from glob import glob
import logging
import os
import random
import regex as re

import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import get_center_by_erosion, get_labels_from_coords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLAREDataset2D(Dataset):
    def __init__(self, data_dir: str, nb_class=5, max_slices_per_image=256,
                 ignore_bg=True, drop_all_bg=True, training=True):
        if training:
            prefix = "FLARETs"
        else:
            prefix = "testing"

        self.data_dir = data_dir
        self.prefix = prefix
        self.nb_class = nb_class
        self.max_slices_per_image = max_slices_per_image
        self.drop_all_bg = drop_all_bg
        self.ignore_bg = ignore_bg

        image_dir = os.path.join(data_dir, "image")
        image_regex = fr"{image_dir}/{prefix}_(?P<id>\d+)_0000.nii.gz"
        image_regex = re.compile(image_regex)
        image_paths = glob(os.path.join(image_dir, f"{prefix}_*.nii.gz"))
        image_ids = []
        invalid_image_paths = []
        for path in image_paths:
            match = image_regex.match(path)
            if match is not None:
                image_ids.append(match.group("id"))
            else:
                logger.warning(match.group())
                invalid_image_paths.append(path)
        logger.info(f"{len(invalid_image_paths)} invalid image paths")
        logger.debug(invalid_image_paths)

        masks_dir = os.path.join(data_dir, "segment")
        masks_regex = fr"{masks_dir}/{prefix}_(?P<id>\d+).nii.gz"
        masks_regex = re.compile(masks_regex)
        masks_paths = glob(os.path.join(masks_dir, f"{prefix}_*.nii.gz"))
        masks_ids = []
        invalid_masks_paths = []
        for path in masks_paths:
            match = masks_regex.match(path)
            if match is not None:
                masks_ids.append(match.group("id"))
            else:
                invalid_masks_paths.append(path)
        logger.info(f"{len(invalid_masks_paths)} invalid masks paths")
        logger.debug(invalid_masks_paths)

        unmatched_ids = set(image_ids) ^ set(masks_ids)
        logger.info(f"{len(unmatched_ids)} unmatched ids")

        self.ids = list(set(image_ids) & set(masks_ids))
        self.ids = sorted(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image_path = os.path.join(self.data_dir, "image",
                                  f"{self.prefix}_{id}_0000.nii.gz")
        mask_path = os.path.join(self.data_dir, "segment",
                                 f"{self.prefix}_{id}.nii.gz")

        image_3d = nib.load(image_path).get_fdata().transpose(2, 0, 1)
        image_min = image_3d.min()
        image_max = image_3d.max()
        image_3d = (image_3d - image_min) / (image_max - image_min)

        # [H, W, D] -> [D, H, W]
        masks_3d = nib.load(mask_path).get_fdata().transpose(2, 0, 1)

        samples = []
        for image, masks in zip(image_3d, masks_3d):

            image = self.get_image_tensor(image)
            masks = self.get_mask_tensor(masks)
            if masks is None:
                continue

            masks = masks.float()

            point_coords = get_center_by_erosion(masks)
            point_labels = get_labels_from_coords(masks, point_coords)

            sample = {
                'image': image,
                'masks': masks,
                'original_size': (1024, 1024),
                'point_coords': point_coords,
                'point_labels': point_labels,
            }
            samples.append(sample)

        if len(samples) > self.max_slices_per_image:
            samples = random.sample(samples, self.max_slices_per_image)

        return samples

    def get_image_tensor(self, image: np.ndarray):
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)
        image = torch.from_numpy(image).float()

        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_mask_tensor(self, masks: np.ndarray):
        masks = masks.astype(np.uint8)
        if self.drop_all_bg and np.all(masks == 0):
            return None

        # one-hot encoding [H, W] -> [H, W, C]
        masks = np.eye(self.nb_class)[masks]

        # [H, W, C] -> [C, H, W]
        masks = masks.transpose(2, 0, 1)

        # ignore background [C, H, W] -> [C-1, H, W]
        if self.ignore_bg:
            masks = masks[1:]

        # select only non-empty masks [C-1, H, W] -> [?, H, W]
        non_empty_index = np.any(masks, axis=(1, 2))
        masks = masks[non_empty_index]

        masks = torch.from_numpy(masks).float()

        transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        ])

        return transform(masks)


class FLAREDatasetCache2D(Dataset):

    def __init__(self, data_dir: str, nb_class: int = 5,
                 ignore_bg: bool = True, drop_all_bg: bool = True,
                 split: str = "train", seed: int = 42,
                 validate_ratio: float = 0.1):
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache", f"shuffle_{seed}")
        self.nb_class = nb_class
        self.ignore_bg = ignore_bg
        self.drop_all_bg = drop_all_bg

        self.split_path = os.path.join(self.cache_dir, f"{split}")
        if not os.path.exists(self.split_path):
            if split == "train":
                ids, _ = self.get_ids(is_testing=False, seed=seed)
            elif split == "valid":
                _, ids = self.get_ids(is_testing=False, seed=seed)
            elif split == "test":
                ids = self.get_ids(is_testing=True)
            os.makedirs(self.split_path)
            self.preprocess(ids, is_testing=(split == "test"))
        self.ids = glob(os.path.join(self.split_path, "*.pt"))
        self.ids = sorted(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = torch.load(self.ids[idx])
        return sample

    def get_ids(self, is_testing: bool = False, seed: int = 42,
                validate_ratio: float = 0.1):
        if is_testing:
            prefix = "testing"
        else:
            prefix = "FLARETs"

        os.makedirs(self.cache_dir, exist_ok=True)
        image_dir = os.path.join(self.data_dir, "image")

        image_regex = fr"{image_dir}/{prefix}_(?P<id>\d+)_0000.nii.gz"
        image_regex = re.compile(image_regex)
        image_paths = glob(os.path.join(image_dir, f"{prefix}_*.nii.gz"))
        image_ids = []
        invalid_image_paths = []
        for path in image_paths:
            match = image_regex.match(path)
            if match is not None:
                image_ids.append(match.group("id"))
            else:
                logger.warning(match.group())
                invalid_image_paths.append(path)
        logger.info(f"{len(invalid_image_paths)} invalid image paths")
        logger.debug(invalid_image_paths)

        masks_dir = os.path.join(self.data_dir, "segment")
        masks_regex = fr"{masks_dir}/{prefix}_(?P<id>\d+).nii.gz"
        masks_regex = re.compile(masks_regex)
        masks_paths = glob(os.path.join(masks_dir, f"{prefix}_*.nii.gz"))
        masks_ids = []
        invalid_masks_paths = []
        for path in masks_paths:
            match = masks_regex.match(path)
            if match is not None:
                masks_ids.append(match.group("id"))
            else:
                invalid_masks_paths.append(path)
        logger.info(f"{len(invalid_masks_paths)} invalid masks paths")
        logger.debug(invalid_masks_paths)

        unmatched_ids = set(image_ids) ^ set(masks_ids)
        logger.info(f"{len(unmatched_ids)} unmatched ids")

        ids = list(set(image_ids) & set(masks_ids))
        if not is_testing:
            random.seed(seed)
            random.shuffle(ids)
            ids_valid = ids[:int(validate_ratio * len(ids))]
            ids_train = ids[int(validate_ratio * len(ids)):]
            return ids_train, ids_valid
        else:
            return ids

    def preprocess(self, ids, is_testing=False):
        if is_testing:
            prefix = "testing"
        else:
            prefix = "FLARETs"
        for i, id_ in enumerate(ids, 1):
            image_path = os.path.join(self.data_dir, "image",
                                      f"{prefix}_{id_}_0000.nii.gz")
            mask_path = os.path.join(self.data_dir, "segment",
                                     f"{prefix}_{id_}.nii.gz")

            image_3d = nib.load(image_path).get_fdata().transpose(2, 0, 1)
            image_min = image_3d.min()
            image_max = image_3d.max()
            image_3d = (image_3d - image_min) / (image_max - image_min)

            # [H, W, D] -> [D, H, W]
            masks_3d = nib.load(mask_path).get_fdata().transpose(2, 0, 1)

            for j, (image, masks) in enumerate(zip(image_3d, masks_3d), 1):
                print(f"{i:02d}/{len(ids):02d}: {j:04d}/{len(image_3d):04d}",
                      end="\r")
                image = self.get_image_tensor(image)
                masks = self.get_mask_tensor(masks)
                if masks is None:
                    continue

                masks = masks.float()

                point_coords = get_center_by_erosion(masks)
                point_labels = get_labels_from_coords(masks, point_coords)

                sample = {
                    'image': image,
                    'masks': masks,
                    'original_size': (1024, 1024),
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                }
                if is_testing:
                    file_path = os.path.join(self.data_dir, "cache", "test",
                                             f"{id_}_{j:04d}.pt")
                else:
                    file_path = os.path.join(self.split_path,
                                             f"{id_}_{j:04d}.pt")
                torch.save(sample, file_path)

    def get_image_tensor(self, image: np.ndarray):
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)
        image = torch.from_numpy(image).float()

        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_mask_tensor(self, masks: np.ndarray):
        masks = masks.astype(np.uint8)
        if self.drop_all_bg and np.all(masks == 0):
            return None

        # one-hot encoding [H, W] -> [H, W, C]
        masks = np.eye(self.nb_class)[masks]

        # [H, W, C] -> [C, H, W]
        masks = masks.transpose(2, 0, 1)

        # ignore background [C, H, W] -> [C-1, H, W]
        if self.ignore_bg:
            masks = masks[1:]

        # select only non-empty masks [C-1, H, W] -> [?, H, W]
        non_empty_index = np.any(masks, axis=(1, 2))
        masks = masks[non_empty_index]

        masks = torch.from_numpy(masks).float()

        transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
        ])

        return transform(masks)


class FLAREDataset3D(Dataset):
    def __init__(self, data_dir: str, nb_class=5,
                 image_size=(128, 128, 128),
                 training=True, ignore_bg=True, warmstart=False):
        if training:
            prefix = "FLARETs"
        else:
            prefix = "testing"

        self.data_dir = data_dir
        self.image_size = image_size
        self.prefix = prefix
        self.nb_class = nb_class
        self.ignore_bg = ignore_bg
        self.warmstart = warmstart

        image_dir = os.path.join(data_dir, "image")
        image_regex = fr"{image_dir}/{prefix}_(?P<id>\d+)_0000.nii.gz"
        image_regex = re.compile(image_regex)
        image_paths = glob(os.path.join(image_dir, f"{prefix}_*.nii.gz"))
        image_ids = []
        invalid_image_paths = []
        for path in image_paths:
            match = image_regex.match(path)
            if match is not None:
                image_ids.append(match.group("id"))
            else:
                logger.warning(match.group())
                invalid_image_paths.append(path)
        logger.info(f"{len(invalid_image_paths)} invalid image paths")
        logger.debug(invalid_image_paths)

        masks_dir = os.path.join(data_dir, "segment")
        masks_regex = fr"{masks_dir}/{prefix}_(?P<id>\d+).nii.gz"
        masks_regex = re.compile(masks_regex)
        masks_paths = glob(os.path.join(masks_dir, f"{prefix}_*.nii.gz"))
        masks_ids = []
        invalid_masks_paths = []
        for path in masks_paths:
            match = masks_regex.match(path)
            if match is not None:
                masks_ids.append(match.group("id"))
            else:
                invalid_masks_paths.append(path)
        logger.info(f"{len(invalid_masks_paths)} invalid masks paths")
        logger.debug(invalid_masks_paths)

        unmatched_ids = set(image_ids) ^ set(masks_ids)
        logger.info(f"{len(unmatched_ids)} unmatched ids")

        self.ids = list(set(image_ids) & set(masks_ids))
        self.ids = sorted(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        image_path = os.path.join(self.data_dir, "image",
                                  f"{self.prefix}_{id_}_0000.nii.gz")
        image = nib.load(image_path).get_fdata()
        image = self.get_image_tensor(image)
        # print("image", image.shape)

        mask_path = os.path.join(self.data_dir, "segment",
                                 f"{self.prefix}_{id_}.nii.gz")
        masks = nib.load(mask_path).get_fdata()
        masks = self.get_mask_tensor(masks)
        # print("masks", masks.shape)

        point_coords = get_center_by_erosion(masks)
        point_labels = get_labels_from_coords(masks, point_coords)
        if self.warmstart:
            non_empty_slices = masks.sum(dim=(2, 3)) > 0
            boundary_slices = np.where(non_empty_slices.diff())[1].reshape(-1, 2)
            point_coords_new = []
            point_labels_new = []
            for lab, (a, b) in enumerate(boundary_slices, 0):
                range_ = b - a
                slices = [int(a + 0.1 * range_), int(a + 0.9 * range_)]
                prompted_slices = masks[lab, slices]
                slices = torch.Tensor(slices).unsqueeze(-1).unsqueeze(0)
                points = get_center_by_erosion(prompted_slices)
                labels = get_labels_from_coords(prompted_slices, points)
                labels = labels.permute((1, 0))
                points = points.flip(-1).permute((1, 0, 2))
                print(slices.shape, points.shape)

                points = torch.cat((slices, points), dim=-1)

                point_coords_new.append(points)
                point_labels_new.append(labels)

            point_coords_new = torch.cat(point_coords_new, dim=0)
            point_labels_new = torch.cat(point_labels_new, dim=0)
            point_coords_new = torch.cat((point_coords, point_coords_new), dim=1)
            point_labels_new = torch.cat((point_labels, point_labels_new), dim=1)

            point_coords, point_labels = [], []
            for coord, label in zip(point_coords_new, point_labels_new):
                indices = coord[:, 0].sort().indices
                point_coords.append(coord[indices])
                point_labels.append(label[indices])
            point_coords = torch.stack(point_coords, dim=0)
            point_labels = torch.stack(point_labels, dim=0)

        sample = {
            'image': image,
            'masks': masks,
            'original_size': self.image_size,
            'point_coords': point_coords,
            'point_labels': point_labels,
        }
        return sample

    def get_image_tensor(self, image: np.ndarray, mean=123.675, std=58.395):
        image_min = image.min()
        image_max = image.max()
        image = 255 * (image - image_min) / (image_max - image_min)
        image = (image - mean) / std
        image = image.transpose(2, 1, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.image_size, mode='trilinear',
                              align_corners=False)
        return image

    def get_mask_tensor(self, masks: np.ndarray):
        masks = masks.astype(np.uint8)
        masks = masks.transpose(2, 1, 0)
        masks = np.eye(self.nb_class)[masks]
        masks = masks.transpose(3, 0, 1, 2)
        if self.ignore_bg:
            masks = masks[1:]

        # select only non-empty masks
        non_empty_index = np.any(masks, axis=(1, 2, 3))
        masks = masks[non_empty_index]

        masks = torch.from_numpy(masks).float()
        masks = masks.unsqueeze(0)
        masks = F.interpolate(masks, size=self.image_size, mode='trilinear',
                              align_corners=False)
        return masks[0]
