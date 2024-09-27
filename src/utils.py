import logging

import numpy as np
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_center_by_erosion(masks, kernel_size: int = 3, to_xy: bool = True,
                          max_iter: int = 50):
    """_summary_

    Args:
        masks (np.Array): in shape [B, H, W] or [B, D, H, W]
        kernel_size (int, optional): _description_. Defaults to 3.
        to_xy (bool, optional): originally in ij for 2D. If True, turn it to
                                xy for SAM point input. Defaults to True.
        max_iter (int, optional): _description_. Defaults to 50.

    Returns:
        center (np.Array): in shape [B, N=1, ndim=2, 3]
    """
    # device = masks.device
    if torch.is_tensor(masks):
        masks = masks.clone().cpu().numpy()
        # masks = masks.cpu().numpy()
    masks = masks.astype(np.uint8)

    assert masks.ndim in [3, 4], "masks shape [B, H, W] or [B, D, H, W]"
    kernel = np.ones([kernel_size] * (masks.ndim - 1), np.uint8)

    coords = []
    for mask in masks:

        iter_count = 0
        mask_temp = np.copy(mask)
        while iter_count < max_iter:
            mask_temp = binary_erosion(mask, kernel).astype(mask.dtype)
            if (np.sum(mask_temp) == 0) or np.array_equal(mask, mask_temp):
                break
            else:
                mask = mask_temp
                iter_count += 1

        mask_coords = np.stack(np.where(mask), axis=-1)
        num_candidate = mask_coords.shape[0]
        idx = np.random.randint(0, num_candidate, 1)
        coords.append(mask_coords[idx])

    coords = np.stack(coords, axis=0)
    coords = torch.Tensor(coords)
    if to_xy and masks.ndim == 3:
        coords = coords.flip(-1)

    return coords


def get_labels_from_coords(masks: torch.Tensor, coords, xy_coord=True):
    assert masks.ndim in [3, 4], "masks shape [B, H, W] or [B, D, H, W]"
    assert coords.ndim == 3, "coords should be in shape [B, N=1, 2 or 3]"

    if masks.ndim == 3:
        b, h, w = masks.size()
        flat_masks = masks.view(b, -1)
        if xy_coord:
            coords = coords.flip(-1)
        flat_indices = coords[:, :, 0] * w + coords[:, :, 1]
        flat_indices = flat_indices.long()
        labels = torch.gather(flat_masks, dim=-1, index=flat_indices)
    else:
        b, d, h, w = masks.size()
        flat_masks = masks.view(b, -1)
        flat_indices = coords[:, :, 0] * w * h + coords[:, :, 1] * w + coords[:, :, 2]
        flat_indices = flat_indices.long()
        labels = torch.gather(flat_masks, dim=-1, index=flat_indices)
    return labels


def get_coords_of_tensor_max(tensor: torch.Tensor, to_xy=True):
    """Get the coordinates of the maximum value in the tensor.

    Args:
        tensor (torch.Tensor): [B, C=1, H, W] or [B, C=1, D, H, W]
        to_xy (bool, optional): Whether to convert the coordinates to xy format
                                Only for 2D tensor. Defaults to True.

    Returns:
        coords (torch.Tensor): [B, C=1, 2 or 3]
    """
    assert tensor.ndim in [4, 5], "tensor shape [B, C=1, (D), H, W]"
    if tensor.ndim == 4:
        n, c, h, w = tensor.size()
        flat_tensor = tensor.view(n, c, -1)
        flat_indices = flat_tensor.argmax(dim=-1)
        coords = torch.stack((flat_indices // w,
                              flat_indices % w), dim=-1)
        if to_xy:
            coords = coords.flip(-1)
    elif tensor.ndim == 5:
        n, c, d, h, w = tensor.size()
        flat_tensor = tensor.view(n, c, -1)
        flat_indices = flat_tensor.argmax(dim=-1)
        coords = torch.stack((flat_indices // (w * h),
                              (flat_indices % (w * h)) // w,
                              flat_indices % w), dim=-1)
    return coords


def get_eig_from_probs(probs, original_size=(1024, 1024),
                       blur_sigma=10., blur_size=21, eps=1e-10):
    """This function calculates the full resolution eigenvalues. It takes
    the low resolution probs from multi-output SAM and calculates the EIG
    with the close-form solution. Then it upsamples the EIG to the original
    size and applies Gaussian blur.

    Args:
        probs (torch.Tensor): the low res. probability inferenced from
                               multi-output SAM, in [B, C=3, (D), H=256, W=256]
                               torch.float32
        original_size (tuple, optional): The original size of the image.
                                         Defaults to (1024, 1024).
        blur_sigma (float, optional): Sigma of the Gaussian blur.
                                      Defaults to 10.0
        blur_size (int, optional): The size of the Gaussian blur filter.
                                   positive odd number defaults to 21.

    Returns:
        eig_full (torch.Tensor): The full resolution EIG map, in
                                 [B, N=1, D=1024, H=1024, W=1024]
                                 torch.float32
    """

    left = (probs ** probs) * ((1 - probs) ** (1 - probs))
    left = torch.clamp(left, eps, 1 - eps)
    left = torch.log(left)
    left = torch.mean(left, dim=1, keepdim=True)
    theta_bar = torch.mean(probs, dim=1, keepdim=True)
    right = (theta_bar ** theta_bar) * ((1 - theta_bar) ** (1 - theta_bar))
    right = torch.clamp(right, eps, 1 - eps)
    right = torch.log(right)
    eig_low_res = left - right
    if len(original_size) == 2:
        eig_high_res = F.interpolate(eig_low_res, size=original_size,
                                     mode='bilinear', align_corners=False)
    elif len(original_size) == 3:
        eig_high_res = F.interpolate(eig_low_res, size=original_size,
                                     mode='trilinear', align_corners=False)
    # eig_high_res = GaussianBlur(blur_size, blur_sigma)(eig_high_res)
    return eig_high_res
