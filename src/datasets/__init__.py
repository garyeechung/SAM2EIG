import torch

from .flare import FLAREDatasetCache2D as FLAREDataset


def convert_to_device_collate_fn(samples, device='cuda'):
    for sample in samples:
        for key, val in sample.items():
            if torch.is_tensor(val):
                sample[key] = val.to(device)
    return samples


def batch_in_batch_out_fn(batch, device='cuda'):
    return batch


if __name__ == "__main__":
    print(FLAREDataset.__name__)
