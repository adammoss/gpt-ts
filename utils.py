import torch
from torch.nn import functional as F


def randint(low, high=None, size=None, device=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        size = low.shape if isinstance(low, torch.Tensor) else high.shape
    return torch.randint(2 ** 63 - 1, size=size, device=device) % (high - low) + low


def get_last_masked_index(mask):
    # Get the last index where the mask = 1 (observed)
    B, T = mask.shape
    return T - torch.argmax(torch.flip((mask == 1).long(), [1]), 1) - 1


def get_random_masked_index(mask):
    return torch.squeeze(torch.multinomial(mask.float(), 1))



