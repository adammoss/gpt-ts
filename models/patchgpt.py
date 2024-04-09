import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class PatchGPT(nn.Module):
    def __init__(self, *, seq_len, patch_size, dim, channels):
        super().__init__()
        self.patch_size = patch_size

        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        # x is (B, C, L)
        x = F.pad(x, (0, self.patch_size - x.size(-1) % self.patch_size))
        x = self.to_patch_embedding(x)
        return x
