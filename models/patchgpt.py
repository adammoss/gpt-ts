import torch
import torch.nn as nn
from torch.nn import functional as F

from models.gpt import Block

from einops.layers.torch import Rearrange

from types import SimpleNamespace


class PatchGPT(nn.Module):
    def __init__(self, patch_size, channels, n_head, n_embd, n_positions, n_layer, dropout=0, n_static=0, n_labels=0,
                 position_embedding='absolute', pretrain=True, **kwargs):
        super().__init__()
        self.patch_size = patch_size

        patch_dim = channels * patch_size

        self.to_patches = Rearrange('b c (n p) -> b n (p c)', p=patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_embd),
            nn.LayerNorm(n_embd),
        )

        self.position_embedding_table = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_positions, dropout=dropout,
                                           position_embedding=position_embedding) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.prediction_head = nn.Linear(n_embd, patch_size * channels)
        if n_labels > 0:
            self.class_head = nn.Linear(n_embd, n_labels)
            self.pretrain = pretrain
        else:
            self.pretrain = True
        if n_static > 0:
            self.static = nn.Linear(n_static, n_embd)
        self.position_embedding = position_embedding

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, labels=None, mask=None, static=None):
        B, _, T = x.shape  # x is (B, channels, T)
        x = F.pad(x, (0, self.patch_size - T % self.patch_size))
        patch_input = self.to_patches(x)  # (B, T, patch_size * channels)
        x = self.to_patch_embedding(patch_input)  # (B, T, C)
        if mask is not None:
            mask = F.pad(mask, (0, self.patch_size - T % self.patch_size))
            patch_mask = self.to_patches(mask)
            patch_mask = torch.count_nonzero(patch_mask, -1) > 0
            patch_mask = patch_mask.to(torch.int32)
        else:
            patch_mask = None
        B, T, _ = x.shape
        if self.position_embedding == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T,C)
            x = x + pos_emb  # (B,T,C)
        if static is not None:
            static = static[:, None, :]
            static_emb = self.static(static)  # (B, 1, C)
            x = x + static_emb
        for block in self.blocks:
            x = block(x, attention_mask=patch_mask)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        if self.pretrain:
            logits = None
            patch_pred = self.prediction_head(x)  # (B,T, patch_size * channels)
            loss = F.mse_loss(patch_pred, patch_input, reduction='none')
        else:
            patch_pred = None
            logits = self.class_head(x)  # (B,T,num_labels)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, patch_pred=patch_pred, loss=loss,
                               patch_mask=patch_mask)
