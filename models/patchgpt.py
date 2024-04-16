import torch
import torch.nn as nn
from torch.nn import functional as F

from models.gpt import Block

from einops.layers.torch import Rearrange
from transformers import PretrainedConfig, PreTrainedModel

from types import SimpleNamespace


class PatchGPTConfig(PretrainedConfig):
    model_type = "patchgpt"

    def __init__(
            self,
            patch_size: int = 7,
            n_channels: int = 3,
            n_head: int = 6,
            n_embd: int = 36,
            n_positions: int = 1024,
            n_layer: int = 6,
            dropout: float = 0.0,
            n_static: int = 0,
            n_labels: int = 0,
            position_embedding: str = 'absolute',
            head_type: str = 'pretrain_lm',
            random_mask_ratio: float = 0.0,
            **kwargs,
    ):
        assert head_type in ['pretrain_lm', 'pretrain_mask', 'classification']
        if head_type == 'classification':
            assert n_labels > 0
        if head_type == 'pretrain_mask':
            assert random_mask_ratio > 0
        if random_mask_ratio > 0:
            assert head_type == 'pretrain_mask'
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.dropout = dropout
        self.n_static = n_static
        self.n_labels = n_labels
        self.position_embedding = position_embedding
        self.head_type = head_type
        self.random_mask_ratio = random_mask_ratio
        super().__init__(**kwargs)


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.to_patches = Rearrange('b c (n p) -> b n (p c)', p=patch_size)

    def forward(self, x):
        B, T, C = x.shape  # X is (B, T, C)
        x = torch.transpose(x, 1, 2)  # (B, C, T)
        x = F.pad(x, (0, self.patch_size - T % self.patch_size))  # Pad the last dimension (T)
        return self.to_patches(x)  # (B, num patches, patch size * C)


class UnPatchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.from_patches = Rearrange('b n (p c) -> b c (n p)', p=patch_size)

    def forward(self, x):
        B, T, C = x.shape  # (B, num patches, patch size * C)
        x = self.from_patches(x)  # (B, C, T)
        return torch.transpose(x, 1, 2)  # (B, T, C)


class PatchGPT(PreTrainedModel):
    config_class = PatchGPTConfig

    def __init__(self, config: PatchGPTConfig):
        super().__init__(config)
        self.patch_size = config.patch_size

        patch_dim = config.n_channels * config.patch_size

        self.patchify = Patchify(config.patch_size)
        self.unpatchify = UnPatchify(config.patch_size)

        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.n_embd),
            nn.LayerNorm(config.n_embd),
        )

        self.position_embedding = config.position_embedding
        self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)

        is_causal = config.head_type in ['pretrain_lm', 'classification']
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head, config.n_positions, dropout=config.dropout,
                                           position_embedding=config.position_embedding,
                                           is_causal=is_causal) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm

        self.head_type = config.head_type
        self.prediction_head = nn.Linear(config.n_embd, config.patch_size * config.n_channels)
        if config.n_labels > 0:
            self.class_head = nn.Linear(config.n_embd, config.n_labels)

        if config.n_static > 0:
            self.static = nn.Linear(config.n_static, config.n_embd)

        self.random_mask_ratio = config.random_mask_ratio

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, labels=None, attention_mask=None, static=None):
        # x is (B, T, channels), mask is also (B, T, channels)
        patch_x = self.patchify(x)  # (B, new T (num patches), patch_size * channels)
        x = self.patch_embedding(patch_x)  # (B, T, C)

        if self.head_type == 'pretrain_lm':
            patch_x = patch_x[:, 1:, :]
            x = x[:, :-1, :]

        B, T, _ = x.shape  # Get new T

        if attention_mask is not None:
            patch_mask = self.patchify(attention_mask)
            # Check if any valid tokens in channel dimension
            patch_mask = torch.count_nonzero(patch_mask, -1) > 0  # Now (B, T)
            patch_mask = patch_mask.to(torch.int32)
            if self.head_type == 'pretrain_lm':
                patch_mask = patch_mask[:, :-1]
        else:
            patch_mask = torch.ones((B, T)).to(x.device)

        patch_indices = patch_mask.nonzero()
        if self.random_mask_ratio > 0:
            patch_indices = patch_indices[
                torch.randperm(patch_indices.size(0))[:int(patch_indices.size(0) * self.random_mask_ratio)]]
            # These tokens are now not attended to
            patch_mask[patch_indices[:, 0], patch_indices[:, 1]] = 0

        if self.position_embedding == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, C)
            x = x + pos_emb  # (B,T,C)

        if static is not None:
            static = static[:, None, :]
            static_emb = self.static(static)  # (B, 1, C)
            x = x + static_emb

        for block in self.blocks:
            x = block(x, attention_mask=patch_mask)  # (B, T, C)

        x = self.ln_f(x)  # (B, T, C)

        if self.head_type in ['pretrain_lm', 'pretrain_mask']:
            logits = None
            patch_labels = None
            patch_pred = self.prediction_head(x)  # (B, T, patch_size * channels)
            x_pred = self.unpatchify(patch_pred)
            loss = F.mse_loss(patch_pred, patch_x, reduction='none')
            loss = loss.mean(dim=-1)
            loss = loss[patch_indices[:, 0], patch_indices[:, 1]].mean()
        elif self.head_type == 'classification' and labels is not None:
            patch_pred = None
            x_pred = None
            logits = self.class_head(x)  # (B, T, num_labels)
            patch_labels = labels.unsqueeze(-1).expand(B, T).clone()
            # Set non-attended loss to be ignored
            patch_indices = (patch_mask == 0).nonzero()
            patch_labels[patch_indices[:, 0], patch_indices[:, 1]] = -100
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), patch_labels.view(B * T))
        else:
            logits = None
            loss = None
            patch_labels = None
            patch_pred = None
            x_pred = None

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, loss=loss, labels=patch_labels,
                               patch_pred=patch_pred, mask=patch_mask, x_pred=x_pred)
