"""
GPT like model, adapted from https://github.com/karpathy/minGPT to include attention masks for padded sequences
and relative position encoding
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PretrainedConfig, PreTrainedModel

from types import SimpleNamespace


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, n_positions, dropout=0, position_embedding=None, is_causal=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_positions, n_positions)))
        self.dropout = nn.Dropout(dropout)
        self.max_position_embeddings = n_positions
        self.position_embedding = position_embedding
        self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, head_size)
        self.is_causal = is_causal

    def forward(self, x, attention_mask=None):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if self.position_embedding == 'relative_key':
            position_ids_l = torch.arange(T, dtype=torch.long, device=x.device).view(-1, 1)
            position_ids_r = torch.arange(T, dtype=torch.long, device=x.device).view(1, -1)
            distance = position_ids_l - position_ids_r  # (T, T, hs)
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            relative_position_scores = torch.einsum("bld,lrd->blr", q, positional_embedding)  # (B, T, T)
            wei = wei + relative_position_scores

        if self.is_causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :]  # (B, 1, T)
            # We have 1's for tokens in mask we *want* to attend to
            attention_mask = (1 - attention_mask) * torch.finfo(wei.dtype).min
            # Broadcast to (B, T, T)
            wei = wei + attention_mask

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values

        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_size, n_embd, n_positions, dropout=0.0, position_embedding=None, is_causal=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, n_positions, dropout=dropout, is_causal=is_causal,
                                         position_embedding=position_embedding) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        out = torch.cat([h(x, attention_mask=attention_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, n_positions, dropout=0, position_embedding=None, is_causal=True):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, n_positions,
                                     dropout=dropout, position_embedding=position_embedding, is_causal=is_causal)
        self.ffwd = FeedFoward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModelConfig(PretrainedConfig):
    model_type = "gptmodel"

    def __init__(
            self,
            vocab_size: int = 3501,
            n_head: int = 6,
            n_embd: int = 36,
            n_positions: int = 1024,
            n_layer: int = 6,
            dropout: float = 0.0,
            n_static: int = 0,
            n_labels: int = 0,
            position_embedding: str = 'absolute',
            is_causal: bool = True,
            head_type: str = 'lm',
            **kwargs,
    ):
        assert head_type in ['lm', 'classification']
        if head_type == 'classification':
            assert n_labels > 0
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.dropout = dropout
        self.n_static = n_static
        self.n_labels = n_labels
        self.position_embedding = position_embedding
        self.is_causal = is_causal
        self.head_type = head_type
        super().__init__(**kwargs)


class GenerateMixin:

    def generate(self, idx, max_new_tokens, static=None, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        self.eval()
        if hasattr(self.config, 'n_positions'):
            n_positions = self.config.n_positions
        else:
            n_positions = 1024
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -n_positions:]
            # get the predictions
            output = self(idx_cond, static=static)
            # focus only on the last time step
            logits = output.logits[:, -1, :] / temperature  # becomes (B, C)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return idx


class GPTModel(GenerateMixin, PreTrainedModel):
    config_class = GPTModelConfig

    def __init__(self, config: GPTModelConfig):
        super().__init__(config)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        self.position_embedding = config.position_embedding
        self.position_embedding_table = nn.Embedding(config.n_positions, config.n_embd)

        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head, config.n_positions, dropout=config.dropout,
                                           position_embedding=config.position_embedding,
                                           is_causal=config.is_causal) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm

        self.head_type = config.head_type
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        if config.n_labels > 0:
            self.class_head = nn.Linear(config.n_embd, config.n_labels)

        if config.n_static > 0:
            self.static = nn.Linear(config.n_static, config.n_embd)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None, attention_mask=None, static=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx)  # (B,T,C)

        if self.position_embedding == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
            x = x + pos_emb  # (B,T,C)

        if static is not None:
            static = static[:, None, :]
            static_emb = self.static(static)  # (B, 1, C)
            x = x + static_emb

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)

        if self.head_type == 'lm':
            logits = self.lm_head(x)  # (B,T,vocab_size)
        else:
            logits = self.class_head(x)  # (B,T,num_labels)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, loss=loss, labels=labels, mask=attention_mask)
