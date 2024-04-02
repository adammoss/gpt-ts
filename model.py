"""
GPT like model, adapted from https://github.com/karpathy/minGPT to include attention masks for padded sequences
and relative position encoding
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from types import SimpleNamespace


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, n_positions, dropout=0, position_embedding=None):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_positions, n_positions)))
        self.dropout = nn.Dropout(dropout)
        self.max_position_embeddings = n_positions
        self.position_embedding = position_embedding
        self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, head_size)

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

    def __init__(self, n_head, head_size, n_embd, n_positions, dropout=0.0, position_embedding=None):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, n_positions, dropout=dropout,
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

    def __init__(self, n_embd, n_head, n_positions, dropout=0, position_embedding=None):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, n_positions,
                                     dropout=dropout, position_embedding=position_embedding)
        self.ffwd = FeedFoward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_head, n_embd, n_positions, n_layer, dropout=0, n_static=0, n_labels=0,
                 position_embedding='absolute', use_lm_head=True):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_positions, dropout=dropout,
                                           position_embedding=position_embedding) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if n_labels > 0:
            self.class_head = nn.Linear(n_embd, n_labels)
            self.use_lm_head = use_lm_head
        else:
            self.use_lm_head = False
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
        if self.use_lm_head:
            logits = self.lm_head(x)  # (B,T,vocab_size)
        else:
            logits = self.class_head(x)  # (B,T,num_labels)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, loss=loss)


class AutoRegressiveRNN(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, n_labels=0,
                 num_layers=1, dropout=0, n_static=0, use_lm_head=True):
        """
        Initialize the autoregressive RNN model with an embedding layer.

        Parameters:
            vocab_size (int): The size of the vocabulary (number of unique tokens)
            n_embd (int): The size of each embedding vector
            n_hidden (int): The number of features in the hidden state `h`
            output_size (int): The number of features in the output
            num_layers (int, optional): Number of recurrent layers. Default: 1
        """
        super(AutoRegressiveRNN, self).__init__()
        self.n_hidden = n_hidden
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)
        # RNN layer
        self.rnn = nn.GRU(n_embd, n_hidden, num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer that outputs the predictions
        self.lm_head = nn.Linear(n_hidden, vocab_size)
        if n_labels > 0:
            self.class_head = nn.Linear(n_hidden, n_labels)
            self.use_lm_head = use_lm_head
        else:
            self.use_lm_head = False
        if n_static > 0:
            self.static = nn.Linear(n_static, n_embd)

    def forward(self, x, labels=None, attention_mask=None, static=None):
        """
        Forward pass through the model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_length), containing indices
                        in the range [0, vocab_size)

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Embedding layer output shape: (batch_size, seq_length, n_embd)
        x = self.embedding(x)

        if static is not None:
            static = static[:, None, :]
            static_emb = self.static(static)  # (B, 1, C)
            x = x + static_emb

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.n_hidden).to(x.device)

        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)

        if self.use_lm_head:
            logits = self.lm_head(out)  # (B,T,vocab_size)
        else:
            logits = self.class_head(out)  # (B,T,num_labels)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, loss=loss)
