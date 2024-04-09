import torch
import torch.nn as nn
from torch.nn import functional as F

from types import SimpleNamespace


class AutoRegressiveRNN(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, n_labels=0,
                 num_layers=1, dropout=0, n_static=0, use_lm_head=True, **kwargs):
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
