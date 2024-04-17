import torch
import torch.nn as nn
from torch.nn import functional as F

from models.gpt import GenerateMixin

from transformers import PretrainedConfig, PreTrainedModel

from types import SimpleNamespace


class RNNConfig(PretrainedConfig):
    model_type = "rnn"

    def __init__(
            self,
            vocab_size: int = 3501,
            n_embd: int = 36,
            n_hidden: int = 16,
            n_layer: int = 6,
            dropout: float = 0.0,
            n_static: int = 0,
            n_labels: int = 0,
            head_type: str = 'lm',
            **kwargs,
    ):
        assert head_type in ['lm', 'classification']
        if head_type == 'classification':
            assert n_labels > 0
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.dropout = dropout
        self.n_static = n_static
        self.n_labels = n_labels
        self.head_type = head_type
        super().__init__(**kwargs)


class AutoRegressiveRNN(PreTrainedModel, GenerateMixin):
    config_class = RNNConfig

    def __init__(self, config: RNNConfig):
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
        self.n_hidden = config.n_hidden
        self.n_layer = config.n_layer

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # RNN layer
        self.rnn = nn.GRU(config.n_embd, config.n_hidden, config.n_layer, batch_first=True, dropout=config.dropout)

        self.head_type = config.head_type
        self.lm_head = nn.Linear(config.n_hidden, config.vocab_size)
        if config.n_labels > 0:
            self.class_head = nn.Linear(config.n_hidden, config.n_labels)

        if config.n_static > 0:
            self.static = nn.Linear(config.n_static, config.n_embd)

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

        if self.head_type == 'lm':
            logits = self.lm_head(out)  # (B,T,vocab_size)
        else:
            logits = self.class_head(out)  # (B,T,num_labels)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), labels.view(B * T))

        # For compat with Hugging face output
        return SimpleNamespace(logits=logits, loss=loss, labels=labels)
