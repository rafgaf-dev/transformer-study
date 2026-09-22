"""Token embeddings and positional encoding.

Reference: "Attention Is All You Need", sections 3.4 and 3.5.
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Maps token ids to d_model-dimensional vectors (paper section 3.4)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # TODO: define the embedding table.

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) integer token ids.

        Returns:
            (batch, seq_len, d_model)
        """
        # TODO: implement (note the scaling mentioned at the end of 3.4)
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (paper section 3.5).

    Attention on its own has no notion of order; this injects position
    information by adding a position-dependent vector to each embedding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # TODO: precompute a (max_len, d_model) table of encodings and store it
        # with self.register_buffer(...) so it moves with .to(device) but is
        # not a trainable parameter.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) token embeddings.

        Returns:
            (batch, seq_len, d_model) embeddings with position info added.
        """
        # TODO: implement
        raise NotImplementedError
