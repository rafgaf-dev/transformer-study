"""Position-wise feed-forward network.

Reference: "Attention Is All You Need", section 3.3.
"""

import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied independently to every position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: define layers.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # TODO: implement
        raise NotImplementedError
