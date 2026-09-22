"""Encoder and decoder layers.

Reference: "Attention Is All You Need", sections 3.1 and 5.4 (residual
dropout). Each sub-layer is wrapped in a residual connection followed by
layer normalisation.

Design choice for you to make: the paper uses "post-norm"
(norm after the residual add). Most modern models use "pre-norm"
(norm before the sub-layer). Try both and compare how training behaves.
"""

import torch
from torch import nn

from transformer.attention import MultiHeadAttention
from transformer.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Self-attention -> feed-forward, each with residual + layer norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: define sub-layers, layer norms and dropout.

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:        (batch, src_len, d_model)
            src_mask: padding mask for the source, see masks.make_padding_mask.

        Returns:
            (batch, src_len, d_model)
        """
        # TODO: implement
        raise NotImplementedError


class DecoderLayer(nn.Module):
    """Masked self-attention -> cross-attention -> feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: define sub-layers, layer norms and dropout.

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch, tgt_len, d_model) decoder input.
            memory:   (batch, src_len, d_model) encoder output.
            src_mask: padding mask for the source (used in cross-attention).
            tgt_mask: causal (+ padding) mask for the target (self-attention).

        Returns:
            (batch, tgt_len, d_model)
        """
        # TODO: implement
        raise NotImplementedError
