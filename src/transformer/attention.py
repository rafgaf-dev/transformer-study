"""Attention mechanisms.

Reference: "Attention Is All You Need" (Vaswani et al., 2017), sections 3.2.1
and 3.2.2. https://arxiv.org/abs/1706.03762
"""

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention (paper section 3.2.1).

    Args:
        query: (..., query_len, d_k)
        key:   (..., key_len, d_k)
        value: (..., key_len, d_v)
        mask:  optional bool tensor broadcastable to (..., query_len, key_len).
               True = may attend, False = blocked. See masks.py.
        dropout: optional dropout applied to the attention weights.

    Returns:
        output:  (..., query_len, d_v)
        weights: (..., query_len, key_len) - each row sums to 1. Return these
                 so you can visualise what the model attends to.
    """
    # we need to matmul query and key (excluding the last dimension)
    # then we scale by sqrt(d_k) as according to the paper
    # then we apply a mask if one exists.
    # then we get the weights from a softmax
    # the resulting weight matrix is matmuled with the value matrix

    #create query_len by key_len matrix
    query_key = query @ key.mT

    #get and apply scale
    scale = query.size(-1) ** 0.5
    query_key_scaled = query_key / scale

    #apply mask
    if mask is not None:
        masked_query_key_scaled = query_key_scaled.masked_fill(~mask, float('-inf'))
    else:
        masked_query_key_scaled = query_key_scaled

    #apply softmax
    weights = torch.softmax(masked_query_key_scaled, -1)    

    #get outputs by matmul weights and values
    outputs = weights @ value

    return (outputs, weights)


class MultiHeadAttention(nn.Module):
    """Multi-head attention (paper section 3.2.2).

    Used three ways in the Transformer:
        - encoder self-attention      (query = key = value = encoder input)
        - decoder masked self-attention
        - decoder cross-attention     (query from decoder, key/value from encoder)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: define the learnable projections and dropout.

        # Populated on each forward pass so you can inspect/plot attention.
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, query_len, d_model)
            key:   (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask:  optional bool mask broadcastable to
                   (batch, num_heads, query_len, key_len).

        Returns:
            (batch, query_len, d_model)
        """
        # TODO: implement
        raise NotImplementedError
