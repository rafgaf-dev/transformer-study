"""Attention masks.

Convention used throughout this repo: a mask is a boolean tensor where
``True`` means "this position MAY be attended to" and ``False`` means
"block this position". Masks are shaped so they broadcast against attention
scores of shape (batch, num_heads, query_len, key_len).
"""

import torch


def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mask out padding tokens so nothing attends to them.

    Args:
        tokens: (batch, seq_len) integer token ids.
        pad_id: the id used for padding.

    Returns:
        Bool tensor of shape (batch, 1, 1, seq_len).
    """
    # TODO: implement
    raise NotImplementedError


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Mask that stops each position from attending to later positions.

    Used in the decoder's self-attention so predictions for position i can
    only depend on positions <= i.

    Args:
        seq_len: length of the target sequence.
        device: device to create the mask on.

    Returns:
        Bool tensor of shape (1, 1, seq_len, seq_len).
    """
    # TODO: implement
    raise NotImplementedError
