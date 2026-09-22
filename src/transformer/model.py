"""The full encoder-decoder Transformer.

Reference: "Attention Is All You Need", section 3 and Figure 1.
"""

import torch
from torch import nn

from transformer.embeddings import PositionalEncoding, TokenEmbedding
from transformer.layers import DecoderLayer, EncoderLayer
from transformer.masks import make_causal_mask, make_padding_mask


class Encoder(nn.Module):
    """A stack of `num_layers` EncoderLayers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: define the layer stack (hint: nn.ModuleList).

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(batch, src_len, d_model) -> (batch, src_len, d_model)"""
        # TODO: implement
        raise NotImplementedError


class Decoder(nn.Module):
    """A stack of `num_layers` DecoderLayers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: define the layer stack.

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, tgt_len, d_model) -> (batch, tgt_len, d_model)"""
        # TODO: implement
        raise NotImplementedError


class Transformer(nn.Module):
    """Encoder-decoder Transformer that outputs next-token logits.

    Defaults are the paper's "base" model sizes. For toy tasks on a laptop,
    something like d_model=128, num_heads=4, num_layers=2, d_ff=512 is plenty.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        # TODO: define embeddings, positional encoding, encoder, decoder and
        # the final projection to vocabulary logits.

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """(batch, src_len) token ids -> (batch, src_len, d_model) memory"""
        # TODO: implement
        raise NotImplementedError

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """(batch, tgt_len) token ids -> (batch, tgt_len, tgt_vocab_size) logits"""
        # TODO: implement
        raise NotImplementedError

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token ids.
            tgt: (batch, tgt_len) target token ids (shifted right, i.e. starting
                 with a BOS token - this is "teacher forcing").

        Returns:
            (batch, tgt_len, tgt_vocab_size) logits.
        """
        # TODO: build the masks, then encode and decode.
        raise NotImplementedError
