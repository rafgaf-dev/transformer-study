import torch

from transformer.layers import DecoderLayer, EncoderLayer
from transformer.masks import make_causal_mask


def test_encoder_layer_shape():
    layer = EncoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(2, 7, 32)

    assert layer(x).shape == (2, 7, 32)


def test_decoder_layer_shape():
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)

    out = layer(x, memory, tgt_mask=make_causal_mask(5))

    assert out.shape == (2, 5, 32)
