import torch

from transformer.layers import DecoderLayer, EncoderLayer
from transformer.masks import make_causal_mask


def test_encoder_layer_shape():
    layer = EncoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(2, 7, 32)

    assert layer(x).shape == (2, 7, 32)


def test_encoder_layer_uses_src_mask():
    torch.manual_seed(0)
    layer = EncoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(1, 7, 32)
    # Last two source positions are padding.
    src_mask = torch.tensor([True] * 5 + [False] * 2).view(1, 1, 1, 7)

    out = layer(x, src_mask)
    x_changed = x.clone()
    x_changed[:, 5:] = torch.randn(1, 2, 32)
    out_changed = layer(x_changed, src_mask)

    # Real positions must not be affected by what's in the padding.
    assert torch.allclose(out[:, :5], out_changed[:, :5], atol=1e-5)


def test_encoder_layer_mixes_information_across_positions():
    torch.manual_seed(0)
    layer = EncoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(1, 7, 32)

    out = layer(x)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(32)
    out_changed = layer(x_changed)

    # Through self-attention, every position should see the change at the end.
    assert not torch.allclose(out[:, 0], out_changed[:, 0], atol=1e-5)


def test_decoder_layer_shape():
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)

    out = layer(x, memory, tgt_mask=make_causal_mask(5))

    assert out.shape == (2, 5, 32)


def test_decoder_layer_is_causal():
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(1, 5, 32)
    memory = torch.randn(1, 7, 32)
    tgt_mask = make_causal_mask(5)

    out = layer(x, memory, tgt_mask=tgt_mask)
    x_changed = x.clone()
    x_changed[:, -1] = torch.randn(32)
    out_changed = layer(x_changed, memory, tgt_mask=tgt_mask)

    assert torch.allclose(out[:, :-1], out_changed[:, :-1], atol=1e-5)


def test_decoder_layer_attends_to_memory():
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(1, 5, 32)
    tgt_mask = make_causal_mask(5)

    out_a = layer(x, torch.randn(1, 7, 32), tgt_mask=tgt_mask)
    out_b = layer(x, torch.randn(1, 7, 32), tgt_mask=tgt_mask)

    # Cross-attention means the encoder output must influence every position.
    assert not torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-5)


def test_decoder_layer_uses_src_mask():
    torch.manual_seed(0)
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    x = torch.randn(1, 5, 32)
    memory = torch.randn(1, 7, 32)
    src_mask = torch.tensor([True] * 5 + [False] * 2).view(1, 1, 1, 7)
    tgt_mask = make_causal_mask(5)

    out = layer(x, memory, src_mask, tgt_mask)
    memory_changed = memory.clone()
    memory_changed[:, 5:] = torch.randn(1, 2, 32)
    out_changed = layer(x, memory_changed, src_mask, tgt_mask)

    assert torch.allclose(out, out_changed, atol=1e-5)
