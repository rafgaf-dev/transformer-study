import torch

from transformer.feedforward import PositionwiseFeedForward


def test_feedforward_shape():
    ff = PositionwiseFeedForward(d_model=32, d_ff=128, dropout=0.0)
    x = torch.randn(2, 10, 32)

    assert ff(x).shape == (2, 10, 32)


def test_feedforward_is_position_wise():
    ff = PositionwiseFeedForward(d_model=32, d_ff=128, dropout=0.0)
    x = torch.randn(1, 10, 32)

    full = ff(x)
    single = ff(x[:, 3:4])

    # Each position is processed independently of the others.
    assert torch.allclose(full[:, 3:4], single, atol=1e-6)
