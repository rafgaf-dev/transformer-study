import torch

from transformer.masks import make_causal_mask, make_padding_mask


def test_padding_mask_shape_and_values():
    tokens = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    mask = make_padding_mask(tokens, pad_id=0)

    assert mask.shape == (2, 1, 1, 4)
    assert mask.dtype == torch.bool
    expected = torch.tensor([[True, True, False, False], [True, True, True, False]])
    assert torch.equal(mask[:, 0, 0, :], expected)


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4)

    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == torch.bool
    # Position i may attend to positions 0..i only.
    assert torch.equal(mask[0, 0], torch.tril(torch.ones(4, 4, dtype=torch.bool)))
