import torch

from transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from transformer.masks import make_causal_mask


def test_sdpa_shapes_and_weights_sum_to_one():
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 7, 16)
    v = torch.randn(2, 7, 32)

    out, weights = scaled_dot_product_attention(q, k, v)

    assert out.shape == (2, 5, 32)
    assert weights.shape == (2, 5, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_sdpa_respects_mask():
    q = k = v = torch.randn(1, 1, 4, 8)
    mask = make_causal_mask(4)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    # Nothing above the diagonal should receive any attention.
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[0, 0][upper] == 0)


def test_sdpa_matches_pytorch_reference():
    q, k, v = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    mask = make_causal_mask(6)

    out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    assert torch.allclose(out, expected, atol=1e-5)


def test_multi_head_attention_shapes():
    mha = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.0)
    query = torch.randn(2, 5, 64)
    memory = torch.randn(2, 9, 64)

    out = mha(query, memory, memory)

    assert out.shape == (2, 5, 64)
    assert mha.attention_weights is not None
    assert mha.attention_weights.shape == (2, 8, 5, 9)
