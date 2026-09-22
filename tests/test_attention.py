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


def test_multi_head_attention_has_four_projections():
    """Section 3.2.2: the heads' W_Q, W_K, W_V projections together amount to
    three d_model x d_model matrices, plus the output projection W_O."""
    d_model = 64
    mha = MultiHeadAttention(d_model=d_model, num_heads=8, dropout=0.0)
    n_params = sum(p.numel() for p in mha.parameters())

    # With or without bias terms.
    assert n_params in (4 * d_model * d_model, 4 * d_model * d_model + 4 * d_model)

    # And every one of them is actually used in the forward pass.
    x = torch.randn(2, 5, d_model)
    mha(x, x, x).sum().backward()
    unused = [name for name, p in mha.named_parameters() if p.grad is None]
    assert not unused, f"parameters not used in forward: {unused}"


def test_multi_head_attention_is_permutation_equivariant():
    """Attention has no built-in notion of order: shuffling the sequence should
    just shuffle the output. This fails if heads are split/merged in a way that
    mixes up positions (e.g. a .view() without the right .transpose())."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 32)
    perm = torch.randperm(6)

    out = mha(x, x, x)
    out_permuted = mha(x[:, perm], x[:, perm], x[:, perm])

    assert torch.allclose(out[:, perm], out_permuted, atol=1e-5)


def test_multi_head_attention_keeps_batch_elements_independent():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(3, 6, 32)

    out_batched = mha(x, x, x)
    out_single = mha(x[1:2], x[1:2], x[1:2])

    assert torch.allclose(out_batched[1:2], out_single, atol=1e-5)


def test_multi_head_attention_ignores_masked_keys():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(1, 6, 32)
    # Last two key positions are padding.
    mask = torch.tensor([True, True, True, True, False, False]).view(1, 1, 1, 6)

    out = mha(x, x, x, mask=mask)
    x_changed = x.clone()
    x_changed[:, 4:] = torch.randn(1, 2, 32)
    out_changed = mha(x[:, :4], x_changed, x_changed, mask=mask)

    # Changing masked-out keys/values must not change the output.
    assert torch.allclose(out[:, :4], out_changed, atol=1e-5)
    assert torch.all(mha.attention_weights[..., 4:] == 0)
