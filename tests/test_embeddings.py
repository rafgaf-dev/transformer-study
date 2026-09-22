import torch

from transformer.embeddings import PositionalEncoding, TokenEmbedding


def test_token_embedding_shape():
    emb = TokenEmbedding(vocab_size=100, d_model=32)
    tokens = torch.randint(0, 100, (2, 10))

    assert emb(tokens).shape == (2, 10, 32)


def test_token_embedding_same_token_same_vector():
    emb = TokenEmbedding(vocab_size=100, d_model=32)
    out = emb(torch.tensor([[7, 3, 7]]))

    assert torch.equal(out[0, 0], out[0, 2])
    assert not torch.equal(out[0, 0], out[0, 1])


def test_positional_encoding_shape_and_not_trainable():
    pe = PositionalEncoding(d_model=32, max_len=50, dropout=0.0)
    x = torch.zeros(2, 10, 32)

    out = pe(x)

    assert out.shape == (2, 10, 32)
    assert len(list(pe.parameters())) == 0, "positional encoding should be a buffer, not a parameter"


def test_positional_encoding_differs_by_position():
    pe = PositionalEncoding(d_model=32, max_len=50, dropout=0.0)
    out = pe(torch.zeros(1, 10, 32))

    # Every position should get a distinct encoding.
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 3], out[0, 7])


def test_positional_encoding_is_added_to_input():
    pe = PositionalEncoding(d_model=32, max_len=50, dropout=0.0)
    x = torch.randn(2, 10, 32)

    encoding = pe(torch.zeros(2, 10, 32))

    assert torch.allclose(pe(x), x + encoding, atol=1e-6)
    # The same encoding is used for every item in the batch.
    assert torch.allclose(encoding[0], encoding[1])


def test_positional_encoding_matches_paper():
    """Spot-checks a few values that follow directly from the formulas in section 3.5."""
    d_model = 32
    pe = PositionalEncoding(d_model=d_model, max_len=50, dropout=0.0)
    enc = pe(torch.zeros(1, 50, d_model))[0]

    # Position 0: sin(0) = 0 in even dims, cos(0) = 1 in odd dims.
    assert torch.allclose(enc[0, 0::2], torch.zeros(d_model // 2), atol=1e-6)
    assert torch.allclose(enc[0, 1::2], torch.ones(d_model // 2), atol=1e-6)

    # Dimensions 0 and 1 have the highest frequency: sin(pos) and cos(pos).
    positions = torch.arange(50, dtype=torch.float32)
    assert torch.allclose(enc[:, 0], torch.sin(positions), atol=1e-5)
    assert torch.allclose(enc[:, 1], torch.cos(positions), atol=1e-5)

    # Frequencies fall off geometrically across dimension pairs: check the
    # second pair and the last pair.
    for i in (1, d_model // 2 - 1):
        freq = 1 / 10000 ** (2 * i / d_model)
        assert torch.allclose(enc[:, 2 * i], torch.sin(positions * freq), atol=1e-5)
        assert torch.allclose(enc[:, 2 * i + 1], torch.cos(positions * freq), atol=1e-5)

    # All values come from sin/cos, so they lie in [-1, 1].
    assert enc.abs().max() <= 1.0 + 1e-6

    # Higher dimensions vary more slowly than lower ones.
    assert (enc[1:, 0] - enc[:-1, 0]).abs().mean() > (enc[1:, -2] - enc[:-1, -2]).abs().mean()
