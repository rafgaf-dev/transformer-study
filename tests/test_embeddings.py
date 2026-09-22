import torch

from transformer.embeddings import PositionalEncoding, TokenEmbedding


def test_token_embedding_shape():
    emb = TokenEmbedding(vocab_size=100, d_model=32)
    tokens = torch.randint(0, 100, (2, 10))

    assert emb(tokens).shape == (2, 10, 32)


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
