import torch

from transformer.model import Transformer


def small_model() -> Transformer:
    return Transformer(
        src_vocab_size=20,
        tgt_vocab_size=30,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=100,
        pad_id=0,
    )


def test_transformer_output_shape():
    model = small_model()
    src = torch.randint(1, 20, (2, 7))
    tgt = torch.randint(1, 30, (2, 5))

    assert model(src, tgt).shape == (2, 5, 30)


def test_decoder_cannot_see_the_future():
    model = small_model().eval()
    src = torch.randint(1, 20, (1, 7))
    tgt = torch.randint(1, 30, (1, 5))

    logits = model(src, tgt)
    # Changing the last target token must not affect predictions at earlier positions.
    tgt_changed = tgt.clone()
    tgt_changed[0, -1] = (tgt[0, -1] % 29) + 1
    logits_changed = model(src, tgt_changed)

    assert torch.allclose(logits[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_model_can_overfit_one_batch():
    """If the model can't memorise a single batch, something is wired wrong."""
    torch.manual_seed(0)
    model = small_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    src = torch.randint(1, 20, (4, 6))
    tgt = torch.randint(1, 30, (4, 6))
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

    for _ in range(300):
        logits = model(src, tgt_in)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 30), tgt_out.reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert loss.item() < 0.1
