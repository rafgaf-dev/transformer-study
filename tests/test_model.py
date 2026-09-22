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


def test_every_parameter_is_used():
    model = small_model()
    src = torch.randint(1, 20, (2, 7))
    tgt = torch.randint(1, 30, (2, 5))

    model(src, tgt).sum().backward()

    unused = [name for name, p in model.named_parameters() if p.grad is None]
    assert not unused, f"parameters not used in forward: {unused}"


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


def test_source_padding_is_ignored():
    """Adding extra padding to the source must not change the predictions."""
    model = small_model().eval()
    src = torch.randint(1, 20, (1, 7))
    tgt = torch.randint(1, 30, (1, 5))
    src_padded = torch.cat([src, torch.zeros(1, 3, dtype=torch.long)], dim=1)

    assert torch.allclose(model(src, tgt), model(src_padded, tgt), atol=1e-5)


def test_target_positions_are_encoded():
    """With a target of one repeated token, the only thing distinguishing the
    positions is the positional encoding - so predictions should differ."""
    model = small_model().eval()
    src = torch.randint(1, 20, (1, 7))
    tgt = torch.full((1, 5), 4)

    logits = model(src, tgt)

    assert not torch.allclose(logits[0, 0], logits[0, 3], atol=1e-4)


def test_source_positions_are_encoded():
    """Without positional encoding the encoder can't tell token order, so
    shuffling the source would leave the output unchanged."""
    torch.manual_seed(0)
    model = small_model().eval()
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    tgt = torch.randint(1, 30, (1, 5))

    logits = model(src, tgt)
    logits_shuffled = model(src.flip(1), tgt)

    assert not torch.allclose(logits, logits_shuffled, atol=1e-4)


def test_eval_mode_is_deterministic():
    model = Transformer(src_vocab_size=20, tgt_vocab_size=30, d_model=32, num_heads=4,
                        num_layers=2, d_ff=64, dropout=0.5).eval()
    src = torch.randint(1, 20, (2, 7))
    tgt = torch.randint(1, 30, (2, 5))

    # Dropout must switch off in eval mode.
    assert torch.equal(model(src, tgt), model(src, tgt))


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
