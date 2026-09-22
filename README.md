# transformer-study

See how transformers work, by building one from scratch in PyTorch.

The code under `src/transformer/` is a skeleton: every class and function has
its signature, a docstring with the expected tensor shapes, and a `TODO`. The
implementation is yours to write.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check it works:

```bash
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```

## Layout

```
src/transformer/
    masks.py        padding and causal masks
    attention.py    scaled dot-product attention, multi-head attention
    embeddings.py   token embeddings, sinusoidal positional encoding
    feedforward.py  position-wise feed-forward network
    layers.py       encoder and decoder layers
    model.py        encoder, decoder, full Transformer
tests/              one test file per module, to check your implementation
scripts/train.py    training script skeleton for a toy task
notebooks/          for experiments and visualising attention
```

## Suggested path

Work through the modules in this order, and run the matching tests as you go:

| Step | Module           | Paper section | Test                               |
|------|------------------|---------------|------------------------------------|
| 1    | `masks.py`       | 3.2.3         | `pytest tests/test_masks.py`       |
| 2    | `attention.py`   | 3.2.1, 3.2.2  | `pytest tests/test_attention.py`   |
| 3    | `embeddings.py`  | 3.4, 3.5      | `pytest tests/test_embeddings.py`  |
| 4    | `feedforward.py` | 3.3           | `pytest tests/test_feedforward.py` |
| 5    | `layers.py`      | 3.1, 5.4      | `pytest tests/test_layers.py`      |
| 6    | `model.py`       | 3, Figure 1   | `pytest tests/test_model.py`       |
| 7    | `scripts/train.py` | 5           | train on sequence reversal         |

Then open a notebook (`jupyter lab`) and plot `MultiHeadAttention.attention_weights`
from your trained model to see what each head learned.

## Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017. The original paper.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar. Visual walkthrough.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), Harvard NLP. A line-by-line implementation. Save it for after you've tried each piece yourself.
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy. Video building a decoder-only model.

## Ideas for after

- Swap post-norm for pre-norm and compare training stability.
- Build a decoder-only (GPT-style) model from the same pieces and train it on character-level text.
- Replace sinusoidal positions with learned embeddings or RoPE.
