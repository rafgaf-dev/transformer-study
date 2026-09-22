"""Train the Transformer on a toy task.

A good first task is sequence reversal: given [3, 7, 1, 9], output [9, 1, 7, 3].
It needs no dataset download, trains in minutes on a laptop, and the model can
only solve it by learning to attend to the right source positions, which makes
the attention maps easy to interpret.

Run with:
    python scripts/train.py
"""

import torch

from transformer.model import Transformer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # TODO: 1. Define special tokens (PAD, BOS, EOS) and a vocabulary.
    # TODO: 2. Write a function that generates random (src, tgt) batches.
    # TODO: 3. Build the model and an optimiser.
    # TODO: 4. Training loop: teacher forcing, cross-entropy loss (ignore PAD).
    # TODO: 5. Greedy decoding to check the model's outputs on new sequences.
    # TODO: 6. (Optional) Save a checkpoint and plot attention in a notebook.


if __name__ == "__main__":
    main()
