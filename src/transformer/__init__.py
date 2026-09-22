"""A from-scratch Transformer, built piece by piece.

Suggested order of implementation (each builds on the previous):

    1. masks.py        - padding and causal (look-ahead) masks
    2. attention.py    - scaled dot-product attention, multi-head attention
    3. embeddings.py   - token embeddings, sinusoidal positional encoding
    4. feedforward.py  - position-wise feed-forward network
    5. layers.py       - encoder and decoder layers (residuals + layer norm)
    6. model.py        - encoder stack, decoder stack, full Transformer
"""
