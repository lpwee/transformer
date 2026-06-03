"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from transformer_lm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """
    n = data.shape[0]
    if n - context_length < 1:
        raise ValueError(
            f"data length ({n}) must exceed context_length ({context_length})"
        )

    starts = torch.randint(0, n - context_length, (batch_size,))
    offsets = torch.arange(context_length)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, T)
    x = data[idx].to(device=device, dtype=torch.long)
    y = data[idx + 1].to(device=device, dtype=torch.long)
    return x, y


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """
    if context_length is None:
        context_length = model.context_length

    ids = list(prompt_ids)
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        ctx = torch.tensor(ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(ctx)[:, -1, :] / temperature  # (1, V)
        probs = softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(int(next_id))
    return ids
