"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """
    max_x = x.max(dim=dim, keepdim=True).values
    top = torch.exp(x - max_x)
    bot = torch.exp(x - max_x).sum(dim=dim, keepdim=True)
    return top / bot

def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    return x * torch.sigmoid(x)


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    B, T, V = logits.shape

    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = m.squeeze(-1) + torch.exp(logits - m).sum(dim=-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    per_position = logsumexp - target_logits
    return per_position.mean()