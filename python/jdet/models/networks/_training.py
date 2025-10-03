"""Utilities shared across network wrappers for training mode handling."""

from __future__ import annotations

from typing import Any


def set_module_training_mode(module: Any, mode: bool) -> None:
    """Best-effort setter for a submodule's training mode.

    Some modules in the original Jittor-based codebase expose ``train`` without
    accepting a boolean ``mode`` argument, whereas PyTorch expects
    ``train(mode: bool)``.  This helper normalises the behaviour across both
    implementations by attempting to call ``train`` with ``mode`` first and
    falling back to a no-argument invocation when the signature is incompatible.
    """

    if module is None:
        return

    train_fn = getattr(module, "train", None)
    if train_fn is None:
        return

    try:
        train_fn(mode)
    except TypeError:
        train_fn()
