from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from jdet.utils.registry import LOSSES


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    avg_factor: Optional[float] = None,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if weight is not None:
        loss = loss * weight

    if reduction == "mean":
        if avg_factor is None:
            avg_factor = max(loss.numel(), 1)
        loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid: bool = True, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        if not use_sigmoid:
            raise NotImplementedError("Only sigmoid focal loss is supported")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
    ) -> torch.Tensor:
        reduction = reduction_override if reduction_override else self.reduction
        if target.dim() == pred.dim() - 1:
            target_one_hot = pred.new_zeros(pred.shape)
            valid = (target >= 0) & (target < pred.shape[-1])
            target_one_hot[valid, target[valid].long()] = 1
            target = target_one_hot
        else:
            target = target.type_as(pred)
        loss = sigmoid_focal_loss(
            pred,
            target,
            weight=weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss * self.loss_weight

    def execute(self, *args, **kwargs):  # pragma: no cover - legacy API
        return self.forward(*args, **kwargs)
