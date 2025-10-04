from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from jdet.utils.registry import LOSSES


def weighted_cross_entropy(pred: torch.Tensor, label: torch.Tensor, weight: torch.Tensor, avg_factor: Optional[float] = None, reduce: bool = True) -> torch.Tensor:
    if avg_factor is None:
        avg_factor = max((weight > 0).sum().item(), 1.0)
    loss = F.cross_entropy(pred, label, reduction="none") * weight
    if reduce:
        return loss.sum() / avg_factor
    return loss / avg_factor


def _expand_binary_labels(labels: torch.Tensor, label_weights: torch.Tensor, label_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze(1)
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand_as(bin_labels)
    return bin_labels, bin_label_weights


def weighted_binary_cross_entropy(pred: torch.Tensor, label: torch.Tensor, weight: torch.Tensor, avg_factor: Optional[float] = None) -> torch.Tensor:
    if pred.ndim != label.ndim:
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max((weight > 0).sum().item(), 1.0)
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction="none") * weight.float()
    return loss.sum() / avg_factor


@LOSSES.register_module()
class CrossEntropyLossForRcnn(nn.Module):
    def __init__(self, use_sigmoid: bool = False, use_mask: bool = False, loss_weight: float = 1.0) -> None:
        super().__init__()
        assert not (use_sigmoid and use_mask)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        if self.use_mask:
            raise NotImplementedError("Mask cross entropy is not implemented in the PyTorch port")

    def forward(self, cls_score: torch.Tensor, label: torch.Tensor, label_weight: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.use_sigmoid:
            loss = weighted_binary_cross_entropy(cls_score, label, label_weight, *args, **kwargs)
        else:
            loss = weighted_cross_entropy(cls_score, label, label_weight, *args, **kwargs)
        return loss * self.loss_weight

    def execute(self, *args, **kwargs):  # pragma: no cover
        return self.forward(*args, **kwargs)


@LOSSES.register_module()
class WeightCrossEntropyLoss(CrossEntropyLossForRcnn):
    pass


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, avg_factor: Optional[float] = None, reduction: str = "mean") -> torch.Tensor:
    target = target.view(-1)
    loss = F.cross_entropy(pred, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        if avg_factor is None:
            avg_factor = max(loss.shape[0], 1)
        loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def binary_cross_entropy_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if class_weight is not None:
        raise NotImplementedError("class_weight is not supported")
    if pred.ndim != label.ndim:
        label = label.float().view_as(pred)
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), reduction="none")
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
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean", use_bce: bool = False, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.use_bce = use_bce
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
        if self.use_bce:
            loss = binary_cross_entropy_loss(pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor)
        else:
            loss = cross_entropy_loss(pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight

    def execute(self, *args, **kwargs):  # pragma: no cover
        return self.forward(*args, **kwargs)
