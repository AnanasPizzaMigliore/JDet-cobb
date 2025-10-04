from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from jdet.utils.registry import LOSSES


def _obb_to_poly(boxes: torch.Tensor) -> torch.Tensor:
    ctr = boxes[..., :2]
    wh = boxes[..., 2:4]
    theta = boxes[..., 4]
    cos, sin = torch.cos(theta), torch.sin(theta)

    dx = wh[..., 0] / 2
    dy = wh[..., 1] / 2

    corners = torch.stack(
        [
            torch.stack([-dx, -dy], dim=-1),
            torch.stack([dx, -dy], dim=-1),
            torch.stack([dx, dy], dim=-1),
            torch.stack([-dx, dy], dim=-1),
        ],
        dim=-2,
    )
    rot = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1),
    ], dim=-2)
    rotated = torch.matmul(corners, rot)
    rotated = rotated + ctr.unsqueeze(-2)
    return rotated.reshape(*boxes.shape[:-1], 8)


def bbox2type(bboxes: torch.Tensor, to_type: str) -> torch.Tensor:
    if to_type == "poly":
        if bboxes.size(-1) == 5:
            return _obb_to_poly(bboxes)
        if bboxes.size(-1) == 8:
            return bboxes
        raise ValueError("Unsupported bbox format")
    raise NotImplementedError


def get_bbox_areas(bboxes: torch.Tensor) -> torch.Tensor:
    if bboxes.size(-1) == 5:
        return bboxes[..., 2] * bboxes[..., 3]
    polys = bboxes.view(bboxes.size(0), -1, 2)
    x = polys[..., 0]
    y = polys[..., 1]
    area = 0.5 * torch.abs(x * y.roll(-1, dims=-1) - y * x.roll(-1, dims=-1)).sum(dim=-1)
    return area


def _shoelace(pts: torch.Tensor) -> torch.Tensor:
    x = pts[..., 0]
    y = pts[..., 1]
    return 0.5 * torch.abs((x * y.roll(-1, dims=-1) - y * x.roll(-1, dims=-1)).sum(dim=-1))


def _convex_hull(points: torch.Tensor) -> torch.Tensor:
    # points: (N, 2)
    pts = points.unique(dim=0)
    if pts.size(0) <= 1:
        return pts
    pts = pts[pts[:, 0].argsort()]
    lower = []
    for p in pts:
        while len(lower) >= 2:
            cross = (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            cross = (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0])
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = torch.stack(lower[:-1] + upper[:-1]) if len(lower) + len(upper) > 2 else torch.stack(lower + upper)
    return hull


def _segment_intersection(p1, p2, q1, q2) -> Tuple[bool, torch.Tensor]:
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if torch.abs(denom) < 1e-7:
        return False, torch.zeros_like(p1)
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = p1 + t * r
        return True, intersection
    return False, torch.zeros_like(p1)


def _poly_intersection(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    intersections = []
    for i in range(pts1.size(0)):
        a1 = pts1[i]
        a2 = pts1[(i + 1) % pts1.size(0)]
        for j in range(pts2.size(0)):
            b1 = pts2[j]
            b2 = pts2[(j + 1) % pts2.size(0)]
            hit, point = _segment_intersection(a1, a2, b1, b2)
            if hit:
                intersections.append(point)
    # points inside each polygon
    def inside(poly, pts):
        if pts.numel() == 0:
            return torch.empty((0, 2), device=poly.device, dtype=poly.dtype)
        edges = torch.roll(poly, shifts=-1, dims=0) - poly
        normals = torch.stack([-edges[:, 1], edges[:, 0]], dim=-1)
        inside_points = []
        for p in pts:
            rel = p - poly
            cross = (normals * rel).sum(dim=-1)
            if (cross >= -1e-6).all():
                inside_points.append(p)
        if not inside_points:
            return torch.empty((0, 2), device=poly.device, dtype=poly.dtype)
        return torch.stack(inside_points)

    pts = []
    if intersections:
        pts.append(torch.stack(intersections))
    inside1 = inside(pts1, pts2)
    inside2 = inside(pts2, pts1)
    if inside1.numel() > 0:
        pts.append(inside1)
    if inside2.numel() > 0:
        pts.append(inside2)
    if not pts:
        return torch.empty((0, 2), device=pts1.device, dtype=pts1.dtype)
    all_pts = torch.cat(pts, dim=0)
    hull = _convex_hull(all_pts)
    return hull


def poly_iou_loss(pred: torch.Tensor, target: torch.Tensor, linear: bool = False, eps: float = 1e-6, weight: Optional[torch.Tensor] = None, reduction: str = "mean", avg_factor: Optional[float] = None) -> torch.Tensor:
    areas1 = get_bbox_areas(pred)
    areas2 = get_bbox_areas(target)
    pred_poly = bbox2type(pred, "poly").view(pred.size(0), -1, 2)
    target_poly = bbox2type(target, "poly").view(target.size(0), -1, 2)

    overlaps = []
    for p, t in zip(pred_poly, target_poly):
        inter = _poly_intersection(p, t)
        if inter.numel() == 0:
            overlaps.append(p.new_tensor(0.0))
        else:
            overlaps.append(_shoelace(inter.unsqueeze(0))[0])
    overlap = torch.stack(overlaps)

    ious = (overlap / (areas1 + areas2 - overlap + eps)).clamp(min=eps)
    loss = 1 - ious if linear else -ious.log()
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
class PolyIoULoss(nn.Module):
    def __init__(self, linear: bool = False, eps: float = 1e-6, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super().__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        reduction = reduction_override if reduction_override else self.reduction
        loss = poly_iou_loss(
            pred,
            target,
            weight=weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss * self.loss_weight

    def execute(self, *args, **kwargs):  # pragma: no cover
        return self.forward(*args, **kwargs)
