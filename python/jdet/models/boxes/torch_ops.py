from __future__ import annotations

import math

import torch


def regular_theta(theta: torch.Tensor, mode: str = "180", start: float = -math.pi / 2) -> torch.Tensor:
    if mode not in {"180", "360"}:
        raise ValueError("mode must be '180' or '360'")
    cycle = 2 * math.pi if mode == "360" else math.pi
    theta = theta - start
    theta = torch.remainder(theta, cycle)
    return theta + start


def regular_obb(obboxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h, theta = torch.unbind(obboxes, dim=-1)
    condition = w > h
    w_regular = torch.where(condition, w, h)
    h_regular = torch.where(condition, h, w)
    theta_regular = torch.where(condition, theta, theta + math.pi / 2)
    theta_regular = regular_theta(theta_regular)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


def mintheta_obb(obboxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h, theta = torch.unbind(obboxes, dim=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + math.pi / 2)
    abs_theta1 = theta1.abs()
    abs_theta2 = theta2.abs()
    mask = abs_theta1 < abs_theta2
    w_regular = torch.where(mask, w, h)
    h_regular = torch.where(mask, h, w)
    theta_regular = torch.where(mask, theta1, theta2)
    return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


def distance2obb(points: torch.Tensor, distance: torch.Tensor, max_shape=None) -> torch.Tensor:
    distance, theta = torch.split(distance, [4, 1], dim=1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    matrix = torch.stack([torch.cat([cos, sin], dim=1), torch.cat([-sin, cos], dim=1)], dim=1)
    wh = distance[:, :2] + distance[:, 2:]
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset = torch.bmm(matrix, offset_t.unsqueeze(-1)).squeeze(-1)
    ctr = points + offset
    obbs = torch.cat([ctr, wh, theta], dim=1)
    return regular_obb(obbs)


def rotated_box_to_poly(boxes: torch.Tensor) -> torch.Tensor:
    ctr = boxes[:, :2]
    wh = boxes[:, 2:4]
    theta = boxes[:, 4]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    wx = wh[:, 0] / 2
    hy = wh[:, 1] / 2

    corners = torch.stack(
        [
            torch.stack([-wx, -hy], dim=-1),
            torch.stack([wx, -hy], dim=-1),
            torch.stack([wx, hy], dim=-1),
            torch.stack([-wx, hy], dim=-1),
        ],
        dim=1,
    )
    rot = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=1,
    )
    rotated = torch.matmul(corners, rot) + ctr.unsqueeze(1)
    return rotated.reshape(boxes.size(0), -1)
