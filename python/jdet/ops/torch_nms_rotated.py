from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torchvision.ops import nms_rotated


def multiclass_nms_rotated(
    multi_bboxes: torch.Tensor,
    multi_scores: torch.Tensor,
    score_thr: float,
    nms_cfg: Dict,
    max_num: int = -1,
    score_factors: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.size(1) > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, 1:]

    if score_factors is not None:
        scores = scores * score_factors[:, None]

    results = []
    labels = []
    iou_thr = nms_cfg.get("iou_thr", 0.1)

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        valid = cls_scores > score_thr
        if valid.sum() == 0:
            continue
        cls_bboxes = bboxes[valid, cls_id]
        cls_scores = cls_scores[valid]
        boxes_deg = cls_bboxes.clone()
        boxes_deg[:, 4] = boxes_deg[:, 4] * 180.0 / math.pi
        keep = nms_rotated(boxes_deg, cls_scores, iou_thr)
        cls_bboxes = cls_bboxes[keep]
        cls_scores = cls_scores[keep]
        res = torch.cat([cls_bboxes, cls_scores[:, None]], dim=1)
        results.append(res)
        labels.append(torch.full((res.size(0),), cls_id, dtype=torch.long, device=multi_bboxes.device))

    if not results:
        return multi_bboxes.new_zeros((0, 6)), multi_bboxes.new_zeros((0,), dtype=torch.long)

    results = torch.cat(results, dim=0)
    labels = torch.cat(labels, dim=0)
    order = torch.argsort(results[:, -1], descending=True)
    if max_num > 0:
        order = order[:max_num]
    results = results[order]
    labels = labels[order]
    return results, labels
