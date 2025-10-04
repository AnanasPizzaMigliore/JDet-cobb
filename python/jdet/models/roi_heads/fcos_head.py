from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from jdet.models.boxes.torch_ops import distance2obb, mintheta_obb, rotated_box_to_poly
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import bias_init_with_prob, normal_init
from jdet.ops.torch_nms_rotated import multiclass_nms_rotated
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS, LOSSES, build_from_cfg


INF = 1e8


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def execute(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.forward(x)


@HEADS.register_module()
class FCOSHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: Tuple[int, ...] = (4, 8, 16, 32, 64),
        conv_bias: str | bool = "auto",
        regress_ranges: Tuple[Tuple[int, int], ...] = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
        center_sampling: bool = False,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = False,
        centerness_on_reg: bool = False,
        scale_theta: bool = True,
        loss_cls: dict = dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox: dict = dict(type="PolyIoULoss", loss_weight=1.0),
        loss_centerness: dict = dict(type="CrossEntropyLoss", use_bce=True, loss_weight=1.0),
        norm_cfg: Optional[dict] = dict(type="GN", num_groups=32, requires_grad=True),
        test_cfg: Optional[dict] = None,
        conv_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.scale_theta = scale_theta

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.reg_dim = 4
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.conv_bias = conv_bias
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_centerness = build_from_cfg(loss_centerness, LOSSES)
        self.test_cfg = test_cfg or {}
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self) -> None:
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.init_weights()

    def _init_cls_convs(self) -> None:
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

    def _init_reg_convs(self) -> None:
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias,
                )
            )

    def _init_predictor(self) -> None:
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.reg_dim, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_theta:
            self.scale_t = Scale(1.0)

    def init_weights(self) -> None:
        for m in self.cls_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_theta, std=0.01)

    def forward(self, feats: Tuple[torch.Tensor, ...], targets: Optional[List[dict]] = None):
        cls_scores, bbox_preds, theta_preds, centernesses = multi_apply(
            self.forward_single, feats, self.scales, self.strides
        )
        if self.training:
            assert targets is not None
            return self.loss(cls_scores, bbox_preds, theta_preds, centernesses, targets)
        return self.get_bboxes(cls_scores, bbox_preds, theta_preds, centernesses, targets or [])

    def execute(self, *args, **kwargs):  # pragma: no cover
        return self.forward(*args, **kwargs)

    def forward_single(self, x: torch.Tensor, scale: Scale, stride: int):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_pred = scale(bbox_pred)
        if self.norm_on_bbox:
            bbox_pred = torch.relu(bbox_pred)
            if not self.training:
                bbox_pred = bbox_pred * stride
        else:
            bbox_pred = bbox_pred.exp()

        theta_pred = self.conv_theta(reg_feat)
        if self.scale_theta:
            theta_pred = self.scale_t(theta_pred)
        return cls_score, bbox_pred, theta_pred, centerness

    def loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        theta_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        targets: List[dict],
    ) -> dict:
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels_list, bbox_targets_list = self.get_targets(all_level_points, targets)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for cls in cls_scores]
        flatten_bbox_preds = [bbox.permute(0, 2, 3, 1).reshape(-1, 4) for bbox in bbox_preds]
        flatten_theta_preds = [theta.permute(0, 2, 3, 1).reshape(-1, 1) for theta in theta_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=0)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=0)
        flatten_theta_preds = torch.cat(flatten_theta_preds, dim=0)
        flatten_centerness = torch.cat(flatten_centerness, dim=0)
        flatten_labels = torch.cat(labels_list, dim=0)
        flatten_bbox_targets = torch.cat(bbox_targets_list, dim=0)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points], dim=0)

        flatten_bbox_preds = torch.cat([flatten_bbox_preds, flatten_theta_preds], dim=1)
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).squeeze(1)
        num_pos = pos_inds.numel()

        cls_target = flatten_cls_scores.new_zeros(flatten_cls_scores.shape)
        if num_pos > 0:
            pos_labels = flatten_labels[pos_inds].long()
            cls_target[pos_inds, pos_labels] = 1
        loss_cls = self.loss_cls(flatten_cls_scores, cls_target, avg_factor=num_pos + num_imgs)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2obb(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2obb(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum().clamp(min=1e-6),
            )
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_centerness = pos_centerness.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    def get_bboxes(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        theta_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        targets: List[dict],
        rescale: bool = True,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        result_list = []
        for img_id in range(cls_scores[0].size(0)):
            cls_score_list = [cls_scores[i][img_id] for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_levels)]
            theta_pred_list = [theta_preds[i][img_id] for i in range(num_levels)]
            centerness_list = [centernesses[i][img_id] for i in range(num_levels)]
            img_shape = targets[img_id]["img_size"] if targets else None
            scale_factor = targets[img_id].get("scale_factor") if targets else None
            det_bboxes = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                theta_pred_list,
                centerness_list,
                mlvl_points,
                img_shape,
                scale_factor,
                rescale,
            )
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        theta_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        mlvl_points: List[torch.Tensor],
        img_shape,
        scale_factor,
        rescale: bool = False,
    ):
        cfg = self.test_cfg
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, theta_pred, centerness, points in zip(
            cls_scores, bbox_preds, theta_preds, centernesses, mlvl_points
        ):
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            theta_pred = theta_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = torch.cat([bbox_pred, theta_pred], dim=1)
            nms_pre = cfg.get("nms_pre", -1)
            centerness = centerness + cfg.get("centerness_factor", 0.0)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds]
                scores = scores[topk_inds]
                points = points[topk_inds]
                centerness = centerness[topk_inds]
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=0)
        if rescale and scale_factor is not None:
            scale_tensor = torch.as_tensor(scale_factor, device=mlvl_bboxes.device, dtype=mlvl_bboxes.dtype)
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / scale_tensor
        mlvl_scores = torch.cat(mlvl_scores, dim=0)
        padding = mlvl_scores.new_zeros((mlvl_scores.size(0), 1))
        mlvl_centerness = torch.cat(mlvl_centerness, dim=0)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.get("score_thr", 0.05),
            cfg.get("nms", {}),
            cfg.get("max_per_img", -1),
            score_factors=mlvl_centerness,
        )
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    def get_points(self, featmap_sizes, dtype, device) -> List[torch.Tensor]:
        mlvl_points = []
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(self._get_points_single(featmap_size, self.strides[i], dtype, device))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device) -> torch.Tensor:
        h, w = featmap_size
        device = torch.device(device)
        x_range = torch.arange(w, device=device, dtype=dtype)
        y_range = torch.arange(h, device=device, dtype=dtype)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        stride_tensor = torch.tensor(stride, dtype=dtype, device=device)
        points = points * stride_tensor + stride_tensor / 2
        return points

    def get_targets(self, points: List[torch.Tensor], targets: List[dict]):
        num_levels = len(points)
        expand_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand(points[i].size(0), -1)
            for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expand_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [p.size(0) for p in points]

        gt_bboxes_list = [t["rboxes"] for t in targets]
        gt_labels_list = [t["labels"] for t in targets]
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox.split(num_points, 0) for bbox in bbox_targets_list]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return (
                points.new_full((num_points,), self.num_classes, dtype=torch.long),
                points.new_zeros((num_points, 5)),
            )

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_theta = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos, sin = torch.cos(gt_theta), torch.sin(gt_theta)
        matrix = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(matrix, offset.unsqueeze(-1)).squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack([left, top, right, bottom], dim=-1)

        inside_gt_bbox_mask = bbox_targets.min(dim=-1).values > 0
        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_center_bbox_mask = (offset.abs() < stride).all(dim=-1)
            inside_gt_bbox_mask = inside_gt_bbox_mask & inside_center_bbox_mask

        max_regress_distance = bbox_targets.max(dim=-1).values
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1]
        )

        areas[~inside_gt_bbox_mask] = INF
        areas[~inside_regress_range] = INF
        min_area_inds = areas.argmin(dim=1)
        min_area = areas[torch.arange(num_points, device=areas.device), min_area_inds]

        labels = gt_labels[min_area_inds] - 1
        labels[min_area == INF] = self.num_classes

        bbox_targets = bbox_targets[torch.arange(num_points, device=bbox_targets.device), min_area_inds]
        theta_targets = gt_theta[torch.arange(num_points, device=gt_theta.device), min_area_inds]
        bbox_targets = torch.cat([bbox_targets, theta_targets.squeeze(-1)], dim=1)
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets: torch.Tensor) -> torch.Tensor:
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        eps = 1e-6
        lr_min = left_right.min(dim=-1).values
        lr_max = left_right.max(dim=-1).values.clamp(min=eps)
        tb_min = top_bottom.min(dim=-1).values
        tb_max = top_bottom.max(dim=-1).values.clamp(min=eps)
        centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
        return centerness
