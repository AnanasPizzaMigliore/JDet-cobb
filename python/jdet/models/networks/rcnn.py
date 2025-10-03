from math import e
import torch
from torch import nn

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class RCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self,backbone,neck=None,rpn=None,bbox_head=None):
        super(RCNN,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn = build_from_cfg(rpn,HEADS)
        self.bbox_head = build_from_cfg(bbox_head,HEADS)
        
    def forward(self, images, targets):
        '''
        Args:
            images (torch.Tensor): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections
            losses (dict): losses
        '''

        features = self.backbone(images)
        
        if self.neck:
            features = self.neck(features)

        proposals_list, rpn_losses = self.rpn(features,targets)

        output = self.bbox_head(features, proposals_list, targets)

        if self.training:
            output.update(rpn_losses)

        return output

    # NOTE: retain the execute entry point for backward compatibility with
    # the original Jittor-based pipeline. ``execute`` simply delegates to
    # :meth:`forward`, which is the idiomatic PyTorch inference entry point.
    def execute(self, images, targets):
        return self.forward(images, targets)

    def train(self, mode: bool = True):
        super(RCNN, self).train(mode)
        self.backbone.train(mode)
        if self.neck:
            self.neck.train(mode)
        if self.rpn:
            self.rpn.train(mode)
        if self.bbox_head:
            self.bbox_head.train(mode)
        return self
