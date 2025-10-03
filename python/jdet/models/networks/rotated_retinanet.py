import jittor as jt 
from jittor import nn 

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS

from ._training import set_module_training_mode


@MODELS.register_module()
class RotatedRetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,bbox_head=None):
        super(RotatedRetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.bbox_head = build_from_cfg(bbox_head,HEADS)

    def train(self, mode: bool = True):
        try:
            super().train(mode)
        except TypeError:
            super().train()

        set_module_training_mode(self.backbone, mode)
        if getattr(self, "neck", None):
            set_module_training_mode(self.neck, mode)
        if getattr(self, "bbox_head", None):
            set_module_training_mode(self.bbox_head, mode)

        return self

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            outputs: train mode will be losses val mode will be results
        '''
        features = self.backbone(images)
        
        if self.neck:
            features = self.neck(features)
        
        outputs = self.bbox_head(features, targets)
        
        return outputs
