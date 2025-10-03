from jdet.utils.registry import MODELS

from ._training import set_module_training_mode
from .single_stage import SingleStageDetector

@MODELS.register_module()
class FCOS(SingleStageDetector):
    
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