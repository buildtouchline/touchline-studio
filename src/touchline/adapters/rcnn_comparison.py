"""Explicit CPU/CUDA official R-CNN comparison; never replaces trained weights."""
from pathlib import Path
from .torchvision import TorchvisionDetector
from ..data.manifest import sha256_file

WEIGHTS_SHA256='dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'

class RcnnComparisonDetector(TorchvisionDetector):
    architecture='fasterrcnn_resnet50_fpn_v2'

    def __init__(self,checkpoint,*,threshold=.1,min_size=1080,max_size=4096,device='cpu'):
        if device not in ('cpu','cuda'):raise ValueError('Explicit CPU or CUDA device required')
        if not 0<=threshold<=1 or type(min_size) is not int or type(max_size) is not int or not 0<min_size<=max_size:raise ValueError('Valid threshold and resize required')
        checkpoint=Path(checkpoint)
        if checkpoint.is_symlink() or sha256_file(checkpoint)!=WEIGHTS_SHA256:raise ValueError('Pinned official comparison weights required')
        import torch,torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA explicitly requested but unavailable')
        self.torch=torch;self.device=device;self.actual_device=device;self.threshold=threshold
        self.model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,num_classes=91,min_size=min_size,max_size=max_size,box_score_thresh=.1)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);self.model.to(device).eval()
        self.actual_device=str(next(self.model.parameters()).device)
        self._inference_calls=0;self._input_device=None;self._detector_wall_seconds=0.
        self._provenance={'adapter':'torchvision-rcnn-comparison-v1','architecture':self.architecture,'checkpoint':WEIGHTS_SHA256,'torch':torch.__version__,'torchvision':torchvision.__version__,'device':device,'model_parameter_device':self.actual_device,'threshold':threshold,'min_size':min_size,'max_size':max_size,'weights_source':'official Torchvision COCO_V1','trained_on_touchline':False,'selection':'owner selected for analysis; trained MobileNet retained'}
