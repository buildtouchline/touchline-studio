"""Keep pretrained training and download-free reload normalization identical."""


def freeze_batchnorm(module):
    import torch
    from torchvision.ops import FrozenBatchNorm2d
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            replacement.load_state_dict({k: v for k, v in child.state_dict().items() if k != 'num_batches_tracked'})
            setattr(module, name, replacement)
        else:
            freeze_batchnorm(child)


def empty_candidate():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=2)
    # Torchvision uses FrozenBN for pretrained weights but BatchNorm for weights=None.
    freeze_batchnorm(model.backbone)
    return model
