"""SE, CBAM and Coordinate Attention; sources and choices in ATTENTION.md."""
import torch
from torch import nn
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect.train import DetectionTrainer

LAYER = 6


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                    nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                     nn.BatchNorm2d(1, eps=1e-5, momentum=0.01))

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)
        maximum = x.amax(dim=(2, 3), keepdim=True)
        x = x * torch.sigmoid(self.shared(avg) + self.shared(maximum))
        spatial_input = torch.cat((x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)), dim=1)
        return x * self.spatial(spatial_input).sigmoid()


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.encode = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                    nn.BatchNorm2d(hidden), nn.Hardswish())
        self.height_gate = nn.Conv2d(hidden, channels, 1)
        self.width_gate = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        along_height = x.mean(dim=3, keepdim=True)
        along_width = x.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.encode(torch.cat((along_height, along_width), dim=2))
        h, w = encoded.split((height, width), dim=2)
        return x * self.height_gate(h).sigmoid() * self.width_gate(w.transpose(2, 3)).sigmoid()


class AttentionC2f(C2f):
    def forward(self, x):
        return self.attention(super().forward(x))

    def forward_split(self, x):
        return self.attention(super().forward_split(x))


MODULES = {"se": SE, "cbam": CBAM, "ca": CoordinateAttention}


def attach_attention(model, variant):
    if variant not in MODULES:
        raise ValueError(f"Unknown attention variant: {variant}")
    block = model.model[LAYER]
    if type(block) is not C2f or block.cv2.conv.out_channels != 128:
        raise ValueError("Expected unmodified YOLOv8n layer 6 with 128 output channels")
    # Retain all existing C2f modules and state-dict names for comparable transfer.
    # Only its forward dispatch and an additional registered submodule change.
    block.__class__ = AttentionC2f
    with torch.random.fork_rng(devices=[]):
        block.attention = MODULES[variant](128)
    block.np = sum(p.numel() for p in block.parameters())
    block.type = f"C2f+{variant}"
    model.yaml = dict(model.yaml, attention_variant=variant, attention_layer=LAYER)
    return model


class AttentionTrainer(DetectionTrainer):
    variant = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_attention(model, self.variant)
        if weights is not None:
            model.load(weights)
        return model


class SETrainer(AttentionTrainer):
    variant = "se"


class CBAMTrainer(AttentionTrainer):
    variant = "cbam"


class CATrainer(AttentionTrainer):
    variant = "ca"


TRAINERS = {"se": SETrainer, "cbam": CBAMTrainer, "ca": CATrainer}
