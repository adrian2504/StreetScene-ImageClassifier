from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

def build_transfer_model(name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("fc.")]
        head_params = list(m.fc.parameters())
    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("fc.")]
        head_params = list(m.fc.parameters())
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in m.named_parameters() if not n.startswith("classifier.")]
        head_params = list(m.classifier.parameters())
    else:
        raise ValueError(f"Unknown transfer model: {name}")

    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    m._backbone_params = backbone_params  # type: ignore[attr-defined]
    m._head_params = head_params          # type: ignore[attr-defined]
    return m
