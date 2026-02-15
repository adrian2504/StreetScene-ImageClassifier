from __future__ import annotations
from typing import Any, Dict, List
import torch.nn as nn
from .baseline_cnn import BaselineCNN
from .transfer import build_transfer_model

def build_model(cfg: Dict[str, Any], labels: List[str]) -> nn.Module:
    mcfg = cfg["model"]
    name = mcfg["name"]
    num_classes = len(labels) if mcfg.get("num_classes", "auto") == "auto" else int(mcfg["num_classes"])

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=float(mcfg.get("dropout", 0.2)))
    else:
        return build_transfer_model(
            name=name,
            num_classes=num_classes,
            pretrained=bool(mcfg.get("pretrained", True)),
            freeze_backbone=bool(mcfg.get("freeze_backbone", False)),
        )
