from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(name: str, optimizer, epochs: int):
    name = (name or "").lower()
    if name in ("", "none", None):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")
