from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register()

    def _register(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations")

        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
