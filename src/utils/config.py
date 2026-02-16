from __future__ import annotations
import os
import yaml
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def env_override(cfg: Dict[str, Any]) -> Dict[str, Any]:
  
    root = os.environ.get("ARTIFACTS_ROOT")
    if root:
        cfg.setdefault("output", {})
        cfg["output"]["root_dir"] = root
    return cfg
