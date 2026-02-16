from __future__ import annotations
import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_manifest, ManifestImageDataset
from src.models.factory import build_model
from src.utils.metrics import compute_metrics
from src.utils.viz import save_confusion_matrix
from src.utils.config import load_yaml
from src.utils.seed import set_seed

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., artifacts/latest or artifacts/<run_name_ts>")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}. Did you point to the right run dir?")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(int(cfg.get("seed", 42)))
    spec = load_manifest(cfg["data"]["manifest_csv"])
    labels = spec.labels

    df = spec.df
    df_split = df[df["split"] == args.split].copy()
    ds = ManifestImageDataset(df_split, labels, image_size=int(cfg["data"]["image_size"]), augment=False)
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]))

    model = build_model(cfg, labels)
    dev = get_device()
    model.to(dev)

    ckpt_path = os.path.join(run_dir, "best.pt") if os.path.exists(os.path.join(run_dir, "best.pt")) else os.path.join(run_dir, "last.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(dev)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.cpu().tolist())

    res = compute_metrics(y_true, y_pred, labels)
    out = {
        "split": args.split,
        "checkpoint": os.path.basename(ckpt_path),
        "metrics": res.metrics,
    }
    with open(os.path.join(run_dir, f"eval_{args.split}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(run_dir, f"eval_{args.split}_report.txt"), "w", encoding="utf-8") as f:
        f.write(res.report_text)
    save_confusion_matrix(res.cm, labels, os.path.join(run_dir, f"confusion_matrix_{args.split}.png"))

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
