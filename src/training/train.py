from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_yaml, env_override
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics
from src.utils.viz import save_confusion_matrix
from src.data.dataset import load_manifest, ManifestImageDataset
from src.models.factory import build_model
from src.training.optim import build_optimizer, build_scheduler
from src.training.early_stop import EarlyStopping

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_run_dir(root: str, run_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{run_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    latest = os.path.join(root, "latest")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        elif os.path.exists(latest):
            # if it's a folder, donot delete it
            pass
        os.symlink(os.path.basename(run_dir), latest)
    except Exception:
        with open(os.path.join(root, "LATEST_RUN.txt"), "w", encoding="utf-8") as f:
            f.write(run_dir)

    return run_dir

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, labels, dev: torch.device):
    if len(loader.dataset) == 0:
        return None

    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(dev)
        y = y.to(dev)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    res = compute_metrics(y_true, y_pred, labels)
    return {"metrics": res.metrics, "cm": res.cm, "report": res.report_text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = env_override(load_yaml(args.config))
    set_seed(int(cfg.get("seed", 42)))

    dev = get_device()
    pin = (dev.type == "cuda")

    spec = load_manifest(cfg["data"]["manifest_csv"])
    labels = spec.labels

    image_size = int(cfg["data"].get("image_size", 224))
    bs = int(cfg["data"].get("batch_size", 32))
    nw = int(cfg["data"].get("num_workers", 2))

    df = spec.df
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    train_ds = ManifestImageDataset(df_train, labels, image_size=image_size, augment=True)
    val_ds = ManifestImageDataset(df_val, labels, image_size=image_size, augment=False)
    test_ds = ManifestImageDataset(df_test, labels, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_model(cfg, labels).to(dev)
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.0))
    opt_name = cfg["train"].get("optimizer", "adamw")
    sch_name = cfg["train"].get("scheduler", "none")

    optimizer = build_optimizer(opt_name, model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_scheduler(sch_name, optimizer, epochs=epochs)

    es_cfg = cfg["train"].get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    early = EarlyStopping(
        patience=int(es_cfg.get("patience", 3)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    ) if es_enabled else None

    out_root = cfg.get("output", {}).get("root_dir", "artifacts")
    os.makedirs(out_root, exist_ok=True)
    run_dir = make_run_dir(out_root, cfg.get("run_name", "run"))

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(run_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    has_val = len(val_ds) > 0
    has_test = len(test_ds) > 0

    best_metric = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(dev)
            y = y.to(dev)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * x.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        train_loss = running / max(1, len(train_ds))

        # Validation
        tracked = None
        val_out = run_eval(model, val_loader, labels, dev) if has_val else None
        if val_out is not None:
            tracked = float(val_out["metrics"].get("macro_f1", 0.0))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_out["metrics"] if val_out else {},
            "lr": optimizer.param_groups[0]["lr"],
        })
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save best only if val exists
        if tracked is not None and tracked > best_metric:
            best_metric = tracked
            torch.save({"model_state": model.state_dict(), "labels": labels}, os.path.join(run_dir, "best.pt"))
            with open(os.path.join(run_dir, "best_val_report.txt"), "w", encoding="utf-8") as f:
                f.write(val_out["report"])
            save_confusion_matrix(val_out["cm"], labels, os.path.join(run_dir, "confusion_matrix_val.png"))

        # Always save last
        torch.save({"model_state": model.state_dict(), "labels": labels}, os.path.join(run_dir, "last.pt"))

        if early is not None and tracked is not None and early.step(tracked):
            break

    # Final evaluation
    ckpt_path = os.path.join(run_dir, "best.pt") if os.path.exists(os.path.join(run_dir, "best.pt")) else os.path.join(run_dir, "last.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])

    metrics_out = {
        "best_val_macro_f1": best_metric if has_val else None,
        "checkpoint_used": os.path.basename(ckpt_path),
        "test": {},
        "note": "",
    }

    if has_test:
        test_out = run_eval(model, test_loader, labels, dev)
        metrics_out["test"] = test_out["metrics"] if test_out else {}
        if test_out is not None:
            with open(os.path.join(run_dir, "test_report.txt"), "w", encoding="utf-8") as f:
                f.write(test_out["report"])
            save_confusion_matrix(test_out["cm"], labels, os.path.join(run_dir, "confusion_matrix_test.png"))
    else:
        metrics_out["note"] = "test split empty; add more data or adjust split ratios to enable test evaluation."

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Run saved to: {run_dir}")

if __name__ == "__main__":
    main()