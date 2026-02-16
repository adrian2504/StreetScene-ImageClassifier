from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
import json

RUN_RE = re.compile(r"Run saved to:\s*(.*)$")

def run_one(config_path: str) -> str:
    cmd = [sys.executable, "-m", "src.training.train", "--config", config_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Training failed for {config_path}\n\n{out}")

    run_dir = None
    for line in out.splitlines():
        m = RUN_RE.search(line.strip())
        if m:
            run_dir = m.group(1).strip()
    if not run_dir:
        raise RuntimeError(f"Could not find run dir in output for {config_path}")
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/experiment_summary.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for cfg in args.configs:
        run_dir = run_one(cfg)
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        rows.append({
            "config": cfg,
            "run_dir": run_dir,
            "best_val_macro_f1": m.get("best_val_macro_f1"),
            "test_accuracy": (m.get("test") or {}).get("accuracy"),
            "test_macro_f1": (m.get("test") or {}).get("macro_f1"),
            "checkpoint_used": m.get("checkpoint_used"),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()  