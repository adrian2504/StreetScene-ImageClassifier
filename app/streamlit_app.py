from __future__ import annotations
import json
import os
from io import BytesIO
from typing import List, Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm

from src.models.factory import build_model
from src.explain.gradcam import GradCAM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_run(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    labels_path = os.path.join(run_dir, "labels.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    ckpt_path = os.path.join(run_dir, "best.pt") if os.path.exists(os.path.join(run_dir, "best.pt")) else os.path.join(run_dir, "last.pt")

    if not os.path.exists(cfg_path) or not os.path.exists(labels_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Run dir missing required files (config.json, labels.json, best.pt/last.pt)")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    dev = get_device()
    model = build_model(cfg, labels)
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    model.eval()

    image_size = int(cfg["data"].get("image_size", 224))
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, labels, tfm, dev, cfg, metrics, ckpt_path

def risk_score(probs: List[float], labels: List[str]) -> float:
    weights = {l: 0.2 for l in labels}
    for k in ["pothole", "debris", "flooded_road", "construction_zone"]:
        if k in weights:
            weights[k] = 1.0
    s = 0.0
    for p, l in zip(probs, labels):
        s += p * weights.get(l, 0.2)
    return float(min(1.0, max(0.0, s)))

def find_last_conv(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def overlay_cam_on_image(img: Image.Image, cam_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    cam_map = np.clip(cam_map, 0, 1)
    heat = cm.get_cmap("jet")(cam_map)[:, :, :3]  # RGB
    heat = (heat * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(img.size)

    base = img.convert("RGB")
    blended = Image.blend(base, heat_img, alpha=alpha)
    return blended

st.set_page_config(page_title="StreetScene Classifier", layout="centered")
st.title("StreetScene / RoadRiskVision — Classifier (v0.2)")

run_dir = st.text_input("Run directory", value="artifacts/latest")
uploaded = st.file_uploader("Upload a street/road image", type=["jpg", "jpeg", "png", "webp"])

col1, col2 = st.columns(2)
with col1:
    load_btn = st.button("Load model")
with col2:
    show_cam = st.checkbox("Show Grad-CAM (why?)", value=False)

if load_btn:
    try:
        model, labels, tfm, dev, cfg, metrics, ckpt_path = load_run(run_dir)
        st.session_state.update({
            "model": model,
            "labels": labels,
            "tfm": tfm,
            "dev": dev,
            "cfg": cfg,
            "metrics": metrics,
            "ckpt_path": ckpt_path
        })
        st.success(f"Loaded: {ckpt_path}")
    except Exception as e:
        st.error(str(e))

if "metrics" in st.session_state and st.session_state["metrics"]:
    st.subheader("Latest run metrics")
    st.json(st.session_state["metrics"])

    cm_val = os.path.join(run_dir, "confusion_matrix_val.png")
    cm_test = os.path.join(run_dir, "confusion_matrix_test.png")
    if os.path.exists(cm_val):
        st.image(cm_val, caption="Confusion Matrix (val)")
    if os.path.exists(cm_test):
        st.image(cm_test, caption="Confusion Matrix (test)")

if uploaded is not None and "model" in st.session_state:
    img = Image.open(BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Input")

    x = st.session_state["tfm"](img).unsqueeze(0).to(st.session_state["dev"])

    with torch.no_grad():
        logits = st.session_state["model"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())

    labels = st.session_state["labels"]
    pred_label = labels[pred_idx]

    st.subheader(f"Prediction: {pred_label}")
    st.write({"risk_score": risk_score(probs, labels)})

    top = sorted(list(zip(labels, probs)), key=lambda t: t[1], reverse=True)[: min(5, len(labels))]
    st.table([{"label": l, "prob": float(p)} for l, p in top])

    # Grad-CAM
    if show_cam:
        target = find_last_conv(st.session_state["model"])
        if target is None:
            st.warning("Could not find a Conv2d layer for Grad-CAM.")
        else:
            gradcam = GradCAM(st.session_state["model"], target)
            cam_t = gradcam(x, class_idx=pred_idx) 
            cam_np = cam_t.squeeze().detach().cpu().numpy()

            # Resize cam to original image size for overlay
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
            cam_np = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize(img.size)) / 255.0

            blended = overlay_cam_on_image(img, cam_np)
            st.image(blended, caption="Grad-CAM overlay")
else:
    st.caption("Tip: Train first, then set run directory to artifacts/latest and click Load model.")