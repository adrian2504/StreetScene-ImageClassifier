from __future__ import annotations
import json
import os
from io import BytesIO
from typing import List

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.factory import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_run(run_dir: str):

    cfg_path = os.path.join(run_dir, "config.json")
    labels_path = os.path.join(run_dir, "labels.json")
    ckpt_path = os.path.join(run_dir, "best.pt") if os.path.exists(os.path.join(run_dir, "best.pt")) else os.path.join(run_dir, "last.pt")


    if not os.path.exists(cfg_path) or not os.path.exists(labels_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Run dir missing required files (config.json, labels.json, best.pt/last.pt)")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)


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


    return model, labels, tfm, dev

def risk_score(probs: List[float], labels: List[str]) -> float:
    # Simple demo: map hazards to higher weights
    weights = {l: 0.2 for l in labels}
    for k in ["pothole", "debris", "flooded_road", "construction_zone"]:
        if k in weights:
            weights[k] = 1.0

    # hazard score in [0,1]
    s = 0.0
    for p, l in zip(probs, labels):
        s += p * weights.get(l, 0.2)
    return float(min(1.0, max(0.0, s)))

st.set_page_config(page_title="RoadRiskVision", layout="centered")
st.title("RoadRiskVision — Street Hazard Classifier (v0.1)")

run_dir = st.text_input("Run directory", value="artifacts/latest")
uploaded = st.file_uploader("Upload a street/road image", type=["jpg", "jpeg", "png", "webp"])

if st.button("Load model"):
    try:
        model, labels, tfm, dev = load_run(run_dir)
        st.session_state["model"] = model
        st.session_state["labels"] = labels
        st.session_state["tfm"] = tfm
        st.session_state["dev"] = dev
        st.success("Loaded.")
    except Exception as e:
        st.error(str(e))

if uploaded is not None and "model" in st.session_state:
    img = Image.open(BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    x = st.session_state["tfm"](img).unsqueeze(0).to(st.session_state["dev"])
    with torch.no_grad():
        logits = st.session_state["model"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())

    labels = st.session_state["labels"]
    pred_label = labels[pred_idx]
    st.subheader(f"Prediction: {pred_label}")
    st.write({"risk_score": risk_score(probs, labels)})

    # show top-k
    k = min(5, len(labels))
    top = sorted(list(zip(labels, probs)), key=lambda t: t[1], reverse=True)[:k]
    st.write("Top probabilities:")
    st.table([{"label": l, "prob": float(p)} for l, p in top])

else:
    st.caption("Tip: Train first, then set run directory to artifacts/latest and click Load model.")
