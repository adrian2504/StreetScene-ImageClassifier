# RoadRiskVision (v0.1)

End-to-end **street/road hazard image classifier** 
data → preprocessing → training (3 models) → optimization → evaluation →  UI.

## 1) Quickstart

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add images
Put images in:
```
data/raw/<class_name>/*.jpg
```
Example classes:
- pothole
- debris
- flooded_road
- construction_zone
- normal_road
- I will add some more random images

### Build manifest + split
```bash
python -m src.data.make_manifest --raw_dir data/raw --out data/dataset_manifest.csv
python -m src.data.split --manifest data/dataset_manifest.csv --out data/dataset_manifest.csv
```

### Train baseline CNN
```bash
python -m src.training.train --config configs/baseline.yaml
```

### Evaluate a checkpoint
```bash
python -m src.training.evaluate --run_dir artifacts/latest
```

### Run UI (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

## 2) Repo layout
- `src/data/` : manifest, split, preprocessing utils
- `src/models/` : baseline + transfer models
- `src/training/` : train/eval loops + metrics + artifacts
- `app/` : Streamlit demo (upload → predict)

## 3) Artifacts
Training writes into `artifacts/`:
- `best.pt`, `last.pt`
- `labels.json`
- `metrics.json`
- `confusion_matrix.png`

## 4) Notes
- This v0.1 is classification-only
