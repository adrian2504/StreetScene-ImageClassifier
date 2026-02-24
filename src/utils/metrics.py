from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

@dataclass
class EvalResult:
    metrics: Dict[str, float]
    cm: np.ndarray
    report_text: str

def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> EvalResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report = classification_report(
    y_true, y_pred, target_names=labels, digits=4, zero_division=0
)
    return EvalResult(metrics={"accuracy": acc, "macro_f1": macro_f1}, cm=cm, report_text=report)
