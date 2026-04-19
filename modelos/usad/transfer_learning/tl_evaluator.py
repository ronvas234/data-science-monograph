from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from usad import UsadModel, testing


class AnomalyScorer:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def score(self, model: UsadModel, loader: DataLoader) -> np.ndarray:
        model.eval()
        results = testing(model, loader, alpha=self.alpha, beta=self.beta)
        flat = []
        for r in results:
            flat.append(r.detach().cpu().numpy().reshape(-1))
        return np.concatenate(flat) if flat else np.array([])


class ThresholdSelector:
    """Selects a threshold from a ROC curve at the operating point TPR = 1 - FPR."""

    def select(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float(np.median(y_score))
        fpr, tpr, thr = roc_curve(y_true, y_score)
        idx = np.argmin(np.abs(tpr - (1 - fpr)))
        return float(thr[idx])


@dataclass
class SplitMetrics:
    split_name: str
    n_samples: int
    n_positive: int
    accuracy: float
    f1: float
    auc: float

    def as_row(self) -> dict:
        return {
            "split": self.split_name,
            "n": self.n_samples,
            "positives": self.n_positive,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "auc": round(self.auc, 4),
        }


class MetricsReporter:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def evaluate(
        self, split_name: str, y_true: np.ndarray, y_score: np.ndarray
    ) -> SplitMetrics:
        y_pred = (y_score >= self.threshold).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")
        return SplitMetrics(
            split_name=split_name,
            n_samples=len(y_true),
            n_positive=int(y_true.sum()),
            accuracy=float(acc),
            f1=float(f1),
            auc=float(auc),
        )
