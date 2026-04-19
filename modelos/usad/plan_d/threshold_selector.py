from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.metrics import f1_score, roc_curve


class ThresholdSelector(Protocol):
    def select(self, scores: np.ndarray, labels: np.ndarray) -> float: ...


class YoudenJSelector:
    """Umbral que maximiza la estadística J de Youden (TPR − FPR)."""

    def select(self, scores: np.ndarray, labels: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        return float(thresholds[best_idx])


class F1OptimalSelector:
    """Umbral que maximiza el F1-score. Barrido lineal sobre cuantiles."""

    def __init__(self, num_candidates: int = 200) -> None:
        self._n = num_candidates

    def select(self, scores: np.ndarray, labels: np.ndarray) -> float:
        scores = np.asarray(scores)
        labels = np.asarray(labels).astype(int)
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, self._n))
        best_t, best_f1 = float(candidates[0]), -1.0
        for t in candidates:
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return best_t
