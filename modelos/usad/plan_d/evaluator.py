from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .model_adapter import SingleChannelUSAD


@dataclass
class ROCResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float
    confusion: np.ndarray  # 2x2: [[TN, FP], [FN, TP]]


class ROCEvaluator:
    """Calcula scores de anomalía y ROC sobre un DataLoader."""

    def __init__(self, device: torch.device, alpha: float = 0.5, beta: float = 0.5) -> None:
        self._device = device
        self._alpha = alpha
        self._beta = beta

    @torch.no_grad()
    def scores(self, model: SingleChannelUSAD, loader: DataLoader) -> np.ndarray:
        model.eval().to(self._device)
        out = []
        for [batch] in loader:
            batch = batch.to(self._device, non_blocking=True)
            s = model.anomaly_score(batch, alpha=self._alpha, beta=self._beta)
            out.append(s.detach().cpu().numpy())
        return np.concatenate(out)

    def roc(self, scores: np.ndarray, labels: np.ndarray) -> ROCResult:
        if len(np.unique(labels)) < 2:
            return ROCResult(
                fpr=np.array([0.0, 1.0]),
                tpr=np.array([0.0, 1.0]),
                thresholds=np.array([np.inf, -np.inf]),
                auc=float("nan"),
            )
        fpr, tpr, thr = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        return ROCResult(fpr=fpr, tpr=tpr, thresholds=thr, auc=float(auc))


class ClassificationEvaluator:
    """Evalúa métricas dado un umbral fijo."""

    def evaluate(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float
    ) -> ClassificationMetrics:
        preds = (scores >= threshold).astype(int)
        labels = labels.astype(int)
        auc = (
            roc_auc_score(labels, scores)
            if len(np.unique(labels)) >= 2
            else float("nan")
        )
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        return ClassificationMetrics(
            accuracy=float(accuracy_score(labels, preds)),
            f1=float(f1_score(labels, preds, zero_division=0)),
            precision=float(precision_score(labels, preds, zero_division=0)),
            recall=float(recall_score(labels, preds, zero_division=0)),
            auc=float(auc),
            confusion=cm,
        )
