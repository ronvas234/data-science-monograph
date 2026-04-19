from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluator import ROCEvaluator
from .model_adapter import SingleChannelUSAD


@dataclass
class DetectionResult:
    scores: np.ndarray
    predictions: np.ndarray  # 0/1 per window
    threshold: float


class AnomalyDetector:
    """Clase de alto nivel: score + threshold → decisión.

    Depende de un ``ROCEvaluator`` inyectado (DIP).
    """

    def __init__(self, model: SingleChannelUSAD, evaluator: ROCEvaluator, threshold: float) -> None:
        self._model = model
        self._eval = evaluator
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def detect(self, loader: DataLoader) -> DetectionResult:
        scores = self._eval.scores(self._model, loader)
        preds = (scores >= self._threshold).astype(int)
        return DetectionResult(scores=scores, predictions=preds, threshold=self._threshold)
