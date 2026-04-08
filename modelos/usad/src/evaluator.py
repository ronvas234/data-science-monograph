"""
Principios SOLID aplicados:
- S: AnomalyEvaluator solo se ocupa de scoring, threshold y métricas
- I: Métodos separados para scoring (compute_scores), threshold (find_threshold)
     y métricas (compute_metrics) — el notebook usa solo lo que necesita
- D: recibe el modelo y el loader por inyección
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class AnomalyEvaluator:
    """Calcula scores de anomalía, busca threshold óptimo y reporta métricas."""

    def __init__(self, device: torch.device, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            device: dispositivo de cómputo
            alpha: peso del error de decoder1 en el score
            beta: peso del error de decoder2 (chained) en el score
        """
        self.device = device
        self.alpha = alpha
        self.beta = beta

    def compute_scores(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """
        Calcula scores de anomalía por ventana.

        Score = alpha * MSE(data, w1) + beta * MSE(data, w2_chained)

        Args:
            model: UsadModel entrenado
            loader: DataLoader que retorna (data,) o (data, mask)

        Returns:
            array de shape (N,) con scores de anomalía
        """
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                data = batch[0].to(self.device)
                w1 = model.decoder1(model.encoder(data))
                w2 = model.decoder2(model.encoder(w1))
                score = (
                    self.alpha * torch.mean((data - w1) ** 2, dim=1)
                    + self.beta * torch.mean((data - w2) ** 2, dim=1)
                )
                scores.append(score.cpu().numpy())
        return np.concatenate(scores)

    def find_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """
        Busca el threshold que maximiza la métrica indicada sobre datos de validación.

        Args:
            scores: array de scores de anomalía
            labels: array binario (0=normal, 1=anomalía)
            metric: 'f1' (default) o 'accuracy'

        Returns:
            threshold óptimo (float)
        """
        thresholds = np.linspace(scores.min(), scores.max(), num=500)
        best_threshold = thresholds[0]
        best_score = -1.0

        for t in thresholds:
            preds = (scores >= t).astype(int)
            if metric == "f1":
                val = f1_score(labels, preds, zero_division=0)
            else:
                val = accuracy_score(labels, preds)
            if val > best_score:
                best_score = val
                best_threshold = t

        return float(best_threshold)

    def compute_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> Dict:
        """
        Calcula las métricas de evaluación requeridas.

        Args:
            scores: array de scores de anomalía
            labels: etiquetas reales (0=normal, 1=anomalía)
            threshold: umbral de clasificación

        Returns:
            dict con accuracy, precision, recall, f1, confusion_matrix
        """
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, preds)

        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "confusion_matrix": cm,
            "threshold": threshold,
            "n_anomalies_predicted": int(preds.sum()),
            "n_anomalies_real": int(labels.sum()),
        }
