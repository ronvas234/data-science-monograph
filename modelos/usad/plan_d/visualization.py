from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluator import ROCResult


def plot_roc(train_roc: ROCResult, val_roc: ROCResult, title: str = "ROC — Train vs Val"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(train_roc.fpr, train_roc.tpr, color="#3366CC", lw=2,
            label=f"Train (AUC={train_roc.auc:.3f})")
    ax.plot(val_roc.fpr, val_roc.tpr, color="#DC3912", lw=2,
            label=f"Val (AUC={val_roc.auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_loss_history(history):
    epochs = [h.epoch for h in history]
    loss1 = [h.val_loss1 for h in history]
    loss2 = [h.val_loss2 for h in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, loss1, "-o", label="val_loss1")
    ax.plot(epochs, loss2, "-o", label="val_loss2")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Pérdida de validación vs. época")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_score_histogram(scores: np.ndarray, labels: np.ndarray, threshold: float):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        [scores[labels == 0], scores[labels == 1]],
        bins=60, stacked=True,
        color=["#82E0AA", "#EC7063"],
        label=["Normal", "Anomalía"],
    )
    ax.axvline(threshold, color="black", linestyle="--", label=f"thr={threshold:.4f}")
    ax.set_xlabel("Score de anomalía")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de scores por clase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Matriz de confusión"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomalía"],
        yticklabels=["Normal", "Anomalía"],
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig


def plot_anomaly_timeline(
    timestamps: pd.Series,
    values: np.ndarray,
    true_labels_per_timestamp: np.ndarray,
    predicted_labels_per_window: np.ndarray,
    window_size: int,
):
    """Serie temporal con anomalías reales (verdes) y predichas (rojas).

    Cada ventana predicha se mapea al timestamp del último elemento de la ventana.
    """
    window_end_idx = np.arange(len(predicted_labels_per_window)) + window_size - 1
    predicted_ts = timestamps.iloc[window_end_idx].values
    predicted_anom_mask = predicted_labels_per_window == 1
    true_anom_mask = true_labels_per_timestamp == 1

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(timestamps, values, color="#3366CC", lw=0.8, label="Temperatura")
    ax.scatter(
        timestamps[true_anom_mask], values[true_anom_mask],
        color="green", s=18, label="Anomalía real (flag)", zorder=3,
    )
    ax.scatter(
        predicted_ts[predicted_anom_mask],
        values[window_end_idx[predicted_anom_mask]],
        color="red", s=14, marker="x", label="Anomalía predicha", zorder=4,
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Detección de anomalías — test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig
