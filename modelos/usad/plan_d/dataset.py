from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TemperatureWindowDataset(Dataset):
    """Dataset de ventanas de temperatura para USAD single-channel.

    Cada item es un tensor ``(window_size,)`` ya escalado. Se conservan las
    labels por ventana para evaluación; no se pasan al modelo en training.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray | None = None) -> None:
        if labels is not None and len(windows) != len(labels):
            raise ValueError("windows and labels must have the same length")
        self._x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        self._y = (
            torch.from_numpy(np.asarray(labels, dtype=np.int64))
            if labels is not None
            else None
        )

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int):
        # Devuelve [x] para mantener compatibilidad con el loop original
        # del USAD (`for [batch] in loader`).
        return [self._x[idx]]

    @property
    def labels(self) -> np.ndarray | None:
        return None if self._y is None else self._y.numpy()

    @property
    def tensor(self) -> torch.Tensor:
        return self._x
