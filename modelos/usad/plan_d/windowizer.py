from __future__ import annotations

import numpy as np


class SlidingWindowizer:
    """Convierte una serie 1D en una matriz de ventanas deslizantes.

    Produce un tensor de forma ``(N, window_size)`` donde
    ``N = (len(series) - window_size) // stride + 1``.
    """

    def __init__(self, window_size: int, stride: int = 1) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._w = window_size
        self._s = stride

    @property
    def window_size(self) -> int:
        return self._w

    def transform(self, series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=np.float32).reshape(-1)
        n_windows = (len(series) - self._w) // self._s + 1
        if n_windows <= 0:
            raise ValueError(f"Series length {len(series)} shorter than window size {self._w}")
        idx = np.arange(self._w)[None, :] + self._s * np.arange(n_windows)[:, None]
        return series[idx].astype(np.float32)

    def window_labels(self, flags: np.ndarray) -> np.ndarray:
        """Etiqueta cada ventana como anómala si contiene al menos un flag=1."""
        flags = np.asarray(flags, dtype=np.int64).reshape(-1)
        windowed = self.transform(flags.astype(np.float32))
        return (windowed.sum(axis=1) > 0).astype(np.int64)
