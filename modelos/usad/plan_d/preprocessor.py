from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerPersistable:
    """Envoltorio fino sobre ``MinMaxScaler`` que se persiste con joblib.

    Se ajusta únicamente sobre el split de entrenamiento para evitar data leakage.
    """

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, values: np.ndarray) -> "MinMaxScalerPersistable":
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        self._scaler.fit(values)
        self._fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        return self._scaler.transform(values).reshape(-1)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)

    @classmethod
    def load(cls, path: Path) -> "MinMaxScalerPersistable":
        obj = cls()
        obj._scaler = joblib.load(path)
        obj._fitted = True
        return obj
