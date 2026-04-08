"""
Responsabilidad única (S): normalización z-score sobre series de temperatura.
Nunca usa MinMaxScaler. El valor centinela -2000 se excluye del fit pero se
preserva en la transformación para que la lógica de masking pueda detectarlo.
"""
import pandas as pd
import numpy as np


class ZScoreNormalizer:
    """Normalización z-score: (x - mean) / std.

    El ajuste (fit) se realiza SOLO sobre datos de entrenamiento limpios
    (t_mask != -2000) para no contaminar las estadísticas con anomalías.
    """

    def __init__(self):
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, series: pd.Series) -> "ZScoreNormalizer":
        """
        Calcula media y desviación estándar excluyendo el centinela -2000.

        Args:
            series: Serie con valores de temperatura (solo datos E normales)

        Returns:
            self (para encadenamiento)
        """
        clean = series[series != -2000.0].dropna()
        if len(clean) == 0:
            raise ValueError("No hay datos limpios para ajustar el normalizador.")
        self.mean_ = float(clean.mean())
        self.std_ = float(clean.std())
        if self.std_ == 0.0:
            raise ValueError("Desviación estándar es 0 — serie constante.")
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Aplica z-score. Los valores -2000 NO se normalizan y se preservan
        para que PlanBMaskBuilder los detecte correctamente.

        Args:
            series: Serie a transformar

        Returns:
            Serie normalizada (sentinelas -2000 intactos)
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Llamar fit() antes de transform().")

        sentinel_mask = series == -2000.0
        normalized = (series - self.mean_) / self.std_
        normalized[sentinel_mask] = -2000.0  # restaurar centinelas
        return normalized

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Ajusta y transforma en un solo paso."""
        return self.fit(series).transform(series)
