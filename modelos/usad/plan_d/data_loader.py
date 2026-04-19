from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


class SiataCsvLoader:
    """Lee un CSV de SIATA y devuelve un DataFrame normalizado.

    Responsabilidad única: leer el archivo, parsear fechas, descartar la
    columna Split original (el usuario la exige ignorar) y ordenar por tiempo.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self._cfg = data_config

    def load(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df[self._cfg.timestamp_column] = pd.to_datetime(df[self._cfg.timestamp_column])
        df = df.sort_values(self._cfg.timestamp_column).reset_index(drop=True)

        if self._cfg.split_column_to_discard in df.columns:
            df = df.drop(columns=[self._cfg.split_column_to_discard])

        required = {self._cfg.timestamp_column, self._cfg.value_column, self._cfg.anomaly_flag_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        df[self._cfg.anomaly_flag_column] = (df[self._cfg.anomaly_flag_column] != 0).astype(int)
        return df
