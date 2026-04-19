from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            rows.append(
                {
                    "split": name,
                    "rows": len(df),
                    "anomalies": int(df["flag"].sum()),
                    "anomaly_rate_pct": round(100.0 * df["flag"].mean(), 3),
                    "start": df.iloc[0, 0] if len(df) else None,
                    "end": df.iloc[-1, 0] if len(df) else None,
                }
            )
        return pd.DataFrame(rows)


class ChronologicalSplitter:
    """Reparticiona el DataFrame según las reglas del Plan D.

    - Train: el tramo continuo más temprano (sin gap grande posterior).
    - Val:   el primer ``val_fraction_of_test_segment`` del tramo posterior al gap.
    - Test:  el resto del tramo posterior al gap.

    El gap se detecta como el salto temporal más grande de la serie.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self._cfg = data_config

    def split(self, df: pd.DataFrame) -> SplitResult:
        ts_col = self._cfg.timestamp_column
        df = df.sort_values(ts_col).reset_index(drop=True)

        diffs = df[ts_col].diff()
        gap_idx = int(np.argmax(diffs.values.astype("timedelta64[s]").astype(np.int64)))
        if gap_idx <= 0 or gap_idx >= len(df):
            raise ValueError("Could not detect a temporal gap to split on.")

        pre_gap = df.iloc[:gap_idx].reset_index(drop=True)
        post_gap = df.iloc[gap_idx:].reset_index(drop=True)

        val_size = int(len(post_gap) * self._cfg.val_fraction_of_test_segment)
        val = post_gap.iloc[:val_size].reset_index(drop=True)
        test = post_gap.iloc[val_size:].reset_index(drop=True)

        return SplitResult(train=pre_gap, val=val, test=test)
