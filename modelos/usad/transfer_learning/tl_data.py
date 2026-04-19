from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class SiataCsvLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        expected = {"fecha_hora", "t", "flag", "Split"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        return df


class SplitExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def by_split(self, split_value: str, flag_zero_only: bool = False) -> pd.DataFrame:
        subset = self.df[self.df["Split"] == split_value]
        if flag_zero_only:
            subset = subset[subset["flag"] == 0]
        return subset.reset_index(drop=True)


class TemperatureNormalizer:
    def __init__(self):
        self.min_: Optional[float] = None
        self.max_: Optional[float] = None

    def fit(self, values: np.ndarray) -> "TemperatureNormalizer":
        self.min_ = float(np.min(values))
        self.max_ = float(np.max(values))
        if self.max_ == self.min_:
            raise ValueError("Cannot fit normalizer: min equals max.")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Normalizer must be fit before transform.")
        scaled = (values - self.min_) / (self.max_ - self.min_)
        return np.clip(scaled, 0.0, 1.0)


@dataclass
class WindowedData:
    windows: np.ndarray
    labels: np.ndarray
    end_indices: np.ndarray


class WindowBuilder:
    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size

    def build(self, series: np.ndarray, flags: np.ndarray) -> WindowedData:
        if series.shape[0] != flags.shape[0]:
            raise ValueError("series and flags must have equal length")
        n = series.shape[0] - self.window_size + 1
        if n <= 0:
            raise ValueError("Series too short for window_size")
        windows = np.lib.stride_tricks.sliding_window_view(series, self.window_size)[:n]
        flag_windows = np.lib.stride_tricks.sliding_window_view(flags, self.window_size)[:n]
        labels = (flag_windows.sum(axis=1) > 0).astype(np.int64)
        end_indices = np.arange(self.window_size - 1, self.window_size - 1 + n)
        return WindowedData(
            windows=windows.astype(np.float32),
            labels=labels,
            end_indices=end_indices,
        )


def make_loader(
    windows: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    tensor = torch.from_numpy(windows).float()
    flat = tensor.view(tensor.size(0), -1)
    return DataLoader(
        TensorDataset(flat),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def build_splits(
    csv_path: str, window_size: int
) -> Tuple[TemperatureNormalizer, dict]:
    df = SiataCsvLoader(csv_path).load()
    extractor = SplitExtractor(df)
    e_train = extractor.by_split("E", flag_zero_only=True)
    e_all = extractor.by_split("E", flag_zero_only=False)
    t_all = extractor.by_split("T", flag_zero_only=False)
    v_all = extractor.by_split("V", flag_zero_only=False)

    normalizer = TemperatureNormalizer().fit(e_train["t"].to_numpy())
    builder = WindowBuilder(window_size)

    def to_windows(subset: pd.DataFrame) -> Tuple[WindowedData, pd.DataFrame]:
        t = normalizer.transform(subset["t"].to_numpy())
        w = builder.build(t, subset["flag"].to_numpy())
        return w, subset

    return normalizer, {
        "E_train": to_windows(e_train),
        "E_all": to_windows(e_all),
        "T": to_windows(t_all),
        "V": to_windows(v_all),
    }
