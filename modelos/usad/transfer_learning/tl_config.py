from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TLConfig:
    csv_path: str = "data/plan_a/68.csv"
    checkpoint_path: str = "model.pth"

    window_size: int = 12
    n_channels: int = 1
    pretrained_n_sensors: int = 51
    pretrained_w_size: int = 612
    z_size: int = 120

    sensor_idx: int = 1
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-4
    alpha: float = 0.5
    beta: float = 0.5
    seed: int = 42

    split_values: Tuple[str, str, str] = field(default_factory=lambda: ("E", "T", "V"))

    @property
    def w_size(self) -> int:
        return self.window_size * self.n_channels
