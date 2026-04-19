from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Path

    @property
    def usad_root(self) -> Path:
        return self.repo_root / "modelos" / "usad"

    @property
    def pretrained_checkpoint(self) -> Path:
        return self.usad_root / "model.pth"

    @property
    def data_csv(self) -> Path:
        return self.usad_root / "data" / "plan_a" / "68.csv"

    @property
    def plan_d_root(self) -> Path:
        return self.usad_root / "plan_d"

    @property
    def checkpoints_dir(self) -> Path:
        return self.plan_d_root / "checkpoints"

    @property
    def finetuned_checkpoint(self) -> Path:
        return self.checkpoints_dir / "siata_68.pth"

    @property
    def scaler_path(self) -> Path:
        return self.checkpoints_dir / "scaler_siata_68.joblib"


@dataclass(frozen=True)
class DataConfig:
    window_size: int = 12
    stride: int = 1
    val_fraction_of_test_segment: float = 0.30
    anomaly_flag_column: str = "flag"
    timestamp_column: str = "fecha_hora"
    value_column: str = "t"
    split_column_to_discard: str = "Split"


@dataclass(frozen=True)
class ModelConfig:
    pretrained_n_sensors: int = 51
    pretrained_window_size: int = 12
    latent_size: int = 120
    donor_sensor_index: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 256
    lr_border: float = 1e-3
    lr_inner: float = 1e-4
    early_stopping_patience: int = 5
    freeze_inner_layers: bool = True
    alpha: float = 0.5
    beta: float = 0.5
    seed: int = 42
