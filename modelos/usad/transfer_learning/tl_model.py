from typing import Dict

import torch
import torch.nn as nn

from usad import Decoder, Encoder, UsadModel


class TransferEncoder(nn.Module):
    """Encoder that preserves the pretrained hidden dims (306 -> 153 -> latent)
    while accepting a smaller input (w_size = window_size * n_channels).
    """

    def __init__(self, in_size: int, hidden1: int, hidden2: int, latent_size: int):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        return self.relu(self.linear3(out))


class TransferDecoder(nn.Module):
    """Decoder mirroring the pretrained hidden dims (latent -> 153 -> 306 -> out_size)."""

    def __init__(self, latent_size: int, hidden2: int, hidden1: int, out_size: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, hidden2)
        self.linear2 = nn.Linear(hidden2, hidden1)
        self.linear3 = nn.Linear(hidden1, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        return self.sigmoid(self.linear3(out))


class SubMatrixWeightAdapter:
    """Adapts a USAD SWaT checkpoint (51 sensors, w_size=612) to a single-channel
    model (w_size=12) by slicing encoder.linear1 (columns) and decoder.linear3
    (rows + bias) to the chosen sensor's 12 positions.

    Flatten order in the source notebook: view(batch, 12, 51) -> (batch, 612)
    => column k + s*51 of encoder.linear1.weight corresponds to timestep s, sensor k.
    """

    def __init__(
        self,
        checkpoint_path: str,
        sensor_idx: int,
        n_sensors: int = 51,
        window_size: int = 12,
        map_location: str = "cpu",
    ):
        if not 0 <= sensor_idx < n_sensors:
            raise ValueError(f"sensor_idx must be in [0, {n_sensors})")
        self.checkpoint_path = checkpoint_path
        self.sensor_idx = sensor_idx
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.map_location = map_location

    def _sensor_columns(self) -> torch.Tensor:
        return torch.tensor(
            [self.sensor_idx + s * self.n_sensors for s in range(self.window_size)],
            dtype=torch.long,
        )

    def build_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        ckpt = torch.load(self.checkpoint_path, map_location=self.map_location)
        cols = self._sensor_columns()

        encoder = {k: v.clone() for k, v in ckpt["encoder"].items()}
        encoder["linear1.weight"] = encoder["linear1.weight"][:, cols].contiguous()

        adapted = {"encoder": encoder}
        for dec_key in ("decoder1", "decoder2"):
            dec = {k: v.clone() for k, v in ckpt[dec_key].items()}
            dec["linear3.weight"] = dec["linear3.weight"][cols, :].contiguous()
            dec["linear3.bias"] = dec["linear3.bias"][cols].contiguous()
            adapted[dec_key] = dec
        return adapted


class TransferUsadModel(UsadModel):
    """Single-channel USAD whose hidden dims match the pretrained 51-ch. model,
    so pretrained weights (after sub-matrix slicing of the boundary layers) can
    be loaded directly. Substitutable for `UsadModel` (Liskov): inherits all
    training_step / validation_step / *_epoch_end methods.
    """

    def __init__(
        self,
        w_size: int,
        z_size: int,
        hidden1: int = 306,
        hidden2: int = 153,
    ):
        super().__init__(w_size, z_size)
        # Replace the default halving-based layers with ones shaped to the pretrained dims.
        self.encoder = TransferEncoder(w_size, hidden1, hidden2, z_size)
        self.decoder1 = TransferDecoder(z_size, hidden2, hidden1, w_size)
        self.decoder2 = TransferDecoder(z_size, hidden2, hidden1, w_size)

    def load_adapted(self, adapted_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.encoder.load_state_dict(adapted_state["encoder"])
        self.decoder1.load_state_dict(adapted_state["decoder1"])
        self.decoder2.load_state_dict(adapted_state["decoder2"])


# Re-export for convenience.
__all__ = [
    "SubMatrixWeightAdapter",
    "TransferEncoder",
    "TransferDecoder",
    "TransferUsadModel",
    "Encoder",
    "Decoder",
]
