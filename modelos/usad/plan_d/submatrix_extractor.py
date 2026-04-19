from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .sensor_selector import SensorSelector, ZerothSensor


@dataclass
class SubmatrixWeights:
    """Pesos listos para inicializar un ``SingleChannelUSAD``.

    Todos los tensores viven en CPU; el adapter los mueve al device que
    corresponda.
    """

    encoder_state: dict
    decoder1_state: dict
    decoder2_state: dict
    chosen_sensor: int


class SubmatrixExtractor:
    """Extrae los pesos de un solo sensor del checkpoint pre-entrenado.

    Convención de layout (verificada vs. el notebook USAD.ipynb): la
    entrada plana al encoder es el resultado de hacer ``x.reshape(-1)`` sobre
    una matriz ``(window_size, n_sensors)`` en orden C (row-major), por lo que
    ``flat_index = t * n_sensors + s``. Esto permite reinterpretar
    ``linear1.weight`` de forma ``(hidden, window_size * n_sensors)`` como
    ``(hidden, window_size, n_sensors)``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        sensor_selector: SensorSelector | None = None,
    ) -> None:
        self._cfg = model_config
        self._selector = sensor_selector or ZerothSensor()

    def extract(self, checkpoint_path: Path) -> SubmatrixWeights:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._validate(checkpoint)

        n_sensors = self._cfg.pretrained_n_sensors
        w = self._cfg.pretrained_window_size

        k = (
            self._cfg.donor_sensor_index
            if self._cfg.donor_sensor_index is not None
            else self._selector.select(checkpoint, n_sensors, w)
        )
        if not 0 <= k < n_sensors:
            raise ValueError(f"donor_sensor_index {k} out of range [0, {n_sensors})")

        encoder_state = self._slice_encoder(checkpoint["encoder"], w, n_sensors, k)
        decoder1_state = self._slice_decoder(checkpoint["decoder1"], w, n_sensors, k)
        decoder2_state = self._slice_decoder(checkpoint["decoder2"], w, n_sensors, k)

        return SubmatrixWeights(
            encoder_state=encoder_state,
            decoder1_state=decoder1_state,
            decoder2_state=decoder2_state,
            chosen_sensor=k,
        )

    @staticmethod
    def _validate(checkpoint: dict) -> None:
        required_keys = {"encoder", "decoder1", "decoder2"}
        missing = required_keys - set(checkpoint.keys())
        if missing:
            raise ValueError(f"Checkpoint missing keys: {missing}")

    @staticmethod
    def _slice_encoder(state: dict, w: int, n_sensors: int, k: int) -> dict:
        # linear1.weight: (hidden, w*n_sensors) → (hidden, w, n_sensors) → (hidden, w)
        weight = state["linear1.weight"]
        weight_r = weight.view(weight.shape[0], w, n_sensors)
        sliced_w = weight_r[:, :, k].contiguous().clone()

        # bias es independiente del canal → se conserva
        bias = state["linear1.bias"].clone()

        new_state = {
            "linear1.weight": sliced_w,
            "linear1.bias": bias,
            "linear2.weight": state["linear2.weight"].clone(),
            "linear2.bias": state["linear2.bias"].clone(),
            "linear3.weight": state["linear3.weight"].clone(),
            "linear3.bias": state["linear3.bias"].clone(),
        }
        return new_state

    @staticmethod
    def _slice_decoder(state: dict, w: int, n_sensors: int, k: int) -> dict:
        # linear3.weight: (w*n_sensors, hidden) → (w, n_sensors, hidden) → (w, hidden)
        weight = state["linear3.weight"]
        weight_r = weight.view(w, n_sensors, weight.shape[1])
        sliced_w = weight_r[:, k, :].contiguous().clone()

        # bias: (w*n_sensors,) → (w, n_sensors) → (w,)
        bias = state["linear3.bias"].view(w, n_sensors)[:, k].contiguous().clone()

        new_state = {
            "linear1.weight": state["linear1.weight"].clone(),
            "linear1.bias": state["linear1.bias"].clone(),
            "linear2.weight": state["linear2.weight"].clone(),
            "linear2.bias": state["linear2.bias"].clone(),
            "linear3.weight": sliced_w,
            "linear3.bias": bias,
        }
        return new_state
