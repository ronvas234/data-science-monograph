from __future__ import annotations

from typing import Protocol

import torch


class SensorSelector(Protocol):
    """Estrategia para escoger cuál de los ``n_sensors`` donantes usamos."""

    def select(self, state_dict: dict, n_sensors: int, window_size: int) -> int: ...


class ZerothSensor:
    """Selector trivial: siempre elige el sensor 0. Útil como baseline."""

    def select(self, state_dict: dict, n_sensors: int, window_size: int) -> int:
        return 0


class StatsBasedSensor:
    """Elige el sensor cuyas columnas tienen la menor norma L2 media.

    Heurística pragmática: columnas con menor norma suelen corresponder
    a sensores con dinámica más suave, que se parecen más a una serie de
    temperatura diaria.
    """

    def select(self, state_dict: dict, n_sensors: int, window_size: int) -> int:
        w = state_dict["encoder"]["linear1.weight"]
        # (306, 12*51) → (306, 12, 51) → norma por sensor
        w_reshaped = w.view(w.shape[0], window_size, n_sensors)
        per_sensor_norm = torch.linalg.vector_norm(w_reshaped, dim=(0, 1))
        return int(torch.argmin(per_sensor_norm).item())
