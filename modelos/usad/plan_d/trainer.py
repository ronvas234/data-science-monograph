from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model_adapter import SingleChannelUSAD


# =====================================================================
# Strategy pattern: qué parámetros entrenar y con qué LR.
# =====================================================================

class FineTuneStrategy(ABC):
    @abstractmethod
    def build_param_groups(
        self, model: SingleChannelUSAD, cfg: TrainingConfig
    ) -> tuple[list[dict], list[dict]]:
        """Devuelve dos listas de param_groups: una por cada optimizador USAD.

        El optimizador 1 actualiza ``encoder + decoder1``.
        El optimizador 2 actualiza ``encoder + decoder2``.
        """


def _border_params(module, cfg: TrainingConfig) -> list[torch.nn.Parameter]:
    return list(module.linear1.parameters()) + list(module.linear3.parameters())


def _inner_params(module, cfg: TrainingConfig) -> list[torch.nn.Parameter]:
    return list(module.linear2.parameters())


class FreezeInner(FineTuneStrategy):
    """Congela las capas internas (linear2) del encoder y ambos decoders.
    Solo entrena las capas borde que fueron rehechas con la submatriz."""

    def build_param_groups(self, model, cfg):
        for m in [model.encoder, model.decoder1, model.decoder2]:
            for p in _inner_params(m, cfg):
                p.requires_grad = False
            # linear3 del encoder es una capa interna también (153→120) pero
            # se deja entrenable porque no recibió submatriz.
        # También congelamos linear3 de encoder (no afectado por canal).
        for p in model.encoder.linear3.parameters():
            p.requires_grad = False
        # decoder.linear1 (latent→153) tampoco depende del canal → congelar.
        for m in [model.decoder1, model.decoder2]:
            for p in m.linear1.parameters():
                p.requires_grad = False

        groups1 = [
            {"params": _border_params(model.encoder, cfg), "lr": cfg.lr_border},
            {"params": _border_params(model.decoder1, cfg), "lr": cfg.lr_border},
        ]
        groups2 = [
            {"params": _border_params(model.encoder, cfg), "lr": cfg.lr_border},
            {"params": _border_params(model.decoder2, cfg), "lr": cfg.lr_border},
        ]
        return groups1, groups2


class FullFinetune(FineTuneStrategy):
    """Entrena todos los parámetros, con LR distintos para borde vs interior."""

    def build_param_groups(self, model, cfg):
        for p in model.parameters():
            p.requires_grad = True

        groups1 = [
            {"params": _border_params(model.encoder, cfg), "lr": cfg.lr_border},
            {"params": _inner_params(model.encoder, cfg) + list(model.encoder.linear3.parameters()), "lr": cfg.lr_inner},
            {"params": _border_params(model.decoder1, cfg), "lr": cfg.lr_border},
            {"params": _inner_params(model.decoder1, cfg) + list(model.decoder1.linear1.parameters()), "lr": cfg.lr_inner},
        ]
        groups2 = [
            {"params": _border_params(model.encoder, cfg), "lr": cfg.lr_border},
            {"params": _inner_params(model.encoder, cfg) + list(model.encoder.linear3.parameters()), "lr": cfg.lr_inner},
            {"params": _border_params(model.decoder2, cfg), "lr": cfg.lr_border},
            {"params": _inner_params(model.decoder2, cfg) + list(model.decoder2.linear1.parameters()), "lr": cfg.lr_inner},
        ]
        return groups1, groups2


# =====================================================================
# Trainer
# =====================================================================

@dataclass
class EpochResult:
    epoch: int
    val_loss1: float
    val_loss2: float


class TransferLearningTrainer:
    """Orquesta el fine-tuning adversarial de USAD.

    Inyecta por constructor: la ``FineTuneStrategy`` y una función que crea
    optimizadores (por defecto Adam) — Dependency Inversion.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        strategy: FineTuneStrategy,
        device: torch.device,
        optimizer_ctor=torch.optim.Adam,
    ) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._device = device
        self._opt_ctor = optimizer_ctor

    def fit(
        self,
        model: SingleChannelUSAD,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_best_to: Path | None = None,
    ) -> list[EpochResult]:
        model.to(self._device)
        groups1, groups2 = self._strategy.build_param_groups(model, self._cfg)
        opt1 = self._opt_ctor(groups1)
        opt2 = self._opt_ctor(groups2)

        history: list[EpochResult] = []
        best_val = float("inf")
        best_state: dict | None = None
        patience_left = self._cfg.early_stopping_patience

        for epoch in range(1, self._cfg.epochs + 1):
            model.train()
            for [batch] in train_loader:
                batch = batch.to(self._device, non_blocking=True)

                loss1, _ = model.training_step(batch, epoch)
                opt1.zero_grad(set_to_none=True)
                loss1.backward()
                opt1.step()

                _, loss2 = model.training_step(batch, epoch)
                opt2.zero_grad(set_to_none=True)
                loss2.backward()
                opt2.step()

            # Validación
            model.eval()
            outputs = []
            for [batch] in val_loader:
                batch = batch.to(self._device, non_blocking=True)
                outputs.append(model.validation_step(batch, epoch))
            result = model.validation_epoch_end(outputs)
            model.epoch_end(epoch, result)

            val_combined = result["val_loss1"] + abs(result["val_loss2"])
            history.append(EpochResult(epoch, result["val_loss1"], result["val_loss2"]))

            if val_combined < best_val - 1e-6:
                best_val = val_combined
                best_state = copy.deepcopy(model.state_dict())
                patience_left = self._cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if save_best_to is not None:
            save_best_to.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_best_to)

        return history
