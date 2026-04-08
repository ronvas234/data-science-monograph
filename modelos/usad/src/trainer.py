"""
Principios SOLID aplicados:
- S: cada trainer tiene una responsabilidad (entrenar stage 1 o stage 2)
- O: BaseTrainer está cerrado a modificaciones; Stage1/Stage2 lo extienden
- L: Stage2Trainer es sustituible por Stage1Trainer en cualquier contexto BaseTrainer
- D: el modelo y los optimizadores se inyectan, no se instancian internamente
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from masking import PlanBMaskBuilder


class BaseTrainer(ABC):
    """Entrenador abstracto para USAD. Define la interfaz de entrenamiento."""

    def __init__(
        self,
        model: nn.Module,
        optimizer1: torch.optim.Optimizer,
        optimizer2: torch.optim.Optimizer,
        device: torch.device,
    ):
        """
        Args:
            model: instancia de UsadModel
            optimizer1: optimiza encoder + decoder1
            optimizer2: optimiza encoder + decoder2
            device: dispositivo de cómputo (CPU/GPU)
        """
        self.model = model
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.device = device

    @abstractmethod
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Ejecuta una época de entrenamiento."""
        ...

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Calcula pérdidas de validación (mismo para Stage 1 y Stage 2)."""
        self.model.eval()
        losses1, losses2 = [], []
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(self.device)
                n = epoch + 1
                z = self.model.encoder(data)
                w1 = self.model.decoder1(z)
                w2 = self.model.decoder2(z)
                w3 = self.model.decoder2(self.model.encoder(w1))
                l1 = (1 / n) * torch.mean((data - w1) ** 2) + (1 - 1 / n) * torch.mean((data - w3) ** 2)
                l2 = (1 / n) * torch.mean((data - w2) ** 2) - (1 - 1 / n) * torch.mean((data - w3) ** 2)
                losses1.append(l1.item())
                losses2.append(l2.item())
        return {
            "val_loss1": sum(losses1) / len(losses1),
            "val_loss2": sum(losses2) / len(losses2),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Ciclo de entrenamiento completo.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: número de épocas
            verbose: imprimir pérdidas por época

        Returns:
            history: lista de dicts con val_loss1 y val_loss2 por época
        """
        history = []
        for epoch in range(epochs):
            self._train_epoch(train_loader, epoch)
            result = self._validate_epoch(val_loader, epoch)
            history.append(result)
            if verbose:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"val_loss1: {result['val_loss1']:.4f}  "
                    f"val_loss2: {result['val_loss2']:.4f}"
                )
        return history


class Stage1Trainer(BaseTrainer):
    """
    Stage 1: entrenamiento estándar USAD sobre datos limpios de entrenamiento.
    El DataLoader retorna tensores (data,) — solo una columna, sin máscara.
    """

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        self.model.train()
        n = epoch + 1
        for batch in train_loader:
            data = batch[0].to(self.device)

            # Entrenar AE1 (encoder + decoder1)
            loss1, _ = self._compute_losses(data, n)
            loss1.backward()
            self.optimizer1.step()
            self.optimizer1.zero_grad()

            # Entrenar AE2 (encoder + decoder2)
            _, loss2 = self._compute_losses(data, n)
            loss2.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()

    def _compute_losses(
        self, data: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.model.encoder(data)
        w1 = self.model.decoder1(z)
        w2 = self.model.decoder2(z)
        w3 = self.model.decoder2(self.model.encoder(w1))
        loss1 = (1 / n) * torch.mean((data - w1) ** 2) + (1 - 1 / n) * torch.mean((data - w3) ** 2)
        loss2 = (1 / n) * torch.mean((data - w2) ** 2) - (1 - 1 / n) * torch.mean((data - w3) ** 2)
        return loss1, loss2


class Stage2Trainer(BaseTrainer):
    """
    Stage 2: fine-tuning con masking del Plan B.
    El DataLoader retorna tensores (data, mask) — dos columnas.
    Las ventanas con centinela -2000 en mask reciben pérdida 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask_builder = PlanBMaskBuilder()

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        self.model.train()
        n = epoch + 1
        for batch in train_loader:
            data = batch[0].to(self.device)
            mask = batch[1].to(self.device)

            # Construir máscara de pérdida: (batch, 1) — 1.0 si ventana limpia
            loss_weight = self._mask_builder.build_loss_mask(mask)  # (batch, 1)

            # Entrenar AE1
            loss1, _ = self._compute_masked_losses(data, mask, loss_weight, n)
            loss1.backward()
            self.optimizer1.step()
            self.optimizer1.zero_grad()

            # Entrenar AE2
            _, loss2 = self._compute_masked_losses(data, mask, loss_weight, n)
            loss2.backward()
            self.optimizer2.step()
            self.optimizer2.zero_grad()

    def _compute_masked_losses(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        loss_weight: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.model.encoder(data)
        w1 = self.model.decoder1(z)
        w2 = self.model.decoder2(z)
        w3 = self.model.decoder2(self.model.encoder(w1))

        # Pérdida por muestra: (batch,)
        mse_w1 = torch.mean((data - w1) ** 2, dim=1, keepdim=True)
        mse_w2 = torch.mean((data - w2) ** 2, dim=1, keepdim=True)
        mse_w3 = torch.mean((data - w3) ** 2, dim=1, keepdim=True)

        # Aplicar máscara de pérdida
        total_weight = loss_weight.sum().clamp(min=1.0)
        loss1 = ((1 / n) * mse_w1 + (1 - 1 / n) * mse_w3) * loss_weight
        loss2 = ((1 / n) * mse_w2 - (1 - 1 / n) * mse_w3) * loss_weight

        return loss1.sum() / total_weight, loss2.sum() / total_weight
