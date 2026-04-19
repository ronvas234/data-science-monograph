from __future__ import annotations

import torch
import torch.nn as nn

from .submatrix_extractor import SubmatrixWeights


class _SingleChannelEncoder(nn.Module):
    """Encoder con shapes fijas al pre-entrenado (306, 153, 120) para permitir
    carga directa de las capas internas. Solo cambia la entrada (window_size → 306).
    """

    def __init__(self, window_size: int, latent_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(window_size, 306)
        self.linear2 = nn.Linear(306, 153)
        self.linear3 = nn.Linear(153, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        return self.relu(self.linear3(out))


class _SingleChannelDecoder(nn.Module):
    """Decoder simétrico con shapes fijas al pre-entrenado (153, 306, window_size)."""

    def __init__(self, latent_size: int, window_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_size, 153)
        self.linear2 = nn.Linear(153, 306)
        self.linear3 = nn.Linear(306, window_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        return self.sigmoid(self.linear3(out))


class SingleChannelUSAD(nn.Module):
    """USAD para un solo canal (temperatura). Mantiene el mismo API que el
    ``UsadModel`` original: ``training_step``, ``validation_step``, ``epoch_end``.
    """

    def __init__(self, window_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder = _SingleChannelEncoder(window_size, latent_size)
        self.decoder1 = _SingleChannelDecoder(latent_size, window_size)
        self.decoder2 = _SingleChannelDecoder(latent_size, window_size)

    @classmethod
    def from_submatrix(
        cls,
        weights: SubmatrixWeights,
        window_size: int,
        latent_size: int,
    ) -> "SingleChannelUSAD":
        model = cls(window_size=window_size, latent_size=latent_size)
        model.encoder.load_state_dict(weights.encoder_state, strict=True)
        model.decoder1.load_state_dict(weights.decoder1_state, strict=True)
        model.decoder2.load_state_dict(weights.decoder2_state, strict=True)
        return model

    # === Loop de entrenamiento USAD original ===
    def training_step(self, batch: torch.Tensor, n: int):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = (1.0 / n) * torch.mean((batch - w1) ** 2) + (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
        loss2 = (1.0 / n) * torch.mean((batch - w2) ** 2) - (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch: torch.Tensor, n: int):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            loss1 = (1.0 / n) * torch.mean((batch - w1) ** 2) + (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
            loss2 = (1.0 / n) * torch.mean((batch - w2) ** 2) - (1.0 - 1.0 / n) * torch.mean((batch - w3) ** 2)
        return {"val_loss1": loss1, "val_loss2": loss2}

    @staticmethod
    def validation_epoch_end(outputs):
        loss1 = torch.stack([x["val_loss1"] for x in outputs]).mean().item()
        loss2 = torch.stack([x["val_loss2"] for x in outputs]).mean().item()
        return {"val_loss1": loss1, "val_loss2": loss2}

    @staticmethod
    def epoch_end(epoch: int, result: dict) -> None:
        print(
            f"Epoch [{epoch}] val_loss1: {result['val_loss1']:.6f} "
            f"val_loss2: {result['val_loss2']:.6f}"
        )

    @torch.no_grad()
    def anomaly_score(
        self, batch: torch.Tensor, alpha: float = 0.5, beta: float = 0.5
    ) -> torch.Tensor:
        w1 = self.decoder1(self.encoder(batch))
        w2 = self.decoder2(self.encoder(w1))
        return alpha * torch.mean((batch - w1) ** 2, dim=1) + beta * torch.mean((batch - w2) ** 2, dim=1)
