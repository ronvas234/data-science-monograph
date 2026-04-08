"""
Responsabilidad única (S): construir máscaras de pérdida para el Plan B.
El centinela -2000 indica anomalías durante entrenamiento que deben ignorarse.
"""
import torch


class PlanBMaskBuilder:
    """
    Genera máscaras binarias para el training loss del Stage 2.

    El Plan B usa t_mask = -2000 para marcar anomalías en E y V.
    Durante el entrenamiento, las ventanas con centinela deben tener
    contribución 0 en la función de pérdida.
    """

    SENTINEL: float = -2000.0

    def build_loss_mask(self, window_mask: torch.Tensor) -> torch.Tensor:
        """
        Construye máscara de pérdida a nivel de ventana.

        Una ventana entera se descarta (peso 0) si CUALQUIER timestep
        en ella contiene el centinela -2000.

        Args:
            window_mask: tensor de shape (batch, window_size) con valores de t_mask

        Returns:
            tensor de shape (batch, 1) con valores 1.0 (ventana limpia) o 0.0 (anomalía)
        """
        has_sentinel = (window_mask == self.SENTINEL).any(dim=1, keepdim=True)  # (batch, 1)
        return (~has_sentinel).float()

    def build_timestep_mask(self, window_mask: torch.Tensor) -> torch.Tensor:
        """
        Construye máscara a nivel de timestep individual.

        Cada timestep se pondera con 0 si contiene centinela, 1 si es normal.

        Args:
            window_mask: tensor de shape (batch, window_size)

        Returns:
            tensor de shape (batch, window_size) con 1.0 o 0.0
        """
        return (window_mask != self.SENTINEL).float()
