"""
Responsabilidad única (S): construir ventanas deslizantes sobre series temporales.
No normaliza ni modifica los valores — solo empaqueta subsecuencias.
Las ventanas que contienen NaN reales (no centinelas -2000) se descartan.
"""
import numpy as np
from typing import Tuple, Optional


class SlidingWindowBuilder:
    """Genera ventanas deslizantes de tamaño fijo sobre un array 1D."""

    def __init__(self, window_size: int = 60):
        """
        Args:
            window_size: número de pasos de tiempo por ventana (default: 60 min)
        """
        if window_size < 1:
            raise ValueError(f"window_size debe ser >= 1, recibido: {window_size}")
        self.window_size = window_size

    def build(
        self,
        data: np.ndarray,
        mask_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Construye ventanas deslizantes descartando las que contienen NaN reales.

        Args:
            data: array 1D de valores normalizados (t o t normalizada)
            mask_data: array 1D de t_mask (sin normalizar, para detectar -2000).
                       Si es None, no se retorna máscara.

        Returns:
            Tupla (windows_data, windows_mask) donde:
            - windows_data: shape (N, window_size) — datos normalizados
            - windows_mask: shape (N, window_size) — t_mask crudas (o None)
        """
        n = len(data)
        if n < self.window_size:
            raise ValueError(
                f"Serie de longitud {n} < window_size={self.window_size}"
            )

        num_windows = n - self.window_size + 1
        indices = np.arange(num_windows)

        # Construir todas las ventanas de datos
        windows_data = np.array(
            [data[i : i + self.window_size] for i in indices], dtype=np.float32
        )

        # Descartar ventanas con NaN reales (NA verdaderos, no centinelas -2000)
        sentinel = -2000.0
        has_real_nan = np.array(
            [
                np.any(np.isnan(windows_data[i]) & (windows_data[i] != sentinel))
                for i in range(len(windows_data))
            ]
        )
        # NaN en numpy no == -2000 nunca, así que la condición correcta es simplemente:
        has_real_nan = np.array(
            [np.any(np.isnan(windows_data[i])) for i in range(len(windows_data))]
        )
        valid_mask = ~has_real_nan
        windows_data = windows_data[valid_mask]

        if mask_data is not None:
            windows_mask = np.array(
                [mask_data[i : i + self.window_size] for i in indices], dtype=np.float32
            )
            windows_mask = windows_mask[valid_mask]
            return windows_data, windows_mask

        return windows_data, None
