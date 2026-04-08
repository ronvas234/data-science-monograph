"""
Responsabilidad única (S): cargar y particionar datos CSV de SIATA.
No realiza transformaciones, normalización ni relleno de valores faltantes.
"""
import pandas as pd


class SIATADataLoader:
    """Carga datos de una estación SIATA y expone los splits E/V/T."""

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: ruta al CSV con columnas fecha_hora, t, flag, Split, t_mask
        """
        self.df = pd.read_csv(csv_path, parse_dates=["fecha_hora"])

    def get_split(self, split: str) -> pd.DataFrame:
        """
        Retorna el subconjunto de datos correspondiente al split indicado.
        No aplica fillna ni dropna — los NA se preservan tal cual.

        Args:
            split: 'E' (entrenamiento), 'V' (validación) o 'T' (prueba)

        Returns:
            DataFrame filtrado con todas las columnas originales
        """
        valid_splits = {"E", "V", "T"}
        if split not in valid_splits:
            raise ValueError(f"split debe ser uno de {valid_splits}, recibido: {split!r}")
        return self.df[self.df["Split"] == split].copy()

    def get_train_normal_mask(self) -> pd.Series:
        """
        Retorna máscara booleana sobre split E donde t_mask != -2000.
        Se usa para calcular estadísticas de normalización sobre datos limpios.
        """
        train_df = self.get_split("E")
        return train_df[train_df["t_mask"] != -2000.0]["t_mask"]
