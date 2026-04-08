"""
Responsabilidad única (S): exportar resultados del experimento como Markdown.
No realiza cálculos — solo formatea la salida.
"""
import os
from datetime import datetime
from typing import Dict

import numpy as np


class MarkdownReporter:
    """Genera un reporte .md con métricas, configuración y confusion matrix."""

    def export(
        self,
        metrics: Dict,
        config: Dict,
        output_path: str,
        stage: str = "Stage 2 (Transfer Learning)",
    ) -> str:
        """
        Exporta el reporte final como archivo Markdown.

        Args:
            metrics: dict retornado por AnomalyEvaluator.compute_metrics()
            config: dict con hiperparámetros del experimento
            output_path: ruta donde guardar el .md
            stage: descripción del stage de entrenamiento

        Returns:
            contenido del reporte como string
        """
        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# Reporte: Transfer Learning USAD — Estación 203 (SIATA)

**Fecha:** {timestamp}
**Estrategia:** Opción 3 — Two-Stage Masking Training (Plan B)
**Stage evaluado:** {stage}

---

## Configuración del Experimento

| Parámetro | Valor |
|---|---|
| window_size | {config.get('window_size', 'N/A')} |
| w_size | {config.get('w_size', 'N/A')} |
| z_size | {config.get('z_size', 'N/A')} |
| batch_size | {config.get('batch_size', 'N/A')} |
| epochs_stage1 | {config.get('epochs_stage1', 'N/A')} |
| epochs_stage2 | {config.get('epochs_stage2', 'N/A')} |
| alpha (scoring) | {config.get('alpha', 0.5)} |
| beta (scoring) | {config.get('beta', 0.5)} |
| normalización | z-score (media y std) |
| dispositivo | {config.get('device', 'N/A')} |

---

## Métricas en Test (Split T — Agosto/Septiembre 2023)

| Métrica | Valor |
|---|---|
| **Accuracy** | {metrics['accuracy']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1-Score** | {metrics['f1']:.4f} |
| **Threshold** | {metrics['threshold']:.6f} |
| Anomalías predichas | {metrics['n_anomalies_predicted']} |
| Anomalías reales | {metrics['n_anomalies_real']} |

---

## Confusion Matrix

```
                 Predicho Normal  Predicho Anomalía
Real Normal           {tn:>8}          {fp:>8}
Real Anomalía         {fn:>8}          {tp:>8}
```

- **TP** (Verdaderos Positivos): {tp}
- **TN** (Verdaderos Negativos): {tn}
- **FP** (Falsos Positivos): {fp}
- **FN** (Falsos Negativos): {fn}

---

## Descripción del Enfoque

### Plan B — Two-Stage Masking Training

1. **Stage 1 — Pre-entrenamiento:** USAD entrenado solo con datos de
   entrenamiento limpios (t_mask ≠ -2000). El modelo aprende la distribución
   normal de temperatura de la estación 203.

2. **Stage 2 — Transfer + Fine-tuning con Masking:** Se cargan los pesos del
   Stage 1 y se re-entrena con todos los datos E+V. Las ventanas que contienen
   el centinela -2000 reciben peso 0 en la función de pérdida, de modo que el
   modelo no aprende a reconstruir anomalías.

3. **Evaluación:** El modelo se prueba en el split T (sin máscara), donde la
   estación 203 presenta una falla masiva en agosto-septiembre. El threshold
   se selecciona maximizando F1 sobre el split de validación.

### Restricciones cumplidas

- [x] Normalización con media y desviación estándar (sin MinMaxScaler)
- [x] Sin fillna() ni dropna() en el pipeline
- [x] Métricas: accuracy, precision, recall, f1, confusion_matrix
- [x] Principios SOLID en la arquitectura del código
- [x] Ejecutable en Google Colab desde GitHub

---

*Generado automáticamente por MarkdownReporter — Transfer Learning USAD → SIATA*
"""

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content
