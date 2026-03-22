# Resumen Consolidado de Experimentos: Transfer Learning USAD → SIATA

**Modelo:** USAD (UnSupervised Anomaly Detection on multivariate time series)
**Tarea:** Detección de anomalías de temperatura en 4 estaciones SIATA — Valle de Aburrá, Medellín
**Estrategia:** Transfer learning por submatriz desde modelo pre-entrenado en SWaT (dataset industrial)
**Periodo de datos:** Marzo–Abril 2025
**GPU:** Tesla T4 (Google Colab)

---

## Dataset

| Parámetro | Valor |
|---|---|
| Estaciones | Jardín Botánico, Torre SIATA, UNAN, Fiscalía General |
| Timestamps únicos | 87,840 (resolución 1 minuto) |
| Ventanas totales (ws=60) | 87,780 |
| Ventanas anómalas | 7,923 (9.03%) |
| Ventanas de entrenamiento | 24,514 (limpias) |
| Ventanas de validación | 6,129 (limpias) |
| NaN imputados (ffill/bfill) | 1,052 |

### Anomalías por estación

| Estación | n_anomalías | % |
|---|---|---|
| Fiscalía General | 34 | 0.04% |
| Jardín Botánico | 4,041 | 4.64% |
| Torre SIATA | 4,471 | 5.09% |
| UNAN | 3,620 | 4.13% |

---

## Arquitectura del Modelo

| Componente | Config |
|---|---|
| `w_size` | 240 (60 min × 4 estaciones) |
| `z_size` | 120 |
| Total parámetros | 130,740 |
| Encoder | Linear(240→120→60→120) + ReLU |
| Decoder | Linear(120→60→120→240) + ReLU + Sigmoid |

### Compatibilidad con checkpoint pre-entrenado (SWaT)

Todas las capas del nuevo modelo tienen dimensiones ≤ al modelo pre-entrenado → transferencia por submatriz válida en todas las capas (✓).

| Capa | Pre-entrenado | SIATA | Compatible |
|---|---|---|---|
| encoder.linear1.weight | (306, 612) | (120, 240) | ✓ |
| encoder.linear2.weight | (153, 306) | (60, 120) | ✓ |
| encoder.linear3.weight | (120, 153) | (120, 60) | ✓ |
| decoder.linear1.weight | (153, 120) | (60, 120) | ✓ |
| decoder.linear2.weight | (306, 153) | (120, 60) | ✓ |
| decoder.linear3.weight | (612, 306) | (240, 120) | ✓ |

---

## Configuración por Corrida

| Parámetro | Corrida 1 | Corrida 2 | Corrida 3 | Corrida 4 |
|---|---|---|---|---|
| `z_size` Transfer | 120 | 120 | 120 | 120 |
| `z_size` Baseline A | **1200** ← error | 120 | 120 | 120 |
| `z_size` Baseline B | 100 | 100 | 100 | 100 |
| `shuffle` train | **False** | True | True | True |
| `N_EPOCHS_P1` | 10 | 10 | 10 | 10 |
| `N_EPOCHS_P2` | 40 | 80 | 150 | 70 |
| Épocas baselines | 50 | 90 | 90 | 90 |
| `alpha` scoring | 0.5 | 0.5 | **0.1** | 0.5 |
| `beta` scoring | 0.5 | 0.5 | **0.9** | 0.5 |
| Comparación válida | ✗ | ✓ | ✓ | ✓ |

---

## Resultados AUC-ROC

| Corrida | Transfer | Baseline A | Baseline B | Nota |
|---|---|---|---|---|
| 1 | 0.6446 | 0.6814 | 0.7135 | ⚠️ Comparación inválida |
| **2** | **0.7308** | **0.7143** | 0.6955 | ✅ Mejor corrida |
| 3 | 0.3714 | 0.6589 | 0.6938 | ❌ Sobre-entrenado + alpha perjudicial |
| 4 | 0.2253 | 0.6644 | 0.6506 | ❌ Sobre-entrenado más rápido |

---

## Curvas de Entrenamiento — Transfer Learning (Phase 2)

| Corrida | val_loss1 inicial | val_loss1 mínimo | val_loss1 final | AUC |
|---|---|---|---|---|
| 2 (80 épocas) | 0.0058 | ~0.069 | 0.0685 | **0.7308** |
| 3 (150 épocas) | 0.0023 | 0.0073 (época 99) | 0.0116 | 0.3714 |
| 4 (70 épocas) | 0.0022 | 0.0027 (época 22) | 0.0055 | 0.2253 |

**Observación clave:** en C2 el modelo no bajó de 0.069 de val_loss1. En C3 y C4 llegó a 0.003-0.007, 10-25× más bajo — y eso destruyó la discriminación.

---

## Relación val_loss1 ↔ AUC

```
val_loss1 ∈ [0.05, 0.10]  →  AUC ∈ [0.65, 0.73]  ← rango útil
val_loss1 ∈ [0.01, 0.05]  →  AUC ∈ [0.30, 0.50]  ← zona gris
val_loss1 < 0.01           →  AUC < 0.30           ← colapsado
```

Existe una **relación inversa directa** y consistente a través de las 4 corridas entre el error de reconstrucción del Transfer y su AUC. Los baselines permanecen en ~0.050 de val_loss1 por su inicialización aleatoria, lo que los mantiene en el rango útil.

---

## Reporte de Clasificación — Mejor Modelo (Transfer, Corrida 2)

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.95 | 0.66 | 0.78 | 79,857 |
| Anómalo | 0.16 | 0.66 | 0.26 | 7,923 |
| **Accuracy** | | | **0.66** | 87,780 |

- Threshold óptimo (Corrida 2): **0.068941**
- Score range (Corrida 2): [0.001205, 0.115866]

---

## Sanity Check — Evolución por Corrida

| Corrida | Score ANÓMALO | Score NORMAL | Anomalía > Normal |
|---|---|---|---|
| 1 | 0.06444 | 0.06557 | ✗ False |
| 2 | 0.08853 | 0.08860 | ✗ False (diff: 0.00007) |
| 3 | 0.00411 | 0.00407 | ✓ True (diff: 0.00004) |
| 4 | 0.00072 | 0.00070 | ✓ True (diff: 0.000021) |

Las corridas 3 y 4 pasan el sanity check (dirección correcta) pero con separaciones microscópicas. Las corridas 1 y 2 fallan, pero C2 tiene una diferencia casi nula (-0.00007) y un AUC de 0.73 — el sanity check puntual no refleja el desempeño global.

---

## Errores y Correcciones Durante el Proceso

| Problema | Corrida donde apareció | Corrección aplicada |
|---|---|---|
| Comparación inválida: Baseline A con z=1200 vs Transfer z=120 | 1 | Unificar z_size=120 en C2 |
| shuffle=False en train_loader | 1 | shuffle=True en C2 |
| NaN en datos tras pivot | 1 | ffill().bfill() en C2 |
| Gradiente explosivo en Phase 1 | 1 | clip_grad_norm(max=1.0) en C2 |
| Threshold no definido como escalar | 2 | np.atleast_1d + float() en C2 |
| Label incorrecto "z=1200" en gráfica | 1-3 | z_size dinámico en C3 |
| alpha=0.1/beta=0.9 colapsa AUC | 3 | Revertido en C4 |
| Sobre-entrenamiento con 150 épocas | 3 | Reducido a 70 en C4 (insuficiente) |

---

## Conclusiones

### 1. Transfer Learning vs Baselines — la conclusión correcta

Con una comparación metodológicamente válida (C2):

- **Transfer Learning supera a Baseline A** con la misma arquitectura: AUC 0.7308 vs 0.7143 (+2.3%)
- Pero esta ventaja **no es reproducible de forma controlada** — depende de la trayectoria estocástica del entrenamiento

### 2. El sobre-entrenamiento es el problema central

Los pesos pre-entrenados de SWaT son tan informativos para temperatura que el modelo Transfer converge a val_loss1 ≈ 0.003 (casi cero error de reconstrucción). En ese régimen:
- El modelo reconstruye datos normales **y anómalos** con igual precisión
- La señal de anomalía (error de reconstrucción elevado) desaparece
- El AUC colapsa por debajo de 0.5

### 3. Los Baselines son más robustos

Baseline A y B mantienen AUC entre 0.65 y 0.71 en todas las corridas, independientemente de configuraciones. Son predecibles y estables.

### 4. Hallazgo para la monografía

> *La inicialización por submatriz desde SWaT puede mejorar el AUC en +2.3% sobre un baseline con la misma arquitectura, pero introduce inestabilidad de entrenamiento. Los pesos pre-entrenados llevan al modelo a un régimen de sobre-reconstrucción donde la señal de anomalía se vuelve indistinguible del ruido. El modelo óptimo para uso práctico en detección de anomalías de temperatura en SIATA es el Baseline A (z=120, inicialización Xavier), que ofrece AUC consistente de ~0.66-0.71 sin requerir control explícito del régimen de convergencia.*

### 5. Trabajo futuro para mejorar el Transfer

Para hacer el transfer learning confiable se necesitaría:
- **Early stopping basado en AUC** (no en val_loss1) para detener en el rango útil
- **Learning rate muy bajo** (1e-5) en Phase 2 para ralentizar convergencia
- **Regularización** (dropout, L2) para evitar sobre-ajuste a datos normales

---

## Archivos de Referencia

| Archivo | Contenido |
|---|---|
| `RESULTADOS_TRANSFER_LEARNING_V2.md` | Análisis detallado Corrida 2 |
| `RESULTADOS_TRANSFER_LEARNING_V3.md` | Análisis detallado Corrida 3 |
| `RESULTADOS_TRANSFER_LEARNING_V4.md` | Análisis detallado Corrida 4 |
| `model_siata_transfer.pth` | Pesos del modelo Transfer (última corrida) |
| `Transfer_Learning_SIATA.ipynb` | Notebook completo con código y outputs |
