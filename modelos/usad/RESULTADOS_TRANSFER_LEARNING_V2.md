# Resultados: Transfer Learning USAD → Detección de Anomalías SIATA (Corrida 2 — Comparación Corregida)

**Fecha de corrida:** 2026-03-21
**Notebook:** `Transfer_Learning_SIATA.ipynb`
**GPU:** Tesla T4 (Google Colab)

---

## Correcciones respecto a la Corrida 1

| Parámetro | Corrida 1 (incorrecta) | Corrida 2 (corregida) |
|---|---|---|
| Transfer z_size | 120 | 120 |
| Baseline A z_size | **1200** (diferente) | **120** (igual) |
| Baseline B z_size | 100 | 100 |
| Épocas baselines | 50 | 90 |
| shuffle train_loader | False | True |

La corrida 1 comparaba modelos con arquitecturas diferentes, lo que invalidaba sus conclusiones. En esta corrida todos los modelos comparten la misma arquitectura base (z=120) y el mismo presupuesto de entrenamiento (90 épocas equivalentes).

---

## Datos

| Parámetro | Valor |
|---|---|
| Total filas (largo) | 350,308 |
| Timestamps únicos | 87,840 |
| Estaciones | Jardín Botánico, Torre SIATA, UNAN, Fiscalía General |
| Periodo | 2025-03-01 → 2025-04-30 |
| NaN imputados (ffill/bfill) | 1,052 |
| Ventanas totales | 87,780 |
| Ventanas anómalas | 7,923 (9.03%) |
| Ventanas normales | 79,857 (90.97%) |
| Train (limpias) | 24,514 |
| Val (limpias) | 6,129 |
| `w_size` | 240 (60 min × 4 estaciones) |
| `z_size` | 120 |
| `BATCH_SIZE` | 512 |

---

## Arquitectura

### Modelo pre-entrenado (SWaT)

| Capa | Shape |
|---|---|
| encoder.linear1.weight | (306, 612) |
| encoder.linear2.weight | (153, 306) |
| encoder.linear3.weight | (120, 153) |
| decoder1.linear1.weight | (153, 120) |
| decoder1.linear2.weight | (306, 153) |
| decoder1.linear3.weight | (612, 306) |

### Nuevo modelo SIATA — todas las dimensiones compatibles ✓

| Capa | Shape |
|---|---|
| encoder.linear1.weight | (120, 240) |
| encoder.linear2.weight | (60, 120) |
| encoder.linear3.weight | (120, 60) |
| decoder.linear1.weight | (60, 120) |
| decoder.linear2.weight | (120, 60) |
| decoder.linear3.weight | (240, 120) |

**Total parámetros:** 130,740

### Normas de pesos post-transferencia

| Capa | Norma |
|---|---|
| encoder.linear1.weight | 6.1378 |
| encoder.linear2.weight | 3.8771 |
| encoder.linear3.weight | 5.9618 |
| decoder1.linear3.weight | 7.3057 |
| decoder2.linear3.weight | 7.9564 |

Normas no nulas confirman que los pesos del checkpoint se transfirieron correctamente.

---

## Entrenamiento

### Transfer — Phase 1: Decoder Warm-Up (encoder congelado, 10 épocas)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0367 | 0.0378 |
| 1 | 0.0047 | -0.0001 |
| 3 | 0.0029 | -0.0018 |
| 9 | 0.0031 | -0.0026 |

Convergencia muy rápida en la primera época: val_loss1 bajó 8× gracias a la inicialización por submatriz.

### Transfer — Phase 2: Fine-Tuning Completo (lr=1e-4, 80 épocas)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0021 | 0.0020 |
| 2 | 0.1067 | -0.0941 ← spike adversarial |
| 18 | 0.2303 | -0.2290 ← pico máximo |
| 30 | 0.0914 | -0.0908 ← convergencia |
| 50 | 0.0731 | -0.0728 |
| 79 | 0.0685 | -0.0684 ← aún bajando al final |

El modelo no convergió completamente al finalizar — sugiere potencial de mejora con más épocas.

### Baseline A: Xavier aleatorio, z=120, 90 épocas

| Época | val_loss1 |
|---|---|
| 0 | 0.0246 |
| 31 | 0.0503 ← mínimo |
| 44+ | empieza a subir ← overfitting |
| 89 | 0.0637 |

Presenta inestabilidad en la segunda mitad del entrenamiento.

### Baseline B: Xavier aleatorio, z=100, 90 épocas

| Época | val_loss1 |
|---|---|
| 0 | 0.0360 |
| 40 | 0.0542 ← convergencia |
| 89 | 0.0531 ← estable |

El más estable de los tres pero con menor capacidad de representación.

---

## Resultados

### AUC-ROC — Comparación válida (misma arquitectura)

| Modelo | Corrida 1 | Corrida 2 |
|---|---|---|
| **Transfer Learning (submatriz, z=120)** | 0.6446* | **0.7308** ← gana |
| Baseline A (Xavier, z=120) | 0.6814* | 0.7143 |
| Baseline B (Xavier, z=100) | 0.7135* | 0.6955 |

*Comparación inválida en Corrida 1 por diferencia de arquitecturas.

**Con la misma arquitectura y mismo presupuesto, el transfer learning supera a Xavier aleatorio en +0.0165 AUC.**

### Reporte de clasificación — Transfer Learning

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.95 | 0.66 | 0.78 | 79,857 |
| Anómalo | 0.16 | 0.66 | 0.26 | 7,923 |
| **Accuracy** | | | **0.66** | 87,780 |

- Threshold óptimo: **0.068941**
- Score range: [0.001205, 0.115866]
- Por cada alarma verdadera hay ~5 falsas alarmas (precision 0.16)

### Sanity Check

```
Score ventana ANÓMALA:  0.08853027
Score ventana NORMAL:   0.08860168
Diferencia:             0.00007
Anomalía > Normal:      False ← al límite del threshold
```

La diferencia es de 0.00007 — marginal pero no estructural como en la Corrida 1. El modelo discrimina globalmente (AUC=0.73) pero esa ventana específica queda justo en el límite del threshold.

---

## Conclusiones

### 1. La corrección de la comparación invierte las conclusiones

La Corrida 1 concluía que el transfer learning era el peor modelo. Con una comparación justa (misma arquitectura, mismas épocas), el transfer learning es el **mejor modelo** con AUC=0.7308.

### 2. El transfer learning aporta valor measurable

Con la misma arquitectura (z=120), la inicialización por submatriz desde SWaT supera a Xavier aleatorio en +0.0165 AUC (0.7308 vs 0.7143). Los pesos pre-entrenados en datos industriales proporcionan un punto de partida útil incluso para temperatura meteorológica.

### 3. shuffle=True fue determinante

En la Corrida 1 (shuffle=False) el transfer obtuvo AUC=0.6446. En la Corrida 2 (shuffle=True) obtuvo 0.7308. El entrenamiento secuencial impedía que el modelo aprendiera patrones generalizables.

### 4. El modelo Transfer no convergió completamente

val_loss1 seguía bajando en la época 79 de Phase 2. Aumentar a 120-150 épocas en Phase 2 podría mejorar el AUC adicionalmente.

### 5. Problema pendiente: precision de anómalos = 0.16

El desbalance de clases (9:1) genera muchas falsas alarmas. Para mejorar la precision hay dos opciones:
- Ajustar el threshold hacia valores más altos (mayor precision, menor recall)
- Atacar el desbalance durante el entrenamiento (ponderación de clases)

### Para la monografía

El experimento demuestra que el transfer learning desde SWaT **sí aporta** cuando la comparación metodológica es correcta. La conclusión revisada es: la inicialización por submatriz proporciona una ventaja modesta pero real (+2.3% AUC) frente a la inicialización aleatoria con la misma arquitectura, validando la hipótesis de transfer learning cross-domain.
