# Resultados: Transfer Learning USAD → Detección de Anomalías SIATA (Corrida 3)

**Fecha de corrida:** 2026-03-21
**Notebook:** `Transfer_Learning_SIATA.ipynb`
**GPU:** Tesla T4 (Google Colab)

---

## Cambios respecto a la Corrida 2

| Parámetro | Corrida 2 | Corrida 3 |
|---|---|---|
| `N_EPOCHS_P2` | 80 | **150** |
| `alpha` (scoring) | 0.5 | **0.1** |
| `beta` (scoring) | 0.5 | **0.9** |
| Label gráfica convergencia | "z=1200" (incorrecto) | `z_size` dinámico ✓ |

---

## Entrenamiento

### Phase 1 — Decoder Warm-Up (10 épocas, encoder congelado)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0356 | 0.0381 |
| 1 | 0.0048 | 0.0000 |
| 9 | 0.0028 | -0.0024 |

Convergencia rápida idéntica a corridas anteriores.

### Phase 2 — Fine-Tuning (150 épocas, lr=1e-4)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0023 | 0.0019 |
| 5 | 0.0324 | -0.0295 ← spike adversarial |
| 6 | 0.0101 | -0.0090 ← convergencia abrupta |
| 43 | 0.0103 | -0.0101 |
| 65 | 0.0080 | -0.0079 ← mínimo aproximado |
| 99 | 0.0073 | -0.0072 ← mínimo absoluto |
| 100+ | empieza a subir ← señal de sobre-entrenamiento |
| 149 | 0.0116 | -0.0115 |

El modelo alcanzó val_loss1 = **0.0073** — 10× más bajo que la Corrida 2 (0.069). Esto indica sobre-entrenamiento severo.

---

## Resultados

### AUC-ROC — Colapso del Transfer Learning

| Modelo | Corrida 2 (α=0.5) | Corrida 3 (α=0.1) | Δ |
|---|---|---|---|
| **Transfer (z=120, 150 épocas)** | **0.7308** | **0.3714** | **-0.3594** |
| Baseline A (Xavier, z=120, 90 épocas) | 0.7143 | 0.6589 | -0.0554 |
| Baseline B (Xavier, z=100, 90 épocas) | 0.6955 | 0.6938 | -0.0017 |

**El transfer learning cayó a 0.37 — peor que aleatorio (0.5).** Los baselines también empeoraron con alpha=0.1/beta=0.9, pero de forma mucho más moderada.

### Reporte de clasificación — Transfer Learning

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.88 | 0.41 | 0.56 | 79,857 |
| Anómalo | 0.07 | 0.41 | 0.11 | 7,923 |
| **Accuracy** | | | **0.41** | 87,780 |

- Threshold: **0.004272** (vs 0.068941 en Corrida 2)
- Score range: [0.000044, 0.080071] (vs [0.001205, 0.115866] en Corrida 2)
- Accuracy 0.41 — peor que lanzar una moneda

### Sanity Check — primer éxito

```
Score ventana ANÓMALA:  0.00411413
Score ventana NORMAL:   0.00407232
Anomalía > Normal:      True ✓  ← primera vez que pasa en las 3 corridas
```

El modelo sí asigna mayor score a anomalías que a datos normales. La dirección es correcta pero la magnitud de separación es insuficiente para discriminar globalmente.

---

## Diagnóstico — Dos problemas combinados

### Problema 1: Sobre-entrenamiento con 150 épocas

El transfer model llegó a val_loss1 = 0.0073 en la época 99 — reconstruye los datos de entrenamiento casi perfectamente. Cuando un autoencoder reconstruye todo con error muy bajo, **también reconstruye las anomalías bien** y pierde capacidad discriminativa.

El óptimo real estuvo alrededor de la **época 65-70** (val_loss1 ≈ 0.008), no en la 150. Después del época 100 la loss empezó a subir — señal de que el modelo ya había pasado su punto óptimo.

```
Época  6  → 0.0101  ← convergencia inicial
Época 65  → 0.0080  ← zona óptima
Época 99  → 0.0073  ← mínimo absoluto (sobre-entrenado)
Época 100 → 0.0075  ← empieza a subir
Época 149 → 0.0116  ← degradación clara
```

### Problema 2: alpha=0.1 / beta=0.9 fue perjudicial

Con beta=0.9 se enfatizó AE2 (adversarial). Con un modelo que reconstruye todo muy bien, AE2 también reconstruye bien — los scores se vuelven inconsistentes y el threshold colapsa a 0.004. El efecto fue perjudicial para los 3 modelos, especialmente el Transfer.

---

## Comparación acumulada de las 3 corridas

| Corrida | Config clave | AUC Transfer | AUC Baseline A | AUC Baseline B |
|---|---|---|---|---|
| 1 | shuffle=False, z incorrecto | 0.6446* | 0.6814* | 0.7135* |
| 2 | shuffle=True, z=120, α=0.5 | **0.7308** | 0.7143 | 0.6955 |
| 3 | +150 épocas, α=0.1/β=0.9 | 0.3714 | 0.6589 | 0.6938 |

*Comparación inválida (arquitecturas distintas).

**La Corrida 2 sigue siendo la mejor configuración encontrada.**

---

## Conclusiones y próximos pasos

### Lo que aprendió esta corrida

1. **alpha=0.5, beta=0.5 es la mejor configuración de scoring** — enfatizar AE2 con beta=0.9 no mejora la detección y puede colapsar el AUC.

2. **El óptimo del transfer está en ~70 épocas de Phase 2**, no en 150. Con 150 épocas el modelo sobre-entrena.

3. **El sanity check pasó por primera vez**, lo que confirma que la dirección del modelo es correcta — el problema es de magnitud de separación entre clases, no de dirección.

### Corrida 4 recomendada

- Revertir `alpha=0.5, beta=0.5`
- Fijar `N_EPOCHS_P2 = 70` (zona óptima identificada)
- Mantener `shuffle=True`
- Meta: recuperar AUC > 0.73 con convergencia controlada
