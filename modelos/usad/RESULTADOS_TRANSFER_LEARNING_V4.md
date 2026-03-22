# Resultados: Transfer Learning USAD → Detección de Anomalías SIATA (Corrida 4)

**Fecha de corrida:** 2026-03-21
**Notebook:** `Transfer_Learning_SIATA.ipynb`
**GPU:** Tesla T4 (Google Colab)

---

## Cambios respecto a la Corrida 3

| Parámetro | Corrida 3 | Corrida 4 |
|---|---|---|
| `N_EPOCHS_P2` | 150 | **70** |
| `alpha` (scoring) | 0.1 | **0.5** |
| `beta` (scoring) | 0.9 | **0.5** |

Objetivo: revertir los cambios perjudiciales de C3 y usar el rango óptimo (~época 65-70) identificado en esa corrida.

---

## Entrenamiento

### Phase 1 — Decoder Warm-Up (10 épocas, encoder congelado)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0365 | 0.0379 |
| 1 | 0.0046 | -0.0001 |
| 9 | 0.0027 | -0.0023 |

Convergencia rápida consistente con corridas anteriores.

### Phase 2 — Fine-Tuning (70 épocas, lr=1e-4)

| Época | val_loss1 | val_loss2 |
|---|---|---|
| 0 | 0.0022 | 0.0019 |
| 3 | 0.1231 | -0.1109 ← spike adversarial |
| 4 | 0.1266 | -0.1183 ← pico máximo |
| 11 | 0.0086 | -0.0080 ← bajada abrupta |
| 22 | 0.0028 | -0.0026 ← plateau total |
| 22-38 | ~0.0027 | ~-0.0026 ← completamente plano |
| 40+ | sube | ← inestabilidad tardía |
| 69 | 0.0055 | -0.0054 |

**El modelo alcanzó plateau en val_loss1 = 0.0027 en la época 22** — reconstrucción casi perfecta desde esa época. Las 48 épocas restantes fueron innecesarias.

---

## Resultados

### AUC-ROC — peor resultado en las 4 corridas

| Modelo | AUC |
|---|---|
| **Transfer Learning (z=120, 70 épocas)** | **0.2253** ← peor que aleatorio |
| Baseline A (Xavier, z=120, 90 épocas) | 0.6644 |
| Baseline B (Xavier, z=100, 90 épocas) | 0.6506 |

### Reporte de clasificación — Transfer Learning

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.81 | 0.30 | 0.44 | 79,857 |
| Anómalo | 0.04 | 0.30 | 0.07 | 7,923 |
| **Accuracy** | | | **0.30** | 87,780 |

- Threshold: **0.001056** (colapso total vs 0.069 en C2)
- Score range: [0.000374, 0.076556]

### Sanity Check

```
Score ventana ANÓMALA:  0.00072511
Score ventana NORMAL:   0.00070437
Diferencia:             0.000021
Anomalía > Normal:      True ✓
```

La dirección es correcta pero la separación es microscópica — insuficiente para discriminar globalmente.

---

## Diagnóstico — Convergencia excesivamente rápida

### Relación inversa entre val_loss1 y AUC

| Corrida | val_loss1 final Transfer | AUC Transfer |
|---|---|---|
| C4 | **0.0027** | 0.2253 |
| C3 | 0.0073 | 0.3714 |
| C2 | 0.0685 | **0.7308** |
| Baseline A (todas) | ~0.045-0.058 | ~0.66-0.71 |
| Baseline B (todas) | ~0.050-0.055 | ~0.65-0.71 |

**A menor val_loss1 del Transfer, menor AUC.** Existe una relación inversa directa y consistente en las 4 corridas.

### Por qué ocurre

Los pesos transferidos desde SWaT inicializan el modelo en un punto donde ya puede reconstruir señales de temperatura con bajo error desde las primeras épocas. Cuando el error de reconstrucción cae a ~0.003:

1. El modelo reconstruye datos **normales** con error ~0.003
2. El modelo reconstruye datos **anómalos** también con error ~0.003
3. La diferencia entre ambos es de **0.000021** — indetectable como señal de anomalía

Los baselines permanecen en ~0.050, donde la diferencia entre reconstruir datos normales vs anómalos es aún significativa y el threshold puede separar las clases.

### Rango "útil" de val_loss1

Basado en las 4 corridas, el rango donde USAD discrimina bien es:

```
val_loss1 ∈ [0.05, 0.10]  ← baselines, discriminación buena
val_loss1 ∈ [0.03, 0.05]  ← zona gris
val_loss1 < 0.01          ← sobre-entrenado, discriminación colapsa
```

La Corrida 2 del Transfer (AUC=0.73) fue la única que quedó en el rango útil, por las dinámicas adversariales aleatorias de esa ejecución específica.

---

## Resumen acumulado de las 4 corridas

| Corrida | Config clave | AUC Transfer | AUC Baseline A | AUC Baseline B | Nota |
|---|---|---|---|---|---|
| 1 | shuffle=False, z incorrecto | 0.6446* | 0.6814* | 0.7135* | Comparación inválida |
| **2** | shuffle=True, α=0.5, 80 épocas | **0.7308** | **0.7143** | 0.6955 | **Mejor corrida** |
| 3 | +150 épocas, α=0.1/β=0.9 | 0.3714 | 0.6589 | 0.6938 | Sobre-entrenado + alpha malo |
| 4 | 70 épocas, α=0.5/β=0.5 | 0.2253 | 0.6644 | 0.6506 | Sobre-entrenado más rápido |

*Comparación inválida (arquitecturas distintas).

---

## Conclusiones finales del experimento

### 1. El Transfer Learning es inestable

La Corrida 2 mostró AUC=0.7308 pero fue resultado de una trayectoria de convergencia específica que no se puede reproducir de forma controlada. Las corridas 3 y 4, con distintas configuraciones de épocas, siempre terminaron sobre-entrenadas.

### 2. Los Baselines son más robustos y predecibles

Baseline A y B oscilaron entre 0.65-0.71 AUC en todas las corridas, independientemente de la configuración. Son más confiables para producción.

### 3. El régimen de val_loss1 es determinante

USAD necesita operar con val_loss1 en el rango [0.05, 0.10] para discriminar bien. El Transfer llega a ese rango solo por accidente (C2); en C3 y C4 lo supera rápidamente.

### 4. Para controlar esto en el futuro

Se necesitaría **early stopping** basado en AUC de validación (no en val_loss1), o un **learning rate muy bajo** (1e-5 en Phase 2) para ralentizar la convergencia del Transfer y mantenerlo en el rango útil.

### 5. Recomendación para la monografía

El experimento documenta un hallazgo importante: **la inicialización por submatriz desde SWaT puede mejorar el AUC (+2.3% sobre baseline) pero introduce inestabilidad de entrenamiento porque los pesos pre-entrenados llevan al modelo a un régimen de sobre-reconstrucción que destruye la señal de anomalía.** El mejor modelo para uso práctico es el **Baseline A (z=120, Xavier aleatorio)** por su estabilidad y AUC comparable (0.66-0.71 consistente en todas las corridas).
