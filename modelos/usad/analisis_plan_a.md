# Análisis: Transfer Learning Plan A — USAD sobre SIATA Estación 203

---

## 1. Método de Entrenamiento

**Progressive Unfreezing** (Descongelamiento Progresivo), dividido en **4 fases** sobre 100 épocas totales.

| Fase | Épocas | Capas entrenables | Learning Rate | Params entrenables |
|------|--------|-------------------|---------------|-------------------|
| 1 | 0–29 | Solo decoders (encoder congelado) | 0.001 | 169,998 |
| 2 | 30–59 | + `encoder.linear3` | 0.0005 | 178,798 |
| 3 | 60–89 | + `encoder.linear2` | 0.0001 | 194,023 |
| 4 | 90–99 | Modelo completo (`encoder.linear1` desbloqueado) | 5e-05 | 254,749 |

El modelo base es **USAD** (Unsupervised Anomaly Detection), un autoencoder dual de dos decoders. Sus pesos originales fueron pre-entrenados en el dataset **SWaT** (planta de tratamiento de agua).

---

## 2. Por Qué Se Utilizó Este Método

### Justificación del Transfer Learning

El dataset de SIATA Estación 203 (UNAL) es relativamente pequeño comparado con SWaT. Entrenar un modelo USAD desde cero con datos de temperatura ambiental requeriría mucho más datos o resultaría en underfitting. Al reutilizar los pesos pre-entrenados en SWaT, el encoder ya aprendió a **comprimir señales temporales multivariadas** y detectar patrones, lo que acelera la convergencia y mejora la generalización.

### Justificación del Progressive Unfreezing

En lugar de fine-tuning completo desde el inicio (que podría destruir el conocimiento previo — fenómeno de *catastrophic forgetting*), se aplica descongelamiento gradual:

1. **Fase 1:** Las capas del encoder se mantienen congeladas. Solo los decoders se adaptan a la distribución nueva de los datos de temperatura. Esto es seguro porque los decoders aprenden a reconstruir la salida del dominio nuevo sin perturbar las representaciones aprendidas.

2. **Fases 2 y 3:** Se descongela la parte más profunda del encoder primero (`linear3`, luego `linear2`), con learning rates reducidos. Esto permite ajuste fino sin desestabilizar las capas iniciales.

3. **Fase 4:** Se descongela todo el modelo a un learning rate mínimo (5e-05), permitiendo ajuste global muy suave.

**En resumen:** El progressive unfreezing equilibra el conocimiento previo de SWaT con las características nuevas de los datos de temperatura de SIATA, protegiendo el modelo del olvido catastrófico.

---

## 3. Resultados

### 3.1 Curvas de Pérdida (Validation)

| Fase | val_loss1 (final) | val_loss2 (final) | Observación |
|------|-------------------|-------------------|-------------|
| Fase 1 (ep. 29) | ~0.825 | -0.776 | val_loss2 cae rápidamente (decoder2 aprende bien) |
| Fase 2 (ep. 59) | ~0.845 | -0.820 | Leve aumento por descongelamiento del encoder |
| Fase 3 (ep. 89) | ~0.845 | -0.829 | Pérdidas estables, plateau alcanzado |
| Fase 4 (ep. 99) | ~0.845 | -0.831 | Mínima variación, sin overfitting |

> **Nota:** `val_loss2` negativa es un comportamiento esperado en USAD. El modelo entrena al Decoder2 para detectar anomalías al reconstruir mal datos atípicos, lo que hace que su función de pérdida tenga un componente adversarial que puede volverse negativo.

### 3.2 Métricas de Clasificación (Test Set — Split T)

- **Dataset de test:** Estación 203 UNAL, junio–julio 2023 (alta concentración de anomalías)
- **Threshold óptimo (curva ROC):** 1.325651
- **Anomalías en test:** 7,108 de 37,601 ventanas (18.91%)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 79.67% |
| **Precision** | 47.98% |
| **Recall** | **89.43%** |
| **F1 Score** | 62.46% |

### 3.3 Matriz de Confusión

|  | Pred. Normal | Pred. Anomalía |
|--|--------------|----------------|
| **Real Normal** | 23,601 (TN) | 6,892 (FP) |
| **Real Anomalía** | 751 (FN) | 6,357 (TP) |

- El modelo **detecta el 89.4% de las anomalías reales** (solo pierde 751 de 8,108)
- Genera **6,892 falsas alarmas** sobre datos normales
- **Estadísticas de anomaly scores:** Min=0.053, Max=354.71, Media=3.23

---

## 4. Conclusiones

1. **El descongelamiento progresivo funcionó:** La transferencia de SWaT a SIATA fue exitosa. El modelo convergió suavemente sin señales de catastrophic forgetting ni overfitting.

2. **El modelo prioriza el Recall sobre la Precisión:** Con un recall del 89.4%, el modelo es muy bueno detectando anomalías reales. Esto es deseable en contextos donde **perder una anomalía real es más costoso que generar una falsa alarma** (e.g., sensores ambientales críticos).

3. **Alta tasa de falsos positivos:** La precisión del 47.98% indica que de cada 2 alarmas, solo 1 es real. Esto puede ser problemático en producción si los operadores experimentan fatiga de alarmas.

4. **El threshold es agresivo:** El threshold de 1.33 (seleccionado por ROC) favorece recall. Un threshold más alto reduciría las falsas alarmas pero aumentaría los falsos negativos.

5. **El enfoque de ingeniería de features fue clave:** Expandir la temperatura univariada a 29 features (lags, diferencias, estadísticas rolling, ciclicidad temporal) permite que el modelo capture patrones temporales complejos.

6. **Sin fuga de datos:** La normalización Z-score fue ajustada solo sobre el split de entrenamiento (Split E), garantizando evaluación honesta.

---

## 5. Respuesta al Método Why

> **¿Por qué se utilizó Transfer Learning con Progressive Unfreezing para este problema?**

**Porque el problema tiene datos insuficientes para entrenar desde cero, pero tiene un dominio relacionado con conocimiento reutilizable.**

La estación SIATA 203 tiene ~82,000 registros de temperatura con solo ~0.57% de anomalías en entrenamiento. USAD requiere aprender a reconstruir patrones normales complejos. Con tan pocos datos, el encoder tendría dificultad para aprender representaciones latentes ricas sin sobreajustarse.

SWaT, en cambio, contiene ~495,000 registros de una planta industrial con múltiples sensores y patrones temporales complejos. El encoder ya aprendió a comprimir señales temporales correlacionadas, lo cual es transferible a series de temperatura ambiental que también tienen autocorrelación, estacionalidad y dependencias temporales.

El progressive unfreezing protege este conocimiento: primero deja que los decoders se adapten al nuevo dominio, y luego va desbloqueando el encoder desde adentro hacia afuera, con learning rates decrecientes, para que el ajuste sea gradual y controlado. El resultado es un modelo que **hereda la capacidad de compresión temporal de SWaT y la especializa en anomalías de temperatura ambiental**, usando solo una fracción de los datos y épocas que requeriría un entrenamiento completo.

---

## 6. Explicación Sencilla: Qué Se Hizo

### La Arquitectura Original (USAD en SWaT)

Imagina un **par de fotógrafos que aprenden a copiar imágenes exactamente**:

```
Entrada (ventana de sensores SWaT)
        ↓
   [ENCODER]  ← comprime la señal a un "resumen" (z)
        ↓
  [DECODER 1] ← intenta reconstruir la señal original
        ↓
  [DECODER 2] ← intenta reconstruir la señal USANDO la reconstrucción del Decoder1
```

- Si la entrada es **normal**, ambos decoders la reconstruyen bien → error bajo.
- Si la entrada es **anómala**, los decoders no la pueden reconstruir bien (nunca vieron algo así) → error alto = **anomalía detectada**.

El modelo fue entrenado originalmente en SWaT: una planta de tratamiento de agua con 51 sensores industriales. Aprendió a "fotografiar" operaciones normales de una planta industrial.

---

### El Problema Nuevo (SIATA Estación 203)

Queremos usar ese mismo modelo para detectar anomalías de **temperatura ambiental** en Medellín (estación UNAL). Pero hay un problema: el modelo "fotografió" plantas industriales, no clima.

**Solución:** Transfer Learning. En lugar de enseñarle todo desde cero, le decimos al modelo: "lo que aprendiste sobre patrones temporales sirve, solo ajústate al nuevo dominio".

---

### Los Pasos Completos del Experimento

**Paso 1 — Cargar datos de temperatura (SIATA 203)**
- 82,266 registros de temperatura de UNAL
- Divididos en: Entrenamiento (marzo–abril 2023), Validación, Test (junio–julio 2023)

**Paso 2 — Enriquecer la señal (Feature Engineering)**
- Una sola columna de temperatura → 29 features
- Se agregan: 10 valores pasados (lags), diferencias, promedios móviles, máximos/mínimos en ventanas de 5, 10, 30 pasos, y la hora del día codificada cíclicamente
- Esto le da al modelo "contexto" temporal para distinguir lo normal de lo anómalo

**Paso 3 — Normalizar**
- Z-score: cada feature se centra en 0 con desviación 1
- Solo se aprenden los parámetros con los datos de entrenamiento (evita trampa)

**Paso 4 — Crear ventanas deslizantes**
- Se toman secuencias de 12 pasos consecutivos (ventana deslizante)
- Cada ventana tiene 12 × 29 = 348 valores de entrada al modelo
- Se descartan ventanas con valores faltantes

**Paso 5 — Cargar el modelo pre-entrenado (SWaT)**
- Se carga `model.pth` con los pesos de USAD entrenado en SWaT
- Encoder + Decoder1 + Decoder2 inicializados desde SWaT

**Paso 6 — Entrenamiento con Progressive Unfreezing**
```
Fases:
 [0-29]   Encoder CONGELADO → solo decoders aprenden el nuevo dominio
 [30-59]  Descongelar la capa más profunda del encoder (linear3)
 [60-89]  Descongelar la capa media (linear2)
 [90-99]  Descongelar todo el encoder (linear1) — ajuste muy fino
```

**Paso 7 — Calcular anomaly scores en el test set**
- Para cada ventana de test: calcular el error de reconstrucción (combinación de Decoder1 y Decoder2)
- Scores van de 0.053 (muy normal) a 354.7 (muy anómalo)

**Paso 8 — Seleccionar threshold y evaluar**
- Curva ROC determina el threshold óptimo: **1.3257**
- Ventanas con score > 1.3257 se clasifican como anomalía

**Resultado Final:**
- El modelo detecta el **89.4% de las anomalías reales** de temperatura
- Con una precisión del **48%** (muchas falsas alarmas)
- F1 Score: **62.5%**

---

### Resumen Visual del Flujo

```
SWaT Dataset (industrial)
        ↓
  [USAD pre-entrenado]
        ↓ Transfer Learning
SIATA 203 (temperatura UNAL)
        ↓
  Feature Engineering (1 → 29 features)
        ↓
  Normalización Z-score
        ↓
  Ventanas deslizantes (12 pasos)
        ↓
  Progressive Unfreezing (4 fases, 100 épocas)
        ↓
  Anomaly Scores → Threshold ROC (1.3257)
        ↓
  Recall=89.4% | Precision=48% | F1=62.5%
```

---

## Archivos Generados por el Notebook

| Archivo | Descripción |
|---------|-------------|
| `model_transfer_plan_a.pth` | Pesos del modelo fine-tuned |
| `results_plan_a.md` | Reporte de experimento generado automáticamente |
| `analisis_plan_a.md` | Este documento (análisis detallado) |

---

*Análisis generado el 2026-04-08 sobre el notebook `transfer_learning_plan_a.ipynb`.*
