# Reporte de Análisis: Transfer Learning USAD — Estación 203

> Fecha de análisis: 2026-04-08
> Branch: `feature/transfer-learning-plan-b`
> Notebook: `transfer_learning_203.ipynb`

---

## 1. Método de Entrenamiento

### Plan B — Entrenamiento en Dos Etapas con Masking

Se implementó una estrategia de **transfer learning de dos etapas** basada en el algoritmo USAD (UnSupervised Anomaly Detection), con la variante "Plan B: Two-Stage Masking Training".

---

### Etapa 1: Pre-entrenamiento sobre datos limpios

| Parámetro       | Valor                                             |
|-----------------|---------------------------------------------------|
| Datos           | Split E, solo filas limpias (`t_mask ≠ -2000`)   |
| Epochs          | 50                                                |
| Batch Size      | 512                                               |
| Ventanas        | 130,818                                           |
| Optimizador     | Adam (dos instancias separadas)                   |

**Funciones de pérdida:**

```
Loss1 = (1/n) × MSE(data, w1)  +  (1 - 1/n) × MSE(data, w3)
Loss2 = (1/n) × MSE(data, w2)  -  (1 - 1/n) × MSE(data, w3)
```

Donde `n` es el número de epoch actual (el factor `1/n` decae progresivamente).

**Pase hacia adelante:**
```
w1 = Decoder1(Encoder(batch))       # reconstrucción directa
w2 = Decoder2(Encoder(batch))       # ruta paralela
w3 = Decoder2(Encoder(w1))          # ruta encadenada (componente adversarial)
```

**Objetivo**: establecer una base sólida de reconstrucción normal de temperatura.

---

### Etapa 2: Fine-tuning con Masking (Transfer Learning)

| Parámetro       | Valor                                               |
|-----------------|-----------------------------------------------------|
| Datos           | Split E completo + Split V completo (con masking)   |
| Inicialización  | Pesos guardados de la Etapa 1 (`model_stage1.pth`)  |
| Epochs          | 50                                                  |
| Batch Size      | 512                                                 |
| Ventanas        | ~191,000 (133,907 train + 57,355 val)               |

**Estrategia de masking:**
- El valor centinela `-2000` marca anomalías/faltantes en el entrenamiento.
- Cualquier ventana que contenga al menos un `-2000` recibe `loss_weight = 0.0`.
- Las ventanas limpias reciben `loss_weight = 1.0`.
- Las pérdidas se normalizan por el total de ventanas válidas.

**Resultado**: el modelo no aprende a reconstruir anomalías; solo aprende comportamiento normal.

---

## 2. Por Qué Se Utilizó Este Método

### Problema a resolver
La estación 203 tiene datos con anomalías masivas de temperatura durante agosto–septiembre 2023. El modelo USAD estándar fallaría si estas anomalías entraran directamente al entrenamiento, pues aprendería a reconstruirlas bien y ya no podría detectarlas.

### Motivaciones concretas

| Razón | Explicación |
|---|---|
| **Datos contaminados** | Los splits de entrenamiento contienen anomalías reales marcadas con centinela. No se podía simplemente eliminar o rellenar los valores. |
| **Masking en pérdida** | Ignorar ventanas con anomalías durante el entrenamiento evita que el modelo aprenda patrones anómalos. |
| **Transfer learning** | Primero entrenar solo con datos limpios da una inicialización robusta, luego el fine-tuning adapta el modelo al dominio real sin partir desde cero. |
| **Epoch-wise weighting** | El factor `1/n` en las pérdidas proporciona regularización implícita: al inicio el modelo aprende reconstrucción básica; al final refuerza la componente adversarial. |
| **Sin hard freezing** | No se congelaron capas explícitamente; la buena inicialización + el masking actúan como regularizadores naturales. |

---

## 3. Resultados

### Métricas finales en Split T (Test)

| Métrica              | Valor         |
|----------------------|---------------|
| **Accuracy**         | 45.32%        |
| **Precision**        | 0.0000        |
| **Recall**           | 0.0000        |
| **F1-Score**         | 0.0000        |
| Threshold (F1-opt.)  | 8,016.03      |
| Ventanas predichas como anomalía | 0 / 95,823 |
| Ventanas reales anomalías       | 52,396 / 95,823 |

### Matriz de Confusión

```
                    Pred. Normal   Pred. Anomalía
Real Normal            43,427             0
Real Anomalía          52,396             0
```

### Estadísticas del Anomaly Score (Split T)

| Estadístico | Valor    |
|-------------|----------|
| Mínimo      | 0.000486 |
| Máximo      | 4.708494 |
| Media       | 0.736713 |
| **Threshold** | **8,016.03** |

> **Problema crítico**: el threshold calculado sobre validación (8,016) es ~1,700× mayor que el score máximo del test (4.71). Todas las ventanas de test son clasificadas como normales.

---

## 4. Conclusiones

### 4.1 Falla del Threshold — Distribución Shift
El umbral óptimo de F1 se calculó sobre el Split V (validación), donde los anomaly scores deben haberse encontrado en el rango de miles. Sin embargo, en el Split T (agosto–septiembre 2023) los scores están entre 0 y 5. Esto indica un **cambio de distribución (distribution shift)** entre validación y test.

### 4.2 El Modelo No Detecta Ninguna Anomalía
Con threshold >> max(score_test), el recall es 0 y el F1 es 0. El modelo no sirve en su estado actual para detectar las anomalías masivas del período de test.

### 4.3 El Método es Arquitectónicamente Correcto, pero Falla en Generalización
La lógica de masking es válida. El problema no es el algoritmo en sí, sino que:
- El modelo puede haberse sobre-adaptado a los patrones del Split V.
- El Split T tiene características de temperatura radicalmente distintas (falla masiva de sensor).
- El threshold debe calcularse de otra forma (percentil fijo, estadística del score de entrenamiento, etc.).

### 4.4 Accuracy Engañosa
El 45.32% de accuracy proviene de clasificar todo como normal, y el Split T tiene ~45% de ventanas normales. Esta métrica no tiene valor informativo aquí.

---

## 5. Método Why — ¿Por Qué Falló?

| Pregunta | Respuesta |
|---|---|
| **¿Por qué se eligió este enfoque?** | Para entrenar USAD con datos contaminados sin que aprenda anomalías, usando masking en la pérdida y transfer learning entre etapas. |
| **¿Por qué falló la detección?** | El threshold se derivó de scores de validación que son órdenes de magnitud mayores que los scores de test, haciendo que ninguna ventana sea detectada. |
| **¿Por qué hay distribution shift?** | El Split T corresponde a un evento extremo (falla masiva de sensor) cuya magnitud en el espacio de reconstrucción es completamente distinta a lo que el modelo vio en validación. |
| **¿Por qué el modelo no aprendió a discriminar?** | Posiblemente la Etapa 2 con datos masivos y masking fue suficiente para que el modelo reconstruya bien todo, incluso lo que debería ser "difícil de reconstruir". |
| **¿Qué se debe cambiar?** | 1) Calcular el threshold a partir de la distribución de scores del conjunto de entrenamiento (p.ej. percentil 99). 2) Evaluar si el fine-tuning de la Etapa 2 debe limitarse (menos epochs, learning rate bajo, o freezing parcial). 3) Revisar si la escala de los anomaly scores de validación vs test es correcta. |

---

## 6. Explicación Sencilla: ¿Qué se Hizo?

### El problema
Tenemos datos de temperatura de la estación 203. En cierto período, el sensor falló y registró temperaturas completamente fuera de rango. Queremos que un modelo aprenda qué es "temperatura normal" y luego avise cuando detecte algo raro.

El reto: los datos de entrenamiento también contienen algunas de esas temperaturas raras. Si el modelo las ve durante el entrenamiento, aprenderá que también son normales, y nunca las marcará como anomalías.

### La solución intentada

**Paso 1 — Solo datos buenos (Etapa 1)**
Se extraen únicamente los registros sin problemas (marcados con `-2000` = excluidos), se construyen ventanas de 60 minutos, y se entrena el modelo durante 50 ciclos. El modelo aprende: "así se ve la temperatura normal".

**Paso 2 — Datos completos con máscara (Etapa 2)**
Ahora se usan todos los datos, incluyendo los contaminados. Pero se aplica una "máscara de pérdida": si una ventana de 60 minutos tiene algún valor raro, esa ventana no cuenta para el aprendizaje (su contribución al error se pone a 0). El modelo parte de lo aprendido en el Paso 1 y se afina con los datos reales, sin aprender los valores anómalos.

**Paso 3 — Detección**
Una vez entrenado, el modelo intenta reconstruir cada ventana nueva. Si la reconstrucción es mala (error alto), se marca como anomalía. El umbral que decide "qué tan malo es malo" se calcula sobre datos de validación.

### ¿Qué salió mal?
El umbral calculado en validación fue de ~8,000, pero en el test los scores máximos son ~4.7. El modelo reconstruye todo demasiado bien (o el umbral quedó altísimo), y no detecta nada. Es como calibrar una balanza para detectar objetos de más de 10 kg, y luego poner objetos de 1 g.

---

## 7. Arquitectura del Modelo USAD

### Visión General

```
              ┌─────────────────────────────────┐
              │            ENCODER               │
              │   60 → 30 → 15 → 30 (latent)    │
              └──────────────┬──────────────────┘
                             │ z (latent vector, dim=30)
              ┌──────────────┴──────────────────┐
              │                                  │
    ┌─────────▼──────────┐           ┌──────────▼─────────┐
    │     DECODER 1       │           │     DECODER 2       │
    │  30 → 15 → 30 → 60  │           │  30 → 15 → 30 → 60  │
    └─────────┬───────────┘           └──────────┬──────────┘
              │ w1                               │ w2
              │
     ┌────────▼────────────────────┐
     │  Encoder(w1) → Decoder2     │  ← componente adversarial
     └────────────────────────────┘
              │ w3
```

### Dimensiones por Capa

| Componente | Capas         | Dimensiones        | Activación |
|------------|---------------|--------------------|------------|
| Encoder    | Linear → ReLU | 60 → 30 → 15 → 30  | ReLU       |
| Decoder1   | Linear → ReLU | 30 → 15 → 30 → 60  | Sigmoid    |
| Decoder2   | Linear → ReLU | 30 → 15 → 30 → 60  | Sigmoid    |

### Arquitectura Anterior vs Actual

| Aspecto                | Antes (USAD estándar) | Plan B (actual)               |
|------------------------|-----------------------|-------------------------------|
| Entrenamiento          | Una sola etapa        | Dos etapas                    |
| Datos de entrada       | Solo datos limpios o sin filtrar | Clean + masked |
| Manejo de anomalías    | Eliminación o relleno | Masking en pérdida (centinel) |
| Transfer learning      | No                    | Sí (Stage1 → Stage2)          |
| Threshold              | Percentil fijo        | F1-optimo sobre validación    |
| Resultado              | Desconocido           | F1 = 0 (falla en test)        |

---

## 8. Resumen de Datos

| Split | Filas      | Centineles | Ventanas (W=60) | Anomalías reales |
|-------|------------|------------|-----------------|------------------|
| E (Train) | 133,966 | 3,089 | 133,907 | — |
| V (Val)   | 57,414  | 184   | 57,355  | — |
| T (Test)  | 95,882  | 0     | 95,823  | 52,396 ventanas (54.7%) |

- **Normalización**: Z-Score ajustado solo sobre datos limpios de E (media=21.24°C, std=3.14°C)
- **Ventana deslizante**: tamaño=60, stride=1, etiqueta=1 si cualquier timestep tiene flag=1

---

## 9. Próximos Pasos Recomendados

1. **Recalcular el threshold** usando percentil 99 o 99.9 sobre los scores del conjunto de entrenamiento en lugar del F1-óptimo sobre validación.
2. **Visualizar la distribución de scores** en train, val y test por separado para confirmar el distribution shift.
3. **Evaluar congelamiento parcial**: congelar el Encoder durante la Etapa 2 para preservar mejor la representación aprendida en datos limpios.
4. **Reducir epochs de Etapa 2** o usar learning rate más bajo para evitar que el modelo "olvide" lo aprendido (catastrophic forgetting).
5. **Probar normalización por separado** para el Split T; si las temperaturas del período fallido están en un rango muy diferente, la normalización entrenada en E puede distorsionar los scores.
