# Plan: Transfer Learning USAD → Detección de Anomalías de Temperatura SIATA

## Contexto

El modelo USAD fue pre-entrenado en el dataset SWaT (51 sensores industriales, w_size=612, z_size=1200). Se quiere reutilizar ese conocimiento para detectar anomalías de temperatura en las 4 estaciones meteorológicas SIATA del Valle de Aburrá (Medellín), usando datos de marzo-abril 2025.

**Desafío principal:** Desajuste de dimensiones — el modelo original tiene w_size=612 (12×51) y el nuevo tendrá w_size=240 (60×4). No es posible copiar pesos directamente capa por capa. La estrategia usada es **inicialización por submatriz**: como la nueva arquitectura es estrictamente más pequeña en cada capa, se copia la submatriz superior-izquierda `W[:out_new, :in_new]` de los pesos pre-entrenados como punto de partida.

**Por qué z_size=1200 (igual al original):** Es la capa donde más transferencia hay — encoder.linear3 tiene el 100% de las filas compatibles (ambas son 1200), y decoder1/decoder2.linear1 tienen el 100% de las columnas compatibles. Esto maximiza la información transferida del espacio latente pre-entrenado.

---

## Parámetros del nuevo modelo

| Parámetro | Valor | Justificación |
|---|---|---|
| `window_size` | 60 | 1 hora de contexto a resolución 1-minuto |
| `w_size` | 240 | 60 timesteps × 4 estaciones |
| `z_size` | 1200 | Igual al pre-entrenado → máxima transferencia en capa latente |
| `BATCH_SIZE` | 512 | Dataset más pequeño que SWaT |
| Épocas Fase 1 | 10 | Warm-up solo decoders (encoder congelado) |
| Épocas Fase 2 | 40 | Fine-tuning completo, lr=1e-4 |

**Dimensiones por capa:**
- Encoder: 240→120→60→1200
- Decoder: 1200→60→120→240

---

## Archivos a modificar / crear

| Archivo | Cambio |
|---|---|
| `usad.py` | Agregar función `load_pretrained_submatrix()` al final |
| `Transfer_Learning_SIATA.ipynb` | Crear — notebook principal (nuevo archivo) |
| `utils.py` | Sin cambios |
| `model.pth` | Solo lectura (fuente de pesos) |

---

## Paso 1: Modificar `usad.py` — agregar helper

Agregar al final del archivo:

```python
def load_pretrained_submatrix(model, checkpoint_path):
    """
    Inicializa un UsadModel más pequeño copiando las submatrices superiores-izquierdas
    de un checkpoint pre-entrenado. Requiere que todas las dimensiones del nuevo modelo
    sean <= las del modelo pre-entrenado.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    def copy_sub(src_state, dst_module):
        dst_state = dst_module.state_dict()
        new_state = {}
        for key in dst_state:
            src = src_state[key]
            dst = dst_state[key]
            if src.dim() == 2:  # matriz de pesos
                new_state[key] = src[:dst.shape[0], :dst.shape[1]].clone()
            elif src.dim() == 1:  # vector de bias
                new_state[key] = src[:dst.shape[0]].clone()
        dst_module.load_state_dict(new_state)

    copy_sub(ckpt['encoder'], model.encoder)
    copy_sub(ckpt['decoder1'], model.decoder1)
    copy_sub(ckpt['decoder2'], model.decoder2)
    return model
```

---

## Paso 2: Notebook `Transfer_Learning_SIATA.ipynb`

### Sección 1 — Imports y Setup
```python
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
from utils import *
from usad import *
device = get_default_device()
```

### Sección 2 — Preprocesamiento de datos
```python
# 1. Cargar CSV y pivotar a formato ancho
df = pd.read_csv('data_new/temperatura_estaciones_marzo_abril_2025.csv',
                 parse_dates=['fecha_hora'])

# 2. Extraer labels antes de pivotar
labels_long = df[['fecha_hora', 'calidad_dudosa']].copy()

# 3. Pivot: index=fecha_hora, columns=codigo, values=t
df_wide = df.pivot(index='fecha_hora', columns='codigo', values='t').sort_index()
# Resultado: ~87,577 filas × 4 columnas de temperatura

# 4. Label por timestamp (True si CUALQUIER estación tiene calidad dudosa)
labels_per_ts = labels_long.groupby('fecha_hora')['calidad_dudosa'].any().astype(int)
labels_per_ts = labels_per_ts.reindex(df_wide.index, fill_value=0)

# 5. Normalizar (fit SOLO sobre datos normales)
normal_mask = (labels_per_ts.values == 0)
scaler = MinMaxScaler()
scaler.fit(df_wide.values[normal_mask])
df_scaled = scaler.transform(df_wide.values)

# 6. Sliding windows (window_size=60, w_size=240)
window_size = 60
data = df_scaled
labels_arr = labels_per_ts.values

windows = data[np.arange(window_size)[None,:] + np.arange(len(data)-window_size)[:,None]]
windows = windows.reshape(len(windows), -1)  # (N, 240)

# 7. Labels por ventana: mayoría simple (>30 de 60 timesteps anómalos)
window_labels = np.array([
    1 if np.sum(labels_arr[i:i+window_size]) > window_size // 2 else 0
    for i in range(len(data) - window_size)
])

# 8. Split: entrenamiento solo con ventanas 100% limpias
clean_mask = np.array([
    np.sum(labels_arr[i:i+window_size]) == 0
    for i in range(len(data) - window_size)
])
clean_windows = windows[clean_mask]
split = int(0.8 * len(clean_windows))
windows_train = clean_windows[:split]
windows_val   = clean_windows[split:]
windows_test  = windows
y_test        = window_labels

# 9. DataLoaders
w_size = 240; z_size = 1200; BATCH_SIZE = 512
train_loader = data_utils.DataLoader(
    data_utils.TensorDataset(torch.from_numpy(windows_train).float()),
    batch_size=BATCH_SIZE, shuffle=False)
val_loader = data_utils.DataLoader(
    data_utils.TensorDataset(torch.from_numpy(windows_val).float()),
    batch_size=BATCH_SIZE, shuffle=False)
test_loader = data_utils.DataLoader(
    data_utils.TensorDataset(torch.from_numpy(windows_test).float()),
    batch_size=BATCH_SIZE, shuffle=False)
```

### Sección 3 — Inicialización por submatriz (Transfer Learning)
```python
model = UsadModel(w_size, z_size)
model = to_device(model, device)
model = load_pretrained_submatrix(model, 'model.pth')
# Verificar: imprimir normas de pesos para confirmar que no son Xavier aleatorio
```

### Sección 4 — Fase 1: Decoder Warm-Up (10 épocas, encoder congelado)
```python
for param in model.encoder.parameters():
    param.requires_grad = False

opt1 = torch.optim.Adam(model.decoder1.parameters())
opt2 = torch.optim.Adam(model.decoder2.parameters())
history_phase1 = []
for epoch in range(10):
    for [batch] in train_loader:
        batch = to_device(batch, device)
        loss1, loss2 = model.training_step(batch, epoch+1)
        loss1.backward(); opt1.step(); opt1.zero_grad()
        loss1, loss2 = model.training_step(batch, epoch+1)
        loss2.backward(); opt2.step(); opt2.zero_grad()
    result = evaluate(model, val_loader, epoch+1)
    model.epoch_end(epoch, result)
    history_phase1.append(result)
```

### Sección 5 — Fase 2: Fine-Tuning completo (40 épocas, lr=1e-4)
```python
for param in model.encoder.parameters():
    param.requires_grad = True

history_phase2 = training(40, model, train_loader, val_loader,
                          opt_func=lambda p: torch.optim.Adam(p, lr=1e-4))
plot_history(history_phase1 + history_phase2)
```

### Sección 6 — Guardar modelo
```python
torch.save({'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()},
           'model_siata_transfer.pth')
```

### Sección 7 — Evaluación
```python
results = testing(model, test_loader, alpha=0.5, beta=0.5)
y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                         results[-1].flatten().detach().cpu().numpy()])
threshold = ROC(y_test, y_pred)       # de utils.py — sin cambios
histogram(y_test, y_pred)             # de utils.py — sin cambios
y_pred_binary = (y_pred >= threshold).astype(int)
confusion_matrix(y_test, y_pred_binary)  # de utils.py — sin cambios
```

### Sección 8 — Ablación (para la monografía)
Entrenar dos baselines adicionales con la misma arquitectura y datos:
- **Baseline A**: `UsadModel(240, 1200)` con inicialización Xavier aleatoria, 50 épocas
- **Baseline B**: `UsadModel(240, 100)` con z_size reducido, 50 épocas

Comparar AUC-ROC y curvas de convergencia entre los 3 modelos.

---

## Verificación / Test end-to-end

1. Ejecutar todas las celdas del notebook secuencialmente sin errores
2. Confirmar shapes intermedias: `windows.shape == (87517, 240)` aprox.
3. Confirmar que `load_pretrained_submatrix` carga sin errores y las normas de pesos son ≠ de una inicialización Xavier
4. Phase 1: val_loss1 debe decrecer en los 10 epochs (decoders aprendiendo)
5. Phase 2: val_loss1 y val_loss2 deben seguir el patrón del notebook original (loss2 negativa → AE2 adversarial activo)
6. `ROC()` debe producir AUC > 0.5 (mejor que azar)
7. Comparar AUC entre los 3 modelos (transfer vs baselines)

---

## Cómo correr en Google Colab

**Sí, este modelo corre perfectamente en Colab.** Colab tiene PyTorch, sklearn y pandas pre-instalados, y ofrece GPU gratuita (T4) que acelera el entrenamiento.

### Pasos para correr en Colab

**Opción A — Subir archivos manualmente:**
1. Ir a [colab.research.google.com](https://colab.research.google.com) y crear un nuevo notebook
2. En el panel izquierdo, pestaña "Files" (ícono de carpeta), subir:
   - `usad.py`
   - `utils.py`
   - `model.pth`
   - `data_new/temperatura_estaciones_marzo_abril_2025.csv`
3. Activar GPU: `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`
4. Copiar el contenido del notebook `Transfer_Learning_SIATA.ipynb` o abrirlo directamente

**Opción B — Montar Google Drive (recomendada para monografía):**
```python
# Celda 1: montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Celda 2: ir al directorio del proyecto
import os
os.chdir('/content/drive/MyDrive/monografia/modelos/usad')  # ajustar ruta
```

**Opción C — Clonar desde GitHub (si subes el repo):**
```python
!git clone https://github.com/tu-usuario/tu-repo.git
%cd tu-repo
```

### Verificar GPU disponible
```python
# Agregar al inicio del notebook
import torch
print(torch.cuda.is_available())    # debe ser True en Colab con GPU
print(torch.cuda.get_device_name(0))  # ej: "Tesla T4"
# get_default_device() en utils.py detectará la GPU automáticamente
```

### Instalar dependencias si faltan
```python
# Normalmente no es necesario en Colab, pero por si acaso:
!pip install scikit-learn seaborn -q
```

### Tiempo estimado de ejecución en Colab (T4 GPU)
| Fase | Tiempo aprox. |
|---|---|
| Preprocesamiento | 1-2 min |
| Fase 1 (10 épocas, encoder congelado) | 2-3 min |
| Fase 2 (40 épocas, fine-tuning) | 8-12 min |
| Evaluación y gráficas | 1 min |
| **Total** | **~15 min** |

---

## Cómo probar el modelo

### Prueba rápida (sanity check)
```python
# Después de cargar el modelo, probar con un batch pequeño
model.eval()
sample = torch.from_numpy(windows_test[:32]).float()
sample = to_device(sample, device)
with torch.no_grad():
    z = model.encoder(sample)
    w1 = model.decoder1(z)
    w2 = model.decoder2(model.encoder(w1))
    score = 0.5*torch.mean((sample-w1)**2, axis=1) + 0.5*torch.mean((sample-w2)**2, axis=1)
print("Scores shape:", score.shape)      # debe ser (32,)
print("Score range:", score.min().item(), "-", score.max().item())
print("Shapes OK:", w1.shape == sample.shape)  # debe ser True
```

### Probar con una ventana de temperatura anómala conocida
```python
# Buscar un timestamp con calidad dudosa conocida
anomaly_idx = np.where(window_labels == 1)[0][0]
normal_idx = np.where(window_labels == 0)[0][0]

sample_anomaly = torch.from_numpy(windows_test[anomaly_idx:anomaly_idx+1]).float()
sample_normal  = torch.from_numpy(windows_test[normal_idx:normal_idx+1]).float()
sample_anomaly = to_device(sample_anomaly, device)
sample_normal  = to_device(sample_normal, device)

with torch.no_grad():
    score_a = testing(model, data_utils.DataLoader(
        data_utils.TensorDataset(sample_anomaly), batch_size=1))
    score_n = testing(model, data_utils.DataLoader(
        data_utils.TensorDataset(sample_normal),  batch_size=1))

print(f"Score anomalía: {score_a[0].item():.6f}")
print(f"Score normal:   {score_n[0].item():.6f}")
# El score de anomalía debe ser mayor que el de normal
```

### Cargar modelo guardado y re-evaluar
```python
checkpoint = torch.load('model_siata_transfer.pth', map_location=device)
model2 = UsadModel(w_size, z_size)
model2 = to_device(model2, device)
model2.encoder.load_state_dict(checkpoint['encoder'])
model2.decoder1.load_state_dict(checkpoint['decoder1'])
model2.decoder2.load_state_dict(checkpoint['decoder2'])
# Re-ejecutar testing() para verificar que los resultados son reproducibles
```

---

## Paso adicional: crear PLAN.md en el proyecto

Al implementar, también crear `/mnt/c/personal/monografia/modelos/usad/PLAN.md`
con este mismo plan, para tenerlo junto al código del proyecto.

---

## Archivos críticos

- `usad.py` — agregar `load_pretrained_submatrix()` al final
- `utils.py` — sin cambios, reusar `ROC`, `histogram`, `confusion_matrix`, `plot_history`
- `model.pth` — fuente de pesos pre-entrenados (solo lectura)
- `data_new/temperatura_estaciones_marzo_abril_2025.csv` — datos SIATA
- `USAD.ipynb` — referencia para patrones de ventanas, DataLoaders y checkpoints
