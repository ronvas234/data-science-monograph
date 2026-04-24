"""Patch transfer_learning_siata_68_v4.ipynb — aplica los 9 cambios del plan."""
import json, copy, re, sys

NB_PATH = "modelos/usad/transfer_learning_siata_68_v4.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

def code_cell(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines if isinstance(lines, list) else [lines],
    }

def md_cell(lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines if isinstance(lines, list) else [lines],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 — Título / changelog (celda 0)
# ─────────────────────────────────────────────────────────────────────────────
old_title = "".join(cells[0]["source"])
new_title = old_title.replace("v2 — Alineado con Monografia", "v4 — Alineado con Monografia")

changelog = """\n\n**Cambios v4** respecto a v2:\n- `epochs = 10`: entrenamiento limitado a 10 épocas máximo\n- `FiltFiltPreprocessor`: Butterworth lowpass (zero-phase) ANTES de Z-score\n- `precision_recall_curve_plot`: argmax(F1) → argmax(balanced_accuracy)\n- `metics`: agrega balanced_accuracy\n- `reconstruir_serie_usad_dual` + `dataset_error_usad_dual`: score combinado α·MSE_d1 + β·MSE_d2\n- Celda diagnóstico TL: arquitectura, estadísticas de pesos, verificación submatriz\n"""
new_title = new_title.rstrip() + changelog

cells[0]["source"] = [new_title]
print("[OK] celda 0 — título/changelog")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 — Config.epochs = 10 (celda 4)
# ─────────────────────────────────────────────────────────────────────────────
src4 = "".join(cells[4]["source"])
src4 = re.sub(r"epochs:\s*int\s*=\s*100", "epochs: int = 10   # v4: limitado a 10 épocas máximo", src4)
cells[4]["source"] = [src4]
print("[OK] celda 4 — epochs=10")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 — Insertar FiltFiltPreprocessor (2 nuevas celdas entre orig 5 y 6)
#           Después de celda 5 (markdown DataModule), antes de celda 6 (code load)
# ─────────────────────────────────────────────────────────────────────────────
filtfilt_md = md_cell(
    "## 2b. FiltFiltPreprocessor — Suavizado Butterworth antes de Z-score\n"
)

filtfilt_code_src = """\
from scipy.signal import butter, filtfilt

class FiltFiltPreprocessor:
    \"\"\"Butterworth lowpass bidireccional (zero-phase) aplicado a señal cruda.
    Se ejecuta ANTES de ZScoreNormalizer. No necesita manejo de sentinel
    (este notebook no usa valores centinela en los datos de train).
    \"\"\"

    def __init__(self, cutoff_hz: float = 0.05, order: int = 2):
        self._cutoff = cutoff_hz
        self._order  = order

    def apply(self, series: pd.Series) -> pd.Series:
        \"\"\"Aplica filtfilt zero-phase. pd y np disponibles desde celda 6.\"\"\"
        b, a = butter(self._order, self._cutoff, btype='low', analog=False)
        filtered = filtfilt(b, a, series.values.astype(np.float64))
        return pd.Series(filtered.astype(np.float32), index=series.index)


print('FiltFiltPreprocessor OK')
"""
filtfilt_code = code_cell([filtfilt_code_src])

# Insert after original index 5 → between [5] and [6]
cells.insert(6, filtfilt_code)
cells.insert(6, filtfilt_md)
print("[OK] celdas 6-7 — FiltFiltPreprocessor insertadas (era orig idx 5→6)")

# After insertion: orig [6] → [8], orig [7] → [9], etc.

# ─────────────────────────────────────────────────────────────────────────────
# Paso 4 — Normalización (ahora en índice 9, era orig 7)
# ─────────────────────────────────────────────────────────────────────────────
new_norm_src = """\
# ── Paso 1: suavizado Butterworth (zero-phase) ANTES de Z-score ───────────────
preprocessor = FiltFiltPreprocessor(cutoff_hz=0.05, order=2)

data_train['t_filtered'] = preprocessor.apply(data_train['t'])
data_val['t_filtered']   = preprocessor.apply(data_val['t'])
data_test['t_filtered']  = preprocessor.apply(data_test['t'])

std_cruda    = data_train[data_train['flag'] == 0]['t'].std()
std_filtrada = data_train[data_train['flag'] == 0]['t_filtered'].std()
print(f'Std señal cruda    (train, flag==0): {std_cruda:.4f}°C')
print(f'Std señal filtrada (train, flag==0): {std_filtrada:.4f}°C')

# ── Paso 2: Z-score sobre señal filtrada ──────────────────────────────────────
media = data_train[data_train['flag'] == 0]['t_filtered'].mean()
std   = data_train[data_train['flag'] == 0]['t_filtered'].std()

print(f'Media (filtrada): {media:.2f}°C | Std (filtrada): {std:.2f}°C')

train_norm = ((data_train['t_filtered'] - media) / std).values
val_norm   = ((data_val['t_filtered']   - media) / std).values
test_norm  = ((data_test['t_filtered']  - media) / std).values

print(f'\\ntrain_norm — min: {train_norm.min():.3f} | max: {train_norm.max():.3f} | mean: {train_norm.mean():.3f}')
"""
cells[9]["source"] = [new_norm_src]
print("[OK] celda 9 — normalización filtfilt→Z-score (era orig 7)")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 5 — Helper functions (ahora en índice 13, era orig 11)
# ─────────────────────────────────────────────────────────────────────────────
src13 = "".join(cells[13]["source"])

# 5a — agregar balanced_accuracy_score al import
src13 = src13.replace(
    "from sklearn.metrics import (\n    precision_recall_curve, f1_score, accuracy_score,\n    precision_score, recall_score,",
    "from sklearn.metrics import (\n    precision_recall_curve, f1_score, accuracy_score,\n    balanced_accuracy_score,          # v4\n    precision_score, recall_score,",
)
print("[OK] 5a — balanced_accuracy_score importado")

# 5b — agregar reconstruir_serie_usad_dual después de reconstruir_serie_usad
dual_recon = """

def reconstruir_serie_usad_dual(model, data_norm_arr, config, media, std, device):
    \"\"\"Score combinado USAD: retorna (t_pred_d1, t_pred_d2) en escala original.

    w1 = decoder1(encoder(batch))
    w2 = decoder2(encoder(w1))   ← decoder2 sobre reconstrucción de AE1
    \"\"\"
    model.eval()
    n = len(data_norm_arr)
    w = config.window_size
    n_windows = n - w

    acc_w1 = np.zeros(n, dtype=np.float64)
    acc_w2 = np.zeros(n, dtype=np.float64)
    cnt    = np.zeros(n, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, n_windows, config.batch_size):
            end     = min(i + config.batch_size, n_windows)
            idx_mat = np.arange(w)[None, :] + np.arange(i, end)[:, None]
            batch   = torch.FloatTensor(data_norm_arr[idx_mat].astype(np.float32)).to(device)

            z  = model.encoder(batch)
            w1 = model.decoder1(z)
            w2 = model.decoder2(model.encoder(w1))   # USAD testing step

            w1_np = w1.cpu().numpy()
            w2_np = w2.cpu().numpy()

            for k, j in enumerate(range(i, end)):
                acc_w1[j:j + w] += w1_np[k]
                acc_w2[j:j + w] += w2_np[k]
                cnt[j:j + w]    += 1

    cnt[cnt == 0] = 1
    t_pred_d1 = (acc_w1 / cnt) * std + media
    t_pred_d2 = (acc_w2 / cnt) * std + media
    return t_pred_d1, t_pred_d2
"""

# Locate end of reconstruir_serie_usad (find the return statement + newline, then add dual after)
anchor = "    return t_pred\n\n\ndef dataset_error_usad("
replacement = "    return t_pred\n" + dual_recon + "\ndef dataset_error_usad("
if anchor in src13:
    src13 = src13.replace(anchor, replacement)
    print("[OK] 5b — reconstruir_serie_usad_dual insertada")
else:
    print("[WARN] 5b — anchor no encontrado, intentando alternativa")
    # Try a more flexible approach
    alt_anchor = "    return t_pred\n\n\ndef dataset_error_usad("
    if alt_anchor in src13:
        src13 = src13.replace(alt_anchor, "    return t_pred\n" + dual_recon + "\ndef dataset_error_usad(")
        print("[OK] 5b alt — reconstruir_serie_usad_dual insertada")
    else:
        # Find what's around "return t_pred"
        idx = src13.find("return t_pred")
        print(f"  context around 'return t_pred': {repr(src13[idx:idx+80])}")

# 5c — agregar dataset_error_usad_dual después de dataset_error_usad
dual_dataset = """

def dataset_error_usad_dual(df_original, t_pred_d1, t_pred_d2, alpha=0.5, beta=0.5):
    \"\"\"DataFrame con score combinado: error = α·MSE_d1 + β·MSE_d2.

    t_predict = t_predict_d1 para compatibilidad con plot_series().
    \"\"\"
    df = pd.DataFrame({
        't':            df_original['t'].values,
        'flag':         df_original['flag'].values,
        't_predict':    t_pred_d1,
        't_predict_d2': t_pred_d2,
    }, index=df_original.index)
    df['error_d1'] = (df['t'] - df['t_predict']) ** 2
    df['error_d2'] = (df['t'] - df['t_predict_d2']) ** 2
    df['error']    = alpha * df['error_d1'] + beta * df['error_d2']
    return df
"""

# Insert after dataset_error_usad function (before precision_recall_curve_plot)
anchor2 = "    return df\n\n\ndef precision_recall_curve_plot("
replacement2 = "    return df\n" + dual_dataset + "\ndef precision_recall_curve_plot("
if anchor2 in src13:
    src13 = src13.replace(anchor2, replacement2)
    print("[OK] 5c — dataset_error_usad_dual insertada")
else:
    print("[WARN] 5c — anchor no encontrado")
    idx = src13.find("def precision_recall_curve_plot(")
    print(f"  context before PR curve fn: {repr(src13[max(0,idx-100):idx])}")

# 5d — actualizar precision_recall_curve_plot (F1 → balanced_accuracy)
new_pr_fn = """def precision_recall_curve_plot(df_concat):
    \"\"\"PR curve + argmax(balanced_accuracy) — v4: penaliza FP y FN por igual.\"\"\"
    precision_arr, recall_arr, thresholds = precision_recall_curve(
        df_concat['flag'], df_concat['error']
    )
    ba_scores = [
        balanced_accuracy_score(df_concat['flag'], (df_concat['error'] >= thr).astype(int))
        for thr in thresholds
    ]
    best_idx   = int(np.argmax(ba_scores))
    umbral_opt = float(thresholds[best_idx])

    plt.figure(figsize=(6, 4))
    plt.plot(recall_arr, precision_arr, linewidth=1)
    plt.scatter(
        recall_arr[best_idx], precision_arr[best_idx], color='red', zorder=5,
        label=f'BA óptimo: {ba_scores[best_idx]:.4f}\\nUmbral θ = {umbral_opt:.6f}',
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall — Validación (v4: BA óptimo)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f'Umbral óptimo (Balanced Acc): {umbral_opt:.6f}')
    return umbral_opt
"""

# Find and replace the existing precision_recall_curve_plot function
pr_start = src13.find("def precision_recall_curve_plot(")
if pr_start == -1:
    print("[WARN] 5d — precision_recall_curve_plot no encontrada")
else:
    # Find the end: next top-level def or end of string
    rest = src13[pr_start:]
    next_def = rest.find("\ndef ", 5)
    if next_def == -1:
        next_def = len(rest)
    old_pr_fn = rest[:next_def + 1]  # include leading \n of next def
    src13 = src13.replace(old_pr_fn, new_pr_fn + "\n")
    print("[OK] 5d — precision_recall_curve_plot actualizada")

# 5e — actualizar metics() con balanced_accuracy
old_metics = """def metics(df_concat):
    y_true = df_concat['flag'].values
    y_pred = df_concat['flag_pred'].values
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}')
    print()
    cm = sk_confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)"""

new_metics = """def metics(df_concat):
    y_true = df_concat['flag'].values
    y_pred = df_concat['flag_pred'].values
    print(f'Accuracy:     {accuracy_score(y_true, y_pred):.4f}')
    print(f'Balanced Acc: {balanced_accuracy_score(y_true, y_pred):.4f}')   # v4
    print(f'Precision:    {precision_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'Recall:       {recall_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'F1-Score:     {f1_score(y_true, y_pred, zero_division=0):.4f}')
    print()
    cm = sk_confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)"""

if old_metics in src13:
    src13 = src13.replace(old_metics, new_metics)
    print("[OK] 5e — metics() actualizada con balanced_accuracy")
else:
    # Try a more flexible replacement
    metics_start = src13.find("def metics(df_concat):")
    if metics_start != -1:
        metics_rest = src13[metics_start:]
        metics_end_rel = metics_rest.find("\ndef ", 5)
        if metics_end_rel == -1:
            metics_end_rel = len(metics_rest)
        old_metics_fn = metics_rest[:metics_end_rel]
        print(f"  found metics: {repr(old_metics_fn[:200])}")
        src13 = src13.replace(old_metics_fn, new_metics)
        print("[OK] 5e alt — metics() actualizada")
    else:
        print("[WARN] 5e — metics() no encontrada")

cells[13]["source"] = [src13]
print("[OK] celda 13 — helper functions completo")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 6 — Insertar diagnóstico TL (2 nuevas celdas después de índice 15)
#           orig [13] → [15] (después de las 2 inserciones de FiltFilt)
#           Insertar DESPUÉS de celda 15 (TL adapter code)
# ─────────────────────────────────────────────────────────────────────────────
tl_diag_md = md_cell(
    "## 4b. Diagnóstico Transfer Learning — Arquitectura y verificación de pesos\n"
)

tl_diag_code_src = """\
# === Diagnóstico Transfer Learning ===
checkpoint_tl = torch.load(config.pretrained_path, map_location='cpu')

print('=== Comparación de arquitecturas ===')
enc_w_orig = checkpoint_tl['encoder']['linear1.weight']  # [306, 612]
print(f'Modelo origen (SWaT):   linear1.weight {list(enc_w_orig.shape)}  (612D input)')
print(f'Modelo destino (68):    linear1.weight {list(model.encoder.linear1.weight.shape)}  ({config.w_size_new}D input)')
print(f'z_size nuevo: {config.z_size_new}  (bottleneck)')

# Sub-matriz extraída
out_rows = config.w_size_new // 2
w_sub_orig = enc_w_orig[:out_rows, :config.w_size_new]
w_new      = model.encoder.linear1.weight.detach()

print('\\n=== Estadísticas de pesos (encoder.linear1) ===')
print(f'Sub-matriz origen {list(w_sub_orig.shape)}: mean={w_sub_orig.mean():.4f}  std={w_sub_orig.std():.4f}  norm={w_sub_orig.norm():.2f}')
print(f'Pesos transferidos {list(w_new.shape)}: mean={w_new.mean():.4f}  std={w_new.std():.4f}  norm={w_new.norm():.2f}')

# Verificar coherencia
match = torch.allclose(w_new, w_sub_orig)
print(f'\\nlinear1.weight es submatriz exacta del checkpoint: {"OK" if match else "ADVERTENCIA: no coincide"}')
print('\\nNota: solo encoder.linear1 se transfiere. linear2, linear3 y todos los decoders son aleatorios.')

# Params totales vs origen
total_new  = sum(p.numel() for p in model.parameters())
total_orig = sum(v.numel() for v in checkpoint_tl['encoder'].values()) + \\
             sum(v.numel() for v in checkpoint_tl.get('decoder1', {}).values())
print(f'\\nParámetros modelo nuevo:  {total_new:,}')
print(f'Porcentaje transferido:   {w_new.numel() / total_new * 100:.1f}% del modelo nuevo')
"""
tl_diag_code = code_cell([tl_diag_code_src])

# Insert after index 15 (TL adapter code, was orig 13)
cells.insert(16, tl_diag_code)
cells.insert(16, tl_diag_md)
print("[OK] celdas 16-17 — diagnóstico TL insertadas")

# After 2nd insertion batch: orig [14]→[18], orig [15]→[19], orig [18]→[22], orig [22]→[26], orig [34]→[38]

# ─────────────────────────────────────────────────────────────────────────────
# Paso 7 — Inferencia val (ahora índice 22, era orig 18)
# ─────────────────────────────────────────────────────────────────────────────
new_val_src = """\
# ── Reconstrucción dual val (α·MSE_d1 + β·MSE_d2) ────────────────────────────
print("Reconstruyendo serie de validación (dual decoder)...")
t_pred_d1_val, t_pred_d2_val = reconstruir_serie_usad_dual(
    model, val_norm, config, media, std, device
)
df_concat_val = dataset_error_usad_dual(
    data_val, t_pred_d1_val, t_pred_d2_val, config.alpha, config.beta
)

print(f"error_d1: {df_concat_val['error_d1'].mean():.4f} | "
      f"error_d2: {df_concat_val['error_d2'].mean():.4f} | "
      f"combined: {df_concat_val['error'].mean():.4f}")
"""
cells[22]["source"] = [new_val_src]
print("[OK] celda 22 — inferencia val dual (era orig 18)")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 8 — Inferencia test (ahora índice 26, era orig 22)
# ─────────────────────────────────────────────────────────────────────────────
new_test_src = """\
# ── Reconstrucción dual test ──────────────────────────────────────────────────
print("Reconstruyendo serie de test (dual decoder)...")
t_pred_d1_test, t_pred_d2_test = reconstruir_serie_usad_dual(
    model, test_norm, config, media, std, device
)
df_concat_test = dataset_error_usad_dual(
    data_test, t_pred_d1_test, t_pred_d2_test, config.alpha, config.beta
)

df_concat_test['flag_pred'] = (df_concat_test['error'] >= umbral).astype(int)

print(f"error combined — test: {df_concat_test['error'].mean():.4f}")
print(f"Anomalías detectadas en Test: {df_concat_test['flag_pred'].sum()} / {len(df_concat_test)}")
"""
cells[26]["source"] = [new_test_src]
print("[OK] celda 26 — inferencia test dual (era orig 22)")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 9 — save_path (ahora índice 38, era orig 34)
# ─────────────────────────────────────────────────────────────────────────────
src38 = "".join(cells[38]["source"])
src38 = src38.replace(
    'save_path = "modelos/usad/model_siata_68_transfer_v2.pth"',
    'save_path = "modelos/usad/model_siata_68_transfer_v4.pth"',
)
cells[38]["source"] = [src38]
print("[OK] celda 38 — save_path v2→v4 (era orig 34)")

# ─────────────────────────────────────────────────────────────────────────────
# Guardar
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"] = cells
with open(NB_PATH, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nTotal celdas final: {len(cells)} (esperado: 39)")
print("Notebook guardado.")
