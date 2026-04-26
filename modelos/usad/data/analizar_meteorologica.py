#!/usr/bin/env python3
"""
Analisis y Limpieza de Datos Meteorologicos SIATA
Estaciones: 203 (UNAN), 201 (Torre SIATA), 478 (Fiscalia General), 68 (Jardin Botanico)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "meteorologica")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

ESTACIONES = {
    203: {"nombre": "UNAN", "archivo": "Estacion_meteorologica_203_2012-10-30_2025-10-31.csv"},
    201: {"nombre": "Torre SIATA", "archivo": "Estacion_meteorologica_201_2012-12-28_2025-10-31.csv"},
    478: {"nombre": "Fiscalia General", "archivo": "Estacion_meteorologica_478_2021-01-01_2025-10-31.csv"},
    68:  {"nombre": "Jardin Botanico", "archivo": "Estacion_meteorologica_68_2013-02-01_2025-10-31.csv"},
}

COLUMNAS_ESPERADAS = ["codigo", "fecha_hora", "h", "t", "pr", "vv", "vv_max", "dv", "dv_max", "p", "calidad"]

COLUMNAS_METEOROLOGICAS = ["h", "t", "pr", "vv", "vv_max", "dv", "dv_max", "p"]

VALOR_FALTANTE = -999.0

# Rangos validos para validacion (Medellin, ~1500 msnm)
RANGOS_VALIDOS = {
    "h":      (0, 100),       # Humedad relativa %
    "t":      (5, 45),        # Temperatura C
    "pr":     (800, 900),     # Presion atmosferica hPa
    "vv":     (0, 50),        # Velocidad viento promedio m/s
    "vv_max": (0, 80),        # Velocidad viento maxima m/s
    "dv":     (0, 360),       # Direccion viento promedio grados
    "dv_max": (0, 360),       # Direccion viento maximo grados
    "p":      (0, 200),       # Precipitacion mm
}

# Extraccion de temperatura - rango de anios a filtrar
FILTRO_ANIO_INICIO = 2020
FILTRO_ANIO_FIN = 2025

COLUMNAS_SALIDA_METEOROLOGICA = [
    "codigo", "estacion_nombre", "fecha_hora",
    "t", "calidad", "calidad_dudosa", "temperatura_dudosa",
]

INDICES_CALIDAD = {
    1: "Calidad confiable del dato en tiempo real",
    2: "Calidad confiable del dato no obtenido en tiempo real",
    151: "Calidad dudosa en datos de todas las variables (4+)",
    1511: "Calidad dudosa en dato de precipitacion en tiempo real",
    153: "Calidad dudosa en dato de temperatura en tiempo real",
    154: "Calidad dudosa en dato de humedad relativa en tiempo real",
    155: "Calidad dudosa en dato de presion atmosferica en tiempo real",
    156: "Calidad dudosa en dato de magnitud de viento en tiempo real",
    1561: "Calidad dudosa en dato de magnitud de viento promedio en tiempo real",
    1562: "Calidad dudosa en dato de magnitud de viento maximo en tiempo real",
    157: "Calidad dudosa en dato de direccion del viento en tiempo real",
    1571: "Calidad dudosa en dato de direccion del viento promedio en tiempo real",
    1572: "Calidad dudosa en dato de direccion del viento maximo en tiempo real",
}


def cargar_estacion(codigo, info):
    """Carga un CSV de estacion y valida columnas."""
    ruta = os.path.join(DATA_DIR, info["archivo"])
    print(f"  Cargando estacion {codigo} ({info['nombre']}) desde {info['archivo']}...")

    df = pd.read_csv(ruta, dtype={"calidad": "Int64"}, engine="python", encoding="utf-8", on_bad_lines="skip")

    if list(df.columns) != COLUMNAS_ESPERADAS:
        print(f"    ADVERTENCIA: Columnas inesperadas en estacion {codigo}: {list(df.columns)}")

    df["estacion_nombre"] = info["nombre"]
    print(f"    {len(df):,} registros cargados.")
    return df


def detectar_faltantes(df, estacion_nombre):
    """Detecta valores -999 y NaN por columna."""
    resultados = {}
    for col in COLUMNAS_METEOROLOGICAS:
        n_minus999 = (df[col] == VALOR_FALTANTE).sum()
        n_nan = df[col].isna().sum()
        resultados[col] = {"faltantes_-999": int(n_minus999), "NaN_nativos": int(n_nan)}
    return resultados


def detectar_fuera_de_rango(df):
    """Detecta valores fuera de rango fisico valido (excluyendo -999)."""
    resultados = {}
    for col, (vmin, vmax) in RANGOS_VALIDOS.items():
        mask_valido = (df[col] != VALOR_FALTANTE) & df[col].notna()
        valores_validos = df.loc[mask_valido, col]
        fuera = ((valores_validos < vmin) | (valores_validos > vmax)).sum()
        resultados[col] = int(fuera)
    return resultados


def detectar_inconsistencias_viento(df):
    """Detecta filas donde vv_max < vv (excluyendo -999)."""
    mask = (
        (df["vv"] != VALOR_FALTANTE) & (df["vv_max"] != VALOR_FALTANTE) &
        df["vv"].notna() & df["vv_max"].notna() &
        (df["vv_max"] < df["vv"])
    )
    return int(mask.sum())


def detectar_gaps_temporales(df, umbral_minutos=5):
    """Detecta gaps mayores al umbral en la serie temporal."""
    df_sorted = df.sort_values("fecha_hora")
    diff = df_sorted["fecha_hora"].diff()
    umbral = pd.Timedelta(minutes=umbral_minutos)
    gaps = diff[diff > umbral]
    return gaps


def analizar_calidad(df):
    """Analiza distribucion de indices de calidad."""
    return df["calidad"].value_counts().sort_index().to_dict()


def generar_reporte(estaciones_data, reporte_path):
    """Genera el archivo TXT con el reporte de datos problematicos."""
    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE DATOS PROBLEMATICOS - ESTACIONES METEOROLOGICAS SIATA\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for codigo, data in estaciones_data.items():
            nombre = ESTACIONES[codigo]["nombre"]
            df = data["df"]
            f.write("-" * 80 + "\n")
            f.write(f"ESTACION {codigo} - {nombre}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total de registros: {len(df):,}\n")
            f.write(f"Periodo: {df['fecha_hora'].min()} a {df['fecha_hora'].max()}\n\n")

            # Faltantes
            f.write(">>> DATOS FALTANTES (-999 y NaN) por variable:\n")
            faltantes = data["faltantes"]
            for col, vals in faltantes.items():
                total_falt = vals["faltantes_-999"] + vals["NaN_nativos"]
                pct = total_falt / len(df) * 100
                f.write(f"  {col:8s}: {vals['faltantes_-999']:>10,} (-999) + {vals['NaN_nativos']:>8,} (NaN) = {total_falt:>10,} ({pct:.2f}%)\n")

            filas_con_algun_faltante = 0
            filas_todo_faltante = 0
            for _, row in df.iterrows() if len(df) < 100 else df.head(0).iterrows():
                pass
            # Calcular mas eficientemente
            mask_any_999 = (df[COLUMNAS_METEOROLOGICAS] == VALOR_FALTANTE).any(axis=1)
            mask_all_999 = (df[COLUMNAS_METEOROLOGICAS] == VALOR_FALTANTE).all(axis=1)
            filas_con_algun_faltante = int(mask_any_999.sum())
            filas_todo_faltante = int(mask_all_999.sum())
            f.write(f"\n  Filas con al menos un campo -999: {filas_con_algun_faltante:,} ({filas_con_algun_faltante/len(df)*100:.2f}%)\n")
            f.write(f"  Filas con TODOS los campos -999:  {filas_todo_faltante:,} ({filas_todo_faltante/len(df)*100:.2f}%)\n\n")

            # Fuera de rango
            f.write(">>> DATOS FUERA DE RANGO VALIDO (excluyendo -999):\n")
            fuera = data["fuera_rango"]
            for col, n in fuera.items():
                rmin, rmax = RANGOS_VALIDOS[col]
                f.write(f"  {col:8s}: {n:>10,} valores fuera de [{rmin}, {rmax}]\n")

            # Inconsistencia viento
            n_viento = data["inconsistencias_viento"]
            f.write(f"\n  Inconsistencias vv_max < vv: {n_viento:,}\n\n")

            # Calidad
            f.write(">>> DISTRIBUCION DE INDICES DE CALIDAD:\n")
            calidad_dist = data["calidad_dist"]
            for idx, count in sorted(calidad_dist.items()):
                desc = INDICES_CALIDAD.get(idx, "Indice acumulativo (combinacion)")
                pct = count / len(df) * 100
                f.write(f"  {idx:>10}: {count:>12,} registros ({pct:.2f}%) - {desc}\n")

            n_dudoso = sum(count for idx, count in calidad_dist.items() if idx not in (1, 2))
            f.write(f"\n  Total registros con calidad dudosa (indice != 1,2): {n_dudoso:,} ({n_dudoso/len(df)*100:.2f}%)\n\n")

            # Gaps temporales
            gaps = data["gaps"]
            f.write(f">>> GAPS TEMPORALES (> 5 minutos): {len(gaps):,} detectados\n")
            if len(gaps) > 0:
                f.write("  Top 10 gaps mas grandes:\n")
                for i, (idx, gap) in enumerate(gaps.nlargest(10).items()):
                    fecha = df.loc[idx, "fecha_hora"]
                    f.write(f"    {i+1}. {gap} antes de {fecha}\n")

            # Ejemplos de filas problematicas
            f.write("\n>>> EJEMPLOS DE FILAS PROBLEMATICAS (primeras 10 con -999):\n")
            filas_999 = df[mask_any_999].head(10)
            if len(filas_999) > 0:
                f.write(filas_999.to_string(index=False) + "\n")
            else:
                f.write("  Ninguna\n")

            # Ejemplos fuera de rango
            f.write("\n>>> EJEMPLOS DE FILAS CON VALORES FUERA DE RANGO (primeras 10):\n")
            mask_fuera = pd.Series(False, index=df.index)
            for col, (vmin, vmax) in RANGOS_VALIDOS.items():
                mask_col = (df[col] != VALOR_FALTANTE) & df[col].notna() & ((df[col] < vmin) | (df[col] > vmax))
                mask_fuera = mask_fuera | mask_col
            filas_fuera = df[mask_fuera].head(10)
            if len(filas_fuera) > 0:
                f.write(filas_fuera.to_string(index=False) + "\n")
            else:
                f.write("  Ninguna\n")

            f.write("\n\n")

        # Resumen global
        f.write("=" * 80 + "\n")
        f.write("RESUMEN GLOBAL\n")
        f.write("=" * 80 + "\n")
        total_registros = sum(len(d["df"]) for d in estaciones_data.values())
        total_faltantes = sum(
            sum(v["faltantes_-999"] + v["NaN_nativos"] for v in d["faltantes"].values())
            for d in estaciones_data.values()
        )
        f.write(f"Total registros procesados: {total_registros:,}\n")
        f.write(f"Total celdas con datos faltantes: {total_faltantes:,}\n")

    print(f"  Reporte guardado en: {reporte_path}")


def generar_diccionario(output_path):
    """Genera el archivo TXT con el diccionario de datos."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DICCIONARIO DE DATOS - ESTACIONES METEOROLOGICAS SIATA\n")
        f.write("Red Meteorologica del Valle de Aburra\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("FUENTE: Sistema de Alerta Temprana de Medellin y el Valle de Aburra (SIATA)\n")
        f.write("        Proyecto del Area Metropolitana del Valle de Aburra\n")
        f.write("        www.siata.gov.co\n\n")

        f.write("-" * 80 + "\n")
        f.write("ESTACIONES INCLUIDAS\n")
        f.write("-" * 80 + "\n")
        for codigo, info in sorted(ESTACIONES.items()):
            f.write(f"  Codigo {codigo:>4d} - {info['nombre']:<20s} | Archivo: {info['archivo']}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("DESCRIPCION DE CAMPOS (COLUMNAS)\n")
        f.write("-" * 80 + "\n\n")

        campos = [
            ("codigo", "entero", "-",
             "Identificador numerico unico de la estacion meteorologica. "
             "Valores posibles en este dataset: 68, 201, 203, 478."),

            ("fecha_hora", "datetime", "YYYY-MM-DD HH:MM:SS",
             "Marca temporal del registro. La red meteorologica registra un dato "
             "cada minuto. La frecuencia esperada entre registros consecutivos es "
             "de 1 minuto. Las estaciones mas antiguas (68, 201, 203) tienen datos "
             "desde 2012-2013; la estacion 478 desde 2021."),

            ("h", "float", "% (porcentaje)",
             "Humedad relativa del aire. Proporcion de vapor de agua presente en "
             "el aire respecto al maximo que podria contener a la misma temperatura. "
             "Rango fisico valido: [0, 100]. Valores fuera de este rango son anomalos."),

            ("t", "float", "grados Celsius (C)",
             "Temperatura ambiente medida por la estacion. Rango esperado para "
             "Medellin (ubicado a ~1500 metros sobre el nivel del mar): [5, 45]. "
             "El clima de Medellin es subtropical de montana con temperaturas "
             "tipicas entre 15 y 30 C."),

            ("pr", "float", "hectopascales (hPa)",
             "Presion atmosferica. Dado que Medellin se encuentra a ~1500 msnm, "
             "la presion atmosferica es menor que al nivel del mar (~1013 hPa). "
             "Rango esperado: [800, 900] hPa."),

            ("vv", "float", "metros por segundo (m/s)",
             "Magnitud de la velocidad promedio del viento durante el minuto de "
             "medicion. Rango valido: [0, 50] m/s. Valores negativos son fisicamente "
             "imposibles."),

            ("vv_max", "float", "metros por segundo (m/s)",
             "Magnitud de la velocidad maxima del viento (rafaga) registrada durante "
             "el minuto de medicion. Rango valido: [0, 80] m/s. Restriccion: debe "
             "ser mayor o igual a vv (velocidad promedio). Si vv_max < vv, el "
             "registro es inconsistente."),

            ("dv", "float", "grados (0-360)",
             "Direccion promedio del viento durante el minuto de medicion. Se mide "
             "en grados desde el norte geografico en sentido horario: 0/360 = Norte, "
             "90 = Este, 180 = Sur, 270 = Oeste. Rango valido: [0, 360]."),

            ("dv_max", "float", "grados (0-360)",
             "Direccion del viento en el momento de la rafaga maxima (vv_max). "
             "Misma convencion que dv. Rango valido: [0, 360]."),

            ("p", "float", "milimetros (mm)",
             "Precipitacion acumulada durante el minuto de medicion. "
             "Rango valido: >= 0. Valores negativos son fisicamente imposibles. "
             "Las estaciones meteorologicas usan unicamente el pluviometro 1."),

            ("calidad", "entero", "-",
             "Indice de calidad acumulativo del registro segun el sistema SIATA. "
             "Los indices son acumulativos:\n"
             "    1    = Calidad confiable del dato en tiempo real\n"
             "    2    = Calidad confiable del dato no obtenido en tiempo real\n"
             "    151  = Calidad dudosa en 4 o mas variables\n"
             "    1511 = Calidad dudosa en precipitacion\n"
             "    153  = Calidad dudosa en temperatura\n"
             "    154  = Calidad dudosa en humedad relativa\n"
             "    155  = Calidad dudosa en presion atmosferica\n"
             "    156  = Calidad dudosa en magnitud de viento (promedio + max)\n"
             "    1561 = Calidad dudosa en magnitud de viento promedio\n"
             "    1562 = Calidad dudosa en magnitud de viento maximo\n"
             "    157  = Calidad dudosa en direccion del viento (promedio + max)\n"
             "    1571 = Calidad dudosa en direccion del viento promedio\n"
             "    1572 = Calidad dudosa en direccion del viento maximo\n"
             "  El primer digito indica procedencia: 1=tiempo real, 2=importacion.\n"
             "  El segundo digito '5' indica calidad dudosa.\n"
             "  Los digitos restantes indican las variables afectadas.\n"
             "  Combinaciones: ej. 1534 = temp + humedad dudosas, 1515 = precipitacion + presion dudosas."),

            ("calidad_dudosa", "booleano (True/False)", "-",
             "Campo agregado durante la limpieza. True si el indice de calidad "
             "original es distinto de 1 y 2, indicando que al menos una variable "
             "tiene calidad dudosa en ese registro."),

            ("estacion_nombre", "texto", "-",
             "Nombre legible de la estacion meteorologica. Campo agregado durante "
             "la limpieza para facilitar la identificacion. Valores: 'UNAN', "
             "'Torre SIATA', 'Fiscalia General', 'Jardin Botanico'."),
        ]

        for nombre, tipo, unidad, descripcion in campos:
            f.write(f"Campo: {nombre}\n")
            f.write(f"  Tipo: {tipo}\n")
            f.write(f"  Unidad: {unidad}\n")
            f.write(f"  Descripcion: {descripcion}\n\n")

        f.write("-" * 80 + "\n")
        f.write("VALOR ESPECIAL: -999\n")
        f.write("-" * 80 + "\n")
        f.write("En los datos originales del SIATA, el valor -999 en cualquier campo\n")
        f.write("numerico indica un DATO FALTANTE. En los CSV limpios generados por\n")
        f.write("este proceso, los valores -999 fueron reemplazados por NaN (Not a Number),\n")
        f.write("y las filas donde TODAS las variables meteorologicas eran -999 fueron\n")
        f.write("eliminadas.\n\n")

        f.write("-" * 80 + "\n")
        f.write("FRECUENCIA Y RESOLUCION TEMPORAL\n")
        f.write("-" * 80 + "\n")
        f.write("La red meteorologica registra un dato cada minuto. Inicialmente\n")
        f.write("(2010-2012) la resolucion era de 5 minutos. Pueden existir gaps\n")
        f.write("temporales por fallas en la transmision o el equipo.\n\n")

        f.write("-" * 80 + "\n")
        f.write("NOTAS IMPORTANTES\n")
        f.write("-" * 80 + "\n")
        f.write("- Los datos provienen del SIATA y deben ser procesados por el solicitante\n")
        f.write("  para ventanas de tiempo distintas a 1 minuto.\n")
        f.write("- La ubicacion de las estaciones puede consultarse en el Geoportal del SIATA.\n")
        f.write("- Los datos de calidad dudosa fueron conservados con un flag (calidad_dudosa=True)\n")
        f.write("  para que el modelo de ML/LLM pueda aprender de patrones completos.\n")

    print(f"  Diccionario guardado en: {output_path}")


def es_temperatura_dudosa(calidad):
    """
    Detecta si el indice de calidad implica temperatura dudosa.
    Segun el documento SIATA, los indices son acumulativos:
      - Primer digito: 1=tiempo real, 2=importacion
      - Segundo digito '5': indica calidad dudosa
      - Digito '3' tras el '5': temperatura es una de las variables dudosas
      - Indices 151 / 251: 4 o mas variables dudosas (temperatura incluida)
    Ejemplos: 153, 1534, 1535, 15311, 1536, 153... cualquier combinacion con '3'.
    """
    try:
        s = str(int(calidad))
    except (ValueError, TypeError):
        return False
    # Todas las variables dudosas => temperatura tambien lo es
    if s in ("151", "251"):
        return True
    # Segundo digito '5' y '3' presente en el resto => temperatura dudosa
    if len(s) >= 3 and s[1] == "5" and "3" in s[2:]:
        return True
    return False


def extraer_meteorologica_rango(df, nombre, anio_inicio, anio_fin):
    """
    Extrae todos los registros del rango de anios con todas las variables meteorologicas
    disponibles (h, t, pr, vv, vv_max, dv, dv_max, p) mas flags de calidad.
    Nota: la variable Radiacion (W/m2) no esta presente en los archivos descargados del SIATA.
    """
    df_out = df.copy()

    # Filtro por rango de anios (inclusive en ambos extremos)
    df_out = df_out[
        (df_out["fecha_hora"].dt.year >= anio_inicio) &
        (df_out["fecha_hora"].dt.year <= anio_fin)
    ].copy()

    # Flag global: cualquier variable con calidad dudosa
    df_out["calidad_dudosa"] = ~df_out["calidad"].isin([1, 2])
    # Flag especifico: temperatura dudosa (util para analisis de temperatura)
    df_out["temperatura_dudosa"] = df_out["calidad"].apply(es_temperatura_dudosa)

    # Reemplazar -999 por NaN en todas las variables meteorologicas
    for col in COLUMNAS_METEOROLOGICAS:
        df_out[col] = df_out[col].replace(VALOR_FALTANTE, np.nan)

    df_out = df_out[COLUMNAS_SALIDA_METEOROLOGICA].copy()

    n_dudosa_t = df_out["temperatura_dudosa"].sum()
    n_dudosa_any = df_out["calidad_dudosa"].sum()
    print(
        f"  Estacion {nombre}: {len(df_out):,} registros {anio_inicio}-{anio_fin} | "
        f"temperatura dudosa: {n_dudosa_t:,} | "
        f"cualquier variable dudosa: {n_dudosa_any:,}"
    )
    return df_out


def generar_csv_meteorologica_rango(estaciones_data, output_dir, anio_inicio, anio_fin):
    """
    Combina y guarda CSV con todas las variables meteorologicas disponibles
    (h, t, pr, vv, vv_max, dv, dv_max, p) para el rango de anios indicado.
    Variables incluidas segun PDF SIATA: precipitacion, temperatura, presion atmosferica,
    humedad relativa, velocidad promedio/maxima del viento, direccion promedio/maxima del viento.
    Radiacion (W/m2) no esta disponible en los archivos descargados.
    """
    fragmentos = []
    for codigo, data in estaciones_data.items():
        nombre = ESTACIONES[codigo]["nombre"]
        df_met = extraer_meteorologica_rango(data["df"], nombre, anio_inicio, anio_fin)
        fragmentos.append(df_met)

    df_final = pd.concat(fragmentos, ignore_index=True)
    df_final = df_final.sort_values(["fecha_hora", "codigo"]).reset_index(drop=True)

    ruta_salida = os.path.join(output_dir, f"meteorologica_estaciones_{anio_inicio}_{anio_fin}.csv")
    df_final.to_csv(ruta_salida, index=False)

    n_dudosa_t = df_final["temperatura_dudosa"].sum()
    n_dudosa_any = df_final["calidad_dudosa"].sum()
    print(f"  CSV guardado en: {ruta_salida}")
    print(f"  Total registros: {len(df_final):,}")
    print(f"  Variables: h (humedad), t (temperatura), pr (presion), vv/vv_max (viento), dv/dv_max (dir. viento), p (precipitacion)")
    print(f"  Con temperatura dudosa:          {n_dudosa_t:,} ({n_dudosa_t / len(df_final) * 100:.2f}%)")
    print(f"  Con cualquier variable dudosa:   {n_dudosa_any:,} ({n_dudosa_any / len(df_final) * 100:.2f}%)")
    return ruta_salida


def limpiar_y_guardar(df, codigo, nombre, output_dir):
    """Limpia datos y guarda CSV."""
    df_clean = df.copy()

    # Reemplazar -999 por NaN en columnas meteorologicas
    for col in COLUMNAS_METEOROLOGICAS:
        df_clean[col] = df_clean[col].replace(VALOR_FALTANTE, np.nan)

    # Eliminar filas donde TODAS las variables meteorologicas son NaN
    mask_all_nan = df_clean[COLUMNAS_METEOROLOGICAS].isna().all(axis=1)
    filas_eliminadas = int(mask_all_nan.sum())
    df_clean = df_clean[~mask_all_nan]

    # Agregar flag de calidad dudosa
    df_clean["calidad_dudosa"] = ~df_clean["calidad"].isin([1, 2])

    # Nombre limpio para archivo
    nombre_archivo = nombre.replace(" ", "")
    ruta_salida = os.path.join(output_dir, f"estacion_{codigo}_{nombre_archivo}_limpio.csv")
    df_clean.to_csv(ruta_salida, index=False)

    print(f"  Estacion {codigo} ({nombre}): {len(df_clean):,} registros guardados ({filas_eliminadas:,} eliminados por ser completamente faltantes)")
    return ruta_salida


def main():
    print("=" * 60)
    print("ANALISIS Y LIMPIEZA DE DATOS METEOROLOGICOS SIATA")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Paso 1: Cargar datos
    print("\n[PASO 1] Cargando datos de las 4 estaciones...")
    estaciones_data = {}
    for codigo, info in ESTACIONES.items():
        df = cargar_estacion(codigo, info)
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        estaciones_data[codigo] = {"df": df}

    # Paso 2: Detectar faltantes
    print("\n[PASO 2] Detectando datos faltantes (-999 y NaN)...")
    for codigo, data in estaciones_data.items():
        data["faltantes"] = detectar_faltantes(data["df"], ESTACIONES[codigo]["nombre"])
        print(f"  Estacion {codigo}: analisis de faltantes completado")

    # Paso 3: Detectar anomalias
    print("\n[PASO 3] Detectando datos fuera de rango y anomalias...")
    for codigo, data in estaciones_data.items():
        data["fuera_rango"] = detectar_fuera_de_rango(data["df"])
        data["inconsistencias_viento"] = detectar_inconsistencias_viento(data["df"])
        data["calidad_dist"] = analizar_calidad(data["df"])
        data["gaps"] = detectar_gaps_temporales(data["df"])
        print(f"  Estacion {codigo}: analisis de anomalias completado")

    # Paso 4: Generar reporte
    print("\n[PASO 4] Generando reporte de datos problematicos...")
    reporte_path = os.path.join(OUTPUT_DIR, "reporte_datos_problematicos.txt")
    generar_reporte(estaciones_data, reporte_path)

    # Paso 5: Generar diccionario
    print("\n[PASO 5] Generando diccionario de datos...")
    diccionario_path = os.path.join(OUTPUT_DIR, "diccionario_datos_meteorologicos.txt")
    generar_diccionario(diccionario_path)

    # Paso 6: Limpiar y guardar
    print("\n[PASO 6] Limpiando datos y generando CSV limpios...")
    for codigo, data in estaciones_data.items():
        nombre = ESTACIONES[codigo]["nombre"]
        limpiar_y_guardar(data["df"], codigo, nombre, OUTPUT_DIR)

    # Paso 7: Extraer todas las variables meteorologicas del rango FILTRO_ANIO_INICIO-FILTRO_ANIO_FIN
    print(f"\n[PASO 7] Extrayendo datos meteorologicos {FILTRO_ANIO_INICIO}-{FILTRO_ANIO_FIN}...")
    print(f"  Variables: h, t, pr, vv, vv_max, dv, dv_max, p (Radiacion no disponible en archivos fuente)")
    generar_csv_meteorologica_rango(estaciones_data, OUTPUT_DIR, FILTRO_ANIO_INICIO, FILTRO_ANIO_FIN)

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print(f"Archivos generados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
