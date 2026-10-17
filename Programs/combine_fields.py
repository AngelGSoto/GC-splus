#!/usr/bin/env python3
"""
combine_fields.py
Combina los CSVs de cada campo en un único catálogo deduplicado,
conservando para cada fuente la medición con mejor SNR promedio.
"""

import pandas as pd
import numpy as np
import glob

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
INPUT_DIR = "Results_Corrected"
OUTPUT_FILE = "Results_Corrected/all_fields_photometry_COMPLETE.csv"
COORD_TOL = 1.0 / 3600.0  # 1 arcsec en grados

# ------------------------------------------------------------
# Cargar todos los archivos individuales
# ------------------------------------------------------------
files = sorted(glob.glob(f"{INPUT_DIR}/CenA*_photometry_COMPLETE.csv"))
if not files:
    print("No se encontraron archivos en", INPUT_DIR)
    exit()

df_list = [pd.read_csv(f) for f in files]
combined = pd.concat(df_list, ignore_index=True)

print(f"Entradas totales: {len(combined)}")

# ------------------------------------------------------------
# Crear columnas temporales de coordenadas redondeadas
# ------------------------------------------------------------
combined['RA_round'] = np.round(combined['RAJ2000'] / COORD_TOL) * COORD_TOL
combined['DEC_round'] = np.round(combined['DEJ2000'] / COORD_TOL) * COORD_TOL

# Columnas de SNR para decidir cuál medición es mejor
snr_cols = [c for c in combined.columns if c.startswith('SNR_')]

# ------------------------------------------------------------
# Agrupar y elegir mejor fila por grupo
# ------------------------------------------------------------
results = []
for (ra, dec), group in combined.groupby(['RA_round', 'DEC_round']):
    if len(group) == 1:
        best = group.iloc[0]
    else:
        avg_snr = group[snr_cols].replace(0, np.nan).mean(axis=1)
        if avg_snr.notna().any():
            best_idx = avg_snr.idxmax()
        else:
            best_idx = group.index[0]
        best = group.loc[best_idx]
    results.append(best)

deduped = pd.DataFrame(results)

# Eliminar columnas temporales
deduped = deduped.drop(columns=['RA_round', 'DEC_round'])

print(f"Después de deduplicar: {len(deduped)} fuentes únicas")

# ------------------------------------------------------------
# Guardar catálogo final
# ------------------------------------------------------------
deduped.to_csv(OUTPUT_FILE, index=False)
print(f"Catálogo final guardado en: {OUTPUT_FILE}")
