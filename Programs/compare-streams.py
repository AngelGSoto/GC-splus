#!/usr/bin/env python3
"""
Cross-match de los GCs (resultados CIGALE) con las subestructuras del halo de NGC 5128.
Basado en Crnojević+2016 Tabla 2 y Crnojević+2019 Tabla 5.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
Z_SOLAR = 0.02
REDUCED_CHI2_THRESHOLD = 2.0
DISTANCE_NGC5128_MPC = 3.8            # distancia a Cen A (Mpc)
SEARCH_RADIUS_KPC = 15.0              # radio de asociación en kpc proyectados

# Factor de conversión: kpc -> grados
KPC_TO_DEG = (1.0 / (DISTANCE_NGC5128_MPC * 1000.0)) * (180.0 / np.pi)

# ============================================================================
# 1. CARGAR DATOS DE GCs (coordenadas + parámetros CIGALE)
# ============================================================================
print("="*70)
print("CARGANDO DATOS DE GCs")
print("="*70)

# Catálogo fotométrico (contiene RA, DEC, T17ID)
phot_cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
print(f"  • Fotometría: {len(phot_cat)} objetos")

# Resultados CIGALE (edades, metalicidades, chi2)
try:
    cigale_table = Table.read('out/results.fits', hdu=1)
    cigale_df = cigale_table.to_pandas()
    print(f"  • CIGALE: {len(cigale_df)} objetos")
except Exception as e:
    print(f"  ❌ Error leyendo out/results.fits: {e}")
    exit()

# Verificar columnas necesarias
required_cols = ['id', 'bayes.stellar.age_m_star', 'bayes.stellar.metallicity',
                 'bayes.stellar.metallicity_err', 'best.reduced_chi_square']
for col in required_cols:
    if col not in cigale_df.columns:
        print(f"  ❌ Columna faltante en CIGALE: {col}")
        exit()

# Calcular [Fe/H] y errores
met = cigale_df['bayes.stellar.metallicity'].values
met_err = cigale_df['bayes.stellar.metallicity_err'].values
valid = (met > 0) & (met_err > 0)
FeH = np.full_like(met, np.nan, dtype=float)
FeH_err = np.full_like(met, np.nan, dtype=float)
if valid.any():
    FeH[valid] = np.log10(met[valid] / Z_SOLAR)
    rel_err = met_err[valid] / met[valid]
    FeH_err[valid] = rel_err / np.log(10)

# Construir DataFrame de GCs
gc_df = pd.DataFrame({
    'ID': cigale_df['id'].astype(str).str.strip(),
    'age_Gyr': cigale_df['bayes.stellar.age_m_star'].values / 1000.0,
    'FeH': FeH,
    'FeH_err': FeH_err,
    'reduced_chi2': cigale_df['best.reduced_chi_square'].values,
})

# Aplicar filtro de chi2
gc_df = gc_df[(gc_df['reduced_chi2'] <= REDUCED_CHI2_THRESHOLD) & (gc_df['FeH'].notna())].copy()
print(f"  • GCs con buena calidad (χ²≤{REDUCED_CHI2_THRESHOLD}): {len(gc_df)}")

# Unir con coordenadas (usando T17ID del catálogo fotométrico)
if 'T17ID' in phot_cat.columns:
    phot_cat.rename(columns={'T17ID': 'ID'}, inplace=True)
elif 'ID' not in phot_cat.columns:
    print("  ❌ El catálogo fotométrico no tiene columna 'T17ID' ni 'ID'")
    exit()

phot_cat['ID'] = phot_cat['ID'].astype(str).str.strip()
gc_df = gc_df.merge(phot_cat[['ID', 'RAJ2000', 'DEJ2000']], on='ID', how='inner')
print(f"  • GCs con coordenadas: {len(gc_df)}")

# ============================================================================
# 2. DEFINIR SUBESTRUCTURAS (Crnojević+2016 Tabla 2)
# ============================================================================
print("\n" + "="*70)
print("DEFINIENDO SUBESTRUCTURAS")
print("="*70)

# =============================================================
# IMPORTANTE: Aquí puedes seleccionar qué subestructuras quieres.
# Para comparar con corrientes estelares, se recomienda usar:
#   - Dw3 core, Dw3 South, Dw3 North, Dw3 SE, Dw3 NW, Second stream
# Las otras (Shells, ESO) son opcionales.
# =============================================================
substructures = [
    # Corrientes principales de Dw3
    ("Dw3 core",      "13:30:21.5", "-42:11:33", -0.61, 0.01),
    ("Dw3 South",     "13:31:02.6", "-42:21:54", -0.86, 0.01),
    ("Dw3 North",     "13:28:27.6", "-41:51:00", -0.80, 0.01),
    # Otras partes de la corriente
    ("Dw3 SE",        "13:32:18.0", "-42:36:54", None, None),
    ("Dw3 NW",        "13:23:57.6", "-41:42:00", None, None),
    ("Second stream", "13:31:54.0", "-41:51:00", None, None),
    # Opcionales: caparazones y otras estructuras
    # ("Shell 1",       "13:26:45.6", "-42:22:30", None, None),
    # ("Shell 2",       "13:23:21.6", "-43:14:06", None, None),
    # ("ESO 324-24",    "13:27:32.4", "-41:33:00", None, None),
    # ("ESO 324-24 S",  "13:27:03.6", "-41:37:30", None, None),
]

stream_names = []
stream_coords = []
stream_feh_lit = []
stream_feh_err_lit = []

for name, ra_str, dec_str, feh, err in substructures:
    c = SkyCoord(ra_str, dec_str, unit=('hourangle', 'deg'))
    stream_names.append(name)
    stream_coords.append(c)
    stream_feh_lit.append(feh)
    stream_feh_err_lit.append(err)
    print(f"  • {name:15}  RA={ra_str}  Dec={dec_str}")

# ============================================================================
# 3. ASIGNAR CADA GC A LA SUBESTRUCTURA MÁS CERCANA (dentro del radio)
# ============================================================================
print(f"\nRadio de búsqueda: {SEARCH_RADIUS_KPC} kpc proyectados = {SEARCH_RADIUS_KPC*KPC_TO_DEG:.3f}°")

# Obtener coordenadas de los GCs
gc_coords = SkyCoord(ra=gc_df['RAJ2000'].values, dec=gc_df['DEJ2000'].values, unit='deg')

assigned = []
dist_kpc = []

for i, gc in enumerate(gc_coords):
    # Calcular separación con cada subestructura (evitando el error de tipo)
    seps_deg = [gc.separation(sc).deg for sc in stream_coords]
    min_sep_deg = np.min(seps_deg)
    if min_sep_deg <= SEARCH_RADIUS_KPC * KPC_TO_DEG:
        idx = np.argmin(seps_deg)
        assigned.append(stream_names[idx])
        dist_kpc.append(min_sep_deg / KPC_TO_DEG)
    else:
        assigned.append("None")
        dist_kpc.append(np.nan)

gc_df['Assigned_Stream'] = assigned
gc_df['Dist_to_Stream_kpc'] = dist_kpc

print(f"  • GCs asignados a alguna subestructura: {(gc_df['Assigned_Stream'] != 'None').sum()}")
print(f"  • GCs sin asignación: {(gc_df['Assigned_Stream'] == 'None').sum()}")

# ============================================================================
# 4. ESTADÍSTICAS POR SUBESTRUCTURA
# ============================================================================
print("\n" + "="*70)
print("ESTADÍSTICAS POR SUBESTRUCTURA")
print("="*70)

results = []
for name, lit_feh, lit_err in zip(stream_names, stream_feh_lit, stream_feh_err_lit):
    sub = gc_df[gc_df['Assigned_Stream'] == name]
    n = len(sub)
    if n == 0:
        results.append({
            'Stream': name,
            'N': 0,
            'FeH_lit': lit_feh,
            'FeH_lit_err': lit_err,
            'FeH_gc_mean': np.nan,
            'FeH_gc_std': np.nan,
            'Age_gc_mean': np.nan,
            'Age_gc_std': np.nan,
        })
        continue
    
    feh_mean = sub['FeH'].mean()
    feh_std = sub['FeH'].std()
    age_mean = sub['age_Gyr'].mean()
    age_std = sub['age_Gyr'].std()
    
    results.append({
        'Stream': name,
        'N': n,
        'FeH_lit': lit_feh,
        'FeH_lit_err': lit_err,
        'FeH_gc_mean': feh_mean,
        'FeH_gc_std': feh_std,
        'Age_gc_mean': age_mean,
        'Age_gc_std': age_std,
    })
    
    print(f"\n{name}:")
    print(f"  • N = {n}")
    if lit_feh is not None:
        print(f"  • [Fe/H]_lit = {lit_feh:.2f} ± {lit_err:.2f}")
    print(f"  • [Fe/H]_GC = {feh_mean:.2f} ± {feh_std:.2f}")
    print(f"  • Edad_GC = {age_mean:.1f} ± {age_std:.1f} Gyr")

# ============================================================================
# 5. GENERAR TABLA LaTeX
# ============================================================================
latex_lines = []
latex_lines.append(r"\begin{table*}")
latex_lines.append(r"\centering")
latex_lines.append(r"\small")
latex_lines.append(r"\caption{Propiedades de los cúmulos globulares asociados a subestructuras del halo de NGC\,5128}")
latex_lines.append(r"\label{tab:gc_halo_substructures}")
latex_lines.append(r"\begin{tabular}{lcccccc}")
latex_lines.append(r"\hline")
latex_lines.append(r"Subestructura & $N_{\rm GC}$ & [Fe/H]$_{\rm stream}$ & [Fe/H]$_{\rm GC}$ & Edad$_{\rm GC}$ (Gyr) \\")
latex_lines.append(r"\hline")

for r in results:
    if r['N'] == 0:
        continue
    # Formatear metalicidad de literatura (si existe)
    if r['FeH_lit'] is not None:
        feh_lit_str = f"${r['FeH_lit']:.2f} \\pm {r['FeH_lit_err']:.2f}$"
    else:
        feh_lit_str = "---"
    feh_gc_str = f"${r['FeH_gc_mean']:.2f} \\pm {r['FeH_gc_std']:.2f}$"
    age_str = f"${r['Age_gc_mean']:.1f} \\pm {r['Age_gc_std']:.1f}$"
    latex_lines.append(f"{r['Stream']} & {r['N']} & {feh_lit_str} & {feh_gc_str} & {age_str} \\\\")

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\parbox{\linewidth}{\small \textit{Notas:} Asociación espacial con radio de búsqueda de 15 kpc proyectados. Las metalicidades estelares de referencia provienen de \citet{Crnojevic:2019} (Tabla~5) para Dw3 core, South y North; para las demás subestructuras no hay datos publicados. Las edades y metalicidades de los GCs son de nuestro ajuste SED (CIGALE).}")
latex_lines.append(r"\end{table*}")

print("\n" + "="*70)
print("CÓDIGO LaTeX PARA TU ARTÍCULO")
print("="*70)
for line in latex_lines:
    print(line)

# ============================================================================
# 6. GUARDAR RESULTADOS EN CSV
# ============================================================================
import os
os.makedirs("Streams", exist_ok=True)

output_csv = "Streams/gc_assignment_to_streams.csv"
gc_df.to_csv(output_csv, index=False)
print(f"\n✓ Archivo guardado: {output_csv}")

results_df = pd.DataFrame(results)
results_df.to_csv("Streams/stream_statistics.csv", index=False)
print("✓ Archivo guardado: Streams/stream_statistics.csv")

print("\n" + "="*70)
print("PROCESO COMPLETADO")
print("="*70)
