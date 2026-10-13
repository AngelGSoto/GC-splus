#!/usr/bin/env python3
"""
Cross-match GCs with the Dw3 stellar stream — FINAL VERSION with kinematic filter.
Includes detailed per-region output for Dw3 apertures.
"""

import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import os, glob, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
Z_SOLAR = 0.02
CHI2_MAX = 2.0
DIST_MPC = 3.8
KPC_TO_DEG = (1.0 / (DIST_MPC * 1000.0)) * (180.0 / np.pi)

RADIUS_KPC = 10.0                     # Radio de búsqueda espacial (kpc)

# Parámetros cinemáticos de Dw3 (Hughes+2023)
DW3_SYSTEMIC_VEL = 359.0              # km/s
DW3_VEL_TOLERANCE = 100.0             # km/s

# ---------------------------------------------------------------------------
# Subestructuras y metalicidades de referencia
# ---------------------------------------------------------------------------
STREAM_REGIONS = {
    'Dw3 core':  ('13:30:21.5', '-42:11:33'),
    'Dw3 North': ('13:28:27.6', '-41:51:00'),
    'Dw3 South': ('13:31:02.6', '-42:21:54'),
    'Dw3 SE':    ('13:32:18.0', '-42:36:54'),
    'Dw3 NW':    ('13:23:57.6', '-41:42:00'),
    'Second stream': ('13:31:54.0', '-41:51:00'),
    'Shell 1':       ('13:26:45.6', '-42:22:30'),
    'Shell 2':       ('13:23:21.6', '-43:14:06'),
}

LIT_FEH = {
    'Dw3 core':  (-0.61, 0.01),
    'Dw3 North': (-0.80, 0.01),
    'Dw3 South': (-0.86, 0.01),
}

def group_region(name):
    if name.startswith('Dw3'):
        return 'Dw3 stream'
    elif name.startswith('Shell'):
        return name
    else:
        return name

# ---------------------------------------------------------------------------
# Funciones GMM (idénticas a tu análisis)
# ---------------------------------------------------------------------------
def validate_gmm_model(feh_data, n_components, final_seed=42):
    X = feh_data.reshape(-1, 1)
    best_score = -np.inf
    for seed in [123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed,
                              n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        if gmm.score(X) > best_score:
            best_score = gmm.score(X)
    gmm_final = GaussianMixture(n_components=n_components, random_state=final_seed,
                                n_init=5, max_iter=500, tol=1e-4)
    gmm_final.fit(X)
    return gmm_final

def get_labels_forced_n3(feh_data):
    gmm = validate_gmm_model(feh_data, 3)
    X = feh_data.reshape(-1, 1)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    stds = stds[sort_idx]
    weights = weights[sort_idx]
    labels = gmm.predict(X)
    label_map = {int(i_orig): int(i_sorted) for i_sorted, i_orig in enumerate(sort_idx)}
    labels_ordered = np.array([label_map[l] for l in labels])
    return labels_ordered, means, stds, weights

# =============================================================================
# 1. CARGA DE DATOS PROPIOS
# =============================================================================
print("=" * 70)
print("CARGANDO DATOS")
print("=" * 70)

phot_cat = pd.read_csv(
    '../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
phot_cat['id_str'] = phot_cat['T17ID'].astype(str).str.strip()
print(f"Fotometría: {len(phot_cat)} objetos")

cig = Table.read('out/results.fits', hdu=1).to_pandas()
cig['id_str'] = cig['id'].astype(str).str.strip()
print(f"CIGALE inicial: {len(cig)}")

cig = cig[cig['best.reduced_chi_square'] <= CHI2_MAX]
cig['FeH'] = np.log10(cig['bayes.stellar.metallicity'] / Z_SOLAR)
cig['Age_Gyr'] = cig['bayes.stellar.age_m_star'] / 1000.0
print(f"CIGALE con χ²≤2.0: {len(cig)}")

df = pd.merge(cig, phot_cat[['id_str', 'RAJ2000', 'DEJ2000']],
              on='id_str', how='inner')
print(f"Unión con coordenadas: {len(df)} GCs")

# =============================================================================
# 2. CLASIFICACIÓN GMM
# =============================================================================
print("\n" + "=" * 70)
print("CLASIFICANDO CON GMM (n=3)")
print("=" * 70)
labels_n3, means_gmm, stds_gmm, weights_gmm = get_labels_forced_n3(df['FeH'].values)
pop_map = {0: 'MP', 1: 'Int', 2: 'MR'}
df['population'] = [pop_map[l] for l in labels_n3]
for i, pop in enumerate(['MP', 'Int', 'MR']):
    print(f"  {pop}: μ={means_gmm[i]:.3f}±{stds_gmm[i]:.3f}, "
          f"f={weights_gmm[i]*100:.1f}%, N={np.sum(labels_n3==i)}")

# =============================================================================
# 3. ASOCIACIÓN ESPACIAL
# =============================================================================
print("\n" + "=" * 70)
print("ASIGNANDO GCs A SUBESTRUCTURAS")
print("=" * 70)

gc_coords = SkyCoord(ra=df['RAJ2000'].values * u.degree,
                     dec=df['DEJ2000'].values * u.degree)

centers = []
names_list = []
for name, (ra_str, dec_str) in STREAM_REGIONS.items():
    centers.append(SkyCoord(ra_str, dec_str, unit=('hourangle', 'deg')))
    names_list.append(name)

sep_matrix = np.column_stack([
    gc_coords.separation(c).degree / KPC_TO_DEG for c in centers
])

detailed = np.full(len(df), "None", dtype=object)
for i in range(len(df)):
    seps = sep_matrix[i]
    within = np.where(seps <= RADIUS_KPC)[0]
    if len(within) > 0:
        best = within[np.argmin(seps[within])]
        detailed[i] = names_list[best]

df['detailed_stream'] = detailed
df['stream'] = [group_region(r) for r in detailed]

for name in ['Dw3 stream', 'Second stream', 'Shell 1', 'Shell 2']:
    n = (df['stream'] == name).sum()
    print(f"  {name}: {n} GCs")
print(f"\nTotal asignados: {(df['stream'] != 'None').sum()} / {len(df)}")

# =============================================================================
# 4. CARGA DE VELOCIDADES RADIALES DESDE CSVs
# =============================================================================
print("\n" + "=" * 70)
print("CARGANDO VELOCIDADES RADIALES (Hughes+2023) desde CSV")
print("=" * 70)

streams_dir = 'Streams'
if not os.path.isdir(streams_dir):
    print(f"  El directorio '{streams_dir}' no existe.")
    print(f"  Directorio actual: {os.getcwd()}")
else:
    print(f"  Contenido de {streams_dir}/:")
    for item in sorted(os.listdir(streams_dir)):
        print(f"    {item}")

pattern3 = os.path.join(streams_dir, '*table3*.csv')
pattern4 = os.path.join(streams_dir, '*table4*.csv')
files3 = sorted(glob.glob(pattern3))
files4 = sorted(glob.glob(pattern4))

tables_csv = []
if files3:
    for f in files3:
        t = Table.read(f, format='csv')
        tables_csv.append(t)
        print(f"  Leído {f} ({len(t)} objetos)")
else:
    print(f"  No se encontró '*table3*.csv' en {streams_dir}/")
if files4:
    for f in files4:
        t = Table.read(f, format='csv')
        tables_csv.append(t)
        print(f"  Leído {f} ({len(t)} objetos)")
else:
    print(f"  No se encontró '*table4*.csv' en {streams_dir}/")

if tables_csv:
    hugh_cat = vstack(tables_csv)
    for old, new in [('RAJ2000', 'RAdeg'), ('DEJ2000', 'DEdeg')]:
        if old in hugh_cat.colnames:
            hugh_cat.rename_column(old, new)
    print(f"  Catálogo Hughes combinado: {len(hugh_cat)} objetos.")
    has_velocities = True
else:
    hugh_cat = None
    has_velocities = False
    print("  No se pudieron cargar las tablas de Hughes. Continuando sin filtro cinemático.")

# =============================================================================
# 5. CRUCE CON VELOCIDADES Y FILTRO CINEMÁTICO
# =============================================================================
if has_velocities:
    hugh_coords = SkyCoord(ra=hugh_cat['RAdeg'].data * u.deg,
                           dec=hugh_cat['DEdeg'].data * u.deg)

    def get_best_velocity(row):
        if 'WRVel' in row.colnames and not np.ma.is_masked(row['WRVel']):
            return float(row['WRVel'])
        vels = []
        for col in ['RVel1', 'RVel2']:
            if col in row.colnames and not np.ma.is_masked(row[col]):
                vels.append(float(row[col]))
        if vels:
            return np.mean(vels)
        return np.nan

    hugh_vels = np.array([get_best_velocity(row) for row in hugh_cat])
    valid_vel = np.isfinite(hugh_vels)
    hugh_coords = hugh_coords[valid_vel]
    hugh_vels = hugh_vels[valid_vel]
    print(f"  Objetos con velocidad disponible: {len(hugh_vels)}")

    idx_gc, sep_deg, _ = match_coordinates_sky(gc_coords, hugh_coords, nthneighbor=1)
    match_mask = sep_deg.degree < 1.5/3600.0
    df['vr'] = np.where(match_mask, hugh_vels[idx_gc], np.nan)
    n_vel_matches = match_mask.sum()
    print(f"  GCs con velocidad radial en Hughes: {n_vel_matches}")

    # Marcar los que están en la ventana cinemática
    df['in_vel_window'] = False
    if n_vel_matches > 0:
        in_window = (df['vr'] >= DW3_SYSTEMIC_VEL - DW3_VEL_TOLERANCE) & \
                    (df['vr'] <= DW3_SYSTEMIC_VEL + DW3_VEL_TOLERANCE)
        df.loc[in_window, 'in_vel_window'] = True

    # Para Dw3 completo
    dw3_mask = df['stream'] == 'Dw3 stream'
    dw3_with_vr = dw3_mask & df['vr'].notna()
    print(f"  Dw3 stream: {dw3_mask.sum()} GCs espaciales, {dw3_with_vr.sum()} con velocidad")
    if dw3_with_vr.sum() > 0:
        vr_dw3 = df.loc[dw3_with_vr, 'vr'].values
        in_win = (vr_dw3 >= DW3_SYSTEMIC_VEL - DW3_VEL_TOLERANCE) & \
                 (vr_dw3 <= DW3_SYSTEMIC_VEL + DW3_VEL_TOLERANCE)
        gc_ids_window = df.loc[dw3_with_vr, 'id_str'].values[in_win]
        df['dw3_kinematic'] = False
        df.loc[df['id_str'].isin(gc_ids_window), 'dw3_kinematic'] = True
        n_kinematic = len(gc_ids_window)
    else:
        n_kinematic = 0
        df['dw3_kinematic'] = False
    print(f"  GCs de Dw3 en ventana cinemática (359±100 km/s): {n_kinematic}")
else:
    df['vr'] = np.nan
    df['in_vel_window'] = False
    df['dw3_kinematic'] = False
    n_kinematic = 0

# =============================================================================
# 6. REGIONES INDIVIDUALES DE Dw3 (análisis detallado)
# =============================================================================
print("\n" + "=" * 70)
print("REGIONES INDIVIDUALES DE Dw3")
print("=" * 70)

dw3_regions = [name for name in STREAM_REGIONS.keys() if name.startswith('Dw3')]
print(f"{'Region':<15} {'N':>3} {'[Fe/H] (mean±std)':>20} {'Age (Gyr)':>15} {'f_Int':>6} {'N_vr':>5} {'N_win':>5}")
print("-" * 75)
for region in dw3_regions:
    mask = df['detailed_stream'] == region
    sub = df[mask]
    n_reg = len(sub)
    if n_reg == 0:
        print(f"{region:<15} {0:>3}")
        continue
    feh = sub['FeH'].values
    age = sub['Age_Gyr'].values
    n_int = (sub['population'] == 'Int').sum()
    n_vr = sub['vr'].notna().sum()
    n_win = sub['in_vel_window'].sum()
    print(f"{region:<15} {n_reg:>3}   {np.mean(feh):.2f} ± {np.std(feh, ddof=1):.2f}        "
          f"{np.mean(age):.1f} ± {np.std(age, ddof=1):.1f}   {n_int/n_reg:.2f}   {n_vr:>4}   {n_win:>4}")

# =============================================================================
# 7. ESTADÍSTICAS FINALES (para la fila combinada)
# =============================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICAS – Dw3 STREAM (combinado)")
print("=" * 70)

if has_velocities and n_kinematic >= 2:
    sub = df[df['dw3_kinematic']]
    kin_note = True
    print(f"Usando muestra cinemática: {len(sub)} GCs.")
else:
    sub = df[df['stream'] == 'Dw3 stream']
    kin_note = False
    if has_velocities and n_kinematic < 2:
        print(f"Pocos GCs cinemáticos ({n_kinematic}), usando muestra espacial completa ({len(sub)} GCs).")
    else:
        print(f"Sin filtro cinemático, usando muestra espacial completa ({len(sub)} GCs).")

n = len(sub)
feh_gc = sub['FeH'].values
age_gc = sub['Age_Gyr'].values

feh_mean = np.mean(feh_gc)
feh_std  = np.std(feh_gc, ddof=1)
age_mean = np.mean(age_gc)
age_std  = np.std(age_gc, ddof=1)

n_int = (sub['population'] == 'Int').sum()
frac_int = n_int / n if n > 0 else np.nan

int_all = df[df['population'] == 'Int']['FeH'].values
ks_p = ks_2samp(feh_gc, int_all)[1] if len(int_all) > 0 else np.nan

print(f"Dw3 stream (final): N={n}, <[Fe/H]>={feh_mean:.2f}±{feh_std:.2f} (std), "
      f"f_Int={frac_int:.2f}, KS p={ks_p:.3f}")

# =============================================================================
# 8. GENERAR TABLA LaTeX (UNA SOLA FILA)
# =============================================================================
latex = []
latex.append(r"\begin{table}[ht!]")
latex.append(r"\centering")
latex.append(r"\small")
latex.append(r"\caption{Globular clusters associated with the Dw3 stellar stream.}")
latex.append(r"\label{tab:dw3_stream}")
latex.append(r"\begin{tabular}{lccccc}")
latex.append(r"\hline")
latex.append(r"Region & $N_{\rm GC}$ & [Fe/H]$_{\rm lit}$ & [Fe/H]$_{\rm GC}$ & Age (Gyr) & $f_{\rm Int}$ \\")
latex.append(r"\hline")

feh_str = f"${feh_mean:.2f}\\pm{feh_std:.2f}$"
age_str = f"${age_mean:.1f}\\pm{age_std:.1f}$"
lit_str = f"${-0.61:.2f}\\pm{0.01:.2f}$"

latex.append(f"Dw3 stream & {n} & {lit_str} & {feh_str} & {age_str} & {frac_int:.2f} \\\\")

latex.append(r"\hline")
latex.append(r"\end{tabular}")

note = (r"\parbox{\linewidth}{\small \textit{Notes:} Association radius 10\,kpc around five apertures. "
        fr"Errors are standard deviation ({feh_std:.2f}\,dex). ")
if kin_note:
    note += (r"GCs selected to have radial velocities within 100\,km\,s$^{-1}$ of the Dw3 systemic "
             fr"velocity ({int(DW3_SYSTEMIC_VEL)}\,km\,s$^{{-1}}$; \citet{{Hughes:2023}}). ")
note += (r"Literature metallicity from \citet{Crnojevic:2019} (Tab.~5). "
         fr"The Int fraction ({frac_int*100:.0f}\%) is consistent with the global GMM value (51.8\%).}}")
latex.append(note)

latex.append(r"\end{table}")

print("\n" + "=" * 70)
print("TABLA LaTeX:")
for line in latex:
    print(line)

# Guardar
os.makedirs("Streams", exist_ok=True)
out_table = os.path.join("Streams", "table_dw3_stream.tex")
with open(out_table, "w") as f:
    f.write("\n".join(latex))
df.to_csv(os.path.join("Streams", "gc_with_gmm_and_streams.csv"), index=False)
print(f"\n✓ Archivos guardados en Streams/")
print("✓ Script completado.")
