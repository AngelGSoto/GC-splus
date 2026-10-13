#!/usr/bin/env python3
"""
Cross-match GCs with ALL halo substructures — FULL TABLE, NO KINEMATICS.
- Uses GMM n=3 labels.
- Assignment: nearest centre within RADIUS_KPC.
- Output: LaTeX table with all regions (Dw3 detailed + combined groups).
- Includes literature [Fe/H] where available.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import os, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
Z_SOLAR = 0.02
CHI2_MAX = 2.0
DIST_MPC = 3.8
KPC_TO_DEG = (1.0 / (DIST_MPC * 1000.0)) * (180.0 / np.pi)

RADIUS_KPC = 10.0        # Radio de búsqueda (kpc)

# ---------------------------------------------------------------------------
# Subestructuras y metalicidades de referencia (Crnojević+2019, Tabla 5)
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

# Metalicidades de referencia disponibles (solo para algunas regiones de Dw3)
LIT_FEH = {
    'Dw3 core':  (-0.61, 0.01),
    'Dw3 North': (-0.80, 0.01),
    'Dw3 South': (-0.86, 0.01),
    # Dw3 SE, Dw3 NW, Second stream, Shells: sin datos
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
# CARGA DE DATOS
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
# CLASIFICACIÓN GMM
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
# ASOCIACIÓN ESPACIAL
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
# FUNCIÓN AUXILIAR DE ESTADÍSTICAS
# =============================================================================
def compute_stats(sub):
    n = len(sub)
    if n == 0:
        return None
    feh = sub['FeH'].values
    age = sub['Age_Gyr'].values
    feh_mean = np.mean(feh)
    feh_std = np.std(feh, ddof=1) if n > 1 else np.nan
    age_mean = np.mean(age)
    age_std = np.std(age, ddof=1) if n > 1 else np.nan
    n_int = (sub['population'] == 'Int').sum()
    frac_int = n_int / n
    int_all = df[df['population'] == 'Int']['FeH'].values
    ks_p = ks_2samp(feh, int_all)[1] if len(int_all) > 0 else np.nan
    return {
        'N': n,
        'FeH_mean': feh_mean,
        'FeH_std': feh_std,
        'Age_mean': age_mean,
        'Age_std': age_std,
        'Frac_Int': frac_int,
        'KS_p': ks_p,
    }

# =============================================================================
# ESTADÍSTICAS PARA TODAS LAS REGIONES
# =============================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICAS POR REGIÓN (TODAS LAS SUBESTRUCTURAS)")
print("=" * 70)

# Regiones individuales (detailed)
individual_regions = [name for name in STREAM_REGIONS.keys() if not name.startswith('Shell')]
# Agregar Shells solo si tienen GCs
shell_regions = [name for name in ['Shell 1', 'Shell 2'] if (df['detailed_stream'] == name).sum() > 0]
all_individual = individual_regions + shell_regions

# Grupos combinados
combined_groups = ['Dw3 stream', 'Second stream', 'Shell 1', 'Shell 2']
# Solo incluir los combinados que tengan GCs
combined_present = []
for grp in combined_groups:
    if (df['stream'] == grp).sum() > 0:
        combined_present.append(grp)

results = []

# Procesar regiones individuales
for reg in all_individual:
    sub = df[df['detailed_stream'] == reg]
    stats = compute_stats(sub)
    if stats is None:
        continue
    lit_feh = LIT_FEH.get(reg, (None, None))
    results.append({
        'Region': reg,
        **stats,
        'FeH_lit': lit_feh[0],
        'FeH_lit_err': lit_feh[1],
    })

# Procesar grupos combinados
for grp in combined_present:
    sub = df[df['stream'] == grp]
    stats = compute_stats(sub)
    if stats is None:
        continue
    # Para Dw3 stream usamos el valor del core como referencia global
    lit_val = (-0.61, 0.01) if grp == 'Dw3 stream' else (None, None)
    results.append({
        'Region': grp + ' (all)',
        **stats,
        'FeH_lit': lit_val[0],
        'FeH_lit_err': lit_val[1],
    })

# Imprimir en pantalla
print(f"{'Region':<20} {'N':>3}  {'[Fe/H] (mean±std)':>22}  {'Age (Gyr)':>15}  {'f_Int':>6}  {'KS p':>6}")
print("-" * 85)
for r in results:
    feh_str = f"{r['FeH_mean']:.2f} ± {r['FeH_std']:.2f}" if not np.isnan(r['FeH_std']) else f"{r['FeH_mean']:.2f}"
    age_str = f"{r['Age_mean']:.1f} ± {r['Age_std']:.1f}" if not np.isnan(r['Age_std']) else f"{r['Age_mean']:.1f}"
    ks_str = f"{r['KS_p']:.3f}" if not np.isnan(r['KS_p']) else "nan"
    print(f"{r['Region']:<20} {r['N']:>3}  {feh_str:>22}  {age_str:>15}  {r['Frac_Int']:.2f}  {ks_str:>6}")

# =============================================================================
# GENERAR TABLA LaTeX COMPLETA
# =============================================================================
def fmt_plusminus(mean, std):
    if np.isnan(std):
        return f"${mean:.2f}$"
    else:
        return f"${mean:.2f}\\pm{std:.2f}$"

latex = []
latex.append(r"\begin{table*}[ht!]")
latex.append(r"\centering")
latex.append(r"\small")
latex.append(r"\caption{Globular clusters associated with all halo substructures.}")
latex.append(r"\label{tab:all_substructures}")
latex.append(r"\begin{tabular}{lcccccc}")
latex.append(r"\hline")
latex.append(r"Region & $N_{\rm GC}$ & [Fe/H]$_{\rm lit}$ & [Fe/H]$_{\rm GC}$ & Age (Gyr) & $f_{\rm Int}$ & KS $p$\\")
latex.append(r"\hline")

for r in results:
    lit_str = f"${r['FeH_lit']:.2f}\\pm{r['FeH_lit_err']:.2f}$" if r['FeH_lit'] is not None else "---"
    feh_str = fmt_plusminus(r['FeH_mean'], r['FeH_std'])
    age_str = fmt_plusminus(r['Age_mean'], r['Age_std'])
    ks_str = f"${r['KS_p']:.3f}$" if not np.isnan(r['KS_p']) else "---"
    latex.append(f"{r['Region']} & {r['N']} & {lit_str} & {feh_str} & {age_str} & {r['Frac_Int']:.2f} & {ks_str} \\\\")

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\parbox{\linewidth}{\small \textit{Notes:} Association radius 10\,kpc. Errors are standard deviations. Literature metallicities from \citet{Crnojevic:2019}. KS $p$-value compares the GCs in each region with the global Int population.}")
latex.append(r"\end{table*}")

print("\n" + "=" * 70)
print("TABLA LaTeX COMPLETA:")
for line in latex:
    print(line)

# Guardar
os.makedirs("Streams", exist_ok=True)
with open("Streams/table_all_substructures.tex", "w") as f:
    f.write("\n".join(latex))
df.to_csv("Streams/gc_with_gmm_and_streams.csv", index=False)
print("\n✓ Archivo guardado: Streams/table_all_substructures.tex")
print("✓ Script completado.")
