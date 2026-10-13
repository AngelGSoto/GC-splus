#!/usr/bin/env python3
# analyze_MR_population_standalone.py
#
# Version ejecutable directa: reusa la carga de datos y crossmatch de
# comparison_cigale_literature_final_optimal.py (mismo directorio) y
# corre el diagnostico de la poblacion MR (vieja, metal-rica ~solar).
#
# Uso:
#   python analyze_MR_population_standalone.py
#
# Requiere estar en el mismo directorio (o ajustar el import de abajo)
# que comparison_cigale_literature_final_optimal.py

import sys
import os
import numpy as np
import pandas as pd
from astropy.table import Table

# ----------------------------------------------------------------------
# Importar las funciones ya existentes del script original.
# Si el archivo tiene otro nombre en tu carpeta, ajusta esta linea.
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    # El nombre del modulo debe ser un identificador valido de Python,
    # por eso se importa con importlib en vez de "import compare-luis...".
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "compare_luis_beasley_svea",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "compare-luis-beasley-svea.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    load_cigale_results_with_errors = _mod.load_cigale_results_with_errors
    simple_crossmatch = _mod.simple_crossmatch
except FileNotFoundError:
    print("⚠️  No se encontro 'compare-luis-beasley-svea.py' en esta carpeta.")
    print("    Corre este script desde la misma carpeta que ese archivo,")
    print("    o ajusta la ruta en el bloque de import de arriba.")
    sys.exit(1)

# ============================================================================
# PARAMETROS DEL ANALISIS (ajustables)
# ============================================================================
OLD_THRESHOLD = 8.0          # Gyr
MR_FEH_THRESHOLD = -0.2      # dex, umbral aprox para "metal-rico" (MR)
AGE_GRID_MAX = 13.0          # Gyr, limite superior de la grilla CIGALE
EDGE_MARGIN = 1.0            # Gyr
EDGE_THRESHOLD = AGE_GRID_MAX - EDGE_MARGIN   # objetos con age >= 12.0 Gyr
REDUCED_CHI2_THRESHOLD = 2.0
CROSSMATCH_RADIUS = {'beasley': 3.0, 'hernandez': 4.0}   # arcsec

# ============================================================================
# RUTAS (copiadas de tu script original - ajustar si hace falta)
# ============================================================================
PATH_MY_CAT = '../Results_Corrected/all_fields_photometry_OPTIMIZED_1496objects.csv'
PATH_BEASLEY = '../../globular_clusters_ngc5128_Beasley.csv'
PATH_HERNANDEZ = '../../globular_clusters_ngc5128.csv'


def load_all_catalogs():
    print("1. CARGANDO CATALOGOS...")
    my_cat = pd.read_csv(PATH_MY_CAT)
    print(f"    -> catalogo propio: {len(my_cat)} objetos")

    beasley = pd.read_csv(PATH_BEASLEY)
    beasley = beasley.rename(columns={
        'RAJ2000': 'RA', 'DEJ2000': 'DEC',
        '[M/H]': 'FeH', 'e_[M/H]': 'FeH_err',
    })
    beasley['FeH'] = pd.to_numeric(beasley['FeH'], errors='coerce')
    beasley['FeH_err'] = pd.to_numeric(beasley['FeH_err'], errors='coerce')
    print(f"    -> Beasley: {len(beasley)} objetos")

    hernandez = pd.read_csv(PATH_HERNANDEZ)
    hernandez = hernandez.rename(columns={
        'RA (J2000)': 'RA', 'Dec (J2000)': 'DEC',
        '[Fe/H] (dex)': 'FeH', 'Error (dex)': 'FeH_err',
    })
    hernandez['FeH'] = pd.to_numeric(hernandez['FeH'], errors='coerce')
    hernandez['FeH_err'] = pd.to_numeric(hernandez['FeH_err'], errors='coerce')
    print(f"    -> Hernandez: {len(hernandez)} objetos")

    cigale_df = load_cigale_results_with_errors()
    if cigale_df.empty:
        print("❌ No se pudo cargar el catalogo de CIGALE (out/results.fits).")
        sys.exit(1)

    return my_cat, beasley, hernandez, cigale_df


def combine_data(matches_df, cigale_df, source_name):
    """Igual a la funcion del script original: junta crossmatch + CIGALE."""
    if matches_df.empty:
        return pd.DataFrame()

    combined = []
    for _, row in matches_df.iterrows():
        t17id = str(row['T17ID']).strip()

        cigale_match = cigale_df[cigale_df['ID'].str.strip() == t17id]
        if cigale_match.empty:
            cigale_match = cigale_df[cigale_df['ID'].str.contains(t17id, na=False)]

        if not cigale_match.empty:
            cigale_row = cigale_match.iloc[0]
            combined.append({
                'T17ID': t17id,
                'ID_CIGALE': cigale_row['ID'],
                'FeH_CIGALE': cigale_row['FeH_CIGALE'],
                'FeH_err_CIGALE': cigale_row['FeH_err_CIGALE'],
                'age': cigale_row['age_Gyr'],
                'reduced_chi2': cigale_row['reduced_chi2'],
                'chi2': cigale_row['chi2'],
                'FeH_lit': row.get('lit_FeH', np.nan),
                'FeH_err_lit': row.get('lit_FeH_err', np.nan),
                'source': source_name,
                'separation': row.get('separation', np.nan),
            })

    return pd.DataFrame(combined)


def flag_MR_population(cigale_df):
    df = cigale_df.copy()
    df['is_old'] = df['age_Gyr'] > OLD_THRESHOLD
    df['is_MR'] = df['is_old'] & (df['FeH_CIGALE'] >= MR_FEH_THRESHOLD)
    df['is_grid_edge'] = df['age_Gyr'] >= EDGE_THRESHOLD

    n_total = len(df)
    n_old = df['is_old'].sum()
    n_MR = df['is_MR'].sum()
    n_MR_edge = (df['is_MR'] & df['is_grid_edge']).sum()

    print("\n" + "="*70)
    print("DIAGNOSTICO DE LA POBLACION MR (vieja, metal-rica)")
    print("="*70)
    print(f"  - Muestra total (chi2_r <= {REDUCED_CHI2_THRESHOLD}): {n_total}")
    print(f"  - Objetos viejos (age > {OLD_THRESHOLD} Gyr): {n_old}")
    print(f"  - Objetos MR (old & [Fe/H] >= {MR_FEH_THRESHOLD}): {n_MR} "
          f"({100*n_MR/max(n_total,1):.1f}% de la muestra)")
    print(f"  - De ellos, en el borde de grilla (age >= {EDGE_THRESHOLD} Gyr): "
          f"{n_MR_edge} ({100*n_MR_edge/max(n_MR,1):.1f}% de MR)")
    print("-"*70)
    return df


def crossmatch_MR_with_literature(df_flagged, my_cat, beasley, hernandez):
    df_beasley = combine_data(
        simple_crossmatch(my_cat, beasley, CROSSMATCH_RADIUS['beasley']),
        df_flagged, 'Beasley')
    df_hernandez = combine_data(
        simple_crossmatch(my_cat, hernandez, CROSSMATCH_RADIUS['hernandez']),
        df_flagged, 'Hernandez')

    lit_all = pd.concat([df_beasley, df_hernandez], ignore_index=True)
    if lit_all.empty:
        print("\n⚠️  No hubo cruces con literatura en general (revisar radios "
              "de busqueda / IDs).")
        return lit_all

    flag_cols = df_flagged.set_index('ID')[['is_MR', 'is_grid_edge', 'age_Gyr']]
    lit_all = lit_all.merge(flag_cols, left_on='ID_CIGALE', right_index=True,
                             how='left')

    n_MR_matched = lit_all['is_MR'].sum()
    n_MR_edge_matched = (lit_all['is_MR'] & lit_all['is_grid_edge']).sum()

    print("\nCRUCE MR x LITERATURA ESPECTROSCOPICA")
    print("-"*70)
    print(f"  - Objetos MR con contraparte espectroscopica: {n_MR_matched}")
    print(f"  - De ellos, en el borde de grilla: {n_MR_edge_matched}")

    if n_MR_matched > 0:
        mr_lit = lit_all[lit_all['is_MR']]
        mae_MR = np.mean(np.abs(mr_lit['FeH_CIGALE'] - mr_lit['FeH_lit']))
        bias_MR = np.mean(mr_lit['FeH_CIGALE'] - mr_lit['FeH_lit'])
        print(f"  - MAE (MR vs literatura): {mae_MR:.3f} dex")
        print(f"  - Bias (MR vs literatura): {bias_MR:+.3f} dex")
        print(f"  - [Fe/H]_lit min/mediana/max: "
              f"{mr_lit['FeH_lit'].min():.2f} / "
              f"{mr_lit['FeH_lit'].median():.2f} / "
              f"{mr_lit['FeH_lit'].max():.2f}")
        n_lit_confirms = (mr_lit['FeH_lit'] >= MR_FEH_THRESHOLD).sum()
        print(f"  - Espectroscopia TAMBIEN los clasifica como metal-ricos: "
              f"{n_lit_confirms}/{n_MR_matched}")
        print("\n  Detalle de objetos MR con contraparte espectroscopica:")
        cols_show = ['T17ID', 'source', 'age_Gyr', 'FeH_CIGALE', 'FeH_lit',
                     'is_grid_edge']
        print(mr_lit[cols_show].sort_values('age_Gyr', ascending=False)
              .to_string(index=False))
    else:
        print("  *** Ningun objeto MR tiene contraparte espectroscopica ***")
        print("      directa. No confirma ni descarta el artefacto -> ")
        print("      apoyarse en el chequeo GMM sin borde (mas abajo) y en")
        print("      la comparacion con BC03.")
    print("="*70)
    return lit_all


def check_metallicity_grid_pileup(df_flagged, z_grid=None, tol_dex=0.05):
    """
    Chequea si la subpoblacion MR se acumula cerca de los nodos discretos
    de la grilla de metalicidad de CIGALE (Apendice E: 5 valores de Z,
    de 0.0004 a 0.03). Un pileup en el nodo mas alto (Z=0.03,
    [Fe/H] ~ +0.176) explicaria un sesgo sistematico hacia MR similar
    al que se ve en el cruce con espectroscopia, sin necesidad de un
    problema en la grilla de edad.

    z_grid: lista de valores de Z usados en CIGALE. Si es None, se usa
    un grid log-espaciado de 5 puntos entre 0.0004 y 0.03, que es lo
    que dice el paper, pero CONVIENE reemplazarlo por los valores
    EXACTOS que usaste en pcigale.ini para que el chequeo sea preciso.
    """
    if z_grid is None:
        # Aproximacion: ajustar con los valores reales del pcigale.ini
        z_grid = np.geomspace(0.0004, 0.03, 5)
        print("  ⚠️  Usando grid de Z aproximado (geomspace). Reemplazar")
        print("      z_grid con los 5 valores exactos de tu pcigale.ini")
        print("      para un chequeo preciso.\n")

    Z_SOLAR = 0.02
    feh_nodes = np.log10(z_grid / Z_SOLAR)

    print("="*70)
    print("CHEQUEO DE PILEUP EN LOS NODOS DE LA GRILLA DE METALICIDAD")
    print("="*70)
    print("  Nodos de [Fe/H] disponibles en la grilla CIGALE:")
    for z, feh in zip(z_grid, feh_nodes):
        print(f"    Z={z:.4f}  ->  [Fe/H]={feh:+.3f} dex")
    print(f"  Tolerancia para considerar 'en el nodo': ±{tol_dex} dex\n")

    df = df_flagged.copy()
    df['dist_to_nearest_node'] = df['FeH_CIGALE'].apply(
        lambda x: np.min(np.abs(feh_nodes - x)) if np.isfinite(x) else np.nan
    )
    df['nearest_node'] = df['FeH_CIGALE'].apply(
        lambda x: feh_nodes[np.argmin(np.abs(feh_nodes - x))]
        if np.isfinite(x) else np.nan
    )
    df['at_node'] = df['dist_to_nearest_node'] <= tol_dex

    # Foco en el nodo mas alto (el mas relevante para el sesgo hacia MR)
    top_node = feh_nodes.max()
    df['at_top_node'] = (np.abs(df['FeH_CIGALE'] - top_node) <= tol_dex)

    mr = df[df['is_MR']]
    n_mr = len(mr)
    n_mr_at_any_node = mr['at_node'].sum()
    n_mr_at_top_node = mr['at_top_node'].sum()

    print(f"  - N(MR) = {n_mr}")
    print(f"  - MR dentro de ±{tol_dex} dex de CUALQUIER nodo: "
          f"{n_mr_at_any_node} ({100*n_mr_at_any_node/max(n_mr,1):.1f}%)")
    print(f"  - MR dentro de ±{tol_dex} dex del nodo MAS ALTO "
          f"([Fe/H]={top_node:+.3f}): "
          f"{n_mr_at_top_node} ({100*n_mr_at_top_node/max(n_mr,1):.1f}%)")

    # Comparacion: lo mismo para MP e Int, como control
    for pop_name, mask in [('MP (referencia)', df['is_old'] &
                             (df['FeH_CIGALE'] < -0.8)),
                            ('Int (referencia)', df['is_old'] &
                             (df['FeH_CIGALE'] >= -0.8) &
                             (df['FeH_CIGALE'] < MR_FEH_THRESHOLD))]:
        sub = df[mask]
        if len(sub) > 0:
            frac = 100 * sub['at_node'].sum() / len(sub)
            print(f"  - [control] {pop_name}: {sub['at_node'].sum()}/"
                  f"{len(sub)} en algun nodo ({frac:.1f}%)")

    # Histograma de texto simple de FeH_CIGALE dentro de MR, para ver
    # si hay un pico angosto cerca del nodo mas alto
    print("\n  Distribucion de [Fe/H]_CIGALE dentro de MR (bins de 0.02 dex):")
    if n_mr > 0:
        bins = np.arange(MR_FEH_THRESHOLD, mr['FeH_CIGALE'].max() + 0.02, 0.02)
        counts, edges = np.histogram(mr['FeH_CIGALE'], bins=bins)
        for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
            if c > 0:
                bar = '#' * c
                marker = '  <-- nodo mas alto' if e0 <= top_node <= e1 else ''
                print(f"    [{e0:+.3f}, {e1:+.3f}): {c:3d} {bar}{marker}")

    print("="*70)
    return df


def rerun_gmm_excluding_edge(df_flagged):
    from sklearn.mixture import GaussianMixture

    old = df_flagged[df_flagged['is_old']].copy()
    old_clean = old[~old['is_grid_edge']].copy()

    print("\nRE-AJUSTE GMM EXCLUYENDO EL BORDE DE GRILLA")
    print("-"*70)
    print(f"  - N (old, original) = {len(old)}")
    print(f"  - N (old, sin borde) = {len(old_clean)}  "
          f"({len(old) - len(old_clean)} excluidos)")

    def fit_3comp(sample, label):
        X = sample['FeH_CIGALE'].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42,
                               n_init=10).fit(X)
        order = np.argsort(gmm.means_.flatten())
        means = gmm.means_.flatten()[order]
        weights = gmm.weights_[order]
        print(f"\n  [{label}]")
        for name, mu, w in zip(['MP', 'Int', 'MR'], means, weights):
            n_comp = int(w * len(sample))
            print(f"    {name}: mu={mu:+.3f} dex, f={100*w:.1f}%, N~{n_comp}")
        return means, weights

    means_orig, weights_orig = fit_3comp(old, "Muestra completa (con borde)")
    means_clean, weights_clean = fit_3comp(old_clean, "Sin objetos de borde")

    delta_mu_MR = means_clean[-1] - means_orig[-1]
    delta_f_MR = 100 * (weights_clean[-1] - weights_orig[-1])
    print("\n  DIFERENCIA EN COMPONENTE MR:")
    print(f"    delta_mu_MR = {delta_mu_MR:+.3f} dex")
    print(f"    delta_f_MR  = {delta_f_MR:+.1f} puntos porcentuales")
    print("="*70)
    return means_orig, weights_orig, means_clean, weights_clean


def main():
    my_cat, beasley, hernandez, cigale_df = load_all_catalogs()
    df_flagged = flag_MR_population(cigale_df)
    crossmatch_MR_with_literature(df_flagged, my_cat, beasley, hernandez)
    check_metallicity_grid_pileup(df_flagged)
    try:
        rerun_gmm_excluding_edge(df_flagged)
    except ImportError:
        print("\n⚠️  sklearn no disponible: salteando el re-ajuste GMM.")
        print("    Instalar con: pip install scikit-learn")


if __name__ == "__main__":
    main()
