#!/usr/bin/env python3
# compare_XSL_BC03.py
# Comparación directa de resultados CIGALE con XSL y BC03
# Enfocado en la subpoblación MR y validación con espectroscopía.

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

# ============================================================
# RUTAS (AJUSTAR SEGÚN TU ESTRUCTURA DE CARPETAS)
# ============================================================
XSL_RESULTS = 'CIGALE-DEcam-extinction/out/results.fits'
BC03_RESULTS = 'CIGALE-DEcam-extinction_bc03_comparison/out/results.fits'
MY_CAT = 'Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv'
BEASLEY_CAT = '../globular_clusters_ngc5128_Beasley.csv'
HERNANDEZ_CAT = '../globular_clusters_ngc5128.csv'

CHI2_THRESHOLD = 2.0
OLD_THRESHOLD = 8.0      # Gyr
MR_FEH_THRESHOLD = -0.2  # dex

FIG_DIR = 'figures_comparison'
os.makedirs(FIG_DIR, exist_ok=True)

def load_cigale_results(path):
    """Carga resultados de CIGALE y convierte Z a [Fe/H] en dex."""
    if not os.path.exists(path):
        print(f"❌ No se encuentra {path}")
        return None

    with fits.open(path) as hdul:
        data = Table(hdul[1].data).to_pandas()

    # Metalicidad lineal Z y su error
    Z = data['bayes.stellar.metallicity'].values
    Z_err = data['bayes.stellar.metallicity_err'].values

    # Convertir a [Fe/H] usando Z_sun = 0.02
    Z_sun = 0.02
    FeH = np.log10(Z / Z_sun)
    # Propagación de error aproximada en dex
    FeH_err = np.abs((1.0 / np.log(10.0)) * (Z_err / Z))

    data['FeH_CIGALE'] = FeH
    data['FeH_err_CIGALE'] = FeH_err

    # Edad: convertir de Myr a Gyr si es necesario
    if data['bayes.stellar.age_m_star'].max() > 1000:
        data['age_Gyr'] = data['bayes.stellar.age_m_star'] / 1000.0
    else:
        data['age_Gyr'] = data['bayes.stellar.age_m_star']

    # Renombrar columnas de chi2
    data = data.rename(columns={
        'id': 'ID',
        'best.reduced_chi_square': 'reduced_chi2',
        'best.chi_square': 'chi2',
    })
    return data

def load_literature():
    """Carga catálogos de Beasley y Hernández con nombres estándar."""
    beasley = pd.read_csv(BEASLEY_CAT)
    beasley = beasley.rename(columns={
        'RAJ2000': 'RA', 'DEJ2000': 'DEC',
        '[M/H]': 'FeH', 'e_[M/H]': 'FeH_err',
    })
    beasley['FeH'] = pd.to_numeric(beasley['FeH'], errors='coerce')
    beasley['FeH_err'] = pd.to_numeric(beasley['FeH_err'], errors='coerce')

    hernandez = pd.read_csv(HERNANDEZ_CAT)
    hernandez = hernandez.rename(columns={
        'RA (J2000)': 'RA', 'Dec (J2000)': 'DEC',
        '[Fe/H] (dex)': 'FeH', 'Error (dex)': 'FeH_err',
    })
    hernandez['FeH'] = pd.to_numeric(hernandez['FeH'], errors='coerce')
    hernandez['FeH_err'] = pd.to_numeric(hernandez['FeH_err'], errors='coerce')
    return beasley, hernandez

def simple_crossmatch(my_cat, lit_cat, radius_arcsec):
    """Crossmatch por coordenadas usando separación angular."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c1 = SkyCoord(ra=my_cat['RA'], dec=my_cat['DEC'], unit='deg')
    c2 = SkyCoord(ra=lit_cat['RA'], dec=lit_cat['DEC'], unit='deg')
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    max_sep = radius_arcsec * u.arcsec
    sep_constraint = d2d < max_sep
    matches = []
    for i, (match_i, sep) in enumerate(zip(idx, d2d)):
        if sep_constraint[i]:
            matches.append({
                'T17ID': my_cat.iloc[i]['T17ID'],
                'lit_FeH': lit_cat.iloc[match_i]['FeH'],
                'lit_FeH_err': lit_cat.iloc[match_i].get('FeH_err', np.nan),
                'separation': sep.to(u.arcsec).value,
            })
    return pd.DataFrame(matches)

def compute_metrics(df_good, lit_matches, label):
    """Calcula métricas para la muestra completa y para MR."""
    print(f"\n{'='*60}")
    print(f"RESULTADOS PARA {label}")
    print(f"{'='*60}")

    n_total = len(df_good)
    n_old = (df_good['age_Gyr'] > OLD_THRESHOLD).sum()
    n_MR = (df_good['is_MR']).sum()
    print(f"  Muestra (χ²≤{CHI2_THRESHOLD}): {n_total}")
    print(f"  Viejos (>{OLD_THRESHOLD} Gyr): {n_old}")
    print(f"  MR (viejos y [Fe/H]≥{MR_FEH_THRESHOLD}): {n_MR}")

    if lit_matches.empty:
        print("  ⚠️ No hay coincidencias espectroscópicas.")
        return

    combined = lit_matches.merge(df_good, left_on='T17ID', right_on='ID', how='inner')
    if combined.empty:
        print("  ⚠️ No se pudo combinar con resultados.")
        return

    mae = np.mean(np.abs(combined['FeH_CIGALE'] - combined['lit_FeH']))
    bias = np.mean(combined['FeH_CIGALE'] - combined['lit_FeH'])
    print(f"\n  Métricas globales (N={len(combined)}):")
    print(f"    MAE = {mae:.3f} dex")
    print(f"    Bias = {bias:+.3f} dex")

    mr_combined = combined[combined['is_MR']]
    n_mr = len(mr_combined)
    print(f"\n  Submuestra MR con espectroscopía (N={n_mr}):")
    if n_mr > 0:
        mae_mr = np.mean(np.abs(mr_combined['FeH_CIGALE'] - mr_combined['lit_FeH']))
        bias_mr = np.mean(mr_combined['FeH_CIGALE'] - mr_combined['lit_FeH'])
        n_true_mr = (mr_combined['lit_FeH'] >= MR_FEH_THRESHOLD).sum()
        print(f"    MAE = {mae_mr:.3f} dex")
        print(f"    Bias = {bias_mr:+.3f} dex")
        print(f"    Espectroscopía confirma MR: {n_true_mr}/{n_mr}")
        print("    Detalle (T17ID, age, FeH_CIGALE, FeH_lit):")
        for _, row in mr_combined.sort_values('FeH_CIGALE', ascending=False).iterrows():
            print(f"      {row['T17ID']:>10s}  {row['age_Gyr']:5.2f}  "
                  f"{row['FeH_CIGALE']:+.3f}  {row['lit_FeH']:+.3f}")
    else:
        print("    No hay objetos MR con espectroscopía.")

def make_comparison_plot(xsl_df, bc03_df):
    """Genera diagramas edad-metalicidad para XSL y BC03."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, label in zip(axes, [xsl_df, bc03_df], ['XSL', 'BC03']):
        good = df[df['reduced_chi2'] <= CHI2_THRESHOLD]
        ax.scatter(good['FeH_CIGALE'], good['age_Gyr'], s=5, alpha=0.4,
                   c='gray', label='Todos')
        mr = good[(good['age_Gyr'] > OLD_THRESHOLD) &
                  (good['FeH_CIGALE'] >= MR_FEH_THRESHOLD)]
        ax.scatter(mr['FeH_CIGALE'], mr['age_Gyr'], s=15, alpha=0.8,
                   c='red', label=f'MR (N={len(mr)})')
        ax.axhline(OLD_THRESHOLD, ls='--', color='k', lw=0.8)
        ax.axvline(MR_FEH_THRESHOLD, ls='--', color='k', lw=0.8)
        ax.set_xlabel('[Fe/H] (dex)')
        ax.set_ylabel('Edad (Gyr)')
        ax.set_title(label)
        ax.legend()
    plt.tight_layout()
    outfig = os.path.join(FIG_DIR, 'age_metallicity_XSL_vs_BC03.pdf')
    plt.savefig(outfig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Figura guardada en {outfig}")

def main():
    print("="*70)
    print("COMPARACIÓN XSL vs BC03")
    print("="*70)

    # Cargar catálogo propio
    my_cat = pd.read_csv(MY_CAT)
    print(f"Catálogo propio: {len(my_cat)} objetos")

    # Ajustar nombres de coordenadas si es necesario
    if 'RA' not in my_cat.columns or 'DEC' not in my_cat.columns:
        rename_map = {}
        for col in my_cat.columns:
            if col.upper() in ['RA', 'RAJ2000', 'RA_ICRS', 'RA (J2000)', 'RA_DEG']:
                rename_map[col] = 'RA'
            elif col.upper() in ['DEC', 'DEJ2000', 'DEC_ICRS', 'DEC (J2000)', 'DE_DEG', 'DEC_DEG']:
                rename_map[col] = 'DEC'
        if 'RA' in rename_map.values() and 'DEC' in rename_map.values():
            my_cat = my_cat.rename(columns=rename_map)
            print("  Coordenadas renombradas a RA/DEC.")
        else:
            print("❌ No se pudieron identificar columnas RA/DEC en el catálogo propio.")
            print("   Columnas disponibles:", my_cat.columns.tolist())
            sys.exit(1)

    beasley, hernandez = load_literature()

    print("\nCargando resultados XSL...")
    xsl_df = load_cigale_results(XSL_RESULTS)
    if xsl_df is None:
        sys.exit(1)
    print(f"  XSL: {len(xsl_df)} objetos")

    print("\nCargando resultados BC03...")
    bc03_df = load_cigale_results(BC03_RESULTS)
    if bc03_df is None:
        sys.exit(1)
    print(f"  BC03: {len(bc03_df)} objetos")

    # Aplicar cortes de calidad y marcar MR
    for df in [xsl_df, bc03_df]:
        df['is_old'] = df['age_Gyr'] > OLD_THRESHOLD
        df['is_MR'] = df['is_old'] & (df['FeH_CIGALE'] >= MR_FEH_THRESHOLD)

    print("\nRealizando crossmatches...")
    beasley_matches = simple_crossmatch(my_cat, beasley, 3.0)
    beasley_matches['source'] = 'Beasley'
    hernandez_matches = simple_crossmatch(my_cat, hernandez, 4.0)
    hernandez_matches['source'] = 'Hernandez'
    all_matches = pd.concat([beasley_matches, hernandez_matches], ignore_index=True)
    print(f"  Total coincidencias espectroscópicas: {len(all_matches)}")

    compute_metrics(xsl_df[xsl_df['reduced_chi2'] <= CHI2_THRESHOLD],
                    all_matches, 'XSL')
    compute_metrics(bc03_df[bc03_df['reduced_chi2'] <= CHI2_THRESHOLD],
                    all_matches, 'BC03')

    make_comparison_plot(xsl_df, bc03_df)

if __name__ == "__main__":
    main()
