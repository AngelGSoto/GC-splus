#!/usr/bin/env python3
# robustness_check_noMR.py
# Verifica la robustez del GMM y gradientes al excluir SOLO los metal-rich VIEJOS
# (edad > 8 Gyr y [Fe/H] >= -0.2), conservando los metal-rich jóvenes.

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.mixture import GaussianMixture
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ========== RUTAS ==========
RESULTS_FITS = 'out/results.fits'          # resultados XSL
CATALOG_CSV  = '../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv'

# ========== PARÁMETROS ==========
CHI2_THRESHOLD = 2.0
FEH_CUT = -0.2   # umbral para considerar metal-rich
AGE_CUT = 8.0    # umbral de edad para considerar viejo

RA_CENTER, DEC_CENTER = 201.365063, -43.019112
DISTANCE_MPC = 3.8

# ========== FUNCIONES ==========
def load_and_prepare():
    """Carga resultados, aplica cortes y devuelve DataFrame limpio."""
    with fits.open(RESULTS_FITS) as hdul:
        data = Table(hdul[1].data).to_pandas()

    # Convertir Z a [Fe/H]
    Z = data['bayes.stellar.metallicity'].values
    Z_sun = 0.02
    data['FeH'] = np.log10(Z / Z_sun)

    # Edad en Gyr
    if data['bayes.stellar.age_m_star'].max() > 1000:
        data['Age_Gyr'] = data['bayes.stellar.age_m_star'] / 1000.0
    else:
        data['Age_Gyr'] = data['bayes.stellar.age_m_star']

    # Corte de chi2
    data = data[data['best.reduced_chi_square'] <= CHI2_THRESHOLD].copy()
    print(f"Después de χ²≤{CHI2_THRESHOLD}: {len(data)} objetos")

    # Excluir SOLO los MR viejos (edad > AGE_CUT y [Fe/H] >= FEH_CUT)
    data = data[~((data['Age_Gyr'] > AGE_CUT) & (data['FeH'] >= FEH_CUT))].copy()
    print(f"Después de excluir MR viejos (edad > {AGE_CUT} Gyr y [Fe/H] ≥ {FEH_CUT}): "
          f"{len(data)} objetos")

    return data

def add_coordinates(df):
    """Agrega coordenadas RA/DEC desde el catálogo."""
    try:
        cat = pd.read_csv(CATALOG_CSV)
        ra_col = 'RAJ2000' if 'RAJ2000' in cat.columns else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in cat.columns else 'DEC'
        df['id_str'] = df['id'].astype(str).str.strip()
        cat['T17ID_str'] = cat['T17ID'].astype(str).str.strip()
        df = pd.merge(df, cat[[ra_col, dec_col, 'T17ID_str']],
                      left_on='id_str', right_on='T17ID_str', how='left')
        df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
        print(f"  Coordenadas cargadas para {df['RA'].notna().sum()} objetos")
        return df
    except Exception as e:
        print(f"  ⚠️ No se pudieron cargar coordenadas: {e}")
        return df

def projected_radii(df):
    """Calcula radios proyectados en kpc."""
    valid = df['RA'].notna() & df['DEC'].notna()
    if not valid.any():
        return None, valid
    coords = SkyCoord(ra=df.loc[valid, 'RA'].values*u.degree,
                      dec=df.loc[valid, 'DEC'].values*u.degree)
    center = SkyCoord(ra=RA_CENTER*u.degree, dec=DEC_CENTER*u.degree)
    radii = DISTANCE_MPC * 1000 * np.pi * center.separation(coords).degree / 180.0
    return radii, valid

def run_gmm(feh, max_n=3):
    """Corre GMM para n=2 y n=3 y devuelve resultados."""
    X = feh.reshape(-1, 1)
    results = {}
    for n in [2, 3]:
        best_gmm, best_score = None, -np.inf
        for seed in [42, 123, 456]:
            gmm = GaussianMixture(n_components=n, random_state=seed,
                                  n_init=5, max_iter=500, tol=1e-4)
            gmm.fit(X)
            if gmm.score(X) > best_score:
                best_score = gmm.score(X)
                best_gmm = gmm
        bic = best_gmm.bic(X)
        aic = best_gmm.aic(X)
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_.flatten()
        sort_idx = np.argsort(means)
        means = means[sort_idx]
        stds = stds[sort_idx]
        weights = weights[sort_idx]
        results[n] = {
            'gmm': best_gmm,
            'bic': bic, 'aic': aic,
            'means': means, 'stds': stds, 'weights': weights,
        }
        print(f"\n  GMM n={n}:")
        print(f"    BIC = {bic:.1f}, AIC = {aic:.1f}")
        for i in range(n):
            print(f"    Comp {i+1}: [Fe/H] = {means[i]:.3f} ± {stds[i]:.3f}, "
                  f"f = {weights[i]*100:.1f}%")
    return results

def assign_labels(feh, gmm, n):
    """Asigna etiquetas ordenadas por metalicidad creciente."""
    labels_orig = gmm.predict(feh.reshape(-1, 1))
    means = gmm.means_.flatten()
    sort_idx = np.argsort(means)
    label_map = {orig: new for new, orig in enumerate(sort_idx)}
    return np.array([label_map[l] for l in labels_orig])

def compute_gradient(feh, labels, radii):
    """Calcula gradiente para cada población (asumiendo n=3 o n=2)."""
    valid = np.isfinite(radii) & np.isfinite(feh)
    feh_v = feh[valid]
    labels_v = labels[valid]
    log_r = np.log10(radii[valid])

    n_pop = len(np.unique(labels_v))
    gradients = {}
    for i in range(n_pop):
        mask = labels_v == i
        if mask.sum() < 10:
            continue
        slope, intercept, r, p, se = linregress(log_r[mask], feh_v[mask])
        gradients[f'Pop {i+1}'] = {
            'slope': slope, 'slope_err': se, 'r': r, 'p': p, 'N': mask.sum()
        }
    return gradients

def main():
    print("="*70)
    print("ANÁLISIS DE ROBUSTEZ: EXCLUYENDO SOLO MR VIEJOS")
    print("="*70)

    df = load_and_prepare()
    df = add_coordinates(df)
    radii, valid = projected_radii(df)

    feh = df['FeH'].values
    print(f"\nMuestra final para GMM: {len(feh)} objetos")

    # GMM con n=2 y n=3
    results = run_gmm(feh)

    # Si n=3, mostramos gradientes para las 3 poblaciones
    if 3 in results:
        print("\n--- Gradientes para n=3 (sin MR viejos) ---")
        gmm3 = results[3]['gmm']
        labels3 = assign_labels(feh, gmm3, 3)
        if radii is not None:
            grad3 = compute_gradient(feh, labels3, radii)
            for pop, gr in grad3.items():
                print(f"  {pop}: slope = {gr['slope']:.3f} ± {gr['slope_err']:.3f}, "
                      f"p = {gr['p']:.4f}, N = {gr['N']}")

    # Si n=2, gradientes para 2 poblaciones
    if 2 in results:
        print("\n--- Gradientes para n=2 ---")
        gmm2 = results[2]['gmm']
        labels2 = assign_labels(feh, gmm2, 2)
        if radii is not None:
            grad2 = compute_gradient(feh, labels2, radii)
            for pop, gr in grad2.items():
                print(f"  {pop}: slope = {gr['slope']:.3f} ± {gr['slope_err']:.3f}, "
                      f"p = {gr['p']:.4f}, N = {gr['N']}")

    print("\n" + "="*70)
    print("FIN DEL ANÁLISIS DE ROBUSTEZ")
    print("="*70)

if __name__ == "__main__":
    main()
