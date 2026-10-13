#!/usr/bin/env python3
# gmm_ngc5128_full_analysis.py
# Análisis GMM completo (1D y 2D) para NGC 5128 GCs
# Incluye diagrama color-color y contornos de densidad en 2D.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, linregress, gaussian_kde
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import datetime
import json

# ============================================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})

RA_CENTER, DEC_CENTER = 201.365063, -43.019112
DISTANCE_MPC = 3.8

# Paleta de colores
COLORS = ['#2E86AB', '#A23B72', '#C73E1D', '#F18F01', '#6A4C93', '#1982C4', '#4C9F70']
COLORS_3 = COLORS[:3]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def calculate_bayes_factor(bic1, bic2):
    return np.exp((bic2 - bic1) / 2)

def interpret_bayes_factor(bf):
    if bf > 1:
        if bf > 150: return "Decisive evidence for M1"
        elif bf > 20: return "Strong evidence for M1"
        elif bf > 3:  return "Substantial evidence for M1"
        else:         return "No significant evidence"
    else:
        bf_inv = 1 / bf
        if bf_inv > 150: return "Decisive evidence for M2"
        elif bf_inv > 20: return "Strong evidence for M2"
        elif bf_inv > 3:  return "Substantial evidence for M2"
        else:             return "No significant evidence"

def calculate_projected_radii(df):
    if 'RA' not in df.columns or 'DEC' not in df.columns:
        return None, None
    valid = df['RA'].notna() & df['DEC'].notna()
    if not valid.any():
        return None, None
    coords = SkyCoord(ra=df.loc[valid, 'RA'].values*u.degree,
                      dec=df.loc[valid, 'DEC'].values*u.degree)
    center = SkyCoord(ra=RA_CENTER*u.degree, dec=DEC_CENTER*u.degree)
    radii = DISTANCE_MPC * 1000 * np.pi * center.separation(coords).degree / 180.0
    return radii, valid

# ============================================================================
# GMM 1D (solo metalicidad)
# ============================================================================
def run_gmm_1D(feh_data, label=''):
    X = feh_data.reshape(-1, 1)
    results = {}
    for n in range(2, 6):
        best_gmm, best_score = None, -np.inf
        for seed in [42, 123, 456, 789, 999]:
            gmm = GaussianMixture(n_components=n, random_state=seed,
                                  n_init=5, max_iter=500, tol=1e-4)
            gmm.fit(X)
            if gmm.score(X) > best_score:
                best_score = gmm.score(X)
                best_gmm = gmm
        bic = float(best_gmm.bic(X))
        aic = float(best_gmm.aic(X))
        labels = best_gmm.predict(X)
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_.flatten()
        sort_idx = np.argsort(means)
        means, stds, weights = means[sort_idx], stds[sort_idx], weights[sort_idx]
        separations = [abs(means[i+1]-means[i])/np.sqrt(stds[i]**2+stds[i+1]**2) for i in range(n-1)]
        label_map = {int(old): int(new) for new, old in enumerate(sort_idx)}
        labels_ordered = np.array([label_map[l] for l in labels])
        results[n] = {
            'bic': bic, 'aic': aic, 'gmm': best_gmm,
            'means': means, 'stds': stds, 'weights': weights,
            'separations': separations,
            'labels': labels_ordered
        }
    best_n = min(results.keys(), key=lambda x: results[x]['bic'])
    print(f"\nGMM 1D {label} -> óptimo n={best_n} (BIC={results[best_n]['bic']:.1f})")
    return results, best_n

# ============================================================================
# GMM 2D (Fe/H + logAge)
# ============================================================================
def validate_gmm_2D(X, n):
    best_gmm, best_score = None, -np.inf
    for seed in [42, 123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n, random_state=seed, n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        if gmm.score(X) > best_score:
            best_score = gmm.score(X)
            best_gmm = gmm
    return best_gmm

def run_gmm_2D(feh, logAge, label=''):
    X_raw = np.column_stack([feh, logAge])
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    results = {}
    for n in range(2, 6):
        gmm = validate_gmm_2D(X, n)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        labels = gmm.predict(X)
        means_std = gmm.means_
        means_orig = scaler.inverse_transform(means_std)
        weights = gmm.weights_.flatten()
        sort_idx = np.argsort(means_orig[:,0])
        means_orig = means_orig[sort_idx]
        weights = weights[sort_idx]
        label_map = {int(old): int(new) for new, old in enumerate(sort_idx)}
        labels_ordered = np.array([label_map[l] for l in labels])
        emp_means = np.array([np.mean(X_raw[labels_ordered==i], axis=0) for i in range(n)])
        emp_stds  = np.array([np.std(X_raw[labels_ordered==i], axis=0) for i in range(n)])
        # separaciones en Fe/H
        covs_std = gmm.covariances_
        separations = []
        for i in range(n-1):
            var_i = (scaler.scale_[0]**2) * covs_std[sort_idx[i],0,0]
            var_j = (scaler.scale_[0]**2) * covs_std[sort_idx[i+1],0,0]
            sep = abs(means_orig[i+1,0]-means_orig[i,0]) / np.sqrt(var_i+var_j)
            separations.append(float(sep))
        results[n] = {
            'bic': bic, 'aic': aic, 'gmm': gmm, 'scaler': scaler,
            'means': emp_means, 'stds': emp_stds, 'weights': weights,
            'separations': separations, 'labels': labels_ordered
        }
    best_n = min(results.keys(), key=lambda x: results[x]['bic'])
    print(f"GMM 2D {label} -> óptimo n={best_n} (BIC={results[best_n]['bic']:.1f})")
    return results, best_n

# ============================================================================
# FIGURAS MEJORADAS
# ============================================================================
def plot_2D_with_kde_contours(feh, logAge, labels, means, n_comp, title, outfile, output_dir):
    """Scatter 2D con contornos KDE por componente."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = COLORS[:n_comp]
    # Scatter suave
    for i in range(n_comp):
        mask = labels == i
        ax.scatter(feh[mask], logAge[mask], s=20, alpha=0.4, color=colors[i],
                   label=f'Comp {i+1} (N={mask.sum()})', edgecolors='none', rasterized=True)
    # Contornos KDE
    for i in range(n_comp):
        mask = labels == i
        if mask.sum() < 10:
            continue
        x = feh[mask]
        y = logAge[mask]
        try:
            kde = gaussian_kde(np.vstack([x, y]))
            xgrid = np.linspace(x.min()-0.2, x.max()+0.2, 100)
            ygrid = np.linspace(y.min()-0.1, y.max()+0.1, 100)
            Xg, Yg = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            # niveles: 1σ, 2σ (aproximados por percentiles)
            levels = np.percentile(Z.ravel(), [68, 95])
            ax.contour(Xg, Yg, Z, levels=levels, colors=colors[i], linewidths=2, alpha=0.8)
        except:
            pass
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('log$_{10}$(Age / Gyr)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', outfile)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile} guardada")

def plot_color_color(df, labels, n_comp, title, outfile, output_dir):
    """Diagrama g-i vs i-z coloreado por componente GMM."""
    if not {'gmag', 'imag', 'zmag'}.issubset(df.columns):
        print("⚠️  Faltan columnas de magnitud para color-color.")
        return
    # Calcular colores
    gi = df['gmag'] - df['imag']
    iz = df['imag'] - df['zmag']
    mask = gi.notna() & iz.notna()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = COLORS[:n_comp]
    for i in range(n_comp):
        m = mask & (labels == i)
        ax.scatter(gi[m], iz[m], s=30, alpha=0.6, color=colors[i],
                   label=f'Comp {i+1} (N={m.sum()})', edgecolors='none', rasterized=True)
    ax.set_xlabel('g - i (mag)')
    ax.set_ylabel('i - z (mag)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', outfile)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Diagrama color-color guardado: {base}.pdf")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print("ANÁLISIS GMM COMPLETO (1D + 2D) PARA NGC 5128 GCs")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"GMM_NGC5128_Full_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    print(f"📁 Resultados: {outdir}")

    # 1. Cargar datos -------------------------------------------------
    print("\n[1] Cargando datos...")
    try:
        table = Table.read('out/results.fits', hdu=1)
        df = table.to_pandas()
    except Exception as e:
        print(f"❌ Error leyendo results.fits: {e}")
        return

    df = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)
    Q1, Q3 = df['FeH'].quantile(0.01), df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= Q1-1.5*(Q3-Q1)) & (df['FeH'] <= Q3+1.5*(Q3-Q1))]

    df['Age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0
    df['logAge'] = np.log10(df['Age_Gyr'])

    # Coordenadas y fotometría
    try:
        cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
        ra_col = 'RAJ2000' if 'RAJ2000' in cat.columns else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in cat.columns else 'DEC'
        df['id_str'] = df['id'].astype(str).str.strip()
        cat['T17ID_str'] = cat['T17ID'].astype(str).str.strip()
        # Añadir magnitudes si existen
        mag_cols = ['gmag', 'imag', 'zmag']
        cols_to_merge = [ra_col, dec_col, 'T17ID_str'] + [c for c in mag_cols if c in cat.columns]
        df = pd.merge(df, cat[cols_to_merge], left_on='id_str', right_on='T17ID_str', how='left')
        df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
        print(f"  ✓ Coordenadas y fotometría cargadas.")
    except Exception as e:
        print(f"  ⚠️  Sin catálogo extendido: {e}")

    feh_full = df['FeH'].values
    age_full = df['logAge'].values
    print(f"  Muestra completa: {len(feh_full)} GCs")

    # 2. GMM 1D --------------------------------------------------------
    print("\n[2] GMM 1D (solo metalicidad)...")
    res1d_full, best1d_full = run_gmm_1D(feh_full, label='Full')
    labels_1d = res1d_full[best1d_full]['labels']

    # 3. GMM 2D --------------------------------------------------------
    print("\n[3] GMM 2D (Fe/H + logAge)...")
    res2d_full, best2d_full = run_gmm_2D(feh_full, age_full, label='Full')
    labels_2d = res2d_full[best2d_full]['labels']
    means_2d = res2d_full[best2d_full]['means']

    # 4. Figuras -------------------------------------------------------
    print("\n[4] Generando figuras...")

    # Figura 1D tradicional
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    best = res1d_full[best1d_full]
    ax = axes[0]
    ax.hist(feh_full, bins=30, density=True, alpha=0.3, color='gray', edgecolor='black')
    x = np.linspace(feh_full.min()-0.3, feh_full.max()+0.3, 500)
    pdf_tot = np.zeros_like(x)
    for i in range(best1d_full):
        pdf = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_tot += pdf
        ax.plot(x, pdf, '--', color=COLORS[i])
    ax.plot(x, pdf_tot, 'k-', lw=2.5)
    ax.set_xlabel('[Fe/H]'); ax.set_ylabel('Density')
    ax.set_title(f'GMM 1D (n={best1d_full})')
    # BIC panel
    ax2 = axes[1]
    ns = sorted(res1d_full.keys())
    bics = [res1d_full[n]['bic'] for n in ns]
    ax2.plot(ns, bics, 'o-')
    ax2.set_xlabel('n'); ax2.set_ylabel('BIC')
    ax2.set_title('Model selection 1D')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figures', 'Fig1D_metallicity_BIC.pdf'))
    plt.close()

    # Figura 2D mejorada con contornos KDE
    plot_2D_with_kde_contours(feh_full, age_full, labels_2d, means_2d, best2d_full,
                              title=f'2D GMM (n={best2d_full}) with KDE contours',
                              outfile='Fig2D_FeH_logAge_contours',
                              output_dir=outdir)

    # Diagrama color-color
    plot_color_color(df, labels_2d, best2d_full,
                     title=f'Color-color (n={best2d_full} components)',
                     outfile='Fig_color_color_g_i_vs_i_z',
                     output_dir=outdir)

    # 5. Tablas resumen ------------------------------------------------
    print("\n[5] Guardando tablas...")
    lines = []
    lines.append("RESUMEN GMM NGC 5128")
    lines.append("="*50)
    lines.append(f"\n1D (solo Fe/H): óptimo n={best1d_full}, BIC={res1d_full[best1d_full]['bic']:.1f}")
    for i in range(best1d_full):
        lines.append(f"  Comp{i+1}: [Fe/H]={best['means'][i]:.3f}±{best['stds'][i]:.3f}, f={best['weights'][i]:.2f}")
    lines.append(f"\n2D (Fe/H + logAge): óptimo n={best2d_full}, BIC={res2d_full[best2d_full]['bic']:.1f}")
    means2 = res2d_full[best2d_full]['means']
    stds2 = res2d_full[best2d_full]['stds']
    weights2 = res2d_full[best2d_full]['weights']
    for i in range(best2d_full):
        n_obj = np.sum(labels_2d==i)
        lines.append(f"  Comp{i+1}: [Fe/H]={means2[i,0]:.3f}±{stds2[i,0]:.3f}, logAge={means2[i,1]:.2f}±{stds2[i,1]:.2f}, f={weights2[i]:.2f}, N={n_obj}")
    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print("  ✓ summary.txt")

    # 6. JSON ----------------------------------------------------------
    json_data = {
        '1D': {'best_n': best1d_full, 'bic': res1d_full[best1d_full]['bic'],
               'means': best['means'].tolist(), 'stds': best['stds'].tolist(), 'weights': best['weights'].tolist()},
        '2D': {'best_n': best2d_full, 'bic': res2d_full[best2d_full]['bic'],
               'means': means2.tolist(), 'stds': stds2.tolist(), 'weights': weights2.tolist()}
    }
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    print("  ✓ results.json")

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()
