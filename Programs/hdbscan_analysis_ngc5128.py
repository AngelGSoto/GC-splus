#!/usr/bin/env python3
# gmm_hdbscan_ngc5128_final.py
# Análisis completo de cúmulos globulares de NGC 5128
# Métodos: GMM 1D, GMM 2D (Fe/H + logAge), HDBSCAN, Monte Carlo de errores de edad.
# Figuras: histogramas 1D, scatter 2D con contornos KDE, diagrama color-color.
# Resultados: tablas, JSON, archivo de resumen.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan 
from sklearn.metrics import adjusted_rand_score
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

# Paleta de colores (hasta 7 componentes)
COLORS = ['#2E86AB', '#A23B72', '#C73E1D', '#F18F01', '#6A4C93', '#1982C4', '#4C9F70']

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
# GMM 1D
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
# GMM 2D
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
# HDBSCAN
# ============================================================================
def run_hdbscan(feh, logAge, min_cluster_size=30, min_samples=5):
    """Aplica HDBSCAN en espacio estandarizado y devuelve etiquetas."""
    X_raw = np.column_stack([feh, logAge])
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        cluster_selection_epsilon=0.2, metric='euclidean')
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"HDBSCAN: encontró {n_clusters} clusters (min_cluster_size={min_cluster_size})")
    return labels, n_clusters

# ============================================================================
# Monte Carlo de errores de edad (estabilidad GMM 2D)
# ============================================================================
def monte_carlo_age_errors(df, labels_ref, n_comp=5, n_iter=100):
    """Perturba edades y recalcula GMM 2D con n fijo. Retorna ARI medio."""
    age_err = df['bayes.stellar.age_m_star_err'].values / 1000.0  # Gyr
    feh = df['FeH'].values
    ari_scores = []
    for i in range(n_iter):
        age_pert = df['Age_Gyr'].values + np.random.normal(0, age_err)
        age_pert = np.clip(age_pert, 0.1, None)
        logAge_pert = np.log10(age_pert)
        X_raw = np.column_stack([feh, logAge_pert])
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        gmm = GaussianMixture(n_components=n_comp, random_state=i, n_init=3, max_iter=300)
        gmm.fit(X)
        labels_raw = gmm.predict(X)
        means = scaler.inverse_transform(gmm.means_)
        sort_idx = np.argsort(means[:,0])
        label_map = {old: new for new, old in enumerate(sort_idx)}
        labels_mc = np.array([label_map[l] for l in labels_raw])
        ari_scores.append(adjusted_rand_score(labels_ref, labels_mc))
    return np.mean(ari_scores), np.std(ari_scores)

# ============================================================================
# FIGURAS
# ============================================================================
def plot_1D_summary(feh, results, best_n, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    best = results[best_n]
    ax = axes[0]
    ax.hist(feh, bins=30, density=True, alpha=0.3, color='gray', edgecolor='black')
    x = np.linspace(feh.min()-0.3, feh.max()+0.3, 500)
    pdf_tot = np.zeros_like(x)
    for i in range(best_n):
        pdf = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_tot += pdf
        ax.plot(x, pdf, '--', color=COLORS[i], label=f'Comp {i+1}')
    ax.plot(x, pdf_tot, 'k-', lw=2.5, label='Total')
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('Density')
    ax.set_title(f'GMM 1D (n={best_n})')
    ax.legend()
    ax2 = axes[1]
    ns = sorted(results.keys())
    bics = [results[n]['bic'] for n in ns]
    ax2.plot(ns, bics, 'o-', color='#D55E00')
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('BIC')
    ax2.set_title('Model selection 1D')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'Fig1D_metallicity.pdf'))
    plt.close()

def plot_2D_kde_contours(feh, logAge, labels, n_comp, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_comp):
        mask = labels == i
        ax.scatter(feh[mask], logAge[mask], s=20, alpha=0.3, color=COLORS[i],
                   label=f'Comp {i+1} (N={mask.sum()})', edgecolors='none', rasterized=True)
        if mask.sum() > 10:
            x = feh[mask]
            y = logAge[mask]
            kde = gaussian_kde(np.vstack([x, y]))
            xgrid = np.linspace(x.min()-0.2, x.max()+0.2, 80)
            ygrid = np.linspace(y.min()-0.1, y.max()+0.1, 80)
            Xg, Yg = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            levels = np.percentile(Z.ravel(), [68, 95])
            ax.contour(Xg, Yg, Z, levels=levels, colors=COLORS[i], linewidths=2)
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('log$_{10}$(Age / Gyr)')
    ax.set_title(f'GMM 2D (n={n_comp}) with KDE contours')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'Fig2D_FeH_logAge_kde.pdf'))
    plt.close()

def plot_color_color(df, labels, n_comp, output_dir):
    if not {'gmag', 'imag', 'zmag'}.issubset(df.columns):
        print("⚠️  No hay columnas de color, omitiendo diagrama color-color.")
        return
    gi = df['gmag'] - df['imag']
    iz = df['imag'] - df['zmag']
    mask = gi.notna() & iz.notna()
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n_comp):
        m = mask & (labels == i)
        ax.scatter(gi[m], iz[m], s=30, alpha=0.6, color=COLORS[i],
                   label=f'Comp {i+1} (N={m.sum()})', edgecolors='none', rasterized=True)
    ax.set_xlabel('g - i (mag)')
    ax.set_ylabel('i - z (mag)')
    ax.set_title(f'Color-color diagram (n={n_comp} GMM 2D)')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'Fig_color_color.pdf'))
    plt.close()

def plot_hdbscan_comparison(feh, logAge, labels_hdb, output_dir):
    unique_labels = set(labels_hdb)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters+1))
    for i, lab in enumerate(sorted(unique_labels)):
        mask = labels_hdb == lab
        if lab == -1:
            col = 'gray'
            lbl = f'Noise (N={mask.sum()})'
        else:
            col = colors[i]
            lbl = f'Cluster {lab} (N={mask.sum()})'
        ax.scatter(feh[mask], logAge[mask], s=20, alpha=0.5, color=col, label=lbl, rasterized=True)
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('log$_{10}$(Age / Gyr)')
    ax.set_title(f'HDBSCAN clustering (min_cluster_size=30)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'Fig_HDBSCAN_comparison.pdf'))
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print("ANÁLISIS GMM + HDBSCAN PARA NGC 5128 GCs")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"NGC5128_Full_Analysis_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    print(f"📁 Resultados: {outdir}")

    # 1. Cargar datos -------------------------------------------------
    print("\n[1] Cargando datos...")
    try:
        table = Table.read('out/results.fits', hdu=1)
        df = table.to_pandas()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    df = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)
    Q1, Q3 = df['FeH'].quantile(0.01), df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= Q1-1.5*(Q3-Q1)) & (df['FeH'] <= Q3+1.5*(Q3-Q1))]
    df['Age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0
    df['logAge'] = np.log10(df['Age_Gyr'])

    # Catálogo con coordenadas y fotometría
    try:
        cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
        ra_col = 'RAJ2000' if 'RAJ2000' in cat.columns else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in cat.columns else 'DEC'
        df['id_str'] = df['id'].astype(str).str.strip()
        cat['T17ID_str'] = cat['T17ID'].astype(str).str.strip()
        cols = [ra_col, dec_col, 'T17ID_str'] + [c for c in ['gmag','imag','zmag'] if c in cat.columns]
        df = pd.merge(df, cat[cols], left_on='id_str', right_on='T17ID_str', how='left')
        df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
    except Exception as e:
        print(f"⚠️  Catálogo no disponible: {e}")

    feh_full = df['FeH'].values
    age_full = df['logAge'].values
    print(f"  Muestra completa: {len(feh_full)} GCs")

    # 2. GMM 1D --------------------------------------------------------
    print("\n[2] GMM 1D...")
    res1d, best1d = run_gmm_1D(feh_full, label='Full')
    labels_1d = res1d[best1d]['labels']

    # 3. GMM 2D --------------------------------------------------------
    print("\n[3] GMM 2D...")
    res2d, best2d = run_gmm_2D(feh_full, age_full, label='Full')
    labels_2d = res2d[best2d]['labels']
    means_2d = res2d[best2d]['means']

    # 4. HDBSCAN -------------------------------------------------------
    print("\n[4] HDBSCAN...")
    labels_hdb, n_hdb = run_hdbscan(feh_full, age_full, min_cluster_size=30, min_samples=5)

    # 5. Monte Carlo de errores de edad (para GMM 2D n=best2d) ---------
    print(f"\n[5] Monte Carlo de estabilidad (n={best2d})...")
    ari_mean, ari_std = monte_carlo_age_errors(df, labels_2d, n_comp=best2d, n_iter=50)
    print(f"  ARI medio = {ari_mean:.3f} ± {ari_std:.3f}")

    # 6. Figuras -------------------------------------------------------
    print("\n[6] Generando figuras...")
    plot_1D_summary(feh_full, res1d, best1d, outdir)
    plot_2D_kde_contours(feh_full, age_full, labels_2d, best2d, outdir)
    plot_color_color(df, labels_2d, best2d, outdir)
    plot_hdbscan_comparison(feh_full, age_full, labels_hdb, outdir)

    # 7. Guardar resultados --------------------------------------------
    print("\n[7] Guardando tablas y JSON...")
    lines = ["RESULTADOS NGC 5128 GCs", "="*50]
    lines.append(f"GMM 1D: n={best1d}, BIC={res1d[best1d]['bic']:.1f}")
    lines.append("Componentes 1D:")
    for i in range(best1d):
        lines.append(f"  C{i+1}: [Fe/H]={res1d[best1d]['means'][i]:.3f}±{res1d[best1d]['stds'][i]:.3f}, f={res1d[best1d]['weights'][i]:.2f}")
    lines.append(f"\nGMM 2D: n={best2d}, BIC={res2d[best2d]['bic']:.1f}")
    lines.append("Componentes 2D:")
    for i in range(best2d):
        n_obj = np.sum(labels_2d==i)
        lines.append(f"  C{i+1}: [Fe/H]={means_2d[i,0]:.3f}±{res2d[best2d]['stds'][i,0]:.3f}, logAge={means_2d[i,1]:.2f}±{res2d[best2d]['stds'][i,1]:.2f}, f={res2d[best2d]['weights'][i]:.2f}, N={n_obj}")
    lines.append(f"\nHDBSCAN: {n_hdb} clusters (min_cluster_size=30)")
    lines.append(f"\nEstabilidad GMM 2D (Monte Carlo errores edad): ARI = {ari_mean:.3f}±{ari_std:.3f}")

    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines))

    json_data = {
        '1D': {'best_n': best1d, 'bic': res1d[best1d]['bic'],
               'means': res1d[best1d]['means'].tolist(), 'stds': res1d[best1d]['stds'].tolist(), 'weights': res1d[best1d]['weights'].tolist()},
        '2D': {'best_n': best2d, 'bic': res2d[best2d]['bic'],
               'means': means_2d.tolist(), 'stds': res2d[best2d]['stds'].tolist(), 'weights': res2d[best2d]['weights'].tolist()},
        'HDBSCAN': {'n_clusters': n_hdb, 'labels': labels_hdb.tolist()},
        'MonteCarlo_ARI': {'mean': ari_mean, 'std': ari_std}
    }
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(json_data, f, indent=2)

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*70)

if __name__ == "__main__":
    main()
