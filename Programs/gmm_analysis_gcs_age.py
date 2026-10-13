#!/usr/bin/env python3
# Análisis GMM 2D (Fe/H + logAge) usando número óptimo de componentes según BIC
#
# Genera:
#   Fig 1: Full sample – distribución marginal en [Fe/H] (modelo óptimo)
#   Fig 2: Old GCs – distribución marginal en [Fe/H] (modelo óptimo)
#   Fig 3: Full sample – scatter 2D con elipses (modelo óptimo)
#   Fig 4: Full sample – scatter de gradientes (n componentes óptimo)
#   Fig 5: Full sample – comparación de pendientes (n componentes óptimo)
#   Fig 6: Old GCs – scatter de gradientes (n componentes óptimo)
#   Fig 7: Old GCs – comparación de pendientes (n componentes óptimo)
#   Fig 8: Full sample – binned gradients (n componentes óptimo, multipanel)
#   Tablas completas y JSON con resultados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, linregress
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from astropy.table import Table
import matplotlib.colors as mcolors
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

# Paleta extendida para hasta 6 componentes
COLORS = ['#2E86AB', '#A23B72', '#C73E1D', '#F18F01', '#6A4C93', '#1982C4']

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


def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none',
                       edge_alpha=None, fill_alpha=None, **kwargs):
    """Dibuja una elipse de confianza con alphas separados para borde y relleno."""
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # --- Extraer edgecolor si viene en kwargs, si no hereda del relleno ---
    if 'edgecolor' in kwargs:
        edgecolor = kwargs.pop('edgecolor')
    else:
        edgecolor = facecolor if facecolor != 'none' else 'none'

    # --- Aseguramos que no quede alpha duplicado en kwargs ---
    kwargs.pop('alpha', None)   # eliminamos cualquier alpha suelto

    # Creamos la elipse SIN pasar facecolor/edgecolor (los pondremos después)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)

    # --- Aplicar alpha al relleno ---
    if fill_alpha is not None:
        ellipse.set_facecolor(mcolors.to_rgba(facecolor, alpha=fill_alpha))
    else:
        ellipse.set_facecolor(facecolor)

    # --- Aplicar alpha al borde ---
    if edge_alpha is not None:
        ellipse.set_edgecolor(mcolors.to_rgba(edgecolor, alpha=edge_alpha))
    else:
        ellipse.set_edgecolor(edgecolor)

    # Transformación estándar
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (plt.matplotlib.transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean[0], mean[1]))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
# ============================================================================
# FUNCIONES GMM 2D
# ============================================================================

def validate_gmm_model_2D(X, n_components):
    """Ajusta GMM a datos 2D (ya estandarizados) con múltiples seeds."""
    best_gmm, best_score = None, -np.inf
    for seed in [42, 123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed,
                              n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        if gmm.score(X) > best_score:
            best_score = gmm.score(X)
            best_gmm   = gmm
    return best_gmm

def run_gmm_analysis_2D(feh_data, age_data, label='Full sample'):
    """
    GMM bivariado (Fe/H, log10(Age)) para n=2..5.
    Retorna resultados y mejor n según BIC.
    """
    X_raw = np.column_stack([feh_data, age_data])
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    results = {}
    for n in range(2, 6):
        gmm    = validate_gmm_model_2D(X, n)
        bic    = float(gmm.bic(X))
        aic    = float(gmm.aic(X))
        labels = gmm.predict(X)
        probs  = gmm.predict_proba(X)

        means_std   = gmm.means_
        covs_std    = gmm.covariances_
        means_orig  = scaler.inverse_transform(means_std)
        weights     = gmm.weights_.flatten()

        # Ordenar por [Fe/H] (primera columna)
        sort_idx = np.argsort(means_orig[:, 0])
        means_orig = means_orig[sort_idx]
        weights    = weights[sort_idx]

        # Calidad de membresía
        quality = []
        for i_orig in range(n):
            mask = labels == i_orig
            quality.append(probs[mask, i_orig].mean() if mask.sum() > 0 else 0.0)
        quality = [quality[i] for i in sort_idx]

        # Separaciones solo en [Fe/H]
        separations = []
        for i in range(n - 1):
            var_i = (scaler.scale_[0]**2) * covs_std[sort_idx[i], 0, 0]
            var_j = (scaler.scale_[0]**2) * covs_std[sort_idx[i+1], 0, 0]
            sep = abs(means_orig[i+1, 0] - means_orig[i, 0]) / np.sqrt(var_i + var_j)
            separations.append(float(sep))

        # Reasignar etiquetas ordenadas
        label_map = {int(i_orig): int(i_sorted) for i_sorted, i_orig in enumerate(sort_idx)}
        labels_ordered = np.array([label_map[l] for l in labels])

        # Medias y stds empíricas por componente
        emp_means = np.zeros((n, 2))
        emp_stds  = np.zeros((n, 2))
        for i in range(n):
            mask = labels_ordered == i
            if mask.sum() > 0:
                emp_means[i] = np.mean(X_raw[mask], axis=0)
                emp_stds[i]  = np.std(X_raw[mask], axis=0)
            else:
                emp_means[i] = np.nan
                emp_stds[i]  = np.nan

        results[n] = {
            'bic': bic, 'aic': aic, 'gmm': gmm,
            'means': emp_means,
            'stds': emp_stds,
            'weights': weights,
            'quality': quality,
            'separations': separations,
            'all_gt_2sigma': all(s > 2 for s in separations),
            'min_quality': float(min(quality)),
            'labels': labels_ordered,
            'sort_idx': sort_idx,
            'scaler': scaler,
            'means_orig': means_orig,
            'covs_std': covs_std,
        }

    best_n = min(results.keys(), key=lambda x: results[x]['bic'])

    # Imprimir resumen
    print(f"\n{'='*60}")
    print(f"GMM 2D — {label} (N={len(feh_data)})")
    print(f"{'='*60}")
    print(f"\n{'n':>4} {'BIC':>10} {'ΔBIC':>10} {'AIC':>10} {'Min sep(FeH)':>12} {'All>2σ':>8}")
    print("-"*60)
    for n in sorted(results.keys()):
        delta  = results[n]['bic'] - results[best_n]['bic']
        sep    = f"{min(results[n]['separations']):.2f}σ" if results[n]['separations'] else 'N/A'
        marker = " ← BEST (min BIC)" if n == best_n else ""
        print(f"  {n:>2}  {results[n]['bic']:>10.1f}  {delta:>+10.1f}  "
              f"{results[n]['aic']:>10.1f}  {sep:>12}  "
              f"{str(results[n]['all_gt_2sigma']):>8}{marker}")

    print(f"\nFactores de Bayes (referencia: n={best_n}):")
    for n in sorted(results.keys()):
        if n != best_n:
            delta = results[n]['bic'] - results[best_n]['bic']
            bf    = calculate_bayes_factor(results[best_n]['bic'], results[n]['bic'])
            print(f"  n={best_n} vs n={n}: ΔBIC={delta:+.1f}, "
                  f"BF={bf:.2e} → {interpret_bayes_factor(bf)}")

    print("\n" + "─"*60)
    print("PARÁMETROS COMPLETOS DE TODOS LOS MODELOS (2D):")
    print("─"*60)
    for n in sorted(results.keys()):
        print(f"\nModelo n={n} (BIC={results[n]['bic']:.1f}):")
        for i in range(n):
            n_obj = np.sum(results[n]['labels'] == i)
            feh_mean, age_mean = results[n]['means'][i]
            feh_std,  age_std  = results[n]['stds'][i]
            print(f"  Comp {i+1}: [Fe/H]={feh_mean:.3f}±{feh_std:.3f}, "
                  f"logAge={age_mean:.2f}±{age_std:.2f}, "
                  f"f={results[n]['weights'][i]*100:.1f}%, N={n_obj}")
        if results[n]['separations']:
            sep_str = ', '.join([f'{s:.2f}σ' for s in results[n]['separations']])
            print(f"  Separaciones (Fe/H): {sep_str}")
        print(f"  Min membership prob: {results[n]['min_quality']:.3f}")

    print(f"\n{'─'*60}")
    print(f"Modelo seleccionado (mínimo BIC): n = {best_n}")
    for i in range(best_n):
        n_obj = np.sum(results[best_n]['labels'] == i)
        feh_mean, age_mean = results[best_n]['means'][i]
        feh_std,  age_std  = results[best_n]['stds'][i]
        print(f"  Comp {i+1}: [Fe/H]={feh_mean:.3f}±{feh_std:.3f}, "
              f"logAge={age_mean:.2f}±{age_std:.2f}, "
              f"f={results[best_n]['weights'][i]*100:.1f}%, N={n_obj}")
    print("─"*60)

    return results, best_n

def get_labels_from_best_model(results_dict, best_n):
    """Retorna labels, means, stds, weights para el modelo óptimo (best_n)."""
    best = results_dict[best_n]
    labels = best['labels']
    means = best['means']      # shape (n_comp, 2)
    stds = best['stds']
    weights = best['weights']
    separations = best['separations']
    bic = best['bic']
    aic = best['aic']
    scaler = best['scaler']
    gmm = best['gmm']

    print(f"\n[Modelo óptimo n={best_n}]")
    print(f"  BIC={bic:.1f}, AIC={aic:.1f}")
    if separations:
        print(f"  Separaciones (Fe/H): {[f'{s:.2f}σ' for s in separations]}")
    for i in range(best_n):
        n_obj = np.sum(labels == i)
        print(f"  Comp {i+1}: [Fe/H]={means[i,0]:.3f}±{stds[i,0]:.3f}, "
              f"logAge={means[i,1]:.2f}±{stds[i,1]:.2f}, "
              f"f={weights[i]*100:.1f}%, N={n_obj}")
    return labels, means, stds, weights, separations, bic, aic, scaler, gmm

# ============================================================================
# FUNCIONES DE COORDENADAS Y GRADIENTES
# ============================================================================

def calculate_projected_radii(df):
    if 'RA' not in df.columns or 'DEC' not in df.columns:
        return None, None
    valid = df['RA'].notna() & df['DEC'].notna()
    if not valid.any():
        return None, None
    coords = SkyCoord(ra=df.loc[valid, 'RA'].values*u.degree,
                      dec=df.loc[valid, 'DEC'].values*u.degree)
    center = SkyCoord(ra=RA_CENTER*u.degree, dec=DEC_CENTER*u.degree)
    radii  = DISTANCE_MPC * 1000 * np.pi * center.separation(coords).degree / 180.0
    return radii, valid

def compute_gradient_results_ncomp(feh_v, labels_v, log_r, n_comp):
    """Calcula gradiente radial para cada una de las n_comp poblaciones."""
    gradient_results = {}
    for i in range(n_comp):
        mask = labels_v == i
        if mask.sum() < 10:
            continue
        slope, intercept, r_val, p_val, se = linregress(log_r[mask], feh_v[mask])
        gradient_results[f'Comp{i+1}'] = {
            'slope': float(slope), 'slope_err': float(se),
            'intercept': float(intercept),
            'r_value': float(r_val), 'p_value': float(p_val),
            'n_points': int(mask.sum()),
            'significant': bool(p_val < 0.05),
        }
    return gradient_results

# ============================================================================
# FUNCIONES DE FIGURAS
# ============================================================================

def plot_metallicity_marginal(feh_data, results_dict, best_n, title, outfile, output_dir):
    """Distribución marginal en [Fe/H] del GMM 2D óptimo."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    best = results_dict[best_n]
    colors = COLORS[:best_n]

    ax.hist(feh_data, bins=30, density=True, alpha=0.3,
            color='lightgray', edgecolor='black', linewidth=0.8,
            label=f'N={len(feh_data)}')

    x = np.linspace(feh_data.min()-0.3, feh_data.max()+0.3, 1000)
    pdf_total = np.zeros_like(x)
    for i in range(best_n):
        feh_mean = best['means'][i, 0]
        feh_std  = best['stds'][i, 0]
        pdf_c = best['weights'][i] * norm.pdf(x, feh_mean, feh_std)
        pdf_total += pdf_c
        ax.plot(x, pdf_c, '--', linewidth=2.5, alpha=0.8, color=colors[i])
    ax.plot(x, pdf_total, 'k-', linewidth=3.5, label='GMM fit (marginal)')
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{title} (GMM 2D, n={best_n})')
    ax.legend()
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    base = os.path.join(figures_dir, outfile)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile} guardada: {base}.pdf")

def plot_2D_scatter_with_ellipses(feh, logAge, labels, gmm, scaler, title, outfile, output_dir,
                                  minx=None, maxx=None, miny=None, maxy=None):
    """Figura 2D: Fe/H vs logAge con elipses de confianza (1σ, 2σ)."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    n_comp = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = COLORS[:n_comp]

    for i in range(n_comp):
        mask = labels == i
        ax.scatter(feh[mask], logAge[mask], s=45, alpha=0.6,
                   color=colors[i], label=f'Comp {i+1} (N={mask.sum()})',
                   edgecolors='black', zorder=3)

    means_std = gmm.means_
    covs_std = gmm.covariances_
    sort_idx = np.argsort(scaler.inverse_transform(means_std)[:, 0])
    for i, idx in enumerate(sort_idx):
        mean_std = means_std[idx]
        cov_std = covs_std[idx]
        scale = scaler.scale_
        cov_orig = np.diag(scale) @ cov_std @ np.diag(scale)
        mean_orig = scaler.inverse_transform(mean_std.reshape(1, -1))[0]

        confidence_ellipse(mean_orig, cov_orig, ax, n_std=1.0,
                           edgecolor=colors[i], edge_alpha=0.9,   # borde casi opaco
                           facecolor=colors[i], fill_alpha=0.2,   # relleno semi‑transparente
                           linewidth=2, zorder=4)
        confidence_ellipse(mean_orig, cov_orig, ax, n_std=2.0,
                           edgecolor=colors[i], edge_alpha=0.9,
                           facecolor=colors[i], fill_alpha=0.1,
                           linewidth=1.5, linestyle='--', zorder=4)

    ax.set_xlabel('[Fe/H] (dex)', fontsize=24)
    ax.set_ylabel('log$_{10}$(Age / Gyr)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
        # --- Límites solo si se especifican (None por defecto) ---
    if minx is not None and maxx is not None:
        ax.set_xlim(minx, maxx)
    if miny is not None and maxy is not None:
        ax.set_ylim(miny, maxy)
    #ax.set_title(title)
    ax.legend(loc='lower left', 
              fontsize=17, framealpha=0.9, 
              title_fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    base = os.path.join(figures_dir, outfile)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura 2D guardada: {base}.pdf")



def plot_gradient_scatter_ncomp(df_sp, labels_sp, radii, n_comp, title_str, outfile_name, output_dir):
    """Scatter de gradientes para n componentes."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    valid = np.isfinite(radii) & np.isfinite(df_sp['FeH'].values)
    radii_v = radii[valid]
    feh_v = df_sp['FeH'].values[valid]
    labels_v = labels_sp[valid]
    log_r = np.log10(radii_v)

    gradient_results = compute_gradient_results_ncomp(feh_v, labels_v, log_r, n_comp)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = COLORS[:n_comp]

    for i in range(n_comp):
        mask = labels_v == i
        if mask.sum() < 10:
            continue
        ax.scatter(log_r[mask], feh_v[mask], s=60, alpha=0.55, color=colors[i],
                   label=f'Comp {i+1} (N={mask.sum()})',
                   edgecolors='black', linewidth=0.4, zorder=3)
        comp_key = f'Comp{i+1}'
        if comp_key in gradient_results:
            gr = gradient_results[comp_key]
            x_fit = np.linspace(log_r[mask].min(), log_r[mask].max(), 200)
            y_fit = gr['intercept'] + gr['slope'] * x_fit

            # --- CORRECCIÓN DE LA BANDA DE CONFIANZA ---
            n_p = mask.sum()
            x_mean = log_r[mask].mean()
            Sxx = np.sum((log_r[mask] - x_mean)**2)
            # Error estándar residual (s) a partir de SE(β)
            s_resid = gr['slope_err'] * np.sqrt(Sxx)
            # Banda de confianza de la recta ajustada
            y_err = s_resid * np.sqrt(1/n_p + (x_fit - x_mean)**2 / Sxx)
            # --------------------------------------------

            sig_str = '*' if gr['p_value'] < 0.05 else '(ns)'
            ax.plot(x_fit, y_fit, color=colors[i], linewidth=3, alpha=0.9, zorder=4,
                    label=f"Comp{i+1}: {gr['slope']:.3f}±{gr['slope_err']:.3f} {sig_str}")
            ax.fill_between(x_fit, y_fit-y_err, y_fit+y_err, color=colors[i], alpha=0.15, zorder=2)

    ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=24)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=24)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=15, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ymin = np.nanmin(feh_v); ymax = np.nanmax(feh_v)
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.39*yr, ymax + 0.18*yr)
    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")
    return gradient_results


def plot_slope_comparison_ncomp(gradient_results, n_comp, title_str, outfile_name, output_dir):
    """Barras de pendientes para n componentes."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    comps = list(gradient_results.keys())
    slps = [gradient_results[c]['slope'] for c in comps]
    errs = [gradient_results[c]['slope_err'] for c in comps]
    x_pos = np.arange(len(comps))
    colors = COLORS[:len(comps)]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x_pos, slps, yerr=errs, color=colors, alpha=0.75, capsize=8,
           edgecolor='black', linewidth=1.5,
           error_kw={'elinewidth': 2.5, 'capthick': 2.5})
    for i, c in enumerate(comps):
        p_val = gradient_results[c]['p_value']
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        yoff = errs[i] + 0.004 if slps[i] >= 0 else -errs[i] - 0.010
        ax.text(i, slps[i] + yoff, sig, ha='center', fontsize=18, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)
    xticklabels = [f"{c}\n{slps[i]:.3f}±{errs[i]:.3f}" for i,c in enumerate(comps)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=13)
    ax.set_ylabel('Gradient Slope (dex dex$^{-1}$)', fontsize=22)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")

def plot_binned_gradients_ncomp(df_sp, labels_sp, radii_kpc, n_comp, output_dir):
    """Binned gradients para las n componentes (multipanel)."""
    figures_dir = os.path.join(output_dir, 'figures')
    valid = np.isfinite(radii_kpc) & np.isfinite(df_sp['FeH'].values)
    radii_v = radii_kpc[valid]
    feh_v = df_sp['FeH'].values[valid]
    labels_v = labels_sp[valid]
    log_r = np.log10(radii_v)

    n_cols = min(3, n_comp)
    n_rows = int(np.ceil(n_comp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharey=True)
    if n_comp == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    n_bins = 6
    for idx, i in enumerate(range(n_comp)):
        ax = axes[idx]
        mask_p = labels_v == i
        feh_p = feh_v[mask_p]
        logr_p = log_r[mask_p]
        color = COLORS[i % len(COLORS)]
        if len(feh_p) < 20:
            ax.axis('off')
            continue
        bin_edges = np.percentile(logr_p, np.linspace(0, 100, n_bins+1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        means_b = np.full(n_bins, np.nan)
        sems_b = np.full(n_bins, np.nan)
        ns_b = np.zeros(n_bins, dtype=int)
        for k in range(n_bins):
            mask_b = (logr_p >= bin_edges[k]) & (logr_p < bin_edges[k+1])
            if mask_b.sum() > 3:
                means_b[k] = np.mean(feh_p[mask_b])
                sems_b[k] = np.std(feh_p[mask_b]) / np.sqrt(mask_b.sum())
                ns_b[k] = mask_b.sum()
        valid_b = ~np.isnan(means_b)
        ax.errorbar(bin_centers[valid_b], means_b[valid_b], yerr=sems_b[valid_b],
                    fmt='o', color=color, ecolor=color, elinewidth=2.5,
                    capsize=5, capthick=2, markersize=9, markeredgewidth=1.5,
                    markeredgecolor='black', zorder=5)
        for k in range(n_bins):
            if ns_b[k] > 0:
                ax.annotate(f'N={ns_b[k]}', xy=(bin_centers[k], means_b[k]),
                            xytext=(0, 12), textcoords='offset points',
                            ha='center', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        slope, intercept, r_val, p_val, se = linregress(logr_p, feh_p)
        x_fit = np.linspace(logr_p.min(), logr_p.max(), 100)
        ax.plot(x_fit, intercept + slope*x_fit, '--', color=color, linewidth=2.5, alpha=0.8,
                label=f'{slope:.3f}±{se:.3f} dex/dex')
        ax.set_title(f'Comp {i+1} (N={len(feh_p)})', fontsize=16)
        ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=18)
        if idx % n_cols == 0:
            ax.set_ylabel('Mean [Fe/H] (dex)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='--')
        sig = '*' if p_val < 0.05 else 'ns'
        ax.text(0.05, 0.05, f'p={p_val:.4f} {sig}', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ocultar ejes sobrantes
    for idx in range(n_comp, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Radial Metallicity Gradients — Binned Analysis (2D GMM, n={n_comp})',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    base = os.path.join(figures_dir, f'Fig_Binned_Gradients_n{n_comp}_2D')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Binned gradients guardados: {base}.pdf")

# ============================================================================
# TABLAS Y JSON
# ============================================================================

def print_paper_table_2D_optimal(results_full, best_n_full, results_old, best_n_old,
                                 labels_full_opt, means_full_opt, stds_full_opt, weights_full_opt,
                                 labels_old_opt, means_old_opt, stds_old_opt, weights_old_opt,
                                 grad_all, grad_old, output_dir):
    """Genera tabla de resultados para los modelos óptimos."""
    lines = []
    lines.append("="*70)
    lines.append("TABLA COMPLETA DE RESULTADOS GMM 2D (MODELOS ÓPTIMOS)")
    lines.append("="*70)

    lines.append("\n── TABLA 1: Comparación de modelos GMM 2D ──")
    lines.append(f"{'n':>4} {'BIC (all)':>12} {'ΔBIC (all)':>12} "
                 f"{'BIC (old)':>12} {'ΔBIC (old)':>12}")
    lines.append("-"*64)
    for n in range(2, 6):
        d_all   = results_full[n]['bic'] - results_full[best_n_full]['bic']
        d_old   = results_old[n]['bic']  - results_old[best_n_old]['bic']
        lines.append(f"  {n:>2}  {results_full[n]['bic']:>12.1f}  {d_all:>+12.1f}  "
                     f"{results_old[n]['bic']:>12.1f}  {d_old:>+12.1f}")

    lines.append(f"\nModelo óptimo muestra completa: n={best_n_full}")
    lines.append(f"Modelo óptimo GCs viejos: n={best_n_old}")

    lines.append(f"\n── TABLA 2: Parámetros modelo óptimo muestra completa (n={best_n_full}) ──")
    lines.append(f"{'Comp':>6} {'[Fe/H]':>10} {'σ_FeH':>8} {'logAge':>8} {'σ_logAge':>8} {'Fraction':>10} {'N':>6}")
    lines.append("-"*70)
    for i in range(best_n_full):
        n_obj = int(np.sum(labels_full_opt == i))
        lines.append(f"  {i+1:>4}  {means_full_opt[i,0]:>10.3f}  {stds_full_opt[i,0]:>8.3f}  "
                     f"{means_full_opt[i,1]:>8.2f}  {stds_full_opt[i,1]:>8.2f}  "
                     f"{weights_full_opt[i]:>10.3f}  {n_obj:>6}")

    lines.append(f"\n── TABLA 3: Parámetros modelo óptimo GCs viejos (n={best_n_old}) ──")
    lines.append(f"{'Comp':>6} {'[Fe/H]':>10} {'σ_FeH':>8} {'logAge':>8} {'σ_logAge':>8} {'Fraction':>10} {'N':>6}")
    lines.append("-"*70)
    for i in range(best_n_old):
        n_obj = int(np.sum(labels_old_opt == i))
        lines.append(f"  {i+1:>4}  {means_old_opt[i,0]:>10.3f}  {stds_old_opt[i,0]:>8.3f}  "
                     f"{means_old_opt[i,1]:>8.2f}  {stds_old_opt[i,1]:>8.2f}  "
                     f"{weights_old_opt[i]:>10.3f}  {n_obj:>6}")

    if grad_all:
        lines.append(f"\n── TABLA 4: Gradientes radiales — muestra completa (n={best_n_full}) ──")
        lines.append(f"{'Comp':>6} {'Slope':>10} {'±err':>8} {'r':>8} {'p-value':>10} {'N':>6} {'Sig.':>6}")
        lines.append("-"*56)
        for comp, gr in grad_all.items():
            lines.append(f"  {comp:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                         f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  "
                         f"{gr['n_points']:>6}  {'Yes' if gr['significant'] else 'No':>6}")

    if grad_old:
        lines.append(f"\n── TABLA 5: Gradientes radiales — GCs viejos (n={best_n_old}) ──")
        lines.append(f"{'Comp':>6} {'Slope':>10} {'±err':>8} {'r':>8} {'p-value':>10} {'N':>6} {'Sig.':>6}")
        lines.append("-"*56)
        for comp, gr in grad_old.items():
            lines.append(f"  {comp:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                         f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  "
                         f"{gr['n_points']:>6}  {'Yes' if gr['significant'] else 'No':>6}")

    lines.append("\n" + "="*70)
    output = "\n".join(lines)
    print(output)
    txt_path = os.path.join(output_dir, 'paper_tables_optimal_2D.txt')
    with open(txt_path, 'w') as f:
        f.write(output)
    print(f"\n✓ Tablas guardadas: {txt_path}")
    return output

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ANÁLISIS GMM 2D (Fe/H + logAge) — SELECCIÓN ÓPTIMA POR BIC")
    print("NGC 5128 Globular Cluster System")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = f"GMM_2D_Optimal_BIC_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    print(f"📁 Resultados: {outdir}")

    # ── 1. Cargar datos ──────────────────────────────────────────────
    print("\n[1] Cargando datos...")
    try:
        table = Table.read('out/results.fits', hdu=1)
        df    = table.to_pandas()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    df = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)

    Q1 = df['FeH'].quantile(0.01)
    Q3 = df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= Q1-1.5*(Q3-Q1)) & (df['FeH'] <= Q3+1.5*(Q3-Q1))]

    age_col = 'bayes.stellar.age_m_star'
    df['Age_Gyr'] = df[age_col] / 1000.0
    df['logAge'] = np.log10(df['Age_Gyr'])

    # Coordenadas
    try:
        catalog = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
        ra_col = 'RAJ2000' if 'RAJ2000' in catalog.columns else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in catalog.columns else 'DEC'
        df['id_str'] = df['id'].astype(str).str.strip()
        catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
        df = pd.merge(df, catalog[[ra_col, dec_col, 'T17ID_str']],
                      left_on='id_str', right_on='T17ID_str', how='left')
        df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
        print(f"  ✓ Coordenadas: {df['RA'].notna().sum()} objetos")
    except Exception as e:
        print(f"  ⚠️  Sin coordenadas: {e}")

    feh_full = df['FeH'].values
    age_full = df['logAge'].values
    print(f"  ✓ Muestra completa: {len(feh_full)} GCs")

    df_old = df[df['Age_Gyr'] > 8.0].copy().reset_index(drop=True)
    feh_old = df_old['FeH'].values
    age_old = df_old['logAge'].values
    print(f"  ✓ GCs viejos >8 Gyr: {len(feh_old)} ({len(feh_old)/len(feh_full)*100:.1f}%)")

    # ── 2. GMM 2D ───────────────────────────────────────────────────────
    print("\n[2] GMM 2D — Muestra completa...")
    results_full, best_n_full = run_gmm_analysis_2D(feh_full, age_full, 'All GCs')

    print("\n[3] GMM 2D — GCs viejos (>8 Gyr)...")
    results_old, best_n_old = run_gmm_analysis_2D(feh_old, age_old, 'Old GCs >8 Gyr')

    # ── 3. Clasificación usando modelo óptimo ──────────────────────────
    print(f"\n[4] Clasificación con modelo óptimo n={best_n_full} — 884 GCs:")
    (labels_full_opt, means_full_opt, stds_full_opt, weights_full_opt,
     seps_full_opt, bic_full_opt, aic_full_opt, scaler_full, gmm_full) = get_labels_from_best_model(results_full, best_n_full)

    print(f"\n[4b] Clasificación con modelo óptimo n={best_n_old} — 720 GCs viejos:")
    (labels_old_opt, means_old_opt, stds_old_opt, weights_old_opt,
     seps_old_opt, bic_old_opt, aic_old_opt, scaler_old, gmm_old) = get_labels_from_best_model(results_old, best_n_old)

    # ── 4. Radios ─────────────────────────────────────────────────────
    print("\n[5] Calculando radios proyectados...")
    radii_full, valid_full = calculate_projected_radii(df)
    if radii_full is not None:
        print(f"  ✓ Rango: {radii_full.min():.1f} – {radii_full.max():.0f} kpc")
        df_sp_full = df[valid_full].copy().reset_index(drop=True)
        labels_sp_full = labels_full_opt[valid_full]
    else:
        print("  ⚠️  Sin coordenadas")
        df_sp_full = df.copy().reset_index(drop=True)
        labels_sp_full = labels_full_opt

    radii_old, valid_old = calculate_projected_radii(df_old)
    if radii_old is not None:
        df_sp_old = df_old[valid_old].copy().reset_index(drop=True)
        labels_sp_old = labels_old_opt[valid_old]
    else:
        df_sp_old = df_old.copy().reset_index(drop=True)
        labels_sp_old = labels_old_opt

    # ── 5. Figuras ────────────────────────────────────────────────────
    print("\n[6] Generando figuras...")

    # Figuras marginales (1D)
    plot_metallicity_marginal(feh_full, results_full, best_n_full,
                              title='All GCs – Marginal [Fe/H]',
                              outfile='Fig1_FullSample_Metallicity_Marginal',
                              output_dir=outdir)
    plot_metallicity_marginal(feh_old, results_old, best_n_old,
                              title='Old GCs (>8 Gyr) – Marginal [Fe/H]',
                              outfile='Fig2_OldGCs_Metallicity_Marginal',
                              output_dir=outdir)

    # Figura 2D (scatter + elipses) para la muestra completa
    plot_2D_scatter_with_ellipses(feh_full, age_full, labels_full_opt, gmm_full, scaler_full,
                                  title=f'All GCs: GMM 2D (n={best_n_full}) with 1σ and 2σ ellipses',
                                  outfile='Fig3_FullSample_2D_GMM_ellipses',
                                  output_dir=outdir)

    # Figura 2D (scatter + elipses) para la muestra vieja (mayores que 8Gyr)
    plot_2D_scatter_with_ellipses(feh_old, age_old, labels_old_opt, gmm_old, scaler_old,
                                  title=f'Old GCs: GMM 2D (n={best_n_full}) with 1σ and 2σ ellipses',
                                  outfile='Fig3_OldSample_2D_GMM_ellipses',
                                  output_dir=outdir, miny=0.83, maxy=1.07)

    # Gradientes
    grad_all = {}
    grad_old = {}
    if radii_full is not None:
        grad_all = plot_gradient_scatter_ncomp(
            df_sp_full, labels_sp_full, radii_full, n_comp=best_n_full,
            title_str=f'Radial Gradients — All GCs (2D GMM, n={best_n_full})',
            outfile_name='Fig4_FullSample_Gradient_Scatter',
            output_dir=outdir)
        plot_slope_comparison_ncomp(
            grad_all, n_comp=best_n_full,
            title_str=f'Slope Comparison — All GCs (n={best_n_full})',
            outfile_name='Fig5_FullSample_Slope_Comparison',
            output_dir=outdir)
        plot_binned_gradients_ncomp(df_sp_full, labels_sp_full, radii_full,
                                    n_comp=best_n_full, output_dir=outdir)

    if radii_old is not None:
        grad_old = plot_gradient_scatter_ncomp(
            df_sp_old, labels_sp_old, radii_old, n_comp=best_n_old,
            title_str=f'Radial Gradients — Old GCs (2D GMM, n={best_n_old})',
            outfile_name='Fig6_OldGCs_Gradient_Scatter',
            output_dir=outdir)
        plot_slope_comparison_ncomp(
            grad_old, n_comp=best_n_old,
            title_str=f'Slope Comparison — Old GCs (n={best_n_old})',
            outfile_name='Fig7_OldGCs_Slope_Comparison',
            output_dir=outdir)
        plot_binned_gradients_ncomp(df_sp_old, labels_sp_old, radii_old,
                                    n_comp=best_n_old, output_dir=outdir)

    # ── 6. Tablas ─────────────────────────────────────────────────────
    print("\n[7] Generando tablas...")
    print_paper_table_2D_optimal(
        results_full, best_n_full, results_old, best_n_old,
        labels_full_opt, means_full_opt, stds_full_opt, weights_full_opt,
        labels_old_opt, means_old_opt, stds_old_opt, weights_old_opt,
        grad_all, grad_old, outdir)

    # ── 7. JSON ───────────────────────────────────────────────────────
    json_data = {
        'version': 'gmm_2D_optimal_BIC',
        'date': datetime.datetime.now().isoformat(),
        'full_sample': {
            'best_n': best_n_full,
            'bic': float(bic_full_opt),
            'aic': float(aic_full_opt),
            'components': []
        },
        'old_gcs': {
            'best_n': best_n_old,
            'bic': float(bic_old_opt),
            'aic': float(aic_old_opt),
            'components': []
        },
        'gradients_all': grad_all,
        'gradients_old': grad_old,
    }

    for i in range(best_n_full):
        comp = {
            'index': i+1,
            'feh_mean': float(means_full_opt[i,0]),
            'feh_std': float(stds_full_opt[i,0]),
            'logAge_mean': float(means_full_opt[i,1]),
            'logAge_std': float(stds_full_opt[i,1]),
            'weight': float(weights_full_opt[i]),
            'n_objects': int(np.sum(labels_full_opt == i))
        }
        json_data['full_sample']['components'].append(comp)

    for i in range(best_n_old):
        comp = {
            'index': i+1,
            'feh_mean': float(means_old_opt[i,0]),
            'feh_std': float(stds_old_opt[i,0]),
            'logAge_mean': float(means_old_opt[i,1]),
            'logAge_std': float(stds_old_opt[i,1]),
            'weight': float(weights_old_opt[i]),
            'n_objects': int(np.sum(labels_old_opt == i))
        }
        json_data['old_gcs']['components'].append(comp)

    json_path = os.path.join(outdir, 'gmm_2D_optimal_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON: {json_path}")

    print("\n" + "="*70)
    print("✅ ANÁLISIS GMM 2D CON SELECCIÓN ÓPTIMA COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()
