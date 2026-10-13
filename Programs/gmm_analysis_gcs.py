#!/usr/bin/env python3
# gmm_analysis_final_paper.py
# Análisis GMM completo para el paper — VERSIÓN FINAL (separada)
#
# Genera:
#   Fig 1: Full sample – metallicity distribution + BIC/AIC
#   Fig 2: Old GCs – metallicity distribution + BIC/AIC
#   Fig 3: Full sample – gradient scatter (n=3 forced)
#   Fig 4: Full sample – slope comparison bars (n=3 forced)
#   Fig 5: Old GCs – gradient scatter (n=3 natural)
#   Fig 6: Old GCs – slope comparison bars (n=3 natural)
#   Fig 7: Binned gradients (full sample, 3 panels)
#   Tabla: resultados completos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, linregress
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
# CONFIGURACIÓN DE ESTILO (idéntico al paper original)
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

COLORS_3 = ['#2E86AB', '#A23B72', '#C73E1D']
COLORS_4 = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

POP_NAMES_3 = ['Metal-Poor', 'Intermediate', 'Metal-Rich']
POP_SHORT_3 = ['MP', 'Int', 'MR']

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

def validate_gmm_model(feh_data, n_components):
    X = feh_data.reshape(-1, 1)
    best_gmm, best_score = None, -np.inf
    for seed in [42, 123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed,
                              n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        if gmm.score(X) > best_score:
            best_score = gmm.score(X)
            best_gmm   = gmm
    return best_gmm

def run_gmm_analysis(feh_data, label='Full sample'):
    """GMM para n=2..5, selección por BIC mínimo."""
    X       = feh_data.reshape(-1, 1)
    results = {}

    for n in range(2, 6):
        gmm    = validate_gmm_model(feh_data, n)
        bic    = float(gmm.bic(X))
        aic    = float(gmm.aic(X))
        labels = gmm.predict(X)
        probs  = gmm.predict_proba(X)

        means   = gmm.means_.flatten()
        stds    = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()

        sort_idx = np.argsort(means)
        means    = means[sort_idx]
        stds     = stds[sort_idx]
        weights  = weights[sort_idx]

        quality = []
        for i_orig in range(n):
            mask = labels == i_orig
            quality.append(probs[mask, i_orig].mean() if mask.sum() > 0 else 0.0)
        quality = [quality[i] for i in sort_idx]

        separations = []
        for i in range(n - 1):
            sep = abs(means[i+1] - means[i]) / np.sqrt(stds[i]**2 + stds[i+1]**2)
            separations.append(float(sep))

        label_map      = {int(i_orig): int(i_sorted)
                          for i_sorted, i_orig in enumerate(sort_idx)}
        labels_ordered = np.array([label_map[l] for l in labels])

        results[n] = {
            'bic': bic, 'aic': aic, 'gmm': gmm,
            'means': means, 'stds': stds, 'weights': weights,
            'quality': quality, 'separations': separations,
            'all_gt_2sigma': all(s > 2 for s in separations),
            'min_quality': float(min(quality)),
            'labels': labels_ordered,
            'sort_idx': sort_idx,
        }

    best_n = min(results.keys(), key=lambda x: results[x]['bic'])

    print(f"\n{'='*60}")
    print(f"GMM — {label} (N={len(feh_data)})")
    print(f"{'='*60}")
    print(f"\n{'n':>4} {'BIC':>10} {'ΔBIC':>10} {'AIC':>10} {'Min sep':>10} {'All>2σ':>8}")
    print("-"*56)
    for n in sorted(results.keys()):
        delta  = results[n]['bic'] - results[best_n]['bic']
        sep    = f"{min(results[n]['separations']):.2f}σ"
        marker = " ← BEST (min BIC)" if n == best_n else ""
        print(f"  {n:>2}  {results[n]['bic']:>10.1f}  {delta:>+10.1f}  "
              f"{results[n]['aic']:>10.1f}  {sep:>10}  "
              f"{str(results[n]['all_gt_2sigma']):>8}{marker}")

    print(f"\nFactores de Bayes (referencia: n={best_n}):")
    for n in sorted(results.keys()):
        if n != best_n:
            delta = results[n]['bic'] - results[best_n]['bic']
            bf    = calculate_bayes_factor(results[best_n]['bic'], results[n]['bic'])
            print(f"  n={best_n} vs n={n}: ΔBIC={delta:+.1f}, "
                  f"BF={bf:.2e} → {interpret_bayes_factor(bf)}")

    # ======================================================================
    # NUEVO: Imprimir los parámetros de TODOS los modelos (n=2..5)
    # ======================================================================
    print("\n" + "─"*60)
    print("PARÁMETROS COMPLETOS DE TODOS LOS MODELOS:")
    print("─"*60)
    for n in sorted(results.keys()):
        print(f"\nModelo n={n} (BIC={results[n]['bic']:.1f}):")
        for i in range(n):
            n_obj = np.sum(results[n]['labels'] == i)
            print(f"  Comp {i+1}: [Fe/H]={results[n]['means'][i]:.3f}±{results[n]['stds'][i]:.3f}, "
                  f"f={results[n]['weights'][i]*100:.1f}%, N={n_obj}")
        if results[n]['separations']:
            sep_str = ', '.join([f'{s:.2f}σ' for s in results[n]['separations']])
            print(f"  Separaciones: {sep_str}")
        print(f"  Min membership prob: {results[n]['min_quality']:.3f}")

    # También imprimir el resumen del mejor modelo (opcional, ya está arriba)
    print(f"\n{'─'*60}")
    print(f"Modelo seleccionado (mínimo BIC): n = {best_n}")
    for i in range(best_n):
        n_obj = np.sum(results[best_n]['labels'] == i)
        print(f"  Comp {i+1}: [Fe/H]={results[best_n]['means'][i]:.3f}±"
              f"{results[best_n]['stds'][i]:.3f}, "
              f"f={results[best_n]['weights'][i]*100:.1f}%, N={n_obj}")
    print("─"*60)

    return results, best_n

def get_labels_forced_n3(feh_data):
    """Clasifica feh_data en n=3 componentes (modelo adoptado)."""
    gmm  = validate_gmm_model(feh_data, 3)
    X    = feh_data.reshape(-1, 1)

    means   = gmm.means_.flatten()
    stds    = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    sort_idx = np.argsort(means)
    means    = means[sort_idx]
    stds     = stds[sort_idx]
    weights  = weights[sort_idx]

    probs  = gmm.predict_proba(X)
    labels = gmm.predict(X)

    quality = []
    for i_orig in range(3):
        mask = labels == i_orig
        quality.append(probs[mask, i_orig].mean() if mask.sum() > 0 else 0.0)
    quality = [quality[i] for i in sort_idx]

    separations = []
    for i in range(2):
        sep = abs(means[i+1] - means[i]) / np.sqrt(stds[i]**2 + stds[i+1]**2)
        separations.append(float(sep))

    label_map      = {int(i_orig): int(i_sorted)
                      for i_sorted, i_orig in enumerate(sort_idx)}
    labels_ordered = np.array([label_map[l] for l in labels])

    bic = float(gmm.bic(X))
    aic = float(gmm.aic(X))

    print(f"  BIC={bic:.1f}, AIC={aic:.1f}")
    print(f"  Separaciones: {[f'{s:.2f}σ' for s in separations]}")
    for i in range(3):
        n_obj = np.sum(labels_ordered == i)
        print(f"  {POP_NAMES_3[i]}: [Fe/H]={means[i]:.3f}±{stds[i]:.3f}, "
              f"f={weights[i]*100:.1f}%, N={n_obj}")

    return labels_ordered, means, stds, weights, separations, quality, bic, aic

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

def compute_gradient_results(feh_v, labels_v, log_r):
    """Calcula gradientes radiales para cada población. Devuelve dict."""
    gradient_results = {}
    for i in range(3):
        mask = labels_v == i
        if mask.sum() < 10:
            continue
        slope, intercept, r_val, p_val, se = linregress(log_r[mask], feh_v[mask])
        gradient_results[POP_SHORT_3[i]] = {
            'name': POP_NAMES_3[i],
            'slope': float(slope), 'slope_err': float(se),
            'intercept': float(intercept),
            'r_value': float(r_val), 'p_value': float(p_val),
            'n_points': int(mask.sum()),
            'significant': bool(p_val < 0.05),
        }
    return gradient_results

# ============================================================================
# FIG 1: Full sample – metallicity distribution + BIC/AIC
# ============================================================================

def plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full,
                                         output_dir):
    """Fig 1: Full sample. Left: metallicity distribution (n=4). Right: BIC/AIC."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel izquierdo: distribución
    ax1 = axes[0]
    best = results_full[best_n_full]
    colors = COLORS_4
    pop_short = ['MP', 'IL', 'IH', 'MR']

    ax1.hist(feh_full, bins=30, density=True, alpha=0.3,
             color='lightgray', edgecolor='black', linewidth=0.8,
             label=f'N={len(feh_full)}')

    x = np.linspace(feh_full.min()-0.3, feh_full.max()+0.3, 1000)
    pdf_total = np.zeros_like(x)

    for i in range(best_n_full):
        pdf_c = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_total += pdf_c
        ax1.plot(x, pdf_c, '--', linewidth=2.5, alpha=0.8, color=colors[i],
                 label=f'{pop_short[i]}: μ={best["means"][i]:.2f}±{best["stds"][i]:.2f}')

    ax1.plot(x, pdf_total, 'k-', linewidth=3.5, label='GMM fit', alpha=0.9)

    for i in range(best_n_full):
        xf = np.linspace(best['means'][i]-best['stds'][i],
                         best['means'][i]+best['stds'][i], 100)
        ax1.fill_between(xf, 0,
                         best['weights'][i]*norm.pdf(xf, best['means'][i], best['stds'][i]),
                         color=colors[i], alpha=0.2)

    ax1.set_xlabel('[Fe/H] (dex)', fontsize=22)
    ax1.set_ylabel('Probability Density', fontsize=22)
    ax1.set_title('(a) All GCs (N=884) – Metallicity Distribution', fontsize=17)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(fontsize=13, loc='upper center',
           bbox_to_anchor=(0.55, 1), framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='--')

    sep_str = ', '.join([f'{s:.2f}σ' for s in best['separations']])
    text_str = f"BIC = {best['bic']:.1f}\nAIC = {best['aic']:.1f}\nSep. = {sep_str}"
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes,
             fontsize=13, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel derecho: BIC/AIC
    ax2 = axes[1]
    n_values = sorted(results_full.keys())
    bic_values = [results_full[n]['bic'] for n in n_values]
    aic_values = [results_full[n]['aic'] for n in n_values]

    for i, n in enumerate(n_values):
        is_best = (n == best_n_full)
        color = '#2ca02c' if is_best else '#7f7f7f'
        marker = 'o' if is_best else 's'
        size = 180 if is_best else 90
        lw = 3 if is_best else 1.5

        ax2.scatter(n, bic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6)
        ax2.scatter(n, aic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6, alpha=0.6)

        if results_full[n]['separations']:
            ax2.annotate(f"{min(results_full[n]['separations']):.2f}σ",
                         xy=(n, bic_values[i]),
                         xytext=(0, 14), textcoords='offset points',
                         ha='center', fontsize=13,
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7))

    ax2.plot(n_values, bic_values, '--', color='#D55E00', linewidth=2.5, alpha=0.8)
    ax2.plot(n_values, aic_values, '--', color='#0072B2', linewidth=2.5, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.5, label='BIC'),
        Line2D([0], [0], color='#0072B2', linestyle='--', linewidth=2.5, label='AIC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=13, markeredgewidth=2, markeredgecolor='black',
               label=f'Selected (n={best_n_full}, min. BIC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=9, markeredgewidth=1.5, markeredgecolor='black',
               label='Other models'),
    ]
    ax2.legend(handles=legend_elements, fontsize=16, loc='upper right', framealpha=0.9)
    ax2.set_xlabel('Number of Components', fontsize=22)
    ax2.set_ylabel('Information Criterion', fontsize=22)
    ax2.set_title('(b) Model Selection – All GCs', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_xticks(n_values)
    ax2.text(0.01, 0.01, 'Numbers: min. component separation (σ)',
             transform=ax2.transAxes, fontsize=10, color='gray',
             verticalalignment='bottom')

    plt.tight_layout()
    base = os.path.join(figures_dir, 'FullSample_Metallicity_BIC')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig 1 guardada: {base}.pdf")

# ============================================================================
# FIG 2: Old GCs – metallicity distribution + BIC/AIC
# ============================================================================

def plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old,
                                     output_dir):
    """Fig 2: Old GCs. Left: metallicity distribution (n=3). Right: BIC/AIC."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel izquierdo: distribución
    ax1 = axes[0]
    best = results_old[best_n_old]
    colors = COLORS_3
    pop_short = POP_SHORT_3

    ax1.hist(feh_old, bins=30, density=True, alpha=0.3,
             color='lightgray', edgecolor='black', linewidth=0.8,
             label=f'N={len(feh_old)}')

    x = np.linspace(feh_old.min()-0.3, feh_old.max()+0.3, 1000)
    pdf_total = np.zeros_like(x)

    for i in range(best_n_old):
        pdf_c = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_total += pdf_c
        ax1.plot(x, pdf_c, '--', linewidth=2.5, alpha=0.8, color=colors[i],
                 label=f'{pop_short[i]}: μ={best["means"][i]:.2f}±{best["stds"][i]:.2f}')

    ax1.plot(x, pdf_total, 'k-', linewidth=3.5, label='GMM fit', alpha=0.9)

    for i in range(best_n_old):
        xf = np.linspace(best['means'][i]-best['stds'][i],
                         best['means'][i]+best['stds'][i], 100)
        ax1.fill_between(xf, 0,
                         best['weights'][i]*norm.pdf(xf, best['means'][i], best['stds'][i]),
                         color=colors[i], alpha=0.2)

    ax1.set_xlabel('[Fe/H] (dex)', fontsize=22)
    ax1.set_ylabel('Probability Density', fontsize=22)
    ax1.set_title('(a) Old GCs >8 Gyr (N=720) – Metallicity Distribution', fontsize=17)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(fontsize=13, loc='upper center', framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='--')

    sep_str = ', '.join([f'{s:.2f}σ' for s in best['separations']])
    text_str = f"BIC = {best['bic']:.1f}\nAIC = {best['aic']:.1f}\nSep. = {sep_str}"
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes,
             fontsize=13, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel derecho: BIC/AIC
    ax2 = axes[1]
    n_values = sorted(results_old.keys())
    bic_values = [results_old[n]['bic'] for n in n_values]
    aic_values = [results_old[n]['aic'] for n in n_values]

    for i, n in enumerate(n_values):
        is_best = (n == best_n_old)
        color = '#2ca02c' if is_best else '#7f7f7f'
        marker = 'o' if is_best else 's'
        size = 180 if is_best else 90
        lw = 3 if is_best else 1.5

        ax2.scatter(n, bic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6)
        ax2.scatter(n, aic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6, alpha=0.6)

        if results_old[n]['separations']:
            ax2.annotate(f"{min(results_old[n]['separations']):.2f}σ",
                         xy=(n, bic_values[i]),
                         xytext=(0, 14), textcoords='offset points',
                         ha='center', fontsize=13,
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7))

    ax2.plot(n_values, bic_values, '--', color='#D55E00', linewidth=2.5, alpha=0.8)
    ax2.plot(n_values, aic_values, '--', color='#0072B2', linewidth=2.5, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.5, label='BIC'),
        Line2D([0], [0], color='#0072B2', linestyle='--', linewidth=2.5, label='AIC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=13, markeredgewidth=2, markeredgecolor='black',
               label=f'Selected (n={best_n_old}, min. BIC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=9, markeredgewidth=1.5, markeredgecolor='black',
               label='Other models'),
    ]
    ax2.legend(handles=legend_elements, fontsize=16, loc='upper right', framealpha=0.9)
    ax2.set_xlabel('Number of Components', fontsize=22)
    ax2.set_ylabel('Information Criterion', fontsize=22)
    ax2.set_title('(b) Model Selection – Old GCs >8 Gyr', fontsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_xticks(n_values)
    ax2.text(0.01, 0.01, 'Numbers: min. component separation (σ)',
             transform=ax2.transAxes, fontsize=10, color='gray',
             verticalalignment='bottom')

    plt.tight_layout()
    base = os.path.join(figures_dir, 'OldGCs_Metallicity_BIC')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig 2 guardada: {base}.pdf")

# ============================================================================
# FIG 3: Full sample – gradient scatter (n=3 forced)
# ============================================================================

def plot_gradient_scatter(df_sp, labels_sp, radii, title_str, outfile_name, output_dir):
    """Figura solo con scatter de gradientes + ajustes."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    valid = np.isfinite(radii) & np.isfinite(df_sp['FeH'].values)
    radii_v = radii[valid]
    feh_v = df_sp['FeH'].values[valid]
    labels_v = labels_sp[valid]
    log_r = np.log10(radii_v)

    gradient_results = compute_gradient_results(feh_v, labels_v, log_r)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(3):
        mask = labels_v == i
        if mask.sum() < 10:
            continue

        ax.scatter(log_r[mask], feh_v[mask],
                   s=60, alpha=0.55, color=COLORS_3[i],
                   label=f'{POP_NAMES_3[i]} ({POP_SHORT_3[i]}, N={mask.sum()})',
                   edgecolors='black', linewidth=0.4, zorder=3)

        gr = gradient_results[POP_SHORT_3[i]]
        x_fit = np.linspace(log_r[mask].min(), log_r[mask].max(), 200)
        y_fit = gr['intercept'] + gr['slope'] * x_fit
        n_p = mask.sum()
        x_mean = log_r[mask].mean()
        Sxx = np.sum((log_r[mask] - x_mean)**2)
        s_resid = gr['slope_err'] * np.sqrt(Sxx)        # error estándar residual
        y_err = s_resid * np.sqrt(1/n_p + (x_fit - x_mean)**2 / Sxx)

        sig_str = '*' if gr['p_value'] < 0.05 else '(ns)'
        ax.plot(x_fit, y_fit, color=COLORS_3[i], linewidth=3, alpha=0.9, zorder=4,
                label=f"{POP_SHORT_3[i]}: {gr['slope']:.3f}±{gr['slope_err']:.3f} {sig_str}")
        ax.fill_between(x_fit, y_fit-y_err, y_fit+y_err,
                        color=COLORS_3[i], alpha=0.15, zorder=2)

    ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=24)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=24)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=15, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')

    ymin = np.nanmin(feh_v)
    ymax = np.nanmax(feh_v)
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.39*yr, ymax + 0.18*yr)

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")

    return gradient_results

# ============================================================================
# FIG 4: Full sample – slope comparison bars (n=3 forced)
# ============================================================================

def plot_slope_comparison(gradient_results, title_str, outfile_name, output_dir):
    """Figura solo con barras de pendientes + significancia."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    pops = list(gradient_results.keys())
    slps = [gradient_results[p]['slope'] for p in pops]
    errs = [gradient_results[p]['slope_err'] for p in pops]
    x_pos = np.arange(len(pops))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.bar(x_pos, slps, yerr=errs,
           color=[COLORS_3[i] for i in range(len(pops))],
           alpha=0.75, capsize=8,
           edgecolor='black', linewidth=1.5,
           error_kw={'elinewidth': 2.5, 'capthick': 2.5})

    for i, p in enumerate(pops):
        p_val = gradient_results[p]['p_value']
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        yoff = errs[i] + 0.004 if slps[i] >= 0 else -errs[i] - 0.010
        ax.text(i, slps[i] + yoff, sig,
                ha='center', fontsize=18, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)

    xticklabels = []
    for p in pops:
        gr = gradient_results[p]
        xticklabels.append(f"{p}\n{gr['slope']:.3f}±{gr['slope_err']:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=15)
    ax.set_ylabel('Gradient Slope (dex dex$^{-1}$)', fontsize=22)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    # Recuadro estadístico
    stat_text = "GRADIENT STATISTICS\n" + "─"*30 + "\n"
    for p in pops:
        gr = gradient_results[p]
        stat_text += f"{gr['name']}:\n"
        stat_text += f"  slope = {gr['slope']:.3f}±{gr['slope_err']:.3f}\n"
        stat_text += f"  r = {gr['r_value']:.3f}, p = {gr['p_value']:.4f}\n"
        stat_text += f"  N = {gr['n_points']}, "
        stat_text += f"{'sig.' if gr['significant'] else 'not sig.'}\n\n"

    stat_text += "SLOPE COMPARISONS:\n"
    for i in range(len(pops)):
        for j in range(i+1, len(pops)):
            p1, p2 = pops[i], pops[j]
            gr1, gr2 = gradient_results[p1], gradient_results[p2]
            t_stat = ((gr1['slope'] - gr2['slope']) /
                      np.sqrt(gr1['slope_err']**2 + gr2['slope_err']**2))
            df_t = gr1['n_points'] + gr2['n_points'] - 4
            p_cmp = 2 * (1 - stats.t.cdf(abs(t_stat), df_t))
            stat_text += (f"{p1} vs {p2}: t={t_stat:.2f}, p={p_cmp:.4f} "
                          f"({'diff.' if p_cmp < 0.05 else 'n.s.'})\n")

    ax.text(1.03, 0.98, stat_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='whitesmoke',
                      alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")

# ============================================================================
# FIG 5: Old GCs – gradient scatter (n=3 natural)
# ============================================================================
# (usamos la misma función plot_gradient_scatter con diferente título y nombre)

# ============================================================================
# FIG 6: Old GCs – slope comparison bars (n=3 natural)
# ============================================================================
# (usamos la misma función plot_slope_comparison)

# ============================================================================
# FIG 7: Binned gradients (full sample, 3 panels)
# ============================================================================

def plot_binned_gradients_n3(df_sp, labels_sp, radii_kpc, output_dir):
    figures_dir = os.path.join(output_dir, 'figures')

    valid = np.isfinite(radii_kpc) & np.isfinite(df_sp['FeH'].values)
    radii_v = radii_kpc[valid]
    feh_v = df_sp['FeH'].values[valid]
    labels_v = labels_sp[valid]
    log_r = np.log10(radii_v)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    n_bins = 6

    for ax, i in zip(axes, range(3)):
        mask_p = labels_v == i
        feh_p = feh_v[mask_p]
        logr_p = log_r[mask_p]
        color = COLORS_3[i]

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
                ax.annotate(f'N={ns_b[k]}',
                            xy=(bin_centers[k], means_b[k]),
                            xytext=(0, 12), textcoords='offset points',
                            ha='center', fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.7))

        slope, intercept, r_val, p_val, se = linregress(logr_p, feh_p)
        x_fit = np.linspace(logr_p.min(), logr_p.max(), 100)
        ax.plot(x_fit, intercept + slope*x_fit, '--', color=color,
                linewidth=2.5, alpha=0.8,
                label=f'{slope:.3f}±{se:.3f} dex/dex')

        ax.set_title(f'{POP_NAMES_3[i]} ({POP_SHORT_3[i]})\nN={len(feh_p)}', fontsize=16)
        ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=18)
        if i == 0:
            ax.set_ylabel('Mean [Fe/H] (dex)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle='--')

        sig = '*' if p_val < 0.05 else 'ns'
        ax.text(0.05, 0.05, f'p={p_val:.4f} {sig}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Radial Metallicity Gradients — Binned Analysis (n=3, adopted)',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    base = os.path.join(figures_dir, 'Fig7_Binned_Gradients_n3')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig 7 guardada: {base}.pdf")

# ============================================================================
# TABLA DE RESULTADOS PARA EL PAPER
# ============================================================================

def print_paper_table(results_full, best_n_full,
                      results_old,  best_n_old,
                      means_n3, stds_n3, weights_n3,
                      labels_full_n3,
                      grad_all, grad_old, output_dir):

    lines = []
    lines.append("="*70)
    lines.append("TABLA COMPLETA DE RESULTADOS GMM PARA EL PAPER")
    lines.append("="*70)

    lines.append("\n── TABLA 1: Comparación de modelos GMM ──")
    lines.append(f"{'n':>4} {'BIC (all)':>12} {'ΔBIC (all)':>12} "
                 f"{'BIC (old)':>12} {'ΔBIC (old)':>12} {'Adopted':>8}")
    lines.append("-"*64)
    for n in range(2, 6):
        d_all   = results_full[n]['bic'] - results_full[best_n_full]['bic']
        d_old   = results_old[n]['bic']  - results_old[best_n_old]['bic']
        adopted = "✓" if n == 3 else ""
        lines.append(f"  {n:>2}  {results_full[n]['bic']:>12.1f}  {d_all:>+12.1f}  "
                     f"{results_old[n]['bic']:>12.1f}  {d_old:>+12.1f}  {adopted:>8}")
    lines.append("\nNote: ΔBIC with respect to minimum BIC model in each sample.")
    lines.append("      Adopted model: n=3 (see text).")

    lines.append("\n── TABLA 2: Parámetros del modelo adoptado (n=3, 884 GCs) ──")
    pop_names_long = ['Metal-Poor (MP)', 'Intermediate (Int)', 'Metal-Rich (MR)']
    lines.append(f"{'Population':>22} {'[Fe/H]':>10} {'σ':>8} {'Fraction':>10} {'N':>6}")
    lines.append("-"*60)
    for i in range(3):
        lines.append(f"  {pop_names_long[i]:>20}  {means_n3[i]:>10.3f}  "
                     f"{stds_n3[i]:>8.3f}  {weights_n3[i]:>10.3f}  "
                     f"{int(np.sum(labels_full_n3 == i)):>6}")

    lines.append("\n── TABLA 3: Robustness check (n=3, 720 GCs >8 Gyr) ──")
    best_old = results_old[best_n_old]
    lines.append(f"{'Population':>22} {'[Fe/H]':>10} {'σ':>8} {'Fraction':>10} {'N':>6}")
    lines.append("-"*60)
    for i in range(3):
        n_obj = int(np.sum(best_old['labels'] == i))
        lines.append(f"  {pop_names_long[i]:>20}  {best_old['means'][i]:>10.3f}  "
                     f"{best_old['stds'][i]:>8.3f}  {best_old['weights'][i]:>10.3f}  "
                     f"{n_obj:>6}")

    lines.append("\n── TABLA 4: Gradientes radiales — 884 GCs (n=3 adoptado) ──")
    lines.append(f"{'Pop':>6} {'Slope':>10} {'±err':>8} {'r':>8} "
                 f"{'p-value':>10} {'N':>6} {'Sig.':>6}")
    lines.append("-"*56)
    for p, gr in grad_all.items():
        lines.append(f"  {p:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                     f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  "
                     f"{gr['n_points']:>6}  {'Yes' if gr['significant'] else 'No':>6}")

    if grad_old:
        lines.append("\n── TABLA 5: Gradientes radiales — 720 GCs viejos >8 Gyr ──")
        lines.append(f"{'Pop':>6} {'Slope':>10} {'±err':>8} {'r':>8} "
                     f"{'p-value':>10} {'N':>6} {'Sig.':>6}")
        lines.append("-"*56)
        for p, gr in grad_old.items():
            lines.append(f"  {p:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                         f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  "
                         f"{gr['n_points']:>6}  {'Yes' if gr['significant'] else 'No':>6}")

    lines.append("\n── Separaciones (n=3, muestra completa) ──")
    sep_names = ['MP–Int', 'Int–MR']
    for i, sep in enumerate(results_full[3]['separations']):
        sig = "≥2σ" if sep >= 2 else "<2σ"
        lines.append(f"  {sep_names[i]}: {sep:.2f}σ ({sig})")

    lines.append("\n── Bayes Factors (muestra completa, ref. n=4) ──")
    for n in [2, 3, 5]:
        bf = calculate_bayes_factor(results_full[best_n_full]['bic'],
                                    results_full[n]['bic'])
        lines.append(f"  n=4 vs n={n}: BF={bf:.2e} → {interpret_bayes_factor(bf)}")

    lines.append("\n── Bayes Factors (GCs viejos, ref. n=3) ──")
    for n in [2, 4, 5]:
        bf = calculate_bayes_factor(results_old[best_n_old]['bic'],
                                    results_old[n]['bic'])
        lines.append(f"  n=3 vs n={n}: BF={bf:.2e} → {interpret_bayes_factor(bf)}")

    lines.append("\n" + "="*70)
    output = "\n".join(lines)
    print(output)

    txt_path = os.path.join(output_dir, 'paper_tables_statistics.txt')
    with open(txt_path, 'w') as f:
        f.write(output)
    print(f"\n✓ Tablas guardadas: {txt_path}")

    return output

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ANÁLISIS GMM COMPLETO — VERSIÓN FINAL PARA EL PAPER")
    print("NGC 5128 Globular Cluster System")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = f"GMM_Final_Paper_{timestamp}"
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

    df        = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar   = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)

    Q1 = df['FeH'].quantile(0.01)
    Q3 = df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= Q1-1.5*(Q3-Q1)) & (df['FeH'] <= Q3+1.5*(Q3-Q1))]

    age_col = 'bayes.stellar.age_m_star'
    if age_col not in df.columns:
        print(f"❌ Columna '{age_col}' no encontrada.")
        return
    df['Age_Gyr'] = df[age_col] / 1000.0

    # Coordenadas
    try:
        catalog  = pd.read_csv('../Results_Corrected/'
                               'all_fields_photometry_OPTIMIZED_1394objects.csv')
        ra_col   = 'RAJ2000' if 'RAJ2000' in catalog.columns else 'RA'
        dec_col  = 'DEJ2000' if 'DEJ2000' in catalog.columns else 'DEC'
        df['id_str']         = df['id'].astype(str).str.strip()
        catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
        df = pd.merge(df, catalog[[ra_col, dec_col, 'T17ID_str']],
                      left_on='id_str', right_on='T17ID_str', how='left')
        df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
        print(f"  ✓ Coordenadas: {df['RA'].notna().sum()} objetos")
    except Exception as e:
        print(f"  ⚠️  Sin coordenadas: {e}")

    feh_full = df['FeH'].values
    print(f"  ✓ Muestra completa: {len(feh_full)} GCs")

    df_old  = df[df['Age_Gyr'] > 8.0].copy().reset_index(drop=True)
    feh_old = df_old['FeH'].values
    print(f"  ✓ GCs viejos >8 Gyr: {len(feh_old)} ({len(feh_old)/len(feh_full)*100:.1f}%)")

    # ── 2. GMM ───────────────────────────────────────────────────────
    print("\n[2] GMM — Muestra completa...")
    results_full, best_n_full = run_gmm_analysis(feh_full, 'All GCs')

    print("\n[3] GMM — GCs viejos (>8 Gyr)...")
    results_old, best_n_old = run_gmm_analysis(feh_old, 'Old GCs >8 Gyr')

    # ── 3. Clasificación n=3 forzado ─────────────────────────────────
    print("\n[4] Clasificación n=3 — 884 GCs:")
    (labels_full_n3, means_n3, stds_n3, weights_n3,
     seps_n3, _, bic_n3, aic_n3) = get_labels_forced_n3(feh_full)

    print("\n[4b] Clasificación n=3 — 720 GCs viejos:")
    (labels_old_n3, means_old_n3, stds_old_n3, weights_old_n3,
     seps_old_n3, _, bic_old_n3, aic_old_n3) = get_labels_forced_n3(feh_old)

    # ── 4. Radios ─────────────────────────────────────────────────────
    print("\n[5] Calculando radios proyectados...")
    radii_full, valid_full = calculate_projected_radii(df)
    if radii_full is not None:
        print(f"  ✓ Rango: {radii_full.min():.1f} – {radii_full.max():.0f} kpc")
        df_sp_full     = df[valid_full].copy().reset_index(drop=True)
        labels_sp_full = labels_full_n3[valid_full]
    else:
        print("  ⚠️  Sin coordenadas")
        df_sp_full     = df.copy().reset_index(drop=True)
        labels_sp_full = labels_full_n3

    radii_old, valid_old = calculate_projected_radii(df_old)
    if radii_old is not None:
        df_sp_old     = df_old[valid_old].copy().reset_index(drop=True)
        labels_sp_old = labels_old_n3[valid_old]
    else:
        df_sp_old     = df_old.copy().reset_index(drop=True)
        labels_sp_old = labels_old_n3

    # ── 5. Figuras ────────────────────────────────────────────────────
    print("\n[6] Generando figuras...")

    # Figuras 1 y 2: distribución + BIC/AIC
    plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full, outdir)
    plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old, outdir)

    # Figuras de gradientes y comparación de pendientes
    grad_all = {}
    grad_old = {}

    if radii_full is not None:
        grad_all = plot_gradient_scatter(
            df_sp_full, labels_sp_full, radii_full,
            title_str='Radial Metallicity Gradients\nAll GCs (n=3 forced)',
            outfile_name='FullSample_Gradient_Scatter',
            output_dir=outdir)
        plot_slope_comparison(
            grad_all,
            title_str='Slope Comparison\nAll GCs (n=3 forced)',
            outfile_name='Fig4_FullSample_Slope_Comparison',
            output_dir=outdir)

    if radii_old is not None:
        grad_old = plot_gradient_scatter(
            df_sp_old, labels_sp_old, radii_old,
            title_str='Radial Metallicity Gradients\nOld GCs >8 Gyr (n=3 natural)',
            outfile_name='OldGCs_Gradient_Scatter',
            output_dir=outdir)
        plot_slope_comparison(
            grad_old,
            title_str='Slope Comparison\nOld GCs >8 Gyr (n=3 natural)',
            outfile_name='Fig6_OldGCs_Slope_Comparison',
            output_dir=outdir)

    # Figura 7: binneados (opcional)
    if radii_full is not None:
        plot_binned_gradients_n3(df_sp_full, labels_sp_full, radii_full, outdir)

    # ── 6. Tablas ─────────────────────────────────────────────────────
    print("\n[7] Generando tablas...")
    print_paper_table(
        results_full, best_n_full,
        results_old,  best_n_old,
        means_n3, stds_n3, weights_n3,
        labels_full_n3,          # ← agregar esta línea
        grad_all, grad_old, outdir)

    # ── 7. JSON ───────────────────────────────────────────────────────
    json_data = {
        'version': 'gmm_final_paper_v5',
        'date': datetime.datetime.now().isoformat(),
        'adopted_model': {
            'n_components': 3,
            'justification': (
                'n=4 wins BIC in full sample (ΔBIC=7.2 over n=3). '
                'Restricted to old GCs (>8 Gyr), n=3 wins decisively '
                '(ΔBIC=123 over n=2; ΔBIC=2.9 over n=4). '
                'The 4th component is driven by young merger-related GCs.'),
            'full_sample': {
                'means':   [float(x) for x in means_n3],
                'stds':    [float(x) for x in stds_n3],
                'weights': [float(x) for x in weights_n3],
                'bic': float(bic_n3), 'aic': float(aic_n3),
                'separations': [float(s) for s in seps_n3],
            },
            'old_gcs': {
                'means':   [float(x) for x in means_old_n3],
                'stds':    [float(x) for x in stds_old_n3],
                'weights': [float(x) for x in weights_old_n3],
                'bic': float(bic_old_n3), 'aic': float(aic_old_n3),
                'separations': [float(s) for s in seps_old_n3],
            },
        },
        'gmm_comparison': {
            'full_sample': {str(n): {
                'bic': float(r['bic']), 'aic': float(r['aic']),
                'delta_bic': float(r['bic']-results_full[best_n_full]['bic']),
                'separations': [float(s) for s in r['separations']],
            } for n, r in results_full.items()},
            'old_gcs': {str(n): {
                'bic': float(r['bic']), 'aic': float(r['aic']),
                'delta_bic': float(r['bic']-results_old[best_n_old]['bic']),
                'separations': [float(s) for s in r['separations']],
            } for n, r in results_old.items()},
        },
        'gradients_all_gcs': grad_all,
        'gradients_old_gcs': grad_old,
    }

    json_path = os.path.join(outdir, 'gmm_final_paper_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON: {json_path}")

    # ── Resumen final ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"  Muestra completa: {len(feh_full)} GCs → BIC mínimo: n={best_n_full}")
    print(f"  GCs viejos:       {len(feh_old)} GCs  → BIC mínimo: n={best_n_old}")
    print(f"\n  MODELO ADOPTADO: n=3")
    print(f"  ΔBIC(n=4 vs n=3) muestra completa = "
          f"{results_full[4]['bic']-results_full[3]['bic']:+.1f}")
    print(f"  ΔBIC(n=3 vs n=4) GCs viejos       = "
          f"{results_old[3]['bic']-results_old[4]['bic']:+.1f}")
    print(f"  ΔBIC(n=3 vs n=2) GCs viejos       = "
          f"{results_old[3]['bic']-results_old[2]['bic']:+.1f}")

    print(f"\n  Parámetros n=3 (884 GCs):")
    for i in range(3):
        print(f"    • {POP_NAMES_3[i]}: [Fe/H]={means_n3[i]:.3f}±{stds_n3[i]:.3f}, "
                 f"f={weights_n3[i]*100:.1f}%, N={int(np.sum(labels_full_n3 == i))}")

    if grad_all:
        print(f"\n  Gradientes — 884 GCs:")
        for p, gr in grad_all.items():
            sig = "sig." if gr['significant'] else "not sig."
            print(f"    • {p}: {gr['slope']:.3f}±{gr['slope_err']:.3f} "
                  f"(p={gr['p_value']:.4f}, {sig})")

    if grad_old:
        print(f"\n  Gradientes — GCs viejos:")
        for p, gr in grad_old.items():
            sig = "sig." if gr['significant'] else "not sig."
            print(f"    • {p}: {gr['slope']:.3f}±{gr['slope_err']:.3f} "
                  f"(p={gr['p_value']:.4f}, {sig})")

    print(f"\n  Figuras en: {outdir}/figures/")
    figs = [
        "FullSample_Metallicity_BIC.pdf",
        "OldGCs_Metallicity_BIC.pdf",
        "FullSample_Gradient_Scatter.pdf",
        "FullSample_Slope_Comparison.pdf",
        "OldGCs_Gradient_Scatter.pdf",
        "OldGCs_Slope_Comparison.pdf",
        "Binned_Gradients_n3.pdf",
    ]
    for f in figs:
        print(f"    {f}")

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO — LISTO PARA EL PAPER")
    print("="*70)


if __name__ == "__main__":
    main()
