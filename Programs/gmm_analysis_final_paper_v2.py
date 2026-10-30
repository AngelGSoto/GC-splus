#!/usr/bin/env python3
# gmm_analysis_final_paper_v2.py
# Analisis GMM completo para el paper -- VERSION 2 (respuesta al arbitro)
#
# Igual que gmm_analysis_final_paper.py (mismo GMM, misma clasificacion, mismos
# gradientes, mismo estilo de figuras; solo se agrandaron textos pequenos), mas:
#   * Comentario 5: ajuste pesado por los errores de [Fe/H] (con y sin dispersion
#     intrinseca), jackknife, bootstrap con reemplazo (clasificacion fija) y
#     bootstrap con re-ajuste del GMM en cada realizacion.
#   * Comentario 7: distribucion de metalicidades en bins radiales (BIC por bin,
#     GMM libre por bin, fracciones con medias/anchos globales fijos, medias por
#     poblacion por bin).
#   * Comentario 3 (iii): se corre igual en cada corrida de CIGALE (--label) y
#     comparar_gradientes_modelos.py compara los JSON.
#
# Uso (desde la carpeta de la corrida, la que contiene out/results.fits):
#   cd CIGALE-DEcam-extinction
#   python ../../Programs/gmm_analysis_final_paper_v2.py --label paper
#
# Genera (en GMM_Final_Paper_<label>_<fecha>/):
#   Fig 1-7 originales + Fig8_Bootstrap_Slopes + Fig9_MDF_Radial_Bins
#   paper_tables_statistics.txt y gmm_final_paper_results.json

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, linregress
from scipy.optimize import minimize
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
# CONFIGURACION DE ESTILO (identico al paper original)
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
# FUNCIONES AUXILIARES (originales)
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
    """GMM para n=2..5, seleccion por BIC minimo."""
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
    print(f"GMM -- {label} (N={len(feh_data)})")
    print(f"{'='*60}")
    print(f"\n{'n':>4} {'BIC':>10} {'dBIC':>10} {'AIC':>10} {'Min sep':>10} {'All>2s':>8}")
    print("-"*56)
    for n in sorted(results.keys()):
        delta  = results[n]['bic'] - results[best_n]['bic']
        sep    = f"{min(results[n]['separations']):.2f}s"
        marker = " <- BEST (min BIC)" if n == best_n else ""
        print(f"  {n:>2}  {results[n]['bic']:>10.1f}  {delta:>+10.1f}  "
              f"{results[n]['aic']:>10.1f}  {sep:>10}  "
              f"{str(results[n]['all_gt_2sigma']):>8}{marker}")

    print(f"\nFactores de Bayes (referencia: n={best_n}):")
    for n in sorted(results.keys()):
        if n != best_n:
            delta = results[n]['bic'] - results[best_n]['bic']
            bf    = calculate_bayes_factor(results[best_n]['bic'], results[n]['bic'])
            print(f"  n={best_n} vs n={n}: dBIC={delta:+.1f}, "
                  f"BF={bf:.2e} -> {interpret_bayes_factor(bf)}")

    print("\n" + "-"*60)
    print("PARAMETROS COMPLETOS DE TODOS LOS MODELOS:")
    print("-"*60)
    for n in sorted(results.keys()):
        print(f"\nModelo n={n} (BIC={results[n]['bic']:.1f}):")
        for i in range(n):
            n_obj = np.sum(results[n]['labels'] == i)
            print(f"  Comp {i+1}: [Fe/H]={results[n]['means'][i]:.3f}+-{results[n]['stds'][i]:.3f}, "
                  f"f={results[n]['weights'][i]*100:.1f}%, N={n_obj}")
        if results[n]['separations']:
            sep_str = ', '.join([f'{s:.2f}s' for s in results[n]['separations']])
            print(f"  Separaciones: {sep_str}")
        print(f"  Min membership prob: {results[n]['min_quality']:.3f}")

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
    print(f"  Separaciones: {[f'{s:.2f}s' for s in separations]}")
    for i in range(3):
        n_obj = np.sum(labels_ordered == i)
        print(f"  {POP_NAMES_3[i]}: [Fe/H]={means[i]:.3f}+-{stds[i]:.3f}, "
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
    """Calcula gradientes radiales para cada poblacion. Devuelve dict."""
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
# FIG 1 / FIG 2: distribucion de metalicidad + BIC/AIC (textos agrandados)
# ============================================================================

def _plot_metallicity_and_BIC(feh, results, best_n, colors, pop_short, title_a,
                              title_b, base, legend_anchor=None, sel_label=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    best = results[best_n]
    ax1.hist(feh, bins=30, density=True, alpha=0.3,
             color='lightgray', edgecolor='black', linewidth=0.8,
             label=f'N={len(feh)}')

    x = np.linspace(feh.min()-0.3, feh.max()+0.3, 1000)
    pdf_total = np.zeros_like(x)
    for i in range(best_n):
        pdf_c = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_total += pdf_c
        ax1.plot(x, pdf_c, '--', linewidth=2.5, alpha=0.8, color=colors[i],
                   label=f'{pop_short[i]}: μ={best["means"][i]:.2f}, σ={best["stds"][i]:.2f}')
    ax1.plot(x, pdf_total, 'k-', linewidth=3.5, label='GMM fit', alpha=0.9)
    for i in range(best_n):
        xf = np.linspace(best['means'][i]-best['stds'][i],
                         best['means'][i]+best['stds'][i], 100)
        ax1.fill_between(xf, 0,
                         best['weights'][i]*norm.pdf(xf, best['means'][i], best['stds'][i]),
                         color=colors[i], alpha=0.2)

    ax1.set_xlabel('[Fe/H] (dex)', fontsize=22)
    ax1.set_ylabel('Probability Density', fontsize=22)
    ax1.set_title(title_a, fontsize=19)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    if legend_anchor:
        ax1.legend(fontsize=16, loc='upper center', bbox_to_anchor=legend_anchor,
                   framealpha=0.9)
    else:
        ax1.legend(fontsize=16, loc='upper center', framealpha=0.9)
    ax1.set_ylim(0, 1.12 * ax1.get_ylim()[1])
    #ax1.grid(True, alpha=0.2, linestyle='--')

    sep_str = ', '.join([f'{s:.2f}σ' for s in best['separations']])
    text_str = f"BIC = {best['bic']:.1f}\nAIC = {best['aic']:.1f}\nSep. = {sep_str}"
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes,
             fontsize=16, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2 = axes[1]
    n_values = sorted(results.keys())
    bic_values = [results[n]['bic'] for n in n_values]
    aic_values = [results[n]['aic'] for n in n_values]
    for i, n in enumerate(n_values):
        is_best = (n == best_n)
        color = '#2ca02c' if is_best else '#7f7f7f'
        marker = 'o' if is_best else 's'
        size = 180 if is_best else 90
        lw = 3 if is_best else 1.5
        ax2.scatter(n, bic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6)
        ax2.scatter(n, aic_values[i], s=size, color=color, marker=marker,
                    edgecolors='black', linewidth=lw, zorder=6, alpha=0.6)
        if results[n]['separations']:
            ax2.annotate(f"{min(results[n]['separations']):.2f}σ",
                         xy=(n, bic_values[i]),
                         xytext=(0, 14), textcoords='offset points',
                         ha='center', fontsize=16,
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7))
    ax2.plot(n_values, bic_values, '--', color='#D55E00', linewidth=2.5, alpha=0.8)
    ax2.plot(n_values, aic_values, '--', color='#0072B2', linewidth=2.5, alpha=0.8)

    legend_elements = [
        Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.5, label='BIC'),
        Line2D([0], [0], color='#0072B2', linestyle='--', linewidth=2.5, label='AIC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=13, markeredgewidth=2, markeredgecolor='black',
               label=sel_label or f'Selected (n={best_n}, min. BIC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=9, markeredgewidth=1.5, markeredgecolor='black',
               label='Other models'),
    ]
    ax2.legend(handles=legend_elements, fontsize=17, loc='upper right', framealpha=0.9)
    ax2.set_xlabel('Number of Components', fontsize=22)
    ax2.set_ylabel('Information Criterion', fontsize=22)
    ax2.set_title(title_b, fontsize=19)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    #ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_xticks(n_values)
    ax2.text(0.01, 0.01, 'Numbers: min. component separation (σ)',
             transform=ax2.transAxes, fontsize=14, color='gray',
             verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")

def plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full, output_dir):
    pop_short = ['MP', 'IL', 'IH', 'MR'] if best_n_full == 4 else \
        (POP_SHORT_3 if best_n_full == 3 else [f'C{i+1}' for i in range(best_n_full)])
    colors = COLORS_4 if best_n_full == 4 else (COLORS_3 if best_n_full == 3 else
                                                plt.cm.tab10.colors)
    _plot_metallicity_and_BIC(
        feh_full, results_full, best_n_full, colors, pop_short,
        f'(a) All GCs (N={len(feh_full)}) – Metallicity Distribution',
        '(b) Model Selection – All GCs',
        os.path.join(output_dir, 'figures', 'FullSample_Metallicity_BIC'),
        legend_anchor=(0.55, 1))

def plot_full_sample_adopted_n3(feh_full, results_full, output_dir):
    """NUEVA (comentario 6): las 3 componentes ADOPTADAS sobre la muestra completa."""
    bic_best = min(results_full.keys(), key=lambda x: results_full[x]['bic'])
    _plot_metallicity_and_BIC(
        feh_full, results_full, 3, COLORS_3, POP_SHORT_3,
        f'(a) All GCs (N={len(feh_full)}) – Adopted model (n=3)',
        f'(b) Model Selection – All GCs (min. BIC: n={bic_best})',
        os.path.join(output_dir, 'figures', 'FullSample_Adopted_n3_Metallicity_BIC'),
        legend_anchor=(0.55, 1), sel_label='Adopted (n=3)')

def plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old, output_dir):
    colors = COLORS_3 if best_n_old == 3 else COLORS_4
    pop_short = POP_SHORT_3 if best_n_old == 3 else ['MP', 'IL', 'IH', 'MR'][:best_n_old]
    _plot_metallicity_and_BIC(
        feh_old, results_old, best_n_old, colors, pop_short,
        f'(a) Old GCs >8 Gyr (N={len(feh_old)}) – Metallicity Distribution',
        '(b) Model Selection – Old GCs >8 Gyr',
        os.path.join(output_dir, 'figures', 'OldGCs_Metallicity_BIC'))

# ============================================================================
# FIG 3/5: gradientes (textos agrandados)
# ============================================================================

def plot_gradient_scatter(df_sp, labels_sp, radii, title_str, outfile_name, output_dir):
    figures_dir = os.path.join(output_dir, 'figures')
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
        s_resid = gr['slope_err'] * np.sqrt(Sxx)
        y_err = s_resid * np.sqrt(1/n_p + (x_fit - x_mean)**2 / Sxx)
        sig_str = '*' if gr['p_value'] < 0.05 else '(ns)'
        ax.plot(x_fit, y_fit, color=COLORS_3[i], linewidth=3, alpha=0.9, zorder=4,
                label=f"{POP_SHORT_3[i]}: {gr['slope']:.3f}±{gr['slope_err']:.3f} {sig_str}")
        ax.fill_between(x_fit, y_fit-y_err, y_fit+y_err,
                        color=COLORS_3[i], alpha=0.15, zorder=2)

    ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=28)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=28)
    ax.set_title(title_str, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.legend(fontsize=17, loc='lower left', framealpha=0.9)
    #ax.grid(True, alpha=0.2, linestyle='--')
    ymin, ymax = np.nanmin(feh_v), np.nanmax(feh_v)
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.45*yr, ymax + 0.18*yr)

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")
    return gradient_results

# ============================================================================
# FIG 4/6: barras de pendientes (textos agrandados)
# ============================================================================

def plot_slope_comparison(gradient_results, title_str, outfile_name, output_dir):
    figures_dir = os.path.join(output_dir, 'figures')
    pops = list(gradient_results.keys())
    slps = [gradient_results[p]['slope'] for p in pops]
    errs = [gradient_results[p]['slope_err'] for p in pops]
    x_pos = np.arange(len(pops))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x_pos, slps, yerr=errs,
           color=[COLORS_3[i] for i in range(len(pops))],
           alpha=0.75, capsize=8, edgecolor='black', linewidth=1.5,
           error_kw={'elinewidth': 2.5, 'capthick': 2.5})
    for i, p in enumerate(pops):
        p_val = gradient_results[p]['p_value']
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        yoff = errs[i] + 0.004 if slps[i] >= 0 else -errs[i] - 0.010
        ax.text(i, slps[i] + yoff, sig, ha='center', fontsize=20, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)

    xticklabels = [f"{p}\n{gradient_results[p]['slope']:.3f}±{gradient_results[p]['slope_err']:.3f}"
                   for p in pops]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xticklabels, fontsize=18)
    ax.set_ylabel('Gradient Slope (dex dex$^{-1}$)', fontsize=22)
    ax.set_title(title_str, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.grid(True, alpha=0.2, linestyle='--', axis='y')

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
            fontsize=14, verticalalignment='top', linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")

# ============================================================================
# FIG 7: gradientes binneados (textos agrandados)
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
        feh_p, logr_p, color = feh_v[mask_p], log_r[mask_p], COLORS_3[i]
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
                            ha='center', fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        slope, intercept, r_val, p_val, se = linregress(logr_p, feh_p)
        x_fit = np.linspace(logr_p.min(), logr_p.max(), 100)
        ax.plot(x_fit, intercept + slope*x_fit, '--', color=color, linewidth=2.5,
                alpha=0.8, label=f'{slope:.3f}±{se:.3f} dex/dex')
        ax.set_title(f'{POP_NAMES_3[i]} ({POP_SHORT_3[i]})\nN={len(feh_p)}', fontsize=19)
        ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=21)
        if i == 0:
            ax.set_ylabel('Mean [Fe/H] (dex)', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=19)
        ax.legend(fontsize=16, loc='upper right', framealpha=0.9)
        #ax.grid(True, alpha=0.2, linestyle='--')
        sig = '*' if p_val < 0.05 else 'ns'
        ax.text(0.05, 0.05, f'p={p_val:.4f} {sig}', transform=ax.transAxes, fontsize=15,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Radial Metallicity Gradients — Binned Analysis (n=3, adopted)',
                 fontsize=21, fontweight='bold', y=1.02)
    plt.tight_layout()
    base = os.path.join(figures_dir, 'Fig7_Binned_Gradients_n3')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")

# ============================================================================
# NUEVO -- COMENTARIO 5: robustez de las pendientes
# ============================================================================

def fit_wls(x, y, e):
    """Minimos cuadrados pesados por 1/e^2 (sin dispersion intrinseca)."""
    w = 1.0 / e**2
    W = np.sum(w); xm = np.sum(w*x)/W; ym = np.sum(w*y)/W
    b = np.sum(w*(x-xm)*(y-ym)) / np.sum(w*(x-xm)**2)
    return ym - b*xm, b

def fit_weighted_intrinsic(x, y, e):
    """Maxima verosimilitud: y = a + b x, var_i = e_i^2 + s_int^2."""
    lr = linregress(x, y)
    s0 = max(np.var(y - lr.intercept - lr.slope*x) - np.median(e)**2, 1e-4)
    def nll(p):
        a, b, lns = p
        var = e**2 + np.exp(2*lns)
        r = y - a - b*x
        return 0.5*np.sum(r**2/var + np.log(var))
    res = minimize(nll, [lr.intercept, lr.slope, 0.5*np.log(s0)], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 5000})
    a, b, lns = res.x
    return a, b, float(np.exp(lns))

def jackknife_se(x, y):
    n = len(x)
    s = np.empty(n)
    keep = np.ones(n, bool)
    for i in range(n):
        keep[i] = False
        s[i] = linregress(x[keep], y[keep]).slope
        keep[i] = True
    return float(np.sqrt((n-1)/n*np.sum((s - s.mean())**2)))

def _summ(arr):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return dict(median=np.nan, p16=np.nan, p84=np.nan, std=np.nan, p_neg=np.nan, n=0)
    return dict(median=float(np.median(arr)), p16=float(np.percentile(arr, 16)),
                p84=float(np.percentile(arr, 84)), std=float(np.std(arr)),
                p_neg=float(np.mean(arr < 0)), n=int(len(arr)))

def bootstrap_gmm_refit(feh, logr, nboot, rng):
    """Remuestrea toda la muestra, re-ajusta el GMM (n=3) y recalcula pendientes."""
    X = feh.reshape(-1, 1)
    n = len(feh)
    out = np.full((nboot, 3), np.nan)
    for k in range(nboot):
        idx = rng.integers(0, n, n)
        g = GaussianMixture(3, n_init=2, max_iter=500, tol=1e-4,
                            random_state=int(rng.integers(1_000_000_000))).fit(X[idx])
        order = np.argsort(g.means_.flatten())
        inv = np.empty(3, int); inv[order] = np.arange(3)
        lab = inv[g.predict(X[idx])]
        for i in range(3):
            m = lab == i
            if m.sum() >= 10:
                out[k, i] = linregress(logr[idx][m], feh[idx][m]).slope
    return out

def gradient_robustness(feh_v, err_v, labels_v, log_r, nboot, nboot_w, nboot_gmm, rng,
                        err_floor=0.0):
    res = {}
    err_v = np.sqrt(err_v**2 + err_floor**2)
    for i in range(3):
        m = labels_v == i
        if m.sum() < 10:
            continue
        x, y, e = log_r[m], feh_v[m], err_v[m]
        ok = np.isfinite(e) & (e > 0)
        lr = linregress(x, y)
        _, b_wls = fit_wls(x[ok], y[ok], e[ok])
        _, b_int, s_int = fit_weighted_intrinsic(x[ok], y[ok], e[ok])
        nb = len(x)
        bs = np.empty(nboot)
        for k in range(nboot):
            j = rng.integers(0, nb, nb)
            bs[k] = linregress(x[j], y[j]).slope
        xo, yo, eo = x[ok], y[ok], e[ok]
        bw = np.empty(nboot_w)
        for k in range(nboot_w):
            j = rng.integers(0, len(xo), len(xo))
            bw[k] = fit_weighted_intrinsic(xo[j], yo[j], eo[j])[1]
        res[POP_SHORT_3[i]] = {
            'N': int(m.sum()),
            'ols_slope': float(lr.slope), 'ols_err': float(lr.stderr),
            'jackknife_err': jackknife_se(x, y),
            'bootstrap': _summ(bs),
            'wls_slope': float(b_wls),
            'wint_slope': float(b_int), 'wint_sigma_int': s_int,
            'wint_bootstrap': _summ(bw),
            'median_err_FeH': float(np.median(e[ok])),
            'err_floor': float(err_floor),
            'rho_err_logR': float(stats.spearmanr(e[ok], x[ok])[0]),
            'rho_err_FeH': float(stats.spearmanr(e[ok], y[ok])[0]),
            '_bs': bs, '_bw': bw,
        }
    gr = bootstrap_gmm_refit(feh_v, log_r, nboot_gmm, rng)
    for i in range(3):
        if POP_SHORT_3[i] in res:
            res[POP_SHORT_3[i]]['gmm_refit_bootstrap'] = _summ(gr[:, i])
            res[POP_SHORT_3[i]]['_bg'] = gr[:, i]
    return res

def plot_bootstrap_slopes(rob_all, rob_old, output_dir):
    """Fig 8: distribucion bootstrap de las pendientes (fila 1: todos; fila 2: viejos)."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for row, (rob, lab) in enumerate([(rob_all, 'All GCs'), (rob_old, 'Old GCs >8 Gyr')]):
        for i, p in enumerate(POP_SHORT_3):
            ax = axes[row, i]
            if p not in rob:
                ax.axis('off'); continue
            r = rob[p]
            bins = np.linspace(min(np.nanmin(r['_bs']), np.nanmin(r['_bg'])),
                               max(np.nanmax(r['_bs']), np.nanmax(r['_bg'])), 40)
            ax.hist(r['_bs'], bins=bins, density=True, alpha=0.5, color=COLORS_3[i],
                    edgecolor='black', linewidth=0.6)
            ax.hist(r['_bg'][np.isfinite(r['_bg'])], bins=bins, density=True,
                    histtype='step', color='black', linewidth=2.2)
            ax.axvline(r['ols_slope'], color=COLORS_3[i], linewidth=3)
            ax.axvline(0, color='gray', linestyle='--', linewidth=2)
            b = r['bootstrap']
            ax.set_title(f"{POP_NAMES_3[i]} ({p}) – {lab}\n"
                         f"{r['ols_slope']:.3f} (+{b['p84']-b['median']:.3f}/"
                         f"−{b['median']-b['p16']:.3f}), N={r['N']}", fontsize=21)
            ax.set_xlabel('Gradient slope (dex dex$^{-1}$)', fontsize=26)
            if i == 0:
                ax.set_ylabel('Density', fontsize=28)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            #ax.grid(True, alpha=0.2, linestyle='--')
    # Leyenda fuera de los paneles: no tapa ningun histograma
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5, edgecolor='black'),
               Line2D([0], [0], color='black', linewidth=2.2),
               Line2D([0], [0], color='gray', linewidth=3),
               Line2D([0], [0], color='gray', linestyle='--', linewidth=2)]
    labels = ['Bootstrap (fixed membership)', 'Bootstrap (GMM re-fitted)',
              'Least-squares slope', 'Zero slope']
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', 'Fig8_Bootstrap_Slopes')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")


# ============================================================================

def fit_weights_only(f, mu, sd, n_iter=1000):
    w = np.ones(len(mu)) / len(mu)
    for _ in range(n_iter):
        p = w * norm.pdf(f[:, None], mu, sd)
        r = p / p.sum(1, keepdims=True)
        w_new = r.mean(0)
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new; break
        w = w_new
    return w

def mdf_radial_bins(feh, labels, log_r, mu_g, sd_g, nbins, rng, nboot=200):
    edges = np.quantile(log_r, np.linspace(0, 1, nbins+1))
    out = []
    for k in range(nbins):
        m = (log_r >= edges[k]) & (log_r <= edges[k+1]) if k == nbins-1 else \
            (log_r >= edges[k]) & (log_r < edges[k+1])
        f = feh[m]
        X = f.reshape(-1, 1)
        bics = {}
        for n in range(1, 5):
            g = GaussianMixture(n, n_init=5, random_state=42, max_iter=500, tol=1e-4).fit(X)
            bics[n] = float(g.bic(X))
        best = min(bics, key=bics.get)
        g3 = validate_gmm_model(f, 3)
        o = np.argsort(g3.means_.flatten())
        w_fix = fit_weights_only(f, mu_g, sd_g)
        wb = np.array([fit_weights_only(f[rng.integers(0, len(f), len(f))], mu_g, sd_g)
                       for _ in range(nboot)])
        pop_means = []
        for i in range(3):
            mi = labels[m] == i
            pop_means.append(dict(mean=float(np.mean(f[mi])) if mi.sum() else np.nan,
                                  sem=float(np.std(f[mi])/np.sqrt(mi.sum())) if mi.sum() > 1 else np.nan,
                                  N=int(mi.sum())))
        out.append({
            'R_min_kpc': float(10**edges[k]), 'R_max_kpc': float(10**edges[k+1]),
            'N': int(m.sum()), 'bic': bics, 'best_n_bic': int(best),
            'free_n3_means': [float(v) for v in g3.means_.flatten()[o]],
            'free_n3_stds': [float(v) for v in np.sqrt(g3.covariances_.flatten())[o]],
            'free_n3_weights': [float(v) for v in g3.weights_.flatten()[o]],
            'fixed_weights': [float(v) for v in w_fix],
            'fixed_weights_err': [float(v) for v in wb.std(0)],
            'pop_mean_FeH': pop_means,
            '_feh': f,
        })
    return out

def plot_mdf_radial_bins(bins_res, mu_g, sd_g, output_dir):
    """Fig 9: distribucion de [Fe/H] en bins radiales con las componentes globales."""
    nb = len(bins_res)
    fig, axes = plt.subplots(1, nb, figsize=(7*nb, 7), sharey=True)
    axes = np.atleast_1d(axes)
    allf = np.concatenate([b['_feh'] for b in bins_res])
    x = np.linspace(allf.min()-0.3, allf.max()+0.3, 1000)
    ymax = 0.0
    for ax, b in zip(axes, bins_res):
        h = ax.hist(b['_feh'], bins=20, density=True, alpha=0.3, color='lightgray',
                    edgecolor='black', linewidth=0.8, label=f"N={b['N']}")[0]
        tot = np.zeros_like(x)
        for i in range(3):
            pc = b['fixed_weights'][i] * norm.pdf(x, mu_g[i], sd_g[i])
            tot += pc
            ax.plot(x, pc, '--', linewidth=2.5, color=COLORS_3[i], alpha=0.85,
                    label=f"{POP_SHORT_3[i]}: f={b['fixed_weights'][i]:.2f}±{b['fixed_weights_err'][i]:.2f}")
        ax.plot(x, tot, 'k-', linewidth=3, alpha=0.9, label='GMM (global μ, σ)')
        ymax = max(ymax, h.max(), tot.max())
        ax.set_title(f"{b['R_min_kpc']:.0f} < R < {b['R_max_kpc']:.0f} kpc", fontsize=25)
        ax.set_xlabel('[Fe/H] (dex)', fontsize=27)
        ax.tick_params(axis='both', which='major', labelsize=24)
        #ax.grid(True, alpha=0.2, linestyle='--')
        ax.legend(fontsize=17, loc='upper left', framealpha=0.9)
        #means_txt = ', '.join(f'{v:.2f}' for v in b['free_n3_means'])
        #ax.text(0.98, 0.98, f"Min. BIC: n={b['best_n_bic']}\nFree n=3 μ:\n{means_txt}",
        #        transform=ax.transAxes, fontsize=17, verticalalignment='top',
        #        horizontalalignment='right',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_ylabel('Probability Density', fontsize=27)
    for ax in axes:
        ax.set_ylim(0, MDF_YTOP * ymax)
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', 'Fig9_MDF_Radial_Bins')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")


# ============================================================================

def robustness_table_lines(rob, title):
    L = [f"\n── {title} ──",
         f"{'Pop':>4} {'N':>4} {'OLS':>8} {'±OLS':>6} {'±jack':>6} {'boot med':>9} "
         f"{'16-84%':>15} {'P(<0)':>6} {'WLS':>8} {'W+int':>8} {'±boot':>6} {'s_int':>6} "
         f"{'GMMrefit med':>12} {'16-84%':>15} {'P(<0)':>6} {'rho(err,logR)':>13} {'rho(err,FeH)':>12}"]
    for p, r in rob.items():
        b, w, g = r['bootstrap'], r['wint_bootstrap'], r['gmm_refit_bootstrap']
        L.append(f"{p:>4} {r['N']:>4} {r['ols_slope']:>+8.3f} {r['ols_err']:>6.3f} "
                 f"{r['jackknife_err']:>6.3f} {b['median']:>+9.3f} "
                 f"[{b['p16']:+.3f},{b['p84']:+.3f}] {b['p_neg']:>6.3f} "
                 f"{r['wls_slope']:>+8.3f} {r['wint_slope']:>+8.3f} {w['std']:>6.3f} "
                 f"{r['wint_sigma_int']:>6.3f} {g['median']:>+12.3f} "
                 f"[{g['p16']:+.3f},{g['p84']:+.3f}] {g['p_neg']:>6.3f} "
                 f"{r['rho_err_logR']:>+13.2f} {r['rho_err_FeH']:>+12.2f}")
    return L

def mdf_table_lines(bins_res):
    L = ["\n── Distribucion de metalicidades en bins radiales (muestra completa) ──",
         f"{'R (kpc)':>14} {'N':>4} {'BIC best n':>10}  "
         f"{'f_MP':>11} {'f_Int':>11} {'f_MR':>11}   "
         f"{'<FeH>_MP':>9} {'<FeH>_Int':>9} {'<FeH>_MR':>9}   free n=3 means"]
    for b in bins_res:
        fw, fe = b['fixed_weights'], b['fixed_weights_err']
        pm = b['pop_mean_FeH']
        L.append(f"{b['R_min_kpc']:6.1f}-{b['R_max_kpc']:6.1f} {b['N']:>4} {b['best_n_bic']:>10}  "
                 f"{fw[0]:.2f}±{fe[0]:.2f}  {fw[1]:.2f}±{fe[1]:.2f}  {fw[2]:.2f}±{fe[2]:.2f}   "
                 f"{pm[0]['mean']:>+9.3f} {pm[1]['mean']:>+9.3f} {pm[2]['mean']:>+9.3f}   "
                 + ', '.join(f'{v:+.2f}' for v in b['free_n3_means']))
    return L

def print_paper_table(results_full, best_n_full, results_old, best_n_old,
                      means_n3, stds_n3, weights_n3, labels_full_n3,
                      grad_all, grad_old, output_dir, extra_lines):
    lines = ["="*70, "TABLA COMPLETA DE RESULTADOS GMM PARA EL PAPER", "="*70]
    lines.append("\n── TABLA 1: Comparacion de modelos GMM ──")
    lines.append(f"{'n':>4} {'BIC (all)':>12} {'dBIC (all)':>12} "
                 f"{'BIC (old)':>12} {'dBIC (old)':>12} {'Adopted':>8}")
    for n in range(2, 6):
        d_all = results_full[n]['bic'] - results_full[best_n_full]['bic']
        if results_old:
            d_old = results_old[n]['bic'] - results_old[best_n_old]['bic']
            old_txt = f"{results_old[n]['bic']:>12.1f}  {d_old:>+12.1f}"
        else:
            old_txt = f"{'—':>12}  {'—':>12}"
        lines.append(f"  {n:>2}  {results_full[n]['bic']:>12.1f}  {d_all:>+12.1f}  "
                     f"{old_txt}  {'*' if n == 3 else '':>8}")
    lines.append("\n── TABLA 2: Parametros del modelo adoptado (n=3, muestra completa) ──")
    for i, nm in enumerate(['Metal-Poor (MP)', 'Intermediate (Int)', 'Metal-Rich (MR)']):
        lines.append(f"  {nm:>20}  {means_n3[i]:>10.3f}  {stds_n3[i]:>8.3f}  "
                     f"{weights_n3[i]:>10.3f}  {int(np.sum(labels_full_n3 == i)):>6}")
    for title, grad in [("TABLA 4: Gradientes radiales — muestra completa (n=3)", grad_all),
                        ("TABLA 5: Gradientes radiales — GCs viejos >8 Gyr", grad_old)]:
        if not grad:
            continue
        lines.append(f"\n── {title} ──")
        lines.append(f"{'Pop':>6} {'Slope':>10} {'+-err':>8} {'r':>8} {'p-value':>10} {'N':>6}")
        for p, gr in grad.items():
            lines.append(f"  {p:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                         f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  {gr['n_points']:>6}")
    lines.append("\n── Separaciones (n=3, muestra completa) ──")
    for i, sep in enumerate(results_full[3]['separations']):
        lines.append(f"  {['MP-Int', 'Int-MR'][i]}: {sep:.2f}s")
    lines += extra_lines
    lines.append("\n" + "="*70)
    output = "\n".join(lines)
    print(output)
    with open(os.path.join(output_dir, 'paper_tables_statistics.txt'), 'w') as f:
        f.write(output)
    print(f"\nOK tablas: {os.path.join(output_dir, 'paper_tables_statistics.txt')}")

def _clean(d):
    """Quita arrays internos (claves con '_') para el JSON."""
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if not str(k).startswith('_')}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d

# ============================================================================
# PRINCIPAL
# ============================================================================

MDF_YTOP = 1.30   # limite superior del eje Y de Fig9 (multiplo del maximo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--label', default='run', help="nombre de la corrida (p.ej. paper, bc03)")
    ap.add_argument('--results', default='out/results.fits')
    ap.add_argument('--catalog', default='../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    ap.add_argument('--nboot', type=int, default=2000, help="bootstrap OLS (clases fijas)")
    ap.add_argument('--nboot-w', type=int, default=300, help="bootstrap del ajuste pesado")
    ap.add_argument('--nboot-gmm', type=int, default=300, help="bootstrap con re-ajuste del GMM")
    ap.add_argument('--nbins', type=int, default=3, help="bins radiales (comentario 7)")
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--err-floor', type=float, default=0.0,
                    help="error minimo de [Fe/H] (dex) sumado en cuadratura en el ajuste pesado")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"GMM_Final_Paper_{a.label}_{timestamp}"
    if a.err_floor > 0:
        outdir = f"GMM_Final_Paper_{a.label}_floor{a.err_floor:g}_{timestamp}"
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    print(f"Resultados: {outdir}")

    table = Table.read(a.results, hdu=1)
    df = table.to_pandas()
    df = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)
    df['FeH_err'] = (df['bayes.stellar.metallicity_err'] / df['bayes.stellar.metallicity']) / np.log(10)

    Q1 = df['FeH'].quantile(0.01)
    Q3 = df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= Q1-1.5*(Q3-Q1)) & (df['FeH'] <= Q3+1.5*(Q3-Q1))]
    df['Age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0

    catalog = pd.read_csv(a.catalog)
    ra_col = 'RAJ2000' if 'RAJ2000' in catalog.columns else 'RA'
    dec_col = 'DEJ2000' if 'DEJ2000' in catalog.columns else 'DEC'
    df['id_str'] = df['id'].astype(str).str.strip()
    catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
    df = pd.merge(df, catalog[[ra_col, dec_col, 'T17ID_str']],
                  left_on='id_str', right_on='T17ID_str', how='left')
    df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
    print(f"  Coordenadas: {df['RA'].notna().sum()} objetos")

    feh_full = df['FeH'].values
    df_old = df[df['Age_Gyr'] > 8.0].copy().reset_index(drop=True)
    feh_old = df_old['FeH'].values
    print(f"  Muestra completa: {len(feh_full)} | viejos >8 Gyr: {len(feh_old)}")

    results_full, best_n_full = run_gmm_analysis(feh_full, 'All GCs')
    do_old = len(feh_old) >= 50
    if do_old:
        results_old, best_n_old = run_gmm_analysis(feh_old, 'Old GCs >8 Gyr')
    else:
        print(f"\nAVISO: solo {len(feh_old)} GCs con edad > 8 Gyr; se omite el analisis de la "
              "muestra vieja (las edades de esta configuracion no son comparables).")
        results_old, best_n_old = None, None

    print("\nClasificacion n=3 — muestra completa:")
    (labels_full_n3, means_n3, stds_n3, weights_n3, seps_n3, _, bic_n3, aic_n3) = \
        get_labels_forced_n3(feh_full)
    if do_old:
        print("\nClasificacion n=3 — GCs viejos:")
        (labels_old_n3, means_old_n3, stds_old_n3, weights_old_n3, seps_old_n3, _,
         bic_old_n3, aic_old_n3) = get_labels_forced_n3(feh_old)

    radii_full, valid_full = calculate_projected_radii(df)
    df_sp_full = df[valid_full].copy().reset_index(drop=True)
    labels_sp_full = labels_full_n3[valid_full.values]
    if do_old:
        radii_old, valid_old = calculate_projected_radii(df_old)
        df_sp_old = df_old[valid_old].copy().reset_index(drop=True)
        labels_sp_old = labels_old_n3[valid_old.values]
    print(f"  Radios: {radii_full.min():.1f} – {radii_full.max():.0f} kpc")

    # Figuras originales
    plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full, outdir)
    plot_full_sample_adopted_n3(feh_full, results_full, outdir)
    if do_old:
        plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old, outdir)
    grad_all = plot_gradient_scatter(df_sp_full, labels_sp_full, radii_full,
                                     'Radial Metallicity Gradients\nAll GCs (n=3)',
                                     'FullSample_Gradient_Scatter', outdir)
    plot_slope_comparison(grad_all, 'Slope Comparison\nAll GCs (n=3)',
                          'Fig4_FullSample_Slope_Comparison', outdir)
    grad_old = {}
    if do_old:
        grad_old = plot_gradient_scatter(df_sp_old, labels_sp_old, radii_old,
                                         'Radial Metallicity Gradients\nOld GCs >8 Gyr (n=3)',
                                         'OldGCs_Gradient_Scatter', outdir)
        plot_slope_comparison(grad_old, 'Slope Comparison\nOld GCs >8 Gyr (n=3)',
                              'Fig6_OldGCs_Slope_Comparison', outdir)
    plot_binned_gradients_n3(df_sp_full, labels_sp_full, radii_full, outdir)

    # Comentario 5
    print("\n[Comentario 5] Robustez de las pendientes (puede tardar 1-3 min)...")
    def _arrays(dsp, lab, rad):
        v = np.isfinite(rad) & np.isfinite(dsp['FeH'].values)
        return dsp['FeH'].values[v], dsp['FeH_err'].values[v], lab[v], np.log10(rad[v])
    fa, ea, la, ra = _arrays(df_sp_full, labels_sp_full, radii_full)
    rob_all = gradient_robustness(fa, ea, la, ra, a.nboot, a.nboot_w, a.nboot_gmm, rng, a.err_floor)
    rob_old = {}
    if do_old:
        fo, eo, lo, ro = _arrays(df_sp_old, labels_sp_old, radii_old)
        rob_old = gradient_robustness(fo, eo, lo, ro, a.nboot, a.nboot_w, a.nboot_gmm, rng, a.err_floor)
    plot_bootstrap_slopes(rob_all, rob_old, outdir)

    # Comentario 7
    print("\n[Comentario 7] Distribucion de metalicidades en bins radiales...")
    bins_res = mdf_radial_bins(fa, la, ra, means_n3, stds_n3, a.nbins, rng)
    plot_mdf_radial_bins(bins_res, means_n3, stds_n3, outdir)

    extra = (robustness_table_lines(rob_all, "Comentario 5: robustez de pendientes — muestra completa")
             + (robustness_table_lines(rob_old, "Comentario 5: robustez de pendientes — GCs viejos")
                if rob_old else [])
             + ["\n  OLS: ajuste original (linregress); ±jack: jackknife; boot: bootstrap con "
                "reemplazo (clases fijas);",
                "  WLS: pesado por 1/err^2; W+int: pesado con dispersion intrinseca (s_int);",
                "  GMMrefit: bootstrap re-ajustando el GMM n=3 en cada realizacion; "
                "P(<0): fraccion de pendientes negativas."]
             + mdf_table_lines(bins_res))
    print_paper_table(results_full, best_n_full, results_old, best_n_old,
                      means_n3, stds_n3, weights_n3, labels_full_n3,
                      grad_all, grad_old, outdir, extra)

    json_data = {
        'version': 'gmm_final_paper_v2', 'label': a.label, 'results_file': a.results,
        'date': datetime.datetime.now().isoformat(),
        'N_full': int(len(feh_full)), 'N_old': int(len(feh_old)),
        'best_n_bic_full': int(best_n_full),
        'best_n_bic_old': int(best_n_old) if do_old else None,
        'adopted_n3_full': {'means': list(map(float, means_n3)), 'stds': list(map(float, stds_n3)),
                            'weights': list(map(float, weights_n3)),
                            'N': [int(np.sum(labels_full_n3 == i)) for i in range(3)],
                            'separations': list(map(float, seps_n3))},
        'adopted_n3_old': ({'means': list(map(float, means_old_n3)), 'stds': list(map(float, stds_old_n3)),
                            'weights': list(map(float, weights_old_n3)),
                            'N': [int(np.sum(labels_old_n3 == i)) for i in range(3)]} if do_old else None),
        'bic_full': {str(n): float(r['bic']) for n, r in results_full.items()},
        'bic_old': ({str(n): float(r['bic']) for n, r in results_old.items()} if do_old else None),
        'gradients_all_gcs': grad_all, 'gradients_old_gcs': grad_old,
        'robustness_all_gcs': _clean(rob_all), 'robustness_old_gcs': _clean(rob_old),
        'mdf_radial_bins': _clean(bins_res),
    }
    with open(os.path.join(outdir, 'gmm_final_paper_results.json'), 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"OK JSON: {os.path.join(outdir, 'gmm_final_paper_results.json')}")

    print("\nCONTROL (paper, Tabla 2): MP -0.099±0.025 (240), Int -0.001±0.024 (435), "
          "MR -0.043±0.010 (209); viejos: -0.092 (175), +0.030 (369), -0.045 (176)")
    for p, gr in grad_all.items():
        print(f"  todos  {p}: {gr['slope']:+.3f}±{gr['slope_err']:.3f} (N={gr['n_points']})")
    for p, gr in grad_old.items():
        print(f"  viejos {p}: {gr['slope']:+.3f}±{gr['slope_err']:.3f} (N={gr['n_points']})")


if __name__ == "__main__":
    main()
