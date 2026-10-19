#!/usr/bin/env python3

# Analisis GMM completo para el paper -- VERSION CORREGIDA (bootstrap reproducible)
#
# Igual que tu gmm_analysis_final_paper.py (mismo GMM, misma clasificacion, mismos
# gradientes, mismo bootstrap: remuestreo con reemplazo + perturbacion segun los
# errores individuales de [Fe/H], seleccion por BIC minimo), con estas correcciones:
#   1. Semilla fija (--seed): los porcentajes del bootstrap son reproducibles.
#   2. Estabilidad del modelo n=3: mediana y percentiles 16-84 de las medias,
#      anchos y fracciones de MP/Int/MR en las replicas (tabla + JSON).
#   3. Los ajustes que fallan se cuentan y se informan (antes se ignoraban).
#   4. Figura del bootstrap con las barras centradas en cada numero de componentes.
#   5. Textos pequenos de las figuras agrandados (mismo estilo).
#   6. Opciones por linea de comandos: --nboot, --seed, --input, --catalog.
#
# Uso (desde la carpeta de la corrida, la que contiene out/results.fits):
#   python ../../Programs/gmm_analysis_final_paper_bootstrap.py
#   python ../../Programs/gmm_analysis_final_paper_bootstrap.py --nboot 200 --seed 2026

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
# CONFIGURACION POR DEFECTO (se puede cambiar con la linea de comandos)
# ============================================================================
INPUT_FITS = 'out/results.fits'
CATALOG_CSV = '../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv'
N_BOOTSTRAP = 200
SEED = 2026

# ============================================================================
# CONFIGURACION DE ESTILO
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

# Centro de NGC 5128
RA_CENTER, DEC_CENTER = 201.365063, -43.019112
DISTANCE_MPC = 3.8

COLORS_3 = ['#2E86AB', '#A23B72', '#C73E1D']
COLORS_4 = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
COLORS_5 = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7B2D8E']

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
    """Ajusta GMM con multiples semillas y devuelve el mejor modelo."""
    X = feh_data.reshape(-1, 1)
    best_gmm, best_score = None, -np.inf
    for seed in [42, 123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed,
                              n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        score = gmm.score(X)          # log-likelihood
        if score > best_score:
            best_score = score
            best_gmm   = gmm
    return best_gmm

def _pop_names(n):
    if n == 2:
        return COLORS_3[:2], ['MP', 'MR']
    if n == 3:
        return COLORS_3, POP_SHORT_3
    if n == 4:
        return COLORS_4, ['MP', 'IL', 'IH', 'MR']
    return COLORS_5, [f'P{i+1}' for i in range(n)]

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
    """Calcula radios proyectados en kpc a partir de coordenadas RA/DEC."""
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
# BOOTSTRAP: REMUESTREO + PERTURBACION SEGUN ERRORES INDIVIDUALES
# ============================================================================

def bootstrap_analysis_with_quality(feh_data, feh_errors=None, n_bootstrap=200,
                                    perturb_errors=True, label=''):
    """
    En cada replica se remuestrea con reemplazo, se perturba cada [Fe/H] segun su
    error y se selecciona el modelo con menor BIC (n=2..5). Se guardan ademas los
    parametros del modelo n=3 en cada replica para evaluar su estabilidad.
    Nota: perturbar valores que ya incluyen el error de medicion suma el ruido
    una segunda vez, por lo que la prueba es conservadora (favorece menos componentes).
    """
    print("\n" + "="*80)
    print(f"BOOTSTRAP CON PERTURBACION DE ERRORES — {label}")
    print("Remuestreo con reemplazo + perturbacion gaussiana; seleccion por BIC minimo")
    print("="*80)

    n_samples = len(feh_data)

    if feh_errors is None or not perturb_errors:
        feh_errors = np.zeros_like(feh_data)
    else:
        feh_errors = np.asarray(feh_errors)
        if len(feh_errors) != n_samples:
            raise ValueError("feh_errors debe tener el mismo tamano que feh_data")
        feh_errors = np.where(np.isfinite(feh_errors) & (feh_errors > 0), feh_errors, 0.0)

    counts = {2: 0, 3: 0, 4: 0, 5: 0}
    n_failed_fits = 0
    n3_means, n3_stds, n3_weights = [], [], []

    print(f"  Ejecutando {n_bootstrap} replicas bootstrap...")

    for i in range(n_bootstrap):
        if (i + 1) % 50 == 0:
            print(f"    Replica {i+1}/{n_bootstrap}")

        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        sample = feh_data[sample_idx].copy()
        sample_err = feh_errors[sample_idx]
        if perturb_errors:
            sample = sample + np.random.normal(0.0, sample_err)

        best_bic = np.inf
        best_n   = None

        for n in [2, 3, 4, 5]:
            try:
                gmm_temp = validate_gmm_model(sample, n)
            except Exception as exc:          # se cuenta, no se ignora en silencio
                n_failed_fits += 1
                print(f"    AVISO: fallo el ajuste n={n} en la replica {i+1}: {exc}")
                continue
            bic = gmm_temp.bic(sample.reshape(-1, 1))
            if bic < best_bic:
                best_bic = bic
                best_n   = n
            if n == 3:
                means3 = gmm_temp.means_.flatten()
                stds3  = np.sqrt(gmm_temp.covariances_.flatten())
                weights3 = gmm_temp.weights_.flatten()
                idx3 = np.argsort(means3)
                n3_means.append(means3[idx3])
                n3_stds.append(stds3[idx3])
                n3_weights.append(weights3[idx3])

        if best_n is not None:
            counts[best_n] += 1

    total = sum(counts.values())
    success_rate = total / n_bootstrap * 100 if n_bootstrap > 0 else 0

    print(f"\n RESULTADOS BOOTSTRAP (seleccion por BIC) — {label}:")
    print(f"  Replicas con modelo valido: {total}/{n_bootstrap}  "
          f"(ajustes fallidos: {n_failed_fits})")
    for n in [2, 3, 4, 5]:
        pct = counts[n] / total * 100 if total > 0 else 0
        print(f"  • n={n}: {counts[n]} replicas ({pct:.1f}%)")

    # Estabilidad del modelo n=3
    stability = {}
    if n3_means:
        M, S, W = np.array(n3_means), np.array(n3_stds), np.array(n3_weights)
        print("\n  Estabilidad del modelo n=3 (mediana [percentiles 16-84]):")
        for i, name in enumerate(POP_SHORT_3):
            q = {k: np.percentile(arr[:, i], [16, 50, 84]) for k, arr in
                 [('mean', M), ('std', S), ('weight', W)]}
            stability[name] = {k: {'p16': float(v[0]), 'median': float(v[1]), 'p84': float(v[2])}
                               for k, v in q.items()}
            print(f"    {name}: mu = {q['mean'][1]:+.3f} [{q['mean'][0]:+.3f}, {q['mean'][2]:+.3f}]  "
                  f"sigma = {q['std'][1]:.3f} [{q['std'][0]:.3f}, {q['std'][2]:.3f}]  "
                  f"f = {q['weight'][1]:.3f} [{q['weight'][0]:.3f}, {q['weight'][2]:.3f}]")

    return {
        'counts': counts,
        'total': total,
        'success_rate': success_rate,
        'n_failed_fits': n_failed_fits,
        'n3_stability': stability,
    }

# ============================================================================
# FIGURAS (mismo estilo; textos pequenos agrandados)
# ============================================================================

def _plot_metallicity_and_BIC(feh, results, best_n, title_a, title_b, base,
                              legend_anchor=None):
    colors, pop_short = _pop_names(best_n)
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
                 label=f'{pop_short[i]}: μ={best["means"][i]:.2f}±{best["stds"][i]:.2f}')
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
        ax1.legend(fontsize=16, loc='upper center', bbox_to_anchor=legend_anchor, framealpha=0.9)
    else:
        ax1.legend(fontsize=16, loc='upper center', framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='--')
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
                         xy=(n, bic_values[i]), xytext=(0, 14), textcoords='offset points',
                         ha='center', fontsize=16,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    ax2.plot(n_values, bic_values, '--', color='#D55E00', linewidth=2.5, alpha=0.8)
    ax2.plot(n_values, aic_values, '--', color='#0072B2', linewidth=2.5, alpha=0.8)
    legend_elements = [
        Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.5, label='BIC'),
        Line2D([0], [0], color='#0072B2', linestyle='--', linewidth=2.5, label='AIC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=13, markeredgewidth=2, markeredgecolor='black',
               label=f'Selected (n={best_n}, min. BIC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=9, markeredgewidth=1.5, markeredgecolor='black',
               label='Other models'),
    ]
    ax2.legend(handles=legend_elements, fontsize=17, loc='upper right', framealpha=0.9)
    ax2.set_xlabel('Number of Components', fontsize=22)
    ax2.set_ylabel('Information Criterion', fontsize=22)
    ax2.set_title(title_b, fontsize=19)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_xticks(n_values)
    ax2.text(0.01, 0.01, 'Numbers: min. component separation (σ)',
             transform=ax2.transAxes, fontsize=14, color='gray', verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura guardada: {base}.pdf")

def plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full, output_dir):
    _plot_metallicity_and_BIC(
        feh_full, results_full, best_n_full,
        f'(a) All GCs (N={len(feh_full)}) – Metallicity Distribution',
        '(b) Model Selection – All GCs',
        os.path.join(output_dir, 'figures', 'FullSample_Metallicity_BIC'),
        legend_anchor=(0.55, 1))

def plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old, output_dir):
    _plot_metallicity_and_BIC(
        feh_old, results_old, best_n_old,
        f'(a) Old GCs >8 Gyr (N={len(feh_old)}) – Metallicity Distribution',
        '(b) Model Selection – Old GCs >8 Gyr',
        os.path.join(output_dir, 'figures', 'OldGCs_Metallicity_BIC'))

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
        ax.scatter(log_r[mask], feh_v[mask], s=60, alpha=0.55, color=COLORS_3[i],
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
        ax.fill_between(x_fit, y_fit-y_err, y_fit+y_err, color=COLORS_3[i], alpha=0.15, zorder=2)

    ax.set_xlabel(r'$\log_{10}(R\,/\,\mathrm{kpc})$', fontsize=24)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=24)
    ax.set_title(title_str, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=17, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ymin, ymax = np.nanmin(feh_v), np.nanmax(feh_v)
    yr = ymax - ymin
    ax.set_ylim(ymin - 0.45*yr, ymax + 0.18*yr)

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")
    return gradient_results

def plot_slope_comparison(gradient_results, title_str, outfile_name, output_dir):
    figures_dir = os.path.join(output_dir, 'figures')
    pops = list(gradient_results.keys())
    slps = [gradient_results[p]['slope'] for p in pops]
    errs = [gradient_results[p]['slope_err'] for p in pops]
    x_pos = np.arange(len(pops))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(x_pos, slps, yerr=errs, color=[COLORS_3[i] for i in range(len(pops))],
           alpha=0.75, capsize=8, edgecolor='black', linewidth=1.5,
           error_kw={'elinewidth': 2.5, 'capthick': 2.5})
    for i, p in enumerate(pops):
        p_val = gradient_results[p]['p_value']
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        yoff = errs[i] + 0.004 if slps[i] >= 0 else -errs[i] - 0.010
        ax.text(i, slps[i] + yoff, sig, ha='center', fontsize=20, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{p}\n{gradient_results[p]['slope']:.3f}±{gradient_results[p]['slope_err']:.3f}"
                        for p in pops], fontsize=18)
    ax.set_ylabel('Gradient Slope (dex dex$^{-1}$)', fontsize=22)
    ax.set_title(title_str, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')

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
    ax.text(1.03, 0.98, stat_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    base = os.path.join(figures_dir, outfile_name)
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ {outfile_name} guardada: {base}.pdf")

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
            if k == n_bins - 1:
                mask_b = (logr_p >= bin_edges[k]) & (logr_p <= bin_edges[k+1])
            else:
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
        ax.grid(True, alpha=0.2, linestyle='--')
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
    print(f"✓ Fig 7 guardada: {base}.pdf")

def plot_bootstrap_selection(boot_full, boot_old, n_full, n_old, output_dir):
    """Porcentajes de seleccion bootstrap (barras centradas en cada n)."""
    figures_dir = os.path.join(output_dir, 'figures')
    n_values = [2, 3, 4, 5]
    x = np.arange(len(n_values))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, boot, color, title in [
            (axes[0], boot_full, 'steelblue', f'(a) Full sample (N={n_full})'),
            (axes[1], boot_old, 'indianred', f'(b) Old GCs >8 Gyr (N={n_old})')]:
        tot = boot['total']
        pct = [boot['counts'][n]/tot*100 if tot > 0 else 0 for n in n_values]
        bars = ax.bar(x, pct, width, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        for bar, p in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{p:.1f}%', ha='center', va='bottom', fontsize=16)
        ax.set_title(title, fontsize=19)
        ax.set_xlabel('Number of Components', fontsize=20)
        ax.set_ylabel('Selection Percentage (%)', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(n_values)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    plt.suptitle('Bootstrap Model Selection with Perturbed Errors (BIC)',
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    base = os.path.join(figures_dir, 'Fig8_Bootstrap_Selection')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig 8 guardada: {base}.pdf")

# ============================================================================
# TABLA DE RESULTADOS
# ============================================================================

def print_paper_table(results_full, best_n_full, results_old, best_n_old,
                      means_n3, stds_n3, weights_n3, labels_full_n3,
                      grad_all, grad_old, boot_full, boot_old, seed, n_boot,
                      output_dir):
    lines = ["="*70, "TABLA COMPLETA DE RESULTADOS GMM PARA EL PAPER", "="*70]

    lines.append("\n── TABLA 1: Comparacion de modelos GMM ──")
    lines.append(f"{'n':>4} {'BIC (all)':>12} {'ΔBIC (all)':>12} "
                 f"{'BIC (old)':>12} {'ΔBIC (old)':>12} {'Adopted':>8}")
    lines.append("-"*64)
    for n in range(2, 6):
        d_all = results_full[n]['bic'] - results_full[best_n_full]['bic']
        d_old = results_old[n]['bic'] - results_old[best_n_old]['bic']
        lines.append(f"  {n:>2}  {results_full[n]['bic']:>12.1f}  {d_all:>+12.1f}  "
                     f"{results_old[n]['bic']:>12.1f}  {d_old:>+12.1f}  {'✓' if n == 3 else '':>8}")

    lines.append("\n── TABLA 2: Parametros del modelo adoptado (n=3, muestra completa) ──")
    names_long = ['Metal-Poor (MP)', 'Intermediate (Int)', 'Metal-Rich (MR)']
    lines.append(f"{'Population':>22} {'[Fe/H]':>10} {'σ':>8} {'Fraction':>10} {'N':>6}")
    for i in range(3):
        lines.append(f"  {names_long[i]:>20}  {means_n3[i]:>10.3f}  {stds_n3[i]:>8.3f}  "
                     f"{weights_n3[i]:>10.3f}  {int(np.sum(labels_full_n3 == i)):>6}")

    lines.append("\n── TABLA 3: Modelo n=3 en GCs viejos (>8 Gyr) ──")
    best_old = results_old[best_n_old]
    lines.append(f"{'Population':>22} {'[Fe/H]':>10} {'σ':>8} {'Fraction':>10} {'N':>6}")
    for i in range(min(3, best_n_old)):
        lines.append(f"  {names_long[i]:>20}  {best_old['means'][i]:>10.3f}  "
                     f"{best_old['stds'][i]:>8.3f}  {best_old['weights'][i]:>10.3f}  "
                     f"{int(np.sum(best_old['labels'] == i)):>6}")

    for title, grad in [("TABLA 4: Gradientes radiales — muestra completa (n=3)", grad_all),
                        ("TABLA 5: Gradientes radiales — GCs viejos >8 Gyr", grad_old)]:
        if not grad:
            continue
        lines.append(f"\n── {title} ──")
        lines.append(f"{'Pop':>6} {'Slope':>10} {'±err':>8} {'r':>8} {'p-value':>10} {'N':>6} {'Sig.':>6}")
        for p, gr in grad.items():
            lines.append(f"  {p:>4}  {gr['slope']:>10.3f}  {gr['slope_err']:>8.3f}  "
                         f"{gr['r_value']:>8.3f}  {gr['p_value']:>10.4f}  "
                         f"{gr['n_points']:>6}  {'Yes' if gr['significant'] else 'No':>6}")

    lines.append("\n── Separaciones (n=3, muestra completa) ──")
    for i, sep in enumerate(results_full[3]['separations']):
        lines.append(f"  {['MP–Int', 'Int–MR'][i]}: {sep:.2f}σ ({'≥2σ' if sep >= 2 else '<2σ'})")

    lines.append(f"\n── Bayes Factors (muestra completa, ref. n={best_n_full}) ──")
    for n in [m for m in range(2, 6) if m != best_n_full]:
        bf = calculate_bayes_factor(results_full[best_n_full]['bic'], results_full[n]['bic'])
        lines.append(f"  n={best_n_full} vs n={n}: BF={bf:.2e} → {interpret_bayes_factor(bf)}")
    lines.append(f"\n── Bayes Factors (GCs viejos, ref. n={best_n_old}) ──")
    for n in [m for m in range(2, 6) if m != best_n_old]:
        bf = calculate_bayes_factor(results_old[best_n_old]['bic'], results_old[n]['bic'])
        lines.append(f"  n={best_n_old} vs n={n}: BF={bf:.2e} → {interpret_bayes_factor(bf)}")

    lines.append(f"\n── Bootstrap: remuestreo + perturbacion (BIC), {n_boot} replicas, semilla {seed} ──")
    lines.append(f"{'n':>4} {'Full count':>12} {'Full %':>10} {'Old count':>12} {'Old %':>10}")
    for n in [2, 3, 4, 5]:
        c_f, c_o = boot_full['counts'][n], boot_old['counts'][n]
        p_f = c_f / boot_full['total'] * 100 if boot_full['total'] else 0
        p_o = c_o / boot_old['total'] * 100 if boot_old['total'] else 0
        lines.append(f"  {n:>2}  {c_f:>12}  {p_f:>9.1f}%  {c_o:>12}  {p_o:>9.1f}%")
    lines.append(f"  Ajustes fallidos: todos={boot_full['n_failed_fits']}, viejos={boot_old['n_failed_fits']}")

    for lab, boot in [("muestra completa", boot_full), ("GCs viejos", boot_old)]:
        st = boot['n3_stability']
        if not st:
            continue
        lines.append(f"\n── Estabilidad del modelo n=3 en el bootstrap ({lab}) ──")
        lines.append(f"{'Pop':>5} {'mu (16-50-84)':>28} {'sigma (16-50-84)':>26} {'f (16-50-84)':>26}")
        for p in POP_SHORT_3:
            s = st[p]
            lines.append(f"  {p:>3}  {s['mean']['p16']:+.3f} {s['mean']['median']:+.3f} {s['mean']['p84']:+.3f}   "
                         f"{s['std']['p16']:.3f} {s['std']['median']:.3f} {s['std']['p84']:.3f}   "
                         f"{s['weight']['p16']:.3f} {s['weight']['median']:.3f} {s['weight']['p84']:.3f}")

    lines.append("\n" + "="*70)
    output = "\n".join(lines)
    print(output)
    txt_path = os.path.join(output_dir, 'paper_tables_statistics.txt')
    with open(txt_path, 'w') as f:
        f.write(output)
    print(f"\n✓ Tablas guardadas: {txt_path}")
    return output

# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="GMM + bootstrap reproducible (NGC 5128)")
    ap.add_argument('--input', default=INPUT_FITS)
    ap.add_argument('--catalog', default=CATALOG_CSV)
    ap.add_argument('--nboot', type=int, default=N_BOOTSTRAP)
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args()

    print("\n" + "="*70)
    print("ANALISIS GMM COMPLETO — VERSION CORREGIDA (bootstrap reproducible)")
    print("NGC 5128 Globular Cluster System")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = f"GMM_Final_Paper_bootstrap_seed{a.seed}_{timestamp}"
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)
    print(f"Resultados: {outdir}")
    np.random.seed(a.seed)            # <- reproducibilidad del bootstrap
    print(f"Semilla: {a.seed} | replicas bootstrap: {a.nboot}")

    # ── 1. Cargar datos ──────────────────────────────────────────────
    print("\n[1] Cargando datos...")
    table = Table.read(a.input, hdu=1)
    df    = table.to_pandas()

    df        = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar   = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)

    met = df['bayes.stellar.metallicity'].replace(0, np.nan)
    if 'bayes.stellar.metallicity_err' in df.columns:
        df['FeH_err'] = np.abs(df['bayes.stellar.metallicity_err'] / (met * np.log(10)))
    else:
        print("AVISO: no hay columna de error de metalicidad; se usa σ[Fe/H] = 0.571 dex.")
        df['FeH_err'] = 0.571

    p01 = df['FeH'].quantile(0.01)
    p99 = df['FeH'].quantile(0.99)
    iqr = p99 - p01
    df = df[(df['FeH'] >= p01 - 1.5*iqr) & (df['FeH'] <= p99 + 1.5*iqr)]

    age_col = 'bayes.stellar.age_m_star'
    df['Age_Gyr'] = df[age_col] / 1000.0

    catalog  = pd.read_csv(a.catalog)
    ra_col   = 'RAJ2000' if 'RAJ2000' in catalog.columns else 'RA'
    dec_col  = 'DEJ2000' if 'DEJ2000' in catalog.columns else 'DEC'
    df['id_str']         = df['id'].astype(str).str.strip()
    catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
    df = pd.merge(df, catalog[[ra_col, dec_col, 'T17ID_str']],
                  left_on='id_str', right_on='T17ID_str', how='left')
    df = df.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
    print(f"  ✓ Coordenadas: {df['RA'].notna().sum()} objetos")

    feh_full = df['FeH'].values
    feh_err_full = df['FeH_err'].values
    print(f"  ✓ Muestra completa: {len(feh_full)} GCs | mediana σ[Fe/H]: {np.nanmedian(feh_err_full):.3f} dex")

    df_old  = df[df['Age_Gyr'] > 8.0].copy().reset_index(drop=True)
    feh_old = df_old['FeH'].values
    feh_err_old = df_old['FeH_err'].values
    print(f"  ✓ GCs viejos >8 Gyr: {len(feh_old)} ({len(feh_old)/len(feh_full)*100:.1f}%)")

    # ── 2. GMM ───────────────────────────────────────────────────────
    results_full, best_n_full = run_gmm_analysis(feh_full, 'All GCs')
    results_old, best_n_old = run_gmm_analysis(feh_old, 'Old GCs >8 Gyr')

    # ── 3. Clasificacion n=3 ─────────────────────────────────────────
    print("\n[4] Clasificacion n=3 — muestra completa:")
    (labels_full_n3, means_n3, stds_n3, weights_n3,
     seps_n3, _, bic_n3, aic_n3) = get_labels_forced_n3(feh_full)
    print("\n[4b] Clasificacion n=3 — GCs viejos:")
    (labels_old_n3, means_old_n3, stds_old_n3, weights_old_n3,
     seps_old_n3, _, bic_old_n3, aic_old_n3) = get_labels_forced_n3(feh_old)

    # ── 4. Bootstrap ─────────────────────────────────────────────────
    boot_full = bootstrap_analysis_with_quality(feh_full, feh_errors=feh_err_full,
                                                n_bootstrap=a.nboot, label='muestra completa')
    boot_old = bootstrap_analysis_with_quality(feh_old, feh_errors=feh_err_old,
                                               n_bootstrap=a.nboot, label='GCs viejos')

    # ── 5. Radios ─────────────────────────────────────────────────────
    radii_full, valid_full = calculate_projected_radii(df)
    df_sp_full     = df[valid_full].copy().reset_index(drop=True)
    labels_sp_full = labels_full_n3[valid_full.values]
    radii_old, valid_old = calculate_projected_radii(df_old)
    df_sp_old     = df_old[valid_old].copy().reset_index(drop=True)
    labels_sp_old = labels_old_n3[valid_old.values]
    print(f"\n  ✓ Radios: {radii_full.min():.1f} – {radii_full.max():.0f} kpc")

    # ── 6. Figuras ────────────────────────────────────────────────────
    plot_full_sample_metallicity_and_BIC(feh_full, results_full, best_n_full, outdir)
    plot_old_gcs_metallicity_and_BIC(feh_old, results_old, best_n_old, outdir)
    grad_all = plot_gradient_scatter(df_sp_full, labels_sp_full, radii_full,
                                     'Radial Metallicity Gradients\nAll GCs (n=3 forced)',
                                     'FullSample_Gradient_Scatter', outdir)
    plot_slope_comparison(grad_all, 'Slope Comparison\nAll GCs (n=3 forced)',
                          'Fig4_FullSample_Slope_Comparison', outdir)
    grad_old = plot_gradient_scatter(df_sp_old, labels_sp_old, radii_old,
                                     'Radial Metallicity Gradients\nOld GCs >8 Gyr (n=3 natural)',
                                     'OldGCs_Gradient_Scatter', outdir)
    plot_slope_comparison(grad_old, 'Slope Comparison\nOld GCs >8 Gyr (n=3 natural)',
                          'Fig6_OldGCs_Slope_Comparison', outdir)
    plot_binned_gradients_n3(df_sp_full, labels_sp_full, radii_full, outdir)
    plot_bootstrap_selection(boot_full, boot_old, len(feh_full), len(feh_old), outdir)

    # ── 7. Tablas ─────────────────────────────────────────────────────
    print_paper_table(results_full, best_n_full, results_old, best_n_old,
                      means_n3, stds_n3, weights_n3, labels_full_n3,
                      grad_all, grad_old, boot_full, boot_old, a.seed, a.nboot, outdir)

    # ── 8. JSON ───────────────────────────────────────────────────────
    json_data = {
        'version': 'gmm_final_paper_bootstrap_v9',
        'date': datetime.datetime.now().isoformat(),
        'seed': a.seed, 'n_bootstrap': a.nboot,
        'adopted_model': {
            'n_components': 3,
            'full_sample': {'means': [float(x) for x in means_n3],
                            'stds': [float(x) for x in stds_n3],
                            'weights': [float(x) for x in weights_n3],
                            'bic': float(bic_n3), 'aic': float(aic_n3),
                            'separations': [float(s) for s in seps_n3]},
            'old_gcs': {'means': [float(x) for x in means_old_n3],
                        'stds': [float(x) for x in stds_old_n3],
                        'weights': [float(x) for x in weights_old_n3],
                        'bic': float(bic_old_n3), 'aic': float(aic_old_n3),
                        'separations': [float(s) for s in seps_old_n3]},
        },
        'gmm_comparison': {
            'full_sample': {str(n): {'bic': float(r['bic']), 'aic': float(r['aic']),
                                     'delta_bic': float(r['bic']-results_full[best_n_full]['bic'])}
                            for n, r in results_full.items()},
            'old_gcs': {str(n): {'bic': float(r['bic']), 'aic': float(r['aic']),
                                 'delta_bic': float(r['bic']-results_old[best_n_old]['bic'])}
                        for n, r in results_old.items()},
        },
        'bootstrap_resample_perturbed': {
            'full_sample': {'counts': {str(k): int(v) for k, v in boot_full['counts'].items()},
                            'total': boot_full['total'], 'n_failed_fits': boot_full['n_failed_fits'],
                            'n3_stability': boot_full['n3_stability']},
            'old_gcs': {'counts': {str(k): int(v) for k, v in boot_old['counts'].items()},
                        'total': boot_old['total'], 'n_failed_fits': boot_old['n_failed_fits'],
                        'n3_stability': boot_old['n3_stability']},
        },
        'gradients_all_gcs': grad_all,
        'gradients_old_gcs': grad_old,
    }
    json_path = os.path.join(outdir, 'gmm_final_paper_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON: {json_path}")

    print("\n" + "="*70)
    print("RESUMEN DEL BOOTSTRAP (remuestreo + perturbacion), semilla", a.seed)
    for n in [2, 3, 4, 5]:
        p_f = boot_full['counts'][n] / boot_full['total'] * 100 if boot_full['total'] else 0
        p_o = boot_old['counts'][n] / boot_old['total'] * 100 if boot_old['total'] else 0
        print(f"  n={n}: todos {p_f:5.1f}%   viejos {p_o:5.1f}%")
    print("  (Apendice I original: todos 53.5/44.5/2.0 %; viejos 39.5/54.5/6.0 % para n=2/3/>=4)")
    print("="*70)


if __name__ == "__main__":
    main()
