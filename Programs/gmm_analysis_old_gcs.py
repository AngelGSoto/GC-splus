#!/usr/bin/env python3
# gmm_analysis_old_GCs_v2.py
# Análisis GMM para cúmulos globulares viejos (>8 Gyr) en NGC 5128
# VERSIÓN 2: BIC mínimo como criterio primario de selección del modelo.
# Separaciones y membership se reportan como métricas descriptivas (no filtros).
# Estilo de figuras idéntico al original del paper.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from astropy.table import Table
import os
import datetime
import json

# ============================================================================
# CONFIGURACIÓN DE ESTILO (idéntico al original)
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

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_bayes_factor(bic_model1, bic_model2):
    """
    BF(M1 vs M2) = exp((BIC2 - BIC1) / 2)
    BF > 1 → M1 más probable
    BF < 1 → M2 más probable
    """
    return np.exp((bic_model2 - bic_model1) / 2)


def interpret_bayes_factor(bf):
    """Escala de Kass & Raftery (1995)."""
    if bf > 1:
        if bf > 150: return "Evidencia decisiva a favor del primer modelo"
        elif bf > 20: return "Evidencia fuerte a favor del primer modelo"
        elif bf > 3:  return "Evidencia sustancial a favor del primer modelo"
        else:         return "No hay evidencia significativa"
    else:
        bf_inv = 1 / bf
        if bf_inv > 150: return "Evidencia decisiva a favor del segundo modelo"
        elif bf_inv > 20: return "Evidencia fuerte a favor del segundo modelo"
        elif bf_inv > 3:  return "Evidencia sustancial a favor del segundo modelo"
        else:             return "No hay evidencia significativa"


def validate_gmm_model(feh_data, n_components):
    """Ajusta GMM con múltiples inicializaciones y devuelve el mejor."""
    X = feh_data.reshape(-1, 1)
    best_gmm, best_score = None, -np.inf
    for seed in [42, 123, 456, 789, 999]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed,
                              n_init=5, max_iter=500, tol=1e-4)
        gmm.fit(X)
        score = gmm.score(X)
        if score > best_score:
            best_score = score
            best_gmm   = gmm
    return best_gmm

# ============================================================================
# ANÁLISIS GMM — CRITERIO PRINCIPAL: BIC MÍNIMO
# ============================================================================

def perform_robust_gmm_analysis(feh_data):
    """
    Análisis GMM robusto para GCs viejos.
    CRITERIO DE SELECCIÓN: BIC mínimo (criterio primario bayesiano).
    Separaciones y membership se reportan como métricas descriptivas.
    """
    X = feh_data.reshape(-1, 1)

    print("\n" + "="*60)
    print("ANÁLISIS GMM PARA GCs VIEJOS (>8 Gyr)")
    print("Criterio de selección: BIC mínimo")
    print("="*60)

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

        # Calidad: trabajar con índices originales y luego reordenar
        quality = []
        for i_orig in range(n):
            mask = labels == i_orig
            quality.append(probs[mask, i_orig].mean() if mask.sum() > 0 else 0.0)
        quality = [quality[i] for i in sort_idx]

        separations = []
        if n > 1:
            for i in range(n - 1):
                sep = abs(means[i+1] - means[i]) / np.sqrt(stds[i]**2 + stds[i+1]**2)
                separations.append(sep)

        results[n] = {
            'bic': bic,
            'aic': aic,
            'gmm': gmm,
            'means': means,
            'stds': stds,
            'weights': weights,
            'quality': quality,
            'separations': separations,
            'all_greater_2sigma': all(s > 2 for s in separations) if separations else True,
            'min_quality': float(min(quality)) if quality else 0.0,
            'avg_quality': float(np.mean(quality)),
            'sort_idx': sort_idx,
        }

        print(f"\n  n={n}: BIC={bic:.1f}, AIC={aic:.1f}")
        print(f"    Calidad media: {np.mean(quality):.3f}, mínima: {min(quality):.3f}")
        print(f"    Separaciones (σ): {[f'{s:.2f}' for s in separations]}")
        print(f"    Todas >2σ (informativo): {all(s > 2 for s in separations) if separations else 'N/A'}")

    # ----------------------------------------------------------------
    # SELECCIÓN: BIC MÍNIMO (criterio primario)
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("SELECCIÓN DEL MODELO ÓPTIMO (criterio: BIC mínimo)")
    print("="*60)

    best_n      = min(results.keys(), key=lambda x: results[x]['bic'])
    best_reason = f"BIC mínimo entre todos los modelos evaluados (BIC={results[best_n]['bic']:.1f})"

    print(f"\n✅ MODELO SELECCIONADO: n = {best_n}")
    print(f"   Razón: {best_reason}")

    # Tabla comparativa
    print(f"\n{'n':>4} {'BIC':>10} {'ΔBIC':>10} {'AIC':>10} {'Min sep(σ)':>12} {'All>2σ':>8} {'Min quality':>12}")
    print("-"*70)
    for n in sorted(results.keys()):
        delta   = results[n]['bic'] - results[best_n]['bic']
        sep_str = f"{min(results[n]['separations']):.2f}" if results[n]['separations'] else "N/A"
        marker  = " ← BEST" if n == best_n else ""
        print(f"  {n:>2}  {results[n]['bic']:>10.1f}  {delta:>+10.1f}  "
              f"{results[n]['aic']:>10.1f}  {sep_str:>12}  "
              f"{str(results[n]['all_greater_2sigma']):>8}  "
              f"{results[n]['min_quality']:>12.3f}{marker}")

    # Factores de Bayes
    print(f"\n📊 FACTORES DE BAYES (referencia: n={best_n}):")
    for n in sorted(results.keys()):
        if n != best_n:
            delta = results[n]['bic'] - results[best_n]['bic']
            bf    = calculate_bayes_factor(results[best_n]['bic'], results[n]['bic'])
            interp = interpret_bayes_factor(bf)
            print(f"  n={best_n} vs n={n}: ΔBIC={delta:+.1f}, BF={bf:.2e} → {interp}")

    # Nombres y colores según número de componentes
    if best_n == 2:
        pop_names = ['Metal-Poor', 'Metal-Rich']
        pop_short = ['MP', 'MR']
        colors    = ['#2E86AB', '#C73E1D']
    elif best_n == 3:
        pop_names = ['Metal-Poor', 'Intermediate', 'Metal-Rich']
        pop_short = ['MP', 'Int', 'MR']
        colors    = ['#2E86AB', '#A23B72', '#C73E1D']
    elif best_n == 4:
        pop_names = ['Metal-Poor', 'Intermediate-Low', 'Intermediate-High', 'Metal-Rich']
        pop_short = ['MP', 'IL', 'IH', 'MR']
        colors    = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    else:
        pop_names = [f'Pop{i+1}' for i in range(best_n)]
        pop_short = [f'P{i+1}' for i in range(best_n)]
        cmap      = plt.colormaps['viridis']
        colors    = [cmap(i / (best_n - 1)) for i in range(best_n)]

    best     = results[best_n]
    sort_idx = best['sort_idx']
    label_map = {int(i_orig): int(i_sorted) for i_sorted, i_orig in enumerate(sort_idx)}

    gmm            = best['gmm']
    labels_orig    = gmm.predict(X)
    labels_ordered = np.array([label_map[l] for l in labels_orig])

    print(f"\n📊 PARÁMETROS FINALES ({best_n} POBLACIONES):")
    for i in range(best_n):
        n_obj = np.sum(labels_ordered == i)
        print(f"  • {pop_names[i]} ({pop_short[i]}):")
        print(f"    [Fe/H] = {best['means'][i]:.3f} ± {best['stds'][i]:.3f} dex")
        print(f"    Fracción = {best['weights'][i]:.3f} ({best['weights'][i]*100:.1f}%)")
        print(f"    N = {n_obj}    Calidad = {best['quality'][i]:.3f}")

    if best_n > 1:
        print(f"\n📈 SEPARACIONES (métricas descriptivas):")
        for i, sep in enumerate(best['separations']):
            sig = "≥2σ (bien separadas)" if sep >= 2 else "<2σ (solapadas)"
            print(f"  • {pop_short[i]}-{pop_short[i+1]}: {sep:.2f}σ → {sig}")

    return (gmm, labels_ordered, best['means'], best['stds'], best['weights'],
            results, best_n, pop_names, pop_short, colors, sort_idx)

# ============================================================================
# FIGURAS — ESTILO IDÉNTICO AL PAPER ORIGINAL
# ============================================================================

def plot_metallicity_distribution(feh_data, results, best_n, pop_names,
                                  pop_short, colors, output_dir):
    """Figura 1: Distribución de metalicidad con GMM."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    best    = results[best_n]

    ax.hist(feh_data, bins=30, density=True, alpha=0.3,
            color='lightgray', edgecolor='black', linewidth=0.8,
            label=f'Old GCs >8 Gyr (N={len(feh_data)})')

    x         = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 1000)
    pdf_total = np.zeros_like(x)

    for i in range(best_n):
        pdf_comp   = best['weights'][i] * norm.pdf(x, best['means'][i], best['stds'][i])
        pdf_total += pdf_comp
        ax.plot(x, pdf_comp, '--', linewidth=2.5, alpha=0.8, color=colors[i],
                label=f'{pop_short[i]}: μ={best["means"][i]:.2f}±{best["stds"][i]:.2f}')

    ax.plot(x, pdf_total, 'k-', linewidth=3.5, label='GMM fit', alpha=0.9)

    for i in range(best_n):
        x_fill   = np.linspace(best['means'][i] - best['stds'][i],
                               best['means'][i] + best['stds'][i], 100)
        pdf_fill = best['weights'][i] * norm.pdf(x_fill, best['means'][i], best['stds'][i])
        ax.fill_between(x_fill, 0, pdf_fill, color=colors[i], alpha=0.2)

    ax.set_xlabel('[Fe/H] (dex)', fontsize=24)
    ax.set_ylabel('Probability Density', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=16, loc='upper center', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Texto informativo
    sep_str  = ', '.join([f'{s:.2f}σ' for s in best['separations']]) if best['separations'] else 'N/A'
    text_str  = f"Old GCs (age > 8 Gyr)\n"
    text_str += f"N = {len(feh_data)}\n"
    text_str += f"n = {best_n} components (min. BIC)\n"
    text_str += f"BIC = {best['bic']:.1f}\n"
    text_str += f"AIC = {best['aic']:.1f}\n"
    text_str += f"Separations = {sep_str}"

    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            fontsize=15, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    base = os.path.join(figures_dir, 'Metallicity_Distribution_GMM_OldGCs')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura 1 guardada: {base}.pdf")


def plot_model_selection_criteria(results, best_n, output_dir):
    """
    Figura 2: BIC y AIC para todos los modelos.
    El modelo con BIC mínimo se destaca.
    Las separaciones mínimas se anotan sobre cada punto.
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, ax   = plt.subplots(figsize=(10, 7))
    n_values  = sorted(results.keys())
    bic_values = [results[n]['bic'] for n in n_values]
    aic_values = [results[n]['aic'] for n in n_values]

    for i, n in enumerate(n_values):
        is_best = (n == best_n)
        color   = '#2ca02c' if is_best else '#7f7f7f'
        marker  = 'o' if is_best else 's'
        size    = 180 if is_best else 90
        lw      = 3 if is_best else 1.5

        ax.scatter(n, bic_values[i], s=size, color=color, marker=marker,
                   edgecolors='black', linewidth=lw, zorder=6)
        ax.scatter(n, aic_values[i], s=size, color=color, marker=marker,
                   edgecolors='black', linewidth=lw, zorder=6, alpha=0.6)

        # Anotar separación mínima sobre cada punto BIC
        if results[n]['separations']:
            min_sep = min(results[n]['separations'])
            ax.annotate(f'{min_sep:.2f}σ',
                        xy=(n, bic_values[i]),
                        xytext=(0, 14), textcoords='offset points',
                        ha='center', fontsize=11, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.plot(n_values, bic_values, '--', color='#D55E00', linewidth=2.5, alpha=0.8)
    ax.plot(n_values, aic_values, '--', color='#0072B2', linewidth=2.5, alpha=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#D55E00', linestyle='--', linewidth=2.5, label='BIC (Bayesian)'),
        Line2D([0], [0], color='#0072B2', linestyle='--', linewidth=2.5, label='AIC (Akaike)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=14, markeredgewidth=2, markeredgecolor='black',
               label=f'Selected model (n={best_n}, min. BIC)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f7f7f',
               markersize=10, markeredgewidth=1.5, markeredgecolor='black',
               label='Other models'),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc='upper right', framealpha=0.9)
    ax.set_xlabel('Number of Gaussian Components', fontsize=24)
    ax.set_ylabel('Information Criterion', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xticks(n_values)

    ax.text(0.01, 0.01,
            'Numbers above BIC points indicate minimum component separation (σ)',
            transform=ax.transAxes, fontsize=11, color='gray',
            verticalalignment='bottom')

    plt.tight_layout()
    base = os.path.join(figures_dir, 'Model_Selection_Criteria_OldGCs')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura 2 guardada: {base}.pdf")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ANÁLISIS GMM PARA GCs VIEJOS (>8 Gyr) — VERSIÓN 2")
    print("Criterio de selección: BIC mínimo (criterio primario bayesiano)")
    print("Separaciones y membership: métricas descriptivas (no filtros)")
    print("="*70)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = f"GMM_OldGCs_v2_{timestamp}"
    os.makedirs(outdir, exist_ok=True)
    print(f"📁 Resultados en: {outdir}")

    # 1. Cargar datos
    print("\n[1] Cargando datos desde results.fits...")
    try:
        table = Table.read('out/results.fits', hdu=1)
        df    = table.to_pandas()
    except Exception as e:
        print(f"❌ Error cargando results.fits: {e}")
        return

    # Verificar columna de edad
    age_col = 'bayes.stellar.age_m_star'
    if age_col not in df.columns:
        print(f"❌ Columna '{age_col}' no encontrada. Abortando.")
        return
    print(f"   Columna de edad: {age_col}")

    # 2. Filtrar por calidad y calcular [Fe/H]
    df       = df[df['best.reduced_chi_square'] <= 2.0]
    Z_solar  = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)

    # 3. Convertir edad de Myr a Gyr
    df['Age_Gyr'] = df[age_col] / 1000.0

    edad_clean = df['Age_Gyr'].dropna()
    print(f"\n   Estadísticas de edad (muestra completa, Gyr):")
    print(f"     Min: {edad_clean.min():.2f}   Max: {edad_clean.max():.2f}")
    print(f"     Mediana: {edad_clean.median():.2f}   Media: {edad_clean.mean():.2f}")

    # 4. Filtrar GCs viejos (>8 Gyr)
    df_old = df[df['Age_Gyr'] > 8.0].copy()
    if len(df_old) == 0:
        print("❌ No hay GCs con edad > 8 Gyr. Abortando.")
        return
    print(f"\n   ✓ {len(df_old)} GCs seleccionados (edad > 8 Gyr)")
    print(f"   ({len(df_old)/len(df)*100:.1f}% de la muestra con chi2 ≤ 2)")

    # 5. Filtrar outliers en [Fe/H]
    Q1 = df_old['FeH'].quantile(0.01)
    Q3 = df_old['FeH'].quantile(0.99)
    df_old = df_old[(df_old['FeH'] >= Q1 - 1.5*(Q3-Q1)) &
                    (df_old['FeH'] <= Q3 + 1.5*(Q3-Q1))]
    print(f"   Tras filtro IQR: {len(df_old)} GCs")

    feh_data = df_old['FeH'].values
    print(f"   Rango [Fe/H]: [{feh_data.min():.2f}, {feh_data.max():.2f}]")
    print(f"   Media ± σ: {feh_data.mean():.3f} ± {feh_data.std():.3f}")

    # 6. Análisis GMM
    print("\n[2] Ejecutando análisis GMM...")
    (gmm, labels, means, stds, weights, results,
     best_n, pop_names, pop_short, colors, sort_idx) = perform_robust_gmm_analysis(feh_data)

    # 7. Figuras
    print("\n[3] Generando figuras...")
    plot_metallicity_distribution(feh_data, results, best_n, pop_names,
                                  pop_short, colors, outdir)
    plot_model_selection_criteria(results, best_n, outdir)

    # 8. Guardar catálogo clasificado
    print("\n[4] Guardando resultados...")
    df_out = df_old.copy()
    df_out['GMM_Population']   = labels
    df_out['Population_Label'] = [pop_short[l] for l in labels]

    probs = gmm.predict_proba(feh_data.reshape(-1, 1))
    for i_sorted, i_orig in enumerate(sort_idx):
        df_out[f'Prob_{pop_short[i_sorted]}'] = probs[:, i_orig]

    csv_path = os.path.join(outdir, f'old_GCs_classified_{best_n}pop.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"✓ CSV guardado: {csv_path}")

    # 9. Guardar parámetros en JSON
    params = {
        'version': 'gmm_old_GCs_v2_bic_primary',
        'selection_criterion': 'Minimum BIC (Bayesian Information Criterion)',
        'note': ('Separations and membership probabilities are descriptive metrics, '
                 'not used as rejection filters.'),
        'metadata': {
            'date': datetime.datetime.now().isoformat(),
            'age_cut_Gyr': 8.0,
            'age_column': age_col,
            'n_objects_total_chi2cut': len(df),
            'n_objects_old': len(df_old),
            'feh_range': [float(feh_data.min()), float(feh_data.max())],
        },
        'selected_model': {
            'n_components': best_n,
            'pop_names': pop_names,
            'pop_short': pop_short,
            'means':   [float(x) for x in means],
            'stds':    [float(x) for x in stds],
            'weights': [float(x) for x in weights],
            'bic': float(results[best_n]['bic']),
            'aic': float(results[best_n]['aic']),
            'separations_sigma': [float(s) for s in results[best_n]['separations']],
            'quality': [float(q) for q in results[best_n]['quality']],
        },
        'all_models': {}
    }

    for n, r in results.items():
        delta = r['bic'] - results[best_n]['bic']
        bf    = calculate_bayes_factor(results[best_n]['bic'], r['bic']) if n != best_n else 1.0
        params['all_models'][f'n{n}'] = {
            'bic': float(r['bic']),
            'aic': float(r['aic']),
            'delta_bic': float(delta),
            'bayes_factor_vs_best': float(bf),
            'separations_sigma': [float(s) for s in r['separations']],
            'all_sep_gt_2sigma': bool(r['all_greater_2sigma']),
            'min_quality': float(r['min_quality']),
            'is_selected': (n == best_n),
        }

    json_path = os.path.join(outdir, f'gmm_parameters_oldGCs_{best_n}pop.json')
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON guardado: {json_path}")

    # 10. Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"  Muestra: {len(df_old)} GCs viejos (edad > 8 Gyr)")
    print(f"  Modelo seleccionado: n = {best_n} (BIC = {results[best_n]['bic']:.1f})")
    print(f"\n  Comparación de modelos:")
    for n in sorted(results.keys()):
        delta  = results[n]['bic'] - results[best_n]['bic']
        marker = " ← SELECCIONADO" if n == best_n else ""
        print(f"    n={n}: BIC={results[n]['bic']:.1f}  (ΔBIC={delta:+.1f}){marker}")

    print(f"\n  Parámetros del modelo seleccionado:")
    for i in range(best_n):
        n_obj = np.sum(labels == i)
        print(f"    • {pop_names[i]}: [Fe/H]={means[i]:.3f}±{stds[i]:.3f}, "
              f"f={weights[i]*100:.1f}%, N={n_obj}")

    print(f"\n  Separaciones (descriptivas):")
    for i, sep in enumerate(results[best_n]['separations']):
        sig = "≥2σ" if sep >= 2 else "<2σ"
        print(f"    • {pop_short[i]}-{pop_short[i+1]}: {sep:.2f}σ ({sig})")

    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()
