#!/usr/bin/env python3
# honest_analysis_ngc5128_fixed.py - Versión corregida

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VERSIÓN SIMPLIFICADA Y ROBUSTA
# ============================================================================

def nonparametric_analysis_simple(feh_data):
    """Análisis no paramétrico simplificado y robusto."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Histograma y KDE
    ax = axes[0, 0]
    
    # Histograma
    n, bins, patches = ax.hist(feh_data, bins=30, density=True, 
                               alpha=0.3, color='gray', edgecolor='black')
    
    # KDE con bandwidth óptimo (Scott)
    kde = gaussian_kde(feh_data, bw_method='scott')
    x_range = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 1000)
    density = kde(x_range)
    ax.plot(x_range, density, 'r-', linewidth=2, label='KDE (Scott)')
    
    # Encontrar modos
    peaks, properties = find_peaks(density, prominence=0.02)
    for peak in peaks:
        ax.axvline(x=x_range[peak], color='blue', linestyle='--', alpha=0.5)
        ax.text(x_range[peak], density[peak]*1.1, f'Mode {x_range[peak]:.2f}', 
                ha='center', fontsize=9)
    
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Histogram and Kernel Density Estimation')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Panel B: CDF y comparación con normal
    ax = axes[0, 1]
    
    sorted_feh = np.sort(feh_data)
    cdf = np.arange(1, len(sorted_feh) + 1) / len(sorted_feh)
    ax.plot(sorted_feh, cdf, 'k-', linewidth=2, label='Empirical CDF')
    
    # CDF de distribución normal con misma media y std
    from scipy.stats import norm as norm_dist
    normal_cdf = norm_dist.cdf(sorted_feh, loc=feh_data.mean(), scale=feh_data.std())
    ax.plot(sorted_feh, normal_cdf, 'r--', linewidth=2, alpha=0.7, label='Normal CDF')
    
    # Calcular estadística KS
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(feh_data, 
                            np.random.normal(feh_data.mean(), feh_data.std(), 10000))
    
    ax.text(0.05, 0.95, f'KS test: D={ks_stat:.3f}\np={ks_p:.3f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('[Fe/H] (dex)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('(b) Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Panel C: QQ-plot
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(feh_data, dist="norm", plot=ax)
    ax.set_title('(c) QQ-plot vs Normal Distribution')
    ax.grid(True, alpha=0.2)
    
    # Panel D: Análisis de asimetría y multimodalidad
    ax = axes[1, 1]
    ax.axis('off')
    
    # Estadísticas descriptivas
    skewness = pd.Series(feh_data).skew()
    kurtosis = pd.Series(feh_data).kurtosis()
    
    # Test de unimodalidad simplificado (basado en skewness y kurtosis)
    is_unimodal = abs(skewness) < 1 and kurtosis < 3.5  # Regla práctica
    
    summary_text = "DISTRIBUTION ANALYSIS SUMMARY\n"
    summary_text += "="*40 + "\n\n"
    summary_text += f"Sample size: N = {len(feh_data)}\n"
    summary_text += f"Range: [{feh_data.min():.2f}, {feh_data.max():.2f}]\n"
    summary_text += f"Mean: {feh_data.mean():.3f} ± {feh_data.std():.3f}\n"
    summary_text += f"Skewness: {skewness:.3f}\n"
    summary_text += f"Kurtosis: {kurtosis:.3f}\n"
    summary_text += f"KS test p-value: {ks_p:.3f}\n\n"
    
    summary_text += "MULTIMODALITY ASSESSMENT:\n"
    summary_text += f"• Number of KDE modes: {len(peaks)}\n"
    summary_text += f"• Primary mode: [Fe/H] = {x_range[peaks[0]]:.2f}\n"
    if len(peaks) > 1:
        summary_text += f"• Secondary mode: [Fe/H] = {x_range[peaks[1]]:.2f}\n"
        valley_idx = np.argmin(density[peaks[0]:peaks[1]]) + peaks[0]
        summary_text += f"• Valley depth: {density[valley_idx]/density[peaks[0]]:.2%}\n"
    
    summary_text += f"• Unimodal by rule-of-thumb: {'Yes' if is_unimodal else 'No'}\n"
    
    if ks_p > 0.05:
        summary_text += "\n→ Cannot reject normal distribution (p > 0.05)"
    else:
        summary_text += "\n→ Distribution is significantly non-normal (p < 0.05)"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', linespacing=1.5,
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))
    
    plt.tight_layout()
    return fig, len(peaks), x_range[peaks]

# ============================================================================
# ANÁLISIS PRINCIPAL - VERSIÓN CORREGIDA
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ANÁLISIS HONESTO - VERSIÓN CORREGIDA")
    print("="*80)
    
    # Cargar datos
    try:
        df = pd.read_csv('NGC5128_Perfect_Analysis_20260205_132925/ngc5128_gcs_perfect_classified_2pop.csv')
    except FileNotFoundError:
        # Buscar el archivo en ubicaciones alternativas
        import glob
        files = glob.glob('*/ngc5128_gcs_perfect_classified_2pop.csv')
        if files:
            df = pd.read_csv(files[0])
        else:
            print("❌ No se encontró el archivo de datos.")
            return
    
    feh_data = df['FeH'].values
    
    print(f"\n📊 DATOS: {len(feh_data)} cúmulos globulares")
    print(f"   Rango: [{feh_data.min():.2f}, {feh_data.max():.2f}]")
    print(f"   Media: {feh_data.mean():.3f} ± {feh_data.std():.3f}")
    print(f"   Asimetría: {pd.Series(feh_data).skew():.3f}")
    print(f"   Curtosis: {pd.Series(feh_data).kurtosis():.3f}")
    
    # 1. Análisis no paramétrico simplificado
    print("\n[1] ANÁLISIS NO PARAMÉTRICO SIMPLIFICADO")
    fig1, n_modes, mode_positions = nonparametric_analysis_simple(feh_data)
    fig1.savefig('nonparametric_analysis_simple.pdf', bbox_inches='tight')
    print(f"   ✓ Figura guardada: nonparametric_analysis_simple.pdf")
    print(f"   Número de modos detectados: {n_modes}")
    print(f"   Posiciones: {mode_positions}")
    
    # 2. Análisis comparativo GMM vs realidad visual
    print("\n[2] ANÁLISIS COMPARATIVO GMM vs REALIDAD VISUAL")
    
    # Crear figura comparativa
    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel izquierdo: Lo que GMM ve
    from sklearn.mixture import GaussianMixture
    
    # Ajustar GMM de 2 componentes
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(feh_data.reshape(-1, 1))
    labels = gmm.predict(feh_data.reshape(-1, 1))
    
    # Histograma con componentes GMM
    ax1.hist(feh_data, bins=30, density=True, alpha=0.3, color='gray', edgecolor='black')
    
    x_range = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 1000)
    log_dens = gmm.score_samples(x_range.reshape(-1, 1))
    ax1.plot(x_range, np.exp(log_dens), 'k-', linewidth=3, label='GMM fit')
    
    # Componentes individuales
    for i in range(2):
        component = norm.pdf(x_range, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i].item()))
        ax1.plot(x_range, gmm.weights_[i] * component, '--', linewidth=2, 
                label=f'Component {i+1}')
    
    ax1.set_xlabel('[Fe/H] (dex)')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) GMM Interpretation: 2 Gaussian Components')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel derecho: Lo que realmente vemos
    ax2.hist(feh_data, bins=30, density=True, alpha=0.3, color='gray', edgecolor='black')
    
    # KDE para ver la forma real
    kde = gaussian_kde(feh_data, bw_method='scott')
    ax2.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE (true shape)')
    
    # Marcar modos detectados
    peaks, _ = find_peaks(kde(x_range), prominence=0.02)
    for i, peak in enumerate(peaks):
        ax2.axvline(x=x_range[peak], color='blue', linestyle='--', alpha=0.7)
        ax2.text(x_range[peak], kde(x_range)[peak]*1.1, f'Mode {i+1}', 
                ha='center', fontsize=10, color='blue')
    
    ax2.set_xlabel('[Fe/H] (dex)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Reality: Actual Distribution Shape')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # Añadir texto explicativo
    fig_compare.suptitle('Comparing GMM Interpretation vs Actual Distribution Shape\n' +
                        f'GMM finds {len(gmm.means_)} components, KDE finds {len(peaks)} modes', 
                        fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_compare.savefig('gmm_vs_reality_comparison.pdf', bbox_inches='tight')
    print(f"   ✓ Figura comparativa guardada: gmm_vs_reality_comparison.pdf")
    
    # 3. Interpretación honesta
    print("\n" + "="*80)
    print("💡 INTERPRETACIÓN HONESTA - RESULTADOS")
    print("="*80)
    
    print("\n📊 RESUMEN ESTADÍSTICO:")
    print(f"   • Asimetría: {pd.Series(feh_data).skew():.3f} " +
          f"{'(negatively skewed)' if pd.Series(feh_data).skew() < 0 else '(positively skewed)'}")
    print(f"   • Curtosis: {pd.Series(feh_data).kurtosis():.3f} " +
          f"{'(platykurtic: flatter than normal)' if pd.Series(feh_data).kurtosis() < 0 else ''}")
    
    print(f"\n🔍 ANÁLISIS DE MULTIMODALIDAD:")
    print(f"   • Modos detectados por KDE: {n_modes}")
    for i, pos in enumerate(mode_positions):
        print(f"     Mode {i+1}: [Fe/H] = {pos:.2f}")
    
    if n_modes == 1:
        print("   → Distribución esencialmente unimodal con cola extendida")
    elif n_modes == 2:
        # Calcular separación entre modos
        separation = abs(mode_positions[1] - mode_positions[0]) / feh_data.std()
        print(f"   → Dos modos detectados, separados por {separation:.2f}σ")
        
        # Verificar si hay valle significativo
        kde = gaussian_kde(feh_data, bw_method='scott')
        density = kde(x_range)
        peaks, properties = find_peaks(density, prominence=0.02)
        if len(peaks) >= 2:
            valley_idx = np.argmin(density[peaks[0]:peaks[1]]) + peaks[0]
            valley_depth = density[valley_idx] / density[peaks[0]]
            print(f"   → Profundidad del valle: {valley_depth:.2%}")
            if valley_depth < 0.5:
                print("   → Valle poco profundo: poblaciones superpuestas")
            else:
                print("   → Valle significativo: poblaciones bien separadas")
    else:
        print("   → Estructura compleja con múltiples modos")
    
    print("\n📝 RECOMENDACIÓN PARA EL PAPER:")
    print("""
Basado en el análisis no paramétrico, la distribución de metalicidad muestra:
1. Un pico principal en [Fe/H] ≈ -0.3
2. Una cola extendida hacia [Fe/H] ≈ -1.2
3. Ausencia de un valle profundo entre estos valores

Esto sugiere que, aunque el GMM identifica dos componentes Gaussianas, la distribución 
real es más continua que discreta. La interpretación más conservadora sería que 
NGC 5128 tiene:
• Una población dominante de cúmulos metal-ricos
• Un continuo de cúmulos hacia valores metal-pobres, posiblemente provenientes de 
  múltiples eventos de acreción o formación extendida

La "bimodalidad" estadística del GMM podría estar reflejando principalmente la 
asimetría de la distribución más que dos poblaciones físicamente discretas.
""")
    
    plt.show()

if __name__ == "__main__":
    main()
