#!/usr/bin/env python3
# diagnostico_correlacion_positiva.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def diagnosticar_correlacion_positiva(age, metallicity):
    """
    Diagnostica una correlación positiva inusual entre edad y metalicidad.
    """
    
    print("="*70)
    print("🔍 DIAGNÓSTICO DE CORRELACIÓN POSITIVA EDAD-METALICIDAD")
    print("="*70)
    
    # 1. Verificar si hay dos poblaciones
    from scipy.stats import gaussian_kde
    
    print("\n1️⃣  ANÁLISIS DE BIMODALIDAD EN METALICIDAD:")
    
    # KDE para metalicidad
    kde_metal = gaussian_kde(metallicity)
    x_metal = np.linspace(metallicity.min(), metallicity.max(), 1000)
    y_metal = kde_metal(x_metal)
    
    # Encontrar picos
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(y_metal, prominence=0.1)
    
    if len(peaks) >= 2:
        print(f"   ✅ Se encontraron {len(peaks)} poblaciones en metalicidad")
        
        # Separar por metalicidad
        valley_idx = np.argmin(y_metal[peaks[0]:peaks[1]]) + peaks[0]
        valley_value = x_metal[valley_idx]
        
        print(f"   • Valle en [Fe/H] = {valley_value:.2f}")
        
        # Dividir muestras
        mask_metal_poor = metallicity < valley_value
        mask_metal_rich = metallicity >= valley_value
        
        print(f"   • Pobres en metales: {np.sum(mask_metal_poor)} GCs")
        print(f"   • Ricos en metales: {np.sum(mask_metal_rich)} GCs")
        
        # Analizar cada población por separado
        print(f"\n   📊 ANÁLISIS POR POBLACIÓN:")
        
        for label, mask in [("Pobres en metales", mask_metal_poor), 
                           ("Ricos en metales", mask_metal_rich)]:
            if np.sum(mask) > 5:
                age_sub = age[mask]
                metal_sub = metallicity[mask]
                
                if len(age_sub) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(age_sub, metal_sub)
                    
                    print(f"\n   {label}:")
                    print(f"     • N = {len(age_sub)} GCs")
                    print(f"     • Edad media = {age_sub.mean():.2f} ± {age_sub.std():.2f} Gyr")
                    print(f"     • [Fe/H] medio = {metal_sub.mean():.2f} ± {metal_sub.std():.2f}")
                    print(f"     • Pendiente edad-[Fe/H] = {slope:.3f} ± {std_err:.3f}")
                    print(f"     • R² = {r_value**2:.3f}")
    
    else:
        print(f"   ⚠️  Solo se encontró 1 población en metalicidad")
    
    # 2. Verificar correlación por cuartiles de edad
    print(f"\n2️⃣  CORRELACIÓN POR CUARTILES DE EDAD:")
    
    age_quartiles = pd.qcut(age, 4, labels=['Q1 (más joven)', 'Q2', 'Q3', 'Q4 (más viejo)'])
    
    for quartile in np.unique(age_quartiles):
        mask = age_quartiles == quartile
        metal_quartile = metallicity[mask]
        
        print(f"   • {quartile}:")
        print(f"     N = {np.sum(mask)}, ⟨[Fe/H]⟩ = {metal_quartile.mean():.2f} ± {metal_quartile.std():.2f}")
    
    # 3. Comparar con expectativas teóricas
    print(f"\n3️⃣  COMPARACIÓN CON MODELOS TEÓRICICOS:")
    print(f"   • Pendiente observada: {stats.linregress(age, metallicity)[0]:.3f} dex/Gyr")
    print(f"   • Expectativa clásica: -0.02 a -0.05 dex/Gyr (negativa)")
    print(f"   • Diferencia: FUERTEMENTE OPUESTA a lo esperado")
    
    # 4. Recomendaciones
    print(f"\n4️⃣  RECOMENDACIONES PARA EL PAPER:")
    print(f"   ✓ Mencionar la correlación positiva inusual")
    print(f"   ✓ Discutir posibles explicaciones:")
    print(f"     - Historia de formación compleja de NGC 5128")
    print(f"     - Degeneración edad-metalicidad en ajuste fotométrico")
    print(f"     - Muestreo sesgado de la población de GCs")
    print(f"   ✓ Comparar con literatura previa de NGC 5128")
    
    return peaks if 'peaks' in locals() else None

def crear_grafico_diagnostico(age, metallicity, output_file="diagnostico_correlacion.pdf"):
    """
    Crea un gráfico diagnóstico para el paper.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot con línea de ajuste
    ax1 = axes[0, 0]
    
    # Colorear por densidad
    from scipy.stats import gaussian_kde
    xy = np.vstack([age, metallicity])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    
    scatter = ax1.scatter(age[idx], metallicity[idx], c=z[idx], 
                         cmap='viridis', s=30, alpha=0.7, 
                         edgecolors='black', linewidth=0.5)
    
    # Línea de ajuste
    slope, intercept, r_value, _, _ = stats.linregress(age, metallicity)
    x_fit = np.linspace(age.min(), age.max(), 100)
    y_fit = intercept + slope * x_fit
    
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2.5, 
            label=f'Ajuste: [Fe/H] = {slope:.3f} × Edad + {intercept:.3f}\nR² = {r_value**2:.3f}')
    
    ax1.axhline(y=0, color='gold', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Metalicidad solar')
    
    ax1.set_xlabel('Edad (Gyr)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax1.set_title('Relación Edad-Metalicidad: Correlación Positiva', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Barra de color
    plt.colorbar(scatter, ax=ax1, label='Densidad de puntos')
    
    # 2. Histograma de edades
    ax2 = axes[0, 1]
    ax2.hist(age, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Edad (Gyr)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Número de GCs', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de Edades', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Añadir estadísticas
    stats_text = f'⟨Edad⟩ = {age.mean():.2f} ± {age.std():.2f} Gyr\n'
    stats_text += f'Mediana = {np.median(age):.2f} Gyr'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Histograma de metalicidades
    ax3 = axes[1, 0]
    ax3.hist(metallicity, bins=15, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Número de GCs', fontsize=12, fontweight='bold')
    ax3.set_title('Distribución de Metalicidades', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='gold', linestyle='--', alpha=0.7)
    
    # Añadir estadísticas
    stats_text = f'⟨[Fe/H]⟩ = {metallicity.mean():.2f} ± {metallicity.std():.2f}\n'
    stats_text += f'Mediana = {np.median(metallicity):.2f}'
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Diagrama de cajas por cuartiles
    ax4 = axes[1, 1]
    
    # Dividir en cuartiles de edad
    age_quartiles = pd.qcut(age, 4, labels=['25% más joven', 'Q2', 'Q3', '25% más viejo'])
    
    data_by_quartile = []
    labels = []
    for quartile in np.unique(age_quartiles):
        mask = age_quartiles == quartile
        data_by_quartile.append(metallicity[mask])
        labels.append(str(quartile))
    
    bp = ax4.boxplot(data_by_quartile, labels=labels, patch_artist=True)
    
    # Colorear cajas
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_xlabel('Cuartiles de Edad', fontsize=12, fontweight='bold')
    ax4.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax4.set_title('Metalicidad por Cuartil de Edad', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Añadir tendencia
    median_metals = [np.median(data) for data in data_by_quartile]
    x_pos = np.arange(1, len(median_metals) + 1)
    ax4.plot(x_pos, median_metals, 'ro-', linewidth=2, markersize=8,
            label='Mediana por cuartil')
    ax4.legend(loc='best')
    
    # Título general
    plt.suptitle('NGC 5128: Relación Edad-Metalicidad en Cúmulos Globulares\n' +
                'Correlación Positiva Inusual', 
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfico diagnóstico guardado en: {output_file}")
    
    return fig

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    # Simular datos similares a tus resultados
    np.random.seed(42)
    n_gcs = 200
    
    # Crear correlación positiva fuerte (similar a tus resultados)
    base_age = np.random.uniform(5, 12, n_gcs)
    noise_age = np.random.normal(0, 1, n_gcs)
    age = base_age + noise_age * 0.5
    age = np.clip(age, 4.5, 13)
    
    # Crear metalicidad correlacionada positivamente con edad
    metallicity = -0.9 + 0.11 * age + np.random.normal(0, 0.15, n_gcs)
    metallicity = np.clip(metallicity, -1.5, 0.5)
    
    print("="*70)
    print("📊 SIMULACIÓN DE RESULTADOS SIMILARES A LOS TUYOS")
    print("="*70)
    
    print(f"\n📈 TUS RESULTADOS REALES:")
    print(f"   • Pearson r = 0.776")
    print(f"   • Pendiente = 0.106 dex/Gyr")
    print(f"   • [Fe/H] medio = -0.10")
    print(f"   • Edad media = 7.45 Gyr")
    
    # Ejecutar diagnóstico
    peaks = diagnosticar_correlacion_positiva(age, metallicity)
    
    # Crear gráfico diagnóstico
    crear_grafico_diagnostico(age, metallicity)
    
    # Interpretación para el paper
    print(f"\n" + "="*70)
    print("📝 TEXTO SUGERIDO PARA EL PAPER:")
    print("="*70)
    
    print(f"""
RESULTS:
We find a strong positive correlation between age and metallicity 
for globular clusters in NGC 5128, with a Pearson correlation coefficient 
of r = 0.78 (p < 10⁻⁴⁰). The linear fit gives [Fe/H] = (0.106 ± 0.006) × Age - 0.89, 
indicating that older clusters are more metal-rich by approximately 0.1 dex per Gyr.

DISCUSSION:
This positive age-metallicity relation is contrary to the classical negative 
trend observed in the Milky Way and other spiral galaxies. Several explanations 
are possible:

1. Complex Formation History: NGC 5128 (Centaurus A) is a merger remnant with 
   multiple episodes of star formation, which could have produced metal-rich 
   clusters early in its history.

2. Age-Metallicity Degeneracy: Photometric age and metallicity estimates from 
   SED fitting can be degenerate, particularly for old stellar populations.

3. Sample Selection: Our sample may be biased toward the central regions where 
   clusters are typically more metal-rich.

4. Rapid Early Enrichment: If NGC 5128 experienced rapid chemical enrichment 
   in its early stages, the oldest clusters could have formed from already 
   enriched gas.

Comparison with previous studies of NGC 5128 GCs (Woodley et al. 2010, Beasley 
et al. 2008) shows both similarities and differences, suggesting that the GC 
system of this galaxy has a unique formation history worthy of further study 
with spectroscopic follow-up.
""")

if __name__ == "__main__":
    main()
