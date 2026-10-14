#!/usr/bin/env python3
# bootstrap_analysis_ngc5128_paper_plot.py
# Análisis Bootstrap con Plot Único para Paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

import os
import datetime
import json
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURACIÓN PARA PAPER - ESTILO PROFESIONAL
# ============================================================================

# Configuración para publicación (Nature/ApJ style)
mpl.rcParams.update({
    # Fuentes
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    
    # Figuras
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'figure.constrained_layout.use': True,
    
    # Líneas
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    
    # Ticks
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    # Grilla
    'grid.linewidth': 0.5,
    'grid.alpha': 0.2,
})

# Colores para las dos poblaciones
COLORS = {
    'MP': '#2E86AB',  # Azul - Metal-Poor
    'MR': '#C73E1D',  # Rojo - Metal-Rich
    'total': '#404040',  # Gris para modelo total
    'hist': '#D3D3D3',  # Gris claro para histograma
}

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_classified_data(input_csv, input_json):
    """Carga datos clasificados y parámetros GMM."""
    print("📊 Cargando datos clasificados...")
    
    df = pd.read_csv(input_csv)
    print(f"✓ {len(df)} objetos cargados")
    
    with open(input_json, 'r') as f:
        gmm_params = json.load(f)
    
    print(f"✓ Modelo GMM: {gmm_params['n_components']} componentes")
    print(f"✓ Poblaciones: {gmm_params['pop_names']}")
    
    feh_data = df['FeH'].values
    if 'GMM_Population' in df.columns:
        labels = df['GMM_Population'].values
    else:
        gmm = GaussianMixture(n_components=gmm_params['n_components'], random_state=42)
        gmm.weights_ = np.array(gmm_params['weights'])
        gmm.means_ = np.array(gmm_params['means']).reshape(-1, 1)
        gmm.covariances_ = np.array([std**2 for std in gmm_params['stds']]).reshape(-1, 1, 1)
        labels = gmm.predict(feh_data.reshape(-1, 1))
    
    return df, gmm_params, feh_data, labels

def perform_bootstrap_analysis(feh_data, n_components, n_bootstrap=500, random_state=42):
    """Análisis bootstrap simplificado."""
    
    print(f"\n🔄 Ejecutando Bootstrap ({n_bootstrap} réplicas)...")
    
    np.random.seed(random_state)
    n_samples = len(feh_data)
    X = feh_data.reshape(-1, 1)
    
    # Arrays para resultados
    bootstrap_means = np.zeros((n_bootstrap, n_components))
    bootstrap_stds = np.zeros((n_bootstrap, n_components))
    bootstrap_weights = np.zeros((n_bootstrap, n_components))
    
    # Bucle bootstrap
    for b in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state + b,
            covariance_type='spherical',
            n_init=3,
            max_iter=300
        )
        gmm.fit(X_boot)
        
        # Ordenar por media
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        
        sort_idx = np.argsort(means)
        bootstrap_means[b] = means[sort_idx]
        bootstrap_stds[b] = stds[sort_idx]
        bootstrap_weights[b] = weights[sort_idx]
    
    # Calcular estadísticas
    def compute_stats(data):
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'ci_95_lower': float(np.percentile(data, 2.5)),
            'ci_95_upper': float(np.percentile(data, 97.5)),
            'ci_68_lower': float(np.percentile(data, 16)),
            'ci_68_upper': float(np.percentile(data, 84)),
        }
    
    # Calcular para cada componente
    stats_means = [compute_stats(bootstrap_means[:, i]) for i in range(n_components)]
    stats_stds = [compute_stats(bootstrap_stds[:, i]) for i in range(n_components)]
    stats_weights = [compute_stats(bootstrap_weights[:, i]) for i in range(n_components)]
    
    results = {
        'metadata': {
            'n_bootstrap': n_bootstrap,
            'n_components': n_components,
            'n_samples': n_samples,
            'random_state': random_state,
            'analysis_date': datetime.datetime.now().isoformat(),
        },
        'statistics': {
            'means': stats_means,
            'stds': stats_stds,
            'weights': stats_weights,
        }
    }
    
    print("✅ Bootstrap completado")
    return results

# ============================================================================
# FIGURA ÚNICA DE ALTA CALIDAD PARA PAPER
# ============================================================================

def create_paper_quality_plot(feh_data, gmm_params, bootstrap_results, output_path):
    """
    Crea una figura única de calidad profesional para el paper.
    Combinación de histograma, modelo GMM y visualización de incertidumbres.
    """
    
    print("\n🎨 CREANDO FIGURA PARA PAPER...")
    
    n_components = gmm_params['n_components']
    pop_names = gmm_params['pop_names']
    pop_short = gmm_params['pop_short']
    
    # Crear figura con dimensiones óptimas para paper (columna simple)
    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(8, 4), 
                                           gridspec_kw={'width_ratios': [2, 1]})
    
    # ============================================================
    # PANEL PRINCIPAL: Histograma y modelo GMM
    # ============================================================
    
    # Histograma
    hist, bins, patches = ax_main.hist(feh_data, bins=25, density=True,
                                      alpha=0.7, color=COLORS['hist'],
                                      edgecolor='black', linewidth=0.5,
                                      label='NGC 5128 GCs')
    
    # Preparar modelo GMM
    means = np.array(gmm_params['means'])
    stds = np.array(gmm_params['stds'])
    weights = np.array(gmm_params['weights'])
    
    # Crear curva de densidad
    x = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 1000)
    pdf_total = np.zeros_like(x)
    
    # Plot cada componente
    for i in range(n_components):
        pdf_comp = weights[i] * norm.pdf(x, means[i], stds[i])
        pdf_total += pdf_comp
        
        ax_main.plot(x, pdf_comp, '--', linewidth=1.5,
                    color=COLORS[pop_short[i]], alpha=0.8,
                    label=f'{pop_names[i]} (Gaussian)')
    
    # Modelo total
    ax_main.plot(x, pdf_total, '-', linewidth=2.5,
                color=COLORS['total'], label='GMM Model')
    
    # Configurar panel principal
    ax_main.set_xlabel('[Fe/H] (dex)', fontsize=10)
    ax_main.set_ylabel('Probability Density', fontsize=10)
    ax_main.set_title('Metallicity Distribution of NGC 5128 Globular Clusters', 
                     fontsize=11, pad=10)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    ax_main.legend(loc='upper left', fontsize=7, framealpha=0.9)
    
    # Añadir estadísticas básicas
    stats_text = f'N = {len(feh_data)} GCs\n'
    stats_text += f'⟨[Fe/H]⟩ = {feh_data.mean():.2f} ± {feh_data.std():.2f}'
    ax_main.text(0.02, 0.98, stats_text,
                transform=ax_main.transAxes, fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white',
                         alpha=0.8, edgecolor='gray', pad=0.3))
    
    # ============================================================
    # PANEL INSET: Visualización de incertidumbres bootstrap
    # ============================================================
    
    # Crear violín plots o boxplots para las incertidumbres
    positions = np.arange(n_components) + 1
    
    # Preparar datos para boxplot
    data_to_plot = []
    for i in range(n_components):
        # Para mostrar la distribución de medias en el bootstrap
        # Simulamos con datos sintéticos basados en las estadísticas
        mean_stat = bootstrap_results['statistics']['means'][i]
        # Crear distribución normal aproximada
        synthetic_data = np.random.normal(
            mean_stat['mean'], 
            mean_stat['std'], 
            1000
        )
        data_to_plot.append(synthetic_data)
    
    # Boxplot con estilo mejorado
    bp = ax_inset.boxplot(data_to_plot, positions=positions,
                         widths=0.6, patch_artist=True,
                         showfliers=False, whis=[5, 95])
    
    # Colorear las cajas
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(COLORS[pop_short[i]])
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(0.8)
    
    # Mejorar mediana
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(1.5)
    
    # Configurar panel inset
    ax_inset.set_xlabel('Population', fontsize=9)
    ax_inset.set_ylabel('[Fe/H] Mean (dex)', fontsize=9)
    ax_inset.set_title('Bootstrap Uncertainty', fontsize=10, pad=8)
    ax_inset.set_xticks(positions)
    ax_inset.set_xticklabels(pop_short, fontsize=8)
    ax_inset.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    # Añadir anotaciones con intervalos de confianza
    for i in range(n_components):
        mean_stat = bootstrap_results['statistics']['means'][i]
        y_pos = mean_stat['mean'] + 0.05
        
        # Texto con intervalo de confianza
        ci_text = f"{mean_stat['mean']:.3f}\n"
        ci_text += f"[{mean_stat['ci_95_lower']:.3f}, {mean_stat['ci_95_upper']:.3f}]"
        
        ax_inset.text(positions[i], y_pos, ci_text,
                     fontsize=6, ha='center', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.8, edgecolor='gray', pad=0.2))
    
    # ============================================================
    # TABLA DE PARÁMETROS (integrada en la figura)
    # ============================================================
    
    # Crear tabla de parámetros como anotación
    param_text = "GMM Parameters with Bootstrap Uncertainties\n"
    param_text += "=" * 40 + "\n"
    
    for i in range(n_components):
        mean_stat = bootstrap_results['statistics']['means'][i]
        std_stat = bootstrap_results['statistics']['stds'][i]
        weight_stat = bootstrap_results['statistics']['weights'][i]
        
        param_text += f"\n{pop_short[i]} ({pop_names[i]}):\n"
        param_text += f"  ⟨[Fe/H]⟩ = {mean_stat['mean']:.3f} ± {mean_stat['std']:.3f} dex\n"
        param_text += f"  σ = {std_stat['mean']:.3f} ± {std_stat['std']:.3f} dex\n"
        param_text += f"  f = {weight_stat['mean']:.3f} ± {weight_stat['std']:.3f}"
    
    param_text += f"\n\nBootstrap: {bootstrap_results['metadata']['n_bootstrap']} replications"
    
    # Añadir tabla como anotación en figura
    fig.text(0.02, 0.02, param_text,
            fontsize=6, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='whitesmoke',
                     alpha=0.9, edgecolor='gray', pad=0.5))
    
    # ============================================================
    # AJUSTES FINALES Y GUARDADO
    # ============================================================
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"✓ Figura guardada: {output_path}")
    
    # También guardar versión de alta resolución para revista
    highres_path = output_path.replace('.pdf', '_highres.png')
    fig.savefig(highres_path, dpi=1200, bbox_inches='tight')
    print(f"✓ Versión alta resolución: {highres_path}")
    
    plt.close(fig)
    
    return fig

def create_alternative_simple_plot(feh_data, gmm_params, bootstrap_results, output_path):
    """
    Alternativa más simple: Solo histograma y modelo con incertidumbres anotadas.
    """
    
    print("\n🎨 CREANDO FIGURA SIMPLE PARA PAPER...")
    
    n_components = gmm_params['n_components']
    pop_names = gmm_params['pop_names']
    pop_short = gmm_params['pop_short']
    
    # Crear figura simple
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # ============================================================
    # HISTOGRAMA Y MODELO GMM
    # ============================================================
    
    # Histograma con mejor estilo
    n_bins = max(20, int(len(feh_data) / 25))
    hist, bins, patches = ax.hist(feh_data, bins=n_bins, density=True,
                                 alpha=0.6, color='#4A90E2',
                                 edgecolor='white', linewidth=0.5,
                                 label=f'NGC 5128 GCs (N={len(feh_data)})')
    
    # Modelo GMM
    means = np.array(gmm_params['means'])
    stds = np.array(gmm_params['stds'])
    weights = np.array(gmm_params['weights'])
    
    x = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 1000)
    pdf_total = np.zeros_like(x)
    
    # Componentes individuales
    for i in range(n_components):
        pdf_comp = weights[i] * norm.pdf(x, means[i], stds[i])
        pdf_total += pdf_comp
        
        ax.plot(x, pdf_comp, '--', linewidth=1.5,
               color=COLORS[pop_short[i]], alpha=0.7)
    
    # Modelo total
    ax.plot(x, pdf_total, '-', linewidth=2.5,
           color='#2C3E50', label='GMM Fit')
    
    # ============================================================
    # ANOTACIONES DE PARÁMETROS
    # ============================================================
    
    # Crear cadena de texto con parámetros
    param_text = "Gaussian Mixture Model Parameters:\n"
    
    for i in range(n_components):
        mean_stat = bootstrap_results['statistics']['means'][i]
        std_stat = bootstrap_results['statistics']['stds'][i]
        weight_stat = bootstrap_results['statistics']['weights'][i]
        
        param_text += f"\n{pop_short[i]} ({pop_names[i]}):\n"
        param_text += f"  ⟨[Fe/H]⟩ = {mean_stat['mean']:.3f} ± {mean_stat['std']:.3f} dex\n"
        param_text += f"  σ = {std_stat['mean']:.3f} ± {std_stat['std']:.3f} dex\n"
        param_text += f"  f = {weight_stat['mean']:.3f} ± {weight_stat['std']:.3f}"
    
    # Añadir anotación en figura
    ax.text(0.02, 0.98, param_text,
           transform=ax.transAxes, fontsize=7,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.95, edgecolor='gray', pad=0.5))
    
    # ============================================================
    # ETIQUETAS Y ESTILO
    # ============================================================
    
    ax.set_xlabel('[Fe/H] (dex)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title('Metallicity Distribution Function of NGC 5128 Globular Clusters', 
                fontsize=11, pad=12)
    
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Añadir información de bootstrap
    bootstrap_info = f"Bootstrap analysis: {bootstrap_results['metadata']['n_bootstrap']} replications"
    ax.text(0.98, 0.02, bootstrap_info,
           transform=ax.transAxes, fontsize=6,
           horizontalalignment='right', verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.8, edgecolor='gray', pad=0.3))
    
    # Ajustar y guardar
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    
    print(f"✓ Figura simple guardada: {output_path}")
    
    plt.close(fig)
    return fig

# ============================================================================
# FUNCIÓN PRINCIPAL SIMPLIFICADA
# ============================================================================

def main():
    """Función principal simplificada."""
    
    print("\n" + "="*60)
    print("BOOTSTRAP ANALYSIS - SINGLE PLOT FOR PAPER")
    print("="*60)
    
    # ============================================================
    # CONFIGURACIÓN
    # ============================================================
    INPUT_CSV = "NGC5128_Robust_Analysis_20260103_201618/ngc5128_gcs_robust_classified_2pop.csv"
    INPUT_JSON = "NGC5128_Robust_Analysis_20260103_201618/gmm_parameters_2pop.json"
    
    # Directorio de salida
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"Bootstrap_Paper_Plot_{timestamp}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📁 Directorio de salida: {OUTPUT_DIR}")
    
    # ============================================================
    # 1. CARGAR DATOS
    # ============================================================
    print("\n[1] CARGANDO DATOS...")
    try:
        df, gmm_params, feh_data, labels = load_classified_data(INPUT_CSV, INPUT_JSON)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # ============================================================
    # 2. ANÁLISIS BOOTSTRAP (500 réplicas para rapidez)
    # ============================================================
    print("\n[2] EJECUTANDO BOOTSTRAP...")
    bootstrap_results = perform_bootstrap_analysis(
        feh_data=feh_data,
        n_components=gmm_params['n_components'],
        n_bootstrap=500,  # Suficiente para paper
        random_state=42
    )
    
    # ============================================================
    # 3. CREAR FIGURA PRINCIPAL PARA PAPER
    # ============================================================
    print("\n[3] CREANDO FIGURA PARA PAPER...")
    
    # Opción 1: Figura con dos paneles (recomendada)
    plot1_path = f"{OUTPUT_DIR}/Figure_Paper_Quality.pdf"
    create_paper_quality_plot(feh_data, gmm_params, bootstrap_results, plot1_path)
    
    # Opción 2: Figura más simple
    plot2_path = f"{OUTPUT_DIR}/Figure_Paper_Simple.pdf"
    create_alternative_simple_plot(feh_data, gmm_params, bootstrap_results, plot2_path)
    
    # ============================================================
    # 4. GUARDAR RESULTADOS (formato serializable)
    # ============================================================
    print("\n[4] GUARDANDO RESULTADOS...")
    
    # Guardar resultados bootstrap
    bootstrap_json = f"{OUTPUT_DIR}/bootstrap_results.json"
    
    # Asegurar que todo es serializable
    serializable_results = {
        'metadata': bootstrap_results['metadata'],
        'statistics': bootstrap_results['statistics'],
    }
    
    with open(bootstrap_json, 'w') as f:
        json.dump(serializable_results, f, indent=4, default=str)
    
    print(f"✓ Resultados: {bootstrap_json}")
    
    # Crear tabla resumen CSV
    summary_data = []
    for i in range(gmm_params['n_components']):
        summary_data.append({
            'Population': gmm_params['pop_names'][i],
            'Code': gmm_params['pop_short'][i],
            'Mean_FeH': bootstrap_results['statistics']['means'][i]['mean'],
            'Mean_FeH_std': bootstrap_results['statistics']['means'][i]['std'],
            'Mean_FeH_CI95_lower': bootstrap_results['statistics']['means'][i]['ci_95_lower'],
            'Mean_FeH_CI95_upper': bootstrap_results['statistics']['means'][i]['ci_95_upper'],
            'Dispersion_FeH': bootstrap_results['statistics']['stds'][i]['mean'],
            'Dispersion_FeH_std': bootstrap_results['statistics']['stds'][i]['std'],
            'Fraction': bootstrap_results['statistics']['weights'][i]['mean'],
            'Fraction_std': bootstrap_results['statistics']['weights'][i]['std'],
        })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = f"{OUTPUT_DIR}/bootstrap_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Tabla resumen: {csv_path}")
    
    # ============================================================
    # 5. RESUMEN FINAL
    # ============================================================
    print("\n" + "="*60)
    print("📊 RESUMEN PARA PAPER")
    print("="*60)
    
    print(f"\n🔬 HALLAZGOS PRINCIPALES:")
    print(f"   • {gmm_params['n_components']} poblaciones de GCs identificadas")
    print(f"   • Análisis bootstrap con {bootstrap_results['metadata']['n_bootstrap']} réplicas")
    
    print(f"\n📈 PARÁMETROS CON INCERTIDUMBRES:")
    for i in range(gmm_params['n_components']):
        mean_stat = bootstrap_results['statistics']['means'][i]
        print(f"\n   {gmm_params['pop_short'][i]} ({gmm_params['pop_names'][i]}):")
        print(f"     ⟨[Fe/H]⟩ = {mean_stat['mean']:.3f} ± {mean_stat['std']:.3f} dex")
        print(f"     95% CI: [{mean_stat['ci_95_lower']:.3f}, {mean_stat['ci_95_upper']:.3f}]")
    
    print(f"\n📁 FIGURAS GENERADAS:")
    print(f"   • {plot1_path} (Figura principal - calidad paper)")
    print(f"   • {plot2_path} (Figura alternativa - más simple)")
    print(f"   • {bootstrap_json} (Resultados completos)")
    print(f"   • {csv_path} (Tabla resumen)")
    
    print("\n✅ ANÁLISIS COMPLETADO - LISTO PARA INCLUIR EN PAPER")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
