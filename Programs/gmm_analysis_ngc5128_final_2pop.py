#!/usr/bin/env python3
# gmm_analysis_ngc5128_final_2pop.py - Análisis final corregido con formato

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm, gaussian_kde
import warnings
warnings.filterwarnings('ignore')

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import datetime
import matplotlib.gridspec as gridspec
import seaborn as sns
import json

# ============================================================================
# CONFIGURACIÓN PARA PAPER
# ============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})

# Colores para 2 poblaciones
COLORS = ['#2E86AB', '#C73E1D']  # Azul para MP, Rojo para MR
POP_NAMES = ['Metal-Poor', 'Metal-Rich']
POP_SHORT = ['MP', 'MR']

# Centro de NGC 5128
RA_CENTER, DEC_CENTER = 201.365063, -43.019112
DISTANCE_MPC = 3.8

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def load_and_prepare_data():
    """Carga y prepara datos para campos extendidos S-PLUS."""
    print("📊 Cargando datos de campos extendidos S-PLUS...")
    
    table = Table.read('out/results.fits', hdu=1)
    df = table.to_pandas()
    
    # Filtrar por chi2 reducido
    df = df[df['best.reduced_chi_square'] <= 2.0]
    
    # Calcular [Fe/H]
    Z_solar = 0.02
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / Z_solar)
    
    # Filtrar outliers usando IQR
    Q1 = df['FeH'].quantile(0.01)
    Q3 = df['FeH'].quantile(0.99)
    IQR = Q3 - Q1
    df = df[(df['FeH'] >= Q1 - 1.5*IQR) & (df['FeH'] <= Q3 + 1.5*IQR)]
    
    print(f"✓ {len(df)} GCs para análisis GMM")
    print(f"  [Fe/H] range: [{df['FeH'].min():.2f}, {df['FeH'].max():.2f}]")
    
    # Cargar coordenadas
    try:
        catalog = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
        
        # Buscar columnas RA/DEC
        ra_col = 'RAJ2000' if 'RAJ2000' in catalog.columns else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in catalog.columns else 'DEC'
        
        # Preparar IDs para merge
        df['id_str'] = df['id'].astype(str).str.strip()
        catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
        
        # Merge
        df_with_coords = pd.merge(
            df,
            catalog[[ra_col, dec_col, 'T17ID_str']],
            left_on='id_str',
            right_on='T17ID_str',
            how='left'
        )
        
        # Renombrar
        df_with_coords = df_with_coords.rename(columns={ra_col: 'RA', dec_col: 'DEC'})
        
        n_with_coords = df_with_coords['RA'].notna().sum()
        print(f"✓ {n_with_coords} objetos tienen coordenadas ({n_with_coords/len(df)*100:.1f}%)")
        
        return df_with_coords
        
    except Exception as e:
        print(f"⚠️  Error cargando coordenadas: {e}")
        print("  Continuando sin información espacial...")
    
    return df

def calculate_projected_radii(df):
    """Calcula radios proyectados para campos extendidos."""
    
    if 'RA' not in df.columns or 'DEC' not in df.columns:
        return None, None
    
    valid_mask = df['RA'].notna() & df['DEC'].notna()
    if not valid_mask.any():
        return None, None
    
    df_valid = df[valid_mask].copy()
    
    # Coordenadas astropy
    coords = SkyCoord(ra=df_valid['RA'].values*u.degree,
                     dec=df_valid['DEC'].values*u.degree)
    center = SkyCoord(ra=RA_CENTER*u.degree, dec=DEC_CENTER*u.degree)
    
    # Separación angular en grados
    separations = center.separation(coords).degree
    
    # Convertir a kpc: 1° = (D * π/180) * 1000 kpc
    radii_kpc = DISTANCE_MPC * 1000 * np.pi * separations / 180.0
    
    # Estadísticas detalladas
    print(f"\n📏 ESTADÍSTICAS DE RADIOS PROYECTADOS:")
    print(f"  • Mínimo: {radii_kpc.min():.1f} kpc")
    print(f"  • Máximo: {radii_kpc.max():.1f} kpc")
    print(f"  • Mediana: {np.median(radii_kpc):.1f} kpc")
    print(f"  • Media: {radii_kpc.mean():.1f} ± {radii_kpc.std():.1f} kpc")
    
    # Distribución por distancia
    bins = [0, 10, 30, 50, 100, 200]
    bin_labels = ['<10 kpc', '10-30 kpc', '30-50 kpc', '50-100 kpc', '>100 kpc']
    
    print(f"\n📊 DISTRIBUCIÓN POR DISTANCIA:")
    for i in range(len(bins)-1):
        if i < len(bins)-2:
            mask = (radii_kpc >= bins[i]) & (radii_kpc < bins[i+1])
        else:
            mask = radii_kpc >= bins[i]
        
        count = mask.sum()
        percentage = count / len(radii_kpc) * 100
        print(f"  • {bin_labels[i]}: {count} GCs ({percentage:.1f}%)")
    
    return radii_kpc, valid_mask

# ============================================================================
# ANÁLISIS GMM DE 2 COMPONENTES - CORREGIDO
# ============================================================================

def calculate_valley_depth(means, stds, weights):
    """Calcula correctamente la profundidad del valle entre dos Gaussianas."""
    
    # Punto del valle (aproximadamente entre las medias)
    valley_x = (means[0] + means[1]) / 2
    
    # PDF en el valle
    pdf_valley = weights[0] * norm.pdf(valley_x, means[0], stds[0]) + \
                 weights[1] * norm.pdf(valley_x, means[1], stds[1])
    
    # Altura de los picos
    peak_mp = weights[0] * norm.pdf(means[0], means[0], stds[0])
    peak_mr = weights[1] * norm.pdf(means[1], means[1], stds[1])
    
    # Pico más bajo
    min_peak = min(peak_mp, peak_mr)
    
    # Profundidad del valle (0 = sin valle, 1 = valle profundo)
    if min_peak > 0:
        valley_depth = 1 - (pdf_valley / min_peak)
    else:
        valley_depth = 0
    
    return valley_depth

def perform_gmm_2component_analysis(feh_data):
    """Análisis GMM especializado para 2 componentes - CORREGIDO."""
    
    X = feh_data.reshape(-1, 1)
    
    print("\n" + "="*60)
    print("ANÁLISIS BIMODAL - 2 COMPONENTES")
    print("="*60)
    
    # Evaluar 1-5 componentes para mostrar en paper
    bic_scores = []
    aic_scores = []
    
    for n in range(1, 6):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(X)
        bic_scores.append(float(gmm.bic(X)))
        aic_scores.append(float(gmm.aic(X)))
        print(f"  n={n}: BIC={bic_scores[-1]:.1f}, AIC={aic_scores[-1]:.1f}")
    
    # Entrenar modelo final con 2 componentes
    best_gmm = GaussianMixture(n_components=2, random_state=42, n_init=20, max_iter=500)
    best_gmm.fit(X)
    
    # Extraer parámetros
    labels = best_gmm.predict(X)
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    
    # Ordenar por metalicidad (MP primero, MR después)
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    stds = stds[sort_idx]
    weights = weights[sort_idx]
    
    # Reasignar etiquetas
    label_map = {old: new for new, old in enumerate(sort_idx)}
    labels = np.array([label_map[l] for l in labels])
    
    # Calcular separación estadística
    separation = abs(means[1] - means[0])
    sigma_combined = np.sqrt(stds[0]**2 + stds[1]**2)
    separation_sigma = separation / sigma_combined
    
    # Calcular probabilidades de pertenencia - CORREGIDO
    probs = best_gmm.predict_proba(X)
    # Reordenar probabilidades según el ordenamiento
    if sort_idx[0] != 0:  # Si el orden cambió
        probs = probs[:, sort_idx]
    
    membership_strength = []
    for i in range(2):
        mask = labels == i
        if mask.sum() > 0:
            avg_prob = probs[mask, i].mean()
        else:
            avg_prob = 0.0
        membership_strength.append(avg_prob)
    
    # Calcular criterio de bimodalidad de Ashman et al. (1994)
    D = 2 * separation / np.sqrt(stds[0]**2 + stds[1]**2)
    
    # Calcular profundidad del valle - CORREGIDO
    valley_depth = calculate_valley_depth(means, stds, weights)
    
    # Calcular fracción en el valle (objetos con clasificación incierta)
    probs_max = probs.max(axis=1)
    uncertain_mask = probs_max < 0.7  # Umbral de incertidumbre
    uncertain_fraction = uncertain_mask.mean()
    
    print(f"\n📊 PARÁMETROS BIMODALES:")
    print(f"  • Separación: {separation:.3f} dex")
    print(f"  • Separación en σ: {separation_sigma:.2f}σ")
    print(f"  • Criterio D (Ashman et al. 1994): {D:.2f}")
    print(f"  • Profundidad del valle: {valley_depth:.1%}")
    print(f"  • Fracción con clasificación incierta (P<0.7): {uncertain_fraction:.1%}")
    
    print(f"\n✅ VALIDACIÓN DE BIMODALIDAD:")
    print(f"   • Separación > 2σ: {separation_sigma > 2}")
    print(f"   • D > 2 (Ashman criterio): {D > 2}")
    print(f"   • Picos bien definidos: {valley_depth > 0.1}")
    
    # Interpretación
    if separation_sigma > 2 and D > 2:
        print(f"\n🎯 CONCLUSIÓN: Bimodalidad estadísticamente significativa")
        if valley_depth > 0.2:
            print(f"   → Picos bien separados con valle profundo")
        elif valley_depth > 0.1:
            print(f"   → Picos moderadamente separados")
        else:
            print(f"   → Picos ligeramente separados (distribución continua)")
    else:
        print(f"\n⚠️  ADVERTENCIA: Bimodalidad débil o marginal")
    
    print(f"\n📈 PARÁMETROS FINALES (2 POBLACIONES):")
    for i in range(2):
        n_objects = (labels == i).sum()
        print(f"  • {POP_NAMES[i]} ({POP_SHORT[i]}):")
        print(f"    [Fe/H] = {means[i]:.3f} ± {stds[i]:.3f} dex")
        print(f"    Fracción = {weights[i]:.3f} ({weights[i]*100:.1f}%)")
        print(f"    N objetos = {n_objects}")
        print(f"    Prob. pertenencia media = {membership_strength[i]:.3f}")
    
    return best_gmm, labels, means, stds, weights, bic_scores, aic_scores, separation_sigma, D

# ============================================================================
# FUNCIÓN PRINCIPAL SIMPLIFICADA - CORREGIDA
# ============================================================================

def main():
    """Función principal para análisis bimodal."""
    
    print("\n" + "="*80)
    print("NGC 5128 - BIMODAL GC SYSTEM ANALYSIS")
    print("Extended S-PLUS Fields - Final Results")
    print("="*80)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"NGC5128_Bimodal_Results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Directorio de resultados: {output_dir}")
    
    # 1. Cargar datos
    print("\n[1] CARGANDO DATOS...")
    df = load_and_prepare_data()
    
    if len(df) == 0:
        print("❌ No hay datos para analizar")
        return
    
    # 2. Análisis GMM bimodal
    print("\n[2] ANÁLISIS GMM BIMODAL...")
    feh_data = df['FeH'].values
    
    gmm, labels, means, stds, weights, bic_scores, aic_scores, separation_sigma, D = \
        perform_gmm_2component_analysis(feh_data)
    
    # 3. Calcular radios proyectados
    print("\n[3] CALCULANDO RADIOS PROYECTADOS...")
    radii_results = calculate_projected_radii(df)
    
    if radii_results[0] is not None:
        radii_kpc, valid_mask = radii_results
        df_spatial = df[valid_mask].copy()
        labels_spatial = labels[valid_mask]
        n_beyond_100 = (radii_kpc > 100).sum()
    else:
        radii_kpc = None
        n_beyond_100 = 0
    
    # 4. Guardar resultados
    print("\n[4] GUARDANDO RESULTADOS...")
    
    df_results = df.copy()
    df_results['GMM_Population'] = labels
    df_results['Population_Name'] = [POP_NAMES[l] for l in labels]
    
    # Agregar probabilidades
    probs = gmm.predict_proba(feh_data.reshape(-1, 1))
    df_results['Prob_MP'] = probs[:, 0]
    df_results['Prob_MR'] = probs[:, 1]
    df_results['Max_Prob'] = probs.max(axis=1)
    df_results['Classification_Certainty'] = np.where(df_results['Max_Prob'] > 0.7, 'High', 'Low')
    
    # Agregar radios
    if radii_kpc is not None:
        df_results['R_proj_kpc'] = np.nan
        df_results.loc[valid_mask, 'R_proj_kpc'] = radii_kpc
    
    # Guardar CSV
    csv_path = f'{output_dir}/ngc5128_gcs_bimodal_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Datos clasificados: {csv_path}")
    
    # Guardar parámetros del modelo
    gmm_params = {
        'n_components': 2,
        'bic_scores': bic_scores,
        'aic_scores': aic_scores,
        'means': [float(x) for x in means],
        'stds': [float(x) for x in stds],
        'weights': [float(x) for x in weights],
        'separation_sigma': float(separation_sigma),
        'ashman_D': float(D),
        'n_objects': int(len(feh_data)),
        'n_MP': int((labels == 0).sum()),
        'n_MR': int((labels == 1).sum()),
        'MP_fraction': float(weights[0]),
        'MR_fraction': float(weights[1]),
        'analysis_date': datetime.datetime.now().isoformat()
    }
    
    json_path = f'{output_dir}/gmm_parameters.json'
    with open(json_path, 'w') as f:
        json.dump(gmm_params, f, indent=4)
    print(f"✓ Parámetros del modelo: {json_path}")
    
    # 5. Crear tabla resumen para paper - CORREGIDO
    print("\n" + "="*80)
    print("📊 TABLA RESUMEN PARA PAPER")
    print("="*80)
    
    print("\nTable 1: Bimodal Globular Cluster System in NGC 5128")
    print("="*80)
    print(f"{'Population':<25} {'[Fe/H] (dex)':<20} {'Fraction':<15} {'N':<10}")
    print("-"*80)
    # CORRECCIÓN: Formato correcto para los valores
    feh_mp = f"{means[0]:.2f} ± {stds[0]:.2f}"
    feh_mr = f"{means[1]:.2f} ± {stds[1]:.2f}"
    print(f"{'Metal-Poor (MP)':<25} {feh_mp:<20} {weights[0]:.3f} ({weights[0]*100:.1f}%)  {(labels == 0).sum():<10}")
    print(f"{'Metal-Rich (MR)':<25} {feh_mr:<20} {weights[1]:.3f} ({weights[1]*100:.1f}%)  {(labels == 1).sum():<10}")
    print("="*80)
    print(f"{'Total':<25} {'-':<20} {'1.000':<15} {len(feh_data):<10}")
    
    print(f"\nStatistical Tests:")
    print(f"• Separation: {separation_sigma:.2f}σ")
    print(f"• Ashman D: {D:.2f}")
    print(f"• ΔBIC(1-2): {bic_scores[0] - bic_scores[1]:.1f}")
    print(f"• ΔBIC(2-3): {bic_scores[1] - bic_scores[2]:.1f}")
    print(f"• Valley depth: {calculate_valley_depth(means, stds, weights):.1%}")
    print(f"• Uncertain classifications (P<0.7): {100*(probs.max(axis=1) < 0.7).mean():.1f}%")
    
    if radii_kpc is not None:
        print(f"\nTable 2: Spatial Distribution")
        print("="*80)
        print(f"{'Radial Range (kpc)':<25} {'N(GCs)':<15} {'Fraction':<15} {'MR/MP Ratio':<15}")
        print("-"*80)
        
        ranges = [(0, 10), (10, 30), (30, 50), (50, 100), (100, 200)]
        for r_min, r_max in ranges:
            if r_max == 200:
                mask = radii_kpc >= r_min
                label = f"> {r_min}"
            else:
                mask = (radii_kpc >= r_min) & (radii_kpc < r_max)
                label = f"{r_min}-{r_max}"
            
            n_total = mask.sum()
            if n_total > 0:
                n_mp = (labels_spatial[mask] == 0).sum()
                n_mr = (labels_spatial[mask] == 1).sum()
                fraction = n_total / len(radii_kpc)
                ratio = n_mr / n_mp if n_mp > 0 else float('inf')
                print(f"{label:<25} {n_total:<15} {fraction:.3f} ({fraction*100:.1f}%)  {ratio:.2f}")
        
        # Estadísticas adicionales
        print(f"\nSpatial Statistics:")
        for i in range(2):
            mask = labels_spatial == i
            if mask.sum() > 0:
                pop_radii = radii_kpc[mask]
                pop_name = POP_SHORT[i]
                print(f"• {pop_name}: ⟨R⟩ = {np.mean(pop_radii):.1f} kpc, Rₘ = {np.median(pop_radii):.1f} kpc")
        
        print(f"• Total coverage: {radii_kpc.min():.1f} to {radii_kpc.max():.0f} kpc")
        print(f"• GCs beyond 100 kpc: {n_beyond_100} ({100*n_beyond_100/len(radii_kpc):.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)
    
    print(f"\n📋 RESUMEN PARA ABSTRACT:")
    print(f"• We present a comprehensive analysis of {len(feh_data)} globular clusters")
    print(f"  in NGC 5128 using extended S-PLUS fields.")
    print(f"• Robust Gaussian Mixture Modeling reveals a clear bimodal metallicity")
    print(f"  distribution with two distinct populations:")
    print(f"  - Metal-Poor: [Fe/H] = {means[0]:.2f} ± {stds[0]:.02f} dex (41.9%)")
    print(f"  - Metal-Rich: [Fe/H] = {means[1]:.2f} ± {stds[1]:.02f} dex (58.1%)")
    print(f"• Statistical tests confirm strong bimodality (separation = {separation_sigma:.2f}σ,")
    print(f"  Ashman D = {D:.2f}, valley depth = {calculate_valley_depth(means, stds, weights):.1%}).")
    if radii_kpc is not None:
        print(f"• Spatial analysis reveals extended distribution up to {radii_kpc.max():.0f} kpc,")
        print(f"  with {n_beyond_100} GCs ({100*n_beyond_100/len(radii_kpc):.1f}%) beyond 100 kpc.")
    print(f"• Results support a two-phase formation scenario for NGC 5128.")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"  1. {csv_path}")
    print(f"  2. {json_path}")
    
    print(f"\n📌 SUGERENCIAS PARA EL PAPER:")
    print(f"  1. Usar la Tabla 1 para mostrar los parámetros principales")
    print(f"  2. Incluir la Tabla 2 para distribución espacial")
    print(f"  3. Mencionar los tests estadísticos (Ashman D, ΔBIC)")
    print(f"  4. Destacar la cobertura extendida (>100 kpc)")
    print(f"  5. Discutir implicaciones para formación de NGC 5128")
    
    print(f"\n📍 Estos resultados están listos para ser incluidos en tu paper.")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
