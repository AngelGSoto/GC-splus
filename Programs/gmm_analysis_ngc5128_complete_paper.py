#!/usr/bin/env python3
# gmm_bootstrap_final_unified.py - Versión Corregida

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm, gaussian_kde
import warnings
warnings.filterwarnings('ignore')

import os
import datetime
import matplotlib.gridspec as gridspec
import json
from matplotlib.cm import get_cmap

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
})

# Colores base para poblaciones
BASE_COLORS = ['#2166AC', '#B2182B', '#F4A582', '#92C5DE', '#762A83', '#9970AB']
POP_NAMES = ['Metal-Poor', 'Metal-Rich', 'Intermediate-Low', 'Intermediate-High']
POP_SHORT = ['MP', 'MR', 'IL', 'IH']

# Centro de NGC 5128
RA_CENTER, DEC_CENTER = 201.365063, -43.019112
DISTANCE_MPC = 3.8

# ============================================================================
# 1. FUNCIONES DE CARGA DE DATOS
# ============================================================================

def load_and_prepare_data():
    """Carga y prepara datos de campos extendidos S-PLUS."""
    print("📊 Cargando datos de campos extendidos S-PLUS...")
    
    from astropy.table import Table
    
    # Intentar múltiples rutas
    fits_paths = ['out/results.fits', '../out/results.fits', 
                  '../../out/results.fits', './results.fits']
    
    fits_path = None
    for path in fits_paths:
        if os.path.exists(path):
            fits_path = path
            break
    
    if fits_path is None:
        raise FileNotFoundError("No se encontró el archivo results.fits")
    
    print(f"📂 Leyendo archivo FITS: {fits_path}")
    table = Table.read(fits_path, hdu=1)
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
    
    # Cargar coordenadas si están disponibles
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
        
        df = df_with_coords
        
    except Exception as e:
        print(f"⚠️  Error cargando coordenadas: {e}")
        df['RA'] = np.nan
        df['DEC'] = np.nan
    
    print(f"✓ {len(df)} GCs para análisis GMM")
    print(f"  [Fe/H] range: [{df['FeH'].min():.2f}, {df['FeH'].max():.2f}]")
    
    return df

def calculate_projected_radii(df):
    """Calcula radios proyectados para campos extendidos."""
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
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
    
    return radii_kpc, valid_mask

# ============================================================================
# 2. FUNCIONES ESTADÍSTICAS MEJORADAS
# ============================================================================

def calculate_bayes_factor(bic_simpler, bic_complex):
    """Calcula el factor de Bayes aproximado a partir de ΔBIC.
    Bayes Factor = exp(-ΔBIC/2) donde ΔBIC = BIC_complex - BIC_simpler
    BF > 1 favorece al modelo más simple, BF < 1 favorece al modelo más complejo"""
    delta_bic = bic_complex - bic_simpler
    bayes_factor = np.exp(-delta_bic / 2)
    return bayes_factor

def interpret_bayes_factor(bf):
    """Interpreta el factor de Bayes según escala de Kass & Raftery (1995)."""
    if bf < 1:
        bf_inv = 1/bf
        if bf_inv > 150:
            return "Evidencia decisiva contra el modelo más simple"
        elif bf_inv > 20:
            return "Evidencia fuerte contra el modelo más simple"
        elif bf_inv > 3:
            return "Evidencia sustancial contra el modelo más simple"
        else:
            return "No hay evidencia significativa"
    else:
        if bf > 150:
            return "Evidencia decisiva a favor del modelo más simple"
        elif bf > 20:
            return "Evidencia fuerte a favor del modelo más simple"
        elif bf > 3:
            return "Evidencia sustancial a favor del modelo más simple"
        else:
            return "No hay evidencia significativa"

# ============================================================================
# 3. ANÁLISIS GMM CON VALIDACIÓN ROBUSTA - CORREGIDO
# ============================================================================

def validate_gmm_model_with_quality(X, n_components, random_state=42):
    """Valida un modelo GMM con criterios de calidad estrictos - CORREGIDO."""
    
    best_gmm = None
    best_bic = np.inf  # Ahora buscamos el menor BIC
    best_quality = None
    best_separations = None
    best_means = None
    best_stds = None
    best_weights = None
    
    # Probar múltiples inicializaciones
    for seed in [random_state, random_state+100, random_state+200, 
                 random_state+300, random_state+400]:
        gmm = GaussianMixture(n_components=n_components, random_state=seed, 
                             n_init=10, max_iter=500)
        gmm.fit(X)
        
        # Predecir etiquetas y probabilidades
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        
        # Calcular calidad de pertenencia para cada componente
        membership_quality = []
        for i in range(n_components):
            mask = labels == i
            if mask.sum() > 0:
                avg_prob = probs[mask, i].mean()
            else:
                avg_prob = 0.0
            membership_quality.append(avg_prob)
        
        # Calcular separaciones entre componentes
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()
        
        # Ordenar por media
        sort_idx = np.argsort(means)
        means = means[sort_idx]
        stds = stds[sort_idx]
        weights = weights[sort_idx]
        membership_quality = [membership_quality[i] for i in sort_idx]
        
        separations = []
        if n_components > 1:
            for i in range(n_components - 1):
                sep = abs(means[i+1] - means[i])
                sigma = np.sqrt(stds[i]**2 + stds[i+1]**2)
                separations.append(sep / sigma)
        
        # Verificar criterios de calidad
        min_quality = min(membership_quality) if membership_quality else 0
        all_separations_good = all(s > 2.0 for s in separations) if separations else True
        
        # Solo considerar modelos que cumplen criterios de calidad
        if min_quality > 0.7 and all_separations_good:
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_quality = membership_quality
                best_separations = separations
                best_means = means
                best_stds = stds
                best_weights = weights
    
    return best_gmm, best_quality, best_separations, best_means, best_stds, best_weights

def perform_robust_gmm_analysis(feh_data):
    """Análisis GMM robusto con validación de calidad - VERSIÓN CORREGIDA."""
    
    X = feh_data.reshape(-1, 1)
    
    print("\n" + "="*60)
    print("ANÁLISIS GAUSSIAN MIXTURE MODEL - CRITERIOS DE CALIDAD")
    print("="*60)
    
    # Evaluar 2-5 componentes con criterios de calidad
    results = {}
    valid_models = {}
    
    for n in range(2, 6):
        print(f"\n  Analizando n={n} componentes...")
        
        gmm, quality, separations, means, stds, weights = validate_gmm_model_with_quality(X, n)
        
        if gmm is not None:
            # Modelo válido encontrado
            bic = float(gmm.bic(X))
            aic = float(gmm.aic(X))
            
            results[n] = {
                'bic': bic,
                'aic': aic,
                'gmm': gmm,
                'means': means,
                'stds': stds,
                'weights': weights,
                'quality': quality,
                'separations': separations,
                'min_quality': min(quality) if quality else 0,
                'all_greater_2sigma': all(s > 2.0 for s in separations) if separations else True,
                'avg_quality': np.mean(quality),
                'valid': True
            }
            valid_models[n] = results[n]
            
            print(f"    • VÁLIDO: BIC={bic:.1f}, AIC={aic:.1f}")
            print(f"    • Calidad mínima: {min(quality):.3f} ✓")
            print(f"    • Separaciones: {[f'{s:.2f}σ' for s in separations]}")
            print(f"    • Todas >2σ: ✓")
        else:
            results[n] = {'valid': False}
            print(f"    • NO VÁLIDO: No cumple criterios de calidad")
    
    # ============================================================
    # SELECCIÓN DEL MODELO ÓPTIMO - VERSIÓN CORREGIDA
    # ============================================================
    print("\n" + "="*60)
    print("SELECCIÓN DEL MODELO ÓPTIMO - LÓGICA CORREGIDA")
    print("="*60)
    
    if not valid_models:
        print("⚠️  Ningún modelo cumple todos los criterios de calidad.")
        print("   Usando modelo con mejor BIC entre todos...")
        # Filtrar solo modelos con datos completos
        available_models = {n: r for n, r in results.items() if 'bic' in r}
        if available_models:
            best_n = min(available_models.keys(), key=lambda x: available_models[x]['bic'])
            best_reason = "Mejor BIC (ningún modelo cumple criterios estrictos)"
            best_result = results[best_n]
        else:
            raise ValueError("No hay modelos disponibles para selección")
    else:
        # Mostrar modelos válidos
        for n in valid_models:
            print(f"  n={n}: VÁLIDO (BIC={valid_models[n]['bic']:.1f}, calidad={valid_models[n]['min_quality']:.3f})")
        
        # SI HAY MÚLTIPLES MODELOS VÁLIDOS, USAR EL DE MENOR BIC
        if len(valid_models) == 1:
            best_n = list(valid_models.keys())[0]
            best_reason = "Único modelo que cumple criterios de calidad"
            best_result = valid_models[best_n]
        else:
            # CORRECCIÓN: Seleccionar el modelo con menor BIC
            best_n = min(valid_models.keys(), key=lambda x: valid_models[x]['bic'])
            best_result = valid_models[best_n]
            
            # Analizar factores de Bayes correctamente
            print(f"\n📊 ANÁLISIS COMPARATIVO DE MODELOS VÁLIDOS:")
            print(f"  Modelo seleccionado: n={best_n} (BIC={best_result['bic']:.1f})")
            
            for n in valid_models:
                if n != best_n:
                    bf = calculate_bayes_factor(best_result['bic'], valid_models[n]['bic'])
                    interpretation = interpret_bayes_factor(bf)
                    
                    print(f"\n  • Comparación {best_n} vs {n}:")
                    print(f"    ΔBIC = {valid_models[n]['bic'] - best_result['bic']:+.1f}")
                    print(f"    Bayes Factor = {bf:.2e}")
                    print(f"    Interpretación: {interpretation}")
            
            best_reason = f"Menor BIC entre modelos válidos (BIC={best_result['bic']:.1f})"
    
    print(f"\n✅ MODELO SELECCIONADO: n = {best_n}")
    print(f"   Razón: {best_reason}")
    
    # Asignar nombres a poblaciones
    if best_n == 2:
        pop_names = ['Metal-Poor', 'Metal-Rich']
        pop_short = ['MP', 'MR']
        colors = BASE_COLORS[:2]
    elif best_n == 3:
        pop_names = ['Metal-Poor', 'Intermediate', 'Metal-Rich']
        pop_short = ['MP', 'I', 'MR']
        colors = [BASE_COLORS[0], BASE_COLORS[2], BASE_COLORS[1]]
    elif best_n == 4:
        pop_names = ['Metal-Poor', 'Intermediate-Low', 'Intermediate-High', 'Metal-Rich']
        pop_short = ['MP', 'IL', 'IH', 'MR']
        colors = BASE_COLORS[:4]
    elif best_n == 5:
        pop_names = ['Very Metal-Poor', 'Metal-Poor', 'Intermediate', 'Metal-Rich', 'Very Metal-Rich']
        pop_short = ['VMP', 'MP', 'I', 'MR', 'VMR']
        colors = ['#2166AC', '#92C5DE', '#F4A582', '#FDDBC7', '#B2182B']
    else:
        pop_names = [f'Population {i+1}' for i in range(best_n)]
        pop_short = [f'P{i+1}' for i in range(best_n)]
        cmap = get_cmap('viridis')
        colors = [cmap(i/best_n) for i in range(best_n)]
    
    # Reordenar etiquetas según medias ordenadas
    gmm = best_result['gmm']
    labels = gmm.predict(X)
    means = best_result['means']
    
    # Crear mapping para etiquetas ordenadas
    orig_means = gmm.means_.flatten()
    label_map = {}
    for i in range(best_n):
        # Encontrar el componente original correspondiente a la media i-ésima
        orig_idx = np.where(np.abs(orig_means - means[i]) < 1e-10)[0][0]
        label_map[orig_idx] = i
    
    labels = np.array([label_map[l] for l in labels])
    
    print(f"\n📊 PARÁMETROS FINALES ({best_n} POBLACIONES):")
    for i in range(best_n):
        n_objects = (labels == i).sum()
        print(f"  • {pop_names[i]} ({pop_short[i]}):")
        print(f"    [Fe/H] = {means[i]:.3f} ± {best_result['stds'][i]:.3f} dex")
        print(f"    Fracción = {best_result['weights'][i]:.3f} ({best_result['weights'][i]*100:.1f}%)")
        print(f"    N objetos = {n_objects}")
        print(f"    Calidad pertenencia = {best_result['quality'][i]:.3f}")
    
    if best_n > 1:
        print(f"\n📈 SEPARACIONES ENTRE COMPONENTES:")
        for i in range(best_n - 1):
            sep = best_result['separations'][i]
            significance = "✓ SIGNIFICATIVA" if sep > 2 else "✗ NO SIGNIFICATIVA"
            print(f"  • {pop_short[i]}-{pop_short[i+1]}: {sep:.2f}σ ({significance})")
    
    # Calcular factores de Bayes para todos los modelos (incluyendo no válidos)
    print(f"\n📊 RESUMEN DE EVIDENCIA BAYESIANA:")
    if best_n in results and 'bic' in results[best_n]:
        for n in range(2, 6):
            if n != best_n and n in results and 'bic' in results[n]:
                bf = calculate_bayes_factor(results[best_n]['bic'], results[n]['bic'])
                interpretation = interpret_bayes_factor(bf)
                validity = "VÁLIDO" if results[n].get('valid', False) else "NO VÁLIDO"
                print(f"  • BF({best_n} vs {n}) = {bf:.2e} - {interpretation} ({validity})")
    
    return gmm, labels, best_result, results, best_n, pop_names, pop_short, colors

# ============================================================================
# 4. ANÁLISIS BOOTSTRAP MEJORADO
# ============================================================================

def bootstrap_analysis_with_quality(feh_data, n_bootstrap=200):
    """
    Análisis bootstrap que aplica criterios de calidad - MEJORADO.
    """
    
    print("\n" + "="*80)
    print("ANÁLISIS BOOTSTRAP CON CRITERIOS DE CALIDAD")
    print("="*80)
    
    X = feh_data.reshape(-1, 1)
    n_samples = len(feh_data)
    
    # Resultados
    bootstrap_results = {
        'n_components_counts': {2: 0, 3: 0, 4: 0, 5: 0},
        'valid_models_by_n': {2: [], 3: [], 4: [], 5: []},
        'all_means_by_n': {2: [], 3: [], 4: [], 5: []},
        'all_bics_by_n': {2: [], 3: [], 4: [], 5: []}
    }
    
    print(f"  Ejecutando {n_bootstrap} réplicas bootstrap...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 50 == 0:
            print(f"    Réplica {i+1}/{n_bootstrap}")
        
        # Muestra bootstrap
        sample_idx = np.random.choice(n_samples, n_samples, replace=True)
        sample = feh_data[sample_idx]
        X_sample = sample.reshape(-1, 1)
        
        # Probar 2-5 componentes
        best_valid_model = None
        best_valid_bic = np.inf
        
        for n in [2, 3, 4, 5]:
            # Validar modelo con criterios de calidad
            try:
                gmm, quality, separations, means, stds, weights = validate_gmm_model_with_quality(
                    X_sample, n, random_state=i
                )
                
                if gmm is not None:
                    # Modelo válido
                    bic = gmm.bic(X_sample)
                    
                    model_info = {
                        'gmm': gmm,
                        'means': means,
                        'stds': stds,
                        'weights': weights,
                        'quality': quality,
                        'separations': separations,
                        'bic': bic,
                        'aic': gmm.aic(X_sample)
                    }
                    
                    bootstrap_results['valid_models_by_n'][n].append(model_info)
                    bootstrap_results['all_means_by_n'][n].append(means)
                    bootstrap_results['all_bics_by_n'][n].append(bic)
                    
                    # Seleccionar el mejor entre los válidos (menor BIC)
                    if bic < best_valid_bic:
                        best_valid_bic = bic
                        best_valid_model = n
            except Exception as e:
                # Silenciar errores en bootstrap
                continue
        
        # Contar el mejor modelo válido
        if best_valid_model is not None:
            bootstrap_results['n_components_counts'][best_valid_model] += 1
    
    # Calcular estadísticas
    total_counts = sum(bootstrap_results['n_components_counts'].values())
    
    print(f"\n📊 RESULTADOS BOOTSTRAP (con criterios de calidad):")
    print(f"  Total réplicas con modelos válidos: {total_counts}/{n_bootstrap}")
    print(f"  Tasa de éxito: {total_counts/n_bootstrap*100:.1f}%")
    
    if total_counts > 0:
        for n in [2, 3, 4, 5]:
            count = bootstrap_results['n_components_counts'][n]
            percentage = count / total_counts * 100 if total_counts > 0 else 0
            n_models = len(bootstrap_results['valid_models_by_n'][n])
            print(f"  • {n} componentes: {count} réplicas ({percentage:.1f}%), {n_models} modelos válidos")
    
    # Analizar estabilidad de parámetros para cada n
    print(f"\n📊 ESTABILIDAD DE PARÁMETROS POR NÚMERO DE COMPONENTES:")
    
    for n in [2, 3, 4, 5]:
        models_n = bootstrap_results['valid_models_by_n'][n]
        if len(models_n) > 10:  # Suficientes para estadísticas
            # Calcular intervalos de confianza para medias
            all_means = np.array([m['means'] for m in models_n])
            
            # Ordenar por medias (para consistencia)
            sorted_means = []
            for means_arr in all_means:
                sort_idx = np.argsort(means_arr)
                sorted_means.append(means_arr[sort_idx])
            sorted_means = np.array(sorted_means)
            
            print(f"\n  📊 n={n} (basado en {len(models_n)} modelos válidos):")
            
            # Estadísticas de BIC
            bics = np.array([m['bic'] for m in models_n])
            print(f"    BIC: {np.mean(bics):.1f} ± {np.std(bics):.1f}")
            
            # Intervalos de confianza para cada población
            for pop_idx in range(n):
                if pop_idx < sorted_means.shape[1]:
                    pop_means = sorted_means[:, pop_idx]
                    mean_val = np.mean(pop_means)
                    ci_lower = np.percentile(pop_means, 2.5)
                    ci_upper = np.percentile(pop_means, 97.5)
                    std_val = np.std(pop_means)
                    print(f"    Población {pop_idx+1}: ⟨[Fe/H]⟩ = {mean_val:.3f}±{std_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    return bootstrap_results

# ============================================================================
# 5. FIGURAS MEJORADAS
# ============================================================================

def create_comprehensive_paper_figures(feh_data, gmm_results, bootstrap_results, 
                                      labels, pop_names, pop_short, colors, 
                                      radii_kpc=None, output_dir='.'):
    """
    Crea todas las figuras para el paper - VERSIÓN MEJORADA.
    """
    
    print("\n🎨 Generando figuras para paper...")
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # FIGURA 1: ANÁLISIS GMM + BOOTSTRAP
    # ============================================================
    fig1 = plt.figure(figsize=(18, 14))
    gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.3, wspace=0.3)
    
    # Panel A: Distribución [Fe/H] con fit GMM
    ax1 = fig1.add_subplot(gs1[0, 0])
    
    # Histograma
    n, bins, patches = ax1.hist(feh_data, bins=30, density=True,
                               alpha=0.3, color='gray',
                               edgecolor='black', linewidth=0.5,
                               label=f'GC Data (n={len(feh_data)})')
    
    # Fit GMM
    x_fit = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 500)
    
    means = gmm_results['means']
    stds = gmm_results['stds']
    weights = gmm_results['weights']
    n_components = len(means)
    
    pdf_total = np.zeros_like(x_fit)
    for i in range(n_components):
        pdf_comp = weights[i] * norm.pdf(x_fit, means[i], stds[i])
        pdf_total += pdf_comp
        ax1.plot(x_fit, pdf_comp, '--', linewidth=2, 
                color=colors[i], alpha=0.6, 
                label=f'{pop_short[i]}: μ={means[i]:.2f}')
    
    ax1.plot(x_fit, pdf_total, 'k-', linewidth=3, label='Mixture model')
    
    ax1.set_xlabel('[Fe/H] (dex)', fontsize=14)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.set_title('(a) Metallicity Distribution with GMM Fit', fontsize=16)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # Panel B: BIC/AIC para diferentes n
    ax2 = fig1.add_subplot(gs1[0, 1])
    
    # Calcular BIC/AIC para todos los modelos
    n_values = list(range(2, 6))
    bic_values = []
    aic_values = []
    valid_mask = []
    
    for n in n_values:
        if n in gmm_results.get('all_results', {}) and 'bic' in gmm_results['all_results'][n]:
            bic_values.append(gmm_results['all_results'][n]['bic'])
            aic_values.append(gmm_results['all_results'][n]['aic'])
            valid_mask.append(gmm_results['all_results'][n].get('valid', False))
        else:
            # Calcular si no está disponible
            gmm_temp = GaussianMixture(n_components=n, random_state=42, n_init=10)
            gmm_temp.fit(feh_data.reshape(-1, 1))
            bic_values.append(gmm_temp.bic(feh_data.reshape(-1, 1)))
            aic_values.append(gmm_temp.aic(feh_data.reshape(-1, 1)))
            valid_mask.append(False)  # No es válido porque no pasó los criterios
    
    ax2.plot(n_values, bic_values, 'o-', color='#D55E00',
             linewidth=2, markersize=8, label='BIC')
    ax2.plot(n_values, aic_values, 's-', color='#0072B2',
             linewidth=2, markersize=8, label='AIC')
    
    # Marcar modelos válidos con círculos verdes
    for i, n in enumerate(n_values):
        if valid_mask[i]:
            ax2.scatter(n, bic_values[i], s=150, facecolors='none', 
                       edgecolors='green', linewidth=2, zorder=5)
            ax2.scatter(n, aic_values[i], s=150, facecolors='none', 
                       edgecolors='green', linewidth=2, zorder=5)
    
    # Marcar el n seleccionado
    selected_n = n_components
    ax2.axvline(x=selected_n, color='red', linestyle='--',
                alpha=0.7, linewidth=2, label=f'Selected (n={selected_n})')
    
    ax2.set_xlabel('Number of Components', fontsize=14)
    ax2.set_ylabel('Information Criterion', fontsize=14)
    ax2.set_title('(b) Model Selection Criteria', fontsize=16)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle=':')
    ax2.set_xticks(n_values)
    
    # Panel C: Resultados bootstrap
    ax3 = fig1.add_subplot(gs1[0, 2])
    
    bootstrap_counts = bootstrap_results['n_components_counts']
    n_components_list = list(bootstrap_counts.keys())
    counts = list(bootstrap_counts.values())
    total = sum(counts)
    
    if total > 0:
        percentages = [c/total*100 for c in counts]
        
        # Colores según porcentaje
        bar_colors = []
        for i, n in enumerate(n_components_list):
            if n == selected_n:
                bar_colors.append('red')
            elif percentages[i] > 20:
                bar_colors.append('orange')
            else:
                bar_colors.append('gray')
        
        bars = ax3.bar(n_components_list, percentages,
                      color=bar_colors,
                      alpha=0.7, edgecolor='black')
        
        for bar, percent, n in zip(bars, percentages, n_components_list):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percent:.1f}%', ha='center', va='bottom', fontsize=10)
            if n == selected_n:
                ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                        'SELECTED', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        ax3.set_xlabel('Number of Components', fontsize=14)
        ax3.set_ylabel('Bootstrap Optimal (%)', fontsize=14)
        ax3.set_title('(c) Bootstrap Model Stability', fontsize=16)
        ax3.grid(True, alpha=0.2, linestyle=':', axis='y')
        ax3.set_ylim(0, max(percentages) * 1.3)
        ax3.set_xticks(n_components_list)
    else:
        ax3.text(0.5, 0.5, 'No valid models in bootstrap',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('(c) Bootstrap Stability', fontsize=16)
        ax3.axis('off')
    
    # Panel D: [Fe/H] vs Radio (si hay datos espaciales)
    ax4 = fig1.add_subplot(gs1[1, 0])
    
    if radii_kpc is not None and len(radii_kpc) == len(feh_data):
        # Scatter plot con colores por población
        for i in range(n_components):
            mask = labels == i
            if mask.sum() > 0:
                ax4.scatter(radii_kpc[mask], feh_data[mask],
                           s=20, alpha=0.6, color=colors[i],
                           edgecolors='white', linewidth=0.2,
                           label=pop_short[i], rasterized=True)
        
        # Añadir línea de tendencia
        from scipy import stats as scipy_stats
        r_val, r_p = scipy_stats.pearsonr(np.log10(radii_kpc), feh_data)
        
        if r_p < 0.05:
            # Calcular regresión lineal
            coeffs = np.polyfit(np.log10(radii_kpc), feh_data, 1)
            x_line = np.logspace(np.log10(radii_kpc.min()), 
                                np.log10(radii_kpc.max()), 100)
            y_line = np.polyval(coeffs, np.log10(x_line))
            ax4.plot(x_line, y_line, 'k--', linewidth=2, 
                    label=f'r={r_val:.3f}, p={r_p:.3f}')
        
        ax4.set_xlabel('Projected Radius (kpc)', fontsize=14)
        ax4.set_ylabel('[Fe/H] (dex)', fontsize=14)
        ax4.set_title('(d) Metallicity vs Radius', fontsize=16)
        ax4.set_xscale('log')
        ax4.legend(fontsize=10, loc='upper right')
        ax4.grid(True, alpha=0.2, linestyle=':', which='both')
    else:
        # Mostrar separaciones si no hay datos espaciales
        separations = gmm_results.get('separations', [])
        if separations:
            x_pos = range(len(separations))
            bar_colors_sep = ['green' if s > 2 else 'orange' for s in separations]
            bars = ax4.bar(x_pos, separations,
                          color=bar_colors_sep,
                          alpha=0.7, edgecolor='black')
            
            ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='2σ threshold')
            
            # Añadir valores en las barras
            for bar, sep in zip(bars, separations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{sep:.2f}σ', ha='center', va='bottom', fontsize=10)
            
            ax4.set_xlabel('Adjacent Population Pairs', fontsize=14)
            ax4.set_ylabel('Separation (σ)', fontsize=14)
            ax4.set_title('(d) Population Separations', fontsize=16)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'{pop_short[i]}-{pop_short[i+1]}' 
                                for i in range(len(separations))], rotation=45)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.2, linestyle=':', axis='y')
        else:
            ax4.text(0.5, 0.5, 'No spatial data available',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('(d) Spatial Distribution', fontsize=16)
            ax4.axis('off')
    
    # Panel E: Calidad de pertenencia
    ax5 = fig1.add_subplot(gs1[1, 1])
    
    quality = gmm_results.get('quality', [])
    if quality:
        x_pos = range(len(quality))
        bars = ax5.bar(x_pos, quality,
                      color=colors[:len(quality)],
                      alpha=0.7, edgecolor='black')
        
        ax5.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Quality threshold')
        
        # Añadir valores en las barras
        for bar, q in zip(bars, quality):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{q:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax5.set_xlabel('Population', fontsize=14)
        ax5.set_ylabel('Membership Quality', fontsize=14)
        ax5.set_title('(e) Population Membership Quality', fontsize=16)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(pop_short[:len(quality)])
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.2, linestyle=':', axis='y')
        ax5.set_ylim(0, 1.1)
    else:
        ax5.axis('off')
    
    # Panel F: Resumen estadístico
    ax6 = fig1.add_subplot(gs1[1, 2])
    ax6.axis('off')
    
    summary_text = f"{n_components} GC POPULATIONS IN NGC 5128\n"
    summary_text += "Extended S-PLUS Fields Analysis\n"
    summary_text += "="*40 + "\n\n"
    
    for i in range(n_components):
        n_objects = (labels == i).sum()
        summary_text += f"{pop_short[i]}: {pop_names[i]}\n"
        summary_text += f"  ⟨[Fe/H]⟩ = {means[i]:.3f}±{stds[i]:.03f}\n"
        summary_text += f"  f = {weights[i]:.3f} (N={n_objects})\n"
        if i < len(quality):
            summary_text += f"  Quality = {quality[i]:.3f}\n\n"
        else:
            summary_text += "\n"
    
    if n_components > 1 and 'separations' in gmm_results:
        summary_text += "SEPARATIONS:\n"
        for i in range(n_components - 1):
            sep = gmm_results['separations'][i]
            summary_text += f"  {pop_short[i]}-{pop_short[i+1]}: {sep:.2f}σ\n"
    
    summary_text += f"\nTotal GCs: {len(feh_data)}\n"
    
    if radii_kpc is not None:
        summary_text += f"Radial coverage: {radii_kpc.min():.1f}-{radii_kpc.max():.0f} kpc\n"
    
    # Añadir información de bootstrap
    bootstrap_summary = "\nBOOTSTRAP STABILITY:\n"
    for n in [2, 3, 4, 5]:
        count = bootstrap_counts.get(n, 0)
        percentage = count/total*100 if total > 0 else 0
        bootstrap_summary += f"  n={n}: {percentage:.1f}%\n"
    
    summary_text += bootstrap_summary
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', linespacing=1.5,
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))
    
    # Panel G: Distribución radial por población (si hay datos)
    if radii_kpc is not None and len(radii_kpc) == len(feh_data):
        ax7 = fig1.add_subplot(gs1[2, :])
        
        bins = np.logspace(np.log10(1), np.log10(200), 20)
        
        for i in range(n_components):
            mask = labels == i
            if mask.sum() > 0:
                counts, bins_edges = np.histogram(radii_kpc[mask], bins=bins, density=True)
                bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
                
                ax7.step(bin_centers, counts, where='mid',
                        color=colors[i], linewidth=2.5,
                        label=f'{pop_short[i]} (N={(mask).sum()})')
        
        ax7.set_xlabel('Projected Radius (kpc)', fontsize=14)
        ax7.set_ylabel('Normalized Density', fontsize=14)
        ax7.set_title('(g) Radial Distribution by Population', fontsize=16)
        ax7.set_xscale('log')
        ax7.set_yscale('log')
        ax7.legend(fontsize=10, loc='upper right')
        ax7.grid(True, alpha=0.2, linestyle=':', which='both')
    else:
        # Usar el espacio para análisis de robustez
        ax7 = fig1.add_subplot(gs1[2, :])
        ax7.axis('off')
        
        stats_text = "STATISTICAL ROBUSTNESS ASSESSMENT\n"
        stats_text += "="*50 + "\n\n"
        
        # Información de criterios de calidad
        stats_text += "QUALITY CRITERIA:\n"
        stats_text += f"• Membership quality > 0.7: {gmm_results.get('min_quality', 0):.3f} ✓\n"
        if n_components > 1:
            all_good = all(s > 2 for s in gmm_results.get('separations', []))
            stats_text += f"• All separations > 2σ: {all_good} ✓\n"
        
        # Información de bootstrap
        stats_text += "\nBOOTSTRAP VALIDATION:\n"
        n_bootstrap = sum(bootstrap_counts.values())
        stats_text += f"• Total replicates: {n_bootstrap}\n"
        successful_fits = sum(bootstrap_results['n_components_counts'].values())
        stats_text += f"• Successful fits: {successful_fits} ({successful_fits/n_bootstrap*100:.1f}%)\n"
        
        # Modelo seleccionado en bootstrap
        selected_count = bootstrap_counts.get(selected_n, 0)
        selected_percentage = selected_count/total*100 if total > 0 else 0
        stats_text += f"• Selected model (n={selected_n}): {selected_percentage:.1f}%\n"
        
        # Interpretación
        stats_text += "\nINTERPRETATION:\n"
        if selected_n == 2:
            stats_text += "• Strong evidence for classical bimodality\n"
            stats_text += "• All quality criteria satisfied\n"
            stats_text += "• Consistent with hierarchical formation models\n"
        elif selected_n == 5:
            stats_text += "• Evidence for 5 distinct metallicity populations\n"
            stats_text += "• Suggests complex formation/accretion history\n"
            stats_text += "• May indicate multiple merger events\n"
        else:
            stats_text += f"• Evidence for {selected_n} distinct populations\n"
            stats_text += "• Suggests more complex formation history\n"
            stats_text += "• May indicate multiple accretion episodes\n"
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    
    fig1.suptitle('Comprehensive Analysis of Globular Cluster Populations in NGC 5128\n' +
                 f'Robust GMM with Quality Criteria: {n_components} Populations Identified',
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig1.savefig(f'{output_dir}/Figure1_Comprehensive_Analysis.pdf',
                dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Figura 1 guardada: {output_dir}/Figure1_Comprehensive_Analysis.pdf")
    
    # ============================================================
    # FIGURA 2: RESUMEN PARA PAPER
    # ============================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel izquierdo: Distribución [Fe/H] con GMM
    # Histograma con KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(feh_data)
    x_kde = np.linspace(feh_data.min() - 0.3, feh_data.max() + 0.3, 500)
    
    ax1.hist(feh_data, bins=25, density=True, alpha=0.3,
            color='gray', edgecolor='black', linewidth=0.5,
            label='GC Data')
    
    # Fit GMM
    pdf_total = np.zeros_like(x_fit)
    for i in range(n_components):
        pdf_comp = weights[i] * norm.pdf(x_fit, means[i], stds[i])
        pdf_total += pdf_comp
        ax1.plot(x_fit, pdf_comp, '--', linewidth=1.5,
                color=colors[i], alpha=0.6)
    
    ax1.plot(x_fit, pdf_total, 'k-', linewidth=2.5, label='GMM fit')
    ax1.plot(x_kde, kde(x_kde), 'b-', linewidth=1.5, alpha=0.5, label='KDE')
    
    ax1.set_xlabel('[Fe/H] (dex)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'Metallicity Distribution: {n_components} Populations', fontsize=14)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.15, linestyle=':')
    
    # Texto con parámetros
    param_text = f"N = {len(feh_data)} GCs\n"
    for i in range(n_components):
        n_objects = (labels == i).sum()
        param_text += f'{pop_short[i]}: μ={means[i]:.2f}, σ={stds[i]:.03f}\n'
        param_text += f'  f={weights[i]:.2f} (N={n_objects})\n'
    
    ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                     edgecolor='gray', linewidth=0.5))
    
    # Panel derecho: Resultados bootstrap
    ax2 = fig2.add_subplot(1, 2, 2)
    
    if total > 0:
        n_components_list = list(bootstrap_counts.keys())
        counts = list(bootstrap_counts.values())
        percentages = [c/total*100 for c in counts]
        
        # Colormap para las barras
        cmap = get_cmap('viridis')
        bar_colors = [cmap(i/len(n_components_list)) for i in range(len(n_components_list))]
        
        bars = ax2.bar(n_components_list, percentages,
                      color=bar_colors,
                      alpha=0.7, edgecolor='black')
        
        # Resaltar el modelo seleccionado
        if selected_n in n_components_list:
            idx = n_components_list.index(selected_n)
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(3)
        
        for bar, percent, n in zip(bars, percentages, n_components_list):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percent:.1f}%', ha='center', va='bottom', fontsize=9)
            if n == selected_n:
                ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                        'SELECTED', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Bootstrap Optimal (%)', fontsize=12)
        ax2.set_title('Bootstrap Model Stability', fontsize=14)
        ax2.grid(True, alpha=0.15, linestyle=':', axis='y')
        ax2.set_ylim(0, max(percentages) * 1.3)
        ax2.set_xticks(n_components_list)
    else:
        ax2.text(0.5, 0.5, 'No valid bootstrap models',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Bootstrap Analysis', fontsize=14)
        ax2.axis('off')
    
    fig2.suptitle(f'NGC 5128 Globular Clusters: {n_components} Metallicity Populations\n' +
                 f'Robust GMM Analysis with Bootstrap Validation',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig2.savefig(f'{output_dir}/Figure2_Paper_Summary.pdf',
                dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Figura 2 guardada: {output_dir}/Figure2_Paper_Summary.pdf")

# ============================================================================
# 6. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del análisis."""
    
    print("\n" + "="*80)
    print("NGC 5128 GCs - ROBUST GMM WITH BAYESIAN BOOTSTRAP ANALYSIS")
    print("="*80)
    
    # Crear directorio de salida
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"NGC5128_Robust_Bootstrap_Analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Directorio de resultados: {output_dir}")
    
    # 1. Cargar datos
    print("\n[1] CARGANDO Y PREPARANDO DATOS...")
    df = load_and_prepare_data()
    
    if len(df) == 0:
        print("❌ No hay datos para analizar")
        return
    
    feh_data = df['FeH'].values
    print(f"✓ {len(feh_data)} GCs cargados")
    print(f"  [Fe/H]: mean={feh_data.mean():.3f}, std={feh_data.std():.3f}")
    print(f"  Range: [{feh_data.min():.2f}, {feh_data.max():.2f}]")
    
    # 2. Calcular radios proyectados
    print("\n[2] CALCULANDO RADIOS PROYECTADOS...")
    radii_results = calculate_projected_radii(df)
    
    if radii_results[0] is not None:
        radii_kpc, valid_mask = radii_results
        print(f"✓ Radios calculados para {len(radii_kpc)} objetos")
        print(f"  Range: {radii_kpc.min():.1f} - {radii_kpc.max():.0f} kpc")
        
        # Calcular correlación [Fe/H] vs radio
        from scipy import stats as scipy_stats
        r_val, r_p = scipy_stats.pearsonr(np.log10(radii_kpc), feh_data)
        print(f"  Correlación [Fe/H] vs log(R): r={r_val:.3f}, p={r_p:.3f}")
    else:
        radii_kpc = None
        valid_mask = None
        print("⚠️  No se pudieron calcular radios proyectados")
    
    # 3. Análisis GMM robusto con criterios de calidad
    print("\n[3] ANÁLISIS GMM CON CRITERIOS DE CALIDAD...")
    gmm, labels, gmm_results, all_results, best_n, pop_names, pop_short, colors = \
        perform_robust_gmm_analysis(feh_data)
    
    # Guardar todos los resultados en gmm_results
    gmm_results['all_results'] = all_results
    
    # Guardar etiquetas en el DataFrame
    df['GMM_Population'] = labels
    df['Population_Name'] = [pop_names[l] for l in labels]
    df['Population_Short'] = [pop_short[l] for l in labels]
    
    # Agregar probabilidades de pertenencia
    probs = gmm.predict_proba(feh_data.reshape(-1, 1))
    for i in range(best_n):
        df[f'Prob_{pop_short[i]}'] = probs[:, i]
    
    if radii_kpc is not None and valid_mask is not None:
        df['R_proj_kpc'] = np.nan
        df.loc[valid_mask, 'R_proj_kpc'] = radii_kpc
        df['log_R_proj_kpc'] = np.log10(df['R_proj_kpc'])
    
    # 4. Análisis bootstrap con criterios de calidad
    print("\n[4] ANÁLISIS BOOTSTRAP CON CRITERIOS DE CALIDAD...")
    n_bootstrap = 500  # Número aumentado para mejor estadística
    bootstrap_results = bootstrap_analysis_with_quality(feh_data, n_bootstrap=n_bootstrap)
    
    # 5. Crear figuras
    print("\n[5] CREANDO FIGURAS PARA PAPER...")
    # Preparar datos espaciales para las figuras
    if radii_kpc is not None and valid_mask is not None:
        feh_spatial = feh_data[valid_mask]
        labels_spatial = labels[valid_mask]
    else:
        feh_spatial = feh_data
        labels_spatial = labels
    
    create_comprehensive_paper_figures(
        feh_spatial, gmm_results, bootstrap_results,
        labels_spatial, pop_names, pop_short, colors,
        radii_kpc, output_dir
    )
    
    # 6. Guardar resultados
    print("\n[6] GUARDANDO RESULTADOS...")
    
    # Guardar DataFrame con clasificaciones
    csv_path = f'{output_dir}/ngc5128_gcs_classified_{best_n}pop.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Datos clasificados guardados: {csv_path}")
    
    # Guardar parámetros del modelo
    model_params = {
        'analysis_metadata': {
            'date': datetime.datetime.now().isoformat(),
            'version': '2.0_unified_corrected',
            'selection_criteria': 'Quality > 0.7, Separation > 2σ, Lowest BIC',
            'n_objects_total': int(len(feh_data)),
            'dataset': 'Extended S-PLUS fields'
        },
        'selected_model': {
            'n_components': int(best_n),
            'pop_names': pop_names,
            'pop_short': pop_short,
            'means': [float(x) for x in gmm_results['means']],
            'stds': [float(x) for x in gmm_results['stds']],
            'weights': [float(x) for x in gmm_results['weights']],
            'quality': [float(x) for x in gmm_results.get('quality', [])],
            'separations': [float(x) for x in gmm_results.get('separations', [])],
            'bic': float(gmm_results['bic']),
            'aic': float(gmm_results['aic']),
            'min_quality': float(gmm_results.get('min_quality', 0)),
            'avg_quality': float(gmm_results.get('avg_quality', 0))
        },
        'all_models': {
            str(n): {
                'bic': float(all_results[n]['bic']) if 'bic' in all_results[n] else None,
                'aic': float(all_results[n]['aic']) if 'aic' in all_results[n] else None,
                'valid': bool(all_results[n].get('valid', False)),
                'means': [float(x) for x in all_results[n].get('means', [])] if 'means' in all_results[n] else []
            } for n in range(2, 6) if n in all_results
        },
        'bootstrap_results': {
            'n_bootstrap': n_bootstrap,
            'n_components_counts': {str(k): int(v) for k, v in bootstrap_results['n_components_counts'].items()},
            'success_rate': float(sum(bootstrap_results['n_components_counts'].values()) / n_bootstrap * 100)
        },
        'spatial_statistics': {
            'n_objects_with_radii': int(len(radii_kpc)) if radii_kpc is not None else 0,
            'radial_range_kpc': [float(radii_kpc.min()), float(radii_kpc.max())] if radii_kpc is not None else None,
            'radial_median_kpc': float(np.median(radii_kpc)) if radii_kpc is not None else None
        }
    }
    
    json_path = f'{output_dir}/gmm_parameters_{best_n}pop.json'
    with open(json_path, 'w') as f:
        json.dump(model_params, f, indent=4, ensure_ascii=False)
    print(f"✓ Parámetros del modelo: {json_path}")
    
    # 7. Resumen científico completo
    print("\n" + "="*80)
    print("📄 RESUMEN CIENTÍFICO COMPLETO - ANÁLISIS UNIFICADO")
    print("="*80)
    
    total_bootstrap = sum(bootstrap_results['n_components_counts'].values())
    bootstrap_selected = bootstrap_results['n_components_counts'].get(best_n, 0)
    bootstrap_percentage = bootstrap_selected / total_bootstrap * 100 if total_bootstrap > 0 else 0
    
    print(f"\n🎯 HALLAZGOS PRINCIPALES:")
    print(f"   • {best_n} poblaciones identificadas mediante GMM con criterios estrictos")
    
    if best_n == 2:
        print(f"   • CONFIRMACIÓN ROBUSTA DE LA BIMODALIDAD CLÁSICA")
        print(f"   • Separación MP-MR: {gmm_results['separations'][0]:.2f}σ (significativa)")
        print(f"   • Calidad excelente: MP={gmm_results['quality'][0]:.3f}, MR={gmm_results['quality'][1]:.3f}")
        print(f"   • Bootstrap: {bootstrap_percentage:.1f}% de réplicas apoyan 2 componentes")
    elif best_n == 5:
        print(f"   • EVIDENCIA PARA 5 POBLACIONES METÁLICAS DISTINTAS")
        print(f"   • Todas las separaciones >2σ (significativas)")
        print(f"   • Calidad media: {gmm_results['avg_quality']:.3f}")
        print(f"   • Bootstrap: {bootstrap_percentage:.1f}% de réplicas apoyan 5 componentes")
    else:
        print(f"   • EVIDENCIA PARA {best_n} POBLACIONES DISTINTAS")
        print(f"   • Separaciones significativas entre componentes")
        print(f"   • Calidad media: {gmm_results['avg_quality']:.3f}")
        print(f"   • Bootstrap: {bootstrap_percentage:.1f}% de réplicas apoyan {best_n} componentes")
    
    print(f"\n📊 PARÁMETROS DE POBLACIONES:")
    for i in range(best_n):
        n_objects = (labels == i).sum()
        print(f"   • {pop_names[i]} ({pop_short[i]}):")
        print(f"     [Fe/H] = {gmm_results['means'][i]:.3f} ± {gmm_results['stds'][i]:.03f} dex")
        print(f"     Fracción = {gmm_results['weights'][i]:.3f} ({gmm_results['weights'][i]*100:.1f}%, N={n_objects})")
    
    if radii_kpc is not None:
        print(f"\n📍 DISTRIBUCIÓN ESPACIAL:")
        print(f"   • {len(radii_kpc)} GCs con información espacial")
        print(f"   • Cobertura radial extendida: {radii_kpc.min():.1f}-{radii_kpc.max():.0f} kpc")
        if 'r_val' in locals():
            print(f"   • Correlación [Fe/H] vs radio: r={r_val:.3f}, p={r_p:.3f}")
            if r_p < 0.05:
                print(f"     → Gradiente radial significativo detectado")
    
    print(f"\n📈 EVIDENCIA ESTADÍSTICA:")
    print(f"   • Criterios de calidad cumplidos: ✓ Calidad > 0.7, ✓ Separaciones > 2σ")
    print(f"   • Bootstrap validation: {total_bootstrap}/{n_bootstrap} réplicas exitosas")
    print(f"   • Modelo seleccionado en {bootstrap_percentage:.1f}% de réplicas bootstrap")
    
    # Factores de Bayes
    if best_n in all_results and 'bic' in all_results[best_n]:
        print(f"\n📊 FACTORES DE BAYES (comparando con otros modelos):")
        for n in range(2, 6):
            if n != best_n and n in all_results and 'bic' in all_results[n]:
                bf = calculate_bayes_factor(all_results[best_n]['bic'], all_results[n]['bic'])
                if bf > 1:
                    print(f"   • El modelo n={best_n} es {bf:.1e} veces más probable que n={n}")
                else:
                    print(f"   • El modelo n={n} es {1/bf:.1e} veces más probable que n={best_n}")
    
    print(f"\n🔬 IMPLICACIONES ASTROFÍSICAS:")
    if best_n == 2:
        print(f"   • Formación en dos episodios principales distintos")
        print(f"   • Población MR: origen in situ en el bulge/halo interno")
        print(f"   • Población MP: acreción temprana o formación extendida")
        print(f"   • Consistente con modelos de formación jerárquica")
    elif best_n == 5:
        print(f"   • Historia de formación extremadamente compleja")
        print(f"   • Múltiples episodios de formación y acreción")
        print(f"   • Similar a galaxias elípticas masivas con historias de fusión ricas")
        print(f"   • Podría indicar 5 eventos de formación/acreción distintos")
    else:
        print(f"   • Historia de formación más compleja")
        print(f"   • Múltiples episodios de formación o acreción")
        print(f"   • Similar a otras elípticas masivas con sistemas complejos")
    
    print(f"\n📁 ARCHIVOS GENERADOS en '{output_dir}/':")
    print(f"   1. Figure1_Comprehensive_Analysis.pdf - Análisis completo")
    print(f"   2. Figure2_Paper_Summary.pdf - Resumen para paper")
    print(f"   3. ngc5128_gcs_classified_{best_n}pop.csv - Catálogo clasificado")
    print(f"   4. gmm_parameters_{best_n}pop.json - Parámetros y metadatos")
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS UNIFICADO COMPLETADO - LISTO PARA PUBLICACIÓN")
    print("="*80)
    
    # Sugerencia para el paper
    print(f"\n💡 SUGERENCIA PARA LA SECCIÓN DE MÉTODOS:")
    print(f"   'Utilizamos Gaussian Mixture Models (GMM) con validación robusta mediante")
    print(f"   múltiples inicializaciones y criterios estrictos de calidad (calidad de")
    print(f"   pertenencia > 0.7 y separación > 2σ entre componentes). La selección del")
    print(f"   modelo óptimo se basó en el criterio de información bayesiano (BIC),")
    print(f"   complementada con análisis bootstrap ({n_bootstrap} réplicas) para evaluar")
    print(f"   la estabilidad de los resultados.'")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
