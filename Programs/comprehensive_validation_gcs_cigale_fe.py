#!/usr/bin/env python3
# comprehensive_validation_gcs_cigale_fe.py - Validación completa con datos CIGALE de [Fe/H]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
import datetime
from astropy.table import Table
import re

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
})

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
POP_NAMES = ['Metal-Poor', 'Intermediate-Low', 'Intermediate-High', 'Metal-Rich']

# ============================================================================
# 1. FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ============================================================================

def load_catalog(catalog_path):
    """
    Carga el catálogo fotométrico y realiza limpieza básica.
    """
    print(f"📂 Cargando catálogo: {catalog_path}")
    
    try:
        df = pd.read_csv(catalog_path)
        print(f"✓ Catálogo cargado: {len(df)} objetos")
        
        # Mostrar algunas columnas clave
        print("\n🔍 Columnas clave del catálogo fotométrico:")
        key_cols = ['ID', 'RAJ2000', 'DEJ2000', 'Prob', 'Rgc', 'gmag_corr', 'e_gmag_corr']
        for col in key_cols:
            if col in df.columns:
                print(f"  • {col}: presente")
            else:
                print(f"  • {col}: NO presente")
        
        # Buscar columnas ID alternativas
        id_cols = [col for col in df.columns if 'id' in col.lower() or 'ID' in col or 'recno' in col.lower()]
        if id_cols:
            print(f"\n🔍 Columnas ID disponibles en fotometría: {id_cols}")
            # Mostrar primeros valores de cada columna ID
            for id_col in id_cols[:3]:  # Mostrar máximo 3 columnas
                print(f"  - '{id_col}': primeros valores -> {df[id_col].head().tolist()}")
        
        # Verificar criterios de calidad aplicados
        print("\n🔍 CRITERIOS DE CALIDAD APLICADOS:")
        if 'good_filters_count' in df.columns:
            print(f"  • good_filters_count: presente")
            print(f"    - Media: {df['good_filters_count'].mean():.1f}")
            print(f"    - Mín: {df['good_filters_count'].min()}")
            print(f"    - Máx: {df['good_filters_count'].max()}")
        
        if 'Prob' in df.columns:
            print(f"  • Prob: presente")
            print(f"    - Media: {df['Prob'].mean():.3f}")
            print(f"    - ≥0.7: {(df['Prob'] >= 0.7).sum()} objetos")
        
        return df
    
    except Exception as e:
        print(f"❌ Error cargando catálogo: {e}")
        return None

def load_cigale_results(cigale_path='out/results.fits'):
    """
    Carga los resultados de CIGALE desde el archivo FITS.
    """
    print(f"\n🧪 Cargando resultados CIGALE: {cigale_path}")
    
    try:
        # Leer la tabla de CIGALE
        table = Table.read(cigale_path, hdu=1)
        print(f"✓ HDU 1 cargado: {len(table)} filas, {len(table.columns)} columnas")
        
        df_cigale = table.to_pandas()
        
        # Buscar columna ID en CIGALE
        cigale_id_cols = [col for col in df_cigale.columns if col.lower() == 'id']
        
        if cigale_id_cols:
            cigale_id_col = cigale_id_cols[0]
            print(f"✓ Columna ID en CIGALE: '{cigale_id_col}'")
            
            # Convertir bytes a string si es necesario
            if df_cigale[cigale_id_col].dtype == object:
                # Intentar convertir bytes a string
                try:
                    df_cigale[cigale_id_col] = df_cigale[cigale_id_col].apply(
                        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
                    )
                    print(f"✓ IDs convertidos de bytes a string")
                except:
                    df_cigale[cigale_id_col] = df_cigale[cigale_id_col].astype(str)
            
            print(f"  Primeros valores: {df_cigale[cigale_id_col].head().tolist()}")
            
            # Extraer el número del ID (ej: '2421' de 'T17-2421')
            df_cigale['ID_NUM'] = df_cigale[cigale_id_col].apply(extract_id_number)
            print(f"✓ Números extraídos de IDs CIGALE")
            print(f"  Primeros números: {df_cigale['ID_NUM'].head().tolist()}")
            
            # Verificar cuántos se extrajeron correctamente
            n_extracted = df_cigale['ID_NUM'].notna().sum()
            print(f"  ✓ Números extraídos correctamente: {n_extracted}/{len(df_cigale)}")
            
        else:
            print("❌ No se encontró columna ID en CIGALE")
            return None
        
        # Buscar columna de metalicidad
        metal_cols = [col for col in df_cigale.columns if 'metall' in col.lower() or 'metal' in col.lower()]
        if metal_cols:
            metal_col = metal_cols[0]
            print(f"\n✓ Columna de metalicidad encontrada: {metal_col}")
            
            # Calcular [Fe/H] = log10(Z/Z_solar)
            Z_solar = 0.02
            
            # Asegurarse de que los valores sean positivos
            df_cigale[metal_col] = pd.to_numeric(df_cigale[metal_col], errors='coerce')
            df_cigale = df_cigale[df_cigale[metal_col] > 0]
            
            # Calcular [Fe/H]
            df_cigale['FeH'] = np.log10(df_cigale[metal_col] / Z_solar)
            
            # Filtrar por chi2 reducido si existe la columna
            chi2_cols = [col for col in df_cigale.columns if 'chi' in col.lower() and 'reduced' in col.lower()]
            if chi2_cols:
                chi2_col = chi2_cols[0]
                df_cigale[chi2_col] = pd.to_numeric(df_cigale[chi2_col], errors='coerce')
                df_cigale = df_cigale[df_cigale[chi2_col] <= 2.0]
                print(f"✓ Después de filtro {chi2_col} ≤ 2.0: {len(df_cigale)} objetos")
            
            # Filtrar outliers usando IQR
            Q1 = df_cigale['FeH'].quantile(0.01)
            Q3 = df_cigale['FeH'].quantile(0.99)
            IQR = Q3 - Q1
            df_cigale = df_cigale[(df_cigale['FeH'] >= Q1 - 1.5*IQR) & (df_cigale['FeH'] <= Q3 + 1.5*IQR)]
            
            print(f"✓ Después de filtro outliers IQR: {len(df_cigale)} objetos")
            print(f"  • [Fe/H] range: [{df_cigale['FeH'].min():.2f}, {df_cigale['FeH'].max():.2f}]")
            print(f"  • Mean: {df_cigale['FeH'].mean():.3f} ± {df_cigale['FeH'].std():.3f} dex")
            
            # Mantener solo columnas necesarias
            keep_cols = ['ID_NUM', 'FeH']
            if chi2_cols:
                keep_cols.append(chi2_col)
            df_cigale = df_cigale[keep_cols]
            
        else:
            print("❌ No se encontró columna de metalicidad")
            return None
        
        return df_cigale
    
    except Exception as e:
        print(f"❌ Error cargando resultados CIGALE: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_id_number(id_str):
    """
    Extrae el número del ID, manejando diferentes formatos.
    Ejemplos:
    - 'T17-2421' -> 2421
    - '2421' -> 2421
    - b'T17-2421' -> 2421 (después de decode)
    """
    if pd.isna(id_str):
        return None
    
    id_str = str(id_str)
    
    # Si ya es un número, devolverlo
    if id_str.isdigit():
        return int(id_str)
    
    # Buscar patrones como 'T17-2421' o 'T17_2421'
    patterns = [
        r'T17[-_](\d+)',  # T17-2421 o T17_2421
        r'[A-Za-z]+[-_]?(\d+)',  # Cualquier letra seguida de números
        r'(\d+)',  # Solo números en cualquier parte
    ]
    
    for pattern in patterns:
        match = re.search(pattern, id_str)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    
    return None

def match_catalogs_by_recno(df_phot, df_cigale):
    """
    Cruza los catálogos por recno (el número extraído de CIGALE).
    """
    print(f"\n🔄 Cruzando catálogos por recno...")
    print(f"  • Catálogo fotométrico: {len(df_phot)} objetos")
    print(f"  • Catálogo CIGALE: {len(df_cigale)} objetos")
    
    if 'recno' not in df_phot.columns:
        print("❌ El catálogo fotométrico no tiene columna 'recno'")
        return None
    
    if 'ID_NUM' not in df_cigale.columns:
        print("❌ El catálogo CIGALE no tiene columna 'ID_NUM'")
        return None
    
    # Asegurarnos de que recno sea numérico
    df_phot['recno_num'] = pd.to_numeric(df_phot['recno'], errors='coerce')
    df_cigale['ID_NUM_num'] = pd.to_numeric(df_cigale['ID_NUM'], errors='coerce')
    
    # Contar valores válidos
    n_valid_phot = df_phot['recno_num'].notna().sum()
    n_valid_cigale = df_cigale['ID_NUM_num'].notna().sum()
    
    print(f"  ✓ Valores numéricos válidos en fotometría: {n_valid_phot}/{len(df_phot)}")
    print(f"  ✓ Valores numéricos válidos en CIGALE: {n_valid_cigale}/{len(df_cigale)}")
    
    # Hacer el merge por número
    df_combined = pd.merge(
        df_phot[df_phot['recno_num'].notna()], 
        df_cigale[df_cigale['ID_NUM_num'].notna()][['ID_NUM_num', 'FeH']], 
        left_on='recno_num', 
        right_on='ID_NUM_num', 
        how='inner'
    )
    
    # Limpiar columnas temporales
    df_combined = df_combined.drop(['recno_num', 'ID_NUM_num'], axis=1)
    
    print(f"✓ Coincidencias encontradas: {len(df_combined)} objetos")
    print(f"  • Tasa de coincidencia: {len(df_combined)/len(df_phot)*100:.1f}% del catálogo fotométrico")
    print(f"  • Tasa de coincidencia: {len(df_combined)/len(df_cigale)*100:.1f}% del catálogo CIGALE")
    
    if len(df_combined) == 0:
        print("❌ No se encontraron coincidencias")
        return None
    
    # Mostrar ejemplos de coincidencias
    print(f"\n📋 Ejemplos de coincidencias (primeros 5):")
    print(f"  recno (fotometría): {df_combined['recno'].head().tolist()}")
    print(f"  [Fe/H] (CIGALE): {df_combined['FeH'].head().tolist()}")
    
    return df_combined

def check_photometric_quality(df):
    """
    Verifica la calidad de las magnitudes corregidas por extinción.
    Basado en los criterios de tu script de optimización.
    """
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE CALIDAD FOTOMÉTRICA (basado en optimize_catalog_size.py)")
    print("="*80)
    
    print(f"• Total de objetos: {len(df)}")
    
    # 1. Verificar probabilidad
    if 'Prob' in df.columns:
        print(f"\n🎯 PROBABILIDAD GC:")
        print(f"  • Media: {df['Prob'].mean():.3f}")
        print(f"  • ≥0.7: {(df['Prob'] >= 0.7).sum()} objetos ({df['Prob'].mean()/len(df)*100:.1f}%)")
        print(f"  • ≥0.5: {(df['Prob'] >= 0.5).sum()} objetos ({df['Prob'].mean()/len(df)*100:.1f}%)")
    
    # 2. Verificar good_filters_count
    if 'good_filters_count' in df.columns:
        print(f"\n🎯 CONTEO DE FILTROS BUENOS:")
        print(f"  • Media: {df['good_filters_count'].mean():.1f}")
        print(f"  • <4: {(df['good_filters_count'] < 4).sum()} objetos")
        print(f"  • 4-6: {((df['good_filters_count'] >= 4) & (df['good_filters_count'] <= 6)).sum()} objetos")
        print(f"  • ≥7: {(df['good_filters_count'] >= 7).sum()} objetos")
    
    # 3. Verificar errores en filtros S-PLUS (apertura 3)
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    print(f"\n🔍 ERRORES FOTOMÉTRICOS MEDIOS (S-PLUS, apertura 3, corregidos):")
    
    for filt in splus_filters:
        err_col = f'MAGERR_{filt}_3_corr'
        if err_col in df.columns:
            avg_err = df[err_col].mean()
            median_err = df[err_col].median()
            print(f"  • {filt}: media = {avg_err:.3f} mag, mediana = {median_err:.3f} mag")
    
    # 4. Verificar filtros DECam
    decam_filters = ['g', 'r', 'i']
    print(f"\n🔍 ERRORES FOTOMÉTRICOS MEDIOS (DECam, corregidos):")
    
    for filt in decam_filters:
        mag_col = f'{filt}mag_corr'
        err_col = f'e_{filt}mag_corr'
        
        if mag_col in df.columns and err_col in df.columns:
            avg_err = df[err_col].mean()
            median_err = df[err_col].median()
            print(f"  • {filt}: media = {avg_err:.3f} mag, mediana = {median_err:.3f} mag")
            print(f"    - Magnitud media: {df[mag_col].mean():.2f} mag")
    
    return df

def apply_quality_cuts(df, min_prob=0.5, min_good_filters=4):
    """
    Aplica cortes de calidad fotométrica basados en tus criterios.
    """
    print("\n" + "="*80)
    print("✂️  APLICANDO CORTES DE CALIDAD (basado en tus criterios)")
    print("="*80)
    
    initial_count = len(df)
    quality_mask = pd.Series(True, index=df.index)
    
    # 1. Corte por probabilidad GC
    if 'Prob' in df.columns:
        print(f"\n1. Corte por probabilidad GC:")
        prob_mask = df['Prob'] >= min_prob
        print(f"  • Prob ≥ {min_prob}: {prob_mask.sum()}/{len(df)} objetos")
        quality_mask &= prob_mask
    
    # 2. Corte por buen recuento de filtros
    if 'good_filters_count' in df.columns:
        print(f"\n2. Corte por buen conteo de filtros:")
        filter_mask = df['good_filters_count'] >= min_good_filters
        print(f"  • good_filters_count ≥ {min_good_filters}: {filter_mask.sum()}/{len(df)} objetos")
        quality_mask &= filter_mask
    
    # 3. Cortes adicionales por errores fotométricos (opcional)
    print(f"\n3. Cortes opcionales por errores fotométricos:")
    
    # Filtros DECam principales
    decam_filters = ['g', 'r', 'i']
    for filt in decam_filters:
        err_col = f'e_{filt}mag_corr'
        
        if err_col in df.columns:
            # Usar umbral más generoso (0.2 mag) basado en tus criterios
            mask = (df[err_col] < 0.2) & (df[err_col] > 0)
            print(f"  • {filt} error < 0.2 mag: {mask.sum()}/{len(df)} objetos")
            # No aplicamos este corte por ahora, solo informamos
    
    df_filtered = df[quality_mask].copy()
    print(f"\n✅ RESULTADO: {len(df_filtered)}/{initial_count} objetos pasan los cortes "
          f"({len(df_filtered)/initial_count*100:.1f}%)")
    
    return df_filtered

def check_contamination(df):
    """
    Verifica posibles fuentes de contaminación.
    """
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE CONTAMINACIÓN")
    print("="*80)
    
    # 1. Distribución de probabilidad GC
    if 'Prob' in df.columns:
        print("\n1. DISTRIBUCIÓN DE PROBABILIDAD GC:")
        print(f"  • Media: {df['Prob'].mean():.3f}")
        print(f"  • Mediana: {df['Prob'].median():.3f}")
        print(f"  • <0.5: {(df['Prob'] < 0.5).sum():4d} objetos")
        print(f"  • 0.5-0.8: {((df['Prob'] >= 0.5) & (df['Prob'] < 0.8)).sum():4d} objetos")
        print(f"  • 0.8-0.9: {((df['Prob'] >= 0.8) & (df['Prob'] < 0.9)).sum():4d} objetos")
        print(f"  • ≥0.9: {(df['Prob'] >= 0.9).sum():4d} objetos")
    
    # 2. Distribución por campo
    if 'FIELD' in df.columns:
        print("\n2. DISTRIBUCIÓN POR CAMPO:")
        field_counts = df['FIELD'].value_counts().sort_index()
        for field, count in field_counts.items():
            print(f"  • {field}: {count:4d} objetos ({count/len(df)*100:.1f}%)")
    
    # 3. Distribución radial
    if 'Rgc' in df.columns:
        print("\n3. DISTRIBUCIÓN RADIAL (Rgc en arcmin):")
        print(f"  • Media: {df['Rgc'].mean():.1f} ± {df['Rgc'].std():.1f}")
        print(f"  • Mediana: {df['Rgc'].median():.1f}")
        print(f"  • Mín: {df['Rgc'].min():.1f}, Máx: {df['Rgc'].max():.1f}")
        
        # Bins radiales
        max_rgc = df['Rgc'].max()
        bins = [0, 5, 10, 15, 20, 30, 50, 100, max_rgc]
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        
        df_temp = df.copy()
        df_temp['Rgc_bin'] = pd.cut(df_temp['Rgc'], bins=bins, labels=labels, include_lowest=True)
        
        print("\n  📊 Objetos por bin radial:")
        for label in labels:
            count = (df_temp['Rgc_bin'] == label).sum()
            if count > 0:
                print(f"    • {label:8s}: {count:4d} objetos ({count/len(df)*100:.1f}%)")
    
    # 4. Verificar extinción
    if 'E_BV_SFD' in df.columns and 'A_V' in df.columns:
        print("\n4. DISTRIBUCIÓN DE EXTINCIÓN:")
        print(f"  • E(B-V): {df['E_BV_SFD'].mean():.3f} ± {df['E_BV_SFD'].std():.3f}")
        print(f"  • A_V: {df['A_V'].mean():.3f} ± {df['A_V'].std():.3f}")
        print(f"  • Rango E(B-V): [{df['E_BV_SFD'].min():.3f}, {df['E_BV_SFD'].max():.3f}]")

# ============================================================================
# 2. VALIDACIÓN FOTOMÉTRICA
# ============================================================================

def photometric_validation(df):
    """
    Realiza validación cruzada usando diferentes combinaciones de filtros.
    """
    print("\n" + "="*80)
    print("🌈 VALIDACIÓN FOTOMÉTRICA CRUZADA")
    print("="*80)
    
    if 'FeH' not in df.columns:
        print("❌ No hay columna [Fe/H] para correlacionar.")
        return {}
    
    # Definir índices de color clave
    color_indices = {
        '(g-i)_DECam': ('gmag_corr', 'imag_corr'),
        '(g-r)_DECam': ('gmag_corr', 'rmag_corr'),
        '(r-i)_DECam': ('rmag_corr', 'imag_corr'),
        '(F395-F430)_JPLUS': ('MAG_F395_3_corr', 'MAG_F430_3_corr'),
        '(F430-F515)_JPLUS': ('MAG_F430_3_corr', 'MAG_F515_3_corr'),
        '(F515-F660)_JPLUS': ('MAG_F515_3_corr', 'MAG_F660_3_corr'),
    }
    
    results = {}
    
    print("\n🔗 CORRELACIONES ENTRE COLORES Y [Fe/H]:")
    
    for color_name, (mag1, mag2) in color_indices.items():
        if mag1 in df.columns and mag2 in df.columns:
            # Calcular color
            color = df[mag1] - df[mag2]
            
            # Filtrar valores extremos
            q1, q3 = color.quantile(0.01), color.quantile(0.99)
            valid_idx = (color >= q1) & (color <= q3) & df['FeH'].notna()
            
            if valid_idx.sum() > 10:  # Necesitamos suficientes puntos
                color_valid = color[valid_idx]
                feh_valid = df.loc[valid_idx, 'FeH']
                
                # Calcular correlaciones
                pearson_r, pearson_p = stats.pearsonr(color_valid, feh_valid)
                spearman_r, spearman_p = stats.spearmanr(color_valid, feh_valid)
                
                print(f"\n  • {color_name}:")
                print(f"    Pearson:  r = {pearson_r:.3f}, p = {pearson_p:.3e}")
                print(f"    Spearman: ρ = {spearman_r:.3f}, p = {spearman_p:.3e}")
                print(f"    N puntos: {valid_idx.sum()}")
                
                results[color_name] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_points': valid_idx.sum()
                }
            else:
                print(f"  • {color_name}: Insuficientes datos válidos")
        else:
            print(f"  • {color_name}: Columnas no disponibles")
    
    return results

def estimate_FeH_from_colors(df):
    """
    Estima [Fe/H] usando diferentes relaciones empíricas de color.
    """
    print("\n" + "="*80)
    print("🎨 ESTIMACIÓN DE [Fe/H] DESDE COLORES")
    print("="*80)
    
    # 1. Desde (g-i) DECam (relación empírica común para GCs)
    if all(col in df.columns for col in ['gmag_corr', 'imag_corr']):
        g_i = df['gmag_corr'] - df['imag_corr']
        
        # Calibración aproximada para GCs viejos
        # Basado en relaciones de la literatura (Usher et al. 2019, etc.)
        FeH_g_i = 3.5 * g_i - 4.5
        
        df['FeH_g_i'] = FeH_g_i
        
        print("1. DESDE (g-i) DECam:")
        print(f"  • Media: {FeH_g_i.mean():.3f} ± {FeH_g_i.std():.3f} dex")
        print(f"  • Rango: [{FeH_g_i.min():.2f}, {FeH_g_i.max():.2f}] dex")
    
    # 2. Desde (F395-F430) J-PLUS (índice Ca H+K)
    if all(col in df.columns for col in ['MAG_F395_3_corr', 'MAG_F430_3_corr']):
        F395_F430 = df['MAG_F395_3_corr'] - df['MAG_F430_3_corr']
        
        # Calibración para índices Ca H+K
        FeH_CaHK = 4.0 * F395_F430 - 2.5
        
        df['FeH_CaHK'] = FeH_CaHK
        
        print("\n2. DESDE (F395-F430) J-PLUS (Ca H+K):")
        print(f"  • Media: {FeH_CaHK.mean():.3f} ± {FeH_CaHK.std():.3f} dex")
        print(f"  • Rango: [{FeH_CaHK.min():.2f}, {FeH_CaHK.max():.2f}] dex")
    
    return df

def check_extinction_effects(df):
    """
    Verifica si la corrección por extinción es adecuada.
    """
    print("\n" + "="*80)
    print("🌫️  ANÁLISIS DE EFECTOS DE EXTINCIÓN")
    print("="*80)
    
    if 'E_BV_SFD' in df.columns and 'FeH' in df.columns:
        # 1. Correlación básica
        corr = df['FeH'].corr(df['E_BV_SFD'])
        
        print(f"1. CORRELACIÓN [Fe/H] vs E(B-V):")
        print(f"  • Pearson r = {corr:.3f}")
        
        if abs(corr) > 0.3:
            print(f"  ⚠️  Correlación moderada/fuerte detectada")
        elif abs(corr) > 0.1:
            print(f"  ⚠️  Correlación débil detectada")
        else:
            print(f"  ✅ Correlación despreciable")
        
        # 2. Verificar si objetos con alta extinción tienen [Fe/H] sistemáticamente diferente
        high_av_mask = df['E_BV_SFD'] > df['E_BV_SFD'].quantile(0.75)
        low_av_mask = df['E_BV_SFD'] < df['E_BV_SFD'].quantile(0.25)
        
        if high_av_mask.sum() > 10 and low_av_mask.sum() > 10:
            feh_high_av = df.loc[high_av_mask, 'FeH'].mean()
            feh_low_av = df.loc[low_av_mask, 'FeH'].mean()
            
            print(f"\n2. COMPARACIÓN ALTA vs BAJA EXTINCIÓN:")
            print(f"  • Alta E(B-V) (>P75): {feh_high_av:.3f} dex, N={high_av_mask.sum()}")
            print(f"  • Baja E(B-V) (<P25): {feh_low_av:.3f} dex, N={low_av_mask.sum()}")
            print(f"  • Diferencia: {feh_high_av - feh_low_av:.3f} dex")
            
            # Test t para significancia
            t_stat, p_value = stats.ttest_ind(
                df.loc[high_av_mask, 'FeH'].dropna(),
                df.loc[low_av_mask, 'FeH'].dropna()
            )
            print(f"  • Test t: t = {t_stat:.3f}, p = {p_value:.3e}")
            
            if p_value < 0.05 and abs(feh_high_av - feh_low_av) > 0.1:
                print(f"  ⚠️  Diferencia SIGNIFICATIVA - posible efecto de extinción residual")
    
    # 3. Verificar correlaciones con A_V específicas por filtro
    if 'A_V' in df.columns and 'FeH' in df.columns:
        print(f"\n3. CORRELACIÓN [Fe/H] vs A_V:")
        print(f"  • r = {df['FeH'].corr(df['A_V']):.3f}")

# ============================================================================
# 3. VALIDACIÓN ESTADÍSTICA GMM
# ============================================================================

def gmm_robustness_test(feh_data, n_tests=20, subsample_fraction=0.8):
    """
    Prueba la robustez del GMM con diferentes submuestras.
    """
    print("\n" + "="*80)
    print("🎯 TEST DE ROBUSTEZ GMM CON SUBMUESTRAS")
    print("="*80)
    
    if len(feh_data) < 100:
        print(f"❌ Muy pocos datos ({len(feh_data)}) para test de robustez.")
        return None, None
    
    stability_counts = {2: 0, 3: 0, 4: 0, 5: 0}
    all_results = []
    
    print(f"• Datos totales: {len(feh_data)} objetos")
    print(f"• Réplicas: {n_tests}")
    print(f"• Tamaño submuestra: {int(subsample_fraction * len(feh_data))} objetos")
    
    for test in range(n_tests):
        # Tomar submuestra aleatoria
        sample_size = int(subsample_fraction * len(feh_data))
        sample_idx = np.random.choice(len(feh_data), sample_size, replace=False)
        sample = feh_data[sample_idx]
        
        # Probar diferentes números de componentes
        bic_values = []
        aic_values = []
        
        for n in [2, 3, 4, 5]:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    random_state=test * 1000,
                    n_init=3,
                    max_iter=200
                )
                gmm.fit(sample.reshape(-1, 1))
                bic_values.append(gmm.bic(sample.reshape(-1, 1)))
                aic_values.append(gmm.aic(sample.reshape(-1, 1)))
            except Exception as e:
                bic_values.append(np.inf)
                aic_values.append(np.inf)
        
        # Determinar mejor modelo por BIC
        best_n_bic = [2, 3, 4, 5][np.argmin(bic_values)]
        stability_counts[best_n_bic] += 1
        
        # Guardar resultados
        all_results.append({
            'test': test,
            'best_n_bic': best_n_bic,
            'bic_values': bic_values.copy(),
            'aic_values': aic_values.copy()
        })
        
        # Progreso
        if (test + 1) % 5 == 0:
            print(f"  • Réplica {test+1}/{n_tests} completada")
    
    # Resultados finales
    print(f"\n📊 RESULTADOS EN {n_tests} SUBMUESTRAS ({subsample_fraction*100:.0f}% de datos):")
    
    for n in [2, 3, 4, 5]:
        percentage = stability_counts[n] / n_tests * 100
        print(f"  • {n} componentes: {stability_counts[n]:3d} veces ({percentage:5.1f}%)")
    
    # Interpretación
    if stability_counts[4] >= 0.7 * n_tests:
        print(f"\n✅ El modelo de 4 componentes es ALTAMENTE ROBUSTO")
    elif stability_counts[4] >= 0.5 * n_tests:
        print(f"\n✅ El modelo de 4 componentes es MODERADAMENTE ROBUSTO")
    else:
        print(f"\n⚠️  El modelo de 4 componentes es INESTABLE")
    
    return stability_counts, all_results

def analyze_metal_rich_population(df, feh_threshold=-0.3):
    """
    Analiza específicamente la población rica en metales.
    """
    print("\n" + "="*80)
    print("💰 ANÁLISIS DE LA POBLACIÓN RICA EN METALES")
    print("="*80)
    
    if 'FeH' not in df.columns:
        print("❌ No hay datos de [Fe/H] para analizar.")
        return None
    
    # Identificar objetos ricos en metales
    rich_mask = df['FeH'] > feh_threshold
    n_rich = rich_mask.sum()
    
    print(f"• Umbral [Fe/H] > {feh_threshold:.1f} dex")
    print(f"• Objetos ricos en metales: {n_rich}/{len(df)} ({n_rich/len(df)*100:.1f}%)")
    
    if n_rich == 0:
        print("❌ No hay objetos suficientemente ricos en metales.")
        return None
    
    rich_df = df[rich_mask].copy()
    
    print(f"\n📊 PROPIEDADES DE LOS GCs RICOS EN METALES:")
    print(f"  • [Fe/H] medio: {rich_df['FeH'].mean():.3f} ± {rich_df['FeH'].std():.3f} dex")
    print(f"  • Rango [Fe/H]: [{rich_df['FeH'].min():.2f}, {rich_df['FeH'].max():.2f}] dex")
    
    # Análisis espacial
    if 'Rgc' in rich_df.columns:
        print(f"\n📍 DISTRIBUCIÓN ESPACIAL:")
        print(f"  • Rgc medio: {rich_df['Rgc'].mean():.1f} ± {rich_df['Rgc'].std():.1f} arcmin")
        print(f"  • Rango Rgc: [{rich_df['Rgc'].min():.1f}, {rich_df['Rgc'].max():.1f}] arcmin")
        
        # Comparar con población total
        if 'Rgc' in df.columns:
            all_rgc_mean = df['Rgc'].mean()
            print(f"  • Rgc medio (total): {all_rgc_mean:.1f} arcmin")
            print(f"  • Diferencia: {rich_df['Rgc'].mean() - all_rgc_mean:.1f} arcmin")
    
    # Análisis de probabilidad
    if 'Prob' in rich_df.columns:
        print(f"\n🎲 PROBABILIDAD GC:")
        print(f"  • Prob media: {rich_df['Prob'].mean():.3f}")
        print(f"  • <0.7: {(rich_df['Prob'] < 0.7).sum()} objetos")
        print(f"  • 0.7-0.9: {((rich_df['Prob'] >= 0.7) & (rich_df['Prob'] < 0.9)).sum()} objetos")
        print(f"  • ≥0.9: {(rich_df['Prob'] >= 0.9).sum()} objetos")
    
    # Análisis fotométrico
    if all(col in rich_df.columns for col in ['gmag_corr', 'e_gmag_corr']):
        print(f"\n📷 CALIDAD FOTOMÉTRICA (banda g):")
        print(f"  • Magnitud media: {rich_df['gmag_corr'].mean():.2f} mag")
        print(f"  • Error medio: {rich_df['e_gmag_corr'].mean():.3f} mag")
        print(f"  • SNR medio: {(1.0 / rich_df['e_gmag_corr']).mean():.1f}")
    
    # Verificar si son reales o artefactos
    print(f"\n🔍 VERIFICACIÓN DE AUTENTICIDAD:")
    
    # 1. Verificar si están concentrados en un campo específico
    if 'FIELD' in rich_df.columns:
        field_counts = rich_df['FIELD'].value_counts()
        if len(field_counts) == 1:
            print(f"  ⚠️  Todos en un solo campo: {field_counts.index[0]}")
        else:
            print(f"  ✅ Distribuidos en {len(field_counts)} campos")
    
    # 2. Verificar errores sistemáticos
    if all(col in rich_df.columns for col in ['e_gmag_corr', 'e_rmag_corr', 'e_imag_corr']):
        avg_error = rich_df[['e_gmag_corr', 'e_rmag_corr', 'e_imag_corr']].mean().mean()
        if avg_error > 0.2:
            print(f"  ⚠️  Errores fotométricos altos: {avg_error:.3f} mag")
        else:
            print(f"  ✅ Errores fotométricos aceptables: {avg_error:.3f} mag")
    
    return rich_df

def consistency_analysis(df):
    """
    Analiza consistencia entre diferentes estimaciones de metalicidad.
    """
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE CONSISTENCIA ENTRE MÉTODOS")
    print("="*80)
    
    # Buscar todas las columnas de [Fe/H]
    feh_columns = [col for col in df.columns if 'FeH' in col]
    
    if len(feh_columns) < 2:
        print("❌ Se necesitan al menos 2 estimaciones de [Fe/H] para comparar.")
        return None
    
    print(f"• Encontradas {len(feh_columns)} estimaciones de [Fe/H]:")
    for col in feh_columns:
        print(f"  • {col}: {df[col].notna().sum()} valores")
    
    results = {}
    
    # Comparar por pares
    for i in range(len(feh_columns)):
        for j in range(i+1, len(feh_columns)):
            col1, col2 = feh_columns[i], feh_columns[j]
            
            # Filtrar valores válidos
            valid_idx = df[[col1, col2]].dropna().index
            
            if len(valid_idx) > 10:
                data1 = df.loc[valid_idx, col1]
                data2 = df.loc[valid_idx, col2]
                
                # Calcular métricas
                pearson_r, pearson_p = stats.pearsonr(data1, data2)
                spearman_r, spearman_p = stats.spearmanr(data1, data2)
                
                # Diferencia media y RMS
                diff = data1 - data2
                mean_diff = diff.mean()
                std_diff = diff.std()
                rms_diff = np.sqrt(np.mean(diff**2))
                
                print(f"\n🔗 {col1} vs {col2}:")
                print(f"  • Pearson:  r = {pearson_r:.3f}, p = {pearson_p:.3e}")
                print(f"  • Spearman: ρ = {spearman_r:.3f}, p = {spearman_p:.3e}")
                print(f"  • N puntos: {len(valid_idx)}")
                print(f"  • Diferencia media: {mean_diff:.3f} ± {std_diff:.3f} dex")
                print(f"  • RMS diferencia: {rms_diff:.3f} dex")
                
                # Interpretación
                if abs(mean_diff) > 0.5:
                    print(f"  ⚠️  Diferencia sistemática GRANDE")
                elif abs(mean_diff) > 0.2:
                    print(f"  ⚠️  Diferencia sistemática moderada")
                else:
                    print(f"  ✅ Diferencia sistemática pequeña")
                
                if pearson_r < 0.7:
                    print(f"  ⚠️  Correlación DÉBIL entre métodos")
                else:
                    print(f"  ✅ Correlación FUERTE entre métodos")
                
                # Guardar resultados
                results[f"{col1}_vs_{col2}"] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'rms_diff': rms_diff,
                    'n_points': len(valid_idx)
                }
    
    return results

# ============================================================================
# 4. VISUALIZACIONES
# ============================================================================

def create_validation_figures(df, output_dir):
    """
    Crea figuras de validación.
    """
    print("\n" + "="*80)
    print("🎨 CREANDO FIGURAS DE VALIDACIÓN")
    print("="*80)
    
    if 'FeH' not in df.columns:
        print("❌ No hay datos de [Fe/H] para crear figuras.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figura 1: Distribución de [Fe/H] con diferentes estimaciones
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    axes1 = axes1.flatten()
    
    # Panel A: Histograma de [Fe/H] principal
    ax = axes1[0]
    ax.hist(df['FeH'].dropna(), bins=40, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.8, density=True)
    ax.set_xlabel('[Fe/H] (dex)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of [Fe/H] (CIGALE)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Comparación g-i vs [Fe/H]
    if all(col in df.columns for col in ['gmag_corr', 'imag_corr']):
        ax = axes1[1]
        g_i = df['gmag_corr'] - df['imag_corr']
        valid_idx = df['FeH'].notna() & g_i.notna()
        
        if valid_idx.sum() > 10:
            scatter = ax.scatter(g_i[valid_idx], df.loc[valid_idx, 'FeH'],
                                alpha=0.6, s=20, c=df.loc[valid_idx, 'Rgc'] if 'Rgc' in df.columns else 'blue',
                                cmap='viridis')
            ax.set_xlabel('(g-i) color (mag)', fontsize=12)
            ax.set_ylabel('[Fe/H] (dex)', fontsize=12)
            ax.set_title('(g-i) vs [Fe/H]', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            if 'Rgc' in df.columns:
                plt.colorbar(scatter, ax=ax, label='Rgc (arcmin)')
    
    # Panel C: Distribución radial por metalicidad
    if 'Rgc' in df.columns:
        ax = axes1[2]
        
        # Dividir en cuartiles de metalicidad
        try:
            feh_quartiles = pd.qcut(df['FeH'], 4, labels=['Q1 (más pobre)', 'Q2', 'Q3', 'Q4 (más rico)'])
            
            box_data = []
            labels = []
            for quartile in ['Q1 (más pobre)', 'Q2', 'Q3', 'Q4 (más rico)']:
                mask = feh_quartiles == quartile
                if mask.any():
                    box_data.append(df.loc[mask, 'Rgc'].values)
                    labels.append(quartile)
            
            if box_data:
                bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
                for i, box in enumerate(bp['boxes']):
                    box.set_facecolor(plt.cm.viridis(i/len(box_data)))
                    box.set_alpha(0.7)
                
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Projected Radius (arcmin)', fontsize=12)
                ax.set_title('Radial Distribution by [Fe/H] Quartile', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
        except:
            print("⚠️  No se pudo crear el gráfico de cuartiles")
    
    # Panel D: Probabilidad vs [Fe/H]
    if 'Prob' in df.columns:
        ax = axes1[3]
        valid_idx = df['FeH'].notna() & df['Prob'].notna()
        
        if valid_idx.sum() > 10:
            ax.scatter(df.loc[valid_idx, 'FeH'], df.loc[valid_idx, 'Prob'],
                      alpha=0.6, s=20, color='green')
            ax.set_xlabel('[Fe/H] (dex)', fontsize=12)
            ax.set_ylabel('GC Probability', fontsize=12)
            ax.set_title('GC Probability vs [Fe/H]', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
    
    fig1.suptitle('Photometric Validation of GC Metallicities', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig(f'{output_dir}/Figure_Photometric_Validation.pdf', dpi=600, bbox_inches='tight')
    print(f"✓ Figura 1 guardada: {output_dir}/Figure_Photometric_Validation.pdf")
    
    # Figura 2: Análisis de la población rica en metales
    if 'FeH' in df.columns:
        rich_mask = df['FeH'] > -0.3
        
        if rich_mask.sum() > 5:
            fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
            
            # Panel A: Comparación colores
            if all(col in df.columns for col in ['gmag_corr', 'imag_corr', 'rmag_corr']):
                ax = axes2[0, 0]
                
                g_i = df['gmag_corr'] - df['imag_corr']
                g_r = df['gmag_corr'] - df['rmag_corr']
                
                # Scatter con colores diferentes para ricos vs normales
                ax.scatter(g_i[~rich_mask], g_r[~rich_mask], alpha=0.3, s=15,
                          color='blue', label=f'[Fe/H] ≤ -0.3 (n={(~rich_mask).sum()})')
                ax.scatter(g_i[rich_mask], g_r[rich_mask], alpha=0.7, s=30,
                          color='red', label=f'[Fe/H] > -0.3 (n={rich_mask.sum()})')
                
                ax.set_xlabel('(g-i) (mag)', fontsize=12)
                ax.set_ylabel('(g-r) (mag)', fontsize=12)
                ax.set_title('Color-Color Diagram', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Panel B: Histograma comparativo
            ax = axes2[0, 1]
            ax.hist(df.loc[~rich_mask, 'FeH'].dropna(), bins=30, alpha=0.6,
                   color='blue', label='[Fe/H] ≤ -0.3', density=True)
            ax.hist(df.loc[rich_mask, 'FeH'].dropna(), bins=20, alpha=0.6,
                   color='red', label='[Fe/H] > -0.3', density=True)
            ax.axvline(x=-0.3, color='black', linestyle='--', linewidth=1.5)
            ax.set_xlabel('[Fe/H] (dex)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('Metallicity Distribution Comparison', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Panel C: Distribución radial
            if 'Rgc' in df.columns:
                ax = axes2[1, 0]
                
                # Box plot comparativo
                box_data = [df.loc[~rich_mask, 'Rgc'].dropna().values,
                           df.loc[rich_mask, 'Rgc'].dropna().values]
                
                bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                               labels=['[Fe/H] ≤ -0.3', '[Fe/H] > -0.3'])
                bp['boxes'][0].set_facecolor('blue')
                bp['boxes'][1].set_facecolor('red')
                
                ax.set_ylabel('Projected Radius (arcmin)', fontsize=12)
                ax.set_title('Radial Distribution Comparison', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Panel D: Probabilidad GC
            if 'Prob' in df.columns:
                ax = axes2[1, 1]
                
                # Box plot de probabilidad
                box_data = [df.loc[~rich_mask, 'Prob'].dropna().values,
                           df.loc[rich_mask, 'Prob'].dropna().values]
                
                bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                               labels=['[Fe/H] ≤ -0.3', '[Fe/H] > -0.3'])
                bp['boxes'][0].set_facecolor('blue')
                bp['boxes'][1].set_facecolor('red')
                
                ax.set_ylabel('GC Probability', fontsize=12)
                ax.set_title('GC Probability Comparison', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 1.05)
            
            fig2.suptitle('Analysis of Metal-Rich GC Population', fontsize=16, fontweight='bold')
            fig2.tight_layout()
            fig2.savefig(f'{output_dir}/Figure_Metal_Rich_Analysis.pdf', dpi=600, bbox_inches='tight')
            print(f"✓ Figura 2 guardada: {output_dir}/Figure_Metal_Rich_Analysis.pdf")
    
    plt.close('all')

# ============================================================================
# 5. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal de validación completa.
    """
    print("\n" + "="*80)
    print("VALIDACIÓN COMPLETA CON DATOS CIGALE [Fe/H] - NGC 5128")
    print("="*80)
    
    # Crear directorio de salida
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"GC_Validation_CIGALE_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Directorio de resultados: {output_dir}")
    
    # 1. Cargar catálogo fotométrico optimizado
    catalog_path = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
    
    print("\n[1] CARGANDO CATÁLOGO FOTOMÉTRICO OPTIMIZADO...")
    df_phot = load_catalog(catalog_path)
    
    if df_phot is None:
        print("❌ No se pudo cargar el catálogo fotométrico.")
        return
    
    # 2. Cargar resultados CIGALE
    print("\n[2] CARGANDO RESULTADOS CIGALE...")
    df_cigale = load_cigale_results('out/results.fits')
    
    if df_cigale is None:
        print("❌ No se pudieron cargar los resultados de CIGALE.")
        return
    
    # 3. Cruzar catálogos por recno
    print("\n[3] CRUZANDO CATÁLOGOS POR RECNO...")
    df_combined = match_catalogs_by_recno(df_phot, df_cigale)
    
    if df_combined is None or len(df_combined) == 0:
        print("❌ No se pudieron combinar los datos.")
        return
    
    print(f"✓ Datos combinados: {len(df_combined)} objetos")
    if 'FeH' in df_combined.columns:
        print(f"  • [Fe/H] medio: {df_combined['FeH'].mean():.3f} ± {df_combined['FeH'].std():.3f} dex")
    
    # 4. Análisis de calidad fotométrica
    print("\n[4] ANALIZANDO CALIDAD FOTOMÉTRICA...")
    check_photometric_quality(df_combined)
    
    # 5. Aplicar cortes de calidad (basados en tus criterios)
    print("\n[5] APLICANDO CORTES DE CALIDAD...")
    df_clean = apply_quality_cuts(df_combined, min_prob=0.5, min_good_filters=4)
    
    # 6. Verificar contaminación
    print("\n[6] VERIFICANDO CONTAMINACIÓN...")
    check_contamination(df_clean)
    
    # 7. Análisis de efectos de extinción
    print("\n[7] ANALIZANDO EFECTOS DE EXTINCIÓN...")
    check_extinction_effects(df_clean)
    
    # 8. Validación fotométrica cruzada
    print("\n[8] REALIZANDO VALIDACIÓN FOTOMÉTRICA CRUZADA...")
    color_correlations = photometric_validation(df_clean)
    
    # 9. Estimación alternativa de [Fe/H]
    print("\n[9] ESTIMANDO [Fe/H] DESDE COLORES...")
    df_clean = estimate_FeH_from_colors(df_clean)
    
    # 10. Análisis de consistencia
    print("\n[10] ANALIZANDO CONSISTENCIA ENTRE MÉTODOS...")
    consistency_results = consistency_analysis(df_clean)
    
    # 11. Test de robustez GMM
    print("\n[11] REALIZANDO TEST DE ROBUSTEZ GMM...")
    if 'FeH' in df_clean.columns:
        feh_data = df_clean['FeH'].dropna().values
        
        if len(feh_data) > 100:
            stability_counts, gmm_results = gmm_robustness_test(feh_data, n_tests=20)
        else:
            print(f"⚠️  Muy pocos datos ({len(feh_data)}) para test GMM robusto")
            stability_counts = None
    else:
        print("❌ No hay datos de [Fe/H] para test GMM")
        stability_counts = None
    
    # 12. Análisis específico de población rica
    print("\n[12] ANALIZANDO POBLACIÓN RICA EN METALES...")
    rich_df = analyze_metal_rich_population(df_clean, feh_threshold=-0.3)
    
    # 13. Crear figuras
    print("\n[13] CREANDO FIGURAS...")
    create_validation_figures(df_clean, output_dir)
    
    # 14. Guardar resultados
    print("\n[14] GUARDANDO RESULTADOS...")
    
    # Guardar datos limpios
    output_path = f"{output_dir}/gcs_cigale_validated_data.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"✓ Datos validados guardados: {output_path}")
    
    # Guardar resumen
    summary_path = f"{output_dir}/cigale_validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VALIDATION SUMMARY - NGC 5128 GLOBULAR CLUSTERS (CIGALE)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis date: {datetime.datetime.now()}\n")
        f.write(f"Photometric catalog: {catalog_path}\n")
        f.write(f"Objects after quality cuts: {len(df_clean)}\n")
        
        if 'FeH' in df_clean.columns:
            f.write(f"\n[Fe/H] STATISTICS:\n")
            f.write(f"  • Mean: {df_clean['FeH'].mean():.3f} ± {df_clean['FeH'].std():.3f} dex\n")
            f.write(f"  • Range: [{df_clean['FeH'].min():.2f}, {df_clean['FeH'].max():.2f}] dex\n")
            f.write(f"  • Objects with [Fe/H] > -0.3: {(df_clean['FeH'] > -0.3).sum()}\n")
        
        if stability_counts:
            f.write(f"\nGMM ROBUSTNESS TEST (20 subsamples):\n")
            for n in [2, 3, 4, 5]:
                percentage = stability_counts[n] / 20 * 100
                f.write(f"  • {n} components: {stability_counts[n]} times ({percentage:.1f}%)\n")
            
            if stability_counts[4] >= 14:
                f.write(f"\nCONCLUSION: 4-component model is HIGHLY ROBUST\n")
            elif stability_counts[4] >= 10:
                f.write(f"\nCONCLUSION: 4-component model is MODERATELY ROBUST\n")
            else:
                f.write(f"\nCONCLUSION: 4-component model is WEAK\n")
        
        f.write(f"\nFILES GENERATED:\n")
        f.write(f"  1. gcs_cigale_validated_data.csv - Cleaned and validated catalog\n")
        f.write(f"  2. Figure_Photometric_Validation.pdf - Main validation figure\n")
        f.write(f"  3. Figure_Metal_Rich_Analysis.pdf - Metal-rich population analysis\n")
        f.write(f"  4. cigale_validation_summary.txt - This summary file\n")
    
    print(f"✓ Resumen guardado: {summary_path}")
    
    # 15. Resumen final
    print("\n" + "="*80)
    print("🎯 RESUMEN EJECUTIVO DE VALIDACIÓN CIGALE")
    print("="*80)
    
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"  • Objetos iniciales (fotometría): {len(df_phot)}")
    print(f"  • Objetos después de cortes: {len(df_clean)}")
    
    if 'FeH' in df_clean.columns:
        print(f"  • [Fe/H] medio: {df_clean['FeH'].mean():.3f} ± {df_clean['FeH'].std():.3f} dex")
        n_rich = (df_clean['FeH'] > -0.3).sum()
        print(f"  • Objetos con [Fe/H] > -0.3: {n_rich} ({n_rich/len(df_clean)*100:.1f}%)")
    
    if stability_counts:
        print(f"\n🎯 ROBUSTEZ DEL MODELO GMM:")
        print(f"  • 4 componentes seleccionadas en: {stability_counts[4]/20*100:.1f}% de las réplicas")
        
        if stability_counts[4] >= 14:
            print(f"  ✅ CONCLUSIÓN: Modelo ALTAMENTE ROBUSTO")
        elif stability_counts[4] >= 10:
            print(f"  ✅ CONCLUSIÓN: Modelo MODERADAMENTE ROBUSTO")
        else:
            print(f"  ⚠️  CONCLUSIÓN: Modelo DÉBIL")
    
    print(f"\n📁 ARCHIVOS GUARDADOS en '{output_dir}/':")
    print(f"   1. gcs_cigale_validated_data.csv - Catálogo validado")
    print(f"   2. Figure_Photometric_Validation.pdf - Figura principal")
    print(f"   3. Figure_Metal_Rich_Analysis.pdf - Análisis población rica")
    print(f"   4. cigale_validation_summary.txt - Resumen completo")
    
    print("\n" + "="*80)
    print("✅ VALIDACIÓN COMPLETA CIGALE FINALIZADA")
    print("="*80)

# ============================================================================
# EJECUCIÓN SIMPLIFICADA
# ============================================================================

def quick_validation():
    """
    Validación rápida usando archivo clasificado.
    """
    print("\n" + "="*80)
    print("VALIDACIÓN RÁPIDA CON ARCHIVO CLASIFICADO")
    print("="*80)
    
    # Crear directorio de salida
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"Quick_Validation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Directorio de resultados: {output_dir}")
    
    try:
        # 1. Cargar catálogo fotométrico
        catalog_path = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
        print(f"\n[1] Cargando catálogo fotométrico: {catalog_path}")
        df_phot = pd.read_csv(catalog_path)
        print(f"✓ {len(df_phot)} objetos cargados")
        
        # 2. Cargar resultados de CIGALE desde archivo clasificado
        print(f"\n[2] Buscando archivo clasificado más reciente...")
        import glob
        result_files = glob.glob("NGC5128_4pop_Analysis_*/ngc5128_gcs_4pop_classified.csv")
        
        if not result_files:
            print("❌ No se encontraron archivos clasificados.")
            return
        
        latest_file = sorted(result_files)[-1]
        print(f"✓ Cargando: {latest_file}")
        df_class = pd.read_csv(latest_file)
        
        # 3. Verificar que tenemos [Fe/H]
        if 'FeH' not in df_class.columns:
            print("❌ El archivo clasificado no tiene columna 'FeH'")
            return
        
        print(f"✓ {len(df_class)} objetos con [Fe/H] cargados")
        print(f"  • [Fe/H] medio: {df_class['FeH'].mean():.3f} ± {df_class['FeH'].std():.3f} dex")
        
        # 4. Buscar columnas ID
        if 'recno' not in df_phot.columns:
            print("❌ El catálogo fotométrico no tiene columna 'recno'")
            return
        
        if 'ID' not in df_class.columns:
            print("❌ El archivo clasificado no tiene columna 'ID'")
            return
        
        print(f"\n[3] Cruzando por recno/ID...")
        
        # Extraer número del ID en el archivo clasificado
        df_class['ID_NUM'] = df_class['ID'].apply(extract_id_number)
        df_phot['recno_num'] = pd.to_numeric(df_phot['recno'], errors='coerce')
        
        # Hacer el merge
        df_combined = pd.merge(
            df_phot[df_phot['recno_num'].notna()], 
            df_class[['ID_NUM', 'FeH']], 
            left_on='recno_num', 
            right_on='ID_NUM', 
            how='inner'
        )
        
        # Limpiar columnas temporales
        df_combined = df_combined.drop(['recno_num', 'ID_NUM'], axis=1)
        
        print(f"✓ {len(df_combined)} coincidencias encontradas")
        
        if len(df_combined) == 0:
            print("❌ No se encontraron coincidencias")
            return
        
        # 5. Aplicar cortes básicos
        print(f"\n[4] Aplicando cortes básicos...")
        if 'Prob' in df_combined.columns:
            df_clean = df_combined[df_combined['Prob'] >= 0.5].copy()
            print(f"✓ Después de Prob ≥ 0.5: {len(df_clean)} objetos")
        else:
            df_clean = df_combined.copy()
        
        # 6. Análisis GMM rápido
        print(f"\n[5] Realizando test GMM rápido...")
        feh_data = df_clean['FeH'].dropna().values
        
        if len(feh_data) > 100:
            stability_counts, _ = gmm_robustness_test(feh_data, n_tests=10, subsample_fraction=0.8)
        else:
            print(f"⚠️  Muy pocos datos ({len(feh_data)}) para test GMM")
            stability_counts = None
        
        # 7. Guardar resultados
        print(f"\n[6] Guardando resultados...")
        output_path = f"{output_dir}/quick_validated_data.csv"
        df_clean.to_csv(output_path, index=False)
        print(f"✓ Datos guardados: {output_path}")
        
        # 8. Crear figura básica
        print(f"\n[7] Creando figura básica...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_clean['FeH'].dropna(), bins=40, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.8, density=True)
        ax.set_xlabel('[Fe/H] (dex)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of [Fe/H] (Quick Validation)', fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.savefig(f"{output_dir}/quick_validation_histogram.pdf", dpi=600, bbox_inches='tight')
        
        # Resumen
        print(f"\n" + "="*80)
        print("🎯 RESUMEN RÁPIDO")
        print("="*80)
        print(f"• Objetos totales: {len(df_clean)}")
        print(f"• [Fe/H] medio: {df_clean['FeH'].mean():.3f} ± {df_clean['FeH'].std():.3f} dex")
        print(f"• Objetos con [Fe/H] > -0.3: {(df_clean['FeH'] > -0.3).sum()}")
        
        if stability_counts:
            print(f"• Estabilidad 4 componentes: {stability_counts[4]/10*100:.1f}%")
        
        print(f"\n📁 Resultados en: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    
    print("\nVALIDACIÓN COMPLETA CON DATOS CIGALE [Fe/H] PARA NGC 5128")
    print("\nOpciones:")
    print("  1. Validación completa (desde results.fits)")
    print("  2. Validación rápida (desde archivo clasificado)")
    
    try:
        choice = input("\nSelecciona opción (1 o 2): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            quick_validation()
        else:
            print("Opción no válida. Ejecutando validación rápida...")
            quick_validation()
            
    except KeyboardInterrupt:
        print("\n\nEjecución cancelada por el usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
