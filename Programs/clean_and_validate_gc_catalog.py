#!/usr/bin/env python3
"""
clean_and_validate_gc_catalog.py

Limpia y valida el catálogo de cúmulos globulares basado en la calidad fotométrica.
Aplica cortes en SNR y errores para producir un catálogo de alta confiabilidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Carga el catálogo y aplica criterios de limpieza"""
    catalog_file = 'Results/NGC5128_GC_photometry_science_ready.csv'
    df = pd.read_csv(catalog_file)
    print(f"✅ Cargado catálogo con {len(df)} cúmulos globulares")
    
    # Lista de filtros SPLUS
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    
    # Columnas de magnitud calibrada y error
    mag_columns = [f'MAG_{filt}_3' for filt in splus_filters]
    err_columns = [f'MAGERR_{filt}_3' for filt in splus_filters]
    snr_columns = [f'SNR_{filt}_3' for filt in splus_filters]
    
    # Verificar qué columnas existen
    available_mag = [col for col in mag_columns if col in df.columns]
    available_err = [col for col in err_columns if col in df.columns]
    available_snr = [col for col in snr_columns if col in df.columns]
    
    print(f"   Magnitudes disponibles: {len(available_mag)}/{len(splus_filters)}")
    print(f"   Errores disponibles: {len(available_err)}/{len(splus_filters)}")
    print(f"   SNR disponibles: {len(available_snr)}/{len(splus_filters)}")
    
    # Inicializar máscara de validez
    valid_mask = pd.Series(True, index=df.index)
    
    # Criterio 1: Excluir magnitudes fuera de rango razonable (10-30 mag)
    for mag_col in available_mag:
        valid_mask &= (df[mag_col] > 10) & (df[mag_col] < 30)
    
    # Criterio 2: Excluir errores muy grandes (> 1 mag)
    for err_col in available_err:
        valid_mask &= (df[err_col] < 1.0) & (df[err_col] > 0)
    
    # Criterio 3: Excluir SNR muy bajo (< 3)
    for snr_col in available_snr:
        if snr_col in df.columns:
            valid_mask &= (df[snr_col] > 3)
    
    # Criterio 4: Excluir valores NaN en magnitudes clave
    key_mags = ['MAG_F515_3', 'MAG_F660_3', 'MAG_F861_3']  # Filtros más importantes
    for mag_col in key_mags:
        if mag_col in df.columns:
            valid_mask &= df[mag_col].notna()
    
    print(f"   Cúmulos después de limpieza: {valid_mask.sum()}/{len(df)}")
    print(f"   Cúmulos removidos: {len(df) - valid_mask.sum()}")
    
    return df[valid_mask].copy(), df[~valid_mask].copy()

def analyze_cleaned_colors(clean_df):
    """Analiza los colores después de la limpieza"""
    print("\n🎨 ANÁLISIS DE COLORES LIMPIOS")
    print("="*70)
    
    # Calcular colores otra vez para el conjunto limpio
    clean_df['F515_F660_clean'] = clean_df['MAG_F515_3'] - clean_df['MAG_F660_3']
    clean_df['F660_F861_clean'] = clean_df['MAG_F660_3'] - clean_df['MAG_F861_3']
    clean_df['F378_F515_clean'] = clean_df['MAG_F378_3'] - clean_df['MAG_F515_3']
    
    # Crear figura comparativa
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogramas de colores limpios
    colors_to_plot = ['F515_F660_clean', 'F660_F861_clean', 'F378_F515_clean']
    titles = ['F515 - F660 (limpio)', 'F660 - F861 (limpio)', 'F378 - F515 (limpio)']
    
    for i, (color_col, title) in enumerate(zip(colors_to_plot, titles)):
        ax = axes[i//2, i%2]
        color_data = clean_df[color_col].dropna()
        
        if len(color_data) > 0:
            ax.hist(color_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.median(color_data), color='red', linestyle='--', 
                      label=f'Mediana: {np.median(color_data):.3f}')
            ax.set_xlabel(title)
            ax.set_ylabel('Número de GCs')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Diagrama color-color limpio
    valid_clean = (clean_df['F515_F660_clean'].notna() & 
                  clean_df['F660_F861_clean'].notna() & 
                  clean_df['g_i_taylor'].notna())
    
    if valid_clean.sum() > 0:
        sc = axes[1, 1].scatter(clean_df.loc[valid_clean, 'F515_F660_clean'], 
                               clean_df.loc[valid_clean, 'F660_F861_clean'],
                               c=clean_df.loc[valid_clean, 'g_i_taylor'], 
                               alpha=0.7, s=40, cmap='viridis')
        axes[1, 1].set_xlabel('F515 - F660 (limpio)')
        axes[1, 1].set_ylabel('F660 - F861 (limpio)')
        axes[1, 1].set_title('Diagrama Color-Color (limpio)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[1, 1], label='g-i (Taylor)')
    
    plt.tight_layout()
    plt.savefig('Figs/gc_colors_cleaned_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Estadísticas de colores limpios
    print("\n📊 ESTADÍSTICAS DE COLORES LIMPIOS:")
    colors_data = []
    for color_name, color_data in [('F515-F660', clean_df['F515_F660_clean']),
                                  ('F660-F861', clean_df['F660_F861_clean']),
                                  ('F378-F515', clean_df['F378_F515_clean'])]:
        valid_data = color_data.dropna()
        if len(valid_data) > 0:
            colors_data.append({
                'Color': color_name,
                'N': len(valid_data),
                'Mediana': np.median(valid_data),
                'Media': np.mean(valid_data),
                'Std': np.std(valid_data),
                'Mín': np.min(valid_data),
                'Máx': np.max(valid_data)
            })
    
    colors_df = pd.DataFrame(colors_data)
    print(colors_df.to_string(index=False))
    
    return clean_df

def create_high_quality_catalog(clean_df, removed_df):
    """Crea un catálogo de alta calidad y reporta estadísticas"""
    print("\n📁 CREANDO CATÁLOGO DE ALTA CALIDAD")
    print("="*70)
    
    # Seleccionar columnas para el catálogo de alta calidad
    base_columns = ['recno', 'T17ID', 'RAJ2000', 'DEJ2000', 'FIELD']
    
    # Columnas Taylor
    taylor_columns = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 
                     'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag']
    
    # Columnas SPLUS (magnitudes, errores, SNR)
    splus_mags = [col for col in clean_df.columns if col.startswith('MAG_') and '_3' in col]
    splus_errs = [col for col in clean_df.columns if col.startswith('MAGERR_') and '_3' in col]
    splus_snr = [col for col in clean_df.columns if col.startswith('SNR_') and '_3' in col]
    
    # Columnas de colores
    color_columns = [col for col in clean_df.columns if 'clean' in col or 'calibrated' in col]
    
    # Combinar todas las columnas
    all_columns = base_columns + taylor_columns + splus_mags + splus_errs + splus_snr + color_columns
    available_columns = [col for col in all_columns if col in clean_df.columns]
    
    # Crear catálogo de alta calidad
    hq_catalog = clean_df[available_columns].copy()
    
    # Guardar catálogos
    hq_output = 'Results/NGC5128_GC_photometry_high_quality.csv'
    removed_output = 'Results/NGC5128_GC_photometry_removed_sources.csv'
    
    hq_catalog.to_csv(hq_output, index=False)
    removed_df.to_csv(removed_output, index=False)
    
    print(f"✅ Catálogo de alta calidad guardado: {hq_output}")
    print(f"   Cúmulos en catálogo HQ: {len(hq_catalog)}")
    print(f"✅ Catálogo de fuentes removidas guardado: {removed_output}")
    print(f"   Fuentes removidas: {len(removed_df)}")
    
    # Reportar estadísticas de limpieza
    print(f"\n📊 ESTADÍSTICAS DE LIMPIEZA:")
    print(f"   Tasa de retención: {len(hq_catalog)/(len(hq_catalog) + len(removed_df)) * 100:.1f}%")
    
    if len(removed_df) > 0:
        print(f"\n🔍 RAZONES PRINCIPALES DE REMOCIÓN:")
        # Aquí podrías agregar análisis de por qué se removieron fuentes
        
    return hq_catalog

def generate_quality_report(clean_df, hq_catalog, removed_df):
    """Genera un reporte de calidad del catálogo"""
    print("\n📊 REPORTE DE CALIDAD DEL CATÁLOGO")
    print("="*70)
    
    # Estadísticas generales
    total_sources = len(clean_df) + len(removed_df)
    retention_rate = len(clean_df) / total_sources * 100
    
    print(f"📈 ESTADÍSTICAS GENERALES:")
    print(f"   Fuentes totales: {total_sources}")
    print(f"   Fuentes en catálogo HQ: {len(clean_df)}")
    print(f"   Fuentes removidas: {len(removed_df)}")
    print(f"   Tasa de retención: {retention_rate:.1f}%")
    
    # Calcular completitud por filtro
    print(f"\n📷 COMPLETITUD POR FILTRO:")
    filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    for filt in filters:
        mag_col = f'MAG_{filt}_3'
        if mag_col in hq_catalog.columns:
            n_valid = hq_catalog[mag_col].notna().sum()
            completeness = n_valid / len(hq_catalog) * 100
            print(f"   {filt}: {n_valid}/{len(hq_catalog)} ({completeness:.1f}%)")
    
    # Calcular profundidad aproximada (percentil 90 de magnitudes)
    print(f"\n🌌 PROFUNDIDAD APROXIMADA (percentil 90):")
    for filt in filters:
        mag_col = f'MAG_{filt}_3'
        if mag_col in hq_catalog.columns:
            valid_mags = hq_catalog[mag_col].dropna()
            if len(valid_mags) > 0:
                depth = np.percentile(valid_mags, 90)
                print(f"   {filt}: {depth:.2f} mag")
    
    # Estadísticas de errores fotométricos
    print(f"\n📏 ERRORES FOTOMÉTRICOS MEDIANOS:")
    for filt in filters:
        err_col = f'MAGERR_{filt}_3'
        if err_col in hq_catalog.columns:
            median_err = hq_catalog[err_col].median()
            print(f"   {filt}: {median_err:.3f} mag")
    
    # Rango de colores Taylor
    print(f"\n🎨 PROPIEDADES DE COLORES TAYLOR:")
    taylor_colors = {
        'u-g': ('umag', 'gmag'),
        'g-r': ('gmag', 'rmag'),
        'r-i': ('rmag', 'imag'),
        'i-z': ('imag', 'zmag')
    }
    
    for color_name, (mag1, mag2) in taylor_colors.items():
        if mag1 in hq_catalog.columns and mag2 in hq_catalog.columns:
            color_data = hq_catalog[mag1] - hq_catalog[mag2]
            if len(color_data.dropna()) > 0:
                median_color = color_data.median()
                std_color = color_data.std()
                print(f"   {color_name}: {median_color:.3f} ± {std_color:.3f}")

def main():
    """Función principal"""
    print("🚀 LIMPIEZA Y VALIDACIÓN DEL CATÁLOGO DE CÚMULOS GLOBULARES")
    print("="*70)
    
    # 1. Cargar y limpiar datos
    clean_df, removed_df = load_and_clean_data()
    
    if len(clean_df) == 0:
        print("❌ No hay datos válidos después de la limpieza. Revisa los criterios.")
        return
    
    # 2. Analizar colores limpios
    clean_df = analyze_cleaned_colors(clean_df)
    
    # 3. Crear catálogo de alta calidad
    hq_catalog = create_high_quality_catalog(clean_df, removed_df)
    
    # 4. Generar reporte de calidad
    generate_quality_report(clean_df, hq_catalog, removed_df)
    
    print("\n" + "="*70)
    print("🎉 LIMPIEZA COMPLETADA EXITOSAMENTE")
    print("="*70)
    print("📁 CATÁLOGOS GENERADOS:")
    print("   ✅ Results/NGC5128_GC_photometry_high_quality.csv - Catálogo de alta calidad")
    print("   ✅ Results/NGC5128_GC_photometry_removed_sources.csv - Fuentes removidas")
    print("   ✅ Results/gc_colors_cleaned_analysis.png - Análisis visual de colores limpios")
    print("\n🔍 RECOMENDACIONES:")
    print("   • Usa el catálogo de alta calidad para análisis científicos")
    print("   • Revisa las fuentes removidas para entender límites del survey")
    print("   • Los colores ahora deberían tener una dispersión más realista")

if __name__ == '__main__':
    main()
