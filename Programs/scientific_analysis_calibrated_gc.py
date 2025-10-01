#!/usr/bin/env python3
"""
scientific_analysis_calibrated_gc.py

Análisis científico usando las magnitudes de cúmulos globulares calibradas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_calibrated_data():
    """Carga los datos calibrados"""
    calibrated_file = '../Notebooks/taylor_gc_simple_corrected.csv'
    df = pd.read_csv(calibrated_file)
    print(f"✅ Cargados {len(df)} cúmulos globulares calibrados")
    return df

def analyze_color_distributions(df):
    """
    Analiza distribuciones de color con magnitudes calibradas
    """
    print("\n🎨 ANÁLISIS DE DISTRIBUCIONES DE COLOR")
    print("="*70)
    
    # Calcular colores calibrados
    df['F515_F660_calibrated'] = df['MAG_F515_simple_corrected_3'] - df['MAG_F660_simple_corrected_3']
    df['F660_F861_calibrated'] = df['MAG_F660_simple_corrected_3'] - df['MAG_F861_simple_corrected_3']
    df['F378_F515_calibrated'] = df['MAG_F378_simple_corrected_3'] - df['MAG_F515_simple_corrected_3']
    
    # Colores Taylor para comparación
    df['g_i_taylor'] = df['gmag'] - df['imag']
    
    # Crear figura de análisis de colores
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Histograma de colores SPLUS
    axes[0,0].hist(df['F515_F660_calibrated'].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_xlabel('F515 - F660 (calibrado)')
    axes[0,0].set_ylabel('Número de GCs')
    axes[0,0].set_title('Distribución Color F515-F660')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].hist(df['F660_F861_calibrated'].dropna(), bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].set_xlabel('F660 - F861 (calibrado)')
    axes[0,1].set_ylabel('Número de GCs')
    axes[0,1].set_title('Distribución Color F660-F861')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].hist(df['F378_F515_calibrated'].dropna(), bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0,2].set_xlabel('F378 - F515 (calibrado)')
    axes[0,2].set_ylabel('Número de GCs')
    axes[0,2].set_title('Distribución Color F378-F515')
    axes[0,2].grid(True, alpha=0.3)
    
    # 2. Diagramas color-color
    valid_mask = (df['F515_F660_calibrated'].notna() & 
                 df['F660_F861_calibrated'].notna() & 
                 df['g_i_taylor'].notna())
    
    sc1 = axes[1,0].scatter(df.loc[valid_mask, 'F515_F660_calibrated'], 
                           df.loc[valid_mask, 'F660_F861_calibrated'],
                           c=df.loc[valid_mask, 'g_i_taylor'], alpha=0.6, s=30)
    axes[1,0].set_xlabel('F515 - F660 (calibrado)')
    axes[1,0].set_ylabel('F660 - F861 (calibrado)')
    axes[1,0].set_title('Diagrama Color-Color SPLUS')
    axes[1,0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[1,0], label='g-i (Taylor)')
    
    # 3. Comparación con colores Taylor
    axes[1,1].scatter(df.loc[valid_mask, 'g_i_taylor'], 
                     df.loc[valid_mask, 'F515_F660_calibrated'], 
                     alpha=0.6, s=30)
    axes[1,1].set_xlabel('g - i (Taylor)')
    axes[1,1].set_ylabel('F515 - F660 (calibrado)')
    axes[1,1].set_title('Relación SPLUS vs Taylor Colors')
    axes[1,1].grid(True, alpha=0.3)
    
    # 4. Distribución de g-i Taylor
    axes[1,2].hist(df['g_i_taylor'].dropna(), bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,2].set_xlabel('g - i (Taylor)')
    axes[1,2].set_ylabel('Número de GCs')
    axes[1,2].set_title('Distribución Color g-i (Taylor)')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figs/gc_color_analysis_calibrated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Estadísticas de colores
    print("\n📊 ESTADÍSTICAS DE COLORES CALIBRADOS:")
    colors_data = []
    for color_name, color_data in [('F515-F660', df['F515_F660_calibrated']),
                                  ('F660-F861', df['F660_F861_calibrated']),
                                  ('F378-F515', df['F378_F515_calibrated']),
                                  ('g-i Taylor', df['g_i_taylor'])]:
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

def analyze_photometric_quality(df):
    """
    Analiza la calidad fotométrica final después de calibración
    """
    print("\n📏 ANÁLISIS DE CALIDAD FOTOMÉTRICA FINAL")
    print("="*70)
    
    # Calcular residuos finales
    filter_pairs = [
        ('MAG_F378_simple_corrected_3', 'umag'),
        ('MAG_F395_simple_corrected_3', 'umag'),
        ('MAG_F410_simple_corrected_3', 'gmag'),
        ('MAG_F430_simple_corrected_3', 'gmag'),
        ('MAG_F515_simple_corrected_3', 'gmag'),
        ('MAG_F660_simple_corrected_3', 'rmag'),
        ('MAG_F861_simple_corrected_3', 'imag')
    ]
    
    quality_results = []
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (splus_col, taylor_col) in enumerate(filter_pairs):
        if i >= len(axes):
            break
            
        if splus_col not in df.columns or taylor_col not in df.columns:
            continue
        
        valid_mask = (
            df[splus_col].notna() & 
            df[taylor_col].notna() &
            np.isfinite(df[splus_col]) & 
            np.isfinite(df[taylor_col]) &
            (df[splus_col] < 90) & 
            (df[taylor_col] < 90)
        )
        
        if valid_mask.sum() < 10:
            continue
        
        residuals = df.loc[valid_mask, splus_col] - df.loc[valid_mask, taylor_col]
        
        # Estadísticas de calidad
        median_residual = np.median(residuals)
        mad_residual = np.median(np.abs(residuals - median_residual))
        std_residual = np.std(residuals)
        
        quality_results.append({
            'Filtro': splus_col.replace('_simple_corrected_3', ''),
            'Taylor_Filtro': taylor_col,
            'N': len(residuals),
            'Residuo_Mediano': median_residual,
            'MAD_Residual': mad_residual,
            'Std_Residual': std_residual,
            'Rango_Residual': f"{residuals.min():.3f} a {residuals.max():.3f}"
        })
        
        # Gráfico de residuos
        ax = axes[i]
        ax.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(median_residual, color='red', linestyle='--', 
                  label=f'Mediana: {median_residual:.3f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel(f'Residuo ({splus_col} - {taylor_col})')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{splus_col.replace("_simple_corrected_3", "")}\nMAD: {mad_residual:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribución de Residuos después de Calibración', fontsize=16)
    plt.tight_layout()
    plt.savefig('Figs/final_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Resumen de calidad
    quality_df = pd.DataFrame(quality_results)
    print("\n📋 CALIDAD FOTOMÉTRICA FINAL:")
    print("="*70)
    print(quality_df.to_string(index=False))
    
    return quality_df

def create_science_ready_catalog(df):
    """
    Crea un catálogo listo para análisis científico
    """
    print("\n📁 CREANDO CATÁLOGO CIENTÍFICO FINAL")
    print("="*70)
    
    # Seleccionar columnas relevantes para ciencia
    science_columns = [
        'recno', 'T17ID', 'RAJ2000', 'DEJ2000', 'FIELD',
        'umag', 'gmag', 'rmag', 'imag', 'zmag',  # Magnitudes Taylor
        'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag',  # Errores Taylor
    ]
    
    # Añadir magnitudes SPLUS calibradas
    splus_columns = [col for col in df.columns if 'simple_corrected' in col]
    
    # Añadir colores calculados
    color_columns = [col for col in df.columns if 'calibrated' in col and 'F' in col]
    
    # Combinar todas las columnas
    all_columns = science_columns + splus_columns + color_columns
    available_columns = [col for col in all_columns if col in df.columns]
    
    # Crear catálogo científico
    science_catalog = df[available_columns].copy()
    
    # Renombrar columnas para claridad
    rename_dict = {}
    for col in science_catalog.columns:
        if 'simple_corrected' in col:
            new_name = col.replace('_simple_corrected', '')
            rename_dict[col] = new_name
    
    science_catalog.rename(columns=rename_dict, inplace=True)
    
    # Guardar catálogo
    output_file = 'Results/NGC5128_GC_photometry_science_ready.csv'
    science_catalog.to_csv(output_file, index=False)
    
    print(f"✅ Catálogo científico guardado en: {output_file}")
    print(f"   Total de columnas: {len(science_catalog.columns)}")
    print(f"   Total de cúmulos: {len(science_catalog)}")
    print(f"   Columnas principales:")
    for col in science_catalog.columns[:10]:  # Mostrar primeras 10 columnas
        print(f"     - {col}")
    
    return science_catalog

def generate_final_report(quality_df):
    """
    Genera un reporte final del proyecto
    """
    print("\n📄 INFORME FINAL DEL PROYECTO")
    print("="*70)
    print("🎯 CALIBRACIÓN FOTOMÉTRICA DE CÚMULOS GLOBULARES EN NGC 5128")
    print("="*70)
    
    print("\n✅ LOGROS PRINCIPALES:")
    print("   1. Pipeline completo de fotometría de apertura en 7 filtros SPLUS")
    print("   2. Calibración usando estrellas de referencia Gaia DR3")
    print("   3. Corrección de offsets sistemáticos específicos para GCs")
    print("   4. Catálogo científico listo para análisis")
    
    print(f"\n📊 RESUMEN ESTADÍSTICO:")
    print(f"   • Cúmulos globulares procesados: {len(quality_df) if not quality_df.empty else 'N/A'}")
    print(f"   • Filtros SPLUS calibrados: 7 (F378, F395, F410, F430, F515, F660, F861)")
    print(f"   • Campos SPLUS utilizados: CenA01, CenA02")
    
    if not quality_df.empty:
        avg_mad = quality_df['MAD_Residual'].mean()
        avg_offset = quality_df['Residuo_Mediano'].abs().mean()
        
        print(f"\n🎯 PRECISIÓN FINAL:")
        print(f"   • MAD residual promedio: {avg_mad:.3f} mag")
        print(f"   • Offset residual promedio: {avg_offset:.3f} mag")
        print(f"   • Rango típico de residuos: ±{avg_mad*2:.3f} mag")
    
    print(f"\n🔧 HERRAMIENTAS DESARROLLADAS:")
    print(f"   1. Extracción de fotometría de estrellas de referencia")
    print(f"   2. Cálculo de zero points usando Gaia XP")
    print(f"   3. Fotometría de cúmulos globulares con corrección de fondo")
    print(f"   4. Análisis de calidad y validación")
    print(f"   5. Correcciones específicas para GCs")
    
    print(f"\n📈 PRÓXIMOS PASOS CIENTÍFICOS:")
    print(f"   1. Análisis de poblaciones de cúmulos globulares")
    print(f"   2. Estudios de metalicidad usando índices de color narrow-band")
    print(f"   3. Correlación con propiedades estructurales")
    print(f"   4. Comparación con otros sistemas de galaxias")
    
    print(f"\n💾 PRODUCTOS FINALES:")
    print(f"   • NGC5128_GC_photometry_science_ready.csv - Catálogo científico")
    print(f"   • gc_color_analysis_calibrated.png - Análisis de colores")
    print(f"   • final_residuals_distribution.png - Calidad fotométrica")
    print(f"   • gc_simple_corrections.csv - Ecuaciones de corrección")

def main():
    """Función principal"""
    print("🚀 ANÁLISIS CIENTÍFICO CON DATOS CALIBRADOS")
    print("="*70)
    
    # Cargar datos calibrados
    df = load_calibrated_data()
    
    # 1. Análisis de distribuciones de color
    analyze_color_distributions(df)
    
    # 2. Análisis de calidad fotométrica final
    quality_df = analyze_photometric_quality(df)
    
    # 3. Crear catálogo científico
    science_catalog = create_science_ready_catalog(df)
    
    # 4. Generar reporte final
    generate_final_report(quality_df)
    
    print("\n" + "="*70)
    print("🎉 PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("¡Las magnitudes de cúmulos globulares están listas para investigación!")
    print("\nPuedes comenzar tu análisis científico usando:")
    print("📊 NGC5128_GC_photometry_science_ready.csv")

if __name__ == '__main__':
    main()
