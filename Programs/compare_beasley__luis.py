#!/usr/bin/env python3
# comparacion_cigale_beasley_robusta_bayesiano.py
# Versión corregida - No cambia de directorio prematuramente

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
import os
import datetime
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHI2_THRESHOLD = 15.0  # Umbral para chi2 reducido (ajustes malos)
REDSHIFT_NGC5128 = 0.001825
Z_SOLAR = 0.02  # Asplund et al. 2009

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def cargar_datos_cigale_bayesiano(ruta_fits='out/results.fits', chi2_max=10.0):
    """
    Carga resultados de CIGALE, PRIORIZANDO ESTIMACIONES BAYESIANAS.
    """
    print("📊 CARGANDO RESULTADOS BAYESIANOS DE CIGALE")
    print("=" * 60)
    
    try:
        # Leer archivo FITS
        table = Table.read(ruta_fits, hdu=1)
        df = table.to_pandas()
        
        print(f"✅ Datos CIGALE cargados: {len(df)} objetos")
        
        # Verificar columnas
        print(f"\n🔍 VERIFICANDO COLUMNAS DISPONIBLES:")
        relevant_cols = [c for c in df.columns if any(x in str(c).lower() 
                       for x in ['id', 'age', 'metal', 'z', 'chi', 'bayes', 'best'])]
        
        # BUSCAR COLUMNAS CON PRIORIDAD BAYESIANA
        col_id = None
        col_age = None
        col_metal = None
        col_chi2 = None
        
        # Estrategia de búsqueda
        for col in df.columns:
            col_lower = str(col).lower()
            
            if col_id is None and 'id' == col_lower:
                col_id = col
            
            if col_age is None and 'bayes.stellar.age_m_star' in col_lower:
                col_age = col
            elif col_age is None and 'best.stellar.age_m_star' in col_lower:
                col_age = col
            elif col_age is None and 'age' in col_lower and 'm_star' in col_lower:
                col_age = col
            elif col_age is None and 'age' in col_lower:
                col_age = col
            
            if col_metal is None and 'bayes.stellar.metallicity' in col_lower:
                col_metal = col
            elif col_metal is None and 'best.stellar.metallicity' in col_lower:
                col_metal = col
            elif col_metal is None and 'metallicity' in col_lower:
                col_metal = col
            elif col_metal is None and col_lower == 'z' and col != 'redshift':
                col_metal = col
            
            if col_chi2 is None and 'reduced_chi_square' in col_lower:
                col_chi2 = col
            elif col_chi2 is None and 'chi_square' in col_lower:
                col_chi2 = col
            elif col_chi2 is None and 'chi2' in col_lower:
                col_chi2 = col
        
        # Fallbacks
        if col_id is None:
            col_id = df.columns[0]
        
        if col_age is None:
            age_cols = [c for c in df.columns if 'age' in str(c).lower()]
            if age_cols:
                col_age = age_cols[0]
        
        if col_metal is None:
            metal_cols = [c for c in df.columns if 'metal' in str(c).lower() or 
                         (str(c).lower() == 'z' and c != 'redshift')]
            if metal_cols:
                col_metal = metal_cols[0]
        
        print(f"\n🎯 COLUMNAS IDENTIFICADAS:")
        print(f"   • ID: {col_id}")
        print(f"   • Edad: {col_age}" if col_age else "   • Edad: NO ENCONTRADA")
        print(f"   • Metalicidad: {col_metal}" if col_metal else "   • Metalicidad: NO ENCONTRADA")
        print(f"   • χ²: {col_chi2}" if col_chi2 else "   • χ²: NO ENCONTRADA")
        
        if col_age is None or col_metal is None:
            raise ValueError("No se encontraron columnas críticas de edad o metalicidad")
        
        # PREPARAR DATOS
        df_clean = pd.DataFrame()
        df_clean['ID'] = df[col_id].astype(str).str.strip()
        
        # Edad: convertir a Gyr
        age_values = pd.to_numeric(df[col_age], errors='coerce')
        if age_values.max() > 1000:  # Probablemente en Myr
            df_clean['age_Gyr'] = age_values / 1000.0
        else:
            df_clean['age_Gyr'] = age_values
        
        # Metalicidad: determinar si es Z o [Fe/H]
        metal_values = pd.to_numeric(df[col_metal], errors='coerce')
        median_metal = metal_values.median()
        
        if 0.0001 < median_metal < 0.05:  # Valores típicos de Z
            df_clean['FeH_CIGALE'] = np.log10(metal_values / Z_SOLAR)
        elif -2.5 < median_metal < 0.5:  # Valores típicos de [Fe/H]
            df_clean['FeH_CIGALE'] = metal_values
        else:
            df_clean['FeH_CIGALE'] = metal_values
        
        # Añadir χ² si existe
        if col_chi2:
            df_clean['chi2'] = pd.to_numeric(df[col_chi2], errors='coerce')
            if not df_clean['chi2'].isna().all():
                mask_good = df_clean['chi2'] <= chi2_max
                df_clean = df_clean[mask_good].copy()
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Error cargando datos CIGALE: {e}")
        raise

def cargar_coincidencias_beasley(archivo_coincidencias='coincidencias_beasley_phot.csv'):
    """
    Carga las coincidencias ya establecidas entre Beasley y fotometría.
    """
    print(f"\n📊 CARGANDO COINCIDENCIAS BEASLEY-FOTOMETRÍA")
    print("=" * 60)
    
    try:
        df = pd.read_csv(archivo_coincidencias)
        print(f"✅ Coincidencias cargadas: {len(df)} objetos")
        
        # Limpiar datos
        df['T17ID'] = df['T17ID'].astype(str).str.strip()
        df['MH_Beasley'] = pd.to_numeric(df['MH_Beasley'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"❌ Error cargando coincidencias: {e}")
        raise

def combinar_cigale_beasley(cigale_df, beasley_df):
    """
    Combina datos de CIGALE con Beasley usando T17ID como clave.
    """
    print(f"\n🔗 COMBINANDO CIGALE CON BEASLEY")
    print("=" * 60)
    
    # Preparar copias
    cigale_data = cigale_df.copy()
    beasley_data = beasley_df.copy()
    
    # Crear columna de ID limpio en ambos dataframes
    cigale_data['ID_clean'] = cigale_data['ID'].astype(str).str.replace('T17-', '', regex=False).str.strip()
    beasley_data['ID_clean'] = beasley_data['T17ID'].astype(str).str.replace('T17-', '', regex=False).str.strip()
    
    # Intentar merge por ID limpio
    merged = pd.merge(
        cigale_data,
        beasley_data[['ID_clean', 'MH_Beasley', 'Beasley_ID']],
        on='ID_clean',
        how='inner'
    )
    
    print(f"✅ Coincidencias encontradas: {len(merged)}")
    
    if len(merged) > 0:
        # Calcular diferencias
        merged['FeH_diff'] = merged['FeH_CIGALE'] - merged['MH_Beasley']
        merged['abs_diff'] = np.abs(merged['FeH_diff'])
        
        return merged
    
    return None

def analizar_comparacion_completa(comp_df):
    """
    Análisis estadístico completo de la comparación.
    """
    print(f"\n📊 ANÁLISIS ESTADÍSTICO COMPLETO")
    print("=" * 70)
    
    n_total = len(comp_df)
    print(f"📈 MUESTRA FINAL: {n_total} objetos comparados")
    
    # Estadísticas básicas
    print(f"\n📊 ESTADÍSTICAS DE DIFERENCIA Δ[Fe/H]:")
    
    stats = {
        'Media': comp_df['FeH_diff'].mean(),
        'Mediana': comp_df['FeH_diff'].median(),
        'Desviación estándar': comp_df['FeH_diff'].std(),
        'MAE (Error absoluto medio)': comp_df['abs_diff'].mean(),
        'RMS': np.sqrt((comp_df['FeH_diff']**2).mean()),
    }
    
    for key, value in stats.items():
        print(f"   {key:25s}: {value:+.3f} dex")
    
    # Clasificación por calidad
    print(f"\n🎯 CLASIFICACIÓN POR CALIDAD DE ACUERDO:")
    
    categorias = [
        (0.15, "Excelente"),
        (0.30, "Buena"),
        (0.50, "Moderada"),
        (float('inf'), "Pobre")
    ]
    
    for i, (limite, categoria) in enumerate(categorias):
        if i == 0:
            mask = comp_df['abs_diff'] <= limite
        else:
            prev_limite = categorias[i-1][0]
            mask = (comp_df['abs_diff'] > prev_limite) & (comp_df['abs_diff'] <= limite)
        
        count = mask.sum()
        porcentaje = count / n_total * 100
        print(f"   {categoria:10s} (Δ ≤ {limite if limite != float('inf') else '∞'} dex): {count:3d} objetos ({porcentaje:5.1f}%)")
    
    return comp_df

def crear_graficos_profesionales(comp_df, output_dir='.'):
    """
    Crea gráficos profesionales.
    """
    print(f"\n🎨 CREANDO GRÁFICOS PROFESIONALES")
    print("=" * 60)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Comparación 1:1
    ax1 = axes[0, 0]
    
    colors = []
    for _, row in comp_df.iterrows():
        diff = row['abs_diff']
        if diff <= 0.15:
            colors.append('#27ae60')
        elif diff <= 0.30:
            colors.append('#f39c12')
        elif diff <= 0.50:
            colors.append('#e67e22')
        else:
            colors.append('#c0392b')
    
    for idx, (_, row) in enumerate(comp_df.iterrows()):
        ax1.scatter(row['MH_Beasley'], row['FeH_CIGALE'], 
                   color=colors[idx], alpha=0.7, edgecolor='black', s=50)
    
    # Línea de identidad
    min_val = min(comp_df['MH_Beasley'].min(), comp_df['FeH_CIGALE'].min()) - 0.5
    max_val = max(comp_df['MH_Beasley'].max(), comp_df['FeH_CIGALE'].max()) + 0.5
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax1.set_xlabel('[M/H] Beasley [dex]')
    ax1.set_ylabel('[Fe/H] CIGALE [dex]')
    ax1.set_title('Comparación directa')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma diferencias
    ax2 = axes[0, 1]
    
    ax2.hist(comp_df['FeH_diff'], bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=comp_df['FeH_diff'].mean(), color='red', linestyle='--')
    
    ax2.set_xlabel('Δ[Fe/H] [dex]')
    ax2.set_ylabel('Número')
    ax2.set_title('Distribución de discrepancias')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Diferencia vs Edad
    ax3 = axes[1, 0]
    
    scatter = ax3.scatter(comp_df['age_Gyr'], comp_df['FeH_diff'],
                         c=comp_df['MH_Beasley'], cmap='coolwarm',
                         s=50, alpha=0.7, edgecolor='black')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('Edad [Gyr]')
    ax3.set_ylabel('Δ[Fe/H] [dex]')
    ax3.set_title('Discrepancia vs Edad')
    ax3.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax3, label='[M/H] Beasley')
    
    # Gráfico 4: Resumen de métricas
    ax4 = axes[1, 1]
    
    metrics = ['MAE', 'RMS', 'Bias', '% ≤0.15']
    values = [
        comp_df['abs_diff'].mean(),
        np.sqrt((comp_df['FeH_diff']**2).mean()),
        comp_df['FeH_diff'].mean(),
        (comp_df['abs_diff'] <= 0.15).sum() / len(comp_df) * 100
    ]
    
    x_pos = np.arange(len(metrics))
    bars = ax4.bar(x_pos, values[:3] + [values[3]/100], alpha=0.7)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel('Valor')
    ax4.set_title('Métricas de comparación')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i < 3:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center')
        else:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.0f}%', ha='center')
    
    plt.suptitle(f'Comparación CIGALE vs Beasley - {len(comp_df)} objetos', fontsize=14)
    plt.tight_layout()
    
    # Guardar gráficos
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_png = os.path.join(output_dir, f'comparacion_cigale_beasley_{timestamp}.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comparacion_cigale_beasley_latest.png'), 
                dpi=150, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n💾 Gráficos guardados en: {output_dir}")
    
    return output_png

def generar_reporte_tabular(comp_df, output_dir='.'):
    """
    Genera un reporte tabular.
    """
    print(f"\n📝 GENERANDO REPORTE TABULAR")
    print("=" * 60)
    
    # Crear DataFrame de reporte
    report_df = comp_df.copy()
    
    # Ordenar por discrepancia
    report_df = report_df.sort_values('abs_diff', ascending=True)
    
    # Guardar
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f'reporte_comparacion_{timestamp}.csv')
    
    report_df.to_csv(csv_file, index=False)
    report_df.to_csv(os.path.join(output_dir, 'reporte_comparacion_latest.csv'), 
                     index=False)
    
    print(f"💾 Reportes guardados en: {output_dir}")
    
    return csv_file

def main():
    """Función principal del análisis."""
    print("=" * 80)
    print("🔍 COMPARACIÓN ROBUSTA CIGALE (BAYESIANO) vs BEASLEY (2008)")
    print("=" * 80)
    print("Análisis completo con estimaciones Bayesianas de CIGALE")
    print("=" * 80)
    
    # Verificar archivos en el directorio ACTUAL
    current_dir = os.getcwd()
    print(f"\n📁 DIRECTORIO ACTUAL: {current_dir}")
    print("\n📁 VERIFICANDO ARCHIVOS NECESARIOS:")
    
    # Rutas relativas al directorio actual
    ruta_fits = 'out/results.fits'
    ruta_coincidencias = 'coincidencias_beasley_phot.csv'
    
    # Verificar si existen
    if not os.path.exists(ruta_fits):
        print(f"   ❌ Archivo no encontrado: {ruta_fits}")
        print(f"   Buscando alternativas...")
        
        # Buscar alternativas
        posibles_fits = [
            'results.fits',
            '../out/results.fits',
            './out/results.fits',
            'out.fits'
        ]
        
        for posible in posibles_fits:
            if os.path.exists(posible):
                ruta_fits = posible
                print(f"   ✅ Encontrado: {posible}")
                break
        else:
            print(f"   ❌ No se encontró ningún archivo FITS de resultados")
            return False
    else:
        print(f"   ✅ {ruta_fits}")
    
    if not os.path.exists(ruta_coincidencias):
        print(f"   ❌ Archivo no encontrado: {ruta_coincidencias}")
        print(f"   Buscando alternativas...")
        
        posibles_csv = [
            '../coincidencias_beasley_phot.csv',
            './coincidencias_beasley_phot.csv',
            'coincidencias.csv'
        ]
        
        for posible in posibles_csv:
            if os.path.exists(posible):
                ruta_coincidencias = posible
                print(f"   ✅ Encontrado: {posible}")
                break
        else:
            print(f"   ❌ No se encontró el archivo de coincidencias")
            return False
    else:
        print(f"   ✅ {ruta_coincidencias}")
    
    try:
        # Crear directorio para resultados (en el directorio actual)
        results_dir = 'comparacion_resultados'
        os.makedirs(results_dir, exist_ok=True)
        print(f"\n📁 Directorio de resultados: {results_dir}")
        
        # 1. Cargar datos de CIGALE
        print(f"\n" + "="*80)
        print("PASO 1: CARGANDO DATOS CIGALE")
        print("="*80)
        cigale_df = cargar_datos_cigale_bayesiano(ruta_fits, chi2_max=CHI2_THRESHOLD)
        
        # 2. Cargar coincidencias Beasley
        print(f"\n" + "="*80)
        print("PASO 2: CARGANDO COINCIDENCIAS BEASLEY")
        print("="*80)
        beasley_df = cargar_coincidencias_beasley(ruta_coincidencias)
        
        # 3. Combinar datos
        print(f"\n" + "="*80)
        print("PASO 3: COMBINANDO DATOS")
        print("="*80)
        comp_df = combinar_cigale_beasley(cigale_df, beasley_df)
        
        if comp_df is None or len(comp_df) == 0:
            print("❌ No se pudieron combinar los datos")
            return False
        
        # 4. Análisis estadístico
        print(f"\n" + "="*80)
        print("PASO 4: ANÁLISIS ESTADÍSTICO")
        print("="*80)
        comp_df = analizar_comparacion_completa(comp_df)
        
        # 5. Guardar datos combinados
        output_csv = os.path.join(results_dir, 'comparacion_completa.csv')
        comp_df.to_csv(output_csv, index=False)
        print(f"\n💾 Datos combinados guardados: {output_csv}")
        
        # 6. Crear gráficos
        print(f"\n" + "="*80)
        print("PASO 5: CREANDO GRÁFICOS")
        print("="*80)
        plot_file = crear_graficos_profesionales(comp_df, results_dir)
        
        # 7. Generar reporte
        print(f"\n" + "="*80)
        print("PASO 6: GENERANDO REPORTE")
        print("="*80)
        report_file = generar_reporte_tabular(comp_df, results_dir)
        
        # Resumen final
        print(f"\n" + "="*80)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        mae = comp_df['abs_diff'].mean()
        bias = comp_df['FeH_diff'].mean()
        n_excelente = (comp_df['abs_diff'] <= 0.15).sum()
        
        print(f"\n📊 RESULTADOS PRINCIPALES:")
        print(f"   • Objetos analizados: {len(comp_df)}")
        print(f"   • Error absoluto medio (MAE): {mae:.3f} dex")
        print(f"   • Bias (diferencia media): {bias:+.3f} dex")
        print(f"   • Acuerdo excelente (Δ ≤ 0.15 dex): {n_excelente} objetos")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   1. {output_csv} - Datos completos")
        print(f"   2. {plot_file} - Gráficos")
        print(f"   3. {report_file} - Reporte detallado")
        print(f"   4. comparacion_resultados/comparacion_cigale_beasley_latest.png - Últimos gráficos")
        print(f"   5. comparacion_resultados/reporte_comparacion_latest.csv - Último reporte")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n✨ ¡Análisis completado!")
        else:
            print(f"\n❌ Hubo problemas en el análisis.")
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Interrumpido por el usuario.")
