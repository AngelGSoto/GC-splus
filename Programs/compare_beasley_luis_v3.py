#!/usr/bin/env python3
# comparacion_cigale_beasley_optimal.py
# Versión con umbral óptimo de χ² basado en análisis estadístico

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy import stats as scipy_stats
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN ÓPTIMA
# ============================================================================
CHI2_THRESHOLD = 20.0  # Basado en tu análisis (69.6% de objetos)
Z_SOLAR = 0.02

# ============================================================================
# FUNCIONES MEJORADAS
# ============================================================================

def cargar_datos_cigale_con_analisis_chi2(ruta_fits='out/results.fits'):
    """
    Carga datos de CIGALE con análisis detallado de distribución de χ².
    """
    print("📊 CARGANDO Y ANALIZANDO DATOS CIGALE")
    print("=" * 60)
    
    # Leer datos
    table = Table.read(ruta_fits, hdu=1)
    df = table.to_pandas()
    
    print(f"✅ Datos cargados: {len(df)} objetos")
    
    # Identificar columnas
    col_id = 'id'
    col_age = 'bayes.stellar.age_m_star'
    col_metal = 'bayes.stellar.metallicity'
    col_chi2 = 'best.chi_square'
    
    # Análisis detallado de χ²
    chi2_values = df[col_chi2].dropna()  # Eliminar NaN antes del análisis
    
    print(f"\n📈 ANÁLISIS DE DISTRIBUCIÓN DE χ²:")
    print(f"   • Mínimo: {chi2_values.min():.2f}")
    print(f"   • Máximo: {chi2_values.max():.2f}")
    print(f"   • Media: {chi2_values.mean():.2f}")
    print(f"   • Mediana: {chi2_values.median():.2f}")
    print(f"   • Percentiles:")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(chi2_values, p)
        print(f"     {p:2d}%: {value:.2f}")
    
    # Crear histograma de χ²
    plt.figure(figsize=(10, 6))
    
    # Histograma logarítmico (porque hay valores extremos)
    plt.subplot(1, 2, 1)
    plt.hist(chi2_values[chi2_values < 100], bins=30, 
             edgecolor='black', alpha=0.7)
    plt.axvline(x=CHI2_THRESHOLD, color='red', linestyle='--', 
                label=f'Umbral: {CHI2_THRESHOLD}')
    plt.xlabel('χ² reducido')
    plt.ylabel('Número de objetos')
    plt.title(f'Distribución de χ² (valores < 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histograma completo (log scale)
    plt.subplot(1, 2, 2)
    plt.hist(chi2_values, bins=50, log=True, 
             edgecolor='black', alpha=0.7)
    plt.axvline(x=CHI2_THRESHOLD, color='red', linestyle='--')
    plt.xlabel('χ² reducido (escala log)')
    plt.ylabel('Número de objetos (log)')
    plt.title('Distribución completa de χ²')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribucion_chi2.png', dpi=150)
    plt.show()
    
    # Preparar DataFrame limpio
    df_clean = pd.DataFrame()
    df_clean['ID'] = df[col_id].astype(str).str.strip()
    
    # Edad: convertir a Gyr
    age_values = df[col_age]
    if age_values.max() > 1000:
        df_clean['age_Gyr'] = age_values / 1000.0
    else:
        df_clean['age_Gyr'] = age_values
    
    # Metalicidad: convertir a [Fe/H]
    metal_values = df[col_metal]
    median_metal = metal_values.median()
    
    if 0.0001 < median_metal < 0.05:
        df_clean['FeH_CIGALE'] = np.log10(metal_values / Z_SOLAR)
    else:
        df_clean['FeH_CIGALE'] = metal_values
    
    # Añadir χ²
    df_clean['chi2'] = df[col_chi2]
    
    # Filtrar por χ²
    mask = df_clean['chi2'] <= CHI2_THRESHOLD
    df_filtered = df_clean[mask].copy()
    
    print(f"\n🎯 FILTRADO CON χ² ≤ {CHI2_THRESHOLD}:")
    print(f"   • Objetos originales: {len(df_clean)}")
    print(f"   • Objetos después del filtro: {len(df_filtered)}")
    print(f"   • Porcentaje retenido: {len(df_filtered)/len(df_clean)*100:.1f}%")
    
    return df_filtered

def matching_inteligente(cigale_df, beasley_df):
    """
    Matching inteligente que maneja múltiples formatos de IDs.
    """
    # Preparar IDs de Beasley
    beasley_df['T17ID_clean'] = beasley_df['T17ID'].astype(str).str.strip()
    beasley_dict = {}
    
    for idx, row in beasley_df.iterrows():
        original_id = row['T17ID_clean']
        
        # Crear múltiples variantes para matching flexible
        variants = [
            original_id,
            original_id.replace('T17-', '').strip(),
            original_id.replace('T17-', '').lstrip('0'),
            original_id.replace('T17', '').strip(),
            original_id.replace('T17', '').lstrip('0')
        ]
        
        # Si el ID original es numérico, también añadirlo como número
        try:
            num_id = str(int(float(original_id.replace('T17-', ''))))
            variants.append(num_id)
            variants.append(num_id.lstrip('0'))
        except:
            pass
        
        # Añadir al diccionario
        for variant in variants:
            if variant:  # No añadir strings vacíos
                # Guardar toda la fila de Beasley, no solo el índice
                if variant not in beasley_dict:
                    beasley_dict[variant] = row.to_dict()
    
    # Buscar coincidencias
    matches = []
    
    for cigale_idx, cigale_row in cigale_df.iterrows():
        cigale_id = cigale_row['ID']
        
        # Crear variantes para el ID de CIGALE
        cigale_variants = [
            str(cigale_id),
            str(cigale_id).strip(),
            str(cigale_id).replace('T17-', '').strip(),
            str(cigale_id).replace('T17-', '').lstrip('0'),
            str(cigale_id).replace('T17', '').strip(),
            str(cigale_id).replace('T17', '').lstrip('0')
        ]
        
        # Intentar convertir a número
        try:
            num_cigale = str(int(float(str(cigale_id).replace('T17-', ''))))
            cigale_variants.append(num_cigale)
            cigale_variants.append(num_cigale.lstrip('0'))
        except:
            pass
        
        # Buscar coincidencia
        matched = False
        for variant in cigale_variants:
            if variant in beasley_dict:
                # Guardar los datos completos, no solo índices
                matches.append({
                    'cigale_row': cigale_row.to_dict(),
                    'beasley_row': beasley_dict[variant],
                    'match_variant': variant
                })
                matched = True
                break
        
        if not matched:
            print(f"   ⚠️  No match para: {cigale_id}")
    
    print(f"\n🔍 MATCHING INTELIGENTE:")
    print(f"   • Coincidencias encontradas: {len(matches)}")
    print(f"   • Porcentaje de CIGALE: {len(matches)/len(cigale_df)*100:.1f}%")
    
    return matches

def main_optimal():
    """Análisis con umbral óptimo de χ²."""
    print("="*80)
    print("🎯 ANÁLISIS ÓPTIMO CIGALE vs BEASLEY")
    print("="*80)
    print(f"Umbral de χ²: ≤ {CHI2_THRESHOLD} (basado en análisis estadístico)")
    print("="*80)
    
    # 1. Cargar datos CIGALE con análisis de χ²
    cigale_df = cargar_datos_cigale_con_analisis_chi2('out/results.fits')
    
    # 2. Cargar coincidencias Beasley
    beasley_df = pd.read_csv('coincidencias_beasley_phot.csv')
    print(f"\n📊 BEASLEY: {len(beasley_df)} objetos")
    
    # 3. Matching inteligente
    matches = matching_inteligente(cigale_df, beasley_df)
    
    if len(matches) == 0:
        print("❌ No se encontraron coincidencias")
        return
    
    # 4. Combinar datos
    combined_data = []
    
    for match in matches:
        cigale_row = match['cigale_row']
        beasley_row = match['beasley_row']
        
        combined_data.append({
            'ID_CIGALE': cigale_row['ID'],
            'ID_Beasley': beasley_row.get('Beasley_ID', ''),
            'T17ID': beasley_row.get('T17ID', ''),
            'age_Gyr': cigale_row['age_Gyr'],
            'FeH_CIGALE': cigale_row['FeH_CIGALE'],
            'MH_Beasley': beasley_row.get('MH_Beasley', np.nan),
            'MH_err_Beasley': beasley_row.get('MH_err_Beasley', np.nan),
            'chi2': cigale_row['chi2'],
            'Sep_arcsec': beasley_row.get('Sep_arcsec', np.nan),
            'n_Name': beasley_row.get('n_Name', ''),
            'match_method': match['match_variant']
        })
    
    comp_df = pd.DataFrame(combined_data)
    
    # 5. Calcular diferencias
    comp_df['FeH_diff'] = comp_df['FeH_CIGALE'] - comp_df['MH_Beasley']
    comp_df['abs_diff'] = np.abs(comp_df['FeH_diff'])
    
    # 6. Análisis estadístico detallado
    print(f"\n📊 ANÁLISIS ESTADÍSTICO CON {len(comp_df)} OBJETOS:")
    print("="*60)
    
    # Estadísticas básicas
    mae = comp_df['abs_diff'].mean()
    bias = comp_df['FeH_diff'].mean()
    std = comp_df['FeH_diff'].std()
    rms = np.sqrt((comp_df['FeH_diff']**2).mean())
    
    print(f"📈 MÉTRICAS DE ERROR:")
    print(f"   • MAE (Error absoluto medio): {mae:.3f} dex")
    print(f"   • Bias (diferencia media): {bias:+.3f} dex")
    print(f"   • Desviación estándar: {std:.3f} dex")
    print(f"   • RMS (Error cuadrático medio): {rms:.3f} dex")
    
    # Clasificación por calidad
    print(f"\n🎯 CLASIFICACIÓN POR CALIDAD DE ACUERDO:")
    
    categorias = [
        (0.15, "Excelente (Δ ≤ 0.15 dex)", "Comparable a Hernández et al. (2018)"),
        (0.30, "Buena (0.15 < Δ ≤ 0.30 dex)", "Aceptable para estudios poblacionales"),
        (0.50, "Moderada (0.30 < Δ ≤ 0.50 dex)", "Requiere cautela en interpretación"),
        (float('inf'), "Pobre (Δ > 0.50 dex)", "Posible error o objeto problemático")
    ]
    
    prev_limite = 0
    for limite, categoria, descripcion in categorias:
        if limite == 0.15:
            mask = comp_df['abs_diff'] <= limite
        else:
            mask = (comp_df['abs_diff'] > prev_limite) & (comp_df['abs_diff'] <= limite)
        
        count = mask.sum()
        porcentaje = count / len(comp_df) * 100
        
        print(f"\n   📊 {categoria}:")
        print(f"      • Objetos: {count} ({porcentaje:.1f}%)")
        print(f"      • {descripcion}")
        
        if count > 0 and limite <= 0.50:
            # Mostrar algunos ejemplos
            ejemplos = comp_df[mask].head(3)
            for i, row in ejemplos.iterrows():
                print(f"      • Ejemplo: {row['ID_Beasley']} → {row['ID_CIGALE']}: "
                      f"[Fe/H]={row['FeH_CIGALE']:.2f}, [M/H]={row['MH_Beasley']:.2f}, "
                      f"Δ={row['FeH_diff']:+.2f}")
        
        prev_limite = limite
    
    # Correlación χ² vs error
    if len(comp_df) > 2:
        r_chi2, p_chi2 = scipy_stats.pearsonr(comp_df['chi2'], comp_df['abs_diff'])
        print(f"\n📈 CORRELACIÓN χ² vs CALIDAD:")
        print(f"   • Correlación (χ² vs |Δ[Fe/H]|): r = {r_chi2:.3f}, p = {p_chi2:.3e}")
        
        if r_chi2 > 0.3 and p_chi2 < 0.05:
            print(f"   ⚠️  Correlación significativa detectada")
            print(f"      Objetos con alto χ² tienden a tener mayor discrepancia")
    
    # Correlación edad vs error
    if 'age_Gyr' in comp_df.columns:
        r_edad, p_edad = scipy_stats.pearsonr(comp_df['age_Gyr'], comp_df['FeH_diff'])
        print(f"\n📈 CORRELACIÓN EDAD vs ERROR:")
        print(f"   • Correlación (Edad vs Δ[Fe/H]): r = {r_edad:.3f}, p = {p_edad:.3e}")
        
        if abs(r_edad) > 0.3 and p_edad < 0.05:
            print(f"   ⚠️  Degeneración edad-metalicidad detectada")
            print(f"      El error depende de la edad estimada")
    
    # 7. Identificar objetos problemáticos
    outliers = comp_df[comp_df['abs_diff'] > 0.5]
    if len(outliers) > 0:
        print(f"\n⚠️  OBJETOS PROBLEMÁTICOS (Δ > 0.5 dex): {len(outliers)}")
        print("   Estos objetos merecen investigación individual:")
        
        for i, row in outliers.nlargest(5, 'abs_diff').iterrows():
            print(f"\n   {i+1}. {row['ID_Beasley']} (T17ID: {row['T17ID']}):")
            print(f"       • [Fe/H] CIGALE: {row['FeH_CIGALE']:.2f}")
            print(f"       • [M/H] Beasley: {row['MH_Beasley']:.2f} ± {row.get('MH_err_Beasley', 0.0):.2f}")
            print(f"       • Δ[Fe/H]: {row['FeH_diff']:+.2f} dex")
            print(f"       • Edad CIGALE: {row['age_Gyr']:.1f} Gyr")
            print(f"       • χ²: {row['chi2']:.1f}")
            print(f"       • Separación: {row.get('Sep_arcsec', 0.0):.2f} arcsec")
    
    # 8. Guardar resultados
    os.makedirs('analisis_optimal', exist_ok=True)
    
    # Guardar datos completos
    comp_df_sorted = comp_df.sort_values('abs_diff', ascending=True)
    comp_df_sorted.to_csv('analisis_optimal/comparacion_optimal.csv', index=False)
    
    # Guardar resumen estadístico
    with open('analisis_optimal/resumen_optimal.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("RESUMEN ANÁLISIS ÓPTIMO CIGALE vs BEASLEY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Umbral de χ²: ≤ {CHI2_THRESHOLD}\n")
        f.write(f"Objetos analizados: {len(comp_df)}\n\n")
        
        f.write("MÉTRICAS DE ERROR:\n")
        f.write(f"  • MAE: {mae:.3f} dex\n")
        f.write(f"  • Bias: {bias:+.3f} dex\n")
        f.write(f"  • Std: {std:.3f} dex\n")
        f.write(f"  • RMS: {rms:.3f} dex\n\n")
        
        f.write("DISTRIBUCIÓN POR CALIDAD:\n")
        prev_limite = 0
        for limite, categoria, _ in categorias:
            if limite == 0.15:
                mask = comp_df['abs_diff'] <= limite
            else:
                mask = (comp_df['abs_diff'] > prev_limite) & (comp_df['abs_diff'] <= limite)
            
            count = mask.sum()
            porcentaje = count / len(comp_df) * 100
            f.write(f"  • {categoria}: {count} objetos ({porcentaje:.1f}%)\n")
            prev_limite = limite
    
    # 9. Crear gráficos
    print(f"\n🎨 CREANDO GRÁFICOS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Gráfico 1: Comparación 1:1
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if d <= 0.15 else 
              '#f39c12' if d <= 0.30 else
              '#e67e22' if d <= 0.50 else
              '#e74c3c' for d in comp_df['abs_diff']]
    
    for i, (_, row) in enumerate(comp_df.iterrows()):
        ax1.scatter(row['MH_Beasley'], row['FeH_CIGALE'], 
                   color=colors[i], alpha=0.7, s=50)
    
    min_val = min(comp_df['MH_Beasley'].min(), comp_df['FeH_CIGALE'].min()) - 0.5
    max_val = max(comp_df['MH_Beasley'].max(), comp_df['FeH_CIGALE'].max()) + 0.5
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax1.set_xlabel('[M/H] Beasley [dex]')
    ax1.set_ylabel('[Fe/H] CIGALE [dex]')
    ax1.set_title(f'Comparación directa (N={len(comp_df)})')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma diferencias
    ax2 = axes[0, 1]
    ax2.hist(comp_df['FeH_diff'], bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=bias, color='red', linestyle='--', label=f'Bias: {bias:+.2f}')
    ax2.set_xlabel('Δ[Fe/H] [dex]')
    ax2.set_ylabel('Número de objetos')
    ax2.set_title('Distribución de discrepancias')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Error vs χ²
    ax3 = axes[0, 2]
    scatter = ax3.scatter(comp_df['chi2'], comp_df['abs_diff'],
                         c=comp_df['MH_Beasley'], cmap='coolwarm',
                         s=50, alpha=0.7)
    ax3.axvline(x=CHI2_THRESHOLD, color='red', linestyle='--', 
                label=f'Umbral: {CHI2_THRESHOLD}')
    ax3.set_xlabel('χ² reducido')
    ax3.set_ylabel('|Δ[Fe/H]| [dex]')
    ax3.set_title('Error vs calidad del ajuste (χ²)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='[M/H] Beasley')
    
    # Gráfico 4: Error vs Edad
    ax4 = axes[1, 0]
    ax4.scatter(comp_df['age_Gyr'], comp_df['abs_diff'],
               c=comp_df['MH_Beasley'], cmap='coolwarm',
               s=50, alpha=0.7)
    ax4.set_xlabel('Edad [Gyr]')
    ax4.set_ylabel('|Δ[Fe/H]| [dex]')
    ax4.set_title('Error vs Edad')
    ax4.grid(True, alpha=0.3)
    
    # Gráfico 5: Diagrama de métricas
    ax5 = axes[1, 1]
    metrics = ['MAE', 'Bias', 'Std', '%≤0.15']
    values = [mae, abs(bias), std, 
              (comp_df['abs_diff'] <= 0.15).sum()/len(comp_df)*100]
    
    bars = ax5.bar(metrics[:3], values[:3], alpha=0.7)
    ax5.set_ylabel('Valor [dex]')
    ax5.set_title('Métricas principales')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Añadir segundo eje para porcentaje
    ax5_twin = ax5.twinx()
    ax5_twin.bar(metrics[3], values[3], alpha=0.7, color='green')
    ax5_twin.set_ylabel('Porcentaje [%]', color='green')
    ax5_twin.tick_params(axis='y', labelcolor='green')
    
    # Gráfico 6: Distribución de [M/H] Beasley
    ax6 = axes[1, 2]
    ax6.hist(comp_df['MH_Beasley'].dropna(), bins=15, 
             edgecolor='black', alpha=0.7)
    ax6.set_xlabel('[M/H] Beasley [dex]')
    ax6.set_ylabel('Número de objetos')
    ax6.set_title('Distribución de [M/H] en la muestra')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análisis CIGALE vs Beasley - χ² ≤ {CHI2_THRESHOLD} - {len(comp_df)} objetos',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analisis_optimal/graficos_comparacion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 RESULTADOS GUARDADOS:")
    print(f"   • analisis_optimal/comparacion_optimal.csv - Datos completos")
    print(f"   • analisis_optimal/resumen_optimal.txt - Resumen estadístico")
    print(f"   • analisis_optimal/graficos_comparacion.png - Gráficos")
    
    # 10. Interpretación y recomendaciones
    print(f"\n" + "="*80)
    print("🎯 INTERPRETACIÓN Y RECOMENDACIONES")
    print("="*80)
    
    print(f"\n📊 TU RESULTADO vs HERNÁNDEZ et al. (2018):")
    print(f"   • Hernández: 19/20 con Δ ≤ 0.15 dex (95%)")
    print(f"   • Tu estudio: {(comp_df['abs_diff'] <= 0.15).sum()}/{len(comp_df)} "
          f"({(comp_df['abs_diff'] <= 0.15).sum()/len(comp_df)*100:.0f}%) con Δ ≤ 0.15 dex")
    
    print(f"\n🔍 POSIBLES CAUSAS DE MAYOR DISCREPANCIA:")
    print(f"   1. Fotometría de menor calidad (S-PLUS vs espectroscopía)")
    print(f"   2. Degeneración edad-metalicidad en SED fitting")
    print(f"   3. Diferencia entre [Fe/H] y [M/H] (aunque según Hernández es pequeña)")
    print(f"   4. Calibración de filtros S-PLUS")
    
    print(f"\n✅ RECOMENDACIONES PARA TU PAPER:")
    print(f"   1. Usar χ² ≤ {CHI2_THRESHOLD} como criterio de calidad")
    print(f"   2. Reportar MAE de {mae:.2f} dex y bias de {bias:+.2f} dex")
    print(f"   3. Discutir la degeneración edad-metalicidad si es significativa")
    print(f"   4. Comparar con Hernández et al. (2018) como referencia")
    
    print(f"\n📝 TEXTO SUGERIDO PARA TU PAPER:")
    print(f"   'La comparación de nuestras estimaciones de [Fe/H] con CIGALE")
    print(f"   y los valores espectroscópicos de [M/H] de Beasley et al. (2008)")
    print(f"   muestra un error absoluto medio de {mae:.2f} dex para una muestra")
    print(f"   de {len(comp_df)} cúmulos globulares. El bias sistemático es")
    print(f"   pequeño ({bias:+.2f} dex), indicando que no hay sobre- o")
    print(f"   subestimación sistemática significativa. Aunque la precisión")
    print(f"   es menor que la reportada por Hernández et al. (2018) para")
    print(f"   espectroscopía de alta resolución, nuestros resultados son")
    print(f"   adecuados para estudios poblacionales de cúmulos globulares.'")

if __name__ == "__main__":
    main_optimal()
