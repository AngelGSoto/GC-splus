#!/usr/bin/env python3
"""
Script para ANALIZAR los resultados de CIGALE - VERSIÓN MEJORADA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table
import os
import glob
import subprocess

def explore_results_columns():
    """Explorar las columnas disponibles en los resultados"""
    print("🔍 EXPLORANDO COLUMNAS DISPONIBLES")
    print("=" * 50)
    
    try:
        # Cargar resultados
        if os.path.exists('out/results.fits'):
            results = Table.read('out/results.fits').to_pandas()
        elif os.path.exists('out/results.txt'):
            results = pd.read_csv('out/results.txt', delim_whitespace=True)
        else:
            print("❌ No se encontraron archivos de resultados")
            return None
        
        print(f"📊 Número de columnas: {len(results.columns)}")
        print(f"📊 Número de objetos: {len(results)}")
        
        # Mostrar todas las columnas disponibles
        print("\n📋 COLUMNAS DISPONIBLES:")
        for i, col in enumerate(results.columns):
            print(f"   {i+1:3d}. {col}")
        
        # Buscar columnas clave por patrones
        print("\n🔎 BUSCANDO COLUMNAS CLAVE:")
        key_patterns = {
            'Masa estelar': ['stellar.m_star', 'bayes.stellar.m_star', 'best.stellar.m_star'],
            'Edad': ['sfh.age', 'bayes.sfh.age', 'best.sfh.age'],
            'Metalicidad': ['stellar.metallicity', 'bayes.stellar.metallicity', 'best.stellar.metallicity'],
            'Atenuación': ['attenuation.Av', 'bayes.attenuation.Av', 'best.attenuation.Av'],
            'Chi2': ['bayes.chi2', 'best.chi2']
        }
        
        found_columns = {}
        for key, patterns in key_patterns.items():
            for pattern in patterns:
                if pattern in results.columns:
                    found_columns[key] = pattern
                    print(f"   ✅ {key}: {pattern}")
                    break
            else:
                print(f"   ❌ {key}: No encontrado")
        
        return results, found_columns
        
    except Exception as e:
        print(f"❌ Error explorando resultados: {e}")
        return None, {}

def plot_available_properties(results, found_columns):
    """Graficar las propiedades disponibles"""
    print("\n📈 CREANDO GRÁFICOS CON PROPIEDADES DISPONIBLES")
    print("=" * 50)
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Determinar cuántos gráficos podemos hacer
    available_plots = len(found_columns)
    if available_plots == 0:
        print("❌ No se encontraron propiedades para graficar")
        return
    
    # Crear figura con subgráficos
    n_cols = min(3, available_plots)
    n_rows = (available_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if available_plots == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Propiedades de los Cúmulos Globulares - CIGALE', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for prop_name, col_name in found_columns.items():
        if plot_idx >= n_rows * n_cols:
            break
            
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        try:
            data = results[col_name]
            
            # Filtrar valores no numéricos
            if data.dtype == object:
                try:
                    data = pd.to_numeric(data, errors='coerce')
                except:
                    print(f"   ⚠️  No se pudo convertir {col_name} a numérico")
                    continue
            
            # Remover NaN e infinitos
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(data) > 0:
                # Histograma
                ax.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)
                ax.set_xlabel(prop_name)
                ax.set_ylabel('Densidad')
                ax.set_title(f'Distribución de {prop_name}')
                ax.grid(True, alpha=0.3)
                
                # Añadir estadísticas básicas
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Media: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Mediana: {median_val:.3f}')
                ax.legend(fontsize=8)
                
                plot_idx += 1
                print(f"   ✅ Gráfico creado para {prop_name}")
            else:
                print(f"   ⚠️  No hay datos válidos para {prop_name}")
                
        except Exception as e:
            print(f"   ❌ Error graficando {prop_name}: {e}")
    
    # Ocultar ejes vacíos
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('cigale_available_properties.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráficos guardados como: cigale_available_properties.png")

def generate_detailed_summary(results, found_columns):
    """Generar un resumen detallado de las propiedades"""
    print("\n📊 GENERANDO RESUMEN DETALLADO")
    print("=" * 50)
    
    summary_data = {}
    
    for prop_name, col_name in found_columns.items():
        try:
            data = results[col_name]
            
            # Convertir a numérico si es necesario
            if data.dtype == object:
                data = pd.to_numeric(data, errors='coerce')
            
            # Filtrar valores válidos
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(data) > 0:
                stats = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    '25%': data.quantile(0.25),
                    'median': data.median(),
                    '75%': data.quantile(0.75),
                    'max': data.max()
                }
                summary_data[prop_name] = stats
                
                print(f"\n📈 {prop_name} ({col_name}):")
                print(f"   • N: {stats['count']}")
                print(f"   • Media: {stats['mean']:.4f}")
                print(f"   • Mediana: {stats['median']:.4f}")
                print(f"   • Std: {stats['std']:.4f}")
                print(f"   • Rango: {stats['min']:.4f} - {stats['max']:.4f}")
                
        except Exception as e:
            print(f"❌ Error calculando estadísticas para {prop_name}: {e}")
    
    # Guardar resumen en CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data).T
        summary_df.to_csv('detailed_results_summary.csv')
        print(f"\n✅ Resumen detallado guardado como: detailed_results_summary.csv")
    
    return summary_data

def check_sed_plots():
    """Verificar y mostrar información sobre los gráficos SED"""
    print("\n📊 VERIFICANDO GRÁFICOS SED")
    print("=" * 50)
    
    # Buscar archivos PDF de SED
    sed_pdfs = glob.glob("out/*_best_model.pdf")
    
    if sed_pdfs:
        print(f"✅ Encontrados {len(sed_pdfs)} gráficos SED en PDF")
        print("📁 Algunos ejemplos:")
        for pdf in sed_pdfs[:5]:  # Mostrar solo los primeros 5
            print(f"   • {os.path.basename(pdf)}")
        
        if len(sed_pdfs) > 5:
            print(f"   ... y {len(sed_pdfs) - 5} más")
    else:
        print("❌ No se encontraron gráficos SED en PDF")
        print("💡 Ejecuta: pcigale-plots sed  para generarlos")

def check_analysis_plots():
    """Verificar y mostrar información sobre otros gráficos de análisis"""
    print("\n📊 VERIFICANDO OTROS GRÁFICOS DE ANÁLISIS")
    print("=" * 50)
    
    # Buscar diferentes tipos de gráficos
    plot_types = {
        'PDF': glob.glob("out/*_pdf.pdf"),
        'Chi2': glob.glob("out/*_chi2.pdf"),
        'SED': glob.glob("out/*_best_model.pdf")
    }
    
    for plot_type, files in plot_types.items():
        if files:
            print(f"✅ {plot_type}: {len(files)} archivos")
        else:
            print(f"❌ {plot_type}: No encontrados")

def create_interpretation_guide(summary_data):
    """Crear una guía de interpretación para cúmulos globulares"""
    print("\n💡 GUÍA DE INTERPRETACIÓN PARA CÚMULOS GLOBULARES")
    print("=" * 50)
    
    print("\n📚 VALORES TÍPICOS ESPERADOS:")
    print("   • Edad: 10-13 Gyr (poblaciones viejas)")
    print("   • Metalicidad: [Fe/H] ≈ -2.0 a 0.0 (log Z desde ~0.0001 a 0.02)")
    print("   • Masa estelar: 10⁴-10⁶ M☉")
    print("   • Atenuación por polvo: A_V < 0.1 mag (muy baja)")
    
    print("\n🔍 COMPARACIÓN CON TUS RESULTADOS:")
    
    if 'Edad' in summary_data:
        age_data = summary_data['Edad']
        age_median = age_data['median'] / 1000  # Convertir a Gyr
        print(f"   • Edad mediana: {age_median:.1f} Gyr")
        if age_median > 8:
            print("     ✅ Consistente con cúmulos globulares viejos")
        else:
            print("     ⚠️  Edad más joven de lo esperado")
    
    if 'Metalicidad' in summary_data:
        met_data = summary_data['Metalicidad']
        print(f"   • Metalicidad mediana: {met_data['median']:.4f}")
        if 0.0001 <= met_data['median'] <= 0.02:
            print("     ✅ En rango típico de cúmulos globulares")
        else:
            print("     ⚠️  Fuera del rango típico")
    
    if 'Masa estelar' in summary_data:
        mass_data = summary_data['Masa estelar']
        print(f"   • Masa estelar mediana: {mass_data['median']:.2e} M☉")
        if 1e4 <= mass_data['median'] <= 1e7:
            print("     ✅ En rango típico de cúmulos globulares")
        else:
            print("     ⚠️  Fuera del rango típico de masas")
    
    if 'Atenuación' in summary_data:
        av_data = summary_data['Atenuación']
        print(f"   • Atenuación mediana (A_V): {av_data['median']:.3f} mag")
        if av_data['median'] < 0.5:
            print("     ✅ Baja atenuación, como se espera")
        else:
            print("     ⚠️  Atenuación más alta de lo esperado")

def main():
    """Función principal mejorada"""
    print("🚀 ANÁLISIS MEJORADO DE RESULTADOS CIGALE")
    print("=" * 60)
    
    # Explorar columnas disponibles
    results, found_columns = explore_results_columns()
    
    if results is None:
        print("❌ No se pudieron cargar los resultados")
        return
    
    # Generar gráficos con las propiedades disponibles
    plot_available_properties(results, found_columns)
    
    # Generar resumen detallado
    summary_data = generate_detailed_summary(results, found_columns)
    
    # Verificar gráficos SED
    check_sed_plots()
    
    # Verificar otros gráficos
    check_analysis_plots()
    
    # Crear guía de interpretación
    create_interpretation_guide(summary_data)
    
    print("\n🎉 ANÁLISIS COMPLETADO")
    print("=" * 60)
    print("📁 ARCHIVOS GENERADOS:")
    print("   • cigale_available_properties.png - Gráficos de propiedades disponibles")
    print("   • detailed_results_summary.csv   - Resumen estadístico detallado")
    print("   • Gráficos en out/               - SEDs y análisis (si existen)")
    
    print(f"\n💡 PRÓXIMOS PASOS:")
    print("   1. Revisar los gráficos SED en out/ para verificar la calidad del ajuste")
    print("   2. Comparar las propiedades con valores típicos de cúmulos globulares")
    print("   3. Analizar la distribución de masas y edades")
    print("   4. Verificar la consistencia de las metalicidades")

if __name__ == "__main__":
    main()
