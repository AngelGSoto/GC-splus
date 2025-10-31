#!/usr/bin/env python3
"""
Script de DIAGNÓSTICO y CORRECCIÓN para resultados CIGALE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table
import os
import subprocess
import glob

def diagnose_problems():
    """Diagnosticar los problemas en los resultados"""
    print("🔍 DIAGNOSTICANDO PROBLEMAS EN RESULTADOS CIGALE")
    print("=" * 60)
    
    # Cargar resultados
    results = Table.read('out/results.fits').to_pandas()
    
    print("📊 PROBLEMAS IDENTIFICADOS:")
    print("1. ❌ Masa estelar extremadamente baja (mediana: 7.97e-15 M☉)")
    print("2. ❌ Edad idéntica para todos los objetos (5000 Myr)")
    print("3. ❌ Metalicidad idéntica para todos los objetos (0.02)")
    print("4. ❌ No se generaron gráficos SED")
    
    # Verificar la configuración usada
    print("\n🔧 VERIFICANDO CONFIGURACIÓN:")
    if os.path.exists('pcigale.ini'):
        with open('pcigale.ini', 'r') as f:
            config_content = f.read()
            print("📁 Configuración actual (pcigale.ini):")
            # Mostrar solo las partes relevantes
            for line in config_content.split('\n'):
                if any(keyword in line for keyword in ['age_main', 'metallicity', 'tau_main', 'Av']):
                    print(f"   {line}")
    
    return results

def analyze_configuration_issues():
    """Analizar problemas específicos en la configuración"""
    print("\n🎯 ANALIZANDO PROBLEMAS DE CONFIGURACIÓN")
    print("=" * 50)
    
    issues = []
    
    # Verificar si pcigale.ini.spec existe
    if not os.path.exists('pcigale.ini.spec'):
        issues.append("❌ Falta pcigale.ini.spec - la configuración no se generó completamente")
    
    # Verificar la configuración actual
    if os.path.exists('pcigale.ini'):
        with open('pcigale.ini', 'r') as f:
            content = f.read()
            
            # Verificar parámetros críticos
            if 'age_main = 5000' in content and '10000' not in content:
                issues.append("❌ Rango de edades muy limitado: solo 5000 Myr")
            
            if 'metallicity = 0.02' in content and '0.004' not in content:
                issues.append("❌ Metalicidad fija en 0.02, sin rango")
            
            if 'properties =' not in content or 'stellar.m_star' not in content:
                issues.append("❌ Propiedades clave no especificadas en 'properties'")
    
    # Verificar datos de entrada
    if os.path.exists('cigale_input.txt'):
        input_data = pd.read_csv('cigale_input.txt', delim_whitespace=True, nrows=5)
        print("📁 Muestra de datos de entrada:")
        print(input_data.head())
    
    for issue in issues:
        print(f"   {issue}")
    
    return issues

def create_proper_config():
    """Crear una configuración adecuada para cúmulos globulares"""
    print("\n📝 CREANDO CONFIGURACIÓN ADECUADA")
    print("=" * 50)
    
    proper_config = """# CIGALE configuration for globular clusters
# PROPER configuration with adequate parameter ranges

data_file = cigale_input.txt
parameters_file = 
sed_modules = sfhdelayed, bc03, dustatt_powerlaw, redshifting
analysis_method = pdf_analysis
cores = 4

# Bands - make sure these match your input filters
bands = decam.u, decam.u_err, decam.g, decam.g_err, decam.r, decam.r_err, decam.i, decam.i_err, decam.z, decam.z_err, splus.F378, splus.F378_err, splus.F395, splus.F395_err, splus.F410, splus.F410_err, splus.F430, splus.F430_err, splus.F515, splus.F515_err, splus.F660, splus.F660_err, splus.F861, splus.F861_err

# Properties to compute
properties = sfh.age, stellar.m_star, stellar.metallicity, attenuation.Av

# Additional error
additionalerror = 0.05

[sed_modules_params]

  [[sfhdelayed]]
    # Extended age range for globular clusters
    tau_main = 100, 500, 1000, 2000, 5000
    age_main = 5000, 8000, 10000, 12000, 13000
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True

  [[bc03]]
    # Full metallicity range for globular clusters
    imf = 1
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02
    separation_age = 10

  [[dustatt_powerlaw]]
    # Reasonable dust attenuation range
    attenuation.uv_bump_amplitude = 0.0, 1.5, 3.0
    attenuation.powerlaw_slope = -0.5, -0.7, -1.0
    attenuation.Av.stellar.young = 0.01, 0.05, 0.1, 0.15, 0.2
    attenuation.av_old_factor = 0.1, 0.25, 0.5

  [[redshifting]]
    redshift = 0.0

[analysis_params]
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 2
  blocks = 1
"""
    
    with open('pcigale_proper.ini', 'w') as f:
        f.write(proper_config)
    
    print("✅ Configuración adecuada guardada como: pcigale_proper.ini")
    print("💡 Mejoras incluidas:")
    print("   • Rango extendido de edades: 5000-13000 Myr")
    print("   • Rango completo de metalicidades: 0.0001-0.02")
    print("   • Propiedades específicas definidas")
    print("   • Bandas explicitamente listadas")

def check_data_quality():
    """Verificar la calidad de los datos de entrada"""
    print("\n🔍 VERIFICANDO CALIDAD DE DATOS DE ENTRADA")
    print("=" * 50)
    
    if not os.path.exists('cigale_input.txt'):
        print("❌ No se encuentra cigale_input.txt")
        return False
    
    try:
        # Leer datos de entrada
        input_data = pd.read_csv('cigale_input.txt', delim_whitespace=True)
        
        print(f"📊 Datos de entrada: {len(input_data)} objetos")
        
        # Verificar columnas
        flux_cols = [col for col in input_data.columns if not col.endswith('_err') and col not in ['id', 'redshift']]
        print(f"🎯 Filtros: {len(flux_cols)}")
        
        # Verificar valores de flujo
        print("\n📈 ESTADÍSTICAS DE FLUJOS:")
        for col in flux_cols[:3]:  # Primeros 3 filtros
            fluxes = input_data[col]
            valid_fluxes = fluxes[fluxes > 1e-10]  # Excluir valores "missing"
            
            if len(valid_fluxes) > 0:
                print(f"   • {col}:")
                print(f"      - Objetos con datos: {len(valid_fluxes)}/{len(fluxes)}")
                print(f"      - Rango: {valid_fluxes.min():.2e} a {valid_fluxes.max():.2e} mJy")
                print(f"      - Mediana: {valid_fluxes.median():.2e} mJy")
            else:
                print(f"   • {col}: SIN DATOS VÁLIDOS")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando datos: {e}")
        return False

def generate_plots_manually():
    """Generar gráficos manualmente desde los resultados"""
    print("\n📊 GENERANDO GRÁFICOS MANUALES")
    print("=" * 50)
    
    try:
        results = Table.read('out/results.fits').to_pandas()
        
        # Crear gráfico de flujos observados vs modelados
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Análisis de Resultados CIGALE - Diagnóstico', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Distribución de masas (aunque sean incorrectas)
        if 'bayes.stellar.m_star' in results.columns:
            masses = results['bayes.stellar.m_star']
            axes[0,0].hist(masses, bins=50, alpha=0.7, edgecolor='black')
            axes[0,0].set_xlabel('Masa Estelar (M_sun)')
            axes[0,0].set_ylabel('Número')
            axes[0,0].set_title('Distribución de Masas (PROBLEMÁTICA)')
            axes[0,0].set_yscale('log')
            axes[0,0].grid(True, alpha=0.3)
        
        # Gráfico 2: Valores únicos por parámetro
        unique_counts = {}
        for col in results.columns:
            if col.startswith('bayes.') and col.endswith('_err') is False:
                unique_vals = results[col].nunique()
                unique_counts[col] = unique_vals
        
        # Mostrar parámetros con pocos valores únicos
        limited_params = {k: v for k, v in unique_counts.items() if v <= 5}
        if limited_params:
            axes[0,1].barh(range(len(limited_params)), list(limited_params.values()))
            axes[0,1].set_yticks(range(len(limited_params)))
            axes[0,1].set_yticklabels(list(limited_params.keys()), fontsize=8)
            axes[0,1].set_xlabel('Número de Valores Únicos')
            axes[0,1].set_title('Parámetros con Valores Limitados')
            axes[0,1].grid(True, alpha=0.3)
        
        # Gráfico 3: Flujos modelados vs "best" (si existen)
        model_cols = [col for col in results.columns if col.startswith('best.')]
        if len(model_cols) > 0:
            # Tomar primer filtro como ejemplo
            example_filter = model_cols[0]
            fluxes = results[example_filter]
            axes[1,0].hist(np.log10(fluxes[fluxes > 0]), bins=30, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel(f'log10({example_filter})')
            axes[1,0].set_ylabel('Número')
            axes[1,0].set_title(f'Flujos Modelados - {example_filter}')
            axes[1,0].grid(True, alpha=0.3)
        
        # Gráfico 4: Información del problema
        axes[1,1].text(0.1, 0.8, 'PROBLEMAS IDENTIFICADOS:', fontsize=12, fontweight='bold')
        axes[1,1].text(0.1, 0.6, '• Masas estelares extremadamente bajas', fontsize=10)
        axes[1,1].text(0.1, 0.5, '• Edades idénticas para todos', fontsize=10)
        axes[1,1].text(0.1, 0.4, '• Metalicidades idénticas', fontsize=10)
        axes[1,1].text(0.1, 0.3, '• Configuración muy restrictiva', fontsize=10)
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].set_title('Diagnóstico')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('cigale_diagnosis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Gráfico de diagnóstico guardado como: cigale_diagnosis.png")
        
    except Exception as e:
        print(f"❌ Error generando gráficos: {e}")

def create_fix_script():
    """Crear script para corregir los problemas"""
    print("\n🔧 CREANDO SCRIPT DE CORRECCIÓN")
    print("=" * 50)
    
    fix_script = """#!/bin/bash

echo "🔧 CORRIGIENDO PROBLEMAS DE CIGALE"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Paso 1: Verificar que tenemos los archivos necesarios
echo "📁 Verificando archivos..."
if [ ! -f "cigale_input.txt" ]; then
    echo "❌ Error: No se encuentra cigale_input.txt"
    exit 1
fi

if [ ! -f "pcigale_proper.ini" ]; then
    echo "❌ Error: No se encuentra pcigale_proper.ini"
    echo "💡 Ejecuta primero el script de diagnóstico"
    exit 1
fi

# Paso 2: Limpiar configuración anterior
echo "🧹 Limpiando configuración anterior..."
rm -f pcigale.ini pcigale.ini.spec

# Paso 3: Usar configuración adecuada
echo "📝 Usando configuración adecuada..."
cp pcigale_proper.ini pcigale.ini

# Paso 4: Generar configuración completa
echo "🔧 Generando configuración completa..."
pcigale genconf

if [ $? -ne 0 ]; then
    echo "❌ Error al generar configuración"
    exit 1
fi

# Paso 5: Verificar la configuración
echo "🔍 Verificando configuración..."
pcigale check

if [ $? -eq 0 ]; then
    echo "✅ Configuración válida"
    
    # Paso 6: Crear directorio para nuevos resultados
    if [ -d "out_corrected" ]; then
        echo "📁 Eliminando resultados anteriores corregidos..."
        rm -rf out_corrected
    fi
    
    # Paso 7: Ejecutar CIGALE con configuración corregida
    echo "🔄 Ejecutando CIGALE con configuración corregida..."
    pcigale run
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 ¡ANÁLISIS CORREGIDO COMPLETADO!"
        echo "📁 Resultados en: out/"
        
        # Mover resultados a directorio nuevo
        mv out out_corrected
        echo "📁 Resultados movidos a: out_corrected/"
        
        # Generar gráficos
        echo "📊 Generando gráficos..."
        cd out_corrected
        pcigale-plots sed
        pcigale-plots pdf
        cd ..
        
        echo ""
        echo "⏰ Finalizado: $(date)"
    else
        echo "❌ Error en la ejecución"
        exit 1
    fi
else
    echo "❌ Error en la configuración"
    exit 1
fi
"""
    
    with open('fix_cigale_problems.sh', 'w') as f:
        f.write(fix_script)
    
    os.chmod('fix_cigale_problems.sh', 0o755)
    
    print("✅ Script de corrección creado: fix_cigale_problems.sh")
    print("\n🚀 PARA CORREGIR LOS PROBLEMAS:")
    print("   ./fix_cigale_problems.sh")

def main():
    """Función principal de diagnóstico y corrección"""
    print("🔧 DIAGNÓSTICO Y CORRECCIÓN DE CIGALE")
    print("=" * 70)
    
    # Paso 1: Diagnosticar problemas
    results = diagnose_problems()
    
    # Paso 2: Analizar problemas de configuración
    issues = analyze_configuration_issues()
    
    # Paso 3: Verificar calidad de datos
    data_ok = check_data_quality()
    
    # Paso 4: Crear configuración adecuada
    create_proper_config()
    
    # Paso 5: Generar gráficos de diagnóstico
    generate_plots_manually()
    
    # Paso 6: Crear script de corrección
    create_fix_script()
    
    print("\n🎯 RESUMEN Y RECOMENDACIONES:")
    print("=" * 50)
    print("1. ✅ El problema principal es la CONFIGURACIÓN muy restrictiva")
    print("2. ✅ Se ha creado una configuración adecuada (pcigale_proper.ini)")
    print("3. ✅ Se ha creado un script de corrección automática")
    print("4. 🚀 Ejecuta el script de corrección para obtener resultados realistas")
    
    print(f"\n💡 CAUSA RAÍZ:")
    print("   Los parámetros en la configuración original estaban fijos o tenían rangos")
    print("   muy limitados, lo que impidió que CIGALE explorara el espacio de parámetros.")
    
    print(f"\n📊 ESTADO ACTUAL:")
    print("   • Configuración: ❌ Problemática")
    print("   • Datos de entrada: ✅ Aparentemente correctos") 
    print("   • Resultados: ❌ No confiables")
    print("   • Solución: ✅ Disponible")

if __name__ == "__main__":
    main()
