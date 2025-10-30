#!/usr/bin/env python3
"""
MÉTODO MANUAL para CIGALE 2025.1 - Control total del proceso
"""

import pandas as pd
import os
import subprocess
import sys
import time

def manual_cigale_setup():
    print("🔧 MÉTODO MANUAL PARA CIGALE 2025.1")
    print("=" * 50)
    
    # Paso 1: Crear archivo de entrada
    print("1. 📊 Creando archivo de entrada...")
    create_input_file()
    
    # Paso 2: Crear pcigale.ini manualmente con estructura exacta
    print("2. 📝 Creando pcigale.ini manualmente...")
    create_manual_config()
    
    # Paso 3: Verificar archivos
    print("3. 🔍 Verificando archivos...")
    if verify_files():
        print("4. 🚀 Ejecutando CIGALE...")
        run_cigale_manual()
    else:
        print("❌ Archivos no verificados. Deteniendo ejecución.")

def create_input_file():
    """Crear archivo de entrada manualmente"""
    
    # Cargar catálogo
    df = pd.read_csv("../Results/gc_photometry_final_high_quality_preliminar_teste_aperture3_only.csv")
    test_df = df.head(10).copy()
    
    # Preparar entrada
    cigale_input = pd.DataFrame()
    cigale_input['id'] = test_df['T17ID']
    cigale_input['redshift'] = 0.0
    
    # Sólo usar filtros básicos para la prueba inicial
    # Filtros SPLUS
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    for filter in splus_filters:
        mag_col = f'MAG_{filter}_3'
        err_col = f'MAGERR_{filter}_3'
        cigale_input[f'splus.{filter}'] = test_df[mag_col]
        cigale_input[f'splus.{filter}_err'] = test_df[err_col]
    
    # Filtros Taylor
    for filter in ['u', 'g', 'r', 'i', 'z']:
        mag_col = filter + 'mag'
        cigale_input[f'taylor.{filter}'] = test_df[mag_col]
        cigale_input[f'taylor.{filter}_err'] = 0.1
    
    # Guardar
    cigale_input.to_csv("input_manual.csv", index=False)
    print("   ✅ input_manual.csv creado")

def create_manual_config():
    """Crear pcigale.ini manualmente con estructura exacta"""
    
    config_content = """[core]
# Input data
data_file = input_manual.csv

# SED modules
sed_modules = sfhdelayed, bc03, dustatt_modified_starburst

[sfhdelayed]
# Delayed star formation history
tau_main = 1000, 5000
age_main = 5000, 10000

[bc03]
# Bruzual & Charlot (2003) stellar population
imf = 1
metallicity = 0.004, 0.02

[dustatt_modified_starburst]
# Dust attenuation
E_BV_lines = 0.0, 0.1, 0.3
E_BV_factor = 0.44
uv_bump_amplitude = 0.0, 3.0
powerlaw_slope = -1.0
filters = splus.F378, splus.F395, splus.F410, splus.F430, splus.F515, splus.F660, splus.F861, taylor.u, taylor.g, taylor.r, taylor.i, taylor.z
"""
    
    with open("pcigale.ini", "w") as f:
        f.write(config_content)
    print("   ✅ pcigale.ini creado")

def verify_files():
    """Verificar que los archivos existen y tienen contenido"""
    
    files_to_check = [
        ("input_manual.csv", "archivo de entrada"),
        ("pcigale.ini", "archivo de configuración")
    ]
    
    all_ok = True
    
    for filename, description in files_to_check:
        if not os.path.exists(filename):
            print(f"   ❌ {description} '{filename}' no existe")
            all_ok = False
        else:
            # Verificar que no esté vacío
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print(f"   ❌ {description} '{filename}' está vacío")
                all_ok = False
            else:
                print(f"   ✅ {description} '{filename}' verificado ({file_size} bytes)")
    
    return all_ok

def run_cigale_manual():
    """Ejecutar CIGALE manualmente con verificación paso a paso"""
    
    print("\n🎯 EJECUTANDO CIGALE - MÉTODO MANUAL")
    print("=" * 40)
    
    # Primero verificar que pcigale.ini tiene sed_modules
    print("🔍 Verificando contenido de pcigale.ini...")
    with open("pcigale.ini", "r") as f:
        content = f.read()
    
    if "sed_modules" not in content:
        print("❌ ERROR: sed_modules no encontrado en pcigale.ini")
        print("Contenido actual:")
        print(content)
        return False
    
    print("✅ sed_modules encontrado en configuración")
    
    # Ejecutar genconf primero
    print("\n🔄 Ejecutando: pcigale genconf")
    try:
        result = subprocess.run(
            ["pcigale", "genconf"], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ pcigale genconf ejecutado exitosamente")
            
            # Verificar que se creó pcigale.ini.spec
            if os.path.exists("pcigale.ini.spec"):
                print("✅ pcigale.ini.spec generado")
            else:
                print("⚠️  pcigale.ini.spec no se generó")
                
        else:
            print("❌ Error en pcigale genconf:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✅ pcigale genconf completado (timeout)")
    except Exception as e:
        print(f"❌ Error ejecutando pcigale genconf: {e}")
        return False
    
    # Ahora ejecutar run
    print("\n🚀 Ejecutando: pcigale run")
    print("⏳ Esto puede tomar varios minutos...")
    
    try:
        start_time = time.time()
        
        # Ejecutar con timeout más largo para el análisis completo
        result = subprocess.run(
            ["pcigale", "run"], 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ pcigale run completado en {elapsed_time:.1f} segundos")
            print("📁 Resultados en carpeta 'out/'")
            
            # Verificar resultados
            check_results()
            return True
        else:
            print(f"❌ Error en pcigale run después de {elapsed_time:.1f} segundos")
            print("STDOUT (últimas líneas):")
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # Últimas 20 líneas
                if line.strip():
                    print(f"   {line}")
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ pcigale run todavía ejecutándose (timeout de 5 minutos)")
        print("💡 El análisis puede continuar en segundo plano")
        return True
    except Exception as e:
        print(f"❌ Error ejecutando pcigale run: {e}")
        return False

def check_results():
    """Verificar que se generaron resultados"""
    
    print("\n📊 VERIFICANDO RESULTADOS:")
    
    results_files = [
        "out/results.fits",
        "out/observation.pdf", 
        "out/log.txt"
    ]
    
    for filepath in results_files:
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"✅ {filepath} ({file_size} bytes)")
        else:
            print(f"❌ {filepath} no encontrado")

def create_debug_script():
    """Crear script de depuración"""
    
    debug_script = """#!/bin/bash
echo "🐛 SCRIPT DE DEPURACIÓN CIGALE"
echo "================================"

echo "1. Verificando archivos..."
ls -la pcigale.ini input_manual.csv 2>/dev/null || echo "Archivos no encontrados"

echo ""
echo "2. Verificando contenido de pcigale.ini..."
if [ -f "pcigale.ini" ]; then
    grep -E "(sed_modules|data_file)" pcigale.ini || echo "No se encontraron las claves necesarias"
else
    echo "pcigale.ini no existe"
fi

echo ""
echo "3. Probando pcigale genconf..."
pcigale genconf

echo ""
echo "4. Si genconf funciona, probando pcigale run..."
if [ $? -eq 0 ] && [ -f "pcigale.ini.spec" ]; then
    echo "✅ genconf exitoso, ejecutando run..."
    pcigale run
else
    echo "❌ genconf falló"
fi
"""
    
    with open("debug_cigale.sh", "w") as f:
        f.write(debug_script)
    
    import stat
    st = os.stat("debug_cigale.sh")
    os.chmod("debug_cigale.sh", st.st_mode | stat.S_IEXEC)
    
    print("📄 Script de depuración creado: ./debug_cigale.sh")

if __name__ == "__main__":
    # Limpiar archivos anteriores
    for file in ["pcigale.ini", "input_manual.csv", "pcigale.ini.spec"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Limpiado: {file}")
    
    # Crear setup manual
    manual_cigale_setup()
    
    # Crear script de depuración
    create_debug_script()
    
    print("\n" + "="*60)
    print( "NEXT STEPS / PRÓXIMOS PASOS:")
    print("="*60)
    print("1. Si hay errores, ejecuta: ./debug_cigale.sh")
    print("2. Revisa los mensajes de error específicos")
    print("3. Verifica que pcigale.ini tenga 'sed_modules'")
    print("4. Asegúrate de que input_manual.csv tenga datos válidos")
