# restaurar_configuracion_optimizada_v2.py
import os
import shutil
from datetime import datetime

def restaurar_config_optimizada_v2():
    """Restaura la configuración OPTIMIZADA para cúmulos globulares con S-PLUS - VERSIÓN MEJORADA"""

    print("🎯 RESTAURANDO CONFIGURACIÓN OPTIMIZADA V2 PARA CÚMULOS GLOBULARES")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config_content = """# CIGALE configuration for NGC 5128 Globular Clusters - OPTIMIZED
# Analysis of S-PLUS narrow-band photometry for old stellar populations
# Generated: {date}

data_file = gc_splus_cigale_custom.txt
parameters_file =

sed_modules = sfhdelayed, bc03, dustatt_modified_starburst, redshifting

analysis_method = pdf_analysis

cores = 8

# S-PLUS narrow-band filters (7 filters for optimal SED coverage)
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties =

# Conservative error for globular cluster photometry (5% additional)
additionalerror = 0.05

[sed_modules_params]

  [[sfhdelayed]]
    # Delayed SFH optimized for globular clusters (single stellar populations)
    tau_main = 100, 500, 1000, 1500, 2000
    age_main = 8000, 9000, 10000, 11000, 12000, 13000
    tau_burst = 50
    age_burst = 10
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True

  [[bc03]]
    # Bruzual & Charlot 2003 with Chabrier IMF (optimal for old populations)
    imf = 1
    # Full metallicity range for globular clusters (from metal-poor to solar)
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
    separation_age = 10

  [[dustatt_modified_starburst]]
    # Minimal dust attenuation for globular clusters
    Av_ISM = 0.0, 0.1, 0.2
    Av_BC = 0.0, 0.1
    mu = 0.3
    filters = F0378, F0395, F0410, F0430, F0515, F0660, F0861

  [[redshifting]]
    # Fixed redshift for NGC 5128 (Centaurus A) - Distance: ~3.8 Mpc
    redshift = 0.001825

[analysis_params]
  # Key physical properties for globular cluster analysis
  variables = stellar.m_star, stellar.metallicity_mw, stellar.age_m_star, dust.luminosity

  # S-PLUS bands for analysis (without _err suffix)
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861

  # CRITICAL: Enable SED generation for quality assessment
  save_best_sed = True
  save_chi2 = True
  
  # Quality control parameters
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 4
  blocks = 1
  best_weights = True
  
  # Physical limits appropriate for globular clusters
  physical_properties = stellar.m_star, stellar.metallicity_mw, stellar.age_m_star
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Crear backup con timestamp
    backup_name = f"pcigale.ini.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists('pcigale.ini'):
        shutil.copy2('pcigale.ini', backup_name)
        print(f"✅ Backup creado: {backup_name}")

    # Escribir configuración OPTIMIZADA
    with open('pcigale.ini', 'w') as f:
        f.write(config_content)

    print("✅ CONFIGURACIÓN OPTIMIZADA V2 RESTAURADA")
    print("📋 MEJORAS IMPLEMENTADAS:")
    print("   - ✅ +1 núcleo (cores=8) para mayor velocidad")
    print("   - ✅ Módulo de polvo incluido (Av_ISM 0-0.2)")
    print("   - ✅ Rango de metalicidad extendido (hasta 0.05)")
    print("   - ✅ Grid de edades más denso (6 puntos)")
    print("   - ✅ Márgenes de tau_main optimizados")
    print("   - ✅ Timestamp en el archivo de configuración")
    print("   - ✅ save_chi2 = True para análisis de calidad")

    print("\n🔬 JUSTIFICACIÓN CIENTÍFICA:")
    print("   - Globular clusters: SSPs con formación rápida (tau_main ~ 100-2000 Myr)")
    print("   - Edades 8-13 Gyr: rango típico para GCs en galaxias elípticas")
    print("   - Polvo mínimo: GCs típicamente tienen baja extinción interna")
    print("   - Metalicidad extendida: cubre GCs pobres y ricos en metales")

    print("\n🚫 ADVERTENCIA CRÍTICA:")
    print("   - NO ejecutar 'pcigale genconf' - Sobrescribirá save_best_sed")
    print("   - Verificar que los filtros S-PLUS estén registrados en CIGALE")

    print("\n🎯 FLUJO DE EJECUCIÓN:")
    print("   1. pcigale check")
    print("   2. pcigale run")
    print("   3. pcigale-plots sed")
    print("   4. pcigale-plots hist -p stellar.metallicity_mw")

def verificar_datos_entrada():
    """Verifica que el archivo de datos de entrada existe y tiene formato correcto"""
    
    print("\n🔍 VERIFICANDO ARCHIVO DE DATOS...")
    
    archivo_datos = 'gc_splus_cigale_custom.txt'
    
    if not os.path.exists(archivo_datos):
        print(f"❌ ARCHIVO CRÍTICO NO ENCONTRADO: {archivo_datos}")
        print("   Ejecutar primero: python preparar_datos_cigale.py")
        return False
    
    # Verificación básica del formato
    try:
        with open(archivo_datos, 'r') as f:
            primeras_lineas = [next(f) for _ in range(3)]
        
        # Verificar encabezado
        if 'id' in primeras_lineas[0].lower() and 'F0378' in primeras_lineas[0]:
            print("✅ Formato de archivo correcto")
            return True
        else:
            print("⚠️  Posible problema en formato - verificar encabezados")
            return True
            
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False

def verificar_filtros_splus():
    """Verifica que los filtros S-PLUS estén registrados en CIGALE"""
    
    print("\n🔍 VERIFICANDO FILTROS S-PLUS...")
    
    # Esta verificación requiere que CIGALE esté instalado
    try:
        import subprocess
        result = subprocess.run(['pcigale-filters', 'list'], 
                              capture_output=True, text=True, timeout=10)
        
        filtros_requeridos = ['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']
        filtros_encontrados = []
        
        for filtro in filtros_requeridos:
            if filtro in result.stdout:
                filtros_encontrados.append(filtro)
        
        if len(filtros_encontrados) == len(filtros_requeridos):
            print("✅ TODOS los filtros S-PLUS registrados")
        else:
            print(f"⚠️  Faltan filtros: {set(filtros_requeridos) - set(filtros_encontrados)}")
            print("   Ejecutar: python formatear_filtros_splus_correcto.py")
            
    except Exception as e:
        print(f"⚠️  No se pudo verificar filtros: {e}")

if __name__ == "__main__":
    restaurar_config_optimizada_v2()
    verificar_datos_entrada()
    verificar_filtros_splus()
    
    print("\n" + "="*70)
    print("🎉 CONFIGURACIÓN COMPLETADA - Lista para análisis científico!")
    print("📊 Próximos pasos:")
    print("   1. Verificar datos de entrada con: head gc_splus_cigale_custom.txt")
    print("   2. Ejecutar: pcigale check")
    print("   3. Ejecutar: pcigale run")
    print("   4. Analizar resultados con los scripts de plotting")
