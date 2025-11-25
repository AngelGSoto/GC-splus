# crear_config_emiles.py
import os
import shutil
import re
from datetime import datetime

def crear_configuracion_emiles():
    """
    Crea archivo de configuración E-MILES independiente para cúmulos globulares
    en NGC 5128 con datos S-PLUS
    """
    config_content = f"""# CIGALE configuration for NGC 5128 Globular Clusters - E-MILES
# Analysis of S-PLUS narrow-band photometry with E-MILES stellar library
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Target: Globular Clusters in Centaurus A (NGC 5128)

data_file = gc_splus_cigale_custom.txt
parameters_file =

# MODULES: Delayed SFH + E-MILES + minimal dust + redshifting
sed_modules = sfhdelayed, emiles, dustatt_modified_starburst, redshifting

analysis_method = pdf_analysis

# Use available cores (adjust based on your system)
cores = 8

# S-PLUS narrow-band filters (7 filters)
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties =

# Conservative 5% additional error for globular cluster photometry
additionalerror = 0.05

[sed_modules_params]

  [[sfhdelayed]]
    # Delayed star formation history - optimized for globular clusters (SSPs)
    tau_main = 100,500,1000,2000
    age_main = 8000,10000,12000,13000
    tau_burst = 50
    age_burst = 10
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True

  [[emiles]]
    # E-MILES stellar library - OPTIMAL for old stellar populations
    imf = chabrier03
    # Metallicity range covering poor to metal-rich globular clusters
    metallicity = 0.0001,0.0004,0.004,0.008,0.02
    separation_age = 10

  [[dustatt_modified_starburst]]
    # Minimal dust attenuation appropriate for globular clusters
    Av_ISM = 0.0,0.1,0.2
    mu = 0.3

  [[redshifting]]
    # Fixed redshift for NGC 5128 (Centaurus A) - distance ~3.8 Mpc
    redshift = 0.001825

[analysis_params]
  # Key physical properties to extract for globular clusters
  variables = stellar.m_star, stellar.metallicity_mw, stellar.age_m_star

  # S-PLUS bands for analysis (without error suffixes)
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
"""

    config_file = 'pcigale.ini'
    
    # Crear backup si ya existe
    if os.path.exists(config_file):
        backup_name = f"{config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(config_file, backup_name)
        print(f"✅ Backup creado: {backup_name}")

    # Escribir archivo de configuración
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ CONFIGURACIÓN E-MILES CREADA: {config_file}")
    return config_file

def verificar_configuracion(config_file):
    """Verifica que el archivo de configuración se creó correctamente (robusta con regex)"""

    print(f"\n🔍 VERIFICANDO CONFIGURACIÓN: {config_file}")

    if not os.path.exists(config_file):
        print("❌ El archivo de configuración no se creó")
        return False

    with open(config_file, 'r') as f:
        contenido = f.read()

    checks = {
        'E-MILES library': bool(re.search(r'sed_modules\s*=\s*.*emiles', contenido)),
        'Save SEDs enabled': 'save_best_sed = True' in contenido,
        'Correct age range': bool(re.search(r'age_main\s*=\s*8000, ?10000, ?12000, ?13000', contenido)),
        'Metallicity range': bool(re.search(r'metallicity\s*=\s*0\.0001, ?0\.0004, ?0\.004, ?0\.008, ?0\.02', contenido)),
        'Redshift NGC 5128': 'redshift = 0.001825' in contenido,
        'S-PLUS filters': all(filtro in contenido for filtro in ['F0378', 'F0861'])
    }

    todos_correctos = True
    for elemento, estado in checks.items():
        icono = "✅" if estado else "❌"
        print(f"   {icono} {elemento}")
        if not estado:
            todos_correctos = False

    return todos_correctos

def mostrar_instrucciones_ejecucion():
    """Muestra instrucciones para ejecutar CIGALE con esta configuración"""
    print("\n🎯 INSTRUCCIONES DE EJECUCIÓN:")
    print("=" * 50)
    print("1. Verificar que el archivo de datos existe:")
    print("   ls -la gc_splus_cigale_custom.txt")
    print()
    print("2. Verificar que los filtros S-PLUS estén registrados:")
    print("   pcigale-filters list | grep -E '(F0378|F0395|F0410|F0430|F0515|F0660|F0861)'")
    print()
    print("3. Ejecutar CIGALE con la configuración E-MILES:")
    print("   pcigale run --config pcigale.ini --out results_emiles")
    print()
    print("4. Monitorear el progreso:")
    print("   tail -f results_emiles/log.txt")
    print()
    print("📊 PARÁMETROS CIENTÍFICOS INCLUIDOS:")
    print("   • Edades: 8-13 Gyr (rango típico GCs)")
    print("   • Metalicidades: 0.0001 a 0.02 (pobres a ricos en metales)")
    print("   • IMF: Chabrier (óptimo para poblaciones antiguas)")
    print("   • Polvo: Mínimo (Av = 0.0-0.2)")

def verificar_archivo_datos():
    """Verifica que el archivo de datos de entrada existe"""
    archivo_datos = 'gc_splus_cigale_custom.txt'
    
    print(f"\n📁 VERIFICANDO ARCHIVO DE DATOS: {archivo_datos}")
    if not os.path.exists(archivo_datos):
        print("❌ ARCHIVO DE DATOS NO ENCONTRADO")
        print("   Necesitas crear primero: gc_splus_cigale_custom.txt")
        print("   Usa el script de preparación de datos.")
        return False
    
    # Verificación básica del formato
    try:
        with open(archivo_datos, 'r') as f:
            primera_linea = f.readline().strip()
        
        if 'id' in primera_linea.lower() and 'F0378' in primera_linea:
            print("✅ Formato de archivo correcto")
            n_lineas = sum(1 for _ in open(archivo_datos)) - 1
            print(f"✅ {n_lineas} fuentes (excluyendo header)")
            return True
        else:
            print("⚠️  Posible problema en el formato del archivo")
            return True
            
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False

if __name__ == "__main__":
    print("🎯 CREANDO CONFIGURACIÓN E-MILES INDEPENDIENTE")
    print("=" * 60)
    print("Biblioteca: E-MILES (óptima para poblaciones estelares antiguas)")
    print("Target: Cúmulos globulares en NGC 5128 (Centaurus A)")
    print("Fotometría: S-PLUS (7 filtros estrechos)")
    print("=" * 60)
    
    # Crear configuración
    config_file = crear_configuracion_emiles()
    
    # Verificar configuración
    config_ok = verificar_configuracion(config_file)
    
    # Verificar archivo de datos
    datos_ok = verificar_archivo_datos()
    
    # Mostrar instrucciones si todo está bien
    if config_ok and datos_ok:
        mostrar_instrucciones_ejecucion()
        print(f"\n🎉 CONFIGURACIÓN E-MILES LISTA!")
        print("Siguiente paso: Ejecutar 'pcigale run --config pcigale.ini --out results_emiles'")
    else:
        print("\n⚠️  Algunos problemas detectados. Revisa antes de ejecutar CIGALE.")
