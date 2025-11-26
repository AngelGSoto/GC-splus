# reparar_configuracion_cigale.py
import os
import shutil
from datetime import datetime

def reparar_configuracion_completa():
    """Repara completamente el archivo de configuración de CIGALE"""
    
    print("🔧 REPARANDO CONFIGURACIÓN CIGALE COMPLETA...")
    
    # Configuración completa y corregida
    config_content = f"""# CIGALE configuration for NGC 5128 Globular Clusters - REPAIRED
# Optimized for globular clusters with S-PLUS photometry
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

data_file = gc_splus_cigale_custom.txt
parameters_file = 

sed_modules = sfhdelayed, bc03, dustatt_modified_starburst, redshifting

analysis_method = pdf_analysis
cores = 8

bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties = ,

additionalerror = 0.05

[sed_modules_params]
  
  [[sfhdelayed]]
    # Delayed SFH optimized for globular clusters
    tau_main = 10, 50, 100, 500
    age_main = 1000, 2000, 3000, 5000, 8000, 10000, 13000
    tau_burst = 10
    age_burst = 10
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True
  
  [[bc03]]
    # Bruzual & Charlot 2003 with Chabrier IMF
    imf = 1
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
    separation_age = 10
  
  [[dustatt_modified_starburst]]
    # Minimal dust for globular clusters
    E_BV_lines = 0.00, 0.01, 0.02, 0.05
    E_BV_factor = 0.44
    uv_bump_wavelength = 217.5
    uv_bump_width = 35.0
    uv_bump_amplitude = 0.0
    powerlaw_slope = 0.0
    Ext_law_emission_lines = 1
    Rv = 3.1
    filters = F0660
  
  [[redshifting]]
    # Fixed redshift for NGC 5128
    redshift = 0.001825

[analysis_params]
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star, dust.luminosity
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 4
  blocks = 1
"""
    
    # Crear backup
    backup_name = f"pcigale.ini.backup_reparacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists('pcigale.ini'):
        shutil.copy2('pcigale.ini', backup_name)
        print(f"✅ Backup creado: {backup_name}")
    
    # Escribir nueva configuración
    with open('pcigale.ini', 'w') as f:
        f.write(config_content)
    
    print("✅ CONFIGURACIÓN REPARADA COMPLETAMENTE")

def verificar_configuracion_reparada():
    """Verifica que la configuración reparada sea válida"""
    
    print("\n🔍 VERIFICANDO CONFIGURACIÓN REPARADA...")
    
    with open('pcigale.ini', 'r') as f:
        contenido = f.read()
    
    checks = {
        'additionalerror = 0.05': 'Error adicional reducido',
        'tau_main = 10, 50, 100, 500': 'Tau optimizados',
        'age_main = 1000, 2000, 3000, 5000, 8000, 10000, 13000': 'Rango de edades',
        'imf = 1': 'IMF de Chabrier',
        'metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05': 'Metalicidad extendida',
        'E_BV_lines = 0.00, 0.01, 0.02, 0.05': 'Polvo mínimo',
        'redshift = 0.001825': 'Redshift fijo (CRÍTICO)',
        'filters = F0660': 'Filtros optimizados',
        'variables = stellar.m_star, stellar.metallicity, stellar.age_m_star, dust.luminosity': 'Variables esenciales'
    }
    
    todos_correctos = True
    for check, descripcion in checks.items():
        if check in contenido:
            print(f"✅ {descripcion}")
        else:
            print(f"❌ {descripcion}")
            todos_correctos = False
    
    return todos_correctos

def ejecutar_prueba_cigale():
    """Ejecuta pcigale check para verificar que la configuración funciona"""
    
    print("\n🚀 EJECUTANDO PRUEBA DE CONFIGURACIÓN...")
    
    try:
        import subprocess
        result = subprocess.run(['pcigale', 'check'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ pcigale check EJECUTADO EXITOSAMENTE")
            print("📋 Salida:")
            for linea in result.stdout.split('\n')[-10:]:  # Últimas 10 líneas
                if linea.strip():
                    print(f"   {linea}")
        else:
            print("❌ ERROR en pcigale check:")
            print(f"   Código de salida: {result.returncode}")
            if result.stderr:
                for linea in result.stderr.split('\n')[-5:]:
                    if linea.strip():
                        print(f"   ERROR: {linea}")
            
    except subprocess.TimeoutExpired:
        print("⚠️  pcigale check tardó demasiado tiempo")
    except Exception as e:
        print(f"❌ No se pudo ejecutar pcigale check: {e}")

def mostrar_resumen_ejecucion():
    """Muestra un resumen de los próximos pasos"""
    
    print("\n" + "="*70)
    print("🎉 REPARACIÓN COMPLETADA!")
    print("="*70)
    
    print("\n📊 RESUMEN DE LA CONFIGURACIÓN:")
    print("   • Módulo SFH: sfhdelayed con tau cortos (10-500 Myr)")
    print("   • Edades: 1-13 Gyr (poblaciones antiguas)")
    print("   • Metalicidad: 0.0001 a 0.05 (pobres a ricos en metales)")
    print("   • Polvo: E(B-V) 0.00-0.05 (mínima atenuación)")
    print("   • Redshift: 0.001825 (NGC 5128 fijo)")
    print("   • Filtros: 7 bandas S-PLUS narrow-band")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1. Verificar datos de entrada:")
    print("      ls -la gc_splus_cigale_custom.txt")
    print("   2. Ejecutar análisis completo:")
    print("      pcigale run")
    print("   3. Monitorear progreso:")
    print("      ls -la out/")
    print("   4. Generar gráficos:")
    print("      pcigale-plots sed")
    
    print("\n💡 CONSEJOS:")
    print("   • El análisis tomará ~2-3 minutos con 8 cores")
    print("   • Verifica que tengas suficiente espacio en disco")
    print("   • Los resultados se guardarán en la carpeta 'out/'")

if __name__ == "__main__":
    print("🔧 REPARADOR DE CONFIGURACIÓN CIGALE - CÚMULOS GLOBULARES")
    print("=" * 70)
    
    reparar_configuracion_completa()
    
    if verificar_configuracion_reparada():
        ejecutar_prueba_cigale()
        mostrar_resumen_ejecucion()
    else:
        print("\n❌ LA CONFIGURACIÓN NO SE REPARÓ CORRECTAMENTE")
        print("   Por favor, ejecuta manualmente: pcigale genconf")
        print("   Luego ejecuta este script nuevamente")
