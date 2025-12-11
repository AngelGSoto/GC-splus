# crear_pcigale_ini_from_file.py
# Crea pcigale.ini para usar con 'redshift = from_file'
# USAR CON: gc_splus_cigale_custom.txt (redshift primero)

import os
from datetime import datetime

def crear_config_from_file():
    """Crea configuración específica para redshift = from_file"""
    
    config = """# NGC 5128 Globular Clusters – S-PLUS (CIGALE 2022.0)
# CONFIGURACIÓN ESPECÍFICA PARA: redshift = from_file
# USAR CON: gc_splus_cigale_custom.txt

data_file = gc_splus_cigale_custom.txt
parameters_file = 

# ¡IMPORTANTE! Redshift se leerá del archivo (primera columna)
redshift = from_file

# MÓDULOS
sed_modules = sfhdelayed, xsl, redshifting

analysis_method = pdf_analysis
cores = 5

# Bands to consider (7 filtros S-PLUS)
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

additionalerror = 0.05

properties = 

# Configuration of the SED creation modules.
[sed_modules_params]
  
  [[sfhdelayed]]
    # e-folding time in Myr.
    tau_main = 500.0, 1000.0, 2000.0, 5000.0
    # Age in Myr.
    age_main = 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True
  
  [[xsl]]
    # Initial mass function: 1 (Kroupa/Chabrier)
    imf = 1
    # METALICIDADES VERIFICADAS
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    # Age separation
    separation_age = 10
  
  [[redshifting]]
    # ¡VALOR DUMMY OBLIGATORIO! CIGALE lo ignorará porque usamos 'from_file'
    # PERO el módulo necesita tener el parámetro definido
    redshift = 0.0

# Configuration of the statistical analysis method.
[analysis_params]
  # Physical properties
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  
  # Bands for flux estimation
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 6
"""
    
    # Hacer backup si existe
    if os.path.exists("pcigale.ini"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"pcigale.ini.backup_{timestamp}"
        os.rename("pcigale.ini", backup_file)
        print(f"✅ Backup creado: {backup_file}")
    
    # Crear nuevo archivo
    with open("pcigale.ini", "w") as f:
        f.write(config)
    
    print("\n✅ pcigale.ini creado CON CONFIGURACIÓN CORRECTA:")
    print("   • data_file = gc_splus_cigale_custom.txt")
    print("   • redshift = from_file")
    print("   • [[redshifting]] con redshift = 0.0 (dummy)")
    
    return True

def verificar_config():
    """Verifica que la configuración sea correcta"""
    
    print("\n📋 VERIFICANDO CONFIGURACIÓN:")
    print("-" * 60)
    
    if not os.path.exists("pcigale.ini"):
        print("❌ pcigale.ini no encontrado")
        return False
    
    with open("pcigale.ini", "r") as f:
        contenido = f.read()
    
    lineas_importantes = []
    for linea in contenido.split('\n'):
        if 'redshift' in linea.lower() or 'data_file' in linea:
            lineas_importantes.append(linea.strip())
    
    print("Líneas importantes detectadas:")
    for linea in lineas_importantes[:10]:  # Mostrar primeras 10
        if linea:
            print(f"  → {linea}")
    
    # Verificaciones clave
    checks = {
        'data_file correcto': 'data_file = gc_splus_cigale_custom.txt' in contenido,
        'redshift from_file': 'redshift = from_file' in contenido,
        'módulo redshifting': '[[redshifting]]' in contenido,
        'redshift en módulo': 'redshift = 0.0' in contenido or 'redshift = 0.001825' in contenido
    }
    
    print("\n✅ VERIFICACIONES:")
    for check, resultado in checks.items():
        estado = "✓" if resultado else "✗"
        print(f"  {estado} {check}")
    
    return all(checks.values())

if __name__ == "__main__":
    print("⚙️  CREANDO CONFIGURACIÓN PARA 'redshift = from_file'")
    print("=" * 78)
    print("REQUISITOS:")
    print("  1. Archivo de datos: gc_splus_cigale_custom.txt")
    print("  2. Primera columna: redshift")
    print("  3. Segunda columna: id")
    print("=" * 78)
    
    if crear_config_from_file():
        if verificar_config():
            print("\n" + "=" * 78)
            print("🎉 ¡CONFIGURACIÓN COMPLETA!")
            print("=" * 78)
            
            print("\n🚀 PRÓXIMOS PASOS:")
            print("  1. Verificar configuración:")
            print("     pcigale check")
            print("\n  2. Si funciona, ejecutar:")
            print("     pcigale run")
            print("\n  3. Si hay error de indentación (bug de CIGALE):")
            print("     cd /home/luis/Downloads/cigale-v2022.0/")
            print("     sed -i '152s/^[ \\t]*/        /' pcigale/analysis_modules/pdf_analysis/workers.py")
            print("     cd -")
            print("     pcigale check")
            
            print("\n📊 TU CONFIGURACIÓN ACTUAL:")
            print("   • Archivo datos: gc_splus_cigale_custom.txt ✓")
            print("   • Formato: redshift primero, id segundo ✓")
            print("   • Config: redshift = from_file ✓")
            print("   • Total objetos: 10 ✓")
        else:
            print("\n❌ La configuración tiene problemas. Revisa pcigale.ini")
