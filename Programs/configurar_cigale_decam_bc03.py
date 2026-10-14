#!/usr/bin/env python3
# cigale_setup_bc03.py
# Configuración para CIGALE con modelos BC03 (comparación con XSL)
# Rejilla fina en edad y metalicidad, todos los filtros S-PLUS + DECam

import os
import shutil
from datetime import datetime
import sys
import subprocess
import time

def verificar_archivo_datos(archivo):
    """Verifica que el archivo de datos tenga el formato correcto."""
    try:
        with open(archivo, 'r') as f:
            lineas = f.readlines()
        
        if len(lineas) < 2:
            print(f"❌ Error: {archivo} tiene menos de 2 líneas.")
            return False
        
        # Verificar encabezado (primera línea)
        encabezado = lineas[0].strip()
        if encabezado.startswith('#'):
            encabezado = encabezado[1:].strip()
        
        # Verificar que el encabezado tenga al menos una columna
        columnas_encabezado = len(encabezado.split())
        if columnas_encabezado == 0:
            print(f"❌ Error: El encabezado de {archivo} no tiene columnas.")
            return False
        
        # Verificar que al menos una línea de datos tenga el mismo número de columnas
        for i, linea in enumerate(lineas[1:], start=2):
            if linea.strip() and not linea.startswith('#'):
                columnas_datos = len(linea.strip().split())
                if columnas_datos != columnas_encabezado:
                    print(f"❌ Error: Línea {i} tiene {columnas_datos} columnas, pero el encabezado tiene {columnas_encabezado}.")
                    return False
                break
        
        print(f"✅ {archivo} verificado: {len(lineas)} líneas, {columnas_encabezado} columnas.")
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar {archivo}: {e}")
        return False

def limpiar_archivo(archivo_entrada, archivo_salida):
    """Limpia el archivo de datos quitando # del encabezado."""
    try:
        with open(archivo_entrada, 'r') as f_in, open(archivo_salida, 'w') as f_out:
            for linea in f_in:
                if linea.startswith('#'):
                    f_out.write(linea[1:].rstrip() + '\n')
                else:
                    f_out.write(linea.rstrip() + '\n')
        
        # Verificar que el archivo de salida no esté vacío
        with open(archivo_salida, 'r') as f:
            lineas = f.readlines()
        
        if len(lineas) < 2:
            print(f"❌ Error: {archivo_salida} tiene menos de 2 líneas después de limpiar.")
            return False
        
        print(f"✅ {archivo_salida} creado exitosamente.")
        return True
        
    except Exception as e:
        print(f"❌ Error al limpiar el archivo: {e}")
        return False

def crear_configuracion_bc03(archivo_datos):
    """Crea el archivo de configuración pcigale.ini con módulo BC03."""
    
    # PARÁMETROS BC03:
    # - Edades finas (16 valores entre 1-13 Gyr)
    # - Metalicidades: Z = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 (6 valores)
    # - IMF: Chabrier (2003)
    # - SFH delayed con tau variable
    # - Módulo de polvo y emisión IR estándar
    
    config = f"""# =====================================================================
# CIGALE v2022.1 — Configuración con BC03 para comparación con XSL
# Proyecto: Photometric Analysis of Globular Clusters in NGC 5128 (S-PLUS)
# Autor: Luis A. Gutiérrez Soto
# Configuración BC03 - Generada: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# =====================================================================
#
# PARÁMETROS CIENTÍFICOS (BC03):
# • Edades: 16 valores entre 1-13 Gyr (rejilla fina)
# • Metalicidades: Z = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
#   ([Fe/H] ≈ -2.3 a +0.4, cubriendo mejor el rango alto)
# • IMF: Chabrier (2003)
# • Redshift: from_file
# • Polvo: atenuación modificada de Calzetti con bajo E(B-V)
#
# FILTROS: 7 S-PLUS (estrechos) + 5 DECam/SDSS (anchos) = 12 filtros
# S-PLUS: F0378, F0395, F0410, F0430, F0515, F0660, F0861
# DECam/SDSS: u(sdss.up), g(sdss.gp), r(sdss.rp), i(sdss.ip), z(sdss.zp)
#
# NOTA: Usa {archivo_datos} (versión limpia)
# =====================================================================

data_file = {archivo_datos}
parameters_file = 
redshift = from_file
sed_modules = sfhdelayed, bc03, dustatt_modified_starburst, dale2014, redshifting
analysis_method = pdf_analysis
cores = 4

# Bands to consider. To consider uncertainties too, the name of the band
# must be indicated with the _err suffix.
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err, sdss.up, sdss.up_err, sdss.gp, sdss.gp_err, sdss.rp, sdss.rp_err, sdss.ip, sdss.ip_err, sdss.zp, sdss.zp_err

properties = 

additionalerror = 0.05


# Configuration of the SED creation modules.
[sed_modules_params]
  
  [[sfhdelayed]]
    tau_main = 0.1, 1, 2, 3, 5, 50, 100
    # Edades en Myr (misma rejilla fina que en XSL)
    age_main = 1000, 2000, 3000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000
    tau_burst = 1
    age_burst = 1
    f_burst = 0.0
    sfr_A = 1000000
    normalise = True
  
  [[bc03]]
    # Initial mass function: 0 (Salpeter) or 1 (Chabrier)
    imf = 1
    # Metallicity (Z). BC03 accepts: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05
    # Age [Myr] of the separation between the young and the old star
    # populations. The default value is 10 Myr. Set to 0 not to
    # differentiate ages (only an old population).
    separation_age = 10
    
  [[dustatt_modified_starburst]]
    E_BV_lines = 0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
    E_BV_factor = 0.25, 0.5, 0.75
    uv_bump_wavelength = 217.5
    uv_bump_width = 35.0
    uv_bump_amplitude = 0.0
    powerlaw_slope = -1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0
    Ext_law_emission_lines = 1
    Rv = 3.1
    filters = F0378 & F0395 & F0410 & F0430 & F0515 & F0660 & F0861 & sdss.up & sdss.gp & sdss.rp & sdss.ip & sdss.zp
    
  [[dale2014]]
    fracAGN = 0.0
    alpha = 2.0
  
  [[redshifting]]
    redshift = 


# Configuration of the statistical analysis method.
[analysis_params]
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861, sdss.up, sdss.gp, sdss.rp, sdss.ip, sdss.zp
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = True
  redshift_decimals = -2
  blocks = 1

# =====================================================================
# RESUMEN DE LA CONFIGURACIÓN BC03:
# • Modelos totales: 16 (edad) × 6 (metal) × 7 (τ) = 672 modelos
# • Rejilla fina en edad: 16 valores entre 1-13 Gyr
# • Metalicidades BC03: 6 valores (incluye Z=0.0001 y Z=0.05)
# • Filtros: 12 (7 S-PLUS + 5 DECam/SDSS)
# =====================================================================
"""
    
    try:
        with open("pcigale.ini", 'w') as f:
            f.write(config)
        print("✅ pcigale.ini con BC03 creado exitosamente.")
        return True
    except Exception as e:
        print(f"❌ Error al crear pcigale.ini: {e}")
        return False

def ejecutar_cigale_patron_confiable():
    """Ejecuta CIGALE con el patrón confiable (run → genconf → run)."""
    
    print("\n🚀 EJECUTAR CIGALE CON PATRÓN CONFIABLE:")
    print("-" * 40)
    print("  1. Primera ejecución de 'run' (puede fallar)")
    print("  2. Ejecutar 'genconf' (inicializa)")
    print("  3. Segunda ejecución de 'run' (debe funcionar)")
    print("-" * 40)
    
    respuesta = input("¿Ejecutar análisis completo ahora? (s/n): ").strip().lower()
    
    if respuesta != 's':
        print("\n📝 Puedes ejecutar manualmente más tarde con:")
        print("   pcigale run")
        return False
    
    print("\n🔮 EJECUTANDO PATRÓN CONFIABLE...")
    
    # Paso 1: Intento inicial
    print("\n1️⃣  Intento 1: Ejecutando pcigale run...")
    try:
        proceso = subprocess.Popen(['pcigale', 'run'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        time.sleep(10)
        
        if proceso.poll() is None:
            print("   ⏳ El proceso sigue corriendo...")
            print(f"   PID: {proceso.pid}")
            opcion = input("\n¿Continuar esperando (c) o seguir con patrón (p)? (c/p): ").strip().lower()
            if opcion == 'c':
                stdout, stderr = proceso.communicate()
                if proceso.returncode == 0:
                    print("✅ ¡pcigale run funcionó a la primera!")
                    return True
                else:
                    print(f"❌ pcigale run falló (código: {proceso.returncode})")
            else:
                proceso.terminate()
                print("   ⏹️  Proceso detenido, continuando con patrón...")
        else:
            stdout, stderr = proceso.communicate()
            print("   ❌ pcigale run falló rápidamente")
    except Exception as e:
        print(f"❌ Error en primera ejecución: {e}")
    
    # Paso 2: genconf
    print("\n2️⃣  Ejecutando pcigale genconf (inicialización)...")
    if os.path.exists("pcigale.ini"):
        shutil.copy2("pcigale.ini", "pcigale.ini.backup")
        print("   📦 Backup de configuración creado")
    try:
        resultado = subprocess.run(['pcigale', 'genconf'], 
                                 capture_output=True, 
                                 text=True)
        if os.path.exists("pcigale.ini.backup"):
            shutil.move("pcigale.ini.backup", "pcigale.ini")
            print("   ✅ Configuración original restaurada")
        if resultado.returncode == 0:
            print("   ✅ genconf ejecutado correctamente")
        else:
            print(f"   ⚠️  genconf devolvió código {resultado.returncode}")
    except Exception as e:
        print(f"❌ Error ejecutando genconf: {e}")
    
    # Paso 3: Segunda ejecución
    print("\n3️⃣  Intento 2: Ejecutando pcigale run (debería funcionar)...")
    print("   ⏳ Esto puede tomar varias horas...")
    print("   📊 Progreso: tail -f out/log.txt")
    try:
        proceso = subprocess.Popen(['pcigale', 'run'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 universal_newlines=True)
        print(f"   PID: {proceso.pid}")
        print("-" * 40)
        for linea in iter(proceso.stdout.readline, ''):
            linea = linea.strip()
            if linea and "pkg_resources" not in linea:
                print(f"   {linea}")
        proceso.wait()
        if proceso.returncode == 0:
            print("\n✅ ¡pcigale run completado exitosamente!")
            return True
        else:
            print(f"\n❌ pcigale run falló (código: {proceso.returncode})")
            return False
    except KeyboardInterrupt:
        print("\n\n⏹️  Ejecución interrumpida por el usuario")
        if 'proceso' in locals():
            proceso.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error en segunda ejecución: {e}")
        return False

def main():
    print("=" * 70)
    print("🚀 CONFIGURADOR CIGALE BC03 - REJILLA FINA + TODOS FILTROS")
    print("=" * 70)
    print("Usa modelos BC03 en lugar de XSL para comparación.")
    print("=" * 70)
    
    # Archivos
    archivo_original = "gc_splus_cigale_custom.txt"
    archivo_limpio = "gc_splus_cigale_fixed.txt"
    
    # Paso 1: Verificar archivo original
    print("\n1️⃣  VERIFICANDO ARCHIVO DE DATOS ORIGINAL")
    print("-" * 40)
    if not verificar_archivo_datos(archivo_original):
        print("⚠️  Problemas con el archivo de datos. ¿Continuar? (s/n): ", end="")
        respuesta = input().strip().lower()
        if respuesta != 's':
            print("❌ No se puede continuar por problemas en el archivo de datos.")
            sys.exit(1)
    
    # Paso 2: Crear versión limpia
    print("\n2️⃣  CREANDO VERSIÓN LIMPIA DEL ARCHIVO DE DATOS")
    print("-" * 40)
    if not limpiar_archivo(archivo_original, archivo_limpio):
        print("⚠️  Error al crear la versión limpia. ¿Usar archivo existente? (s/n): ", end="")
        respuesta = input().strip().lower()
        if respuesta != 's':
            print("❌ Necesitamos el archivo limpio para continuar.")
            sys.exit(1)
        elif not os.path.exists(archivo_limpio):
            print(f"❌ {archivo_limpio} no existe.")
            sys.exit(1)
    
    # Paso 3: Crear configuración BC03
    print("\n3️⃣  CREANDO CONFIGURACIÓN BC03")
    print("-" * 40)
    if not crear_configuracion_bc03(archivo_limpio):
        print("❌ Error al crear la configuración.")
        sys.exit(1)
    
    # Paso 4: Resumen
    print("\n4️⃣  RESUMEN DE LA CONFIGURACIÓN BC03")
    print("-" * 40)
    print(f"📁 Archivos creados:")
    print(f"   1. {archivo_limpio} (datos limpios)")
    print(f"   2. pcigale.ini (configuración BC03)")
    
    print(f"\n🔧 Configuración científica BC03:")
    print(f"   • Modelos: 672 (16 edades × 6 metalicidades × 7 tau)")
    print(f"   • Edades: 16 valores entre 1-13 Gyr")
    print(f"   • Metalicidades: Z = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05")
    print(f"   • Filtros: 12 (7 S-PLUS estrechos + 5 DECam/SDSS anchos)")
    print(f"   • Ventaja: BC03 tiene mayor cobertura metálica y es estándar")
    
    print(f"\n🔄 Patrón de ejecución:")
    print(f"   1. Intento 1: pcigale run (puede fallar)")
    print(f"   2. Ejecutar: pcigale genconf")
    print(f"   3. Intento 2: pcigale run (debe funcionar)")
    print(f"   ⏱️  Tiempo estimado: similar al de XSL")
    
    # Paso 5: Ejecutar CIGALE con patrón confiable
    ejecutar_cigale_patron_confiable()
    
    print("\n" + "=" * 70)
    print("🎉 ¡PROCESO COMPLETADO!")
    print("=" * 70)
    
    print("\n📝 Comandos útiles después del análisis:")
    print("   pcigale-plots sed         # Generar gráficos SED")
    print("   pcigale-plots pdf         # Generar gráficos PDF")
    print("   tail -f out/log.txt       # Monitorear progreso")
    print("   ls -lh out/              # Ver archivos generados")
    
    print("\n🔍 Para comparar con XSL:")
    print("   python ../Programs/compare_cigale_bc03_xsl.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
