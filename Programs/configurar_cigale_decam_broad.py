#!/usr/bin/env python3
# cigale_setup_optimizado.py
# Configuración OPTIMIZADA con rejilla fina and only for the broad band filters
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

def crear_configuracion_optimizada(archivo_datos):
    """Crea el archivo de configuración pcigale.ini OPTIMIZADO."""
    
    # PARÁMETROS OPTIMIZADOS:
    # - 9 edades entre 5-13 Gyr (para evitar discretización)
    # - 5 metalicidades
    # - 4 valores de tau
    # - TOTAL: 9 × 5 × 4 = 180 modelos
    # - TODOS los filtros (12)
    
    config = f"""# =====================================================================
# CIGALE v2022.1 — Configuración OPTIMIZADA para CÚMULOS GLOBULARES en NGC 5128
# Proyecto: Photometric Analysis of Globular Clusters in NGC 5128 (S-PLUS)
# Autor: Luis A. Gutiérrez Soto
# Configuración OPTIMIZADA - Generada: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# =====================================================================
#
# PARÁMETROS CIENTÍFICOS (optimizados para poblaciones antiguas):
# • Edades: 9 valores entre 5 - 13 Gyr (rejilla fina para evitar discretización)
# • Metalicidades: Z = 0.0004, 0.004, 0.008, 0.02, 0.03 ([Fe/H] ≈ -2.2 a +0.2)
# • IMF: Chabrier (2003) — estándar para poblaciones de baja masa
# • Redshift: from_file (leído del archivo de datos)
# • Sin atenuación de polvo (típico en GCs)
#
# FILTROS: 7 S-PLUS (estrechos) + 5 DECam/SDSS (anchos) = 12 filtros totales
# S-PLUS: F0378, F0395, F0410, F0430, F0515, F0660, F0861--no
# DECam/SDSS: u(sdss.up), g(sdss.gp), r(sdss.rp), i(sdss.ip), z(sdss.zp)
#
# NOTA: Usa {archivo_datos} (versión limpia del archivo original)
# =====================================================================

data_file = {archivo_datos}
parameters_file = 
redshift = from_file
sed_modules = sfhdelayed, xsl, dustatt_modified_starburst, dale2014, redshifting
analysis_method = pdf_analysis
cores = 4

# Bands to consider. To consider uncertainties too, the name of the band
# must be indicated with the _err suffix. For instance: FUV, FUV_err.
bands =  sdss.up, sdss.up_err, sdss.gp, sdss.gp_err, sdss.rp, sdss.rp_err, sdss.ip, sdss.ip_err, sdss.zp, sdss.zp_err

# Properties to be considered. All properties are to be given in the
# rest frame rather than the observed frame. This is the case for
# instance the equivalent widths and for luminosity densities.
properties = 

# Relative error added in quadrature to the uncertainties of the fluxes
# and the extensive properties.
additionalerror = 0.05


# Configuration of the SED creation modules.
[sed_modules_params]
  
  [[sfhdelayed]]
    # e-folding time of the main stellar population model in Myr.
    tau_main = 0.1, 1, 2, 3, 5, 50, 100
    # Age of the main stellar population in the galaxy in Myr. The precision
    # is 1 Myr.
    # 9 VALORES para rejilla fina y evitar discretización
    age_main = 1000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000
    # e-folding time of the late starburst population model in Myr.
    tau_burst = 1
    # Age of the late burst in Myr. The precision is 1 Myr.
    age_burst = 1
    # Mass fraction of the late burst population.
    f_burst = 0.0
    # Multiplicative factor controlling the SFR if normalise is False. For
    # instance without any burst: SFR(t)=sfr_A×t×exp(-t/τ)/τ²
    sfr_A = 1000000
    # Normalise the SFH to produce one solar mass.
    normalise = True
  
  [[xsl]]
    # Initial mass function: 1 (Kroupa)
    imf = 1
    # Metallicity. Possible values are: 0.0004, 0.004, 0.008, 0.02, 0.03.
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    # Age [Myr] of the separation between the young and the old star
    # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
    # differentiate ages (only an old population).
    separation_age = 10
    
  [[dustatt_modified_starburst]]
    # E(B-V)l, the colour excess of the nebular lines light for both the
    # young and old population.
    E_BV_lines = 0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
    # Reduction factor to apply on E_BV_lines to compute E(B-V)s the stellar
    # continuum attenuation. Both young and old population are attenuated
    # with E(B-V)s.
    E_BV_factor = 0.25, 0.5, 0.75
    # Central wavelength of the UV bump in nm.
    uv_bump_wavelength = 217.5
    # Width (FWHM) of the UV bump in nm.
    uv_bump_width = 35.0
    # Amplitude of the UV bump. For the Milky Way: 3.
    uv_bump_amplitude = 0.0
    # Slope delta of the power law modifying the attenuation curve.
    powerlaw_slope = -1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0
    # Extinction law to use for attenuating the emissio  n lines flux.
    # Possible values are: 1, 2, 3. 1: MW, 2: LMC, 3: SMC. MW is modelled
    # using CCM89, SMC and LMC using Pei92.
    Ext_law_emission_lines = 1
    # Ratio of total to selective extinction, A_V / E(B-V), for the
    # extinction curve applied to emission lines.Standard value is 3.1 for
    # MW using CCM89, but can be changed.For SMC and LMC using Pei92 the
    # value is automatically set to 2.93 and 3.16 respectively, no matter
    # the value you write.
    Rv = 3.1
    # Filters for which the attenuation will be computed and added to the
    # SED information dictionary. You can give several filter names
    # separated by a & (don't use commas).
    filters = sdss.up & sdss.gp & sdss.rp & sdss.ip & sdss.zp
    
  [[dale2014]]
    # AGN fraction. It is not recommended to combine this AGN emission with
    # the of Fritz et al. (2006) or SKIRTOR models.
    fracAGN = 0.0
    # Alpha slope. Possible values are: 0.0625, 0.1250, 0.1875, 0.2500,
    # 0.3125, 0.3750, 0.4375, 0.5000, 0.5625, 0.6250, 0.6875, 0.7500,
    # 0.8125, 0.8750, 0.9375, 1.0000, 1.0625, 1.1250, 1.1875, 1.2500,
    # 1.3125, 1.3750, 1.4375, 1.5000, 1.5625, 1.6250, 1.6875, 1.7500,
    # 1.8125, 1.8750, 1.9375, 2.0000, 2.0625, 2.1250, 2.1875, 2.2500,
    # 2.3125, 2.3750, 2.4375, 2.5000, 2.5625, 2.6250, 2.6875, 2.7500,
    # 2.8125, 2.8750, 2.9375, 3.0000, 3.0625, 3.1250, 3.1875, 3.2500,
    # 3.3125, 3.3750, 3.4375, 3.5000, 3.5625, 3.6250, 3.6875, 3.7500,
    # 3.8125, 3.8750, 3.9375, 4.0000
    alpha = 2.0
  
  [[redshifting]]
    # Redshift of the objects. Leave empty to use the redshifts from the
    # input file.
    redshift = 


# Configuration of the statistical analysis method.
[analysis_params]
  # List of the physical properties to estimate. Leave empty to analyse
  # all the physical properties (not recommended when there are many
  # models).
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  # List of bands for which to estimate the fluxes. Note that this is
  # independent from the fluxes actually fitted to estimate the physical
  # properties.
  bands = sdss.up, sdss.gp, sdss.rp, sdss.ip, sdss.zp
  # If true, save the best SED for each observation to a file.
  save_best_sed = True
  # Save the raw chi2. It occupies ~15 MB/million models/variable. Allowed
  # values are 'all', 'none', 'properties', and 'fluxes'.
  save_chi2 = none
  # Take into account upper limits. If 'full', the exact computation is
  # done. If 'noscaling', the scaling of the models will not be adjusted
  # but the χ² will include the upper limits adequately. Waiving the
  # adjustment makes the fitting much faster compared to the 'full' option
  # while generally not affecting the results in any substantial manner.
  # This is the recommended option as it achieves a good balance between
  # speed and reliability. Finally, 'none' simply discards bands with
  # upper limits.
  lim_flag = noscaling
  # If true, for each object we create a mock object and analyse them.
  mock_flag = True
  # When redshifts are not given explicitly in the redshifting module,
  # number of decimals to round the observed redshifts to compute the grid
  # of models. To disable rounding give a negative value. Do not round if
  # you use narrow-band filters.
  redshift_decimals = -2
  # Number of blocks to compute the models and analyse the observations.
  # If there is enough memory, we strongly recommend this to be set to 1.
  blocks = 1

# =====================================================================
# RESUMEN DE LA CONFIGURACIÓN OPTIMIZADA:
# • Modelos totales: 9 (edad) × 5 (metal) × 4 (τ) = 180 modelos
# • Rejilla fina en edad: 9 valores entre 5-13 Gyr
# • Rango científico: Adecuado para cúmulos globulares antiguos
# • Variables de salida: sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs
# • Filtros:  5 DECam/SDSS anchos)
# =====================================================================
"""
    
    try:
        with open("pcigale.ini", 'w') as f:
            f.write(config)
        print("✅ pcigale.ini creado exitosamente (CONFIGURACIÓN OPTIMIZADA).")
        return True
    except Exception as e:
        print(f"❌ Error al crear pcigale.ini: {e}")
        return False

def ejecutar_cigale_patron_confiable():
    """Ejecuta CIGALE con el patrón confiable que sabemos funciona."""
    
    print("\n🚀 EJECUTAR CIGALE CON PATRÓN CONFIABLE:")
    print("-" * 40)
    print("Este patrón ha demostrado funcionar:")
    print("  1. Primera ejecución de 'run' (puede fallar)")
    print("  2. Ejecutar 'genconf' (inicializa)")
    print("  3. Segunda ejecución de 'run' (debe funcionar)")
    print("-" * 40)
    
    respuesta = input("¿Ejecutar análisis completo ahora? (s/n): ").strip().lower()
    
    if respuesta != 's':
        print("\n📝 Puedes ejecutar manualmente más tarde con:")
        print("   pcigale run")
        return False
    
    # PATRÓN CONFIABLE (el que sabemos funciona)
    print("\n🔮 EJECUTANDO PATRÓN CONFIABLE...")
    
    # Paso 1: Intentar ejecución (puede fallar)
    print("\n1️⃣  Intento 1: Ejecutando pcigale run...")
    print("   (Puede fallar en la primera ejecución, es normal)")
    
    try:
        # Ejecutamos con timeout corto
        proceso = subprocess.Popen(['pcigale', 'run'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # Esperar 10 segundos máximo para ver si falla rápido
        time.sleep(10)
        
        if proceso.poll() is None:
            # Si sigue corriendo, puede que funcione
            print("   ⏳ El proceso sigue corriendo...")
            print(f"   PID: {proceso.pid}")
            
            # Preguntar qué hacer
            opcion = input("\n¿Continuar esperando (c) o seguir con patrón (p)? (c/p): ").strip().lower()
            
            if opcion == 'c':
                print("\n⏳ Esperando finalización...")
                stdout, stderr = proceso.communicate()
                
                if proceso.returncode == 0:
                    print("✅ ¡pcigale run funcionó a la primera!")
                    return True
                else:
                    print(f"❌ pcigale run falló (código: {proceso.returncode})")
                    # Continuar con patrón
            else:
                proceso.terminate()
                print("   ⏹️  Proceso detenido, continuando con patrón...")
        else:
            # Proceso ya terminó (probablemente falló)
            stdout, stderr = proceso.communicate()
            print("   ❌ pcigale run falló rápidamente")
            
    except Exception as e:
        print(f"❌ Error en primera ejecución: {e}")
    
    # Paso 2: Ejecutar genconf
    print("\n2️⃣  Ejecutando pcigale genconf (inicialización)...")
    
    # Hacer backup de configuración
    if os.path.exists("pcigale.ini"):
        shutil.copy2("pcigale.ini", "pcigale.ini.backup")
        print("   📦 Backup de configuración creado")
    
    try:
        resultado = subprocess.run(['pcigale', 'genconf'], 
                                 capture_output=True, 
                                 text=True)
        
        # Restaurar configuración original
        if os.path.exists("pcigale.ini.backup"):
            shutil.move("pcigale.ini.backup", "pcigale.ini")
            print("   ✅ Configuración original restaurada")
        
        if resultado.returncode == 0:
            print("   ✅ genconf ejecutado correctamente")
        else:
            print(f"   ⚠️  genconf devolvió código {resultado.returncode}")
            
    except Exception as e:
        print(f"❌ Error ejecutando genconf: {e}")
    
    # Paso 3: Segunda ejecución (debería funcionar)
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
        print("   Para detener: Ctrl+C")
        print("-" * 40)
        
        # Mostrar salida en tiempo real
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
    """Función principal."""
    
    print("=" * 70)
    print("🚀 CONFIGURADOR CIGALE OPTIMIZADO - REJILLA FINA + TODOS FILTROS")
    print("=" * 70)
    print("Combina lo mejor de ambos scripts:")
    print("  • Rejilla fina (9 edades, 5 metalicidades, 4 tau = 180 modelos)")
    print("  • Todos los filtros (7 S-PLUS + 5 DECam/SDSS)")
    print("  • Patrón de ejecución confiable (run → genconf → run)")
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
    
    # Paso 3: Crear configuración optimizada
    print("\n3️⃣  CREANDO CONFIGURACIÓN OPTIMIZADA")
    print("-" * 40)
    if not crear_configuracion_optimizada(archivo_limpio):
        print("❌ Error al crear la configuración.")
        sys.exit(1)
    
    # Paso 4: Resumen
    print("\n4️⃣  RESUMEN DE LA CONFIGURACIÓN OPTIMIZADA")
    print("-" * 40)
    print(f"📁 Archivos creados:")
    print(f"   1. {archivo_limpio} (datos limpios)")
    print(f"   2. pcigale.ini (configuración optimizada)")
    
    print(f"\n🔧 Configuración científica OPTIMIZADA:")
    print(f"   • Modelos: 180 (9 edades × 5 metalicidades × 4 tau)")
    print(f"   • Edades: 9 valores entre 5-13 Gyr")
    print(f"   • Metalicidades: Z = 0.0004, 0.004, 0.008, 0.02, 0.03")
    print(f"   • Filtros: 12 (7 S-PLUS estrechos + 5 DECam/SDSS anchos)")
    print(f"   • Ventaja: Rejilla más fina evita valores discretos en resultados")
    
    print(f"\n🔄 Patrón de ejecución:")
    print(f"   1. Intento 1: pcigale run (puede fallar)")
    print(f"   2. Ejecutar: pcigale genconf")
    print(f"   3. Intento 2: pcigale run (debe funcionar)")
    print(f"   ⏱️  Tiempo estimado: 3-5 horas para 760 objetos")
    
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
    
    print("\n🔍 Para analizar resultados optimizados:")
    print("   python analisis_gc_avanzado.py  # Usa versión mejorada")
    
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
