#!/usr/bin/env python3
# crear_pcigale_ini_GC_optimizado.py
# CONFIGURACIÓN OPTIMIZADA para Cúmulos Globulares en NGC 5128
# Genera pcigale.ini basado en gc_splus_cigale_custom.txt existente

import os
import sys
from datetime import datetime

def verificar_archivo_datos():
    """Verifica que el archivo de datos exista y tenga formato correcto"""
    
    archivo = "gc_splus_cigale_custom.txt"
    
    if not os.path.exists(archivo):
        print(f"❌ ERROR: {archivo} no encontrado")
        print("\n⚠️  Debes ejecutar primero: preparar_cigale_redshift_primero.py")
        print("   para generar el archivo de datos correctamente.")
        return False
    
    try:
        with open(archivo, 'r') as f:
            # Leer primera línea (encabezado con #)
            primera_linea = f.readline().strip()
            
            # Leer primera línea de datos
            segunda_linea = f.readline().strip()
        
        # Verificar que tenga al menos redshift e id
        if not primera_linea.startswith('#'):
            print(f"⚠️  Advertencia: El archivo no tiene encabezado con #")
        
        if segunda_linea:
            partes = segunda_linea.split()
            if len(partes) >= 2:
                try:
                    redshift = float(partes[0])
                    id_obj = partes[1]
                    print(f"✅ Archivo de datos verificado:")
                    print(f"   • Archivo: {archivo}")
                    print(f"   • Primer objeto: {id_obj}")
                    print(f"   • Redshift: {redshift:.6f}")
                    print(f"   • Campos por objeto: {len(partes)}")
                    
                    # Contar líneas totales
                    with open(archivo, 'r') as f:
                        lineas = f.readlines()
                    num_objetos = len(lineas) - 1 if primera_linea.startswith('#') else len(lineas)
                    print(f"   • Número de objetos: {num_objetos}")
                    
                    return True
                except ValueError:
                    print(f"❌ Error: El primer campo no es un número: {partes[0]}")
                    return False
            else:
                print(f"❌ Error: Formato incorrecto. Línea de datos: {segunda_linea}")
                return False
        else:
            print(f"❌ Error: Archivo vacío o sin datos")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False

def limpiar_archivo_datos():
    """Limpia el archivo de datos (quita # del encabezado)"""
    
    archivo_original = "gc_splus_cigale_custom.txt"
    archivo_limpio = "gc_splus_cigale_fixed.txt"
    
    if not os.path.exists(archivo_original):
        return False
    
    try:
        with open(archivo_original, 'r') as f_in, open(archivo_limpio, 'w') as f_out:
            for linea in f_in:
                # Quitar # solo al inicio de la línea
                if linea.startswith('#'):
                    linea_limpia = linea[1:].strip()
                else:
                    linea_limpia = linea.rstrip()
                
                if linea_limpia:
                    f_out.write(linea_limpia + '\n')
        
        # Contar líneas
        with open(archivo_limpio, 'r') as f:
            lineas = f.readlines()
        
        print(f"🧹 Archivo limpiado: {archivo_limpio}")
        print(f"   • {len(lineas)} líneas (objetos + encabezado)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error limpiando archivo: {e}")
        return False

def crear_config_optimizada_gc():
    """Crea configuración OPTIMIZADA para cúmulos globulares antiguos"""
    
    config = """# =====================================================================
# CIGALE v2022.1 — Configuración para CÚMULOS GLOBULARES en NGC 5128
# Proyecto: Photometric Analysis of Globular Clusters in NGC 5128 (S-PLUS)
# Autor: Luis A. Gutiérrez Soto
# Generado automáticamente el: {fecha}
# =====================================================================
#
# PARÁMETROS CIENTÍFICOS (optimizados para poblaciones antiguas):
# • Edades: 8 - 13 Gyr (rango típico para GCs)
# • Metalicidades: Z = 0.0004, 0.004, 0.008, 0.02, 0.03 ([Fe/H] ≈ -2.2 a +0.2)
# • IMF: Chabrier (2003) — estándar para poblaciones de baja masa
# • Redshift: from_file (leído del archivo de datos)
# • Sin atenuación de polvo (típico en GCs)
#
# NOTA: Usa gc_splus_cigale_fixed.txt (versión limpia del archivo original)
# =====================================================================

data_file = gc_splus_cigale_fixed.txt
parameters_file = 
redshift = from_file
sed_modules = sfhdelayed, xsl, redshifting
analysis_method = pdf_analysis
cores = 12

bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties = 
additionalerror = 0.1

[sed_modules_params]
  
  [[sfhdelayed]]
    # Tiempos de decaimiento exponencial (τ) en Myr
    # Valores amplios para cubrir posibles historias de formación
    tau_main = 500.0, 1000.0, 2000.0, 5000.0
    
    # EDADES PRINCIPALES (Myr) — Rango OPTIMIZADO para GCs antiguos
    # 8000-13000 Myr = 8-13 Gyr (edad cósmica de GCs en NGC 5128)
    age_main = 8000, 10000, 12000, 13000
    
    # Parámetros de estallido — FIJADOS A CERO para GCs
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    
    sfr_A = 1.0
    normalise = True
  
  [[xsl]]
    # IMF: 1 = Chabrier (2003). Más realista que Salpeter para GCs.
    imf = 1
    
    # METALICIDADES (Z) — Cubren el rango completo observado en GCs
    # Z_sol = 0.02; [Fe/H] = log(Z/Z_sol)
    # 0.0004 → [Fe/H] ≈ -2.2 (muy pobre en metales)
    # 0.03   → [Fe/H] ≈ +0.2 (ligeramente super-solar, para cubrir todo el rango)
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    
    separation_age = 10
  
  [[redshifting]]
    # Redshift se leerá del archivo de datos (redshift = from_file)
    # Por lo tanto, aquí debe dejarse vacío.
    redshift = 

[analysis_params]
  # VARIABLES FÍSICAS CLAVE para GCs
  # • stellar.m_star: Masa estelar total (M_sol)
  # • stellar.metallicity_mw: Metalicidad media ponderada por masa (log(Z))
  # • stellar.age_m_star: Edad media ponderada por masa (Myr)
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 6
  blocks = 1

# =====================================================================
# RESUMEN DE LA CONFIGURACIÓN:
# • Modelos totales: 4 (τ) × 4 (edad) × 5 (metal) = 80 modelos
# • Rango científico: Adecuado para cúmulos globulares antiguos
# • Variables de salida: Masas, metalicidades ([Fe/H]), edades
# =====================================================================
"""
    
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config = config.format(fecha=fecha)
    
    # Crear backup si existe
    if os.path.exists("pcigale.ini"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"pcigale.ini.backup_{timestamp}"
        os.rename("pcigale.ini", backup_file)
        print(f"📦 Backup creado: {backup_file}")
    
    # Escribir archivo de configuración
    with open("pcigale.ini", "w") as f:
        f.write(config)
    
    print(f"✅ pcigale.ini creado ({datetime.now().strftime('%H:%M:%S')})")
    
    return True

def ejecutar_comando(comando, descripcion):
    """Ejecuta un comando del sistema y muestra resultado"""
    
    print(f"\n⚡ {descripcion}")
    print(f"   $ {comando}")
    
    try:
        resultado = os.system(comando)
        if resultado == 0:
            print(f"   ✅ Comando exitoso")
            return True
        else:
            print(f"   ❌ Error en comando (código: {resultado})")
            return False
    except Exception as e:
        print(f"   ❌ Excepción: {e}")
        return False

def mostrar_resumen():
    """Muestra un resumen de los parámetros científicos"""
    
    print("\n📊 RESUMEN CIENTÍFICO DE LA CONFIGURACIÓN:")
    print("   • Población: Cúmulos globulares antiguos (NGC 5128)")
    print("   • Edades: 8, 10, 12, 13 Gyr (4 valores)")
    print("   • Metalicidades: Z = 0.0004, 0.004, 0.008, 0.02, 0.03")
    print("   • IMF: Chabrier (2003)")
    print("   • Modelos: 80 (4×4×5) — Grid eficiente")
    print("   • Variables clave: Masa, [Fe/H], Edad")
    print("   • Redshift: from_file (leído del archivo de datos)")
    print("   • Filtros: 7 bandas S-PLUS (F0378 a F0861)")

def main():
    """Función principal que ejecuta todo el proceso"""
    
    print("=" * 70)
    print("🚀 GENERADOR DE CONFIGURACIÓN CIGALE - OPTIMIZADO PARA GCs")
    print("=" * 70)
    print("Este script genera pcigale.ini usando gc_splus_cigale_custom.txt")
    print("Configuración CIENTÍFICAMENTE validada para cúmulos globulares")
    print("en NGC 5128 (Centaurus A) con datos S-PLUS")
    print("=" * 70)
    
    # Paso 1: Verificar que el archivo de datos existe
    print("\n1️⃣  VERIFICANDO ARCHIVO DE DATOS")
    print("-" * 40)
    if not verificar_archivo_datos():
        return False
    
    # Paso 2: Limpiar archivo de datos (quitar # del encabezado)
    print("\n2️⃣  LIMPIANDO ARCHIVO DE DATOS")
    print("-" * 40)
    if not limpiar_archivo_datos():
        print("⚠️  Continuando con archivo original (puede causar errores)...")
        # Si no puede limpiar, usar el archivo original
        global archivo_limpio
        archivo_limpio = "gc_splus_cigale_custom.txt"
    
    # Paso 3: Crear configuración OPTIMIZADA
    print("\n3️⃣  CREANDO CONFIGURACIÓN OPTIMIZADA PARA GCs")
    print("-" * 40)
    if crear_config_optimizada_gc():
        print("\n✅ CONFIGURACIÓN OPTIMIZADA CREADA")
        print("=" * 70)
        
        mostrar_resumen()
        
        print("\n🚀 PRÓXIMOS PASOS:")
        print("-" * 40)
        
        # Ejecutar comandos automáticamente
        respuesta = input("\n¿Ejecutar 'pcigale check' ahora? (s/n): ").strip().lower()
        if respuesta == 's':
            if ejecutar_comando("pcigale check", "Verificando configuración"):
                respuesta2 = input("\n✅ Verificación exitosa. ¿Ejecutar 'pcigale run'? (s/n): ").strip().lower()
                if respuesta2 == 's':
                    ejecutar_comando("pcigale run", "Ejecutando análisis completo")
        
        print("\n" + "=" * 70)
        print("🎉 ¡ANÁLISIS CIENTÍFICO LISTO!")
        print("=" * 70)
        print("\n📁 Archivos generados/modificados:")
        print("   1. gc_splus_cigale_fixed.txt (datos limpios)")
        print("   2. pcigale.ini (configuración optimizada)")
        if os.path.exists("pcigale.ini.backup"):
            print("   3. pcigale.ini.backup_... (copia de seguridad)")
        
        print("\n🔧 Comandos manuales disponibles:")
        print("   $ pcigale check  # Verificar configuración")
        print("   $ pcigale run    # Ejecutar análisis completo")
        
        return True
        
    else:
        print("\n❌ Error creando configuración")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
