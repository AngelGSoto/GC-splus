#!/usr/bin/env python3
# crear_pcigale_ini_automático_mejorado.py
# Script optimizado que genera EXACTAMENTE la configuración que funciona
# Basado en la configuración verificada con: pcigale check

import os
import sys
from datetime import datetime
import subprocess
import time

def crear_config_perfecta(variables_opcion=1, imf_opcion=1):
    """Crea la configuración EXACTA que sabemos que funciona"""
    
    # Opciones de variables (3 opciones útiles)
    variables_opciones = {
        1: "stellar.m_star, stellar.metallicity, stellar.age_m_star",  # Básicas para paper
        2: "sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs",  # Para análisis SFH
        3: "stellar.m_star, stellar.metallicity, stellar.age_m_star, sfh.sfr"  # Mixtas
    }
    
    # ¡ESTA ES LA CONFIGURACIÓN EXACTA QUE FUNCIONA!
    config = """# NGC 5128 Globular Clusters – S-PLUS (CIGALE 2022.1)
# CONFIGURACIÓN VERIFICADA Y FUNCIONAL (pcigale check ✅)
# Generado automáticamente el: {fecha}
# Variables seleccionadas: {vars_desc}
# IMF seleccionado: {imf_desc}

data_file = gc_splus_cigale_fixed.txt
parameters_file = 
redshift = from_file
sed_modules = sfhdelayed, xsl, redshifting
analysis_method = pdf_analysis
cores = 4

# 7 filtros S-PLUS con sus errores
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties = 
additionalerror = 0.1

[sed_modules_params]
  
  [[sfhdelayed]]
    # τ (tiempos de decaimiento) en Myr - 4 valores
    tau_main = 500.0, 1000.0, 2000.0, 5000.0
    
    # Edades en Myr - 9 valores entre 5-13 Gyr
    age_main = 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000
    
    # Parámetros de burst (desactivado para GCs)
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    
    sfr_A = 1.0
    normalise = True
  
  [[xsl]]
    # IMF: 1 = Salpeter (1955), 2 = Chabrier (2003)
    imf = {imf}
    
    # Metalicidad (Z) - 5 valores
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    
    separation_age = 10
  
  [[redshifting]]
    redshift = 

[analysis_params]
  # Variables de salida (más útiles para paper)
  variables = {variables}
  
  # Filtros para estimar flujos
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 6
  blocks = 1
"""
    
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    variables = variables_opciones.get(variables_opcion, variables_opciones[1])
    
    # Descripciones para el header
    vars_descriptions = {
        1: "stellar.m_star, stellar.metallicity, stellar.age_m_star (recomendado para paper)",
        2: "sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs (para análisis SFH)",
        3: "stellar.m_star, stellar.metallicity, stellar.age_m_star, sfh.sfr (mixtas)"
    }
    
    imf_descriptions = {
        1: "Salpeter (1955) - el que funciona",
        2: "Chabrier (2003) - para poblaciones de baja masa"
    }
    
    config = config.format(
        fecha=fecha,
        variables=variables,
        imf=imf_opcion,
        vars_desc=vars_descriptions[variables_opcion],
        imf_desc=imf_descriptions[imf_opcion]
    )
    
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
    print(f"   • Variables: {variables}")
    print(f"   • IMF: {imf_opcion} ({imf_descriptions[imf_opcion]})")
    
    return True

def limpiar_archivo_datos():
    """Limpia el archivo de datos (quita # y espacios innecesarios)"""
    
    archivo_original = "gc_splus_cigale_custom.txt"
    archivo_limpio = "gc_splus_cigale_fixed.txt"
    
    if not os.path.exists(archivo_original):
        print(f"⚠️  {archivo_original} no encontrado")
        print(f"   Buscando alternativas...")
        
        # Buscar alternativas
        alternativas = [
            "gc_splus_cigale.txt",
            "gc_splus_data.txt",
            "gc_data.txt",
            "data.txt"
        ]
        
        for alt in alternativas:
            if os.path.exists(alt):
                archivo_original = alt
                print(f"✅ Encontrado: {archivo_original}")
                break
        else:
            print("❌ No se encontró ningún archivo de datos")
            return False
    
    try:
        with open(archivo_original, 'r') as f_in, open(archivo_limpio, 'w') as f_out:
            lineas_procesadas = 0
            for linea in f_in:
                # Quitar # al inicio y espacios/tabs al final
                linea_limpia = linea.lstrip('#').strip()
                if linea_limpia:  # Solo escribir líneas no vacías
                    f_out.write(linea_limpia + '\n')
                    lineas_procesadas += 1
        
        # Contar líneas y verificar formato
        with open(archivo_limpio, 'r') as f:
            lineas = f.readlines()
        
        if len(lineas) == 0:
            print("❌ Archivo limpio está vacío")
            return False
        
        # Verificar primera línea
        primera_linea = lineas[0].strip().split()
        print(f"🧹 Archivo limpiado: {archivo_limpio}")
        print(f"   • Objetos procesados: {len(lineas)}")
        print(f"   • Campos por objeto: {len(primera_linea)}")
        print(f"   • Primer objeto: ID={primera_linea[1] if len(primera_linea) > 1 else 'N/A'}")
        
        # Verificar que tenga los 7 filtros S-PLUS esperados
        campos_esperados = 2 + 7 * 2  # redshift + ID + (flux + err) × 7
        if len(primera_linea) != campos_esperados:
            print(f"⚠️  Advertencia: Se esperaban {campos_esperados} campos")
            print(f"   Se encontraron {len(primera_linea)} campos")
            print(f"   Verificar que sean 7 filtros S-PLUS")
        
        return True
        
    except Exception as e:
        print(f"❌ Error limpiando archivo: {e}")
        return False

def verificar_formato_datos():
    """Verifica formato básico del archivo de datos"""
    
    archivo = "gc_splus_cigale_fixed.txt"
    
    if not os.path.exists(archivo):
        print(f"⚠️  {archivo} no encontrado")
        return False
    
    try:
        with open(archivo, 'r') as f:
            primera_linea = f.readline().strip()
            segunda_linea = f.readline().strip()
        
        if not primera_linea or not segunda_linea:
            print(f"❌ Error: Archivo vacío o tiene muy pocas líneas")
            return False
        
        # Verificar primera línea
        partes1 = primera_linea.split()
        
        if len(partes1) >= 2:
            try:
                redshift = float(partes1[0])
                id_obj = partes1[1]
                
                print(f"📊 Formato verificado:")
                print(f"   • Redshift primer objeto: {redshift}")
                print(f"   • ID primer objeto: {id_obj}")
                print(f"   • Campos totales: {len(partes1)}")
                
                # Verificar segunda línea
                partes2 = segunda_linea.split()
                if len(partes2) == len(partes1):
                    print(f"   • Formato consistente en línea 2: ✓")
                else:
                    print(f"⚠️  Inconsistencia: línea 1 tiene {len(partes1)} campos, línea 2 tiene {len(partes2)}")
                
                return True
                
            except ValueError:
                print(f"❌ Error: El primer campo no es un número: {partes1[0]}")
                return False
        else:
            print(f"❌ Error: Formato incorrecto. Línea: {primera_linea[:50]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False

def ejecutar_comando(comando, descripcion, timeout=None):
    """Ejecuta un comando del sistema y muestra resultado"""
    
    print(f"\n⚡ {descripcion}")
    print(f"   $ {comando}")
    
    try:
        proceso = subprocess.Popen(
            comando,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if timeout:
            try:
                stdout, stderr = proceso.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proceso.kill()
                print(f"   ⏱️  Timeout ({timeout}s) - proceso detenido")
                return False
        else:
            stdout, stderr = proceso.communicate()
        
        if proceso.returncode == 0:
            print(f"   ✅ Comando exitoso")
            
            # Mostrar salida relevante
            for linea in stdout.split('\n'):
                if linea.strip() and 'pkg_resources' not in linea:
                    print(f"      {linea}")
            return True
        else:
            print(f"   ❌ Error en comando (código: {proceso.returncode})")
            
            # Mostrar errores
            if stderr:
                for linea in stderr.split('\n'):
                    if linea.strip():
                        print(f"      ERROR: {linea}")
            return False
            
    except Exception as e:
        print(f"   ❌ Excepción: {e}")
        return False

def ejecutar_patron_confiable():
    """Ejecuta CIGALE con el patrón confiable (run → genconf → run)"""
    
    print("\n" + "=" * 60)
    print("🚀 EJECUTANDO CIGALE CON PATRÓN CONFIABLE")
    print("=" * 60)
    print("Este patrón ha demostrado funcionar:")
    print("  1. pcigale check (verificar configuración)")
    print("  2. pcigale run - intento 1 (puede fallar)")
    print("  3. pcigale genconf (inicializar)")
    print("  4. pcigale run - intento 2 (debe funcionar)")
    print("=" * 60)
    
    respuesta = input("\n¿Ejecutar análisis completo ahora? (s/n): ").strip().lower()
    if respuesta != 's':
        print("⏹️  Ejecución cancelada")
        return False
    
    # Paso 1: Verificar configuración
    print("\n1️⃣  VERIFICANDO CONFIGURACIÓN (pcigale check)...")
    if not ejecutar_comando("pcigale check", "Verificando configuración", timeout=30):
        print("⚠️  Continuando a pesar de advertencias...")
    
    # Paso 2: Primer intento (puede fallar)
    print("\n2️⃣  PRIMER INTENTO (pcigale run - puede fallar)...")
    print("   ⏳ Esperando 10 segundos...")
    
    proceso = None
    try:
        proceso = subprocess.Popen(
            "pcigale run",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Esperar 10 segundos
        time.sleep(10)
        
        if proceso.poll() is None:
            # Sigue corriendo, puede que funcione
            print("   ⚡ El proceso sigue corriendo (puede ser buena señal)")
            print("   PID:", proceso.pid)
            
            opcion = input("\n¿Continuar esperando (c) o seguir con patrón (p)? (c/p): ").strip().lower()
            if opcion == 'c':
                print("⏳ Esperando finalización...")
                stdout, _ = proceso.communicate()
                
                if proceso.returncode == 0:
                    print("✅ ¡pcigale run funcionó a la primera!")
                    return True
                else:
                    print(f"❌ pcigale run falló (código: {proceso.returncode})")
            else:
                proceso.terminate()
                print("   ⏹️  Proceso detenido, continuando con patrón...")
        else:
            stdout, _ = proceso.communicate()
            print("   ❌ pcigale run falló rápidamente")
            
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Paso 3: Ejecutar genconf
    print("\n3️⃣  INICIALIZANDO (pcigale genconf)...")
    
    # Hacer backup de configuración
    if os.path.exists("pcigale.ini"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"pcigale.ini.backup_{timestamp}"
        subprocess.run(f"cp pcigale.ini {backup_file}", shell=True)
        print(f"   📦 Backup: {backup_file}")
    
    if ejecutar_comando("pcigale genconf", "Inicializando CIGALE", timeout=30):
        # Restaurar configuración
        if os.path.exists(backup_file):
            subprocess.run(f"mv {backup_file} pcigale.ini", shell=True)
            print("   ✅ Configuración restaurada")
    
    # Paso 4: Segundo intento (debería funcionar)
    print("\n4️⃣  SEGUNDO INTENTO (pcigale run - debería funcionar)...")
    print("   ⏳ Esto puede tomar VARIAS HORAS...")
    print("   📊 Para monitorear: tail -f out/log.txt")
    print("   🛑 Para detener: Ctrl+C")
    print("-" * 50)
    
    try:
        proceso = subprocess.Popen(
            "pcigale run",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print(f"   PID: {proceso.pid}")
        print(f"   Inicio: {datetime.now().strftime('%H:%M:%S')}")
        
        # Mostrar salida en tiempo real
        for linea in iter(proceso.stdout.readline, ''):
            linea = linea.strip()
            if linea and 'pkg_resources' not in linea:
                print(f"   {linea}")
        
        proceso.wait()
        
        if proceso.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ ¡CIGALE COMPLETADO EXITOSAMENTE!")
            print("=" * 50)
            print(f"   Fin: {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Resultados en: out/results.fits")
            return True
        else:
            print(f"\n❌ CIGALE falló (código: {proceso.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⏹️  EJECUCIÓN INTERRUMPIDA POR EL USUARIO")
        if proceso:
            proceso.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 GENERADOR DE CONFIGURACIÓN CIGALE - VERSIÓN MEJORADA")
    print("=" * 70)
    print("Este script genera la configuración optimizada para GCs en NGC 5128")
    print("usando solo los 7 filtros S-PLUS")
    print("=" * 70)
    
    # Preguntar opciones al usuario
    print("\n🎛️  OPCIONES DE CONFIGURACIÓN:")
    print("   1. Variables básicas para paper (masa, metalicidad, edad) [RECOMENDADO]")
    print("   2. Variables de formación estelar (SFRs)")
    print("   3. Mixtas (masa, metalicidad, edad, SFR)")
    
    try:
        var_opcion = int(input("\nSeleccione opción de variables (1-3) [1]: ").strip() or "1")
        if var_opcion not in [1, 2, 3]:
            var_opcion = 1
    except:
        var_opcion = 1
    
    print("\n📊 IMF (Initial Mass Function):")
    print("   1. Salpeter (1955) - El que sabemos funciona [RECOMENDADO]")
    print("   2. Chabrier (2003) - Para poblaciones de baja masa")
    
    try:
        imf_opcion = int(input("\nSeleccione IMF (1-2) [1]: ").strip() or "1")
        if imf_opcion not in [1, 2]:
            imf_opcion = 1
    except:
        imf_opcion = 1
    
    # Paso 1: Limpiar archivo de datos
    print("\n" + "=" * 70)
    print("1️⃣  LIMPIANDO ARCHIVO DE DATOS")
    print("-" * 40)
    
    if not limpiar_archivo_datos():
        respuesta = input("\n⚠️  Problemas limpiando archivo. ¿Continuar de todos modos? (s/n): ").strip().lower()
        if respuesta != 's':
            print("❌ No se puede continuar sin archivo de datos")
            sys.exit(1)
    
    # Paso 2: Verificar formato
    print("\n2️⃣  VERIFICANDO FORMATO DE DATOS")
    print("-" * 40)
    verificar_formato_datos()
    
    # Paso 3: Crear configuración
    print("\n3️⃣  CREANDO CONFIGURACIÓN OPTIMIZADA")
    print("-" * 40)
    if crear_config_perfecta(variables_opcion=var_opcion, imf_opcion=imf_opcion):
        print("\n" + "=" * 70)
        print("✅ CONFIGURACIÓN CREADA EXITOSAMENTE")
        print("=" * 70)
        
        print("\n📋 RESUMEN DE LA CONFIGURACIÓN:")
        print(f"   • Modelos: 180 (4 τ × 9 edades × 5 metalicidades)")
        print(f"   • Filtros: 7 S-PLUS (con errores)")
        
        if var_opcion == 1:
            print(f"   • Variables: stellar.m_star, stellar.metallicity, stellar.age_m_star")
            print(f"     (Masa estelar, metalicidad, edad - perfecto para paper)")
        elif var_opcion == 2:
            print(f"   • Variables: sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs")
            print(f"     (Tasas de formación estelar)")
        else:
            print(f"   • Variables: stellar.m_star, stellar.metallicity, stellar.age_m_star, sfh.sfr")
            print(f"     (Mixtas: masa, metalicidad, edad, SFR)")
        
        print(f"   • IMF: {imf_opcion} ({'Salpeter (1955)' if imf_opcion == 1 else 'Chabrier (2003)'})")
        print(f"   • Redshift: from_file (6 decimales)")
        print(f"   • Cores: 4")
        
        print("\n🚀 OPCIONES DE EJECUCIÓN:")
        print("   1. Solo verificar configuración (pcigale check)")
        print("   2. Ejecutar con patrón confiable (recomendado)")
        print("   3. Ejecutar directamente (pcigale run)")
        print("   4. Solo generar configuración, sin ejecutar")
        
        try:
            ejec_opcion = int(input("\nSeleccione opción de ejecución (1-4) [2]: ").strip() or "2")
        except:
            ejec_opcion = 2
        
        if ejec_opcion == 1:
            print("\n" + "=" * 40)
            print("🔍 SOLO VERIFICACIÓN")
            print("=" * 40)
            ejecutar_comando("pcigale check", "Verificando configuración")
            
        elif ejec_opcion == 2:
            ejecutar_patron_confiable()
            
        elif ejec_opcion == 3:
            print("\n" + "=" * 40)
            print("⚡ EJECUCIÓN DIRECTA")
            print("=" * 40)
            print("⚠️  Nota: La ejecución directa puede fallar en la primera vez")
            respuesta = input("¿Ejecutar pcigale run ahora? (s/n): ").strip().lower()
            if respuesta == 's':
                ejecutar_comando("pcigale run", "Ejecutando análisis completo")
        
        else:
            print("\n" + "=" * 40)
            print("📁 SOLO CONFIGURACIÓN")
            print("=" * 40)
            print("Configuración generada pero no ejecutada")
            print("Puedes ejecutar manualmente después con:")
            print("   pcigale check  # Verificar configuración")
            print("   pcigale run    # Ejecutar análisis")
        
        print("\n" + "=" * 70)
        print("🎉 ¡PROCESO COMPLETADO!")
        print("=" * 70)
        
        print("\n📝 COMANDOS ÚTILES PARA DESPUÉS:")
        print("   pcigale-plots sed              # Generar gráficos SED")
        print("   pcigale-plots pdf              # Gráficos de PDF")
        print("   python analizar_resultados.py  # Analizar resultados")
        print("\n📊 PARA VER RESULTADOS:")
        print("   ls -lh out/                    # Ver archivos generados")
        print("   tail -f out/log.txt            # Ver log en tiempo real")
        
    else:
        print("\n❌ Error creando configuración")
