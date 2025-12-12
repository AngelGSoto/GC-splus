#!/usr/bin/env python3
# crear_pcigale_ini_automático.py
# Script optimizado que genera EXACTAMENTE la configuración que funciona
# Basado en la configuración verificada con: pcigale check

import os
from datetime import datetime

def crear_config_perfecta():
    """Crea la configuración EXACTA que sabemos que funciona"""
    
    # ¡ESTA ES LA CONFIGURACIÓN EXACTA QUE FUNCIONA!
    config = """# NGC 5128 Globular Clusters – S-PLUS (CIGALE 2022.1)
# CONFIGURACIÓN VERIFICADA Y FUNCIONAL (pcigale check ✅)
# Generado automáticamente el: {fecha}

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
    tau_main = 500.0, 1000.0, 2000.0, 5000.0
    age_main = 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True
  
  [[xsl]]
    imf = 1
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    separation_age = 10
  
  [[redshifting]]
    redshift = 

[analysis_params]
  variables = sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 6
  blocks = 1
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

def limpiar_archivo_datos():
    """Limpia el archivo de datos (quita # y espacios innecesarios)"""
    
    archivo_original = "gc_splus_cigale_custom.txt"
    archivo_limpio = "gc_splus_cigale_fixed.txt"
    
    if not os.path.exists(archivo_original):
        print(f"⚠️  {archivo_original} no encontrado")
        return False
    
    try:
        with open(archivo_original, 'r') as f_in, open(archivo_limpio, 'w') as f_out:
            for linea in f_in:
                # Quitar # al inicio y espacios/tabs al final
                linea_limpia = linea.lstrip('#').strip()
                if linea_limpia:  # Solo escribir líneas no vacías
                    f_out.write(linea_limpia + '\n')
        
        # Contar líneas
        with open(archivo_limpio, 'r') as f:
            lineas = f.readlines()
        
        print(f"🧹 Archivo limpiado: {archivo_limpio}")
        print(f"   • {len(lineas)} objetos procesados")
        print(f"   • Primera línea: {lineas[0].strip() if lineas else 'Vacío'}")
        
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
        
        # Verificar que tenga al menos redshift e id
        partes = primera_linea.split()
        
        if len(partes) >= 2:
            try:
                redshift = float(partes[0])
                id_obj = partes[1]
                print(f"📊 Formato verificado:")
                print(f"   • Redshift: {redshift}")
                print(f"   • ID: {id_obj}")
                print(f"   • Campos totales: {len(partes)}")
                return True
            except ValueError:
                print(f"❌ Error: El primer campo no es un número: {partes[0]}")
                return False
        else:
            print(f"❌ Error: Formato incorrecto. Línea: {primera_linea}")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando archivo: {e}")
        return False

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

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 GENERADOR DE CONFIGURACIÓN CIGALE (VERIFICADA)")
    print("=" * 70)
    print("Este script genera EXACTAMENTE la configuración que sabemos funciona")
    print("basada en pruebas anteriores con pcigale check")
    print("=" * 70)
    
    # Paso 1: Limpiar archivo de datos
    print("\n1️⃣  LIMPIANDO ARCHIVO DE DATOS")
    print("-" * 40)
    if not limpiar_archivo_datos():
        print("⚠️  Continuando sin limpiar archivo...")
    
    # Paso 2: Verificar formato
    print("\n2️⃣  VERIFICANDO FORMATO DE DATOS")
    print("-" * 40)
    verificar_formato_datos()
    
    # Paso 3: Crear configuración
    print("\n3️⃣  CREANDO CONFIGURACIÓN PERFECTA")
    print("-" * 40)
    if crear_config_perfecta():
        print("\n✅ CONFIGURACIÓN CREADA EXITOSAMENTE")
        print("=" * 70)
        
        print("\n📋 RESUMEN DE LA CONFIGURACIÓN:")
        print("   • Modelos: 180 (4×9×5)")
        print("   • Filtros: 7 S-PLUS (con errores)")
        print("   • Variables: sfh.sfr, sfh.sfr10Myrs, sfh.sfr100Myrs")
        print("   • Redshift: from_file (6 decimales)")
        print("   • Cores: 12")
        
        print("\n🚀 PRÓXIMOS PASOS AUTOMÁTICOS:")
        print("-" * 40)
        
        # Ofrecer ejecutar comandos automáticamente
        respuesta = input("\n¿Ejecutar 'pcigale check' ahora? (s/n): ").strip().lower()
        if respuesta == 's':
            if ejecutar_comando("pcigale check", "Verificando configuración con pcigale"):
                respuesta2 = input("\n✅ Verificación exitosa. ¿Ejecutar 'pcigale run' ahora? (s/n): ").strip().lower()
                if respuesta2 == 's':
                    ejecutar_comando("pcigale run", "Ejecutando análisis completo")
        
        print("\n" + "=" * 70)
        print("🎉 ¡PROCESO COMPLETADO!")
        print("=" * 70)
        print("\nSi prefieres ejecutar manualmente:")
        print("   $ pcigale check  # Verificar configuración")
        print("   $ pcigale run    # Ejecutar análisis completo")
        
    else:
        print("\n❌ Error creando configuración")
