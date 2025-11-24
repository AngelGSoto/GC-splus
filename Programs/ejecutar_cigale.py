import subprocess
import os
import time
import sys

def verificar_prerrequisitos():
    """Verifica que todo esté listo para ejecutar CIGALE"""
    print("🔍 VERIFICANDO PRERREQUISITOS")
    print("=" * 50)
    
    archivos_requeridos = [
        'pcigale.ini',
        'gc_splus_cigale_custom.txt',
        'F0378.dat', 'F0395.dat', 'F0410.dat', 'F0430.dat',
        'F0515.dat', 'F0660.dat', 'F0861.dat'
    ]
    
    faltan = False
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"✅ {archivo}")
        else:
            print(f"❌ {archivo} - NO ENCONTRADO")
            faltan = True
    return not faltan

def ejecutar_paso(comando, descripcion, paso_numero):
    """Ejecuta un paso del flujo CIGALE"""
    print(f"\n{paso_numero}. 🚀 {descripcion}")
    print("=" * 40)
    print(f"Comando: {' '.join(comando)}")
    
    inicio = time.time()
    try:
        result = subprocess.run(comando, check=True, capture_output=True, text=True)
        tiempo = time.time() - inicio
        if result.stdout:
            print(f"📤 Salida: {result.stdout.strip()}")
        print(f"✅ {descripcion} completado en {tiempo:.1f} segundos")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {descripcion}: {e}")
        if e.stderr:
            # Mostrar solo las primeras líneas del error para no saturar
            lineas_error = e.stderr.split('\n')[:10]
            print("📥 Primeras líneas de error:")
            for linea in lineas_error:
                if linea.strip():
                    print(f"   {linea}")
        return False
    except FileNotFoundError:
        print(f"❌ Comando no encontrado: {comando[0]}")
        return False

def ejecutar_flujo_completo_corregido():
    """Ejecuta el flujo completo de CIGALE, evitando sobrescribir pcigale.ini"""
    print("🎯 FLUJO COMPLETO CIGALE - VERSIÓN CORREGIDA")
    print("=" * 60)
    print("Usando sfhdelayed en lugar de ssp (más compatible)")
    print("=" * 60)
    
    # Verificar que todo esté listo
    if not verificar_prerrequisitos():
        print("\n❌ Faltan archivos necesarios")
        print("💡 Ejecuta en orden:")
        print("   1. python preparar_datos_cigale.py")
        print("   2. python iniciar_cigale.py")
        print("   3. python configurar_cigale_corregido.py")
        return False

    # Salta pcigale genconf si ya existe el archivo de configuración
    if not os.path.exists('pcigale.ini'):
        if not ejecutar_paso(['pcigale', 'genconf'], 'pcigale genconf', "3"):
            return False
    else:
        print("3. 📝 'pcigale.ini' ya existe, saltando 'pcigale genconf'")
    
    # Paso 4: check
    if not ejecutar_paso(['pcigale', 'check'], 'pcigale check', "4"):
        print("⚠️  pcigale check falló, pero continuando...")
    
    # Paso 5: run (sin timeout porque puede ser largo)
    print(f"\n5. 🚀 pcigale run")
    print("=" * 40)
    print("⏰ Esto puede tomar varios minutos...")
    inicio_run = time.time()
    try:
        result = subprocess.run(['pcigale', 'run'], check=True, capture_output=True, text=True)
        tiempo_run = time.time() - inicio_run
        print(f"✅ pcigale run completado en {tiempo_run/60:.1f} minutos")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en pcigale run: {e}")
        if e.stderr:
            lineas_error = e.stderr.split('\n')[:10]
            print("📥 Primeras líneas de error:")
            for linea in lineas_error:
                if linea.strip():
                    print(f"   {linea}")
        return False

    # Paso 6: plot básico
    ejecutar_paso(['pcigale', 'plot'], 'pcigale plot', "6")
    
    # Paso 7: plots SED especializados
    print(f"\n7. 📊 pcigale-plots sed")
    print("=" * 40)
    try:
        result = subprocess.run(['pcigale-plots', 'sed'], check=True, capture_output=True, text=True)
        print("✅ pcigale-plots sed ejecutado")
    except:
        print("⚠️  pcigale-plots sed no disponible")
    
    # Verificar resultados
    print(f"\n📋 VERIFICANDO RESULTADOS")
    print("=" * 40)
    archivos_resultado = [
        'results.fits',
        'results.txt', 
        'results.png',
        'configuration.txt',
        'SED.pdf',
        'SED.png'
    ]
    for archivo in archivos_resultado:
        if os.path.exists(archivo):
            tamaño = os.path.getsize(archivo) / 1024  # KB
            print(f"✅ {archivo} ({tamaño:.1f} KB)")
        else:
            print(f"❌ {archivo}")
    return True

def main():
    if ejecutar_flujo_completo_corregido():
        print("\n🎉 ¡FLUJO CIGALE COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print("📁 Archivos generados:")
        print("   - results.fits : Resultados completos")
        print("   - results.txt  : Tabla de resultados") 
        print("   - results.png  : Gráficos básicos")
        print("   - SED.pdf/png  : Gráficos SED especializados")
        print("\n📊 Para análisis detallado: python analizar_resultados_cigale.py")
        print("📊 Para más SEDs: python visualizar_seds.py")
    else:
        print("\n❌ El flujo CIGALE falló")
        sys.exit(1)

if __name__ == "__main__":
    main()
