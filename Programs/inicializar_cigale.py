import subprocess
import os
import shutil
import sys

def verificar_archivos_necesarios():
    """Verifica que todos los archivos necesarios existan"""
    
    print("🔍 VERIFICANDO ARCHIVOS NECESARIOS")
    print("=" * 50)
    
    archivos_requeridos = [
        'gc_splus_cigale_custom.txt',
        'F0378.dat', 'F0395.dat', 'F0410.dat', 'F0430.dat',
        'F0515.dat', 'F0660.dat', 'F0861.dat'
    ]
    
    problemas = []
    
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            tamaño = os.path.getsize(archivo)
            if tamaño > 0:
                print(f"✅ {archivo} ({tamaño} bytes)")
            else:
                print(f"❌ {archivo} está vacío")
                problemas.append(f"{archivo} está vacío")
        else:
            print(f"❌ {archivo} no encontrado")
            problemas.append(f"{archivo} no encontrado")
    
    return problemas

def crear_configuracion_robusta():
    """Crea un archivo pcigale.ini más robusto y verificado"""
    
    print("\n🔧 CREANDO CONFIGURACIÓN ROBUSTA")
    print("=" * 50)
    
    # Verificar el archivo de datos para obtener información real
    try:
        with open('gc_splus_cigale_custom.txt', 'r') as f:
            primeras_lineas = [f.readline() for _ in range(3)]
        
        # Analizar las columnas del archivo de datos
        columnas = primeras_lineas[0].strip().split()
        print(f"📋 Columnas en el archivo de datos: {len(columnas)}")
        print(f"   Ejemplo: {columnas[:5]}...")  # Primeras 5 columnas
        
    except Exception as e:
        print(f"⚠️  No se pudo leer el archivo de datos: {e}")
        columnas = []
    
    config_content = f"""# CONFIGURACIÓN CIGALE - CÚMULOS GLOBULARES NGC 5128
# Archivo generado automáticamente

# Archivo de datos
data_file = gc_splus_cigale_custom.txt

# Módulos SED para poblaciones estelares simples
sed_modules = ssp, bc03, redshifting

# Método de análisis
analysis_method = pdf_analysis

# Número de núcleos a usar
cores = 4

# Filtros S-PLUS - deben coincidir con archivos .dat
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

# Error adicional para sistemáticos
additionalerror = 0.05

[sed_modules_params]

  [[ssp]]
    # Edades para cúmulos globulares en Myr
    age = 4000, 6000, 8000, 10000, 12000, 13000
    
    # Metalicidades típicas de GCs
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02
    
    # IMF de Chabrier (más realista para GCs)
    imf = 1

  [[bc03]]
    # Modelos de población estelar de Bruzual & Charlot 2003
    separation_age = 10

  [[redshifting]]
    # Redshift de NGC 5128
    redshift = 0.0018

[analysis_params]
  # Parámetros físicos a recuperar
  variables = stellar.m_star, stellar.metallicity, stellar.age, stellar.lum
  
  # Guardar mejores SEDs
  save_best_sed = True
  
  # No guardar chi2 para ahorrar espacio
  save_chi2 = none
  
  # Manejo de límites superiores
  lim_flag = noscaling
  
  # No generar objetos mock
  mock_flag = False
  
  # Precisión de redshift
  redshift_decimals = 4
"""
    
    with open('pcigale.ini', 'w') as f:
        f.write(config_content)
    print("✅ pcigale.ini creado con configuración robusta")
    
    # Verificar que el archivo se creó correctamente
    if os.path.exists('pcigale.ini'):
        tamaño = os.path.getsize('pcigale.ini')
        print(f"📏 Tamaño de pcigale.ini: {tamaño} bytes")
        return True
    else:
        print("❌ No se pudo crear pcigale.ini")
        return False

def ejecutar_genconf_con_verbose():
    """Ejecuta pcigale genconf mostrando toda la salida"""
    
    print("\n🚀 EJECUTANDO pcigale genconf CON VERBOSE")
    print("=" * 50)
    
    # Opción 1: Intentar con pcigale genconf normal
    print("🔄 Intentando con: pcigale genconf")
    try:
        result = subprocess.run(
            ['pcigale', 'genconf'], 
            capture_output=True, 
            text=True,
            timeout=60  # 60 segundos de timeout
        )
        
        print(f"📤 Salida estándar:\n{result.stdout}")
        print(f"📥 Salida de error:\n{result.stderr}")
        print(f"📟 Código de salida: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ pcigale genconf ejecutado correctamente")
            return True
        else:
            print("❌ pcigale genconf falló")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ pcigale genconf excedió el tiempo límite")
        return False
    except FileNotFoundError:
        print("❌ Comando 'pcigale' no encontrado")
        
        # Opción 2: Intentar con python -m pcigale
        print("\n🔄 Intentando con: python -m pcigale genconf")
        try:
            result = subprocess.run(
                ['python', '-m', 'pcigale', 'genconf'], 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            print(f"📤 Salida estándar:\n{result.stdout}")
            print(f"📥 Salida de error:\n{result.stderr}")
            print(f"📟 Código de salida: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ python -m pcigale genconf ejecutado correctamente")
                return True
            else:
                print("❌ python -m pcigale genconf falló")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando python -m pcigale: {e}")
            return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def verificar_instalacion_cigale():
    """Verifica que CIGALE esté correctamente instalado"""
    
    print("\n🔍 VERIFICANDO INSTALACIÓN DE CIGALE")
    print("=" * 40)
    
    # Intentar diferentes formas de verificar CIGALE
    comandos_verificacion = [
        ['pcigale', '--version'],
        ['python', '-m', 'pcigale', '--version'],
        ['pcigale', '--help'],
        ['python', '-m', 'pcigale', '--help']
    ]
    
    for comando in comandos_verificacion:
        try:
            result = subprocess.run(
                comando, 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ {' '.join(comando)} funciona")
                if 'version' in comando and result.stdout:
                    print(f"   Versión: {result.stdout.strip()}")
                return True
        except:
            continue
    
    print("❌ No se pudo verificar la instalación de CIGALE")
    print("💡 Instala CIGALE con: pip install pcigale")
    return False

def main():
    """Función principal corregida"""
    
    print("🎯 INICIALIZADOR CIGALE - VERSIÓN CORREGIDA")
    print("=" * 60)
    print("Este script soluciona problemas comunes de inicialización")
    print("=" * 60)
    
    # Paso 1: Verificar instalación de CIGALE
    if not verificar_instalacion_cigale():
        sys.exit(1)
    
    # Paso 2: Verificar archivos necesarios
    problemas = verificar_archivos_necesarios()
    if problemas:
        print(f"\n❌ Se encontraron {len(problemas)} problemas:")
        for problema in problemas:
            print(f"   - {problema}")
        print("\n💡 Soluciones:")
        print("   1. Ejecuta primero: python preparar_datos_cigale.py")
        print("   2. Asegúrate de que los archivos .dat estén en el directorio")
        sys.exit(1)
    
    # Paso 3: Crear configuración robusta
    if not crear_configuracion_robusta():
        sys.exit(1)
    
    # Paso 4: Ejecutar genconf con verbose
    if not ejecutar_genconf_con_verbose():
        print("\n❌ No se pudo inicializar CIGALE")
        print("\n🔧 POSIBLES SOLUCIONES:")
        print("   1. Verifica que el archivo gc_splus_cigale_custom.txt tenga el formato correcto")
        print("   2. Asegúrate de que los nombres de los filtros en 'bands' coincidan con los archivos .dat")
        print("   3. Revisa que los archivos .dat tengan el formato correcto (2 columnas: wavelength transmission)")
        print("   4. Ejecuta manualmente: pcigale genconf y revisa los errores")
        sys.exit(1)
    
    # Paso 5: Verificar que se creó configuration.txt
    if os.path.exists('configuration.txt'):
        tamaño = os.path.getsize('configuration.txt')
        print(f"\n✅ configuration.txt generado ({tamaño} bytes)")
        
        # Hacer copia de seguridad
        shutil.copy2('configuration.txt', 'configuration_backup.txt')
        print("✅ Copia de seguridad: configuration_backup.txt")
        
        # Mostrar información del archivo generado
        try:
            with open('configuration.txt', 'r') as f:
                lineas = f.readlines()
            print(f"📄 configuration.txt tiene {len(lineas)} líneas")
        except:
            pass
    else:
        print("❌ configuration.txt no se generó")
        sys.exit(1)
    
    print("\n🎉 INICIALIZACIÓN COMPLETADA EXITOSAMENTE!")
    print("📝 Ahora puedes ejecutar: python ejecutar_cigale.py")

if __name__ == "__main__":
    main()
