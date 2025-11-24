import subprocess
import os
import sys

def iniciar_cigale():
    """Ejecuta pcigale init para crear la configuración inicial"""
    
    print("🎯 INICIAR CIGALE - PASO 1: pcigale init")
    print("=" * 60)
    
    # Verificar que el archivo de datos existe
    if not os.path.exists('gc_splus_cigale_custom.txt'):
        print("❌ No se encuentra gc_splus_cigale_custom.txt")
        print("💡 Ejecuta primero: python preparar_datos_cigale.py")
        return False
    
    print("🔄 Ejecutando pcigale init...")
    try:
        result = subprocess.run(['pcigale', 'init'], check=True, capture_output=True, text=True)
        print("✅ pcigale init ejecutado correctamente")
        
        # Verificar que se creó pcigale.ini
        if os.path.exists('pcigale.ini'):
            print("✅ pcigale.ini creado")
            return True
        else:
            print("❌ pcigale.ini no se creó")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en pcigale init: {e}")
        return False
    except FileNotFoundError:
        print("❌ Comando 'pcigale' no encontrado")
        return False

def main():
    if iniciar_cigale():
        print("\n✅ PASO 1 COMPLETADO: pcigale init")
        print("📝 Ahora modifica pcigale.ini con tu configuración")
        print("💡 Luego ejecuta: python configurar_cigale.py")
    else:
        print("\n❌ Error en pcigale init")
        sys.exit(1)

if __name__ == "__main__":
    main()
