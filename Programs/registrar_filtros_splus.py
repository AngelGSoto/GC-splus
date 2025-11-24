# registrar_filtros_splus_forzado.py
import os
import shutil
import subprocess
import sys

def registrar_filtros_forzado():
    """Registro forzado de filtros S-PLUS en CIGALE"""
    
    print("🔧 REGISTRO FORZADO DE FILTROS S-PLUS")
    print("=" * 50)
    
    # Directorios posibles de filtros de CIGALE
    posibles_directorios = [
        os.path.expanduser('~/.cigale/filters/'),
        os.path.expanduser('~/.local/share/cigale/filters/'),
        '/usr/local/share/cigale/filters/',
        '/usr/share/cigale/filters/',
    ]
    
    # Agregar directorio del entorno virtual
    env_path = os.path.dirname(sys.executable)
    posibles_directorios.append(os.path.join(os.path.dirname(env_path), 'share/cigale/filters/'))
    
    directorio_encontrado = None
    for directorio in posibles_directorios:
        if os.path.exists(directorio):
            directorio_encontrado = directorio
            print(f"✅ Directorio CIGALE encontrado: {directorio}")
            break
    
    if not directorio_encontrado:
        # Crear directorio si no existe
        directorio_encontrado = posibles_directorios[0]
        os.makedirs(directorio_encontrado, exist_ok=True)
        print(f"📁 Directorio creado: {directorio_encontrado}")
    
    # Copiar filtros
    filtros_splus = ['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']
    
    print("\n📋 COPIANDO FILTROS:")
    for filtro in filtros_splus:
        archivo_origen = f"{filtro}.dat"
        archivo_destino = os.path.join(directorio_encontrado, f"{filtro}.dat")
        
        if os.path.exists(archivo_origen):
            shutil.copy2(archivo_origen, archivo_destino)
            print(f"   ✅ {filtro}.dat → {directorio_encontrado}")
        else:
            print(f"   ❌ {archivo_origen} no encontrado")
    
    # Verificar con pcigale-filters
    print("\n🔍 VERIFICANDO CON pcigale-filters:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pcigale.filters', 'list'], 
                              capture_output=True, text=True, check=True)
        
        for filtro in filtros_splus:
            if filtro in result.stdout:
                print(f"   ✅ {filtro} registrado")
            else:
                print(f"   ❌ {filtro} NO registrado")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Error con pcigale-filters: {e}")
    
    return directorio_encontrado

def verificar_con_python():
    """Verificar filtros directamente con Python"""
    
    print("\n🐍 VERIFICANDO CON PYTHON:")
    try:
        from pcigale.filters import Filter
        
        filtros_splus = ['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']
        
        for filtro in filtros_splus:
            try:
                f = Filter(filtro)
                print(f"   ✅ {filtro}: λ_eff = {f.lambda_eff:.1f} Å, ancho = {f.width:.1f} Å")
            except Exception as e:
                print(f"   ❌ {filtro}: {e}")
                
    except ImportError as e:
        print(f"❌ No se pudo importar pcigale.filters: {e}")

if __name__ == "__main__":
    directorio = registrar_filtros_forzado()
    verificar_con_python()
    
    print(f"\n💡 Directorio de filtros: {directorio}")
    print("🎯 Si los filtros no aparecen, prueba reiniciando el terminal")
