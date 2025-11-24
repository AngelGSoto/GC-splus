# crear_filtros_formato_exacto_doc.py
import os

def crear_filtros_formato_exacto():
    """Crea filtros con el formato EXACTO de la documentación de CIGALE"""
    
    print("🎯 CREANDO FILTROS CON FORMATO EXACTO (DOCUMENTACIÓN CIGALE)")
    print("=" * 60)
    
    # Información según documentación - formato EXACTO
    filtros_info = {
        'F0378': 'S-PLUS F0378 narrow-band filter (3780 Å)',
        'F0395': 'S-PLUS F0395 narrow-band filter (3950 Å)',
        'F0410': 'S-PLUS F0410 narrow-band filter (4100 Å)',
        'F0430': 'S-PLUS F0430 narrow-band filter (4300 Å)',
        'F0515': 'S-PLUS F0515 narrow-band filter (5150 Å)',
        'F0660': 'S-PLUS F0660 narrow-band filter (6600 Å)',
        'F0861': 'S-PLUS F0861 narrow-band filter (8610 Å)'
    }
    
    for filtro, descripcion in filtros_info.items():
        archivo_entrada = f"{filtro}.dat"
        archivo_salida = f"{filtro}_cigale.dat"  # Nombre simple
        
        if os.path.exists(archivo_entrada):
            print(f"📊 Creando {filtro}...")
            
            # Leer datos originales
            with open(archivo_entrada, 'r') as f:
                datos_originales = f.read().strip()
            
            # Crear archivo con formato EXACTO según documentación
            with open(archivo_salida, 'w') as f:
                # CABECERA EXACTA según documentación punto #7
                f.write(f"# {filtro}\n")                    # Línea 1: nombre del filtro
                f.write("# energy\n")                      # Línea 2: tipo (energy o photon)
                f.write(f"# {descripcion}\n")              # Línea 3: descripción
                f.write(datos_originales + "\n")           # Datos (longitud onda + transmisión)
            
            # Verificar el formato
            with open(archivo_salida, 'r') as f:
                cabecera = [f.readline().strip() for _ in range(3)]
                primera_linea_datos = f.readline().strip()
            
            print(f"   ✅ {filtro} → {archivo_salida}")
            print(f"   📋 Cabecera: {cabecera}")
            print(f"   📊 Primera línea datos: {primera_linea_datos}")
            
        else:
            print(f"❌ {archivo_entrada} no encontrado")
    
    print(f"\n🎯 COMANDO EXACTO PARA AGREGAR FILTROS:")
    print("pcigale-filters add F0378_cigale.dat F0395_cigale.dat F0410_cigale.dat F0430_cigale.dat F0515_cigale.dat F0660_cigale.dat F0861_cigale.dat")

if __name__ == "__main__":
    crear_filtros_formato_exacto()
