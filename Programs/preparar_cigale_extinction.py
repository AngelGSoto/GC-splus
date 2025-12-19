#!/usr/bin/env python3
# preparar_cigale_redshift_primero_corr.py
# Crea archivo con 'redshift' como primera columna (SOLO FILTROS ESTREÑOS S-PLUS)
# USANDO MAGNITUDES CORREGIDAS POR EXTINCIÓN
# USAR CON: redshift = from_file en pcigale.ini

import pandas as pd
import numpy as np
import os

def abmag_to_mjy(mag_ab):
    """Convierte magnitud AB a flujo en mJy"""
    if mag_ab >= 90 or np.isnan(mag_ab):
        return 0.0
    return 3631.0 * 10**(-mag_ab / 2.5) * 1000

def preparar_datos_redshift_primero(input_file, output_file="gc_splus_cigale_custom.txt", n_objects=None):
    """
    Crea archivo para CIGALE con 'redshift' como primera columna.
    SOLO FILTROS ESTREÑOS S-PLUS (7 filtros)
    USANDO MAGNITUDES CORREGIDAS POR EXTINCIÓN
    Formato correcto para: redshift = from_file
    """
    
    print("🔧 CREANDO ARCHIVO PARA CIGALE (REDSHIFT primero, solo filtros estrechos)")
    print("=" * 78)
    print("⚠️  PARA USAR CON: redshift = from_file en pcigale.ini")
    print("✨ Usando magnitudes CORREGIDAS por extinción")
    print("=" * 78)
    
    if not os.path.exists(input_file):
        print(f"❌ ERROR: {input_file} no encontrado")
        return False
    
    try:
        df = pd.read_csv(input_file)
        print(f"📊 Datos cargados: {len(df)} objetos")
    except Exception as e:
        print(f"❌ Error leyendo {input_file}: {e}")
        return False
    
    if n_objects and n_objects < len(df):
        df = df.head(n_objects).copy()
        print(f"🔬 Usando muestra: {len(df)} objetos")
    
    filter_map = {
        'F378': 'F0378', 'F395': 'F0395', 'F410': 'F0410',
        'F430': 'F0430', 'F515': 'F0515', 'F660': 'F0660', 'F861': 'F0861'
    }
    
    data_rows = []
    used_corrected = True
    warning_shown = False
    
    print("🔄 Procesando objetos (usando magnitudes corregidas por extinción)...")
    for idx, row in df.iterrows():
        obj_id = str(row.get('T17ID', row.get('recno', f"obj_{idx+1}")))
        redshift = 0.001825
        
        # REDSHIFT primero, ID segundo
        values = [redshift, obj_id]
        
        for internal_name, cigale_name in filter_map.items():
            # Primero intentar con columnas corregidas
            mag_col_corr = f'MAG_{internal_name}_3_corr'
            err_col_corr = f'MAGERR_{internal_name}_3_corr'
            
            # Si existen las columnas corregidas, usarlas
            if mag_col_corr in df.columns and err_col_corr in df.columns:
                mag = row.get(mag_col_corr, 99.0)
                err = row.get(err_col_corr, 0.1)
            else:
                # Si no existen, usar las no corregidas y mostrar advertencia (solo una vez)
                if not warning_shown:
                    print(f"⚠️  ADVERTENCIA: Columnas corregidas no encontradas para {internal_name}, usando versiones sin corregir")
                    warning_shown = True
                    used_corrected = False
                
                mag_col = f'MAG_{internal_name}_3'
                err_col = f'MAGERR_{internal_name}_3'
                mag = row.get(mag_col, 99.0)
                err = row.get(err_col, 0.1)
            
            if pd.isna(mag): mag = 99.0
            if pd.isna(err): err = 0.1
                
            if mag >= 90.0:
                flux = 0.0
                flux_err = 99.0
            else:
                flux = abmag_to_mjy(mag)
                flux_err = flux * (np.log(10)/2.5) * err if err > 0 else flux * 0.05
            
            values.append(flux)
            values.append(flux_err)
            
        data_rows.append(values)
    
    if not data_rows:
        print("❌ No se generaron datos válidos")
        return False
    
    header_list = ['redshift', 'id', 'F0378', 'F0378_err', 'F0395', 'F0395_err', 
                   'F0410', 'F0410_err', 'F0430', 'F0430_err', 'F0515', 'F0515_err', 
                   'F0660', 'F0660_err', 'F0861', 'F0861_err']
    
    with open(output_file, 'w') as f:
        f.write('# ' + ' '.join(header_list) + '\n')
        for row in data_rows:
            # Formatear valores
            formatted_values = [f"{row[0]:.6f}", row[1]]
            for val in row[2:]:
                formatted_values.append(f"{val:.6e}")
            
            f.write(' '.join(formatted_values) + "\n")
    
    print(f"\n✅ ARCHIVO CREADO: {output_file}")
    print(f"   • {len(data_rows)} objetos")
    print(f"   • Redshift: {redshift:.6f} (mismo para todos)")
    print(f"   • Columnas: {len(header_list)}")
    print(f"   • Primera columna: {header_list[0]}")
    print(f"   • Filtros: 7 filtros estrechos S-PLUS")
    
    if used_corrected:
        print(f"   • ✨ Usando magnitudes CORREGIDAS por extinción")
    else:
        print(f"   • ⚠️  Usando magnitudes SIN CORREGIR (columnas corregidas no encontradas)")
    
    # Estadísticas de detección
    print(f"\n🔍 ESTADÍSTICAS DE DETECCIÓN (primeros {min(10, len(data_rows))} objetos):")
    for i, (internal_name, cigale_name) in enumerate(filter_map.items()):
        fluxes = [data_rows[j][2 + 2*i] for j in range(min(10, len(data_rows)))]
        detected = sum(1 for f in fluxes if f > 0)
        print(f"   • {cigale_name} ({internal_name}): {detected}/10 objetos con flujo > 0")
    
    print(f"\n📋 EJEMPLO (primer objeto):")
    with open(output_file, 'r') as f:
        print(f"  {f.readline().strip()}")
        second_line = f.readline().strip()
        if len(second_line) > 100:
            print(f"  {second_line[:100]}...")
        else:
            print(f"  {second_line}")
    
    return True

if __name__ == "__main__":
    input_csv = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
    N_OBJECTS_TEST = None  # None para todos
    
    print("=" * 78)
    print("📁 PROCESAMIENTO DE DATOS PARA CIGALE (SOLO FILTROS ESTREÑOS)")
    print("=" * 78)
    print(f"Archivo de entrada: {input_csv}")
    print(f"Filtros incluidos: 7 filtros estrechos S-PLUS")
    print(f"✨ Usando magnitudes CORREGIDAS por extinción")
    print("=" * 78)
    
    success = preparar_datos_redshift_primero(input_csv, n_objects=N_OBJECTS_TEST)
    
    if success:
        print("\n" + "=" * 78)
        print("🚀 CONFIGURACIÓN PARA CIGALE (MISMA QUE ANTES):")
        print("=" * 78)
        print("1. En pcigale.ini usar:")
        print("   data_file = gc_splus_cigale_custom.txt")
        print("   redshift = from_file")
        print("   bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err,")
        print("           F0430, F0430_err, F0515, F0515_err, F0660, F0660_err,")
        print("           F0861, F0861_err")
        print("")
        print("2. Ejecutar:")
        print("   pcigale check")
        print("   pcigale run")
        print("")
        print("3. Para verificar el archivo generado:")
        print("   head -n 3 gc_splus_cigale_custom.txt")
        print("   wc -l gc_splus_cigale_custom.txt")
        print("")
        print("✅ El archivo de salida tiene los mismos nombres de columnas que antes,")
        print("   pero ahora usa magnitudes CORREGIDAS por extinción (si están disponibles).")
    else:
        print("\n❌ Error en el procesamiento")
