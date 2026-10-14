#!/usr/bin/env python3
# preparar_cigale_redshift_primero_decam_corr.py
# Crea archivo con 'redshift' como primera columna
# INCLUYE 7 filtros S-PLUS + 5 filtros DECam (Taylor catalog)
# USA MAGNITUDES CORREGIDAS POR EXTINCIÓN
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
    INCLUYE 12 FILTROS: 7 S-PLUS (estrechos) + 5 DECam (anchos)
    USANDO MAGNITUDES CORREGIDAS POR EXTINCIÓN
    Formato correcto para: redshift = from_file
    """
    
    print("🔧 CREANDO ARCHIVO PARA CIGALE (REDSHIFT primero, con corrección de extinción)")
    print("=" * 78)
    print("⚠️  PARA USAR CON: redshift = from_file en pcigale.ini")
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
    
    # Mapeo de filtros S-PLUS
    splus_filter_map = {
        'F378': 'F0378', 'F395': 'F0395', 'F410': 'F0410',
        'F430': 'F0430', 'F515': 'F0515', 'F660': 'F0660', 'F861': 'F0861'
    }
    
    # Mapeo de filtros DECam (Taylor catalog) a nombres CIGALE SDSS
    decam_filter_map = {
        'u': 'sdss.up',  # Taylor column: umag_corr
        'g': 'sdss.gp',  # Taylor column: gmag_corr
        'r': 'sdss.rp',  # Taylor column: rmag_corr
        'i': 'sdss.ip',  # Taylor column: imag_corr
        'z': 'sdss.zp'   # Taylor column: zmag_corr
    }
    
    data_rows = []
    
    print("🔄 Procesando objetos (usando magnitudes corregidas por extinción)...")
    for idx, row in df.iterrows():
        obj_id = str(row.get('T17ID', row.get('recno', f"obj_{idx+1}")))
        redshift = 0.001825  # Redshift de NGC 5128
        
        # REDSHIFT primero, ID segundo
        values = [redshift, obj_id]
        
        # 1. PROCESAR FILTROS S-PLUS (7 filtros) - USAR VERSIONES CORREGIDAS
        for internal_name, cigale_name in splus_filter_map.items():
            mag_col = f'MAG_{internal_name}_3_corr'      # Magnitud corregida (apertura 3 arcsec)
            err_col = f'MAGERR_{internal_name}_3_corr'   # Error corregido
            
            # Verificar si las columnas corregidas existen
            if mag_col not in df.columns:
                print(f"⚠️  ADVERTENCIA: Columna {mag_col} no encontrada, usando versión sin corregir")
                mag_col = f'MAG_{internal_name}_3'
                err_col = f'MAGERR_{internal_name}_3'
            
            mag = row.get(mag_col, 99.0)
            if pd.isna(mag): mag = 99.0
                
            err = row.get(err_col, 0.1)
            if pd.isna(err): err = 0.1
                
            if mag >= 90.0:
                flux = 0.0
                flux_err = 99.0
            else:
                flux = abmag_to_mjy(mag)
                flux_err = flux * (np.log(10)/2.5) * err if err > 0 else flux * 0.05
            
            values.append(flux)
            values.append(flux_err)
        
        # 2. PROCESAR FILTROS DECam (5 filtros) - USAR VERSIONES CORREGIDAS
        decam_bands = ['u', 'g', 'r', 'i', 'z']
        
        for band in decam_bands:
            mag_col = f'{band}mag_corr'      # Ejemplo: umag_corr, gmag_corr, etc.
            err_col = f'e_{band}mag_corr'    # Ejemplo: e_umag_corr, e_gmag_corr, etc.
            
            # Verificar si las columnas corregidas existen
            if mag_col not in df.columns:
                print(f"⚠️  ADVERTENCIA: Columna {mag_col} no encontrada, usando versión sin corregir")
                mag_col = f'{band}mag'
                err_col = f'e_{band}mag'
            
            if mag_col not in df.columns:
                print(f"❌ ERROR: Columna {mag_col} no encontrada en archivo")
                mag = 99.0
                err = 0.1
            else:
                mag = row.get(mag_col, 99.0)
                if pd.isna(mag): mag = 99.0
                    
                err = row.get(err_col, 0.1)
                if pd.isna(err): err = 0.1
            
            # Convertir magnitud a flujo
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
    
    # Encabezado: redshift, id, 7 filtros S-PLUS, 5 filtros DECam/SDSS
    header_list = ['redshift', 'id']
    
    # Añadir filtros S-PLUS (mismos nombres que antes)
    for cigale_name in splus_filter_map.values():
        header_list.append(cigale_name)
        header_list.append(f'{cigale_name}_err')
    
    # Añadir filtros DECam/SDSS (mismos nombres que antes)
    for cigale_name in decam_filter_map.values():
        header_list.append(cigale_name)
        header_list.append(f'{cigale_name}_err')
    
    # Escribir archivo
    with open(output_file, 'w') as f:
        f.write('# ' + ' '.join(header_list) + '\n')
        for row in data_rows:
            # Formatear valores
            formatted_values = [f"{row[0]:.6f}", row[1]]
            for val in row[2:]:
                if isinstance(val, (int, float)):
                    formatted_values.append(f"{val:.6e}")
                else:
                    formatted_values.append(str(val))
            
            f.write(' '.join(formatted_values) + "\n")
    
    print(f"\n✅ ARCHIVO CREADO: {output_file}")
    print(f"   • {len(data_rows)} objetos")
    print(f"   • Redshift: {redshift:.6f} (mismo para todos)")
    print(f"   • Columnas: {len(header_list)}")
    print(f"   • Primera columna: {header_list[0]}")
    
    print(f"\n📊 RESUMEN DE FILTROS:")
    print(f"   • S-PLUS (7): {', '.join(splus_filter_map.values())}")
    print(f"   • DECam/SDSS (5): {', '.join(decam_filter_map.values())}")
    print(f"   • TOTAL: 12 filtros")
    
    print(f"   • ✨ Usando magnitudes CORREGIDAS por extinción")
    
    # Estadísticas de detección
    print(f"\n🔍 ESTADÍSTICAS DE DETECCIÓN (primeros {min(10, len(data_rows))} objetos):")
    for i, band in enumerate(splus_filter_map.values()):
        fluxes = [data_rows[j][2 + 2*i] for j in range(min(10, len(data_rows)))]
        detected = sum(1 for f in fluxes if f > 0)
        print(f"   • {band}: {detected}/10 objetos con flujo > 0")
    
    for i, band in enumerate(decam_filter_map.values()):
        offset = 2 + 14  # 2 (redshift+id) + 14 (7 filtros S-PLUS × 2)
        fluxes = [data_rows[j][offset + 2*i] for j in range(min(10, len(data_rows)))]
        detected = sum(1 for f in fluxes if f > 0)
        print(f"   • {band}: {detected}/10 objetos con flujo > 0")
    
    print(f"\n📋 EJEMPLO (primer objeto):")
    with open(output_file, 'r') as f:
        print(f"  {f.readline().strip()}")
        second_line = f.readline().strip()
        if len(second_line) > 150:
            print(f"  {second_line[:150]}...")
        else:
            print(f"  {second_line}")
    
    return True

if __name__ == "__main__":
    input_csv = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
    N_OBJECTS_TEST = None  # None para todos, o un número para prueba
    
    print("=" * 78)
    print("📁 PROCESAMIENTO DE DATOS PARA CIGALE (CON CORRECCIÓN DE EXTINCIÓN)")
    print("=" * 78)
    print(f"Archivo de entrada: {input_csv}")
    print(f"Filtros incluidos: 7 S-PLUS + 5 DECam = 12 filtros totales")
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
        print("   bands = F0378, F0378_err, ..., F0861, F0861_err,")
        print("           sdss.up, sdss.up_err, ..., sdss.zp, sdss.zp_err")
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
        print("   pero ahora usa magnitudes CORREGIDAS por extinción.")
    else:
        print("\n❌ Error en el procesamiento")
