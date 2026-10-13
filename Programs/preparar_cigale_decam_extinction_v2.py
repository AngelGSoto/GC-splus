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

def verificar_columnas_extincion(df, band, prefix=''):
    """
    Verifica qué columnas de extinción existen para una banda dada.
    Retorna el nombre de la columna de magnitud corregida preferida.
    """
    # Posibles nombres de columnas corregidas (orden de preferencia)
    posibles_corr = [
        f'{prefix}{band}mag_corr',      # umag_corr, gmag_corr
        f'{prefix}{band}_corr',         # u_corr, g_corr
        f'{prefix}{band}mag_dered',     # umag_dered, gmag_dered
        f'{prefix}{band}_dered',        # u_dered, g_dered
        f'{prefix}{band}mag_extcorr',   # umag_extcorr
        f'{band}mag_corr',              # sin prefix (para columnas simples)
        f'{band}_corr',
    ]
    
    # Posibles nombres de columnas de error corregidas
    posibles_err = [
        f'e_{prefix}{band}mag_corr',    # e_umag_corr
        f'e_{prefix}{band}_corr',       # e_u_corr
        f'err_{prefix}{band}mag_corr',
        f'{prefix}{band}magerr_corr',
        f'e_{band}mag_corr',            # sin prefix
        f'err_{band}mag_corr',
    ]
    
    # Buscar magnitud
    mag_col = None
    for col in posibles_corr:
        if col in df.columns:
            # Verificar que no esté vacía
            if df[col].notna().sum() > 0:
                mag_col = col
                break
    
    # Si no hay corregida, buscar la normal
    if mag_col is None:
        posibles_normal = [
            f'{prefix}{band}mag', f'{prefix}{band}',
            f'{band}mag', f'{band}'
        ]
        for col in posibles_normal:
            if col in df.columns:
                if df[col].notna().sum() > 0:
                    mag_col = col
                    print(f"   ⚠️  Usando {col} (sin corrección de extinción)")
                    break
    
    # Buscar error correspondiente
    err_col = None
    if mag_col:
        # Intentar encontrar error corregido que corresponda
        for col in posibles_err:
            if col in df.columns:
                err_col = col
                break
        
        # Si no hay error corregido, buscar error normal
        if err_col is None:
            posibles_err_normal = [
                f'e_{prefix}{band}mag', f'err_{prefix}{band}mag',
                f'{prefix}{band}magerr', f'{prefix}{band}err',
                f'e_{band}mag', f'err_{band}mag', f'{band}magerr'
            ]
            for col in posibles_err_normal:
                if col in df.columns:
                    err_col = col
                    break
    
    return mag_col, err_col

def preparar_datos_redshift_primero(input_file, output_file="gc_splus_cigale_custom.txt", n_objects=None):
    """
    Crea archivo para CIGALE con 'redshift' como primera columna.
    INCLUYE 12 FILTROS: 7 S-PLUS (estrechos) + 5 DECam (anchos)
    USANDO MAGNITUDES CORREGIDAS POR EXTINCIÓN
    """
    
    print("🔧 CREANDO ARCHIVO PARA CIGALE (REDSHIFT primero)")
    print("=" * 78)
    print("Buscando magnitudes CORREGIDAS POR EXTINCIÓN")
    print("=" * 78)
    
    if not os.path.exists(input_file):
        print(f"❌ ERROR: {input_file} no encontrado")
        return False
    
    try:
        df = pd.read_csv(input_file)
        print(f"📊 Datos cargados: {len(df)} objetos")
        print(f"📋 Columnas disponibles: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error leyendo {input_file}: {e}")
        return False
    
    # Mostrar primeras columnas para diagnóstico
    print(f"\n🔍 Primeras columnas en archivo:")
    for i, col in enumerate(df.columns[:15]):
        print(f"   {i+1}. {col}")
    
    if n_objects and n_objects < len(df):
        df = df.head(n_objects).copy()
        print(f"\n🔬 Usando muestra: {len(df)} objetos")
    
    # Mapeo de filtros S-PLUS
    splus_filter_map = {
        'F378': 'F0378', 'F395': 'F0395', 'F410': 'F0410',
        'F430': 'F0430', 'F515': 'F0515', 'F660': 'F0660', 'F861': 'F0861'
    }
    
    # Mapeo de filtros DECam a nombres CIGALE SDSS
    decam_filter_map = {
        'u': 'sdss.up', 'g': 'sdss.gp', 'r': 'sdss.rp', 
        'i': 'sdss.ip', 'z': 'sdss.zp'
    }
    
    # Verificar qué columnas de extinción existen
    print(f"\n🔎 VERIFICANDO COLUMNAS DE EXTINCIÓN:")
    
    # Verificar S-PLUS
    print(f"\n   S-PLUS filtros:")
    splus_cols_encontradas = {}
    for filt, cigale_name in splus_filter_map.items():
        mag_col = f'MAG_{filt}_3_corr'
        err_col = f'MAGERR_{filt}_3_corr'
        
        mag_col_alt = f'MAG_{filt}_3'
        err_col_alt = f'MAGERR_{filt}_3'
        
        if mag_col in df.columns and df[mag_col].notna().sum() > 0:
            print(f"      ✅ {cigale_name}: {mag_col} ({df[mag_col].notna().sum()} valores válidos)")
            splus_cols_encontradas[cigale_name] = (mag_col, err_col)
        elif mag_col_alt in df.columns:
            print(f"      ⚠️  {cigale_name}: {mag_col_alt} (SIN corrección)")
            splus_cols_encontradas[cigale_name] = (mag_col_alt, err_col_alt)
        else:
            print(f"      ❌ {cigale_name}: No encontrado")
    
    # Verificar DECam
    print(f"\n   DECam filtros:")
    decam_cols_encontradas = {}
    for band, cigale_name in decam_filter_map.items():
        mag_col, err_col = verificar_columnas_extincion(df, band)
        
        if mag_col:
            if 'corr' in mag_col or 'dered' in mag_col:
                print(f"      ✅ {cigale_name}: {mag_col} ({df[mag_col].notna().sum()} valores) [CORREGIDA]")
            else:
                print(f"      ⚠️  {cigale_name}: {mag_col} ({df[mag_col].notna().sum()} valores) [sin corr]")
            decam_cols_encontradas[cigale_name] = (mag_col, err_col)
        else:
            print(f"      ❌ {cigale_name}: No encontrado")
    
    # Procesar datos
    data_rows = []
    usando_corregidas = True
    
    print(f"\n🔄 Procesando {len(df)} objetos...")
    
    for idx, row in df.iterrows():
        obj_id = str(row.get('T17ID', row.get('recno', f"obj_{idx+1}")))
        redshift = 0.001825  # Redshift de NGC 5128
        
        values = [redshift, obj_id]
        
        # Procesar S-PLUS
        for cigale_name in splus_filter_map.values():
            if cigale_name not in splus_cols_encontradas:
                values.extend([0.0, 99.0])
                continue
                
            mag_col, err_col = splus_cols_encontradas[cigale_name]
            
            # Verificar si es corregida
            if 'corr' not in mag_col:
                usando_corregidas = False
            
            mag = row.get(mag_col, 99.0)
            if pd.isna(mag): mag = 99.0
                
            err = row.get(err_col, 0.1) if err_col else 0.1
            if pd.isna(err): err = 0.1
                
            if mag >= 90.0:
                flux = 0.0
                flux_err = 99.0
            else:
                flux = abmag_to_mjy(mag)
                flux_err = flux * (np.log(10)/2.5) * err if err > 0 else flux * 0.05
            
            values.append(flux)
            values.append(flux_err)
        
        # Procesar DECam
        for cigale_name in decam_filter_map.values():
            if cigale_name not in decam_cols_encontradas:
                values.extend([0.0, 99.0])
                continue
                
            mag_col, err_col = decam_cols_encontradas[cigale_name]
            
            # Verificar si es corregida
            if mag_col and 'corr' not in mag_col and 'dered' not in mag_col:
                usando_corregidas = False
            
            mag = row.get(mag_col, 99.0) if mag_col else 99.0
            if pd.isna(mag): mag = 99.0
                
            err = row.get(err_col, 0.1) if err_col else 0.1
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
    
    # Crear encabezado
    header_list = ['redshift', 'id']
    for cigale_name in splus_filter_map.values():
        header_list.append(cigale_name)
        header_list.append(f'{cigale_name}_err')
    for cigale_name in decam_filter_map.values():
        header_list.append(cigale_name)
        header_list.append(f'{cigale_name}_err')
    
    # Escribir archivo
    with open(output_file, 'w') as f:
        f.write('# ' + ' '.join(header_list) + '\n')
        for row in data_rows:
            formatted_values = [f"{row[0]:.6f}", row[1]]
            for val in row[2:]:
                if isinstance(val, (int, float)):
                    formatted_values.append(f"{val:.6e}")
                else:
                    formatted_values.append(str(val))
            f.write(' '.join(formatted_values) + "\n")
    
    # Reporte final
    print(f"\n{'='*78}")
    print(f"✅ ARCHIVO CREADO: {output_file}")
    print(f"   • {len(data_rows)} objetos")
    print(f"   • {len(header_list)} columnas")
    
    if usando_corregidas:
        print(f"\n🌟 ÉXITO: Usando magnitudes CORREGIDAS POR EXTINCIÓN en todos los filtros")
    else:
        print(f"\n⚠️  ADVERTENCIA: ALGUNOS filtros usan magnitudes SIN corregir")
        print(f"    Revisa los mensajes arriba para ver cuáles")
    
    # Estadísticas
    print(f"\n📊 Estadísticas de detección (muestra de 10 objetos):")
    for i, band in enumerate(list(splus_filter_map.values())[:3]):
        fluxes = [r[2 + 2*i] for r in data_rows[:10]]
        detected = sum(1 for f in fluxes if f > 0)
        print(f"   • {band}: {detected}/10 detectados")
    
    return True

if __name__ == "__main__":
    input_csv = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1496objects.csv"
    N_OBJECTS_TEST = None
    
    print("=" * 78)
    print("📁 PREPARACIÓN DE DATOS PARA CIGALE")
    print("=" * 78)
    
    success = preparar_datos_redshift_primero(input_csv, n_objects=N_OBJECTS_TEST)
    
    if success:
        print("\n" + "=" * 78)
        print("🚀 INSTRUCCIONES PARA CIGALE:")
        print("=" * 78)
        print("En pcigale.ini:")
        print("   data_file = gc_splus_cigale_custom.txt")
        print("   redshift = from_file")
        print("   bands = F0378, F0378_err, F0395, F0395_err, ...")
        print("           sdss.up, sdss.up_err, sdss.gp, sdss.gp_err, ...")
        print("\nEjecutar:")
        print("   pcigale check")
        print("   pcigale run")
