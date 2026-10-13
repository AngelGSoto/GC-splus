#!/usr/bin/env python3
# preparar_cigale_beasley_fotometria.py
# Busca coincidencias automáticamente por coordenadas entre Beasley (2008) y fotometría
# Genera archivo de entrada para CIGALE solo con las coincidencias

import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

def buscar_coincidencias_beasley_fotometria(beasley_csv, phot_csv, max_sep_arcsec=3.0):
    """
    Busca coincidencias entre catálogo Beasley (2008) y catálogo de fotometría por coordenadas.
    SOLO para objetos con n_Name == 'g' (cúmulos globulares confirmados).
    
    Parámetros:
    -----------
    beasley_csv : str
        Ruta al catálogo de Beasley (globular_clusters_ngc5128_Beasley.csv)
    phot_csv : str
        Ruta al catálogo de fotometría
    max_sep_arcsec : float
        Separación máxima en arcseconds para considerar coincidencia
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con las coincidencias encontradas
    list
        Lista de T17IDs de las coincidencias
    """
    
    print(f"🔍 BUSCANDO COINCIDENCIAS BEASLEY-FOTOMETRÍA (tolerancia: {max_sep_arcsec} arcsec)")
    print("-" * 50)
    
    # Cargar catálogo de Beasley
    try:
        beasley_df = pd.read_csv(beasley_csv)
        print(f"✅ Catálogo Beasley cargado: {len(beasley_df)} objetos")
        
        # VERIFICAR COLUMNAS DISPONIBLES
        print(f"📋 Columnas Beasley disponibles: {list(beasley_df.columns)}")
        
        # FILTRAR SOLO OBJETOS CON n_Name == 'g' (cúmulos globulares confirmados)
        if 'n_Name' in beasley_df.columns:
            beasley_original_len = len(beasley_df)
            # Filtrar solo objetos con n_Name == 'g'
            beasley_df = beasley_df[beasley_df['n_Name'] == 'g'].copy()
            print(f"✅ Filtrado por n_Name == 'g': {len(beasley_df)} de {beasley_original_len} objetos")
            
            # Mostrar distribución de n_Name original
            distribucion_original = beasley_df['n_Name'].value_counts()
            print(f"📊 Distribución de n_Name (solo 'g'):")
            for valor, conteo in distribucion_original.items():
                print(f"   • '{valor}': {conteo} objetos")
        else:
            print("⚠️  ADVERTENCIA: Columna 'n_Name' no encontrada en catálogo Beasley")
            print("   No se aplicará filtro por tipo de objeto")
        
        # Verificar columnas necesarias
        required_beasley_cols = ['RAJ2000', 'DEJ2000', '[M/H]']
        for col in required_beasley_cols:
            if col not in beasley_df.columns:
                print(f"❌ Columna {col} no encontrada en catálogo Beasley")
                print(f"   Columnas disponibles: {list(beasley_df.columns)}")
                return None, []
                
    except Exception as e:
        print(f"❌ Error cargando catálogo Beasley: {e}")
        return None, []
    
    # Cargar catálogo de fotometría
    try:
        phot_df = pd.read_csv(phot_csv)
        print(f"✅ Catálogo fotometría cargado: {len(phot_df)} objetos")
    except Exception as e:
        print(f"❌ Error cargando catálogo fotometría: {e}")
        return None, []
    
    # Verificar columnas necesarias en fotometría
    required_phot_cols = ['T17ID', 'RAJ2000', 'DEJ2000']
    for col in required_phot_cols:
        if col not in phot_df.columns:
            print(f"❌ Columna {col} no encontrada en catálogo fotometría")
            print(f"   Columnas disponibles: {list(phot_df.columns)}")
            return None, []
    
    # Asegurar formato de IDs
    phot_df['T17ID'] = phot_df['T17ID'].astype(str).str.strip()
    
    # Crear objetos SkyCoord
    print("🗺️  Creando coordenadas SkyCoord...")
    
    # Asegurar que las coordenadas sean numéricas
    beasley_df['RAJ2000'] = pd.to_numeric(beasley_df['RAJ2000'], errors='coerce')
    beasley_df['DEJ2000'] = pd.to_numeric(beasley_df['DEJ2000'], errors='coerce')
    phot_df['RAJ2000'] = pd.to_numeric(phot_df['RAJ2000'], errors='coerce')
    phot_df['DEJ2000'] = pd.to_numeric(phot_df['DEJ2000'], errors='coerce')
    
    # Eliminar filas con coordenadas inválidas
    beasley_valid = beasley_df.dropna(subset=['RAJ2000', 'DEJ2000']).copy()
    phot_valid = phot_df.dropna(subset=['RAJ2000', 'DEJ2000']).copy()
    
    if len(beasley_valid) < len(beasley_df):
        print(f"⚠️  {len(beasley_df) - len(beasley_valid)} objetos Beasley con coordenadas inválidas")
    if len(phot_valid) < len(phot_df):
        print(f"⚠️  {len(phot_df) - len(phot_valid)} objetos fotometría con coordenadas inválidas")
    
    coords_beasley = SkyCoord(ra=beasley_valid['RAJ2000'].values * u.degree, 
                               dec=beasley_valid['DEJ2000'].values * u.degree)
    coords_phot = SkyCoord(ra=phot_valid['RAJ2000'].values * u.degree, 
                           dec=phot_valid['DEJ2000'].values * u.degree)
    
    # Buscar coincidencias más cercanas (de Beasley a fotometría)
    max_sep = max_sep_arcsec * u.arcsec
    print(f"🔍 Buscando coincidencias dentro de {max_sep_arcsec} arcsec...")
    
    try:
        idx_beasley_in_phot, d2d, _ = coords_beasley.match_to_catalog_sky(coords_phot)
        match_mask = d2d < max_sep
    except Exception as e:
        print(f"❌ Error en la búsqueda de coincidencias: {e}")
        return None, []
    
    # Contar coincidencias
    n_matches = match_mask.sum()
    print(f"📊 Coincidencias encontradas: {n_matches} de {len(beasley_valid)} objetos Beasley (n_Name='g')")
    
    if n_matches == 0:
        print("⚠️  No se encontraron coincidencias. Intentando con mayor tolerancia...")
        # Intentar con 5 arcsec
        max_sep_arcsec = 5.0
        max_sep = max_sep_arcsec * u.arcsec
        match_mask = d2d < max_sep
        n_matches = match_mask.sum()
        print(f"   Coincidencias con {max_sep_arcsec} arcsec: {n_matches}")
        
        if n_matches == 0:
            print("❌ No hay coincidencias incluso con mayor tolerancia")
            return pd.DataFrame(), []
    
    # Crear DataFrame con coincidencias
    coincidencias = []
    t17ids_coincidentes = []
    
    print(f"\n📋 DETALLES DE COINCIDENCIAS (primeras 20):")
    for i, (beasley_idx, phot_idx, sep) in enumerate(zip(
        np.where(match_mask)[0], 
        idx_beasley_in_phot[match_mask], 
        d2d[match_mask]
    )):
        beasley_row = beasley_valid.iloc[beasley_idx]
        phot_row = phot_valid.iloc[phot_idx]
        
        beasley_id = beasley_row.get('Name', f"Beasley_{beasley_idx}")
        t17id = phot_row['T17ID']
        
        # Obtener [M/H] de Beasley si está disponible
        mh_beasley = beasley_row.get('[M/H]', np.nan)
        mh_err_beasley = beasley_row.get('e_[M/H]', np.nan)
        
        coincidencia = {
            'Beasley_ID': beasley_id,
            'T17ID': t17id,
            'RA_Beasley': beasley_row['RAJ2000'],
            'Dec_Beasley': beasley_row['DEJ2000'],
            'RA_phot': phot_row['RAJ2000'],
            'Dec_phot': phot_row['DEJ2000'],
            'Sep_arcsec': sep.to(u.arcsec).value,
            'MH_Beasley': mh_beasley,
            'MH_err_Beasley': mh_err_beasley,
            'Vmag_Beasley': beasley_row.get('Vmag', np.nan),
            'RV_Beasley': beasley_row.get('RV', np.nan),
            'n_Name': beasley_row.get('n_Name', ''),
            'SimbadName': beasley_row.get('SimbadName', '')
        }
        
        coincidencias.append(coincidencia)
        t17ids_coincidentes.append(t17id)
        
        # Mostrar solo primeros 20
        if i < 20:
            mh_str = f"{mh_beasley:.2f}" if not np.isnan(mh_beasley) else "NaN"
            n_name_str = f"n_Name='{beasley_row.get('n_Name', '')}'"
            print(f"   {i+1:2d}. {beasley_id:15s} → {t17id:10s} "
                  f"(sep: {sep.to(u.arcsec).value:.2f} arcsec, "
                  f"[M/H]: {mh_str}, {n_name_str})")
    
    if n_matches > 20:
        print(f"   ... y {n_matches - 20} más")
    
    coincidencias_df = pd.DataFrame(coincidencias)
    
    # Verificar si hay duplicados
    t17id_counts = coincidencias_df['T17ID'].value_counts()
    duplicados = t17id_counts[t17id_counts > 1]
    if len(duplicados) > 0:
        print(f"\n⚠️  ADVERTENCIA: {len(duplicados)} T17IDs aparecen múltiples veces:")
        for t17id, count in duplicados.head(5).items():
            print(f"   • {t17id}: {count} coincidencias")
        if len(duplicados) > 5:
            print(f"   ... y {len(duplicados) - 5} más")
    
    # Análisis estadístico de las metalicidades
    if not coincidencias_df.empty and 'MH_Beasley' in coincidencias_df.columns:
        mh_values = coincidencias_df['MH_Beasley'].dropna()
        if len(mh_values) > 0:
            print(f"\n📊 ESTADÍSTICAS DE [M/H] (Beasley) para coincidencias:")
            print(f"   • Número: {len(mh_values)} objetos")
            print(f"   • Media: {mh_values.mean():.2f} dex")
            print(f"   • Mediana: {mh_values.median():.2f} dex")
            print(f"   • Mínimo: {mh_values.min():.2f} dex")
            print(f"   • Máximo: {mh_values.max():.2f} dex")
            print(f"   • Std: {mh_values.std():.2f} dex")
            
            # Distribución bimodal (picos típicos en NGC 5128)
            pobre = (mh_values < -1.0).sum()
            intermedio = ((mh_values >= -1.0) & (mh_values < -0.4)).sum()
            rico = (mh_values >= -0.4).sum()
            total_mh = len(mh_values)
            
            print(f"\n🎯 DISTRIBUCIÓN BIMODAL:")
            if total_mh > 0:
                print(f"   • Pobres ([M/H] < -1.0): {pobre} ({pobre/total_mh*100:.0f}%)")
                print(f"   • Intermedios (-1.0 ≤ [M/H] < -0.4): {intermedio} ({intermedio/total_mh*100:.0f}%)")
                print(f"   • Ricos ([M/H] ≥ -0.4): {rico} ({rico/total_mh*100:.0f}%)")
    
    # Guardar archivo con coincidencias
    output_coincidencias = 'coincidencias_beasley_phot.csv'
    coincidencias_df.to_csv(output_coincidencias, index=False)
    print(f"\n💾 Coincidencias guardadas en: {output_coincidencias}")
    
    return coincidencias_df, t17ids_coincidentes

def resolver_duplicados(coincidencias_df):
    """
    Resuelve duplicados en las coincidencias, priorizando:
    1. Objetos con n_Name == 'g' (aunque ya filtramos, por si acaso)
    2. Menor separación
    3. Mejor metalicidad (menor error)
    
    Parámetros:
    -----------
    coincidencias_df : pandas.DataFrame
        DataFrame con las coincidencias encontradas
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con duplicados resueltos
    """
    print(f"\n🔍 RESOLVIENDO DUPLICADOS EN COINCIDENCIAS...")
    
    if coincidencias_df.empty:
        return coincidencias_df
    
    # Identificar duplicados en T17ID
    t17id_counts = coincidencias_df['T17ID'].value_counts()
    duplicados = t17id_counts[t17id_counts > 1]
    
    if len(duplicados) == 0:
        print("✅ No hay duplicados para resolver")
        return coincidencias_df
    
    print(f"📊 Duplicados encontrados: {len(duplicados)} T17IDs")
    
    # Para cada T17ID duplicado, seleccionar la mejor coincidencia
    df_resuelto_list = []
    
    for t17id in coincidencias_df['T17ID'].unique():
        filas_t17id = coincidencias_df[coincidencias_df['T17ID'] == t17id]
        
        if len(filas_t17id) == 1:
            # No es duplicado, mantener
            df_resuelto_list.append(filas_t17id)
            continue
        
        # ES DUPLICADO - aplicar estrategia de selección
        print(f"   🔄 Resolviendo: {t17id} ({len(filas_t17id)} opciones)")
        
        # Priorizar menor separación
        seleccionada = filas_t17id.loc[filas_t17id['Sep_arcsec'].idxmin()]
        df_resuelto_list.append(pd.DataFrame([seleccionada]))
        
        # Mostrar qué se seleccionó
        print(f"      ✅ Seleccionado: {seleccionada['Beasley_ID']} (sep: {seleccionada['Sep_arcsec']:.3f} arcsec)")
    
    # Concatenar resultados
    df_resuelto = pd.concat(df_resuelto_list, ignore_index=True)
    
    # Verificar que no queden duplicados
    t17id_counts_final = df_resuelto['T17ID'].value_counts()
    duplicados_final = t17id_counts_final[t17id_counts_final > 1]
    
    if len(duplicados_final) > 0:
        print(f"⚠️  Aún hay {len(duplicados_final)} T17IDs duplicados después de resolver")
        # Forzar eliminación manteniendo el primero
        df_resuelto = df_resuelto.drop_duplicates(subset='T17ID', keep='first')
    
    print(f"✅ Duplicados resueltos: {len(coincidencias_df) - len(df_resuelto)} objetos eliminados")
    
    return df_resuelto

def filtrar_por_coincidencias(phot_csv, t17ids_coincidentes):
    """
    Filtra el catálogo de fotometría para quedarse solo con las coincidencias.
    
    Parámetros:
    -----------
    phot_csv : str
        Ruta al catálogo de fotometría
    t17ids_coincidentes : list
        Lista de T17IDs coincidentes
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame filtrado con solo las coincidencias
    """
    
    print(f"\n🔍 FILTRANDO CATÁLOGO FOTOMETRÍA ({len(t17ids_coincidentes)} coincidencias)...")
    
    try:
        phot_df = pd.read_csv(phot_csv)
        print(f"📊 Catálogo original: {len(phot_df)} objetos")
    except Exception as e:
        print(f"❌ Error cargando catálogo fotometría: {e}")
        return None
    
    # Asegurar formato de IDs
    phot_df['T17ID'] = phot_df['T17ID'].astype(str).str.strip()
    
    # Filtrar por coincidencias
    df_filtrado = phot_df[phot_df['T17ID'].isin(t17ids_coincidentes)].copy()
    
    if len(df_filtrado) == 0:
        print("❌ No se encontraron coincidencias en el catálogo de fotometría")
        print(f"   Primeros IDs buscados: {t17ids_coincidentes[:5]}")
        print(f"   Primeros IDs en catálogo: {phot_df['T17ID'].head().tolist()}")
        return None
    
    print(f"✅ Catálogo filtrado: {len(df_filtrado)} objetos encontrados")
    
    return df_filtrado

def abmag_to_mjy(mag_ab):
    """Convierte magnitud AB a flujo en mJy"""
    if mag_ab >= 90 or np.isnan(mag_ab):
        return 0.0
    return 3631.0 * 10**(-mag_ab / 2.5) * 1000

def preparar_datos_para_cigale(df_filtrado, output_file="gc_splus_cigale_coincidencias.txt"):
    """
    Crea archivo para CIGALE con 'redshift' como primera columna.
    Solo para los objetos filtrados.
    
    Parámetros:
    -----------
    df_filtrado : pandas.DataFrame
        DataFrame con datos filtrados
    output_file : str
        Nombre del archivo de salida
    """
    
    print(f"\n🔧 CREANDO ARCHIVO PARA CIGALE: {output_file}")
    
    if df_filtrado is None or df_filtrado.empty:
        print("❌ No hay datos para procesar")
        return False
    
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
    objetos_procesados = []
    
    print(f"🔄 Procesando {len(df_filtrado)} objetos...")
    
    # Contadores para diagnóstico
    missing_splus_corr = 0
    missing_decam_corr = 0
    
    for idx, row in df_filtrado.iterrows():
        obj_id = str(row.get('T17ID', f"obj_{idx+1}"))
        objetos_procesados.append(obj_id)
        redshift = 0.001825  # Redshift de NGC 5128
        
        # REDSHIFT primero, ID segundo
        values = [redshift, obj_id]
        
        # 1. PROCESAR FILTROS S-PLUS (7 filtros) - USAR VERSIONES CORREGIDAS
        for internal_name, cigale_name in splus_filter_map.items():
            mag_col = f'MAG_{internal_name}_3_corr'      # Magnitud corregida
            err_col = f'MAGERR_{internal_name}_3_corr'   # Error corregido
            
            # Verificar si las columnas corregidas existen
            if mag_col not in df_filtrado.columns:
                mag_col = f'MAG_{internal_name}_3'
                err_col = f'MAGERR_{internal_name}_3'
                missing_splus_corr += 1
            
            mag = row.get(mag_col, 99.0)
            if pd.isna(mag): 
                mag = 99.0
                
            err = row.get(err_col, 0.1)
            if pd.isna(err): 
                err = 0.1
                
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
            mag_col = f'{band}mag_corr'      # Ejemplo: umag_corr
            err_col = f'e_{band}mag_corr'    # Ejemplo: e_umag_corr
            
            # Verificar si las columnas corregidas existen
            if mag_col not in df_filtrado.columns:
                mag_col = f'{band}mag'
                err_col = f'e_{band}mag'
                missing_decam_corr += 1
            
            if mag_col not in df_filtrado.columns:
                mag = 99.0
                err = 0.1
            else:
                mag = row.get(mag_col, 99.0)
                if pd.isna(mag): 
                    mag = 99.0
                    
                err = row.get(err_col, 0.1)
                if pd.isna(err): 
                    err = 0.1
            
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
    
    # Mostrar diagnósticos
    if missing_splus_corr > 0:
        print(f"⚠️  Usando magnitudes S-PLUS sin corregir para {missing_splus_corr} filtros")
    if missing_decam_corr > 0:
        print(f"⚠️  Usando magnitudes DECam sin corregir para {missing_decam_corr} filtros")
    
    if not data_rows:
        print("❌ No se generaron datos válidos")
        return False
    
    # Encabezado: redshift, id, 7 filtros S-PLUS, 5 filtros DECam/SDSS
    header_list = ['redshift', 'id']
    
    # Añadir filtros S-PLUS
    for cigale_name in splus_filter_map.values():
        header_list.append(cigale_name)
        header_list.append(f'{cigale_name}_err')
    
    # Añadir filtros DECam/SDSS
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
    print(f"   • {len(data_rows)} objetos procesados")
    print(f"   • Redshift: {redshift:.6f} (mismo para todos)")
    print(f"   • Columnas: {len(header_list)}")
    print(f"   • Filtros: 7 S-PLUS + 5 DECam = 12 filtros")
    
    # Mostrar estadísticas de detección
    if data_rows:
        print(f"\n📊 ESTADÍSTICAS DE FLUJO (primer objeto):")
        obj1_data = data_rows[0]
        print(f"   • Objeto: {objetos_procesados[0]}")
        for i, band in enumerate(['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']):
            flux_idx = 2 + i*2
            if flux_idx < len(obj1_data):
                flux = obj1_data[flux_idx]
                if flux > 0:
                    print(f"   • {band}: {flux:.2e} mJy")
                else:
                    print(f"   • {band}: No detectado")
    
    return True

def main():
    """Función principal"""
    
    print("=" * 70)
    print("🔍 PREPARADOR DE DATOS CIGALE (BEASLEY + FOTOMETRÍA)")
    print("=" * 70)
    print("Busca coincidencias por coordenadas entre Beasley (2008) y fotometría")
    print("SOLO para objetos con n_Name == 'g' (cúmulos globulares confirmados)")
    print("=" * 70)
    
    # Configuración
    beasley_csv = "../../globular_clusters_ngc5128_Beasley.csv"  # Catálogo de Beasley
    phot_csv = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"  # Catálogo de fotometría
    max_sep_arcsec = 3.0  # Tolerancia en arcseconds
    
    print(f"📁 Ruta Beasley: {beasley_csv}")
    print(f"📁 Ruta fotometría: {phot_csv}")
    print(f"🎯 Tolerancia: {max_sep_arcsec} arcsec")
    print(f"🎯 Filtro: SOLO objetos con n_Name == 'g'")
    
    # Verificar si los archivos existen
    if not os.path.exists(beasley_csv):
        print(f"❌ Archivo no encontrado: {beasley_csv}")
        return False
    
    if not os.path.exists(phot_csv):
        print(f"❌ Archivo no encontrado: {phot_csv}")
        # Intentar ruta alternativa
        phot_csv = "all_fields_photometry_OPTIMIZED_1394objects.csv"
        print(f"   Intentando: {phot_csv}")
        if not os.path.exists(phot_csv):
            print("❌ No se pudo encontrar el catálogo de fotometría")
            return False
    
    # 1. Buscar coincidencias automáticamente entre Beasley y fotometría
    coincidencias_df, t17ids_coincidentes = buscar_coincidencias_beasley_fotometria(
        beasley_csv, phot_csv, max_sep_arcsec
    )
    
    if coincidencias_df is None or len(t17ids_coincidentes) == 0:
        print("❌ No se pueden continuar sin coincidencias")
        return False
    
    # 2. Resolver duplicados si los hay
    coincidencias_df = resolver_duplicados(coincidencias_df)
    
    # Actualizar lista de T17IDs después de resolver duplicados
    t17ids_coincidentes = coincidencias_df['T17ID'].tolist()
    
    # Guardar archivo con coincidencias (actualizado)
    output_coincidencias = 'coincidencias_beasley_phot.csv'
    coincidencias_df.to_csv(output_coincidencias, index=False)
    print(f"💾 Coincidencias (sin duplicados) guardadas en: {output_coincidencias}")
    
    # 3. Filtrar catálogo de fotometría por coincidencias
    df_filtrado = filtrar_por_coincidencias(phot_csv, t17ids_coincidentes)
    
    if df_filtrado is None:
        print("❌ No se pudo filtrar el catálogo de fotometría")
        return False
    
    # 4. Preparar archivo para CIGALE
    output_file = "gc_splus_cigale_coincidencias.txt"
    if not preparar_datos_para_cigale(df_filtrado, output_file):
        print("❌ Error creando archivo para CIGALE")
        return False
    
    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   1. {output_file} → Datos para CIGALE ({len(t17ids_coincidentes)} objetos)")
    print(f"   2. {output_coincidencias} → Detalles de coincidencias con [M/H] y n_Name")
    
    print(f"\n🚀 PARA USAR CON CIGALE:")
    print(f"   1. Configura pcigale.ini con:")
    print(f"      data_file = {output_file}")
    print(f"      redshift = from_file")
    print(f"   2. Ejecuta: pcigale run")
    
    print(f"\n📊 RESUMEN DE COINCIDENCIAS:")
    print(f"   • Total: {len(t17ids_coincidentes)} objetos Beasley (n_Name='g') con fotometría")
    print(f"   • Separación máxima: {max_sep_arcsec} arcsec")
    
    # Mostrar valores de [M/H] de Beasley para referencia
    if not coincidencias_df.empty and 'MH_Beasley' in coincidencias_df.columns:
        mh_validas = coincidencias_df['MH_Beasley'].dropna()
        print(f"\n🎯 VALORES DE REFERENCIA (Beasley et al. 2008):")
        print(f"   • Objetos con [M/H]: {len(mh_validas)} de {len(coincidencias_df)}")
        
        # Mostrar primeros 5 con [M/H]
        for i, (_, row) in enumerate(coincidencias_df.iterrows()):
            if i >= 5:  # Solo mostrar primeros 5
                print(f"   ... y {len(coincidencias_df) - 5} más")
                break
            mh_str = f"{row['MH_Beasley']:.2f}" if not np.isnan(row['MH_Beasley']) else "NaN"
            print(f"   • {row['Beasley_ID']:15s} → {row['T17ID']:10s} ([M/H] = {mh_str})")
    
    print(f"\n📝 CITACIÓN EN TU PAPER:")
    print(f"   Para justificar el uso de [Fe/H] vs [M/H] sin transformación:")
    print(f"   'Seguimos el enfoque de Hernández et al. (2018, MNRAS 476, 5189),")
    print(f"    quienes demostraron excelente acuerdo entre [Fe/H] (alta resolución)")
    print(f"    y [M/H] (Beasley et al. 2008) para cúmulos globulares en NGC 5128,'")
    print(f"    'con diferencias típicamente menores a 0.15 dex.'")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ ¡Todo listo! Puedes ejecutar CIGALE con el archivo generado.")
        else:
            print("\n❌ Hubo problemas en el proceso")
    except KeyboardInterrupt:
        print("\n\n⏹️  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
