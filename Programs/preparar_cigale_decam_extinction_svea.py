#!/usr/bin/env python3
# preparar_cigale_coincidencias_auto.py
# Busca coincidencias automáticamente por coordenadas entre catálogo Svea y fotometría
# Genera archivo de entrada para CIGALE solo con las coincidencias

import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

def buscar_coincidencias_por_coordenadas(svea_csv, phot_csv, max_sep_arcsec=3.0):
    """
    Busca coincidencias entre catálogo Svea y catálogo de fotometría por coordenadas.
    
    Parámetros:
    -----------
    svea_csv : str
        Ruta al catálogo de Svea (debe tener columnas: Id, RA (J2000), Dec (J2000), [Fe/H] (dex), Error (dex))
    phot_csv : str
        Ruta al catálogo de fotometría (debe tener columnas: T17ID, RAJ2000, DEJ2000, ...)
    max_sep_arcsec : float
        Separación máxima en arcseconds para considerar coincidencia
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con las coincidencias encontradas
    list
        Lista de T17IDs de las coincidencias
    """
    
    print(f"🔍 BUSCANDO COINCIDENCIAS (tolerancia: {max_sep_arcsec} arcsec)")
    print("-" * 50)
    
    # Cargar catálogo de Svea con nombres de columnas específicos
    try:
        svea_df = pd.read_csv(svea_csv)
        print(f"✅ Catálogo Svea cargado: {len(svea_df)} objetos")
        
        # Verificar y renombrar columnas si es necesario
        rename_dict = {}
        if 'Id' in svea_df.columns:
            rename_dict['Id'] = 'ID'
        if 'RA (J2000)' in svea_df.columns:
            rename_dict['RA (J2000)'] = 'RA'
        if 'Dec (J2000)' in svea_df.columns:
            rename_dict['Dec (J2000)'] = 'Dec'
        if '[Fe/H] (dex)' in svea_df.columns:
            rename_dict['[Fe/H] (dex)'] = 'FeH'
        if 'Error (dex)' in svea_df.columns:
            rename_dict['Error (dex)'] = 'FeH_err'
        
        if rename_dict:
            svea_df = svea_df.rename(columns=rename_dict)
            print(f"   Columnas renombradas: {list(rename_dict.keys())} -> {list(rename_dict.values())}")
            
    except Exception as e:
        print(f"❌ Error cargando catálogo Svea: {e}")
        return None, []
    
    # Cargar catálogo de fotometría
    try:
        phot_df = pd.read_csv(phot_csv)
        print(f"✅ Catálogo fotometría cargado: {len(phot_df)} objetos")
    except Exception as e:
        print(f"❌ Error cargando catálogo fotometría: {e}")
        return None, []
    
    # Verificar columnas necesarias en Svea
    required_svea_cols = ['ID', 'RA', 'Dec']
    for col in required_svea_cols:
        if col not in svea_df.columns:
            print(f"❌ Columna {col} no encontrada en catálogo Svea")
            print(f"   Columnas disponibles: {list(svea_df.columns)}")
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
    coords_svea = SkyCoord(ra=svea_df['RA'].values * u.degree, 
                           dec=svea_df['Dec'].values * u.degree)
    coords_phot = SkyCoord(ra=phot_df['RAJ2000'].values * u.degree, 
                           dec=phot_df['DEJ2000'].values * u.degree)
    
    # Buscar coincidencias más cercanas
    max_sep = max_sep_arcsec * u.arcsec
    print(f"🔍 Buscando coincidencias dentro de {max_sep_arcsec} arcsec...")
    
    try:
        idx_svea_in_phot, d2d, _ = coords_svea.match_to_catalog_sky(coords_phot)
        match_mask = d2d < max_sep
    except Exception as e:
        print(f"❌ Error en la búsqueda de coincidencias: {e}")
        return None, []
    
    # Contar coincidencias
    n_matches = match_mask.sum()
    print(f"📊 Coincidencias encontradas: {n_matches} de {len(svea_df)}")
    
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
            # Mostrar separaciones mínimas para diagnóstico
            print("🔍 Separaciones mínimas encontradas:")
            for i in range(min(5, len(d2d))):
                print(f"   Objeto {i+1}: {d2d[i].to(u.arcsec).value:.1f} arcsec")
            return pd.DataFrame(), []
    
    # Crear DataFrame con coincidencias
    coincidencias = []
    t17ids_coincidentes = []
    
    print(f"\n📋 DETALLES DE COINCIDENCIAS:")
    for i, (svea_idx, phot_idx, sep) in enumerate(zip(
        np.where(match_mask)[0], 
        idx_svea_in_phot[match_mask], 
        d2d[match_mask]
    )):
        svea_row = svea_df.iloc[svea_idx]
        phot_row = phot_df.iloc[phot_idx]
        
        svea_id = svea_row['ID']
        t17id = phot_row['T17ID']
        
        # Obtener [Fe/H] si está disponible
        feh_svea = svea_row.get('FeH', np.nan)
        feh_err_svea = svea_row.get('FeH_err', np.nan)
        
        coincidencia = {
            'Svea_ID': svea_id,
            'T17ID': t17id,
            'RA_Svea': svea_row['RA'],
            'Dec_Svea': svea_row['Dec'],
            'RA_phot': phot_row['RAJ2000'],
            'Dec_phot': phot_row['DEJ2000'],
            'Sep_arcsec': sep.to(u.arcsec).value,
            'FeH_Svea': feh_svea,
            'FeH_err_Svea': feh_err_svea
        }
        
        coincidencias.append(coincidencia)
        t17ids_coincidentes.append(t17id)
        
        print(f"   {i+1:2d}. {svea_id:12s} → {t17id:10s} (sep: {sep.to(u.arcsec).value:.2f} arcsec)")
    
    coincidencias_df = pd.DataFrame(coincidencias)
    
    # Guardar archivo con coincidencias
    output_coincidencias = 'coincidencias_svea_phot.csv'
    coincidencias_df.to_csv(output_coincidencias, index=False)
    print(f"\n💾 Coincidencias guardadas en: {output_coincidencias}")
    
    return coincidencias_df, t17ids_coincidentes

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
        print(f"   IDs buscados: {t17ids_coincidentes[:5]}...")  # Mostrar solo primeros 5
        print(f"   Primeros IDs en catálogo: {phot_df['T17ID'].head().tolist()}")
        return None
    
    print(f"✅ Catálogo filtrado: {len(df_filtrado)} objetos encontrados")
    
    # Mostrar qué objetos se encontraron
    print(f"\n📋 OBJETOS ENCONTRADOS:")
    encontrados = []
    no_encontrados = []
    
    for t17id in t17ids_coincidentes:
        if t17id in df_filtrado['T17ID'].values:
            encontrados.append(t17id)
        else:
            no_encontrados.append(t17id)
    
    for t17id in encontrados[:10]:  # Mostrar solo primeros 10
        print(f"   ✓ {t17id}: ENCONTRADO")
    
    if len(encontrados) > 10:
        print(f"   ... y {len(encontrados) - 10} más")
    
    if no_encontrados:
        print(f"\n⚠️  OBJETOS NO ENCONTRADOS ({len(no_encontrados)}):")
        for t17id in no_encontrados[:5]:
            print(f"   ✗ {t17id}: NO ENCONTRADO")
        if len(no_encontrados) > 5:
            print(f"   ... y {len(no_encontrados) - 5} más")
    
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
    print(f"   • Objetos: {', '.join(objetos_procesados)}")
    print(f"   • Redshift: {redshift:.6f} (mismo para todos)")
    print(f"   • Columnas: {len(header_list)}")
    print(f"   • Filtros: 7 S-PLUS + 5 DECam = 12 filtros")
    
    # Mostrar estadísticas de detección
    print(f"\n📊 ESTADÍSTICAS DE FLUJO (primer objeto):")
    obj1_data = data_rows[0]
    for i, band in enumerate(['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']):
        flux_idx = 2 + i*2
        if flux_idx < len(obj1_data):
            flux = obj1_data[flux_idx]
            print(f"   • {band}: {flux:.2e} mJy")
    
    # Mostrar ejemplo
    print(f"\n📋 EJEMPLO (primer objeto - {objetos_procesados[0]}):")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(f"  Encabezado: {lines[0].strip()}")
        if len(lines) > 1:
            # Mostrar primeros 5 valores de datos
            data_parts = lines[1].strip().split()
            print(f"  Datos (primeros 5 valores): {data_parts[:5]}...")
    
    return True

def main():
    """Función principal"""
    
    print("=" * 70)
    print("🔍 PREPARADOR DE DATOS CIGALE (COINCIDENCIAS AUTOMÁTICAS)")
    print("=" * 70)
    print("Busca coincidencias por coordenadas entre catálogo Svea y fotometría")
    print("Genera archivo de entrada para CIGALE solo con coincidencias")
    print("=" * 70)
    
    # Configuración
    svea_csv = "../../globular_clusters_ngc5128.csv"  # Catálogo de Svea
    phot_csv = "../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"  # Catálogo de fotometría
    max_sep_arcsec = 3.0  # Tolerancia en arcseconds
    
    print(f"📁 Ruta Svea: {svea_csv}")
    print(f"📁 Ruta fotometría: {phot_csv}")
    print(f"🎯 Tolerancia: {max_sep_arcsec} arcsec")
    
    # Verificar si los archivos existen
    if not os.path.exists(svea_csv):
        print(f"❌ Archivo no encontrado: {svea_csv}")
        # Intentar ruta alternativa
        svea_csv = "globular_clusters_ngc5128.csv"
        print(f"   Intentando: {svea_csv}")
        if not os.path.exists(svea_csv):
            print("❌ No se pudo encontrar el catálogo de Svea")
            return False
    
    if not os.path.exists(phot_csv):
        print(f"❌ Archivo no encontrado: {phot_csv}")
        # Intentar ruta alternativa
        phot_csv = "all_fields_photometry_OPTIMIZED_1394objects.csv"
        print(f"   Intentando: {phot_csv}")
        if not os.path.exists(phot_csv):
            print("❌ No se pudo encontrar el catálogo de fotometría")
            return False
    
    # 1. Buscar coincidencias automáticamente
    coincidencias_df, t17ids_coincidentes = buscar_coincidencias_por_coordenadas(
        svea_csv, phot_csv, max_sep_arcsec
    )
    
    if coincidencias_df is None or len(t17ids_coincidentes) == 0:
        print("❌ No se pueden continuar sin coincidencias")
        return False
    
    # 2. Filtrar catálogo de fotometría por coincidencias
    df_filtrado = filtrar_por_coincidencias(phot_csv, t17ids_coincidentes)
    
    if df_filtrado is None:
        print("❌ No se pudo filtrar el catálogo de fotometría")
        return False
    
    # 3. Preparar archivo para CIGALE
    output_file = "gc_splus_cigale_coincidencias.txt"
    if not preparar_datos_para_cigale(df_filtrado, output_file):
        print("❌ Error creando archivo para CIGALE")
        return False
    
    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   1. {output_file} → Datos para CIGALE ({len(t17ids_coincidentes)} objetos)")
    print(f"   2. coincidencias_svea_phot.csv → Detalles de coincidencias")
    
    print(f"\n🚀 PARA USAR CON CIGALE:")
    print(f"   1. Configura pcigale.ini con:")
    print(f"      data_file = {output_file}")
    print(f"      redshift = from_file")
    print(f"   2. Ejecuta: pcigale run")
    
    print(f"\n📊 RESUMEN DE COINCIDENCIAS:")
    print(f"   • Total: {len(t17ids_coincidentes)} objetos")
    print(f"   • Separación máxima: {max_sep_arcsec} arcsec")
    print(f"   • Tiempo estimado para CIGALE: < 5 minutos")
    
    # Mostrar valores de [Fe/H] de Svea para referencia
    if 'FeH_Svea' in coincidencias_df.columns:
        print(f"\n🎯 VALORES DE REFERENCIA (Svea):")
        for _, row in coincidencias_df.iterrows():
            print(f"   • {row['Svea_ID']:12s} ([Fe/H] = {row['FeH_Svea']:.2f} ± {row.get('FeH_err_Svea', 0.0):.2f})")
    
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
