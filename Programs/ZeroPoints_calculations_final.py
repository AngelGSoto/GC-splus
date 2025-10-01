#!/usr/bin/env python3
"""
ZeroPoints_calculations_NO_APER_CORR.py
VERSIÓN para archivos SIN CORRECCIONES DE APERTURA
"""

import pandas as pd
import numpy as np
import json
import glob
import os
import logging
from pathlib import Path
from astropy.stats import sigma_clipped_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

# ✅ MAPEO CRÍTICO: Filtros CSV -> Filtros JSON
FILTER_MAP_CSV_TO_JSON = {
    'F378': 'F0378',
    'F395': 'F0395', 
    'F410': 'F0410',
    'F430': 'F0430',
    'F515': 'F0515', 
    'F660': 'F0660',
    'F861': 'F0861'
}

def find_json_file(field_name, source_id):
    """Buscar archivo JSON - COMPATIBLE CON FORMATO REAL"""
    json_dir = f"gaia_spectra_{field_name}"
    
    if not os.path.exists(json_dir):
        logging.warning(f"JSON directory not found: {json_dir}")
        return None
    
    # ✅ PATRONES basados en estructura real
    patterns = [
        f"{json_dir}/gaia_xp_spectrum_{source_id}-Ref-SPLUS21-magnitude.json",
        f"{json_dir}/gaia_xp_spectrum_{source_id}-Ref.json",
        f"{json_dir}/{source_id}-Ref-SPLUS21-magnitude.json",
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            logging.debug(f"JSON encontrado: {pattern}")
            return pattern
    
    # ✅ Búsqueda flexible
    json_pattern = f"{json_dir}/*{source_id}*.json"
    matches = glob.glob(json_pattern)
    if matches:
        logging.debug(f"JSON encontrado por patrón: {matches[0]}")
        return matches[0]
    
    logging.debug(f"JSON NO encontrado para source_id: {source_id}")
    return None

def calculate_zero_points_no_aper_corr(csv_file):
    """Calcula zero-points para archivos SIN correcciones de apertura"""
    # ✅ EXTRAER NOMBRE DEL CAMPO del nuevo formato de archivo
    filename = Path(csv_file).stem
    
    if '_gaia_xp_matches_3arcsec' in filename:
        field_name = filename.replace('_gaia_xp_matches_3arcsec', '')
    else:
        field_name = filename.replace('_gaia_xp_matches', '').replace('_splus', '')
    
    logging.info(f"🎯 PROCESANDO {field_name} desde {csv_file} (SIN CORRECCIONES APERTURA)")
    
    try:
        # ✅ Leer CSV
        df = pd.read_csv(csv_file, dtype={'source_id': str})
        if 'source_id' in df.columns:
            df['source_id'] = df['source_id'].astype(str).str.strip()
    except Exception as e:
        logging.error(f"❌ Error leyendo {csv_file}: {e}")
        return field_name, None
    
    # ✅ VERIFICAR COLUMNAS DISPONIBLES
    all_columns = df.columns.tolist()
    logging.info(f"📊 Columnas disponibles en CSV: {len(all_columns)}")
    
    # ✅ IDENTIFICAR COLUMNAS DE MAGNITUD DEL NUEVO SCRIPT 1
    available_mag_columns = {}
    for filt in SPLUS_FILTERS:
        mag_col = f'mag_inst_3.0_{filt}'
        if mag_col in df.columns:
            available_mag_columns[filt] = mag_col
            logging.info(f"✅ {filt}: Columna '{mag_col}' encontrada")
            
            # ✅ VERIFICAR VALORES DE CORRECCIÓN DE APERTURA (deberían ser 0.0)
            ac_col = f'aper_corr_3.0_{filt}'
            if ac_col in df.columns:
                unique_ac = df[ac_col].unique()
                if len(unique_ac) == 1 and unique_ac[0] == 0.0:
                    logging.info(f"✅ {filt}: Correcciones de apertura = 0.0 (correcto)")
                else:
                    logging.warning(f"⚠️ {filt}: Correcciones de apertura no son 0.0: {unique_ac}")
        else:
            logging.warning(f"⚠️ {filt}: Columna '{mag_col}' NO encontrada")
    
    if len(available_mag_columns) < 3:
        logging.error(f"❌ Solo {len(available_mag_columns)} filtros tienen datos")
        return field_name, None
    
    # ✅ IDENTIFICAR COLUMNA DE ID
    id_column = 'source_id' if 'source_id' in df.columns else None
    if not id_column:
        logging.error(f"❌ Columna 'source_id' no encontrada")
        return field_name, None
    
    logging.info(f"✅ Usando columna de ID: {id_column}")
    
    # ✅ PROCESAR CADA ESTRELLA
    zp_data = {filt: [] for filt in SPLUS_FILTERS}
    stats = {
        'total_stars': len(df),
        'stars_with_json': 0,
        'stars_with_photometry': 0,
        'json_not_found': 0,
        'json_errors': 0
    }
    
    # ✅ EJEMPLOS PARA DIAGNÓSTICO
    example_count = 0
    example_data = []
    
    for idx, row in df.iterrows():
        source_id = str(row[id_column]).strip()
        
        if not source_id or source_id in ['nan', 'None', '']:
            continue
        
        # Buscar archivo JSON
        json_file = find_json_file(field_name, source_id)
        if not json_file:
            stats['json_not_found'] += 1
            continue
        
        try:
            with open(json_file, 'r') as f:
                synth_data = json.load(f)
            stats['stars_with_json'] += 1
                
        except Exception as e:
            stats['json_errors'] += 1
            continue
        
        # ✅ PROCESAR CADA FILTRO
        for filt in SPLUS_FILTERS:
            if filt not in available_mag_columns:
                continue
                
            mag_col = available_mag_columns[filt]
            inst_mag = row[mag_col]
            
            if pd.isna(inst_mag):
                continue
            
            # ✅ CLAVE EN JSON - USAR MAPEO CORRECTO
            json_key = FILTER_MAP_CSV_TO_JSON[filt]
            
            if json_key not in synth_data:
                # Intentar alternativas
                json_key_alt = filt  # Por si acaso usa el mismo nombre
                if json_key_alt not in synth_data:
                    continue
                json_key = json_key_alt
            
            synth_mag = synth_data[json_key]
            
            if synth_mag is None:
                continue
            
            # ✅ CÁLCULO DE ZERO POINT
            zp = synth_mag - inst_mag
            
            # ✅ FILTRO FÍSICO (zero points típicos para S-PLUS SIN correcciones)
            # SIN correcciones, los zero points deberían ser MÁS ALTOS
            if np.isfinite(zp) and 15 < zp < 30:  # Rango típico para S-PLUS
                zp_data[filt].append(zp)
                
                # ✅ GUARDAR EJEMPLOS PARA DIAGNÓSTICO
                if example_count < 3:
                    example_data.append({
                        'source_id': source_id,
                        'filter': filt,
                        'inst_mag': inst_mag,
                        'synth_mag': synth_mag,
                        'zp': zp
                    })
                    example_count += 1
    
    # ✅ REPORTAR EJEMPLOS DE VERIFICACIÓN
    if example_data:
        logging.info("🔍 EJEMPLOS DE CÁLCULO (SIN CORRECCIONES APERTURA):")
        for example in example_data:
            logging.info(f"  Source: {example['source_id']}, {example['filter']}: "
                        f"inst={example['inst_mag']:.3f}, synth={example['synth_mag']:.3f}, ZP={example['zp']:.3f}")
    
    # ✅ ESTADÍSTICAS FINALES
    logging.info(f"📊 ESTADÍSTICAS {field_name}:")
    logging.info(f"  Estrellas totales: {stats['total_stars']}")
    logging.info(f"  Con JSON encontrado: {stats['stars_with_json']}")
    logging.info(f"  JSON no encontrado: {stats['json_not_found']}")
    logging.info(f"  Errores JSON: {stats['json_errors']}")
    
    # ✅ CALCULAR ZERO POINTS POR FILTRO
    zp_stats = {}
    for filt in SPLUS_FILTERS:
        zps = np.array(zp_data[filt])
        
        if len(zps) >= 5:
            try:
                mean, median, std = sigma_clipped_stats(zps, sigma=2.5)
            except:
                mean, median, std = np.mean(zps), np.median(zps), np.std(zps)
            
            zp_stats[filt] = {
                'mean': float(mean),
                'median': float(median), 
                'std': float(std),
                'n_stars': int(len(zps)),
                'min': float(np.min(zps)),
                'max': float(np.max(zps))
            }
            
            status = "✅" if len(zps) >= 10 else "⚠️ "
            logging.info(f"{status} {filt}: ZP = {median:.3f} ± {std:.3f} (N={len(zps)})")
            
        elif len(zps) > 0:
            median = np.median(zps)
            std = np.std(zps)
            zp_stats[filt] = {
                'mean': float(np.mean(zps)),
                'median': float(median),
                'std': float(std),
                'n_stars': int(len(zps)),
                'min': float(np.min(zps)),
                'max': float(np.max(zps))
            }
            logging.warning(f"⚠️  {filt}: ZP = {median:.3f} ± {std:.3f} (N={len(zps)}) - POCAS ESTRELLAS!")
        else:
            zp_stats[filt] = None
            logging.error(f"❌ {filt}: Sin datos de zero-point")
    
    return field_name, zp_stats

def main():
    """Función principal"""
    # ✅ BUSCAR ARCHIVOS CSV NUEVOS (SIN CORRECCIONES APERTURA)
    csv_patterns = [
        "CenA*_gaia_xp_matches_3arcsec.csv",  # Nuevos archivos
        
    ]
    
    csv_files = []
    for pattern in csv_patterns:
        matches = glob.glob(pattern)
        csv_files.extend(matches)
        if matches:
            logging.info(f"Found {len(matches)} files with pattern: {pattern}")
    
    # Eliminar duplicados
    csv_files = list(set(csv_files))
    
    if not csv_files:
        logging.error("No CSV files found with expected patterns!")
        logging.info("Available CSV files:")
        for f in glob.glob("*.csv"):
            logging.info(f"  {f}")
        return
    
    logging.info(f"Processing {len(csv_files)} CSV files: {csv_files}")
    
    # ✅ PROCESAR CADA ARCHIVO
    all_stats = {}
    for csv_file in csv_files:
        field, stats = calculate_zero_points_no_aper_corr(csv_file)
        if stats:
            all_stats[field] = stats
        else:
            logging.error(f"❌ Failed to process {field}")
    
    # ✅ GUARDAR RESULTADOS
    if all_stats:
        os.makedirs('Results', exist_ok=True)
        
        # Formato detallado
        detailed_data = []
        for field, stats in all_stats.items():
            for filt in SPLUS_FILTERS:
                if stats.get(filt):
                    row = {
                        'field': field,
                        'filter': filt,
                        'zp_mean': stats[filt]['mean'],
                        'zp_median': stats[filt]['median'],
                        'zp_std': stats[filt]['std'],
                        'n_stars': stats[filt]['n_stars']
                    }
                    detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv('Results/all_fields_zero_points_detailed_3arcsec.csv', 
                          index=False, float_format='%.6f')
        logging.info("✅ Detailed results saved (NO APER CORR)")
        
        # Formato SPLUS
        splus_data = []
        for field, stats in all_stats.items():
            row = {'field': field}
            for filt in SPLUS_FILTERS:
                row[filt] = stats[filt]['median'] if stats.get(filt) else np.nan
            splus_data.append(row)
        
        df_splus = pd.DataFrame(splus_data)
        df_splus.to_csv('Results/all_fields_zero_points_splus_format_3arcsec.csv', 
                       index=False, float_format='%.6f')
        logging.info("✅ SPLUS format saved (NO APER CORR)")
        
        # ✅ REPORTAR PROMEDIOS
        logging.info("📊 PROMEDIOS FINALES (SIN CORRECCIONES APERTURA):")
        for filt in SPLUS_FILTERS:
            zps = [stats[filt]['median'] for stats in all_stats.values() if stats.get(filt)]
            if zps:
                avg = np.mean(zps)
                std = np.std(zps)
                logging.info(f"  {filt}: {avg:.3f} ± {std:.3f} (N={len(zps)} campos)")
        
        # ✅ COMPARAR CON ZERO POINTS ANTERIORES (si existen)
        old_zp_file = 'Results/all_fields_zero_points_splus_format_compatible.csv'
        if os.path.exists(old_zp_file):
            logging.info("🔍 COMPARACIÓN CON ZERO POINTS ANTERIORES:")
            old_df = pd.read_csv(old_zp_file)
            for filt in SPLUS_FILTERS:
                if filt in old_df.columns:
                    old_median = old_df[filt].median()
                    new_median = np.mean([stats[filt]['median'] for stats in all_stats.values() if stats.get(filt)])
                    diff = new_median - old_median
                    logging.info(f"  {filt}: Antes={old_median:.3f}, Ahora={new_median:.3f}, Δ={diff:.3f}")
        
    else:
        logging.error("❌ No zero-points calculated!")

if __name__ == "__main__":
    main()
