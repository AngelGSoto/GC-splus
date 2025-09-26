#!/usr/bin/env python3
"""
Simple batch zero point calculation for SPLUS fields
Direct processing without external scripts - CORRECTED VERSION
"""
import pandas as pd
import numpy as np
import json
import glob
import os
import logging
from pathlib import Path
from astropy.stats import sigma_clipped_stats

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('zero_points_batch_simple_corrected.log')
    ]
)

# CORRECCIÓN: Los filtros en los JSON tienen formato F0378, F0395, etc.
SPLUS_FILTERS_JSON = ['F0378', 'F0395', 'F0410', 'F0430', 'F0515', 'F0660', 'F0861']
# Las columnas en el CSV tienen formato F378, F395, etc.
SPLUS_FILTERS_CSV = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']

# Mapeo entre nombres de filtros en JSON y CSV
FILTER_MAPPING = {
    'F0378': 'F378',
    'F0395': 'F395', 
    'F0410': 'F410',
    'F0430': 'F430',
    'F0515': 'F515',
    'F0660': 'F660',
    'F0861': 'F861'
}

INST_MAG_COLS = [f'mag_inst_calibrated_{filt}' for filt in SPLUS_FILTERS_CSV]

def safe_convert_source_id(source_id_value):
    """Convertir source_id de notación científica a entero"""
    try:
        if pd.isna(source_id_value):
            return None
        
        source_id_str = str(source_id_value).strip()
        
        if 'E+' in source_id_str.upper() or 'E-' in source_id_str.upper():
            source_id_float = float(source_id_str)
            return str(int(source_id_float))
        else:
            return str(int(float(source_id_str)))
    except (ValueError, TypeError):
        return None

def find_json_file(field_name, source_id):
    """Encontrar archivo JSON con magnitudes sintéticas"""
    json_dir = f"gaia_spectra_{field_name}"
    
    if not os.path.exists(json_dir):
        logging.warning(f"JSON directory not found: {json_dir}")
        return None
    
    # Patrones de nombres posibles
    patterns = [
        f"{json_dir}/gaia_xp_spectrum_{source_id}-Ref-SPLUS21-magnitude.json",
        f"{json_dir}/gaia_xp_spectrum_{source_id}-SPLUS21-magnitude.json",
        f"{json_dir}/{source_id}-Ref-SPLUS21-magnitude.json"
    ]
    
    for pattern in patterns:
        if os.path.exists(pattern):
            logging.debug(f"Found JSON file: {pattern}")
            return pattern
    
    logging.debug(f"JSON file not found for source_id {source_id} in {json_dir}")
    return None

def calculate_field_zero_points(csv_file):
    """Calcular zero points para un campo específico"""
    field_name = Path(csv_file).stem.replace('_gaia_xp_matches_splus_method', '')
    logging.info(f"Processing field: {field_name}")
    
    try:
        # Leer datos instrumentales
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(df)} stars from {csv_file}")
        
        # Verificar que tenemos las columnas necesarias
        missing_cols = [col for col in INST_MAG_COLS if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")
        
        available_cols = [col for col in INST_MAG_COLS if col in df.columns]
        
        if not available_cols:
            logging.error(f"No instrumental magnitude columns found")
            return None, None
            
    except Exception as e:
        logging.error(f"Error reading {csv_file}: {e}")
        return None, None
    
    # Inicializar almacenamiento de zero points
    zp_data = {filt: [] for filt in SPLUS_FILTERS_CSV}
    valid_stars = 0
    json_files_found = 0
    json_files_processed = 0
    
    # Procesar cada estrella
    for idx, row in df.iterrows():
        if idx % 100 == 0 and idx > 0:
            logging.info(f"  Processing star {idx}/{len(df)}")
        
        # Obtener source_id
        source_id = None
        if 'source_id' in row and pd.notna(row['source_id']):
            source_id = safe_convert_source_id(row['source_id'])
        
        if not source_id:
            continue
        
        # Encontrar archivo JSON
        json_file = find_json_file(field_name, source_id)
        if not json_file:
            continue
        
        json_files_found += 1
        
        # Leer magnitudes sintéticas
        try:
            with open(json_file, 'r') as f:
                synth_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.debug(f"Error reading {json_file}: {e}")
            continue
        
        json_files_processed += 1
        
        # Calcular zero points para cada filtro disponible
        filters_processed = 0
        for inst_col in available_cols:
            # Extraer nombre del filtro del CSV (ej: 'F378' de 'mag_inst_calibrated_F378')
            filt_csv = inst_col.replace('mag_inst_calibrated_', '')
            
            # Obtener el nombre correspondiente en el JSON
            filt_json = None
            for json_filt, csv_filt in FILTER_MAPPING.items():
                if csv_filt == filt_csv:
                    filt_json = json_filt
                    break
            
            if not filt_json:
                continue
            
            # Obtener magnitud instrumental
            inst_mag = row[inst_col]
            if pd.isna(inst_mag) or abs(inst_mag) >= 50.0:
                continue
            
            # Obtener magnitud sintética del JSON
            synth_mag = synth_data.get(filt_json)
            if synth_mag is None or pd.isna(synth_mag) or abs(synth_mag) >= 50.0:
                continue
            
            # Calcular zero point
            zp = synth_mag - inst_mag
            
            # Filtrar valores razonables
            if 5.0 < zp < 30.0:  # Rango ampliado para S-PLUS
                zp_data[filt_csv].append(zp)
                filters_processed += 1
                logging.debug(f"  {filt_csv}: synth={synth_mag:.3f}, inst={inst_mag:.3f}, zp={zp:.3f}")
        
        if filters_processed > 0:
            valid_stars += 1
    
    logging.info(f"Field {field_name}: {json_files_found} JSON found, {json_files_processed} processed, {valid_stars} valid stars")
    
    # Calcular estadísticas para cada filtro
    zp_stats = {}
    for filt in SPLUS_FILTERS_CSV:
        zps = zp_data[filt]
        
        if len(zps) > 0:
            zps_array = np.array(zps)
            
            # Estadísticas básicas
            median_zp = np.median(zps_array)
            mean_zp = np.mean(zps_array)
            std_zp = np.std(zps_array)
            mad = np.median(np.abs(zps_array - median_zp))
            std_mad = 1.4826 * mad
            
            # Estadísticas con sigma clipping
            if len(zps) > 5:
                mean_clipped, median_clipped, std_clipped = sigma_clipped_stats(zps_array, sigma=3.0)
            else:
                mean_clipped, median_clipped, std_clipped = mean_zp, median_zp, std_zp
            
            zp_stats[filt] = {
                'median': median_zp,
                'mean': mean_zp,
                'std': std_zp,
                'mad': mad,
                'std_mad': std_mad,
                'mean_clipped': mean_clipped,
                'median_clipped': median_clipped,
                'std_clipped': std_clipped,
                'n_stars': len(zps),
                'min': np.min(zps_array),
                'max': np.max(zps_array),
                'q25': np.percentile(zps_array, 25) if len(zps) >= 4 else np.nan,
                'q75': np.percentile(zps_array, 75) if len(zps) >= 4 else np.nan
            }
            
            status = "OK" if len(zps) >= 3 else "LOW"
            logging.info(f"  {filt}: {len(zps)} stars, ZP = {median_zp:.3f} ± {std_mad:.3f} ({status})")
        else:
            zp_stats[filt] = None
            logging.warning(f"  {filt}: No valid measurements")
    
    return field_name, zp_stats

def main():
    """Función principal"""
    # Encontrar todos los archivos CSV
    csv_files = glob.glob("CenA*_gaia_xp_matches_splus_method.csv")
    
    if not csv_files:
        logging.error("No CSV files found")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    # Procesar cada campo
    all_fields_stats = {}
    processed_fields = []
    failed_fields = []
    
    for csv_file in sorted(csv_files):
        field_name, zp_stats = calculate_field_zero_points(csv_file)
        
        if zp_stats is not None:
            all_fields_stats[field_name] = zp_stats
            processed_fields.append(field_name)
        else:
            failed_fields.append(field_name)
    
    # Crear directorio de resultados
    os.makedirs('Results', exist_ok=True)
    
    # Guardar resultados detallados por campo
    if all_fields_stats:
        # Archivo detallado (formato largo)
        detailed_rows = []
        for field_name, stats in all_fields_stats.items():
            for filt, data in stats.items():
                if data is not None:
                    detailed_rows.append({
                        'field': field_name,
                        'filter': filt,
                        'median_zp': data['median'],
                        'mean_zp': data['mean'],
                        'std_zp': data['std'],
                        'mad': data['mad'],
                        'std_mad': data['std_mad'],
                        'mean_clipped': data['mean_clipped'],
                        'median_clipped': data['median_clipped'],
                        'std_clipped': data['std_clipped'],
                        'n_stars': data['n_stars'],
                        'min_zp': data['min'],
                        'max_zp': data['max'],
                        'q25': data['q25'],
                        'q75': data['q75']
                    })
        
        if detailed_rows:
            df_detailed = pd.DataFrame(detailed_rows)
            detailed_file = 'Results/all_fields_zero_points_detailed.csv'
            df_detailed.to_csv(detailed_file, index=False)
            logging.info(f"✓ Detailed results saved to {detailed_file}")
        
        # Archivo formato SPLUS (campo, F378, F395, ...)
        splus_rows = []
        for field_name, stats in all_fields_stats.items():
            row = {'field': field_name}
            for filt in SPLUS_FILTERS_CSV:
                if stats.get(filt) is not None:
                    row[filt] = stats[filt]['median']
                else:
                    row[filt] = np.nan
            splus_rows.append(row)
        
        df_splus = pd.DataFrame(splus_rows)
        splus_file = 'Results/all_fields_zero_points_splus_format.csv'
        df_splus.to_csv(splus_file, index=False, float_format='%.6f')
        logging.info(f"✓ SPLUS format results saved to {splus_file}")
        
        # Calcular promedios entre campos
        avg_stats = {}
        for filt in SPLUS_FILTERS_CSV:
            zps = []
            for field_name, stats in all_fields_stats.items():
                if stats.get(filt) is not None:
                    zps.append(stats[filt]['median'])
            
            if zps:
                zps_array = np.array(zps)
                avg_stats[filt] = {
                    'median': np.median(zps_array),
                    'mean': np.mean(zps_array),
                    'std': np.std(zps_array),
                    'n_fields': len(zps)
                }
        
        # Guardar promedios
        avg_rows = []
        for filt, data in avg_stats.items():
            avg_rows.append({
                'filter': filt,
                'average_median_zp': data['median'],
                'average_mean_zp': data['mean'],
                'std_across_fields': data['std'],
                'n_fields': data['n_fields']
            })
        
        if avg_rows:
            df_avg = pd.DataFrame(avg_rows)
            avg_file = 'Results/average_zero_points_across_fields.csv'
            df_avg.to_csv(avg_file, index=False, float_format='%.6f')
            logging.info(f"✓ Average zero points saved to {avg_file}")
            
            # Mostrar resumen de promedios
            logging.info("\n=== AVERAGE ZERO POINTS ACROSS FIELDS ===")
            for _, row in df_avg.iterrows():
                logging.info(f"{row['filter']}: {row['average_median_zp']:.3f} ± {row['std_across_fields']:.3f} (from {row['n_fields']} fields)")
    
    # Resumen final
    logging.info("\n" + "="*50)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*50)
    logging.info(f"Total fields: {len(csv_files)}")
    logging.info(f"Successfully processed: {len(processed_fields)}")
    logging.info(f"Failed: {len(failed_fields)}")
    
    if processed_fields:
        logging.info("\n✓ Processed fields: " + ", ".join(sorted(processed_fields)))
    
    if failed_fields:
        logging.info("\n✗ Failed fields: " + ", ".join(sorted(failed_fields)))

if __name__ == "__main__":
    main()
