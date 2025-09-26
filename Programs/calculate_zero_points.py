#!/usr/bin/env python3
"""
Calculate zero points for SPLUS filters using instrumental and synthetic magnitudes
Updated version for splus_method CSV files - CORRECTED VERSION
"""
from __future__ import print_function
import numpy as np
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from astropy.stats import sigma_clipped_stats
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Filter mapping between CSV columns and JSON keys
filter_mapping = {
    'mag_inst_calibrated_F378': 'F0378',
    'mag_inst_calibrated_F395': 'F0395', 
    'mag_inst_calibrated_F410': 'F0410',
    'mag_inst_calibrated_F430': 'F0430',
    'mag_inst_calibrated_F515': 'F0515',
    'mag_inst_calibrated_F660': 'F0660',
    'mag_inst_calibrated_F861': 'F0861'
}

def safe_convert_source_id(source_id_value):
    """
    Safely convert source_id from scientific notation or other formats
    """
    try:
        if pd.isna(source_id_value):
            return None
        
        # Convert to string first, then handle scientific notation
        source_id_str = str(source_id_value).strip()
        
        # Handle scientific notation (e.g., 6.09472427568452E+018)
        if 'E+' in source_id_str.upper() or 'E-' in source_id_str.upper():
            # Convert scientific notation to integer string
            source_id_float = float(source_id_str)
            source_id_int = int(source_id_float)
            return str(source_id_int)
        else:
            # Try direct conversion to int
            return str(int(float(source_id_str)))  # Handle float representations
    except (ValueError, TypeError) as e:
        logging.warning(f"Error converting source_id {source_id_value}: {e}")
        return None

def find_json_file(json_files_dir, source_id):
    """
    Find JSON file with multiple possible naming patterns
    """
    if not source_id:
        return None
    
    # Try different filename patterns
    patterns = [
        f"gaia_xp_spectrum_{source_id}-Ref-SPLUS21-magnitude.json",
        f"gaia_xp_spectrum_{source_id}-SPLUS21-magnitude.json",
        f"gaia_xp_spectrum_{source_id}_synthetic_mags.json",
        f"{source_id}-Ref-SPLUS21-magnitude.json",
        f"{source_id}-SPLUS21-magnitude.json"
    ]
    
    for pattern in patterns:
        json_file = os.path.join(json_files_dir, pattern)
        if os.path.exists(json_file):
            return json_file
    
    # Also check in parent directory if exists
    parent_dir = os.path.dirname(json_files_dir)
    for pattern in patterns:
        json_file = os.path.join(parent_dir, pattern)
        if os.path.exists(json_file):
            return json_file
    
    return None

def calculate_zero_points(csv_file, json_dir):
    """
    Calculate zero points for SPLUS filters
    
    Parameters:
    csv_file: CSV file with instrumental magnitudes
    json_dir: Directory containing JSON files with synthetic magnitudes
    """
    
    # Read instrumental photometry
    try:
        pho_inst = pd.read_csv(csv_file)
        logging.info(f"Successfully loaded CSV file: {csv_file}")
        logging.info(f"Number of stars: {len(pho_inst)}")
        logging.info(f"Columns found: {list(pho_inst.columns)}")
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}: {e}")
        return {}, {}
    
    # Extract field name from CSV filename
    basename = os.path.basename(csv_file)
    if '_gaia_xp_matches_splus_method.csv' in basename:
        field_name = basename.split('_gaia_xp_matches_splus_method.csv')[0]
    elif '_gaia_xp_matches_corrected.csv' in basename:
        field_name = basename.split('_gaia_xp_matches_corrected.csv')[0]
    else:
        field_name = basename.split('.csv')[0]
        logging.warning(f"Using fallback field name extraction: {field_name}")
    
    # Get the directory where JSON files are stored - CORRECTED APPROACH
    # Try multiple possible locations
    json_locations = [
        json_dir,  # User-provided directory
        os.path.join(json_dir, f"gaia_spectra_{field_name}"),
        os.path.join(os.path.dirname(csv_file), f"gaia_spectra_{field_name}"),
        os.path.dirname(csv_file),  # Same directory as CSV
        json_dir  # Fallback to user directory
    ]
    
    json_files_dir = None
    for location in json_locations:
        if os.path.exists(location):
            json_files_dir = location
            logging.info(f"Found JSON directory: {json_files_dir}")
            break
    
    if json_files_dir is None:
        logging.error(f"JSON directory not found in any location for field {field_name}")
        logging.error(f"Tried locations: {json_locations}")
        return {}, {}
    
    # Initialize arrays for zero point calculation
    zero_points = {filt: [] for filt in filter_mapping.keys()}
    n_stars = len(pho_inst)
    
    logging.info(f"Processing field {field_name} with {n_stars} stars")
    logging.info(f"Using JSON directory: {json_files_dir}")
    
    # Process each star
    valid_stars = 0
    for idx, row in pho_inst.iterrows():
        if idx % 100 == 0:  # Progress indicator
            logging.info(f"Processing star {idx+1}/{n_stars}")
        
        # Get source_id safely - CORRECTED
        source_id = None
        if 'source_id' in row and pd.notna(row['source_id']):
            source_id = safe_convert_source_id(row['source_id'])
        
        if not source_id:
            logging.debug(f"Skipping row {idx}: No valid source_id")
            continue
        
        # Find JSON file - CORRECTED
        json_file = find_json_file(json_files_dir, source_id)
        
        if not json_file:
            logging.debug(f"JSON file not found for source_id {source_id}")
            continue
        
        # Load synthetic magnitudes
        try:
            with open(json_file) as f:
                synth_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            logging.warning(f"Could not load JSON file {json_file}: {e}")
            continue
        
        # Calculate zero points for each filter
        filters_processed = 0
        for inst_filt, synth_filt in filter_mapping.items():
            # Check if the instrumental magnitude column exists
            if inst_filt not in row:
                continue
                
            inst_mag = row[inst_filt]
            
            # Skip invalid magnitudes
            if pd.isna(inst_mag) or abs(inst_mag) >= 50.0:
                continue
            
            if synth_filt in synth_data:
                synth_mag = synth_data[synth_filt]
                
                # Skip invalid synthetic magnitudes
                if pd.isna(synth_mag) or abs(synth_mag) >= 50.0:
                    continue
                
                # Zero point calculation: synthetic mag - instrumental mag
                zp = synth_mag - inst_mag
                
                # Filter reasonable zero points (typically between 15-25 for S-PLUS)
                if 10.0 < zp < 30.0:
                    zero_points[inst_filt].append(zp)
                    filters_processed += 1
                else:
                    logging.debug(f"ZP out of range for {inst_filt}: {zp:.3f}")
        
        if filters_processed > 0:
            valid_stars += 1
    
    logging.info(f"Processed {valid_stars} valid stars with synthetic photometry")
    
    # Calculate statistics for each filter
    zp_results = {}
    for filt, zp_values in zero_points.items():
        if len(zp_values) > 0:  # Show stats even if <5
            zp_array = np.array(zp_values)
            
            # Basic statistics
            median_zp = np.median(zp_array)
            mad = np.median(np.abs(zp_array - median_zp))
            std_mad = 1.4826 * mad
            
            # Sigma-clipped statistics
            if len(zp_values) > 5:
                mean_clipped, median_clipped, std_clipped = sigma_clipped_stats(zp_array, sigma=3.0)
            else:
                mean_clipped, median_clipped, std_clipped = np.mean(zp_array), np.median(zp_array), np.std(zp_array)
            
            zp_results[filt] = {
                'median': median_zp,
                'mad': mad,
                'std_mad': std_mad,
                'mean': np.mean(zp_array),
                'std': np.std(zp_array),
                'mean_clipped': mean_clipped,
                'median_clipped': median_clipped,
                'std_clipped': std_clipped,
                'n_stars': len(zp_values),
                'min': np.min(zp_array),
                'max': np.max(zp_array),
                'q25': np.percentile(zp_array, 25) if len(zp_values) >= 4 else np.nan,
                'q75': np.percentile(zp_array, 75) if len(zp_values) >= 4 else np.nan
            }
            
            status = "SUFFICIENT" if len(zp_values) >= 5 else "INSUFFICIENT"
            logging.info(f"{filt}: {len(zp_values)} measurements, Median ZP = {median_zp:.3f} ± {std_mad:.3f} ({status})")
        else:
            logging.warning(f"{filt}: No valid measurements")
            zp_results[filt] = None
    
    return zp_results, zero_points

# ... (las funciones plot_zero_points y main se mantienen igual, pero actualicé el mínimo de estrellas)

def main():
    parser = argparse.ArgumentParser(
        description="""Calculate zero points for SPLUS filters using robust median statistics
                     for splus_method instrumental photometry - CORRECTED VERSION""")
    
    parser.add_argument("CSV", type=str,
                        help="CSV file with instrumental magnitudes (_gaia_xp_matches_splus_method.csv)")
    
    parser.add_argument("--json-dir", type=str, default=".",
                        help="Directory containing JSON files with synthetic magnitudes")
    
    parser.add_argument("--plot", action="store_true",
                        help="Create plot of zero point distributions")
    
    parser.add_argument("--min-stars", type=int, default=3,  # Reduced minimum
                        help="Minimum number of stars required per filter (default: 3)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.CSV):
        logging.error(f"Input CSV file not found: {args.CSV}")
        return
    
    # Calculate zero points
    zp_results, all_zp_values = calculate_zero_points(args.CSV, args.json_dir)
    
    # Extract field name for output files
    basename = os.path.basename(args.CSV)
    if '_gaia_xp_matches_splus_method.csv' in basename:
        field_name = basename.split('_gaia_xp_matches_splus_method.csv')[0]
    elif '_gaia_xp_matches_corrected.csv' in basename:
        field_name = basename.split('_gaia_xp_matches_corrected.csv')[0]
    else:
        field_name = basename.split('.csv')[0]
    
    # Save results to file
    output_file = f'{field_name}_zero_points_splus_method.csv'
    
    try:
        with open(output_file, 'w') as f:
            f.write("Filter,Median_ZP,MAD,STD_MAD,Mean_ZP,STD,Mean_Clipped,Median_Clipped,STD_Clipped,N_Stars,Min,Max,Q25,Q75\n")
            for filt, data in zp_results.items():
                if data is not None and data['n_stars'] >= args.min_stars:
                    f.write(f"{filt},{data['median']:.6f},{data['mad']:.6f},{data['std_mad']:.6f},"
                           f"{data['mean']:.6f},{data['std']:.6f},"
                           f"{data['mean_clipped']:.6f},{data['median_clipped']:.6f},{data['std_clipped']:.6f},"
                           f"{data['n_stars']},{data['min']:.6f},{data['max']:.6f},"
                           f"{data['q25']:.6f},{data['q75']:.6f}\n")
                else:
                    n_stars = data['n_stars'] if data is not None else 0
                    f.write(f"{filt},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,{n_stars},NaN,NaN,NaN,NaN\n")
        
        logging.info(f"Results saved to {output_file}")
        
        # Create plot if requested
        if args.plot and zp_results:
            plot_zero_points(zp_results, field_name, all_zp_values)
        
        # Print summary
        valid_filters = [f for f, d in zp_results.items() 
                        if d is not None and d['n_stars'] >= args.min_stars]
        
        if valid_filters:
            logging.info(f"✓ Successfully calculated zero points for {len(valid_filters)} filters:")
            for filt in valid_filters:
                data = zp_results[filt]
                logging.info(f"  {filt}: ZP = {data['median']:.3f} ± {data['std_mad']:.3f} (n={data['n_stars']})")
        else:
            logging.warning("✗ No valid zero points calculated for any filter")
            
    except Exception as e:
        logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
