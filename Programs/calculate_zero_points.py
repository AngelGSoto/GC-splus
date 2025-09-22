#!/usr/bin/env python3
"""
Calculate zero points for SPLUS filters using instrumental and synthetic magnitudes
Corrected version for consistency with the new corrected catalogs
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
    'mag_inst_F378': 'F0378',
    'mag_inst_F395': 'F0395', 
    'mag_inst_F410': 'F0410',
    'mag_inst_F430': 'F0430',
    'mag_inst_F515': 'F0515',
    'mag_inst_F660': 'F0660',
    'mag_inst_F861': 'F0861'
}

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
        logging.info(f"Columns found: {list(pho_inst.columns)}")
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}: {e}")
        return {}, {}
    
    # Extract field name from CSV filename
    if '_gaia_xp_matches_corrected.csv' in csv_file:
        field_name = os.path.basename(csv_file).split('_gaia_xp_matches_corrected.csv')[0]
    else:
        # Fallback for different naming conventions
        field_name = os.path.basename(csv_file).split('_gaia_xp_matches')[0]
        logging.warning(f"Using fallback field name extraction: {field_name}")
    
    # Get the directory where JSON files are stored
    json_files_dir = os.path.join(json_dir, f"gaia_spectra_{field_name}")
    
    # Check if JSON directory exists
    if not os.path.exists(json_files_dir):
        logging.warning(f"JSON directory not found: {json_files_dir}")
        # Try alternative location
        json_files_dir_alt = os.path.join(os.path.dirname(csv_file), f"gaia_spectra_{field_name}")
        if os.path.exists(json_files_dir_alt):
            json_files_dir = json_files_dir_alt
            logging.info(f"Using alternative JSON directory: {json_files_dir}")
        else:
            logging.error(f"JSON directory not found in any location for field {field_name}")
            return {}, {}
    
    # Initialize arrays for zero point calculation
    zero_points = {filt: [] for filt in filter_mapping.keys()}
    n_stars = len(pho_inst)
    
    logging.info(f"Processing field {field_name} with {n_stars} stars")
    logging.info(f"JSON directory: {json_files_dir}")
    
    # Process each star
    valid_stars = 0
    for idx, row in pho_inst.iterrows():
        # Check for spectrum_file column with different possible names
        spectrum_file = None
        for col_name in ['spectrum_file', 'spectrum_file_path', 'gaia_spectrum']:
            if col_name in row and pd.notna(row[col_name]):
                spectrum_file = row[col_name]
                break
        
        if spectrum_file is None:
            # Try to construct from source_id
            if 'source_id' in row and pd.notna(row['source_id']):
                source_id = str(int(row['source_id'])) if pd.notna(row['source_id']) else None
                if source_id:
                    spectrum_file = f"gaia_spectra_{field_name}/gaia_xp_spectrum_{source_id}.dat"
            else:
                continue
        
        # Skip NaN values or invalid entries
        if pd.isna(spectrum_file) or not isinstance(spectrum_file, str):
            continue
        
        # Extract source ID from spectrum file name
        try:
            # Handle different filename formats
            if 'gaia_xp_spectrum_' in spectrum_file:
                source_id = spectrum_file.split('gaia_xp_spectrum_')[-1].split('.')[0]
            else:
                # Assume source_id is already in the row
                source_id = str(int(row['source_id'])) if 'source_id' in row and pd.notna(row['source_id']) else None
            
            if not source_id:
                continue
                
            json_file = os.path.join(json_files_dir, f"gaia_xp_spectrum_{source_id}-Ref-SPLUS21-magnitude.json")
            
        except (AttributeError, IndexError, ValueError) as e:
            logging.warning(f"Error extracting source ID from {spectrum_file}: {e}")
            continue
        
        if not os.path.exists(json_file):
            logging.debug(f"JSON file not found: {json_file}")
            # Try alternative naming
            json_file_alt = os.path.join(json_files_dir, f"gaia_xp_spectrum_{source_id}-SPLUS21-magnitude.json")
            if os.path.exists(json_file_alt):
                json_file = json_file_alt
            else:
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
                logging.debug(f"Column {inst_filt} not found in CSV")
                continue
                
            inst_mag = row[inst_filt]
            
            # Skip invalid magnitudes (including the 99.0 placeholder for failed measurements)
            if pd.isna(inst_mag) or inst_mag >= 50.0 or inst_mag <= -50.0:
                continue
            
            if synth_filt in synth_data:
                synth_mag = synth_data[synth_filt]
                
                # Skip invalid synthetic magnitudes
                if pd.isna(synth_mag) or synth_mag >= 50.0 or synth_mag <= -50.0:
                    continue
                
                # CORRECCIÓN: Zero point = synthetic mag - instrumental mag
                # Esta es la definición estándar en fotometría
                zp = synth_mag - inst_mag
                
                # Filter reasonable zero points (typically between 15-25 for S-PLUS)
                if 10.0 < zp < 30.0:
                    zero_points[inst_filt].append(zp)
                    filters_processed += 1
        
        if filters_processed > 0:
            valid_stars += 1
    
    logging.info(f"Processed {valid_stars} valid stars with synthetic photometry")
    
    # Calculate statistics for each filter
    zp_results = {}
    for filt, zp_values in zero_points.items():
        if len(zp_values) > 5:  # Require minimum number of measurements
            # Convert to numpy array for easier handling
            zp_array = np.array(zp_values)
            
            # Use median as the primary statistic (robust against outliers)
            median_zp = np.median(zp_array)
            
            # Calculate robust scatter using MAD (Median Absolute Deviation)
            mad = np.median(np.abs(zp_array - median_zp))
            std_mad = 1.4826 * mad  # Convert MAD to equivalent standard deviation
            
            # Also calculate sigma-clipped statistics for comparison
            mean_clipped, median_clipped, std_clipped = sigma_clipped_stats(zp_array, sigma=3.0)
            
            zp_results[filt] = {
                'median': median_zp,           # Primary statistic
                'mad': mad,                    # Median Absolute Deviation
                'std_mad': std_mad,            # MAD converted to equivalent STD
                'mean': np.mean(zp_array),     # For comparison only
                'std': np.std(zp_array),       # For comparison only
                'mean_clipped': mean_clipped,  # Sigma-clipped mean
                'median_clipped': median_clipped,  # Sigma-clipped median
                'std_clipped': std_clipped,    # Sigma-clipped STD
                'n_stars': len(zp_values),
                'min': np.min(zp_array),
                'max': np.max(zp_array),
                'q25': np.percentile(zp_array, 25),
                'q75': np.percentile(zp_array, 75)
            }
            
            logging.info(f"{filt}: {len(zp_values)} measurements, Median ZP = {median_zp:.3f} ± {std_mad:.3f} (MAD)")
        else:
            logging.warning(f"{filt}: Insufficient data ({len(zp_values)} measurements, need >5)")
            zp_results[filt] = None
    
    return zp_results, zero_points

def plot_zero_points(zero_points_data, field_name, all_zp_values):
    """
    Plot zero point distributions with median values
    """
    if not zero_points_data or all(v is None for v in zero_points_data.values()):
        logging.warning("No data available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    filters = []
    zp_medians = []
    zp_errors_mad = []
    zp_means = []
    zp_errors_std = []
    n_measurements = []
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(zero_points_data)))
    
    # Plot 1: Individual measurements and median values
    for i, (filt, data) in enumerate(zero_points_data.items()):
        if data is not None and len(all_zp_values[filt]) > 0:
            filter_short = filt.replace('mag_inst_', '')
            filters.append(filter_short)
            zp_medians.append(data['median'])
            zp_errors_mad.append(data['std_mad'])
            zp_means.append(data['mean'])
            zp_errors_std.append(data['std'])
            n_measurements.append(data['n_stars'])
            
            # Plot individual measurements with jitter to avoid overplotting
            x_pos = i + np.random.normal(0, 0.05, len(all_zp_values[filt]))
            ax1.scatter(x_pos, all_zp_values[filt], 
                       alpha=0.4, color=colors[i], s=20, 
                       label=f'{filter_short} (n={data["n_stars"]})')
    
    if not filters:  # No valid data to plot
        plt.close(fig)
        return
    
    # Plot median values with MAD error bars
    ax1.errorbar(range(len(filters)), zp_medians, yerr=zp_errors_mad, 
                fmt='o', color='red', markersize=10, capsize=8, 
                label='Median ± MAD', linewidth=3, alpha=0.8)
    
    ax1.set_xticks(range(len(filters)))
    ax1.set_xticklabels(filters, rotation=45, ha='right')
    ax1.set_ylabel('Zero Point (synthetic - instrumental)')
    ax1.set_title(f'Zero Points for Field {field_name} - Corrected Photometry')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add text box with statistics
    textstr = '\n'.join([
        f'Total stars: {sum(n_measurements)}',
        f'Filters: {len(filters)}',
        f'Median ZP range: {min(zp_medians):.2f} - {max(zp_medians):.2f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Plot 2: Comparison between median and mean
    x_pos = np.arange(len(filters))
    width = 0.35
    
    ax2.bar(x_pos - width/2, zp_medians, width, label='Median', 
            yerr=zp_errors_mad, capsize=5, alpha=0.7, color='skyblue',
            error_kw=dict(elinewidth=2, capthick=2))
    ax2.bar(x_pos + width/2, zp_means, width, label='Mean', 
            yerr=zp_errors_std, capsize=5, alpha=0.7, color='lightcoral',
            error_kw=dict(elinewidth=2, capthick=2))
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(filters, rotation=45, ha='right')
    ax2.set_ylabel('Zero Point')
    ax2.set_title('Comparison: Median vs Mean Zero Points (Robust vs Standard)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (med, mean) in enumerate(zip(zp_medians, zp_means)):
        ax2.text(i - width/2, med + zp_errors_mad[i] + 0.1, f'{med:.2f}', 
                ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, mean + zp_errors_std[i] + 0.1, f'{mean:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{field_name}_zero_points_corrected.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Plot saved as {plot_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="""Calculate zero points for SPLUS filters using robust median statistics
                     for corrected instrumental photometry""")
    
    parser.add_argument("CSV", type=str,
                        help="CSV file with corrected instrumental magnitudes (_gaia_xp_matches_corrected.csv)")
    
    parser.add_argument("--json-dir", type=str, default=".",
                        help="Directory containing JSON files with synthetic magnitudes")
    
    parser.add_argument("--plot", action="store_true",
                        help="Create plot of zero point distributions")
    
    parser.add_argument("--min-stars", type=int, default=5,
                        help="Minimum number of stars required per filter (default: 5)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.CSV):
        logging.error(f"Input CSV file not found: {args.CSV}")
        return
    
    # Calculate zero points
    zp_results, all_zp_values = calculate_zero_points(args.CSV, args.json_dir)
    
    # Extract field name for output files
    if '_gaia_xp_matches_corrected.csv' in args.CSV:
        field_name = os.path.basename(args.CSV).split('_gaia_xp_matches_corrected.csv')[0]
    else:
        field_name = os.path.basename(args.CSV).split('.csv')[0]
    
    # Save results to file
    output_file = f'{field_name}_zero_points_corrected.csv'
    
    try:
        with open(output_file, 'w') as f:
            f.write("Filter,Median_ZP,MAD,STD_MAD,Mean_ZP,STD,Mean_Clipped,Median_Clipped,STD_Clipped,N_Stars,Min,Max,Q25,Q75\n")
            for filt, data in zp_results.items():
                if data is not None:
                    f.write(f"{filt},{data['median']:.6f},{data['mad']:.6f},{data['std_mad']:.6f},"
                           f"{data['mean']:.6f},{data['std']:.6f},"
                           f"{data['mean_clipped']:.6f},{data['median_clipped']:.6f},{data['std_clipped']:.6f},"
                           f"{data['n_stars']},{data['min']:.6f},{data['max']:.6f},"
                           f"{data['q25']:.6f},{data['q75']:.6f}\n")
                else:
                    f.write(f"{filt},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,NaN,NaN,NaN,NaN\n")
        
        logging.info(f"Results saved to {output_file}")
        
        # Create plot if requested
        if args.plot:
            plot_zero_points(zp_results, field_name, all_zp_values)
        
        # Print summary
        valid_filters = [f for f, d in zp_results.items() if d is not None]
        if valid_filters:
            logging.info(f"Successfully calculated zero points for {len(valid_filters)} filters")
            for filt in valid_filters:
                data = zp_results[filt]
                logging.info(f"  {filt}: ZP = {data['median']:.3f} ± {data['std_mad']:.3f} (n={data['n_stars']})")
        else:
            logging.warning("No valid zero points calculated for any filter")
            
    except Exception as e:
        logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
