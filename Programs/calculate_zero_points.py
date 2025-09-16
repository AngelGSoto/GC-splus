'''
Calculate zero points for SPLUS filters using instrumental and synthetic magnitudes
'''
from __future__ import print_function
import numpy as np
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from astropy.stats import sigma_clipped_stats

# Filter mapping between CSV columns and JSON keys
filter_mapping = {
    'mag_F378': 'F0378',
    'mag_F395': 'F0395', 
    'mag_F410': 'F0410',
    'mag_F430': 'F0430',
    'mag_F515': 'F0515',
    'mag_F660': 'F0660',
    'mag_F861': 'F0861'
}

def calculate_zero_points(csv_file, json_dir):
    """
    Calculate zero points for SPLUS filters
    
    Parameters:
    csv_file: CSV file with instrumental magnitudes
    json_dir: Directory containing JSON files with synthetic magnitudes
    """
    
    # Read instrumental photometry
    pho_inst = pd.read_csv(csv_file)
    
    # Extract field name from CSV filename
    field_name = os.path.basename(csv_file).split('_gaia_xp_matches.csv')[0]
    
    # Get the directory where JSON files are stored
    json_files_dir = os.path.join(json_dir, f"gaia_spectra_{field_name}")
    
    # Initialize arrays for zero point calculation
    zero_points = {filt: [] for filt in filter_mapping.keys()}
    n_stars = len(pho_inst)
    
    print(f"Processing field {field_name} with {n_stars} stars")
    
    # Process each star
    for idx, row in pho_inst.iterrows():
        spectrum_file = row['spectrum_file']
        
        # Skip NaN values or invalid entries
        if pd.isna(spectrum_file) or not isinstance(spectrum_file, str):
            continue
        
        # Extract source ID from spectrum file name
        try:
            source_id = spectrum_file.split('_')[-1].split('.')[0]
            json_file = os.path.join(json_files_dir, f"gaia_xp_spectrum_{source_id}-Ref-SPLUS21-magnitude.json")
        except (AttributeError, IndexError):
            print(f"Warning: Invalid spectrum_file format: {spectrum_file}")
            continue
        
        if not os.path.exists(json_file):
            print(f"Warning: JSON file not found for {source_id}")
            continue
        
        # Load synthetic magnitudes
        try:
            with open(json_file) as f:
                synth_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load JSON file {json_file}")
            continue
        
        # Calculate zero points for each filter
        for inst_filt, synth_filt in filter_mapping.items():
            if inst_filt in row and synth_filt in synth_data:
                inst_mag = row[inst_filt]
                synth_mag = synth_data[synth_filt]
                
                # Skip invalid magnitudes
                if pd.isna(inst_mag) or pd.isna(synth_mag):
                    continue
                
                # CORRECCIÓN: Zero point = synthetic mag - instrumental mag
                # Esta es la definición estándar en fotometría
                zp = synth_mag - inst_mag
                zero_points[inst_filt].append(zp)
    
    # Calculate statistics for each filter
    zp_results = {}
    for filt, zp_values in zero_points.items():
        if len(zp_values) > 0:
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
            
            print(f"{filt}: {len(zp_values)} stars, Median ZP = {median_zp:.3f} ± {std_mad:.3f} (MAD)")
        else:
            print(f"{filt}: No data available")
            zp_results[filt] = None
    
    return zp_results, zero_points

def plot_zero_points(zero_points_data, field_name, all_zp_values):
    """
    Plot zero point distributions with median values
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    filters = []
    zp_medians = []
    zp_errors_mad = []
    zp_means = []
    zp_errors_std = []
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(zero_points_data)))
    
    # Plot 1: Individual measurements and median values
    for i, (filt, data) in enumerate(zero_points_data.items()):
        if data is not None:
            filter_short = filt.replace('mag_', '')
            filters.append(filter_short)
            zp_medians.append(data['median'])
            zp_errors_mad.append(data['std_mad'])
            zp_means.append(data['mean'])
            zp_errors_std.append(data['std'])
            
            # Plot individual measurements
            ax1.scatter([i] * len(all_zp_values[filt]), all_zp_values[filt], 
                       alpha=0.3, color=colors[i], s=15, label=filter_short if i < 10 else "")
    
    # Plot median values with MAD error bars
    ax1.errorbar(range(len(filters)), zp_medians, yerr=zp_errors_mad, 
                fmt='o', color='red', markersize=8, capsize=5, 
                label='Median ± MAD', linewidth=2)
    
    ax1.set_xticks(range(len(filters)))
    ax1.set_xticklabels(filters, rotation=45)
    ax1.set_ylabel('Zero Point (synthetic - instrumental)')
    ax1.set_title(f'Zero Points for Field {field_name} - Correct Definition')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Comparison between median and mean
    x_pos = np.arange(len(filters))
    width = 0.35
    
    ax2.bar(x_pos - width/2, zp_medians, width, label='Median', 
            yerr=zp_errors_mad, capsize=4, alpha=0.7)
    ax2.bar(x_pos + width/2, zp_means, width, label='Mean', 
            yerr=zp_errors_std, capsize=4, alpha=0.7)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(filters, rotation=45)
    ax2.set_ylabel('Zero Point')
    ax2.set_title('Comparison: Median vs Mean Zero Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{field_name}_zero_points.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="""Calculate zero points for SPLUS filters using robust median statistics""")
    
    parser.add_argument("CSV", type=str,
                        help="CSV file with instrumental magnitudes")
    
    parser.add_argument("--json-dir", type=str, default=".",
                        help="Directory containing JSON files with synthetic magnitudes")
    
    parser.add_argument("--plot", action="store_true",
                        help="Create plot of zero point distributions")
    
    args = parser.parse_args()
    
    # Calculate zero points
    zp_results, all_zp_values = calculate_zero_points(args.CSV, args.json_dir)
    
    # Save results to file
    field_name = os.path.basename(args.CSV).split('_gaia_xp_matches.csv')[0]
    output_file = f'{field_name}_zero_points.csv'
    
    with open(output_file, 'w') as f:
        f.write("Filter,Median_ZP,MAD,STD_MAD,Mean_ZP,STD,Mean_Clipped,Median_Clipped,STD_Clipped,N_Stars,Min,Max,Q25,Q75\n")
        for filt, data in zp_results.items():
            if data is not None:
                f.write(f"{filt},{data['median']:.4f},{data['mad']:.4f},{data['std_mad']:.4f},"
                       f"{data['mean']:.4f},{data['std']:.4f},"
                       f"{data['mean_clipped']:.4f},{data['median_clipped']:.4f},{data['std_clipped']:.4f},"
                       f"{data['n_stars']},{data['min']:.4f},{data['max']:.4f},"
                       f"{data['q25']:.4f},{data['q75']:.4f}\n")
            else:
                f.write(f"{filt},NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,NaN,NaN,NaN,NaN\n")
    
    print(f"Results saved to {output_file}")
    
    # Create plot if requested
    if args.plot and any(data is not None for data in zp_results.values()):
        plot_zero_points(zp_results, field_name, all_zp_values)
        print(f"Plot saved as {field_name}_zero_points.png")
    elif args.plot:
        print("No data available for plotting")

if __name__ == "__main__":
    main()
