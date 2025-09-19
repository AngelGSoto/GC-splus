#!/usr/bin/env python3
"""
Make photo-spectra from observed SPLUS objects for NGC 5128 globular clusters.
Adapted to handle both standard SPLUS and Taylor catalog formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse
import os
from pathlib import Path

# Define filter wavelengths and properties
wl = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
filter_names = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
color = ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", 
         "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"]

def safe_convert(value, default=np.nan):
    """Safely convert a value to float, handling masked and string values."""
    if isinstance(value, (str, np.ma.core.MaskedConstant)):
        if str(value).strip() in ['--', '', 'NaN', 'nan', 'NULL', 'None']:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    elif np.isnan(value) or value is None:
        return default
    else:
        return float(value)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate photo-spectra for globular clusters in NGC 5128")
parser.add_argument("--catalog", type=str, default="all_fields_gc_photometry_merged.csv",
                    help="Input catalog with photometry data")
parser.add_argument("--aper", type=int, default=5, choices=[3, 4, 5, 6],
                    help="Aperture size to use (3, 4, 5, or 6 arcsec)")
parser.add_argument("--id", type=str, help="Specific GC ID to plot (e.g., 'T17-2421')")
parser.add_argument("--min-snr", type=float, default=0.1,  # Reduced from 3.0 to 0.1
                    help="Minimum SNR for plotting filters")
parser.add_argument("--output-dir", type=str, default="./photospectra",
                    help="Output directory for plots")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug output")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Read the photometry catalog
try:
    data = Table.read(args.catalog, format="ascii.csv")
    print(f"Loaded catalog with {len(data)} sources")
    print(f"Available columns: {data.colnames}")
except FileNotFoundError:
    print(f"Error: Catalog file {args.catalog} not found.")
    exit(1)

# Filter data if a specific ID is requested
if args.id:
    id_col = 'T17ID' if 'T17ID' in data.colnames else 'ID'
    mask = data[id_col] == args.id
    if not any(mask):
        print(f"Error: Source {args.id} not found in catalog.")
        exit(1)
    data = data[mask]

# Process each source in the catalog
for source in data:
    source_id = source['T17ID'] if 'T17ID' in source.colnames else source['ID']
    print(f"Processing source {source_id}")
    
    # Extract magnitudes and errors
    mags = []
    mag_errs = []
    snrs = []
    
    # First, check what format we have by looking for Taylor format columns
    has_taylor_format = 'umag' in source.colnames and 'gmag' in source.colnames
    
    if args.debug:
        print(f"Has Taylor format: {has_taylor_format}")
    
    for filter_name in filter_names:
        # Determine which column names to use based on available data
        if has_taylor_format and filter_name in ['U', 'G', 'R', 'I', 'Z']:
            # Use Taylor format for broad-band filters
            taylor_map = {'U': 'umag', 'G': 'gmag', 'R': 'rmag', 'I': 'imag', 'Z': 'zmag'}
            mag_col = taylor_map[filter_name]
            err_col = 'e_' + mag_col
            snr_col = 's_' + mag_col
            
            if args.debug:
                print(f"Using Taylor format for {filter_name}: {mag_col}")
        else:
            # Use SPLUS format for all filters
            mag_col = f'MAG_{filter_name}_{args.aper}'
            err_col = f'MAGERR_{filter_name}_{args.aper}'
            snr_col = f'SNR_{filter_name}_{args.aper}'
            
            if args.debug:
                print(f"Using SPLUS format for {filter_name}: {mag_col}")
        
        # Check if columns exist in the table
        if mag_col not in source.colnames:
            if args.debug:
                print(f"Column {mag_col} not found. Skipping filter {filter_name}.")
            mags.append(np.nan)
            mag_errs.append(np.nan)
            snrs.append(0)
            continue
            
        # Safely convert values to floats, handling masked values and strings
        mag = safe_convert(source[mag_col])
        
        # Handle special values (99.0 indicates measurement issues)
        if mag == 99.0:
            if args.debug:
                print(f"Filter {filter_name} has bad value (99.0). Skipping.")
            mags.append(np.nan)
            mag_errs.append(np.nan)
            snrs.append(0)
            continue
            
        # Get error
        if err_col in source.colnames:
            mag_err = safe_convert(source[err_col], 0.1)
            # For Taylor format, use error-based SNR calculation
            if has_taylor_format and filter_name in ['U', 'G', 'R', 'I', 'Z']:
                snr = 1.0 / mag_err if mag_err > 0 else 100
            else:
                # For SPLUS format, use provided SNR or calculate from error
                if snr_col in source.colnames:
                    snr = safe_convert(source[snr_col], 10)
                else:
                    snr = 1.0 / mag_err if mag_err > 0 else 100
        else:
            mag_err = 0.1
            snr = 10  # Default SNR value
        
        mags.append(mag)
        mag_errs.append(mag_err)
        snrs.append(snr)
        
        print(f"Filter {filter_name}: mag={mag}, err={mag_err}, snr={snr}")
    
    # Check if we have any valid data to plot
    valid_data = sum(~np.isnan(mags) & (np.array(snrs) >= args.min_snr))
    if valid_data == 0:
        print(f"Warning: No valid data found for source {source_id}. Skipping plot.")
        continue
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot connecting line for filters with good measurements
    valid_mask = ~np.isnan(mags) & (np.array(snrs) >= args.min_snr)
    valid_wl = np.array(wl)[valid_mask]
    valid_mags = np.array(mags)[valid_mask]
    
    if len(valid_wl) > 1:
        # Sort by wavelength for proper connecting line
        sort_idx = np.argsort(valid_wl)
        ax.plot(valid_wl[sort_idx], valid_mags[sort_idx], 
                '-', color='gray', alpha=0.7, linewidth=1)
    
    # Plot each filter point with different styles based on SNR
    for w, m, e, s, c, mk in zip(wl, mags, mag_errs, snrs, color, marker):
        if np.isnan(m) or s < args.min_snr:
            continue
        
        # Use different styles based on SNR quality
        if s < 1.0:
            # Low SNR: transparent and smaller
            alpha = 0.5
            markersize = 6
        elif s < 3.0:
            # Medium SNR: partially transparent
            alpha = 0.8
            markersize = 8
        else:
            # High SNR: fully opaque
            alpha = 1.0
            markersize = 8
            
        ax.errorbar(w, m, yerr=e, fmt=mk, color=c, ecolor=c, alpha=alpha,
                   elinewidth=2, capsize=4, capthick=2, markersize=markersize)
    
    # Customize the plot
    ax.set_xlabel('Wavelength (Å)', fontsize=14)
    ax.set_ylabel('Magnitude (AB)', fontsize=14)
    ax.set_title(f'GC {source_id} - Aperture: {args.aper}"', fontsize=16)
    ax.invert_yaxis()  # Magnitude scale inverted
    ax.grid(True, alpha=0.3)
    
    # Add filter labels
    for i, (w, fn) in enumerate(zip(wl, filter_names)):
        if not np.isnan(mags[i]) and snrs[i] >= args.min_snr:
            ax.annotate(fn, (w, mags[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9, alpha=0.7)
    
    # Add text box with source information
    ra_col = 'RAJ2000' if 'RAJ2000' in source.colnames else 'RA'
    dec_col = 'DEJ2000' if 'DEJ2000' in source.colnames else 'DEC'
    textstr = f"RA: {source[ra_col]:.6f}\nDEC: {source[dec_col]:.6f}"
    if 'FIELD' in source.colnames:
        textstr += f"\nField: {source['FIELD']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add SNR information to the plot
    snr_info = f"Min SNR: {args.min_snr}"
    ax.text(0.05, 0.05, snr_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, f"photospectrum_{source_id}_aper{args.aper}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {source_id} to {output_file}")

print("All photo-spectra generated successfully!")
