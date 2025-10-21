#!/usr/bin/env python3
"""
Make photo-spectra from HOMOGENIZED SPLUS photometry for NGC 5128 globular clusters.
Uses the homogenized catalog: Results/all_fields_gc_photometry_properly_homogenized_v3.csv
NOW IN FLUX UNITS (erg/s/cm2/A) with proper error propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse
import os
from pathlib import Path
import pandas as pd

# Define filter wavelengths and properties (in Angstroms)
wl = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
filter_names = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']

# Colors and markers: MISMO COLOR Y SÍMBOLO PARA TAYLOR, MISMO COLOR Y SÍMBOLO PARA SPLUS
# Taylor filters (broad-band): U, G, R, I, Z
taylor_filters = ['U', 'G', 'R', 'I', 'Z']
taylor_color = "#1f77b4"  # AZUL para todos los filtros Taylor
taylor_marker = "s"       # CUADRADO para todos los filtros Taylor

# SPLUS filters (narrow-band): all others
splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
splus_color = "#ff7f0e"   # NARANJA para todos los filtros SPLUS
splus_marker = "o"        # CÍRCULO para todos los filtros SPLUS

# Create mapping dictionaries - MISMO COLOR Y SÍMBOLO PARA CADA TIPO
color_map = {}
marker_map = {}
for filter_name in filter_names:
    if filter_name in taylor_filters:
        color_map[filter_name] = taylor_color
        marker_map[filter_name] = taylor_marker
    else:
        color_map[filter_name] = splus_color
        marker_map[filter_name] = splus_marker

def safe_convert(value, default=np.nan):
    """Safely convert a value to float, handling masked and string values."""
    if isinstance(value, (str, np.ma.core.MaskedConstant)):
        if str(value).strip() in ['--', '', 'NaN', 'nan', 'NULL', 'None', '99.0', '99']:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    elif np.isnan(value) or value is None or value == 99.0:
        return default
    else:
        return float(value)

def magnitude_to_flux(mag, wl_angstrom):
    """
    Convert AB magnitude to flux in erg/s/cm2/A.
    
    Parameters:
    mag: AB magnitude
    wl_angstrom: Wavelength in Angstroms
    
    Returns:
    flux: Flux in erg/s/cm2/A
    """
    if np.isnan(mag) or mag >= 50.0 or mag <= -50.0:
        return np.nan
    
    # Convert magnitude to flux using AB system
    # F = 10^(-0.4 * (mag + 2.41)) / wl^2  [in erg/s/cm2/A]
    flux = (10**(-0.4 * (mag + 2.41))) / (wl_angstrom**2)
    
    # Convert to 1e-15 units for better readability
    flux /= 1e-15
    
    return flux

def flux_error_propagation(mag, mag_err, wl_angstrom):
    """
    Propagate magnitude error to flux error.
    
    Parameters:
    mag: AB magnitude
    mag_err: Magnitude error
    wl_angstrom: Wavelength in Angstroms
    
    Returns:
    flux_err: Flux error in same units as flux
    """
    if np.isnan(mag) or np.isnan(mag_err) or mag_err <= 0:
        return np.nan
    
    # Calculate the conversion factor
    c = (10**(-2.41/2.5)) / (wl_angstrom**2)
    c /= 1e-15  # Convert to 1e-15 units
    
    # Exponent for conversion
    b = -1.0 / 2.5
    
    # Error propagation formula
    flux_err = np.sqrt(((c * 10**(b * mag))**2) * (np.log(10) * b * mag_err)**2)
    
    return flux_err

def get_magnitude_columns_homogenized(catalog_path, aper=3):
    """
    Analyze the HOMOGENIZED catalog to determine available magnitude columns.
    Returns dictionary mapping filter names to column names.
    """
    # Read just the header to check columns
    df = pd.read_csv(catalog_path, nrows=1)
    available_columns = df.columns.tolist()
    
    mag_columns = {}
    
    print("🔍 Analyzing HOMOGENIZED catalog columns...")
    
    # For HOMOGENIZED catalog, we expect:
    # - Taylor filters: umag, gmag, rmag, imag, zmag (ORIGINAL Taylor values)
    # - SPLUS filters: MAG_FXXX_3 (HOMOGENIZED values)
    
    for filter_name in filter_names:
        if filter_name in taylor_filters:
            # Taylor filters - use original Taylor columns
            taylor_map = {'U': 'umag', 'G': 'gmag', 'R': 'rmag', 'I': 'imag', 'Z': 'zmag'}
            base_col = taylor_map[filter_name]
            
            if base_col in available_columns:
                mag_columns[filter_name] = base_col
                print(f"  ✅ {filter_name}: {base_col} (Taylor original)")
            else:
                print(f"  ❌ {filter_name}: {base_col} not found")
                
        else:
            # SPLUS filters - use HOMOGENIZED columns
            splus_col = f'MAG_{filter_name}_{aper}'
            if splus_col in available_columns:
                mag_columns[filter_name] = splus_col
                print(f"  ✅ {filter_name}: {splus_col} (SPLUS homogenized)")
            else:
                print(f"  ❌ {filter_name}: {splus_col} not found")
    
    return mag_columns, available_columns

def get_error_column_homogenized(mag_column, available_columns):
    """
    Get the corresponding error column for a magnitude column in HOMOGENIZED catalog.
    """
    # For SPLUS homogenized format: MAGERR_FXXX_3
    if mag_column.startswith('MAG_'):
        err_column = mag_column.replace('MAG_', 'MAGERR_')
        if err_column in available_columns:
            return err_column
    
    # For Taylor format: e_umag, e_gmag, etc.
    if mag_column in ['umag', 'gmag', 'rmag', 'imag', 'zmag']:
        err_column = f'e_{mag_column}'
        if err_column in available_columns:
            return err_column
        
        # Alternative naming
        err_column = f'{mag_column}_err'
        if err_column in available_columns:
            return err_column
    
    return None

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate photo-spectra for globular clusters in NGC 5128 using HOMOGENIZED photometry")
parser.add_argument("--catalog", type=str, default="Results/all_fields_gc_photometry_final_calibrated.csv",
                    help="Input catalog with HOMOGENIZED photometry data")
parser.add_argument("--aper", type=int, default=3, choices=[2, 3, 4, 5, 6],
                    help="Aperture size to use (3, 4, 5, or 6 arcsec)")
parser.add_argument("--id", type=str, help="Specific GC ID to plot (e.g., 'T17-2421')")
parser.add_argument("--min-snr", type=float, default=1.0,
                    help="Minimum SNR for plotting filters")
parser.add_argument("--output-dir", type=str, default="./photospectra_homogenized_v3",
                    help="Output directory for plots")
parser.add_argument("--ymin", type=float, help="Y-axis minimum value")
parser.add_argument("--ymax", type=float, help="Y-axis maximum value")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug output")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Read the HOMOGENIZED photometry catalog
try:
    print(f"📖 Loading HOMOGENIZED catalog: {args.catalog}")
    data = Table.read(args.catalog, format="ascii.csv")
    print(f"✅ Loaded catalog with {len(data)} sources")
except FileNotFoundError:
    print(f"❌ Error: Catalog file {args.catalog} not found.")
    exit(1)

# Analyze available magnitude columns in HOMOGENIZED catalog
mag_columns, available_columns = get_magnitude_columns_homogenized(args.catalog, args.aper)
print(f"\n📊 Using aperture: {args.aper}\"")

# Filter data if a specific ID is requested
if args.id:
    # Try different possible ID column names
    id_columns = ['T17ID', 'ID', 'id', 'recno']
    id_col = None
    for col in id_columns:
        if col in data.colnames:
            id_col = col
            break
    
    if id_col is None:
        print("❌ Error: No ID column found in catalog.")
        exit(1)
    
    mask = data[id_col] == args.id
    if not any(mask):
        print(f"❌ Error: Source {args.id} not found in catalog.")
        exit(1)
    data = data[mask]
    print(f"🔍 Filtered to source: {args.id}")

# Process each source in the catalog
for source in data:
    # Get source ID
    id_columns = ['T17ID', 'ID', 'id', 'recno']
    source_id = "Unknown"
    for col in id_columns:
        if col in source.colnames:
            source_id = source[col]
            break
    
    print(f"\n🎯 Processing source {source_id}")
    
    # Extract magnitudes and convert to fluxes
    fluxes = []
    flux_errs = []
    snrs = []
    filter_types = []  # Track whether filter is Taylor or SPLUS
    used_columns = []  # Track which columns were actually used
    
    for filter_name in filter_names:
        if filter_name not in mag_columns:
            if args.debug:
                print(f"  ⚠️ No column mapping for {filter_name}. Skipping.")
            fluxes.append(np.nan)
            flux_errs.append(np.nan)
            snrs.append(0)
            filter_types.append("Unknown")
            used_columns.append("None")
            continue
        
        mag_col = mag_columns[filter_name]
        
        # Check if column exists in the table
        if mag_col not in source.colnames:
            if args.debug:
                print(f"  ⚠️ Column {mag_col} not found. Skipping filter {filter_name}.")
            fluxes.append(np.nan)
            flux_errs.append(np.nan)
            snrs.append(0)
            filter_types.append("Unknown")
            used_columns.append("None")
            continue
            
        # Safely convert values to floats
        mag = safe_convert(source[mag_col])
        
        # Handle special values (99.0 indicates measurement issues)
        if mag == 99.0 or np.isnan(mag):
            if args.debug:
                print(f"  ⚠️ Filter {filter_name} has bad magnitude value. Skipping.")
            fluxes.append(np.nan)
            flux_errs.append(np.nan)
            snrs.append(0)
            filter_types.append("Taylor" if filter_name in taylor_filters else "SPLUS")
            used_columns.append(mag_col)
            continue
            
        # Get error column
        err_col = get_error_column_homogenized(mag_col, available_columns)
        if err_col and err_col in source.colnames:
            mag_err = safe_convert(source[err_col], 0.1)
            # Clean extreme errors (common in homogenized catalog)
            if mag_err > 10 or mag_err <= 0:
                mag_err = 0.1
        else:
            mag_err = 0.1  # Default error
        
        # Convert magnitude to flux
        wl_idx = filter_names.index(filter_name)
        wavelength = wl[wl_idx]
        
        flux = magnitude_to_flux(mag, wavelength)
        flux_err = flux_error_propagation(mag, mag_err, wavelength)
        
        # Calculate SNR from magnitude error
        snr = 1.0 / mag_err if mag_err > 0 else 100
        
        fluxes.append(flux)
        flux_errs.append(flux_err)
        snrs.append(snr)
        filter_types.append("Taylor" if filter_name in taylor_filters else "SPLUS")
        used_columns.append(mag_col)
        
        if args.debug:
            print(f"  ✅ Filter {filter_name} ({filter_types[-1]}):")
            print(f"     mag={mag:.3f}, flux={flux:.2e}, err={flux_err:.2e}, snr={snr:.1f}")

    # Check if we have any valid data to plot
    valid_data = sum(~np.isnan(fluxes) & (np.array(snrs) >= args.min_snr))
    if valid_data == 0:
        print(f"⚠️ Warning: No valid data found for source {source_id}. Skipping plot.")
        continue
    
    print(f"📊 Valid data points: {valid_data}/{len(filter_names)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    
    # Set tick parameters
    plt.tick_params(axis='x', labelsize=14, width=2, length=6) 
    plt.tick_params(axis='y', labelsize=14, width=2, length=6)
    
    # Set x-axis limits
    ax.set(xlim=[3000, 9300])
    
    # Set Y-axis range if provided
    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    elif args.ymin is not None:
        plt.ylim(ymin=args.ymin)
    elif args.ymax is not None:
        plt.ylim(ymax=args.ymax)
    
    # Plot connecting line for filters with good measurements
    valid_mask = ~np.isnan(fluxes) & (np.array(snrs) >= args.min_snr)
    valid_wl = np.array(wl)[valid_mask]
    valid_fluxes = np.array(fluxes)[valid_mask]
    
    if len(valid_wl) > 1:
        # Sort by wavelength for proper connecting line
        sort_idx = np.argsort(valid_wl)
        ax.plot(valid_wl[sort_idx], valid_fluxes[sort_idx], 
                '-', color='gray', alpha=0.7, linewidth=1, zorder=1)
    
    # Plot each filter point with different styles based on filter type and SNR
    taylor_handles = []  # For legend
    splus_handles = []   # For legend
    
    for w, f, fe, s, filter_name, filter_type in zip(wl, fluxes, flux_errs, snrs, filter_names, filter_types):
        if np.isnan(f) or s < args.min_snr:
            continue
        
        # Get color and marker based on filter type
        color = color_map[filter_name]
        marker = marker_map[filter_name]
        
        # Use different styles based on SNR quality
        if s < 1.0:
            # Low SNR: transparent and smaller
            alpha = 0.5
            markersize = 8
            edgecolor = 'gray'
            marker_label = ' (low SNR)'
        elif s < 3.0:
            # Medium SNR: partially transparent
            alpha = 0.8
            markersize = 10
            edgecolor = 'k'
            marker_label = ' (medium SNR)'
        else:
            # High SNR: fully opaque
            alpha = 1.0
            markersize = 10
            edgecolor = 'k'
            marker_label = ' (high SNR)'
        
        # Plot the point
        scatter = ax.scatter(w, f, color=color, marker=marker, 
                           edgecolors=edgecolor, s=markersize*20, 
                           alpha=alpha, zorder=3)
        
        # Plot error bars
        ax.errorbar(w, f, yerr=fe, fmt='none', 
                   color=color, alpha=alpha, elinewidth=2, 
                   capsize=4, capthick=2, zorder=2)
        
        # Store handles for legend (only once per filter type)
        if filter_type == 'Taylor' and not taylor_handles:
            taylor_handles.append(scatter)
        elif filter_type == 'SPLUS' and not splus_handles:
            splus_handles.append(scatter)
    
    # Customize the plot
    ax.set_xlabel('Wavelength (Å)', fontsize=16)
    ax.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=16)
    ax.set_title(f'GC {source_id} - HOMOGENIZED Photometry (Aperture: {args.aper}")', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Add filter labels
    for i, (w, fn) in enumerate(zip(wl, filter_names)):
        if not np.isnan(fluxes[i]) and snrs[i] >= args.min_snr:
            ax.annotate(fn, (w, fluxes[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, alpha=0.7)
    
    # Create legend for filter types
    legend_handles = []
    legend_labels = []
    
    if taylor_handles:
        legend_handles.append(taylor_handles[0])
        legend_labels.append('Taylor et al. (broad-band)')
    if splus_handles:
        legend_handles.append(splus_handles[0])
        legend_labels.append('SPLUS (narrow-band, homogenized)')
    
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)
    
    # Add text box with source information
    textstr = f"Source: {source_id}"
    
    # Try to get coordinates
    ra_cols = ['RAJ2000', 'RA', 'ra']
    dec_cols = ['DEJ2000', 'DEC', 'dec']
    ra_val = "N/A"
    dec_val = "N/A"
    
    for ra_col in ra_cols:
        if ra_col in source.colnames:
            ra_val = f"{source[ra_col]:.6f}"
            break
    
    for dec_col in dec_cols:
        if dec_col in source.colnames:
            dec_val = f"{source[dec_col]:.6f}"
            break
    
    textstr += f"\nRA: {ra_val}\nDEC: {dec_val}"
    
    if 'FIELD' in source.colnames:
        textstr += f"\nField: {source['FIELD']}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add SNR information to the plot
    snr_info = f"Min SNR: {args.min_snr}\nValid points: {valid_data}/{len(filter_names)}"
    ax.text(0.02, 0.02, snr_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Add catalog info
    catalog_info = f"HOMOGENIZED Catalog\nAperture: {args.aper}\""
    ax.text(0.98, 0.02, catalog_info, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, f"photospectrum_{source_id}_homogenized_aper{args.aper}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved HOMOGENIZED flux plot for {source_id} to {output_file}")

print(f"\n🎯 All HOMOGENIZED flux photo-spectra generated successfully!")
print(f"📁 Output directory: {args.output_dir}")
print(f"📊 Total sources processed: {len(data)}")
