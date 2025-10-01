#!/usr/bin/env python3
"""
Make photo-spectra from observed SPLUS objects for NGC 5128 globular clusters.
Adapted to handle both standard SPLUS and Taylor catalog formats.
NOW IN FLUX UNITS (erg/s/cm2/A) with proper error propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse
import os
from pathlib import Path

# Define filter wavelengths and properties (in Angstroms)
wl = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
filter_names = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']

# Colors and markers: MISMO COLOR Y SÍMBOLO PARA TAYLOR, MISMO COLOR Y SÍMBOLO PARA T80S
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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate photo-spectra for globular clusters in NGC 5128")
parser.add_argument("--catalog", type=str, default="Results/all_fields_gc_photometry_identical_gc.csv",
                    help="Input catalog with photometry data")
parser.add_argument("--aper", type=int, default=5, choices=[3, 4, 5, 6],
                    help="Aperture size to use (3, 4, 5, or 6 arcsec)")
parser.add_argument("--id", type=str, help="Specific GC ID to plot (e.g., 'T17-2421')")
parser.add_argument("--min-snr", type=float, default=0.1,
                    help="Minimum SNR for plotting filters")
parser.add_argument("--output-dir", type=str, default="./photospectra_flux_corrected",
                    help="Output directory for plots")
parser.add_argument("--ymin", type=float, help="Y-axis minimum value")
parser.add_argument("--ymax", type=float, help="Y-axis maximum value")
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
    
    # Extract magnitudes and convert to fluxes
    fluxes = []
    flux_errs = []
    snrs = []
    filter_types = []  # Track whether filter is Taylor or SPLUS
    
    # First, check what format we have by looking for Taylor format columns
    has_taylor_format = 'umag' in source.colnames and 'gmag' in source.colnames
    
    if args.debug:
        print(f"Has Taylor format: {has_taylor_format}")
    
    for filter_name in filter_names:
        # Determine which column names to use based on available data
        if has_taylor_format and filter_name in taylor_filters:
            # Use Taylor format for broad-band filters
            taylor_map = {'U': 'umag', 'G': 'gmag', 'R': 'rmag', 'I': 'imag', 'Z': 'zmag'}
            mag_col = taylor_map[filter_name]
            err_col = 'e_' + mag_col
            snr_col = 's_' + mag_col
            filter_type = 'Taylor'
            
            if args.debug:
                print(f"Using Taylor format for {filter_name}: {mag_col}")
        else:
            # Use SPLUS format for all filters
            mag_col = f'MAG_{filter_name}_{args.aper}'
            err_col = f'MAGERR_{filter_name}_{args.aper}'
            snr_col = f'SNR_{filter_name}_{args.aper}'
            filter_type = 'SPLUS'
            
            if args.debug:
                print(f"Using SPLUS format for {filter_name}: {mag_col}")
        
        # Check if columns exist in the table
        if mag_col not in source.colnames:
            if args.debug:
                print(f"Column {mag_col} not found. Skipping filter {filter_name}.")
            fluxes.append(np.nan)
            flux_errs.append(np.nan)
            snrs.append(0)
            filter_types.append(filter_type)
            continue
            
        # Safely convert values to floats, handling masked values and strings
        mag = safe_convert(source[mag_col])
        
        # Handle special values (99.0 indicates measurement issues)
        if mag == 99.0 or np.isnan(mag):
            if args.debug:
                print(f"Filter {filter_name} has bad value. Skipping.")
            fluxes.append(np.nan)
            flux_errs.append(np.nan)
            snrs.append(0)
            filter_types.append(filter_type)
            continue
            
        # Get error and calculate flux
        if err_col in source.colnames:
            mag_err = safe_convert(source[err_col], 0.1)
        else:
            mag_err = 0.1
        
        # Convert magnitude to flux
        wl_idx = filter_names.index(filter_name)
        wavelength = wl[wl_idx]
        
        flux = magnitude_to_flux(mag, wavelength)
        flux_err = flux_error_propagation(mag, mag_err, wavelength)
        
        # For Taylor format, use error-based SNR calculation
        if has_taylor_format and filter_name in taylor_filters:
            snr = 1.0 / mag_err if mag_err > 0 else 100
        else:
            # For SPLUS format, use provided SNR or calculate from error
            if snr_col in source.colnames:
                snr = safe_convert(source[snr_col], 10)
            else:
                snr = 1.0 / mag_err if mag_err > 0 else 100
        
        fluxes.append(flux)
        flux_errs.append(flux_err)
        snrs.append(snr)
        filter_types.append(filter_type)
        
        if args.debug:
            print(f"Filter {filter_name} ({filter_type}): mag={mag}, flux={flux:.2e}, err={flux_err:.2e}, snr={snr}")
    
    # Check if we have any valid data to plot
    valid_data = sum(~np.isnan(fluxes) & (np.array(snrs) >= args.min_snr))
    if valid_data == 0:
        print(f"Warning: No valid data found for source {source_id}. Skipping plot.")
        continue
    
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
        
        # Get color and marker based on filter type - AHORA MISMO COLOR POR TIPO
        color = color_map[filter_name]
        marker = marker_map[filter_name]
        
        # Use different styles based on SNR quality
        if s < 1.0:
            # Low SNR: transparent and smaller
            alpha = 0.5
            markersize = 8
            edgecolor = 'gray'
        elif s < 3.0:
            # Medium SNR: partially transparent
            alpha = 0.8
            markersize = 10
            edgecolor = 'k'
        else:
            # High SNR: fully opaque
            alpha = 1.0
            markersize = 10
            edgecolor = 'k'
        
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
    ax.set_title(f'GC {source_id} - Aperture: {args.aper}"', fontsize=18)
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
        legend_labels.append('T80S (narrow-band)')
    
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)
    
    # Add text box with source information
    ra_col = 'RAJ2000' if 'RAJ2000' in source.colnames else 'RA'
    dec_col = 'DEJ2000' if 'DEJ2000' in source.colnames else 'DEC'
    textstr = f"RA: {source[ra_col]:.6f}\nDEC: {source[dec_col]:.6f}"
    if 'FIELD' in source.colnames:
        textstr += f"\nField: {source['FIELD']}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add SNR information to the plot
    snr_info = f"Min SNR: {args.min_snr}\nValid points: {valid_data}/{len(filter_names)}"
    ax.text(0.02, 0.02, snr_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(args.output_dir, f"photospectrum_{source_id}_aper{args.aper}_flux.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved flux plot for {source_id} to {output_file}")

print("All flux photo-spectra generated successfully!")
