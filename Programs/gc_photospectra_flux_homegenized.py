#!/usr/bin/env python3
"""
Make photo-spectra from observed SPLUS objects for NGC 5128 globular clusters.
Adapted to handle both homogenized SPLUS photometry and Taylor catalog formats.
NOW IN FLUX UNITS (erg/s/cm2/A) with proper error propagation.
Uses homogenized SPLUS photometry and includes both aperture sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def extract_photometry_data(source, apertures=[2, 3], debug=False):
    """
    Extract photometry data for a source for multiple apertures.
    
    Returns:
    dict: Dictionary with photometry data organized by filter and aperture
    """
    photometry_data = {}
    
    # Check available columns
    available_columns = source.colnames
    
    # Determine if we have Taylor format
    has_taylor_format = 'umag' in available_columns and 'gmag' in available_columns
    
    if debug:
        logger.info(f"Source has Taylor format: {has_taylor_format}")
        logger.info(f"Available SPLUS columns: {[col for col in available_columns if 'MAG_' in col]}")
    
    # Process each filter
    for filter_name in filter_names:
        photometry_data[filter_name] = {}
        
        # Determine which column names to use based on available data
        if has_taylor_format and filter_name in taylor_filters:
            # Use Taylor format for broad-band filters
            IDmag_col_base = taylor_map[filter_name]
            err_col_base = 'e_' + mag_col_base
            filter_type = 'Taylor'
            
            # Taylor data doesn't have apertures, so we use the same value for all apertures
            mag = safe_convert(source[mag_col_base])
            mag_err = safe_convert(source[err_col_base], 0.1) if err_col_base in available_columns else 0.1
            
            # Calculate SNR for Taylor data
            snr = 1.0 / mag_err if mag_err > 0 else 100
            
            for aperture in apertures:
                photometry_data[filter_name][aperture] = {
                    'mag': mag,
                    'mag_err': mag_err,
                    'snr': snr,
                    'filter_type': filter_type,
                    'wavelength': wl[filter_names.index(filter_name)]
                }
                
        else:
            # Use SPLUS format for narrow-band filters
            filter_type = 'SPLUS'
            
            for aperture in apertures:
                mag_col = f'MAG_{filter_name}_{aperture}'
                err_col = f'MAGERR_{filter_name}_{aperture}'
                snr_col = f'SNR_{filter_name}_{aperture}'
                
                if mag_col in available_columns:
                    mag = safe_convert(source[mag_col])
                    mag_err = safe_convert(source[err_col], 0.1) if err_col in available_columns else 0.1
                    
                    # Get SNR
                    if snr_col in available_columns:
                        snr = safe_convert(source[snr_col], 10)
                    else:
                        snr = 1.0 / mag_err if mag_err > 0 else 100
                else:
                    mag = np.nan
                    mag_err = np.nan
                    snr = 0
                
                photometry_data[filter_name][aperture] = {
                    'mag': mag,
                    'mag_err': mag_err,
                    'snr': snr,
                    'filter_type': filter_type,
                    'wavelength': wl[filter_names.index(filter_name)]
                }
    
    return photometry_data

def plot_photospectrum(source_id, photometry_data, apertures=[2, 3], output_dir="./photospectra_flux_corrected", 
                      min_snr=0.1, ymin=None, ymax=None, debug=False):
    """
    Plot photospectrum for a single source.
    """
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
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    elif ymin is not None:
        plt.ylim(ymin=ymin)
    elif ymax is not None:
        plt.ylim(ymax=ymax)
    
    # Plot for each aperture
    aperture_colors = {2: 'red', 3: 'blue', 4: 'green', 5: 'orange', 6: 'purple'}
    aperture_labels = {2: '2"', 3: '3"', 4: '4"', 5: '5"', 6: '6"'}
    
    taylor_handles = []  # For legend
    splus_handles = []   # For legend
    aperture_handles = [] # For aperture legend
    
    valid_apertures = []
    
    for aperture in apertures:
        # Collect data for this aperture
        fluxes = []
        flux_errs = []
        snrs = []
        filter_types = []
        wavelengths = []
        valid_filters = []
        
        for filter_name in filter_names:
            data = photometry_data[filter_name].get(aperture, {})
            if not data or np.isnan(data.get('mag', np.nan)) or data.get('snr', 0) < min_snr:
                continue
            
            mag = data['mag']
            mag_err = data['mag_err']
            snr = data['snr']
            filter_type = data['filter_type']
            wavelength = data['wavelength']
            
            # Convert to flux
            flux = magnitude_to_flux(mag, wavelength)
            flux_err = flux_error_propagation(mag, mag_err, wavelength)
            
            if not np.isnan(flux):
                fluxes.append(flux)
                flux_errs.append(flux_err)
                snrs.append(snr)
                filter_types.append(filter_type)
                wavelengths.append(wavelength)
                valid_filters.append(filter_name)
        
        # Skip if no valid data for this aperture
        if not fluxes:
            continue
            
        valid_apertures.append(aperture)
        
        # Sort by wavelength for proper connecting line
        sort_idx = np.argsort(wavelengths)
        sorted_wl = np.array(wavelengths)[sort_idx]
        sorted_fluxes = np.array(fluxes)[sort_idx]
        
        # Plot connecting line for this aperture
        if len(sorted_wl) > 1:
            ax.plot(sorted_wl, sorted_fluxes, '-', 
                   color=aperture_colors.get(aperture, 'gray'), 
                   alpha=0.7, linewidth=1, zorder=1,
                   label=f'Aperture {aperture_labels[aperture]}')
        
        # Plot each filter point
        for w, f, fe, s, filter_name, filter_type in zip(wavelengths, fluxes, flux_errs, snrs, valid_filters, filter_types):
            # Get color and marker based on filter type
            color = color_map[filter_name]
            marker = marker_map[filter_name]
            
            # Use different styles based on SNR quality
            if s < 1.0:
                # Low SNR: transparent and smaller
                alpha = 0.5
                markersize = 6
                edgecolor = 'gray'
            elif s < 3.0:
                # Medium SNR: partially transparent
                alpha = 0.8
                markersize = 8
                edgecolor = 'k'
            else:
                # High SNR: fully opaque
                alpha = 1.0
                markersize = 10
                edgecolor = 'k'
            
            # Plot the point with aperture-specific face color but filter-specific edge color
            scatter = ax.scatter(w, f, color=aperture_colors.get(aperture, 'gray'), 
                               marker=marker, edgecolors=color, 
                               s=markersize*20, alpha=alpha, zorder=3,
                               linewidths=2)
            
            # Plot error bars
            ax.errorbar(w, f, yerr=fe, fmt='none', 
                       color=aperture_colors.get(aperture, 'gray'), alpha=alpha, 
                       elinewidth=1, capsize=3, capthick=1, zorder=2)
            
            # Store handles for legend (only once per filter type and aperture)
            if filter_type == 'Taylor' and not taylor_handles:
                taylor_handles.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                               markersize=10, linestyle='None', label='Taylor et al. (broad-band)'))
            elif filter_type == 'SPLUS' and not splus_handles:
                splus_handles.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                              markersize=10, linestyle='None', label='T80S (narrow-band)'))
        
        # Store aperture handle
        if aperture not in [h.get_label() for h in aperture_handles]:
            aperture_handles.append(plt.Line2D([0], [0], color=aperture_colors.get(aperture, 'gray'), 
                                            linewidth=3, label=f'Aperture {aperture_labels[aperture]}'))
    
    # Skip if no valid data at all
    if not valid_apertures:
        logger.warning(f"No valid data found for source {source_id} with min SNR {min_snr}")
        plt.close()
        return
    
    # Customize the plot
    ax.set_xlabel('Wavelength (Å)', fontsize=16)
    ax.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=16)
    ax.set_title(f'GC {source_id} - Homogenized Photometry', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Add filter labels for high SNR points
    for filter_name in filter_names:
        # Use data from the first valid aperture
        for aperture in apertures:
            data = photometry_data[filter_name].get(aperture, {})
            if data and data.get('snr', 0) >= min_snr and not np.isnan(data.get('mag', np.nan)):
                wavelength = data['wavelength']
                flux = magnitude_to_flux(data['mag'], wavelength)
                if not np.isnan(flux):
                    ax.annotate(filter_name, (wavelength, flux), xytext=(5, 5), 
                               textcoords='offset points', fontsize=9, alpha=0.7)
                break
    
    # Create combined legend
    legend_handles = aperture_handles + taylor_handles + splus_handles
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    
    # Add text box with source information
    textstr = f"Min SNR: {min_snr}\nValid apertures: {', '.join([aperture_labels[a] for a in valid_apertures])}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"photospectrum_{source_id}_flux_homogenized.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved flux plot for {source_id} to {output_file}")
    return True

def main():
    """
    Main function to generate photospectra for globular clusters.
    """
    parser = argparse.ArgumentParser(description="Generate photo-spectra for globular clusters in NGC 5128")
    parser.add_argument("--catalog", type=str, 
                        default="CenA01_reference_stars_photometry_v17.csv",
                       #default="Results/all_fields_gc_photometry_corrected_errors_v17.csv",
                       help="Input catalog with homogenized photometry data")
    parser.add_argument("--apertures", type=int, nargs='+', default=[2, 3],
                       help="Aperture sizes to use (e.g., 2 3)")
    parser.add_argument("--id", type=str, help="Specific GC ID to plot (e.g., 'T17-2421')")
    parser.add_argument("--min-snr", type=float, default=0.1,
                       help="Minimum SNR for plotting filters")
    parser.add_argument("--output-dir", type=str, default="./photospectra_flux_homogenized",
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
        logger.info(f"Loading catalog: {args.catalog}")
        data = Table.read(args.catalog, format="ascii.csv")
        logger.info(f"Loaded catalog with {len(data)} sources")
        if args.debug:
            logger.info(f"Available columns: {data.colnames}")
    except FileNotFoundError:
        logger.error(f"Catalog file {args.catalog} not found.")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading catalog: {e}")
        exit(1)
    
    # Filter data if a specific ID is requested
    if args.id:
        id_col = 'T17ID' if 'T17ID' in data.colnames else 'ref_source_id'
        mask = data[id_col] == args.id
        if not any(mask):
            logger.error(f"Source {args.id} not found in catalog.")
            exit(1)
        data = data[mask]
        logger.info(f"Processing specific source: {args.id}")
    
    # Process each source in the catalog
    successful_plots = 0
    for source in data:
        source_id = source['T17ID'] if 'T17ID' in source.colnames else source['ref_source_id']
        
        if args.debug:
            logger.info(f"Processing source {source_id}")
        
        # Extract photometry data for all apertures
        photometry_data = extract_photometry_data(source, apertures=args.apertures, debug=args.debug)
        
        # Plot the photospectrum
        success = plot_photospectrum(
            source_id=source_id,
            photometry_data=photometry_data,
            apertures=args.apertures,
            output_dir=args.output_dir,
            min_snr=args.min_snr,
            ymin=args.ymin,
            ymax=args.ymax,
            debug=args.debug
        )
        
        if success:
            successful_plots += 1
    
    logger.info(f"Successfully generated {successful_plots} flux photo-spectra in {args.output_dir}")

if __name__ == "__main__":
    main()
