#!/usr/bin/env python3
"""
Simple photo-spectra generator for high-quality NGC 5128 globular clusters.
Uses the cleaned catalog with 507 high-quality sources.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Filter wavelengths in Angstroms
filter_wavelengths = {
    'F378': 3785, 'F395': 3950, 'F410': 4100, 'F430': 4300,
    'F515': 5150, 'F660': 6600, 'F861': 8610,
    'u': 3485, 'g': 4803, 'r': 6250, 'i': 7660, 'z': 9110  # Taylor filters
}

# Colors and markers
TAYLOR_COLOR = "#1f77b4"  # Blue for Taylor filters
TAYLOR_MARKER = "s"       # Square for Taylor filters
SPLUS_COLOR = "#ff7f0e"   # Orange for SPLUS filters  
SPLUS_MARKER = "o"        # Circle for SPLUS filters

def magnitude_to_flux(mag, wl_angstrom):
    """Convert AB magnitude to flux in erg/s/cm2/A."""
    if np.isnan(mag) or mag >= 50.0:
        return np.nan
    flux = (10**(-0.4 * (mag + 2.41))) / (wl_angstrom**2)
    return flux / 1e-15  # Convert to 1e-15 units

def plot_photospectrum(source, source_id, output_dir, aperture=3):
    """Plot photo-spectrum for a single source."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    
    # Plot SPLUS filters
    splus_fluxes = []
    splus_wavelengths = []
    
    for filter in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
        mag_col = f'MAG_{filter}_{aperture}'
        err_col = f'MAGERR_{filter}_{aperture}'
        
        if mag_col in source and not pd.isna(source[mag_col]) and source[mag_col] < 90:
            mag = source[mag_col]
            wl = filter_wavelengths[filter]
            flux = magnitude_to_flux(mag, wl)
            
            # Get error
            if err_col in source and not pd.isna(source[err_col]):
                flux_err = magnitude_to_flux(mag - source[err_col], wl) - flux
            else:
                flux_err = 0.1 * flux  # Default 10% error
            
            # Plot point and error bar
            ax.errorbar(wl, flux, yerr=flux_err, 
                       fmt=SPLUS_MARKER, color=SPLUS_COLOR, markersize=8,
                       capsize=4, capthick=2, elinewidth=2, label='SPLUS' if filter == 'F378' else "")
            
            splus_fluxes.append(flux)
            splus_wavelengths.append(wl)
    
    # Plot Taylor filters
    taylor_fluxes = []
    taylor_wavelengths = []
    
    for taylor_filter, splus_name in [('u', 'u'), ('g', 'g'), ('r', 'r'), ('i', 'i'), ('z', 'z')]:
        if taylor_filter + 'mag' in source and not pd.isna(source[taylor_filter + 'mag']):
            mag = source[taylor_filter + 'mag']
            wl = filter_wavelengths[splus_name]
            flux = magnitude_to_flux(mag, wl)
            
            # Get error
            err_col = 'e_' + taylor_filter + 'mag'
            if err_col in source and not pd.isna(source[err_col]):
                flux_err = magnitude_to_flux(mag - source[err_col], wl) - flux
            else:
                flux_err = 0.1 * flux  # Default 10% error
            
            # Plot point and error bar
            #ax.errorbar(wl, flux, yerr=flux_err,
            #           fmt=TAYLOR_MARKER, color=TAYLOR_COLOR, markersize=8,
            #           capsize=4, capthick=2, elinewidth=2, label='Taylor' if taylor_filter == 'u' else "")
            
            #taylor_fluxes.append(flux)
            #taylor_wavelengths.append(wl)
    
    # Connect points with lines (separately for SPLUS and Taylor)
    if len(splus_fluxes) > 1:
        sort_idx = np.argsort(splus_wavelengths)
        ax.plot(np.array(splus_wavelengths)[sort_idx], np.array(splus_fluxes)[sort_idx], 
                '-', color=SPLUS_COLOR, alpha=0.7, linewidth=1, label='_nolegend_')
    
    #if len(taylor_fluxes) > 1:
    #    sort_idx = np.argsort(taylor_wavelengths)
    #    ax.plot(np.array(taylor_wavelengths)[sort_idx], np.array(taylor_fluxes)[sort_idx], 
    #            '-', color=TAYLOR_COLOR, alpha=0.7, linewidth=1, label='_nolegend_')
    
    # Customize plot
    ax.set_xlabel('Wavelength (Å)', fontsize=14)
    ax.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=14)
    ax.set_title(f'GC {source_id}', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best', fontsize=12)
    
    # Add source info
    info_text = f"RA: {source['RAJ2000']:.5f}\nDEC: {source['DEJ2000']:.5f}"
    if 'Prob' in source:
        info_text += f"\nProb: {source['Prob']:.2f}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"photospectrum_{source_id}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main function to generate all photo-spectra."""
    
    # Configuration
    CATALOG_PATH = "Results/gc_photometry_final_high_quality_preliminar_teste_aperture3_only.csv"
    OUTPUT_DIR = "./photospectra_simple_highquality"
    APERTURE = 3  # Using aperture 3 as in your cleaned catalog
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load catalog
    print(f"Loading catalog: {CATALOG_PATH}")
    df = pd.read_csv(CATALOG_PATH)
    print(f"Found {len(df)} high-quality sources")
    
    # Generate photo-spectra for all sources
    successful_plots = 0
    
    for idx, source in df.iterrows():
        source_id = source['T17ID'] if 'T17ID' in source else f"GC_{idx}"
        
        try:
            output_file = plot_photospectrum(source, source_id, OUTPUT_DIR, APERTURE)
            successful_plots += 1
            if successful_plots % 50 == 0:
                print(f"Generated {successful_plots} plots...")
                
        except Exception as e:
            print(f"Error plotting {source_id}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {successful_plots} photo-spectra")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
