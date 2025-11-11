#!/usr/bin/env python3
"""
Simple photo-spectra generator for high-quality NGC 5128 globular clusters.
Uses the cleaned catalog with 507 high-quality sources.
Includes both SPLUS and DECam (Taylor) photometry with correct wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Filter wavelengths in Angstroms (updated with correct values)
filter_wavelengths = {
    # SPLUS filters with correct effective wavelengths
    'F378': 3770,   # J0378 [OII]
    'F395': 3940,   # J0395 Ca H + K  
    'F410': 4094,   # J0410 Hδ
    'F430': 4292,   # J0430 G band
    'F515': 5133,   # J0515 Mgb Triplet
    'F660': 6614,   # J0660 Hα
    'F861': 8611,   # J0861 Ca Triplet
    
    # DECam filters with correct central wavelengths from specs
    'u': 3550,      # DECam u-band (355 nm)
    'g': 4730,      # DECam g-band (473 nm)  
    'r': 6420,      # DECam r-band (642 nm)
    'i': 7840,      # DECam i-band (784 nm)
    'z': 9260       # DECam z-band (926 nm)
}

# Colors and markers
DECAM_COLOR = "#1f77b4"   # Blue for DECam filters
DECAM_MARKER = "s"        # Square for DECam filters
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
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    
    # Plot SPLUS filters
    splus_fluxes = []
    splus_wavelengths = []
    splus_labels = {
        'F378': '[OII] 3770Å', 
        'F395': 'CaHK 3940Å', 
        'F410': 'Hδ 4094Å', 
        'F430': 'G-band 4292Å', 
        'F515': 'Mgb 5133Å', 
        'F660': 'Hα 6614Å', 
        'F861': 'CaT 8611Å'
    }
    
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
                       fmt=SPLUS_MARKER, color=SPLUS_COLOR, markersize=10,
                       capsize=5, capthick=2, elinewidth=2, 
                       label='SPLUS' if filter == 'F378' else "",
                       zorder=3)
            
            # Add filter label
            ax.text(wl, flux * 1.05, splus_labels[filter], fontsize=9, 
                   ha='center', va='bottom', color=SPLUS_COLOR, alpha=0.9,
                   fontweight='bold')
            
            splus_fluxes.append(flux)
            splus_wavelengths.append(wl)
    
    # Plot DECam filters (Taylor et al.)
    decam_fluxes = []
    decam_wavelengths = []
    decam_labels = {
        'u': 'u 3550Å', 'g': 'g 4730Å', 'r': 'r 6420Å', 
        'i': 'i 7840Å', 'z': 'z 9260Å'
    }
    
    for decam_filter in ['u', 'g', 'r', 'i', 'z']:
        mag_col = f'{decam_filter}mag'
        err_col = f'e_{decam_filter}mag'
        
        if mag_col in source and not pd.isna(source[mag_col]) and source[mag_col] < 90:
            mag = source[mag_col]
            wl = filter_wavelengths[decam_filter]
            flux = magnitude_to_flux(mag, wl)
            
            # Get error
            if err_col in source and not pd.isna(source[err_col]):
                flux_err = magnitude_to_flux(mag - source[err_col], wl) - flux
            else:
                flux_err = 0.1 * flux  # Default 10% error
            
            # Plot point and error bar
            ax.errorbar(wl, flux, yerr=flux_err,
                       fmt=DECAM_MARKER, color=DECAM_COLOR, markersize=10,
                       capsize=5, capthick=2, elinewidth=2, 
                       label='DECam' if decam_filter == 'u' else "",
                       zorder=3)
            
            # Add filter label
            ax.text(wl, flux * 1.05, decam_labels[decam_filter], fontsize=9,
                   ha='center', va='bottom', color=DECAM_COLOR, alpha=0.9,
                   fontweight='bold')
            
            decam_fluxes.append(flux)
            decam_wavelengths.append(wl)
    
    # Connect points with lines (separately for SPLUS and DECam)
    if len(splus_fluxes) > 1:
        sort_idx = np.argsort(splus_wavelengths)
        ax.plot(np.array(splus_wavelengths)[sort_idx], np.array(splus_fluxes)[sort_idx], 
                '-', color=SPLUS_COLOR, alpha=0.7, linewidth=2, label='_nolegend_',
                zorder=2)
    
    if len(decam_fluxes) > 1:
        sort_idx = np.argsort(decam_wavelengths)
        ax.plot(np.array(decam_wavelengths)[sort_idx], np.array(decam_fluxes)[sort_idx], 
                '-', color=DECAM_COLOR, alpha=0.7, linewidth=2, label='_nolegend_',
                zorder=2)
    
    # Customize plot
    ax.set_xlabel('Wavelength (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', 
                 fontsize=14, fontweight='bold')
    ax.set_title(f'GC {source_id}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    
    # Set x-axis limits and ticks
    ax.set_xlim(3000, 10000)
    ax.set_xticks([3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    ax.set_xticklabels(['3500', '4000', '5000', '6000', '7000', '8000', '9000', '10000'])
    
    # Improve y-axis scaling
    ax.set_ylim(bottom=0)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], loc='best', fontsize=12, framealpha=0.9)
    
    # Add source info
    info_text = f"RA: {source['RAJ2000']:.5f}\nDEC: {source['DEJ2000']:.5f}"
    if 'Prob' in source:
        info_text += f"\nProb: {source['Prob']:.2f}"
    if 'Rgc' in source and not pd.isna(source['Rgc']):
        info_text += f"\nRgc: {source['Rgc']:.2f} arcmin"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontweight='bold')
    
    # Add wavelength regions annotation
    ax.text(0.98, 0.98, "SPLUS: Narrow + Medium\nDECam: Broadband", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"photospectrum_{source_id}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_file

def main():
    """Main function to generate all photo-spectra."""
    
    # Configuration
    CATALOG_PATH = "Results/gc_photometry_CLEAN_FINAL.csv"
    OUTPUT_DIR = "./photospectra_decam_corrected_allSources"
    APERTURE = 3  # Using aperture 3 as in your cleaned catalog
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load catalog
    print(f"Loading catalog: {CATALOG_PATH}")
    df = pd.read_csv(CATALOG_PATH)
    print(f"Found {len(df)} high-quality sources")
    
    # Print filter info for verification
    print("\n📊 Using filter wavelengths:")
    for filter, wl in filter_wavelengths.items():
        print(f"   {filter}: {wl} Å")
    
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
