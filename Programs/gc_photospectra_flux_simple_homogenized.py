#!/usr/bin/env python3
"""
Quick homogenization test - plot specific GCs with/without homogenization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load catalog
df = pd.read_csv("../Results/gc_photometry_final_high_quality_preliminar_teste_aperture3_only.csv")

# Homogenization offsets
offsets = {
    'F378': 0.091, 'F395': 0.213, 'F410': 0.803, 
    'F430': -0.635, 'F515': -0.087, 'F660': -0.045, 'F861': -0.073
}

filter_wl = {'F378': 3785, 'F395': 3950, 'F410': 4100, 'F430': 4300,
             'F515': 5150, 'F660': 6600, 'F861': 8610,
             'u': 3485, 'g': 4803, 'r': 6250, 'i': 7660, 'z': 9110}

def mag_to_flux(mag, wl):
    return (10**(-0.4 * (mag + 2.41))) / (wl**2) / 1e-15

def quick_comparison(source_id, aperture=3):
    """Quick plot comparing raw vs homogenized SPLUS photometry."""
    
    source = df[df['T17ID'] == source_id].iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw photometry
    for filter in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
        mag_col = f'MAG_{filter}_{aperture}'
        if mag_col in source and not pd.isna(source[mag_col]) and source[mag_col] < 90:
            mag_raw = source[mag_col]
            flux_raw = mag_to_flux(mag_raw, filter_wl[filter])
            ax1.plot(filter_wl[filter], flux_raw, 'o', color='red', markersize=8, 
                    label='SPLUS (raw)' if filter == 'F378' else "")
    
    # Plot Taylor filters
    for tf in ['u', 'g', 'r', 'i', 'z']:
        mag_col = tf + 'mag'
        if mag_col in source and not pd.isna(source[mag_col]):
            flux = mag_to_flux(source[mag_col], filter_wl[tf])
            ax1.plot(filter_wl[tf], flux, 's', color='blue', markersize=8,
                    label='Taylor' if tf == 'u' else "")
    
    ax1.set_title(f'GC {source_id} - Raw Photometry\n(SPLUS appears brighter)')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Homogenized photometry
    for filter in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
        mag_col = f'MAG_{filter}_{aperture}'
        if mag_col in source and not pd.isna(source[mag_col]) and source[mag_col] < 90:
            mag_raw = source[mag_col]
            mag_corrected = mag_raw - offsets[filter]  # Apply offset
            flux_corrected = mag_to_flux(mag_corrected, filter_wl[filter])
            ax2.plot(filter_wl[filter], flux_corrected, 'o', color='green', markersize=8,
                    label='SPLUS (homogenized)' if filter == 'F378' else "")
    
    # Plot Taylor filters (same as before)
    for tf in ['u', 'g', 'r', 'i', 'z']:
        mag_col = tf + 'mag'
        if mag_col in source and not pd.isna(source[mag_col]):
            flux = mag_to_flux(source[mag_col], filter_wl[tf])
            ax2.plot(filter_wl[tf], flux, 's', color='blue', markersize=8,
                    label='Taylor' if tf == 'u' else "")
    
    ax2.set_title(f'GC {source_id} - With Homogenization\n(Better agreement)')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel(r'F$_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"homogenization_comparison_{source_id}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Saved: homogenization_comparison_{source_id}.png")

# Test with a few examples
if __name__ == "__main__":
    # Test with first 3 GCs
    for source_id in df['T17ID'].head(20):
        quick_comparison(source_id)
