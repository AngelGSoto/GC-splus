#!/usr/bin/env python3
'''
Make photo spectra from convolved JPLUS spectra
WITH TRANSMISSION CURVES AND FLEXIBLE Y-LIMITS
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import sys
import argparse
import os
from astropy.io import ascii

magnitude = []
wl = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
color = ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", 
         "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"]
filters = [r'$u$', r'$J0378$', r'$J0395$', r'$J0410$', r'$J0430$', 
           r'$g$', r'$J0515$', r'$r$', r'$J0660$', r'$i$', r'$J0861$', r'$z$']

# Transmission curve files
transmission_files = [
    '../../filters/F0352_uJAVA.dat', 
    '../../filters/F0378.dat', 
    '../../filters/F0395.dat', 
    '../../filters/F0410.dat', 
    '../../filters/F0430.dat', 
    '../../filters/F0475_gSDSS.dat',
    '../../filters/F0515.dat', 
    '../../filters/F0626_rSDSS.dat', 
    '../../filters/F0660.dat', 
    '../../filters/F0769_iSDSS.dat', 
    '../../filters/F0861.dat', 
    '../../filters/F0883_zSDSS.dat'
]

# z-order and alpha for transmission curves
zorder_trans = [1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1]
alpha_trans = [0.15, 0.25, 0.25, 0.35, 0.25, 0.15, 0.25, 0.15, 0.25, 0.15, 0.25, 0.15]

parser = argparse.ArgumentParser(
    description="""Write wave and magnitude of a spectrum""")

parser.add_argument("source", type=str,
                    default="Object-SPLUS-magnitude",
                    help="Name of source, taken the prefix ")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

parser.add_argument("--ymin", required=False, type=float, default=0.15,
                    help="Minimum value for y-axis")

parser.add_argument("--ymax", required=False, type=float, default=0.52,
                    help="Maximum value for y-axis")

args = parser.parse_args()
file_ = args.source + ".json"

def load_file(filename):
    """Load transmission curve data."""
    wll, ress = [], []
    try:
        data = ascii.read(filename)
        for i in data:
            wl_val = i['col1']
            res = i['col2']
            wll.append(wl_val)
            ress.append(res)
    except Exception as e:
        if args.debug:
            print(f"Warning: Could not load {filename}: {e}")
    return wll, ress

# Read magnitude data from JSON
with open(file_) as f:
    data = json.load(f)
    magnitude.append(data["F0352_uJAVA"])
    magnitude.append(data["F0378"])
    magnitude.append(data["F0395"])
    magnitude.append(data["F0410"])
    magnitude.append(data["F0430"])
    magnitude.append(data["F0475_gSDSS"])
    magnitude.append(data["F0515"]) 
    magnitude.append(data["F0626_rSDSS"]) 
    magnitude.append(data["F0660"])
    magnitude.append(data["F0769_iSDSS"]) 
    magnitude.append(data["F0861"]) 
    magnitude.append(data["F0883_zSDSS"])

# Spectrum reading function
def sys(spectra):
    file_ = spectra
    x = np.loadtxt(file_, delimiter=None, skiprows=1, usecols=None,
                   unpack=False, dtype=np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
    return x['Wl'], x['Flux']

# Read Gaia spectrum
x, y = sys("gaia_xp_spectrum_6087393007027689344.dat")

# Read Gaia source_id from header
with open("gaia_xp_spectrum_6087393007027689344.dat", "r") as f:
    gaia_id = "Unknown"
    for line in f:
        if line.startswith("# Gaia XP Spectrum for source_id:"):
            gaia_id = line.strip().split(":")[1].strip()
            break

y /= 1e-14  # Adjust flux scaling

if args.savefig:
    plotfile = file_.replace(".json", "-flux-with-transmission.pdf")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Clean frame without spines on top and right
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    
    plt.tick_params(axis='x', labelsize=25) 
    plt.tick_params(axis='y', labelsize=25)
    
    # Set axis limits using command line arguments or defaults
    ax.set_xlim(xmin=3000, xmax=9500)
    ax.set_ylim(ymin=args.ymin, ymax=args.ymax)
    
    ax.set_xlabel(r'Wavelength $[\mathrm{\AA]}$', fontsize=25)
    ax.set_ylabel(r'Flux $(\mathrm{10^{-14} erg\ s^{-1} cm^{-2} \AA^{-1}})$', fontsize=25)
    
    # ================================================================
    # PLOT TRANSMISSION CURVES (USING SECOND SCRIPT APPROACH)
    # ================================================================
    # Calculate transmission height as a small fraction of y-range
    transmission_height = (args.ymax - args.ymin) * 0.8  # 8% of y-axis range
    
    for f, color_trans, filter_label, zord, alp in zip(transmission_files, color, 
                                                      filters, zorder_trans, alpha_trans):
        try:
            x_trans, y_trans = load_file(f)
            if x_trans and y_trans:
                # Normalize transmission curve to max = 1
                y_trans = np.array(y_trans)
                y_trans = y_trans / y_trans.max()
                
                # Scale transmission curve to fit nicely in plot
                y_scaled = y_trans * transmission_height
                
                # Shift to bottom of plot
                y_shifted = y_scaled + args.ymin
                
                # Plot transmission curve with fill
                ax.fill(x_trans, y_shifted, color=color_trans, alpha=alp, zorder=zord)
                
                # Add filter label at the top of the transmission curve
                max_y = max(y_shifted)
                max_x_index = np.argmax(y_trans)  # Use normalized y_trans for position
                max_x = x_trans[max_x_index]
                
                # Adjust the horizontal alignment, offset, and vertical position based on the filter
                # USING VALUES FROM SECOND SCRIPT BUT ADJUSTED FOR OUR Y-SCALE
                if filter_label == r'$g$':
                    ha = 'left'
                    x_offset = -500  # Move the label to the left
                    y_offset = -0.02  # Slightly above transmission curve
                elif filter_label == r'$r$':
                    ha = 'left'
                    x_offset = -700  # Move the label to the left
                    y_offset = -0.02  # Slightly above transmission curve
                elif filter_label == r'$u$':
                    ha = 'left'
                    x_offset = -350  # Move the label to the left
                    y_offset = -0.2  # Slightly above transmission curve
                elif filter_label == r'$i$':
                    ha = 'center'
                    x_offset = 300  # Move the label to the right
                    y_offset = 0.0  # Slightly above transmission curve
                elif filter_label == r'$z$':
                    ha = 'center'
                    x_offset = 500  # Move the label to the right
                    y_offset = -0.1  # Slightly above transmission curve
                elif filter_label == r'$J0861$':
                    ha = 'center'
                    x_offset = 180  # Move the label to the right
                    y_offset = 0.0  # Slightly above transmission curve
                elif filter_label == r'$J0378$':
                    ha = 'center'
                    x_offset = -100
                    y_offset = -0.15
                elif filter_label == r'$J0395$':
                    ha = 'center'
                    x_offset = -0.0
                    y_offset = -0.1
                elif filter_label == r'$J0410$':
                    ha = 'center'
                    x_offset = 0
                    y_offset = -0.05
                elif filter_label == r'$J0430$':
                    ha = 'center'
                    x_offset = 0
                    y_offset = 0.0
                elif filter_label == r'$J0515$':
                    ha = 'center'
                    x_offset = 0
                    y_offset = -0.0
                elif filter_label == r'$J0660$':
                    ha = 'center'
                    x_offset = 0
                    y_offset = 0.0
                else:
                    ha = 'center'
                    x_offset = 0
                    y_offset = 0.015  # Default offset
                
                # Position label
                label_x = max_x + x_offset
                label_y = max_y + y_offset
                
                ax.text(label_x, label_y, filter_label, 
                        color=color_trans, fontsize=18, ha=ha, va='bottom', 
                        zorder=zord+1, fontweight='bold')
                
                if args.debug:
                    print(f"✓ Loaded transmission curve: {filter_label} at {max_x:.0f}Å")
                    
        except Exception as e:
            if args.debug:
                print(f"✗ Could not load transmission curve {f}: {e}")
    
    # ================================================================
    # PLOT SPECTRUM AND PHOTOMETRY
    # ================================================================
    ax.plot(x, y, linewidth=3.0, color="black", alpha=0.5, zorder=10)
    
    valid_points = 0
    for wl1, mag, colors, marker_ in zip(wl, magnitude, color, marker):
        if mag < 90:  # Valid magnitude threshold
            F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
            F /= 1e-14
            ax.scatter(wl1, F, c=colors, marker=marker_, s=300, 
                      edgecolors='k', linewidth=1.5, zorder=20)
            valid_points += 1
        else:
            if args.debug:
                print(f"Skipping magnitude {mag} at wavelength {wl1}Å")
    
    if args.debug:
        print(f"✓ Plotted {valid_points} valid photometry points")
    
    # Connect photometry points with subtle line
    valid_indices = [i for i, mag in enumerate(magnitude) if mag < 90]
    if len(valid_indices) > 1:
        valid_wl = [wl[i] for i in valid_indices]
        valid_flux = [(10**(-(magnitude[i] + 2.41) / 2.5)) / wl[i]**2 / 1e-14 
                     for i in valid_indices]
        
        sort_idx = np.argsort(valid_wl)
        sorted_wl = np.array(valid_wl)[sort_idx]
        sorted_flux = np.array(valid_flux)[sort_idx]
        
        # Optional: uncomment to connect points
        # ax.plot(sorted_wl, sorted_flux, '-', color='#666666', 
        #         alpha=0.6, linewidth=1.5, zorder=15)
    
    # Add Gaia source_id
    ax.text(0.03, 0.97, f"Gaia source_id: {gaia_id}",
            transform=ax.transAxes,
            fontsize=18, color="black",
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(plotfile, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved: {plotfile}")
    plt.clf()
else:
    # Debug mode - print magnitudes
    print("\n📊 Magnitudes from JSON file:")
    for wl1, mag, filter_name in zip(wl, magnitude, filters):
        if mag < 90:
            F = (10**(-(mag + 2.41) / 2.5)) / wl1**2 / 1e-14
            print(f"  {filter_name} ({wl1}Å): mag={mag:.3f}, flux={F:.4f}")
        else:
            print(f"  {filter_name} ({wl1}Å): mag={mag} (INVALID)")
