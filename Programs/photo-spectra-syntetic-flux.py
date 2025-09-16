'''
Make photo spectra from convolved JPLUS spectra
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
#import seaborn as sns
import sys
import argparse
import os
from colour import Color

magnitude= []
magnitude1= []
magnitude2= []
wl = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
color= ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"]

parser = argparse.ArgumentParser(
    description="""Write wave and magnitude of a spectrum""")

parser.add_argument("source", type=str,
                    default="DdDm-1-HPNe-JPLUS17-magnitude",
                    help="Name of source, taken the prefix ")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
file_ = args.source + ".json"


c = 2.99792458e18 #speed of light in Angstron/sec
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

# Spectrum
def sys(spectra):
    file_ = spectra
    x = np.loadtxt(file_, delimiter = None, skiprows = 1, usecols = None,
                   unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
    return x['Wl'], x['Flux']

x, y = sys("gaia_xp_spectrum_6087393007027689344.dat")
# Leer el source_id desde el header del archivo
with open("gaia_xp_spectrum_6087393007027689344.dat", "r") as f:
    for line in f:
        if line.startswith("# Gaia XP Spectrum for source_id:"):
            gaia_id = line.strip().split(":")[1].strip()
            break
    else:
        gaia_id = "Unknown"

y /= 1e-14
#x, y = sys("DdDm-1.dat")
# x1, y1 = sys("star.dat")
# y1 /= 1e-6 
# x2, y2 = sys("galaxy.sed")
# print(y2)
#y2 /= 1e-8
if args.savefig:
    plotfile = file_.replace(".json", 
                    "-flux.jpg")
    fig = plt.figure(figsize=(14.29, 9))
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(axis='x', labelsize=30) 
    plt.tick_params(axis='y', labelsize=30)
    ax.set_xlim(xmin=3e3,xmax=10000)
    ax.set_ylim(ymin=0.15,ymax=0.52)
    #ax.set_ylim(ymin=-0.1,ymax=1e-10)
    #ax.set_ylim(ymin=0,ymax=12.3)
    #ax1.set_xlabel(r'$\lambda$')
    ax.set_xlabel(r'Wavelength $[\mathrm{\AA]}$', fontsize = 30)
    #ax.set_ylabel(r'Magnitude [AB]', fontsize = 26)
    #ax.set_ylabel(r'Flux $(\mathrm{6^{-16} erg s^{-1} cm^{-2} \AA^{-1}})$', fontsize = 26)
    ax.set_ylabel(r'Flux $(\mathrm{10^{-14} erg\ s^{-1} cm^{-2} \AA^{-1}})$', fontsize = 30)
    ax.plot(x, y, linewidth=3.0, color="black", alpha = 0.6, )
    # ax.plot(x1, y1+4.5, linewidth=2.3, color="black")
    # ax.plot(x2, y2, linewidth=2.3, color="black")
    for wl1, mag, colors, marker_ in zip(wl, magnitude, color, marker):
    # Add a condition to avoid extremely high or infinite magnitudes
        if mag < 90:  # Set an appropriate threshold value based on your data range
            F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
            F /= 1e-14
            ax.scatter(wl1, F, c=colors, marker=marker_, s=400, zorder=3)
        else:
            print(f"Skipping calculation for magnitude: {mag} at wavelength: {wl1}")


    #plt.subplots_adjust(bottom=0.19)
    # Mostrar el Gaia source_id en la figura
    ax.text(0.03, 0.95, f"Gaia source_id: {gaia_id}",
            transform=ax.transAxes,
            fontsize=25, color="black",
            verticalalignment='top')

    # plt.text(0.75, 0.94, 'QSO (z=2.3)',
    #          transform=ax.transAxes, fontsize=25, weight='bold')
    # plt.text(0.75, 0.68, 'Star (A0)',
    #          transform=ax.transAxes, fontsize=25, weight='bold')
    # plt.text(0.71, 0.25, 'Galaxy (z=0.0)',
    #          transform=ax.transAxes, fontsize=25, weight='bold')
    plt.legend(fontsize=20.0)
    #plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.clf()


