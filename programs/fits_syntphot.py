"""
Find the values in the spectrums for file.fits 

Estimate the magnitudes for the any photometryc system 
""" 
from __future__ import print_function
import glob
from syntphot import spec2filterset
from readfilterset import readfilterset
from syntphot import photoconv

from astropy.io import fits
import os
import json
import numpy as np
import argparse
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, getopt
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description="""Estimate Alhambra magnitudes of a spectrum""")

parser.add_argument("source", type=str,
                    default="dddm1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--filters", type=str,
                    default="Alhambra3",
                    help="Filter of the photometric system")

parser.add_argument("--name", type=str,
                    default="HPNe",
                    help="Name to clasify the source in the JSON file")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
regionfile = cmd_args.source + ".fits"

# Get any filters that you need
datadir = "../filters/"
filterfile = cmd_args.filters + ".filter"

f = readfilterset()
f.read(os.path.join(datadir, filterfile))
f.uniform()
f.calc_filteravgwls()

# creates dictionary
magn = OrderedDict({"id": regionfile.replace(".fits", "-{}".format(cmd_args.name))})

h = photoconv()
y = h.fromSDSSfits(f.filterset, regionfile, badpxl_tolerance = 0.5)
for xx, yy in zip(np.unique(f.filterset['ID_filter']), y['m_ab']):
        xx=str(xx).split("b'")[-1].split("'")[0]
        magn[xx] = float(yy)    

# Find the redshift of the source
hdu = fits.open(regionfile)
z = hdu[2].data['Z']
magn["Z"] = float(z)

if cmd_args.debug:
    print("Calculating the magnitude of:", magn["id"])

# Plot magnitude vs filter or wavelenght   
if cmd_args.savefig:
    plotfile = regionfile.replace(".fits", 
                "-{}-{}-magnitude.pdf".format(cmd_args.name, 
                                 cmd_args.filters.split('am')[0]))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
    plt.xlabel(r'Wavelength($\AA$)', size = 14)
    plt.ylabel(r'Magnitude', size = 14)
    ax1.plot(f.filteravgwls, y['m_ab'], 'ko-')
    ax1.set_title(" ".join([cmd_args.source]))
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                      #rotation='vertical', size = 'small')
    plt.margins(0.06)
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(plotfile)

jsonfile = regionfile.replace(".fits", 
               "-{}-{}-magnitude.json".format(cmd_args.name, 
                                 cmd_args.filters.split('am')[0]))
with open(jsonfile, "w") as f:
    json.dump(magn, f, indent=4)
