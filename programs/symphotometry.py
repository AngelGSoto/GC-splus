"""
Find the values in the spectrums (file ascii) 

Estimate the magnitudes for any photometryc system
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
from urllib.parse import parse_qs
import warnings
#with warnings.catch_warnings():
    #warnings.filterwarnings("ignore",category=FutureWarning)
    #import h5py


parser = argparse.ArgumentParser(
    description="""Estimate magnitudes of a spectrum for any photometryc system""")

parser.add_argument("source", type=str,
                    default="DdDm-1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--filters", type=str,
                    default="SPLUS21",
                    help="Filter of the photometryc system")

parser.add_argument("--name", type=str,
                    default="HPNe",
                    help="Name to clasify the source in the JSON file")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
# Construct the regionfile with proper extension handling
if not cmd_args.source.endswith('.dat'):
    regionfile = cmd_args.source + ".dat"
else:
    regionfile = cmd_args.source


# Getting any filter set that you need
datadir = "../../filters/"
filterfile = cmd_args.filters + ".filter"

f = readfilterset()
f.read(os.path.join(datadir, filterfile))
f.uniform()
f.calc_filteravgwls()

# Dictionary
magn = OrderedDict({"id": regionfile.replace(".dat", "-{}".format(cmd_args.name)) })
#magn = {"id": regionfile }

dt = np.dtype([('wl', 'f4'), ('flux', 'f4')])
obs_spec = np.loadtxt(regionfile, delimiter=None, skiprows=1, dtype=dt)

print(obs_spec["wl"])

# Estimate of magnitude of the photometric system
x = spec2filterset(f.filterset, obs_spec, model_spec = None, badpxl_tolerance = 0.5)
for xx, yy in zip(np.unique(f.filterset['ID_filter']), x['m_ab']):
    xx=str(xx).split("b'")[-1].split("'")[0]
    magn[xx] = float(yy)
if cmd_args.debug:
    print("Calculating the magnitude of:", magn["id"])

# Plot magnitude vs filter or wavelenght 
if cmd_args.savefig:
    plotfile = regionfile.replace(".dat", 
                    "-{}-{}-magnitude.pdf".format(cmd_args.name, 
                                 cmd_args.filters.split('am')[0]))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
    plt.tick_params(axis='x', labelsize=19) 
    plt.tick_params(axis='y', labelsize=19)
    ax1.set_xlabel(r'Wavelength($\AA$)', size = 19)
    ax1.set_ylabel(r'Magnitude', size = 19)
    ax1.plot(f.filteravgwls, x['m_ab'], 'ko-')
    ax1.set_title(" ".join([cmd_args.source]))
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
    plt.margins(0.06)
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(plotfile)

# Creates The JSON files with the magnitudes
jsonfile = regionfile.replace(".dat", 
                  "-{}-{}-magnitude.json".format(cmd_args.name, 
                                cmd_args.filters.split('am')[0]))
with open(jsonfile, "w") as f:
    json.dump(magn, f, indent=4)



