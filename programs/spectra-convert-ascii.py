"""
Convert file.fits in file ascii (spectrum)
""" 
from __future__ import print_function
import glob
from astropy.io import fits
import os
import json
import numpy as np
import argparse
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="""Convert file.fits in file ascii""")

parser.add_argument("source", type=str,
                    default="dddm1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
regionfile = cmd_args.source + ".fits"

hdulist = fits.open(regionfile)

x0, wave0, y0, wave = [hdulist[0].header[k] for k in ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")] #"CDELT1")]#"CD1_1")]
wavelength = wave0 + (np.arange(x0) - (y0 - 1))*wave

# Flux = hdulist[0].data
# Flux *= 10**(-16)

# asciifile = regionfile.replace(".fits", ".dat")
# file=open(asciifile,'w') #create file  
# for x,y in zip(wavelength, Flux ):  
#     file.write('%f %s\n'%(x,y))     #assume you separate columns by tabs  
# file.close()     #close file  


# ojo con esto es para espectros que no tienen el flujo

Flux = hdulist[0].data.mean(axis=0)

for Flux_col in Flux:  
   asciifile = regionfile.replace(".fits", ".dat")
   file=open(asciifile,'w') #create file  
   for x,y in zip(wavelength, Flux_col):  
       file.write('%f %s\n'%(x,y))     #assume you separate columns by tabs  
   file.close()     #close file   
