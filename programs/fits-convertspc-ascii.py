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

wavelength = hdulist[1].data.field("Wavelength")
Flux = hdulist[1].data.field("Flux")
Flux *= 1e-17

asciifile = regionfile.replace(".fits", "-CV.dat")
file=open(asciifile,'w') #create file  
for x,y in zip(wavelength, Flux ):  
    file.write('%f %s\n'%(x,y))     #assume you separate columns by tabs  
file.close()     #close file  
