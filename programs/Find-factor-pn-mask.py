"""
Multiplicate factor of scale of the HII spectra
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
import pandas as pd
import matplotlib.pyplot as plt
import sys 

param = sys.argv[1:]

parser = argparse.ArgumentParser(
    description="""Write wave and flux of a spectrum""")

parser.add_argument("source", type=str,
                    default="H-10b.0016",
                    help="Name of blue source, taken the prefix ")

parser.add_argument("source1", type=str,
                    default="H-10r.0016",
                    help="Name of red source, taken the prefix")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + ".dat"
regionfile1 = args.source1 + ".dat"

data = np.loadtxt(regionfile, delimiter = None, skiprows = 0, usecols = None, 
                                          unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
data1 = np.loadtxt(regionfile1, delimiter = None, skiprows = 0, usecols = None, 
                                          unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))

#mask
# mask = (data['Flux']>=0.0)
# mask1 = (data1['Flux']>=0.0)
mask3 = (data['Wl']<=6142.70)
mask4 = (data1['Wl']<=10000.0)

#Scale factor needed to put the same scale flux
factor = input('Enter scale factor: ')

wl, wl1 = data['Wl'][mask3] + 5.0, data1['Wl'][mask4] + 5.0
flux, flux1 = (data['Flux'][mask3]+2.2e-18),  (data1['Flux'][mask4]+1.2e-18)*float(factor)


asciifile = regionfile.replace(".dat", "+r.dat")
file=open(asciifile,'w') #create file  
for x,y in zip(wl, flux):  
    file.write('%f  %s\n'%(x,y))     #assume you separate columns by tabs  
for x,y in zip(wl1, flux1):  
    file.write('%f  %s\n'%(x,y))    
file.close()     #close file  
