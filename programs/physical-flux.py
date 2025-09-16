'''
Convert the flux in pysical units

'''
from __future__ import print_function
import glob
from astropy.io import fits
import os
import json
import numpy as np
import argparse
#import matplotlib
#matplotlib.use("Agg")
import panda as pd
import matplotlib.pyplot as plt
import sys 

param = sys.argv[1:]

parser = argparse.ArgumentParser(
    description="""Write wave and flux of a spectrum""")

parser.add_argument("source", type=str,
                    default="_axper_20111004_014",
                    help="Name of blue source, taken the prefix ")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + ".dat"

data = np.loadtxt(regionfile, delimiter = None, skiprows = 0, usecols = None, 
                                          unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Int', 'f8')]))


#Scale factor needed to put the same scale flux
V = input('Enter V magnitude: ')

wl = data['Wl']
F = 10**(-0.4*V - 8.449 )
flux = data['Int']*F

asciifile = regionfile.replace(".dat", "-cali.dat")
file=open(asciifile,'w') #create file  
for x,y in zip(wl, flux):  
    file.write('%f  %s\n'%(x,y))     #assume you separate columns by tabs  
file.close()     #close file  
