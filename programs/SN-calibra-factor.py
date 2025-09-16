#import matplotlib
#matplotlib.use("Agg")
import panda as pd
import matplotlib.pyplot as plt
import sys 
import argparse
import numpy as np

param = sys.argv[1:]

parser = argparse.ArgumentParser(
    description="""Write wave and flux of a spectrum""")

parser.add_argument("source", type=str,
                    default="H-10b.0016",
                    help="Name of blue source, taken the prefix ")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + ".dat"

data = np.loadtxt(regionfile, delimiter = None, skiprows = 0, usecols = None, 
                                          unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
wl = data['Wl']
flux  = data['Flux'] + 10e-17

asciifile = regionfile.replace(".dat", "cal10.dat")
file=open(asciifile,'w') #create file  
for x,y in zip(wl, flux):  
    file.write('%f  %s\n'%(x,y))     #assume you separate columns by tabs  
file.close()     #close file  
