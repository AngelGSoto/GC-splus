from __future__ import print_function
import numpy as np
from astropy.io import fits
import os
import glob
import json
import matplotlib.pyplot as plt
from astropy.table import Table
from collections import OrderedDict


data = np.loadtxt("DdDm1-3.dat",  delimiter = None, converters = None, skiprows = 0, 
                         usecols = None, unpack = False, ndmin = 0,)
wl = data[:,0]
flux = data[:,1]
flux*= 2.6915348039269138e-14

asciifile = "DdDm1-3-correc.dat"
file=open(asciifile,'w') #create file  
for x,y in zip(wl, flux):  
    file.write('%f  %s\n'%(x,y))     #assume you separate columns by tabs  
file.close()     #close file  
