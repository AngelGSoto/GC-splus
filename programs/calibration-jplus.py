from __future__ import print_function
import numpy as np
from astropy.io import fits
import os
import glob
import json
import matplotlib.pyplot as plt
from astropy.table import Table
from collections import OrderedDict

pattern = "*.apdat"
file_list = glob.glob(pattern)

#magn =  OrderedDict({"id": "H4-1" })
magn =  OrderedDict({"id": "NGC2242" })

for file_name in sorted(file_list):
    x = np.loadtxt(file_name,  delimiter = None, converters = None, skiprows = 0, 
                         usecols = None, unpack = False, ndmin = 0,)
    #y =  1.26 + 0.86*float(x[7]) #H4 1 y = 1.26 + 0.86x
    y = float(x[7])   #y = 7.60 + 0.54x
    #magn[file_name.split('-1_')[-1].split('_s')[0]] = y
    magn[file_name.split('2_')[-1].split('_s')[0]] = y
    
    #print(file_name.split('-1_')[-1].split('_s')[0], x[9])

#jsonfile = "H4-1-jplus-6pix.json"
jsonfile = "NGC2242-jplus-6pix.json"
with open(jsonfile, "w") as f:
    json.dump(magn, f, indent=4)

