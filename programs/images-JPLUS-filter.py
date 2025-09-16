'''
To make JPLUS spectra from JPLUS images
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


pattern = "*.apdat"
file_list = glob.glob(pattern)

for file_name in file_list:
    x = np.loadtxt(file_name,  delimiter = None, converters = None, skiprows = 0, 
                         usecols = None, unpack = False, ndmin = 0,)
    print(file_name ,x[7], x[8], x[9], x[9], x[10])
    plt.plot(x, y)
