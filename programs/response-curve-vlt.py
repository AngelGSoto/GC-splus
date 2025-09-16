'''
Rewritten the trasmission curve
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import sys

filterss = []
wll = []
ress = []

filter1 = "VLT/SII2000trans.dat"
filter2 = "VLT/SII4500trans.dat"

data = np.loadtxt(filter1, delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
for i in data:
    filters = filter1.split('/')[-1].split('tr')[0]
    wl = i[0]
    res = i[1]
    filterss.append(filters)
    wll.append(wl)
    ress.append(res)

data1 = np.loadtxt(filter2, delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
for i in data1:
    filters1 = filter2.split('/')[-1].split('tr')[0]
    wl1 = i[0]
    res1 = i[1]
    res1-=np.float64(0.05)
    filterss.append(filters1)
    wll.append(wl1)
    ress.append(res1)

asciifile = "VLT.filter"
file=open(asciifile,'w') #create file  
for x,y,z in zip(filterss, wll, ress):  
    file.write('%s  %f  %f\n'%(x,y,z))     #assume you separate columns by tabs  
file.close()     #close file  

#DAT =  np.column_stack((filterss, wll, ress))
#np.savetxt('SPLUS.filter', DAT, delimiter=" ") 
