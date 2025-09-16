"""
Create a list with all the filters

wavelenght and filter response 
""" 
from __future__ import print_function
import glob
import os
import numpy as np
from astropy.io import ascii
import tarfile

filterss = []
wll = []
ress = []

# Taken from http://www.splus.iag.usp.br/instrumentation/
pattern = "F*.dat"


file_list = glob.glob(pattern)

for file_name in sorted(file_list):
    data = np.loadtxt(file_name, delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    for i in data:
        #filters = file_name.split('/')[-1].split('tr')[0]
        filters = file_name.split('/')[-1].split('.da')[0]
        wl = i[0]                                                      
        res = i[1]
        filterss.append(filters)
        wll.append(wl)
        ress.append(res)
        #print(filters, wl, res)
print(filterss)      
asciifile = "SPLUS21.filter"
file=open(asciifile,'w') #create file  
for x,y,z in zip(filterss, wll, ress):  
    file.write('%s  %f  %f\n'%(x,y,z))     #assume you separate columns by tabs  
file.close()     #close file  

#DAT =  np.column_stack((filterss, wll, ress))
#np.savetxt('SPLUS.filter', DAT, delimiter=" ") 

