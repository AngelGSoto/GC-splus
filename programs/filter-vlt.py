'''
Reproduces the trasmission curve
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import sys

# Transmission curve
def load_file(filename):
    wll, ress = [], []
    data = np.loadtxt(filename, delimiter=None, converters=None, skiprows=0,
                                       usecols=None, unpack=False, ndmin=0)
    for i in data:
        wl = str(i[0])
        res = i[1]
        #res*= 30
        wll.append(wl)
        ress.append(res)
    return wll, ress

x, y = load_file("SII2000trans.dat")
x1, y1 = load_file("SII4500trans.dat")

y1-=np.float64(0.05)
#x1*=30


#sys.exit()
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
#plt.xlim(6500, 7100)
#plt.xlim(3000, 10000) # cambinated
#plt.ylim(5.1, 10)      #red
#plt.ylim(4.1, 5.65)  #blue
#plt.ylim(0.0, 20)  #Combinated
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18)
plt.plot(x, y, color="black", label="SII2000")
plt.plot(x1, y1, color="green", label="SII4500")
plt.legend(fontsize=15)
plt.xlabel("Wavelength($\AA$)", fontsize= 16)
plt.ylabel("Transmition ", fontsize= 16)
plt.grid()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.126)
#plt.savefig('spectrum-star2-trans.pdf')
#plt.savefig('spectrum-star2-trans-zoon-new.pdf')
plt.savefig('filters-vlt.jpg')
#plt.savefig('star2-red.pdf')
