'''
Reproduces the trasmission curve
'''
from __future__ import print_function
import numpy as np
from astropy.io import fits
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from astropy.io import ascii

def load_file(filename):
    wll, ress = [], []
    data = ascii.read(filename)
    for i in data:
        wl = i['col1']
        res = i['col2']
        wll.append(wl)
        ress.append(res)
    return wll, ress

filters= [ r'$J0378$', r'$J0395$', r'$J0410$', r'$J0430$',  r'$J0515$',  r'$J0660$',   r'$J0861$']
filters1= [r'$u$', r'$g$',  r'$r$',  r'$i$', r'$z$']

files = [ 'F378.dat', 'F395.dat', 
'F410.dat', 'F430.dat', 
'F515.dat', 'F660.dat',  'F861.dat']

files1 = ['U.dat', 'G.dat',  'R.dat', 'I.dat', 'Z.dat']

   
# files = ['F378_transm.txt', 'F395_transm.txt', 'F410_transm.txt', 'F430_transm.txt','F515_transm.txt', 'F660_transm.txt',  
#                 'F861_transm.txt']

# files1 = [ 'uJAVA_transm.txt', 'gSDSS_transm.txt',  'F625_transm.txt',
#           'iSDSS_transm.txt', 'zSDSS_transm.txt']


colors = [ "#9900FF", "#6600FF", "#0000FF", "#009999", "#DD8000", "#CC0066", "#660033"]
colors1 = [ "#CC00FF", "#006600", "#FF0000", "#990033", "#330034"]

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
for f, color, filter_ in zip(files, colors, filters):
    x, y = load_file(f)
    ax1.fill(x,y, color=color, label=filter_, zorder=3)
    
for f, color, filter_ in zip(files1, colors1, filters1):
    x, y = load_file(f)
    ax1.fill(x,y,  color=color, alpha =0.3, linewidth=2.8, label=filter_)

# wll, ress = [], []
# data = np.loadtxt('z_sdss_with_ccd_qe.dat', delimiter=None, converters=None, skiprows=0,
#                                        usecols=None, unpack=False, ndmin=0)
# for i in data:
#     wl = str(i[0]*10)
#     res = str(i[1])
#     wll.append(wl)
#     ress.append(res)
# plt.plot(wll, ress,  color='brown', linewidth=3.8, label='zSDSS')

datadir = "../../../Halo-PNe-spectros/"
fitsfile = "spec-0953-52411-0160_PNG_1359559.fits"
hdulist = fits.open(os.path.join(datadir, fitsfile))

wl = (10**hdulist[1].data.field('loglam'))
flux = 1E-17*hdulist[1].data.field('flux')

#DdDm1

def sys(spectra):
    datadir = "../../../Halo-PNe-spectros/"
    #datadir = "../../../syst-spectros/"
    file_ = spectra
    x = np.loadtxt(os.path.join(datadir, file_), delimiter = None, skiprows = 0, usecols = None,
                   unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
    return x['Wl'], x['Flux']

#x, y = sys("Lin358.txt")
x, y = sys("DdDm-1.dat")
y /=1.5e-13 #DdDm1
#y /=5e-14 #symbiotic

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
plt.ylim(0.0, 0.8)
plt.xlim(2500, 11000)
plt.tick_params(axis='x', labelsize=16) 
plt.tick_params(axis='y', labelsize=16)
#plt.plot(x, y, 'k')
#plt.ylim(0.0, 1.0)    
plt.legend(fontsize='small')
plt.xlabel("Wavelength($\AA$)", fontsize= 18)
plt.ylabel("Transmission", fontsize= 18)
plt.tight_layout()
plt.savefig('splus-filter-PN-DdDm1-2018.pdf')
#plt.savefig('splus-filter-SySt-Lha-2018.jpg')
