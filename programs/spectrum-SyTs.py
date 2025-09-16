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

fitsfile = "star2.fits"
hdulist = fits.open(fitsfile)

hdu = hdulist[0]
nx, wav0, i0, dwav = [hdu.header[k] for k in ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav

Flux = hdulist[0].data
#Flux*=10**-16
#Flux/=19

# STAR 2 spectrum

fitsfile1 = "star2b.fits"
hdulist1 = fits.open(fitsfile1)

hdu1 = hdulist1[0]
nx, wav0, i0, dwav = [hdu1.header[k] for k in ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
wavs1 = wav0 + (np.arange(nx) - (i0 - 1))*dwav

Flux1 = hdulist1[0].data

# Transmission curve
def load_file(filename):
    wll, ress = [], []
    data = np.loadtxt(filename, delimiter=None, converters=None, skiprows=0,
                                       usecols=None, unpack=False, ndmin=0)
    for i in data:
        wl = str(i[0])
        res = i[1]
        res*= 30
        wll.append(wl)
        ress.append(res)
    return wll, ress

# colors = ['black', 'green']
# pattern = "../filters/VLT/*trans.dat"
# file_list = glob.glob(pattern)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for f, color in zip(file_list, colors):
#     x, y = load_file(f)
#     ax1.plot(x,y, label=f.split('T/')[-1].split('tr')[0], c=color, linewidth=1)

x, y = load_file("SII2000trans.dat")
x1, y1 = load_file("SII4500trans.dat")

y1-=np.float64(0.05)
#x1*=30

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
#plt.xlim(6500, 7100)
plt.xlim(3000, 10000) # cambinated
#plt.ylim(5.1, 10)      #red
#plt.ylim(4.1, 5.65)  #blue
plt.ylim(0.0, 20)  #Combinated
plt.plot(wavs1, Flux1*1.08, color="blue") #1.08 to calbration 
plt.plot(wavs, Flux, color="red")
plt.plot(x, y, color="black") #1.08 to calbration 
plt.plot(x1, y1, color="green")
plt.tick_params(axis='x', labelsize=14) 
plt.tick_params(axis='y', labelsize=14)     
plt.legend()
plt.xlabel("Wavelength($\AA$)", fontsize= 15)
plt.ylabel("Flux ", fontsize= 15)
#plt.savefig('spectrum-star2-trans.pdf')
#plt.savefig('spectrum-star2-trans-zoon-new.pdf')
plt.savefig('star2-blue-red.pdf')
#plt.savefig('star2-red.pdf')
