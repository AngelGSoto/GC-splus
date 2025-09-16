'''
Reproduces the trasmission curve
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_file(filename):
    wll, ress = [], []
    data = np.loadtxt(filename, delimiter=None, converters=None, skiprows=0,
                                       usecols=None, unpack=False, ndmin=0)
    for i in data:
        wl = str(i[0])
        res = str(i[1])
        wll.append(wl)
        ress.append(res)
    return wll, ress

filters= [ r'$J0378$', r'$J0395$', r'$J0410$', r'$J0430$',  r'$J0515$',  r'$J0660$',   r'$J0861$']
filters1= [r'$u$', r'$g$',  r'$r$',  r'$i$', r'$z$']

# files = [ 'F348_with_ccd_qe.dat', 'F378_with_ccd_qe.dat', 'F395_with_ccd_qe.dat', 'F410_with_ccd_qe.dat', 
#           'F430_with_ccd_qe.dat', 'filter_F515_average.txt', 'filter_Ha_average.txt', 'F861_with_ccd_qe.dat']

# files1 = ['F348_with_ccd_qe.dat', 'g_sdss_with_ccd_qe.dat',  'filter_rsdss_average.txt', 'i_sdss_with_ccd_qe.dat', 'z_sdss_with_ccd_qe.dat']

files = [ 'JPLUS_J0378_FullEfficiency.tab', 'JPLUS_J0395_FullEfficiency.tab', 'JPLUS_J0410_FullEfficiency.tab', 'JPLUS_J0430_FullEfficiency.tab', 
          'JPLUS_J0515_FullEfficiency.tab', 'JPLUS_J0660_FullEfficiency.tab', 'JPLUS_J0861_FullEfficiency.tab']

files1 = ['JPLUS_uJava_FullEfficiency.tab', 'JPLUS_gSDSS_FullEfficiency.tab',  'JPLUS_rSDSS_FullEfficiency.tab', 'JPLUS_iSDSS_FullEfficiency.tab', 'JPLUS_zSDSS_FullEfficiency.tab']

   
# files = ['F378_transm.txt', 'F395_transm.txt', 'F410_transm.txt', 'F430_transm.txt','F515_transm.txt', 'F660_transm.txt',  
#                 'F861_transm.txt']

# files1 = [ 'uJAVA_transm.txt', 'gSDSS_transm.txt',  'F625_transm.txt',
#           'iSDSS_transm.txt', 'zSDSS_transm.txt']


colors = [ "#9900FF", "#6600FF", "#0000FF", "#009999", "#DD8000", "#CC0066", "#660033"]
colors1 = [ "#CC00FF", "#006600", "#FF0000", "#990033", "#330034"] 
for f, color, filter_ in zip(files, colors, filters):
    x, y = load_file(f)
    plt.fill(x,y, color=color, label=filter_)

for f, color, filter_ in zip(files1, colors1, filters1):
    x, y = load_file(f)
    plt.plot(x,y,  color=color, linewidth=3.8, label=filter_)

def sys(spectra):
    #datadir = "../../Halo-PNe-spectros/"
    datadir = "../../PN-Tesis-Didactica/"
    file_ = spectra
    x = np.loadtxt(os.path.join(datadir, file_), delimiter = None, skiprows = 0, usecols = None,
                   unpack = False, dtype = np.dtype([('Wl', '|f8'), ('Flux', 'f8')]))
    return x['Wl'], x['Flux']

#x, y = sys("mwc574.dat")
# m = x >=3200
# x = x[m]
x, y = sys("PN-M157-medium.dat")
y /=80.0e-15
    
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
#plt.ylim(0.0, 0.8)
#plt.xlim(2500, 10500)
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18)
plt.plot(x, y, linewidth=1.3, color="k")
#plt.ylim(0.0, 1.0)    
plt.legend(fontsize='x-small')
plt.xlabel("Wavelength($\AA$)", fontsize= 18)
plt.ylabel("Transmission", fontsize= 18)
plt.tight_layout()

plt.savefig('jplus-filter-2019.pdf')
