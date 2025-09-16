'''
Reproduces the trasmission curve
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_file(filename):
    wll, ress = [], []
    data = np.loadtxt(filename, delimiter=None, converters=None, skiprows=0,
                                       usecols=None, unpack=False, ndmin=0)
    for i in data:
        wl = str(i[0]*10)
        res = str(i[1]*100)
        wll.append(wl)
        ress.append(res)
    return wll, ress

filters= [ 'J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
filters1= ['uJAVA',   'gSDSS',  'rSDSS',  'iSDSS']

files = [ 'F378_with_ccd_qe.dat', 'F395_with_ccd_qe.dat', 
'F410_with_ccd_qe.dat', 'F430_with_ccd_qe.dat', 
'F515_with_ccd_qe.dat', 'F660_with_ccd_qe.dat',  'F861_with_ccd_qe.dat']

files1 = ['F348_with_ccd_qe.dat', 'g_sdss_with_ccd_qe.dat',  'r_sdss_with_ccd_qe.dat', 
                             'i_sdss_with_ccd_qe.dat']
   
# files = ['F378_transm.txt', 'F395_transm.txt', 'F410_transm.txt', 'F430_transm.txt','F515_transm.txt', 'F660_transm.txt',  
#                 'F861_transm.txt']

# files1 = [ 'uJAVA_transm.txt', 'gSDSS_transm.txt',  'F625_transm.txt',
#           'iSDSS_transm.txt', 'zSDSS_transm.txt']


colors = [ 'green',  'red',  'purple', 'gray', 'sage', 'salmon', 'goldenrod' ]
colors1 = [ 'cyan', 'olive', 'teal', 'magenta'] 
for f, color, filter_ in zip(files, colors, filters):
    x, y = load_file(f)
    plt.fill(x,y, color=color, label=filter_)

for f, color, filter_ in zip(files1, colors1, filters1):
    x, y = load_file(f)
    plt.plot(x,y,  color=color, linewidth=3.8, label=filter_)

wll, ress = [], []
data = np.loadtxt('z_sdss_with_ccd_qe.dat', delimiter=None, converters=None, skiprows=0,
                                       usecols=None, unpack=False, ndmin=0)
for i in data:
    wl = str(i[0]*10)
    res = str(i[1])
    wll.append(wl)
    ress.append(res)
plt.plot(wll, ress,  color='brown', linewidth=3.8, label='zSDSS')
    
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
plt.ylim(0.0, 100.0)
plt.xlim(2500, 10800)
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 
#plt.ylim(0.0, 1.0)    
plt.legend(fontsize='x-small')
plt.xlabel("Wavelength($\AA$)", fontsize= 16)
plt.ylabel("Transmission", fontsize= 16)

plt.savefig('jplus-filter.pdf')
