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
        wl = str(i[0])
        res = str(i[1])
        wll.append(wl)
        ress.append(res)
    return wll, ress

pattern = "Jv*.res"
file_list = glob.glob(pattern)

pattern1 = "*JPAS_0915.res"
file_list1 = glob.glob(pattern1)

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(111)
for f in file_list:
    x, y = load_file(f)
    ax1.plot(x,y, c="gray", label=f.split('_t')[0])

for f in file_list1:
    x, y = load_file(f)
    ax1.plot(x,y, linewidth=2.3, label=f.split('_t')[0])

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")
plt.xlim(3e3, 10500)
plt.ylim(0.0, 0.70)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)     
#plt.legend(fontsize='x-small')
plt.xlabel("Wavelength($\AA$)", fontsize= 18)
plt.ylabel("Transmission", fontsize= 18)

plt.savefig('luis-filter.pdf')

