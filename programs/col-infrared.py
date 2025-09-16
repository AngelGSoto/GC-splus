'''
Make color-color diagram to infrered
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

dHK, dJH = [], []
dHK_, dJH_ = [], []

can_alh = []

f = open('Alhambra-candidates/list-candidates.txt', 'r')
header1 = f.readline()
header2 = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    mag1 = float(columns[47])
    mag2 = float(columns[49])
    mag3 = float(columns[51])
    d_mag1 = mag2 - mag3
    d_mag2 = mag1 - mag2
    print(columns[0].split('0')[-1], mag1, mag2, mag3)
    dHK.append(d_mag1)
    dJH.append(d_mag2)
    can_alh.append(columns[0].split('0')[-1])

dt = np.dtype([ ('J', 'f4'), ('H', 'f4'), ('K', 'f4')])
x = np.loadtxt('Halo-PNe-spectros/list-HPNe.dat', delimiter = None, converters = None, skiprows = 0, 
                         usecols = None, unpack = False, ndmin = 0, dtype = dt)

mag1 = x['J']
mag2 = x['H']
mag3 = x['K']
d_mag1 = mag2 - mag3
d_mag2 = mag1 - mag2
dHK_.append(d_mag1)
dJH_.append(d_mag2)

#print(columns[0])
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-1.5,xmax=2.0)
ax1.set_ylim(ymin=-1.5,ymax=1.5)
plt.xlabel('H - Ks', size = 12)
plt.ylabel('J - H', size = 12)
ax1.scatter(dHK, dJH, c='green', alpha=0.8, marker ='D',  s=35, label='ALHAMBRA candidates')
ax1.scatter(dHK_, dJH_, c='blue', alpha=0.8, marker ='D',  s=35, label='Known HPNe')
for label_, x, y in zip(can_alh, dHK, dJH):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(3,-8), textcoords='offset points', ha='left', va='bottom',)
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, **lgd_kws)
ax1.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('col-infrared.pdf')
