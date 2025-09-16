'''
Make color-color diagram to particular filters
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

pattern = "*-spectros/*-alh-magnitude.json"

file_list = glob.glob(pattern)

d_644, d_768 = [], []
d_644_c, d_768_c = [], []
d_644_E0, d_768_E0 = [], []
d_644_E01, d_768_E01 = [], []
d_644_E02, d_768_E02 = [], []

label = []

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith('catB'):
            F613_c = data['F_613_3']
            F644_c = data['F_644_3']
            F768_c = data['F_768_3']
            diff_644_c = F613_c - F644_c
            diff_768_c = F613_c - F768_c
            d_644_c.append(diff_644_c)
            d_768_c.append(diff_768_c)

        elif data['id'].startswith('DdDm1_L2'):
            F613_E01 = data['F_613_3']
            F644_E01 = data['F_644_3']
            F768_E01 = data['F_768_3']
            diff_644_E01 = F613_E01 - F644_E01
            diff_768_E01 = F613_E01 - F768_E01
            d_644_E01.append(diff_644_E01)
            d_768_E01.append(diff_768_E01)
                
        else:
            F613 = data['F_613_3']
            F644 = data['F_644_3']
            F768 = data['F_768_3']
            diff_644 = F613 - F644
            diff_768 = F613 - F768
            d_644.append(diff_644)
            d_768.append(diff_768)
            if data['id'].endswith('SLOAN'):
                label.append("H41") 
            else:
                label.append(data['id'])
                
sns.set(style="dark", context="talk")                 

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-0.5,xmax=2)
#ax1.set_ylim(ymin=0.40,ymax=2.60)
plt.xlabel('F613W - F768W', size = 12)
plt.ylabel('F613W - F644W', size = 12)
ax1.scatter(d_768, d_644, c='black', alpha=0.6, s=35, label='Halo PNe')
ax1.scatter(d_768_c, d_644_c, c='red', alpha=0.6, label='CVV')
ax1.scatter(d_768_E01, d_644_E01,  c= "purple", alpha=0.6, marker='s', label='L2 dddm1')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(-3,3), textcoords='offset points', ha='left', va='bottom',)
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, ncol=2, fontsize='x-small', **lgd_kws)

ax1.grid()
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('cor-alhambra.pdf')
