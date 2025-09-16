'''
Make color-color diagram to particular filters
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt

pattern = "*-spectros/*-alh-magnitude.json"

file_list = glob.glob(pattern)

nsource = len(file_list)

d_pn = {'d_644':{'id': np.empty((nsource,))}, 'd_768':{'id': np.empty((nsource,))}}
#d_CV = {'d_644': np.empty((nsource,)), 'd_768': np.empty((nsource,))}

objects = {}

for  isource, file_name in enumerate(file_list):
    with open(file_name) as f:
        data = json.load(f)
        F613 = data['F_613_3']
        F644 = data['F_644_3']
        F768 = data['F_768_3']
        diff_644 = F613 - F644
        diff_768 = F613 - F768
        d_pn['d_644'][isource] = diff_644
        d_pn['d_768'][isource] = diff_768
        label_ = data['id']
        if data['id'].endswith('catB'):
            objects[label_] = 1
        else:
            objects[label_] = 0

print(d_pn)
#mask
#mask_pn = np.array([objects[source] == 0 for source in data['id']])
#mask_cv = np.array([objects[source] == 1 for source in data['id']]) 
            
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlim(xmin=-0.5,xmax=2)
#ax1.set_ylim(ymin=0.40,ymax=2.60)
plt.xlabel('F613w - F768w', size = 12)
plt.ylabel('F613w - F644w', size = 12)
#ax1.scatter(d_pn['d_768'][mask_pn], d_pn['d_644'][mask_pn], c='black', alpha=0.6, s=35, label='Halo PNe')
#ax1.scatter(d_pn['d_768'][mask_cv], d_pn['d_644'][mask_cv], c='red', alpha=0.6, label='CVV')
#for label_, x, y in zip(label, d_pn['d_768'], d_pn['d_644']):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(-3,3), textcoords='offset points', ha='left', va='bottom',)
ax1.legend()
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
plt.savefig('cor-alhambra2.pdf')

