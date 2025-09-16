'''
Principal component analysis (J-PLUS)
'''
#from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns
import os.path
from collections import OrderedDict
from scipy.stats import gaussian_kde
import pandas as pd

label=[]
label_dr1=[]
X = []
target = []

pattern =  "*-proof-spectros/*-JPLUS17-magnitude.json"

def clean_nan_inf(M):
    mask_nan = np.sum(np.isnan(M), 1) > 0
    mask_inf = np.sum(np.isinf(M), 1) > 0
    lines_to_discard = np.logical_xor(mask_nan,  mask_inf)
    print("Number of lines to discard:", sum(lines_to_discard))
    M = M[np.logical_not(lines_to_discard), :]
    return M

    
file_list = glob.glob(pattern)

shape = (len(file_list), 12)
print(len(file_list))

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        data = OrderedDict((k, v) for k, v in sorted(data.items(), key=lambda x: x[0]))
    X.append(data["F348"])
    X.append(data["F378"])
    X.append(data["F395"])
    X.append(data["F410"])
    X.append(data["F430"])
    X.append(data["F480_g_sdss"])
    X.append(data["F515"])
    X.append(data["F625_r_sdss"])
    X.append(data["F660"])
    X.append(data["F766_i_sdss"])
    X.append(data["F861"])
    X.append(data["F911_z_sdss"])
        
    
XX = np.array(X).reshape(shape)
print("Data shape:", XX.shape)
m = []

XX = clean_nan_inf(XX)

#Create target to classify the kind of object

#XX = np.array(XX[np.logical_not(np.isnan(XX), np.isinf(XX))])
#target_ = np.array(target_[np.logical_not(np.isnan(target_), np.isinf(target_))])


print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#create the PCA for S-PLUS photometric system
XX = StandardScaler().fit_transform(XX.data)

pca = PCA(n_components=6)
pca.fit(XX)

XX_pca = pca.transform(XX)

#porcentages
print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_[0]) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

#Lista with the objects
#n=10
pc1, pc2, pc3 = [],  [], []

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

pc1.append(XX_pca[:,0])
pc2.append(XX_pca[:,1])
pc3.append(XX_pca[:,2])

print(pc1)
#waights

w1 = pca.components_[0]
w2 = pca.components_[1]
w3 = pca.components_[2]

print('W:',  pca.components_)
print('Ein:',  pca.explained_variance_)

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(-11, 9.0)
# ax1.set_ylim(-3.4, 3.2)
ax1.set_xlim(-27.0, 23.0)
ax1.set_ylim(-7.5, 8.0)
#ax1.set_ylim(-0.5,1.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC2', fontsize= 35)

ax1.scatter(pc1, pc2,  color= sns.xkcd_rgb["cerulean"], s=130, marker='*', alpha=0.8, edgecolor='black', zorder=100.0, label='Obs. J-PLUS objects')

ax1.minorticks_on()

# for label_, x, y in zip(label_dr1, pc1[8], pc2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(pc1[8], dtype=str), np.array(pc2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
for x,y in zip(pc1, pc2):
    ax1.annotate("PN Sp 4-1", (x[0], y[0]), alpha=5,  size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    ax1.annotate("LEDA 101538", (x[1], y[1]), alpha=5, size="x-large",
                   xytext=(105.0, -13.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    ax1.annotate("J-PLUS HII region", (x[2], y[2]), alpha=5, size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103, weight='bold',)
    ax1.annotate("[HLG90] 55", (x[3], y[3]), alpha=5, size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    print(x[2])
#ax2.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='lower center', **lgd_kws)
ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig1-JPLUS-PC1-PC2-v0-onlyjobj.pdf'
save_path = '../../Dropbox/JPAS/paper-phot/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
# ax2.set_xlim(-10.0, 8.0)
# ax2.set_ylim(-2.0, 1.5)
ax2.set_xlim(-27.0, 23.0)
ax2.set_ylim(-6.0, 4.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)

ax2.scatter(pc1, pc3,  color= sns.xkcd_rgb["cerulean"], s=130, marker='*', edgecolor='black', alpha=0.8, zorder=100.0, label='Obs. J-PLUS objects')

ax2.minorticks_on()

for x,y in zip(pc1, pc3):
    ax2.annotate("PN Sp 4-1", (x[0], y[0]), alpha=5, size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    ax2.annotate("LEDA 101538", (x[1], y[1]), alpha=5, size="x-large",
                   xytext=(100.0, 10.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    ax2.annotate("J-PLUS HII region", (x[2], y[2]), alpha=5, size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103, weight='bold',)
    ax2.annotate("[HLG90] 55", (x[3], y[3]), alpha=5, size="x-large",
                   xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)

# for label_, x, y in zip(np.array(label), pc1[1], pc2[1]):
#     ax1.annotate(label_, (x, y), size=10.5, 
#                    xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
    
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'Fig2-JPLUS-PC1-PC3-v0-onlyjobj.pdf'
save_path = '../../Dropbox/JPAS/paper-phot/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()
