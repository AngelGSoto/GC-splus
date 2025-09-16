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
from matplotlib.ticker import FormatStrFormatter

label=[]
label_dr1=[]
X = []
target = []

pattern =  "*-spectros/*-JPLUS17-magnitude.json"

def clean_nan_inf(M):
    mask_nan = np.sum(np.isnan(M), 1) > 0
    mask_inf = np.sum(np.isinf(M), 1) > 0
    lines_to_discard = np.logical_xor(mask_nan,  mask_inf)
    print("Number of lines to discard:", sum(lines_to_discard))
    M = M[np.logical_not(lines_to_discard), :]
    return M

    
file_list = glob.glob(pattern)

shape = (len(file_list), 12)
shape1 = (len(file_list), 13)
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
    target.append(data["F348"])
    target.append(data["F378"])
    target.append(data["F395"])
    target.append(data["F410"])
    target.append(data["F430"])
    target.append(data["F480_g_sdss"])
    target.append(data["F515"])
    target.append(data["F625_r_sdss"])
    target.append(data["F660"])
    target.append(data["F766_i_sdss"])
    target.append(data["F861"])
    target.append(data["F911_z_sdss"])
    if data["id"].endswith("E00_300"):
        target.append(0)
    elif data["id"].endswith("E01_300"):
        target.append(0)
    elif data["id"].endswith("E02_300"):
        target.append(0)
    elif data["id"].endswith("E00_600"):
        target.append(0)
    elif data["id"].endswith("E01_600"):
        target.append(0)
    elif data["id"].endswith("E02_600"):
        target.append(0)
    elif data["id"].endswith("E00_100"):
        target.append(0)
    elif data["id"].endswith("E01_100"):
        target.append(0)
    elif data["id"].endswith("E02_100"):
        target.append(0)
    elif data["id"].endswith("HPNe"):
        target.append(1)
    elif data["id"].endswith("HPNe-"):
        target.append(1)
    # elif data["id"].endswith("DPNe"):
    #     target.append(2)
    elif data["id"].endswith("sys"):
        target.append(2)
    elif data["id"].endswith("extr-SySt"):
        target.append(2)
    elif data["id"].endswith("ngc185"):
        target.append(2)
    elif data["id"].endswith("SySt-ic10"):
        target.append(2)
    elif data["id"].endswith("sys-IPHAS"):
        target.append(2)
    #elif data["id"].endswith("sys-Ext"):
        #target.append(5)
    #elif data["id"].endswith("survey"):
        #target.append(6)
    elif data["id"].endswith("CV"):
        target.append(3)
    elif data["id"].endswith("ExtHII"):
        target.append(4)
    # elif data["id"].endswith("SNR"):
    #     target.append(9)
    elif data["id"].endswith("QSOs-13"):
        target.append(5)
    elif data["id"].endswith("QSOs-24"):
        target.append(5)
    elif data["id"].endswith("QSOs-32"):
        target.append(5)
    elif data["id"].endswith("YSOs"):
        target.append(6)
    # elif data["id"].endswith("DR1jplus"):
    #     target.append(8)
    # elif data["id"].endswith("DR1jplusHash"):
    #     target.append(9)
    elif data["id"].endswith("DR1SplusWDs"):
        target.append(7)
    else:
        target.append(8)
    #label
    # if data['id'].endswith("-1-HPNe-"):
    #     label.append("DdDm-1")
    # if data['id'].endswith("H41-HPNe-"):
    #     label.append("H4-1")
    # if data['id'].endswith("1359559-HPNe-"):
    #     label.append("PNG 135.9+55.9")
    # if data['id'].startswith("ngc"):
    #     label.append("NGC 2242")
    # elif data['id'].startswith("mwc"):
    #     label.append("MWC 574")
    #label objts select
    # if data['id'].endswith("12636-DR1jplus"):
    #     label_dr1.append("J-PLUS object")
    # if data['id'].endswith("4019-DR1jplus"):
    #     label_dr1.append("LEDA 2790884")
    # if data['id'].endswith("12636-DR1jplus"):
    #     label_dr1.append("LEDA 101538")
    # elif data['id'].startswith("18242-DR1jplus"):
    #     label_dr1.append("PN Sp 4-1")
            
print(label)            
XX = np.array(X).reshape(shape)
target_ = np.array(target).reshape(shape1)
print("Data shape:", XX.shape)

m = []

XX = clean_nan_inf(XX)

#Create target to classify the kind of object
target_ = clean_nan_inf(target_)

#XX = np.array(XX[np.logical_not(np.isnan(XX), np.isinf(XX))])
#target_ = np.array(target_[np.logical_not(np.isnan(target_), np.isinf(target_))])

for i in target_:
    m.append(i[12])

m = np.array(m)
print(m.shape)
print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#create the PCA for S-PLUS photometric system
XX1 = StandardScaler().fit_transform(XX.data)

pca = PCA(n_components=6)
pca.fit(XX1)

XX_pca = pca.transform(XX1)

#porcentages
print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_[0]) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

#Lista with the objects
n=11
pc1, pc2, pc3 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

pc1[0].append(XX_pca[m == 0, 0])
pc2[0].append(XX_pca[m == 0, 1])
pc3[0].append(XX_pca[m == 0, 2])
pc1[1].append(XX_pca[m == 1, 0])
pc2[1].append(XX_pca[m == 1, 1])
pc3[1].append(XX_pca[m == 1, 2])
pc1[2].append(XX_pca[m == 2, 0])
pc2[2].append(XX_pca[m == 2, 1])
pc3[2].append(XX_pca[m == 2, 2])
pc1[3].append(XX_pca[m == 3, 0])
pc2[3].append(XX_pca[m == 3, 1])
pc3[3].append(XX_pca[m == 3, 2])
pc1[4].append(XX_pca[m == 4, 0])
pc2[4].append(XX_pca[m == 4, 1])
pc3[4].append(XX_pca[m == 4, 2])
pc1[5].append(XX_pca[m == 5, 0])
pc2[5].append(XX_pca[m == 5, 1])
pc3[5].append(XX_pca[m == 5, 2])
pc1[6].append(XX_pca[m == 6, 0])
pc2[6].append(XX_pca[m == 6, 1])
pc3[6].append(XX_pca[m == 6, 2])
pc1[7].append(XX_pca[m == 7, 0])
pc2[7].append(XX_pca[m == 7, 1])
pc3[7].append(XX_pca[m == 7, 2])
pc1[8].append(XX_pca[m == 8, 0])
pc2[8].append(XX_pca[m == 8, 1])
pc3[8].append(XX_pca[m == 8, 2])
# pc1[9].append(XX_pca[m == 9, 0])
# pc2[9].append(XX_pca[m == 9, 1])
# pc3[9].append(XX_pca[m == 9, 2])
# pc1[10].append(XX_pca[m == 10, 0])
# pc2[10].append(XX_pca[m == 10, 1])
# pc3[10].append(XX_pca[m == 10, 2])

#waights

w1 = pca.components_[0]
w2 = pca.components_[1]
w3 = pca.components_[2]

print('W:',  pca.components_)
print('Ein:',  pca.explained_variance_)

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
ax1.set_xlim(-8.2, 5.7)
ax1.set_ylim(-2.5, 1.5)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.set_xlim(-10, 10)
# ax1.set_ylim(-3.3, 3.2)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC2', fontsize= 35)

#print(A1[0][1], B1[0][1])        
AB = np.vstack([pc1[0], pc2[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': pc1[0], 'y': pc2[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
for x1, y1 in zip(pc1[0], pc2[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax1.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax1.scatter(pc1[0], pc2[0], c= sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', zorder=10.0, edgecolor='black', label='CLOUDY modelled halo PNe')
ax1.scatter(pc1[1], pc2[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='Obs. hPNe')
ax1.scatter(pc1[3], pc2[3], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax1.scatter(pc1[5], pc2[5],  c= "mediumaquamarine", alpha=0.8, s=90, edgecolor='black',  marker='D', label='SDSS QSOs')
ax1.scatter(pc1[8], pc2[8],  c= "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax1.scatter(pc1[2], pc2[2],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
#ax1.scatter(pc1[3], pc2[3],  c= "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax1.scatter(pc1[4], pc2[4],  c= "gray", alpha=0.8, s=90, marker='D', edgecolor='black', zorder=133.0,  label='Obs. HII regions in NGC 55')
ax1.scatter(pc1[6], pc2[6],  c= "lightsalmon", alpha=0.8, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
#ax1.scatter(pc1[8], pc2[8],  c= sns.xkcd_rgb["cerulean"],  s=700, marker='*', edgecolor='black', zorder= 125, label='Sources selected in J-PLUS DR1')
#ax1.scatter(pc1[9], pc2[9],  c= sns.xkcd_rgb["light green"], s=700, marker='*', edgecolor='black', zorder= 125, label='HASH sources in J/S-PLUS DR1')
ax1.scatter(pc1[7], pc2[7], c= sns.xkcd_rgb["mint green"], alpha=0.3, s=150, marker='*', edgecolor='black', zorder= 125, label='S-PLUS DR1 WDs')

# left, bottom, width, height = [0.55, 0.73, 0.28, 0.23]
# ax11 = fig.add_axes([left, bottom, width, height])
# ax11.set_xlim(-8.2, 5.7)
# ax11.set_ylim(-2.5, 2.5)
# for x11, y11 in zip(pc1[0], pc2[0]):
#     x111, y111, z = x11[idx], y11[idx], z[idx]
#     ax11.scatter(x111, y111, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
# ax11.scatter(pc1[2], pc2[2],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0)
# ax11.scatter(pc1[7], pc2[7],  c=sns.xkcd_rgb['mint green'], s=150, cmap=plt.cm.hot, marker='*', zorder=110)

# for label_, x, y in zip(label_dr1, pc1[8], pc2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(pc1[8], dtype=str), np.array(pc2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
# print(pc1[8], pc2[8])
# bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.7, pad=0.1)
# for x,y in zip(pc1[8], pc2[8]):
#     ax1.annotate("2", (x[0], y[0]), alpha=9,  size=21, #PN
#                    xytext=(-8.0, -3.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax1.annotate("3", (x[1], y[1]), alpha=9, size=21, #LEDA 101538
#                    xytext=(15.0, -32.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax1.annotate("1", (x[2], y[2]), alpha=9, size=21, #candidate
#                    xytext=(0.0, -32.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=180, weight='bold',)
#     ax1.annotate("4", (x[3], y[3]), alpha=9, size=21, #2790884
#                    xytext=(-8.0, -3.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#print(x[2])
#ax2.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=19.8, loc='lower center', **lgd_kws)
ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
pltfile = 'Fig1-JPLUS-PC1-PC2-WD.pdf'
save_path = '../../Dropbox/JPAS/Tesis/Fig'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
# ax2.set_xlim(-10.0, 8.0)
# ax2.set_ylim(-2.0, 1.5)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.set_xlim(-8.2, 5.7)
ax2.set_ylim(-1.1, 1.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)
#print(A1[0][1], B1[0][1])        
AB = np.vstack([pc1[0], pc3[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': pc1[0], 'y': pc3[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()

for x1, y1 in zip(pc1[0], pc3[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax2.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax2.scatter(pc1[0], pc3[0], c = sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', edgecolor='black', zorder = 10.0, label='CLOUDY modelled halo PNe')
ax2.scatter(pc1[1], pc3[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', edgecolor='black', alpha=0.8, zorder=120.0, label='Obs. halo PNe')
ax2.scatter(pc1[3], pc3[3],  c = sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black',zorder=3.0, label='SDSS CVs')
ax2.scatter(pc1[5], pc3[5],  c = "mediumaquamarine", alpha=0.8, s=60,  edgecolor='black', marker='D', label='SDSS QSOs')
ax2.scatter(pc1[8], pc3[8],  c = "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax2.scatter(pc1[2], pc3[2],  c = "red", alpha=0.8, s=90, marker='s',edgecolor='black', zorder=3.0, label='Obs SySt')
#ax2.scatter(pc1[3], pc3[3],  c = "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax2.scatter(pc1[4], pc3[4],  c = "gray", alpha=0.8, s=90, marker='D', edgecolor='black', zorder=133.0, label='Obs. HII regions in NGC 55')
ax2.scatter(pc1[6], pc3[6],  c = "lightsalmon", alpha=0.8, s=150, marker='*', edgecolor='black', zorder = 11, label='Obs. YSOs')
#ax2.scatter(pc1[8], pc3[8],  c = sns.xkcd_rgb["cerulean"], s=700, marker='*', edgecolor='black', zorder= 125, label='Obs. select in J-PLUS DR1')
#ax2.scatter(pc1[9], pc3[9],  c= sns.xkcd_rgb["light green"],  s=700, marker='*', edgecolor='black', zorder= 125, label='HASH sources in J-PLUS DR1')
ax2.scatter(pc1[7], pc3[7],  c= sns.xkcd_rgb["mint green"], alpha=0.3, s=150, marker='*', edgecolor='black', zorder= 125, label='DR1 S-PLUS WDs')
ax2.minorticks_on()

# for x,y in zip(pc1[8], pc3[8]):
#     ax2.annotate("2", (x[0], y[0]), alpha=9, size=21, #PN
#                    xytext=(15, -33.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax2.annotate("3", (x[1], y[1]), alpha=9, size=21,
#                    xytext=(15, -33.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax2.annotate("1", (x[2], y[2]), alpha=9, size=21,
#                    xytext=(-3.0, 33.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=180, weight='bold',)
#     ax2.annotate("4", (x[3], y[3]), alpha=9, size=21,
#                    xytext=(5.0, -45.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)

# for label_, x, y in zip(np.array(label), pc1[1], pc2[1]):
#     ax1.annotate(label_, (x, y), size=10.5, 
#                    xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
    
#ax2.grid(which='minor')#, lw=0.3)
ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#pltfile = 'Fig2-JPLUS-PC1-PC3-veri.pdf'
pltfile = 'Fig2-JPLUS-PC1-PC3-WD.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax22 = fig.add_subplot(111)
ax22.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax22.set_xlim(-9, 10)
ax22.set_ylim(-1.2, 1.5)
# ax2.set_xlim(-27.0, 23.0)
# ax2.set_ylim(-7.5, 7.5)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC2', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)
#print(A1[0][1], B1[0][1])        
AB = np.vstack([pc1[0], pc3[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': pc1[0], 'y': pc3[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()

for x1, y1 in zip(pc2[0], pc3[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax22.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax2.scatter(pc1[0], pc3[0], c = sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', edgecolor='black', zorder = 10.0, label='CLOUDY modelled halo PNe')
ax22.scatter(pc2[1], pc3[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', edgecolor='black', alpha=0.8, zorder=120.0, label='Obs. halo PNe')
ax22.scatter(pc2[4], pc3[4],  c = sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax22.scatter(pc2[6], pc3[6],  c = "mediumaquamarine", alpha=0.8, s=60, edgecolor='black',  marker='D', label='SDSS QSOs')
ax22.scatter(pc2[10], pc3[10],  c = "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax22.scatter(pc2[2], pc3[2],  c = "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
ax22.scatter(pc2[3], pc3[3],  c = "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax22.scatter(pc2[5], pc3[5],  c = "gray", alpha=0.8, s=90, marker='D', edgecolor='black',  label='Obs. HII regions in NGC 55')
ax22.scatter(pc2[7], pc3[7],  c = "lightsalmon", alpha=0.8, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
ax22.scatter(pc2[8], pc3[8],  c = sns.xkcd_rgb["cerulean"], alpha=1.0, s=700, marker='*', edgecolor='black', zorder= 113, label='Obs. select in J-PLUS DR1')
ax22.scatter(pc2[9], pc3[9],  c= sns.xkcd_rgb["light green"], alpha=1.0, s=650, marker='*', edgecolor='black', zorder= 113, label='HASH sources in J-PLUS DR1')

ax22.minorticks_on()

# for x,y in zip(pc1[8], pc3[8]):
#     ax2.annotate("PN Sp 4-1", (x[0], y[0]), alpha=9, size=21,
#                    xytext=(78.0, 15.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax2.annotate("LEDA 101538", (x[1], y[1]), alpha=9, size=21,
#                    xytext=(120.0, -35.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)
#     ax2.annotate("J-PLUS HII region", (x[2], y[2]), alpha=9, size=21,
#                    xytext=(-5.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=180, weight='bold',)
#     ax2.annotate("[HLG90] 55", (x[3], y[3]), alpha=9, size=21,
#                    xytext=(5.0, -35.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=150,)

# for label_, x, y in zip(np.array(label), pc1[1], pc2[1]):
#     ax1.annotate(label_, (x, y), size=10.5, 
#                    xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
    
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax22.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'Fig3-JPLUS-PC2-PC3-v0.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#waigt ########################################################
####################################################################

filter_name = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
color= ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
#color= ["#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066"]
marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"]

filter_ = []
for a in range(1, 13):
    filter_.append(a)

plotfile = "jplus-wight1.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_xlim(-10.0, 8.0)
#ax.set_ylim(-0.30, -0.28)
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'$w_1$', size = 45)
#ax.axhline(y=0, c='k')
for wl, mag, colors, marker_ in zip(filter_, w1, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400,  alpha=0.8,  zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
#plt.xticks(filter_)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
##############################################################################
plotfile = "jplus-wight2.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'$w_2$', size = 45)
ax.axhline(y=0, color='k')
for wl, mag, colors, marker_ in zip(filter_, w2, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400, alpha=0.8, zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
#############################################################################
plotfile = "jplus-wight3.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylim(-0.7, 0.6)
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'$w_3$', size = 45)
ax.axhline(y=0, c='k')
for wl, mag, colors, marker_ in zip(filter_, w3, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400, alpha=0.8, zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
