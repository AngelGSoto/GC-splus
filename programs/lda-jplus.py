'''
linear discrimant analysis (J-PLUS)
'''
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
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
        target.append(0)
    elif data["id"].endswith("HPNe-"):
        target.append(0)
    # elif data["id"].endswith("DPNe"):
    #     target.append(2)
    elif data["id"].endswith("sys"): #2
        target.append(1)
    elif data["id"].endswith("extr-SySt"):#2
        target.append(1)
    elif data["id"].endswith("ngc185"):#2
        target.append(1)
    elif data["id"].endswith("SySt-ic10"): #2
        target.append(1)
    elif data["id"].endswith("sys-IPHAS"):#3
        target.append(1)
    #elif data["id"].endswith("sys-Ext"):
        #target.append(5)
    #elif data["id"].endswith("survey"):
        #target.append(6)
    elif data["id"].endswith("CV"): #4
        target.append(2)
    elif data["id"].endswith("ExtHII"): #5
        target.append(3)
    # elif data["id"].endswith("SNR"):
    #     target.append(9)
    elif data["id"].endswith("QSOs-13"):#6
        target.append(4)
    elif data["id"].endswith("QSOs-24"):#6
        target.append(4)
    elif data["id"].endswith("QSOs-32"): #6
        target.append(4)
    elif data["id"].endswith("YSOs"): #7
        target.append(5)
    elif data["id"].endswith("DR1SplusWDs"): #8
        target.append(6)
    # elif data["id"].endswith("DR1jplus"): #9
    #     target.append(7)
    # elif data["id"].endswith("DR1jplusHash"): #10
    #     target.append(8)
    else:
        target.append(7) #11
        
    #label
    if data['id'].endswith("-1-HPNe-"):
        label.append("DdDm-1")
    if data['id'].endswith("H41-HPNe-"):
        label.append("H4-1")
    if data['id'].endswith("1359559-HPNe-"):
        label.append("PNG 135.9+55.9")
    if data['id'].startswith("ngc"):
        label.append("NGC 2242")
    elif data['id'].startswith("mwc"):
        label.append("MWC 574")
    #label objts select
    # if data['id'].endswith("12636-DR1jplus"):
    #     label_dr1.append("J-PLUS object")
    # if data['id'].endswith("17767-DR1jplus"):
    #     label_dr1.append("[HLG90] 55")
    # if data['id'].endswith("12636-DR1jplus"):
    #     label_dr1.append("LEDA 101538")
    # elif data['id'].startswith("18242-DR1jplus"):
    #     label_dr1.append("PN Sp 4-1")

#print(X.shape)
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
print(len(target))
print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#create the PCA for S-PLUS photometric system
#XX1 = StandardScaler().fit_transform(XX, m)
    
#create the LDA for J-PLUS photometric system
lda = LDA(n_components=6)
lda.fit(XX, m)

XX_lda = lda.transform(XX)

##############################################################################
X_new = []
File = "*-nolda/*-JPLUS17-magnitude.json"
file_list1 = glob.glob(File)
for file_name1 in file_list1:
    with open(file_name1) as f1:
        da = json.load(f1)
        da = OrderedDict((k, v) for k, v in sorted(da.items(), key=lambda x: x[0]))
    X_new.append(da["F348"])
    X_new.append(da["F378"])
    X_new.append(da["F395"])
    X_new.append(da["F410"])
    X_new.append(da["F430"])
    X_new.append(da["F480_g_sdss"])
    X_new.append(da["F515"])
    X_new.append(da["F625_r_sdss"])
    X_new.append(da["F660"])
    X_new.append(da["F766_i_sdss"])
    X_new.append(da["F861"])
    X_new.append(da["F911_z_sdss"])
shape_new = (len(file_list1), 12)
XX_new = np.array(X_new).reshape(shape_new)
#XX1_new = StandardScaler().fit_transform(XX_new)
y_pred = lda.predict(XX_new)
print(y_pred)
###############################################################################
#porcentages
#print("Porcentage:", lda.explained_variance_ratio_)
#print("Singular Value:", lda.singular_values_)
# print("Component:", lda.components_[0]) # eigevectors
# print("Sorted components:", lda.explained_variance_) # eigenvalues

#Lista with the objects
n=7   #11
ld1, ld2, ld3 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

ld1[0].append(XX_lda[m == 0, 0])
ld2[0].append(XX_lda[m == 0, 1])
ld3[0].append(XX_lda[m == 0, 2])
ld1[1].append(XX_lda[m == 1, 0])
ld2[1].append(XX_lda[m == 1, 1])
ld3[1].append(XX_lda[m == 1, 2])
ld1[2].append(XX_lda[m == 2, 0])
ld2[2].append(XX_lda[m == 2, 1])
ld3[2].append(XX_lda[m == 2, 2])
ld1[3].append(XX_lda[m == 3, 0])
ld2[3].append(XX_lda[m == 3, 1])
ld3[3].append(XX_lda[m == 3, 2])
ld1[4].append(XX_lda[m == 4, 0])
ld2[4].append(XX_lda[m == 4, 1])
ld3[4].append(XX_lda[m == 4, 2])
ld1[5].append(XX_lda[m == 5, 0])
ld2[5].append(XX_lda[m == 5, 1])
ld3[5].append(XX_lda[m == 5, 2])
ld1[6].append(XX_lda[m == 6, 0])
ld2[6].append(XX_lda[m == 6, 1])
ld3[6].append(XX_lda[m == 6, 2])
ld1[7].append(XX_lda[m == 7, 0])
ld2[7].append(XX_lda[m == 7, 1])
ld3[7].append(XX_lda[m == 7, 2])
# ld1[8].append(XX_lda[m == 8, 0])
# ld2[8].append(XX_lda[m == 8, 1])
# ld3[8].append(XX_lda[m == 8, 2])
# ld1[9].append(XX_lda[m == 9, 0])
# ld2[9].append(XX_lda[m == 9, 1])
# ld3[9].append(XX_lda[m == 9, 2])

#waights


#print('W:',  lda.components_)
# print('Ein:',  lda.explained_variance_)

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(-11, 9.0)
ax1.set_ylim(-22, 20)
# ax1.set_xlim(-16.0, 6.0)
#ax1.set_ylim(-7.5, 8.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'LD1', fontsize= 35)
plt.ylabel(r'LD2', fontsize= 35)

#print(A1[0][1], B1[0][1])        
AB = np.vstack([ld1[0], ld2[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': ld1[0], 'y': ld2[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
for x1, y1 in zip(ld1[0], ld2[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax1.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
# ax1.scatter(ld1[0], ld2[0], c= sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', zorder=10.0, edgecolor='black', label='CLOUDY modelled halo PNe')
#ax1.scatter(ld1[1], ld2[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='Obs. hPNe')
ax1.scatter(ld1[2], ld2[2], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax1.scatter(ld1[4], ld2[4],  c= "mediumaquamarine", alpha=0.8, s=90, edgecolor='black',  marker='D', label='SDSS QSOs')
ax1.scatter(ld1[7], ld2[7],  c= "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax1.scatter(ld1[1], ld2[1],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
# ax1.scatter(ld1[3], ld2[3],  c= "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax1.scatter(ld1[3], ld2[3],  c= "gray", alpha=0.8, s=90, marker='D', edgecolor='black', zorder=133.0, label='Obs. HII regions in NGC 55')
ax1.scatter(ld1[5], ld2[5],  c= "lightsalmon", alpha=1.0, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
ax1.scatter(ld1[6], ld2[6],  c= "mint green", alpha=1.0, s=30, marker='*', edgecolor='black', label='S-PLUS DR1 WDs')
# ax1.scatter(ld1[7], ld2[7],  c= sns.xkcd_rgb["cerulean"], alpha=1.0, s=650, marker='*', edgecolor='black', zorder= 113, label='Sources selected in J-PLUS DR1')
# ax1.scatter(ld1[8], ld2[8],  c= sns.xkcd_rgb["light green"], alpha=1.0, s=650, marker='*', edgecolor='black', linewidth='1.7', zorder= 113, label='HASH sources in J/S-PLUS DR1')

ax1.minorticks_on()

# for label_, x, y in zip(label_dr1, ld1[8], ld2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(ld1[8], dtype=str), np.array(ld2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)

bbox_props = dict(boxstyle="round", fc="w", ec="0.78", alpha=0.6, pad=0.1)
# for x,y in zip(ld1[8], ld2[8]):
#     ax1.annotate("PN Sp 4-1", (x[0], y[0]), alpha=5,  size="x-large",
#                    xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
#     ax1.annotate("LEDA 101538", (x[1], y[1]), alpha=5, size="x-large",
#                    xytext=(105.0, -13.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
#     ax1.annotate("J-PLUS HII region", (x[2], y[2]), alpha=5, size="x-large",
#                    xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103, weight='bold',)
#     ax1.annotate("[HLG90] 55", (x[3], y[3]), alpha=5, size="x-large",
#                    xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
    #print(x[2])
#ax2.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='lower center', **lgd_kws)
ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig1-JPLUS-LD1-LD2-v0.pdf'
save_path = '../../Dropbox/JPAS/Tesis/Fig/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()
####################################################################
#LD1 vs LD3 ########################################################
####################################################################


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
# ax2.set_xlim(-10.0, 8.0)
# ax2.set_ylim(-2.0, 1.5)
#ax2.set_xlim(-18.0, 8.0)
#ax2.set_ylim(-3.8, 4.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'LD1', fontsize= 35)
plt.ylabel(r'LD3', fontsize= 35)
#print(A1[0][1], B1[0][1])        
AB = np.vstack([ld1[0], ld3[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': ld1[0], 'y': ld3[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()

for x1, y1 in zip(ld1[0], ld3[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax2.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax2.scatter(ld1[0], ld3[0], c = sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', edgecolor='black', zorder = 10.0, label='CLOUDY modelled halo PNe')
ax2.scatter(ld1[1], ld3[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', edgecolor='black', alpha=0.8, zorder=120.0, label='Obs. halo PNe')
ax2.scatter(ld1[4], ld3[4],  c = sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax2.scatter(ld1[6], ld3[6],  c = "mediumaquamarine", alpha=0.8, s=60, edgecolor='black',  marker='D', label='SDSS QSOs')
ax2.scatter(ld1[7], ld3[7],  c = "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax2.scatter(ld1[1], ld3[1],  c = "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
#ax2.scatter(ld1[3], ld3[3],  c = "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax2.scatter(ld1[3], ld3[3],  c = "gray", alpha=0.8, s=90, marker='D', edgecolor='black',  zorder=105, label='Obs. HII regions in NGC 55')
ax2.scatter(ld1[5], ld3[5],  c = "lightsalmon", alpha=1.0, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
ax2.scatter(ld1[6], ld2[6],  c= "mint green", alpha=1.0, s=30, marker='*', edgecolor='black', label='S-PLUS DR1 WDs')
# ax2.scatter(ld1[8], ld3[8],  c = sns.xkcd_rgb["cerulean"], alpha=1.0, s=650, marker='*', edgecolor='black', zorder= 113, label='Obs. select in J-PLUS DR1')
# ax2.scatter(ld1[9], ld3[9],  c= sns.xkcd_rgb["light green"], alpha=1.0, s=650, marker='*', edgecolor='black', zorder= 113, label='HASH sources in J-PLUS DR1')

ax2.minorticks_on()

# for x,y in zip(ld1[8], ld3[8]):
#     ax2.annotate("PN Sp 4-1", (x[0], y[0]), alpha=5, size="x-large",
#                    xytext=(78.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
#     ax2.annotate("LEDA 101538", (x[1], y[1]), alpha=5, size="x-large",
#                    xytext=(100.0, 10.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)
#     ax2.annotate("J-PLUS HII region", (x[2], y[2]), alpha=5, size="x-large",
#                    xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103, weight='bold',)
#     ax2.annotate("[HLG90] 55", (x[3], y[3]), alpha=5, size="x-large",
#                    xytext=(-3.0, 5.0), textcoords='offset points', ha='right', va='bottom', bbox=bbox_props, zorder=103,)

# for label_, x, y in zip(np.array(label), ld1[1], ld2[1]):
#     ax1.annotate(label_, (x, y), size=10.5, 
#                    xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
    
#ax2.grid(which='minor')#, lw=0.3)
#ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
pltfile = 'Fig2-JPLUS-LD1-LD3-v0.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()


