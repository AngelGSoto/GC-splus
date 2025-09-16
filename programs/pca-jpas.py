'''
Principal component analysis (JPAS)
'''
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns
from collections import OrderedDict
import operator
from operator import itemgetter
from scipy.stats import gaussian_kde
import pandas as pd
import os.path
import matplotlib.cm as cm
from colour import Color
from matplotlib.ticker import FormatStrFormatter

X = []
target = []

pattern =  "*-spectros/*-JPAS17-magnitude.json"

def clean_nan_inf(M):
    mask_nan = np.sum(np.isnan(M), 1) > 0
    mask_inf = np.sum(np.isinf(M), 1) > 0
    lines_to_discard = np.logical_xor(mask_nan, mask_inf)
    print("Number of lines to discard:", sum(lines_to_discard))
    M = M[np.logical_not(lines_to_discard), :]
    return M
    
file_list = glob.glob(pattern)

shape = (len(file_list), 56)
shape1 = (len(file_list), 57)
print(len(file_list))
for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        data = OrderedDict((k, v) for k, v in sorted(data.items(), key=lambda x: x[0]))
        for k in data.keys():
            if k.startswith('Jv0915'):
                imagename=k
                #print(k)
                X.append(data[imagename])
                target.append(data[imagename])
        if data["id"].endswith("E00"):
            target.append(0)
        elif data["id"].endswith("E01"):
            target.append(0)
        elif data["id"].endswith("E02"):
            target.append(0)
        elif data["id"].endswith("E00_100"):
            target.append(0)
        elif data["id"].endswith("E01_100"):
            target.append(0)
        elif data["id"].endswith("E02_100"):
            target.append(0)
        elif data["id"].endswith("E00_900"):
            target.append(0)
        elif data["id"].endswith("E01_900"):
            target.append(0)
        elif data["id"].endswith("E02_900"):
            target.append(0)    
        elif data["id"].endswith("HPNe"):
            target.append(1)
        elif data["id"].endswith("HPNe-"):
            target.append(1)
    # elif data["id"].endswith("DPNe"):
    #     target.append(2)
        elif data["id"].endswith("sys"):
            target.append(3)
        elif data["id"].endswith("sys-raman"):
            target.append(3)
        elif data["id"].endswith("extr-SySt-raman"):
            target.append(3)
        elif data["id"].endswith("extr-SySt"):
            target.append(3)
        # elif data["id"].endswith("ngc185-raman"):
        #     target.append(3)
        elif data["id"].endswith("SySt-ic10"): 
            target.append(3)
        elif data["id"].endswith("sys-IPHAS"):
            target.append(4)
        elif data["id"].endswith("sys-IPHAS-raman"):
            target.append(4)
    #elif data["id"].endswith("sys-Ext"):
        #target.append(5)
    #elif data["id"].endswith("survey"):
        #target.append(6)
        elif data["id"].endswith("CV"):
            target.append(5)
        elif data["id"].endswith("ExtHII"):
            target.append(6)
    # elif data["id"].endswith("SNR"):
    #     target.append(9)
        elif data["id"].endswith("QSOs-13"):
            target.append(7)
        elif data["id"].endswith("QSOs-24"):
            target.append(7)
        elif data["id"].endswith("QSOs-30"):
            target.append(7)
        elif data["id"].endswith("QSOs-32"):
            target.append(7)
        elif data["id"].endswith("YSOs"):
            target.append(8)
        elif data['id'].endswith("Be"):
            target.append(9)
        else:
            target.append(10)
                           
XX = np.array(X).reshape(shape)
target_ = np.array(target).reshape(shape1)
print("Data shape:", XX.shape)

m = [] #To clasify the objects for example 0 PNe, 1 symbiotic stars, etc-- list

XX = clean_nan_inf(XX)
print("np.inf=", np.where(np.isnan(XX)))
print("np.max=", np.max(abs(XX)))

#XX[XX >= np.nan ] = 0
#target_[XX >= np.inf] = 0

#Create target to classify the kind of object
target_ = clean_nan_inf(target_)

#target_[target_ >= np.nan ] = 0
#target_[target_ >= np.inf] = 0

for i in target_:
    m.append(i[56])

m = np.array(m)
print(m.shape)
print(XX.shape)

# aa =  []
# for a, b in zip(target_, file_list):
#     aa.append(a)
#     aa.append(b)
# print(aa)

np.savetxt("Values_pca_jpas2.txt", target_)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#create the PCA for S-PLUS photometric system
XX = StandardScaler().fit_transform(XX.data)
#create the PCA for JPAS photometric system
pca = PCA(n_components=6)
pca.fit(XX)
XX_pca = pca.transform(XX)

print("Porcentage:", pca.explained_variance_ratio_)

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
# pc1[2].append(XX_pca[m == 2, 0])
# pc2[2].append(XX_pca[m == 2, 1])
# pc3[2].append(XX_pca[m == 2, 2])
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
pc1[9].append(XX_pca[m == 9, 0])
pc2[9].append(XX_pca[m == 9, 1])
pc3[9].append(XX_pca[m == 9, 2])
pc1[10].append(XX_pca[m == 10, 0])
pc2[10].append(XX_pca[m == 10, 1])
pc3[10].append(XX_pca[m == 10, 2])


filter_ = []

for a in range(56):
    filter_.append(a)

filter_name = ['J3495', 'J3781', 'J3900', 'J4000', 'J4100', 'J4200', 'J4301', 'J4400', 'J4501', 'J4600', 'J4701', 'J4801', 'J4900', 'J5001', 'J5101', 'J5201', 'J5300', 'J5400', 'J5500', 'J5600', 'J5700', 'J5799', 'J5898', 'J5999', 'J6098', 'J6199', 'J6300', 'J6401', 'J6500', 'J6600', 'J6700', 'J6800', 'J6900', 'J7000', 'J7100', 'J7200', 'J7300', 'J7400', 'J7500', 'J7600', 'J7700', 'J7800', 'J7900', 'J8000', 'J8100', 'J8201', 'J8300', 'J8400', 'J8500', 'J8600', 'J8700', 'J8800', 'J8900', 'J9000', 'J9100', 'J9669']
    
w1 = pca.components_[0]
w2 = pca.components_[1]
w3 = pca.components_[2]
print(len(filter_))

wl = [3495, 3781, 3900, 4000, 4100, 4200, 4301, 4400, 4501, 4600, 4701, 4801, 4900, 5001, 5101, 5201, 5300, 5400, 5500, 5600, 5700, 5799, 5898,  5999, 6098, 6199, 6300, 6401, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8201, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9669]

#Color of the points
colors = cm.rainbow(np.linspace(0, 1, len(w1)))
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
plotfile = "jpas-pcs-wight1.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
ax.set_ylim(ymin=-0.136,ymax=-0.126)
    #ax1.set_xlabel(r'$\lambda$')
#ax.set_xlabel(r'Wavelength($\AA$)', size = 18)
ax.set_xlabel(r'Wavelength $[\mathrm{\AA]}$', size = 45)
ax.set_ylabel(r'$w_1$', size = 45)
ax.plot(wl, w1, '-k', alpha=0.1)
# for wl, w1, colorss in zip(wl, w1, colors):
ax.scatter(wl, w1, color = colors, marker='o',  cmap=plt.cm.hot, edgecolor='black', s=200, zorder=10)
ax.annotate("He II", (4701-0.1, -0.130), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax.annotate("[O III]", (5001, -0.1264), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax.annotate(r"H$\alpha$", (6600, -0.1277), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax.annotate(r"[S III]", (9100, -0.1305), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
# plt.text(4701, -0.130, 'He II',
#             transform=ax.transAxes, fontsize=60,  fontdict=font)
#plt.xticks(filter_, filter_name, rotation=70)
#plt.xticks(filter_)
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
plt.tight_layout()
save_path = '../../Dropbox/JPAS/Tesis/Fig'
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
#------------------------------------
plotfile = "jpas-pcs-wight2.pdf"
fig = plt.figure(figsize=(14, 9))
ax1 = fig.add_subplot(1,1,1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
ax1.set_ylim(-0.22,0.4)
    #ax1.set_xlabel(r'$\lambda$')
#ax1.set_xlabel(r'Wavelength($\AA$)', size = 18)
ax1.set_xlabel(r'Wavelength $[\mathrm{\AA]}$', size = 45)
ax1.set_ylabel(r'$w_2$', size = 45)
ax1.plot(wl, w2, '-k', alpha=0.1)
ax1.scatter(wl, w2, color = colors, marker='o', cmap=plt.cm.hot, edgecolor='black', s=200, zorder=10)
ax1.annotate("He II", (4701-0.1, 0.21), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom', zorder=15)#, bbox=bbox_props,)
ax1.annotate("[O III]", (5001, 0.3), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax1.annotate(r"H$\alpha$", (6600, 0.3), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax1.annotate(r"[S III]", (9100, -0.01), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
# plt.xticks(filter_, filter_name, rotation=70)
#plt.xticks(filter_)
ax1.axhline(y=0, c='k')
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.tight_layout()
plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
#------------------------------------
plotfile = "jpas-pcs-wight3.pdf"
fig = plt.figure(figsize=(14, 9))
ax2 = fig.add_subplot(1,1,1)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
#ax2.set_xlabel(r'Wavelength($\AA$)', size = 18)
ax2.set_xlabel(r'Wavelength $[\mathrm{\AA]}$', size = 45)
ax2.set_ylabel(r'$w_3$', size = 45)
ax2.plot(wl, w3, '-k', alpha= 0.1)
ax2.scatter(wl, w3, color = colors, marker='o', cmap=plt.cm.hot, edgecolor='black', s=200, zorder=10)
#plt.xticks(filter_, filter_name, rotation=70)
#plt.xticks(filter_)
ax2.annotate("He II", (4701-0.1, 0.15), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom', zorder=15)#, bbox=bbox_props,)
ax2.annotate("[O III]", (5001, -0.319), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax2.annotate(r"H$\alpha$", (6600, -0.29), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
ax2.annotate(r"[S III]", (9100, -0.38), alpha=15, size=29.0,
                   xytext=(24, -18), textcoords='offset points', ha='right', va='bottom')#, bbox=bbox_props,)
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
ax2.axhline(y=0, c='k')
plt.margins(0.06)
plt.tight_layout()
plt.tight_layout()
plt.subplots_adjust(bottom=0.17)
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
#########################################
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax3 = fig.add_subplot(111)
# ax1.set_xlim(-11, 9.0)
ax3.set_ylim(-5.8, 6.5)
# ax1.set_xlim(-27.0, 23.0)
#ax3.set_ylim(-6.0, 22.0)
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
    ax3.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax1.scatter(pc1[0], pc2[0], c= sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', zorder=10.0, edgecolor='black', label='CLOUDY modelled halo PNe')
ax3.scatter(pc1[1], pc2[1],  color= sns.xkcd_rgb["aqua"], s=150, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='Obs. hPNe')
ax3.scatter(pc1[5], pc2[5], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax3.scatter(pc1[7], pc2[7],  c= "mediumaquamarine", alpha=0.8, s=90, edgecolor='black',  marker='D', label='SDSS QSOs')
ax3.scatter(pc1[10], pc2[10],  c= "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax3.scatter(pc1[3], pc2[3],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
ax3.scatter(pc1[4], pc2[4],  c= "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax3.scatter(pc1[6], pc2[6],  c= "gray", alpha=0.8, s=90, marker='D', edgecolor='black', zorder=300.0, label='Obs. HII regions in NGC 55')
ax3.scatter(pc1[8], pc2[8],  c= "lightsalmon", alpha=1.0, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
#ax3.scatter(pc1[9], pc2[9],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=150, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')

print(pc1[4], pc2[4])
# for label_, x, y in zip(label_dr1, pc1[8], pc2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(pc1[8], dtype=str), np.array(pc2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)

#ax2.grid(which='minor')#, lw=0.3)
ax3.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='lower center', **lgd_kws)
ax3.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig1-JPAS17-PC1-PC2.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################


#########################################
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax4 = fig.add_subplot(111)
# ax1.set_xlim(-11, 9.0)
# ax1.set_ylim(-3.4, 3.2)
# ax1.set_xlim(-27.0, 23.0)
# ax1.set_ylim(-7.5, 8.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)

#print(A1[0][1], B1[0][1])        
AB = np.vstack([pc1[0], pc2[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': pc1[0], 'y': pc2[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
for x1, y1 in zip(pc1[0], pc2[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax4.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax1.scatter(pc1[0], pc2[0], c= sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', zorder=10.0, edgecolor='black', label='CLOUDY modelled halo PNe')
ax4.scatter(pc1[1], pc3[1],  color= sns.xkcd_rgb["aqua"], s=150, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='Obs. hPNe')
ax4.scatter(pc1[5], pc3[5], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax4.scatter(pc1[7], pc3[7],  c= "mediumaquamarine", alpha=0.8, s=90, edgecolor='black',  marker='D', label='SDSS QSOs')
ax4.scatter(pc1[10], pc3[10],  c= "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax4.scatter(pc1[3], pc3[3],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
ax4.scatter(pc1[4], pc3[4],  c= "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax4.scatter(pc1[6], pc3[6],  c= "gray", alpha=0.8, s=90, marker='D', edgecolor='black',  zorder=300, label='Obs. HII regions in NGC 55')
ax4.scatter(pc1[8], pc3[8],  c= "lightsalmon", alpha=1.0, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
ax4.scatter(pc1[9], pc3[9],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=150, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')

# for label_, x, y in zip(label_dr1, pc1[8], pc2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(pc1[8], dtype=str), np.array(pc2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)

#ax2.grid(which='minor')#, lw=0.3)
#ax4.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='lower center', **lgd_kws)
ax4.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig2-JPAS17-PC1-PC3.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC2 vs PC3 ########################################################
####################################################################
#########################################

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax5 = fig.add_subplot(111)
# ax1.set_xlim(-11, 9.0)
# ax1.set_ylim(-3.4, 3.2)
# ax1.set_xlim(-27.0, 23.0)
# ax1.set_ylim(-7.5, 8.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC2', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)

#print(A1[0][1], B1[0][1])        
AB = np.vstack([pc1[0], pc2[0]])
z = gaussian_kde(AB)(AB)
df=pd.DataFrame({'x': pc1[0], 'y': pc2[0]})

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
for x1, y1 in zip(pc1[0], pc2[0]):
    x11, y11, z = x1[idx], y1[idx], z[idx]
    ax5.scatter(x11, y11, c=z, s=50, zorder=10, alpha=0.5, edgecolor='')
#ax1.scatter(pc1[0], pc2[0], c= sns.xkcd_rgb["indigo"], alpha=0.5, s=130, marker='o', zorder=10.0, edgecolor='black', label='CLOUDY modelled halo PNe')
ax5.scatter(pc2[1], pc3[1],  color= sns.xkcd_rgb["aqua"], s=150, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='Obs. hPNe')
ax5.scatter(pc2[5], pc3[5], c=sns.xkcd_rgb['pale yellow'], alpha=0.8, s=90, marker='o', edgecolor='black', zorder=3.0, label='SDSS CVs')
ax5.scatter(pc2[7], pc3[7],  c= "mediumaquamarine", alpha=0.8, s=90, edgecolor='black',  marker='D', label='SDSS QSOs')
ax5.scatter(pc2[10], pc3[10],  c= "goldenrod", alpha=0.8, s=120, marker='^', edgecolor='black', label='SDSS SFGs')
ax5.scatter(pc2[3], pc3[3],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='Obs SySt')
ax5.scatter(pc2[4], pc3[4],  c= "red", alpha=0.8, s=120, marker='^', edgecolor='black', zorder=3.0, label='IPHAS SySt')
ax5.scatter(pc2[6], pc3[6],  c= "gray", alpha=0.8, s=90, marker='D', edgecolor='black', zorder=300,  label='Obs. HII regions in NGC 55')
ax5.scatter(pc2[8], pc3[8],  c= "lightsalmon", alpha=1.0, s=150, marker='*', edgecolor='black', label='Obs. YSOs')
ax5.scatter(pc2[9], pc3[9],  c=sns.xkcd_rgb['ultramarine blue'], alpha=0.8, s=150, cmap=plt.cm.hot, marker='*',  edgecolor='black', zorder=110, label='Obs. B[e] stars')

# for label_, x, y in zip(label_dr1, pc1[8], pc2[8]):
#     print(label_, x, y)
#     ax1.annotate(label_dr1, (np.array(pc1[8], dtype=str), np.array(pc2[8], dtype=str)), size=10.5, 
#                  xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)

#ax2.grid(which='minor')#, lw=0.3)
#ax5.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='lower center', **lgd_kws)
ax5.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
pltfile = 'Fig3-JPAS17-PC2-PC3.pdf'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()
