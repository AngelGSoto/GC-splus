'''
Principal component analysis (S-PLUS)
'''
#from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns


X = []
target = []

pattern =  "*-spectros/*-SPLUS18-magnitude.json"

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
    X.append(data["F348_U"])
    X.append(data["F378"])
    X.append(data["F395"])
    X.append(data["F410"])
    X.append(data["F430"])
    X.append(data["F480_G"])
    X.append(data["F515"])
    X.append(data["F625_R"])
    X.append(data["F660"])
    X.append(data["F766_I"])
    X.append(data["F861"])
    X.append(data["F911_Z"])
    target.append(data["F348_U"])
    target.append(data["F378"])
    target.append(data["F395"])
    target.append(data["F410"])
    target.append(data["F430"])
    target.append(data["F480_G"])
    target.append(data["F515"])
    target.append(data["F625_R"])
    target.append(data["F660"])
    target.append(data["F766_I"])
    target.append(data["F861"])
    target.append(data["F911_Z"])
    
    if data["id"].endswith("E00"):
        target.append(0)
    elif data["id"].endswith("E01"):
        target.append(0)
    elif data["id"].endswith("E02"):
        target.append(0)
    elif data["id"].endswith("E00_900"):
        target.append(0)
    elif data["id"].endswith("E01_900"):
        target.append(0)
    elif data["id"].endswith("E02_900"):
        target.append(0)
    elif data["id"].endswith("E00_100"):
        target.append(0)
    elif data["id"].endswith("E01_100"):
        target.append(0)
    elif data["id"].endswith("E02_100"):
        target.append(0)
    elif data["id"].endswith("HPNe"):
        target.append(1)
    # elif data["id"].endswith("DPNe"):
    #     target.append(2)
    elif data["id"].endswith("sys"):
        target.append(3)
    elif data["id"].endswith("extr-SySt"):
        target.append(3)
    elif data["id"].endswith("ngc185"):
        target.append(3)
    elif data["id"].endswith("SySt-ic10"):
        target.append(3)
    elif data["id"].endswith("sys-IPHAS"):
        target.append(4)
    #elif data["id"].endswith("sys-Ext"):
        #target.append(5)
    #elif data["id"].endswith("survey"):
        #target.append(6)
    elif data["id"].endswith("CV"):
        target.append(7)
    elif data["id"].endswith("ExtHII"):
        target.append(8)
    # elif data["id"].endswith("SNR"):
    #     target.append(9)
    elif data["id"].endswith("QSOs-13"):
        target.append(10)
    elif data["id"].endswith("QSOs-24"):
        target.append(11)
    elif data["id"].endswith("QSOs-32"):
        target.append(12)
    # elif data["id"].endswith("SFGs"):
    #     target.append(13)
    # elif data["id"].endswith("QSOs-401"):
    #     target.append(14)
    # elif data["id"].endswith("QSOs-hz"):
    #     target.append(13)
    else:
        target.append(13)
      
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
pca = PCA(n_components=6)
pca.fit(XX)

XX_pca = pca.transform(XX)

#porcentages
print(pca.explained_variance_ratio_)

#plot waits
filter_name = ['uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'gSDSS', 'J0515', 'rSDSS', 'J0660', 'iSDSS', 'J0861', 'zSDSS']
filter_ = []
for a in range(1, 13):
    filter_.append(a)

w1 = pca.components_[0]
w2 = pca.components_[1]
w3 = pca.components_[2]
print(len(filter_))

plotfile = "splus-wight1.pdf"
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1,1,1)
plt.tick_params(axis='x', labelsize=17) 
plt.tick_params(axis='y', labelsize=17)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
ax.set_xlabel(r'Filters', size = 16)
ax.set_ylabel(r'w$_1$', size = 16)
ax.plot(filter_, w1, 'ko-')
plt.xticks(filter_, filter_name, rotation=45)
#plt.xticks(filter_)
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.savefig(plotfile)

plotfile = "splus-wight2.pdf"
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1,1,1)
plt.tick_params(axis='x', labelsize=17) 
plt.tick_params(axis='y', labelsize=17)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
ax.set_xlabel(r'Filters', size = 16)
ax.set_ylabel(r'w$_2$', size = 16)
ax.plot(filter_, w2, 'ko-')
plt.xticks(filter_, filter_name, rotation=45)
#plt.xticks(filter_)
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.savefig(plotfile)

plotfile = "splus-wight3.jpg"
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1,1,1)
plt.tick_params(axis='x', labelsize=17) 
plt.tick_params(axis='y', labelsize=17)
    #ax1.set_xlim(xmin=-0.5,xmax=2)
    #ax1.set_ylim(ymin=15,ymax=-5)
    #ax1.set_xlabel(r'$\lambda$')
ax.set_xlabel(r'Filters', size = 16)
ax.set_ylabel(r'w$_3$', size = 16)
ax.plot(filter_, w3, 'ko-')
plt.xticks(filter_, filter_name, rotation=45)
#plt.xticks(filter_)
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    #plt.xticks(f.filteravgwls, np.unique(f.filterset['ID_filter']), 
                              #rotation='vertical', size = 'small')
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.savefig(plotfile)


target_names = [ "CLOUDY modelled HPNe", "Obs. HPNe", "Obs SySts", "IPHAS SySts", "SDSS CVs", "SDSS QSOs (1.3<z<1.4)", "SDSS QSOs (2.4<z<2.6)", "SDSS QSOs (3.2<z<3.4)", "SDSS SFGs", "Obs. H II regions in NGC 55"]
colors = [ 'green', 'black', 'red', 'red',  'purple', 'mediumaquamarine', 'royalblue', 'goldenrod', 'cyan', 'gray' ]

marker = ['*', 'o', 's', '^', 'o','s', 'D', '^', '<', 'D']
size   = [40, 74, 40, 40, 40, 40, 40, 40, 40, 40]

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set_style('ticks')
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(111)

for color, i, marker, size, target_name in zip(colors, [0,1, 3,  4, 7, 10, 11, 12, 13, 8], marker, size, target_names):
    ax1.scatter(XX_pca[m == i, 0], XX_pca[m == i, 1], color=color, alpha=0.8,
                label=target_name, marker = marker, s=size)
ax1.legend(loc='upper right', fontsize=16.0, ncol=1, scatterpoints=1,  **lgd_kws)
#plt.title('PCA of IRIS dataset')
ax1.set_xlim(-17.0, 32.5)
ax1.set_ylim(-2.8, 7.1)
plt.tick_params(axis='x', labelsize=22) 
plt.tick_params(axis='y', labelsize=22)
plt.xlabel('PC{}'.format(0+1), fontsize=24)
plt.ylabel('PC{}'.format(1+1), fontsize=24)
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax1.minorticks_on()
ax1.grid()
plt.tight_layout()
plt.savefig("Fig1-SPLUS-PC1-PC2.pdf" )#, bbox_extra_artists=(lgd,), bbox_inches='tight')

sys.exit()
fig = plt.figure(figsize=(12, 7))
ax2 = fig.add_subplot(111)
colors = ['green', 'black', 'red', 'red',  'purple', 'gray', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine',  'cyan' ]

marker = ['*', 'o', 's', '^', 'o','s', 'D', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15], marker, target_names):
    ax2.scatter(XX_pca[m == i, 0], XX_pca[m == i, 2], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
#ax2.legend(loc='lower right', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('PC{}'.format(0+1), fontsize=16)
plt.ylabel('PC{}'.format(2+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax2.set_xlim(-20.3, 20.3)
ax2.set_ylim(-4.0, 3.6)
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18)
ax2.minorticks_on()
ax2.grid()
plt.tight_layout()
plt.savefig("plot2-pca-splus.pdf") #, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure(figsize=(12, 7))
ax3 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax3.scatter(XX_pca[m == i, 0], XX_pca[m == i, 3], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax3.legend(loc='upper left', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('PC{}'.format(0+1), fontsize=16)
plt.ylabel('PC{}'.format(3+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax3.set_xlim(-20.3, 20.3)
ax3.set_ylim(-3.0, 4.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
ax3.minorticks_on()
ax3.grid()
plt.tight_layout()
plt.savefig("plot3-pca-splus.pdf") #, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure(figsize=(12, 7))
ax4 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax4.scatter(XX_pca[m == i, 0], XX_pca[m == i, 4], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax4.legend(loc='lower right', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('PC{}'.format(0+1), fontsize=16)
plt.ylabel('PC{}'.format(4+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax4.set_xlim(-20.3, 20.3)
ax4.set_ylim(-2.0, 1.7)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
ax4.minorticks_on()
ax4.grid()
plt.tight_layout()
plt.savefig("plot4-pca-splus.jpg") #, bbox_extra_artists=(lgd,), bbox_inches='tight')
