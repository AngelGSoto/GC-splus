'''
Principal component analysis (S-PLUS)
'''
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns


X = []
target = []

pattern =  "*-spectros/*-SPLUS-magnitude.json"

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
    X.append(data["F348_uJAVA"])
    X.append(data["F378"])
    X.append(data["F395"])
    X.append(data["F410"])
    X.append(data["F430"])
    X.append(data["F480_gSDSS"])
    X.append(data["F515"])
    X.append(data["F625"])
    X.append(data["F660"])
    X.append(data["F770_iSDSS"])
    X.append(data["F861"])
    X.append(data["F910_zSDSS"])
    target.append(data["F348_uJAVA"])
    target.append(data["F378"])
    target.append(data["F395"])
    target.append(data["F410"])
    target.append(data["F430"])
    target.append(data["F480_gSDSS"])
    target.append(data["F515"])
    target.append(data["F625"])
    target.append(data["F660"])
    target.append(data["F770_iSDSS"])
    target.append(data["F861"])
    target.append(data["F910_zSDSS"])
    if data["id"].endswith("HPNe"):
        target.append(0)
    elif data["id"].endswith("DPNe"):
        target.append(1)
    elif data["id"].endswith("E00"):
        target.append(2)
    elif data["id"].endswith("E01"):
        target.append(2)
    elif data["id"].endswith("E02"):
        target.append(2)
    elif data["id"].endswith("sys"):
        target.append(3)
    elif data["id"].endswith("sys-IPHAS"):
        target.append(4)
    elif data["id"].endswith("sys-Ext"):
        target.append(5)
    elif data["id"].endswith("survey"):
        target.append(6)
    elif data["id"].endswith("CV"):
        target.append(7)
    elif data["id"].endswith("ExtHII"):
        target.append(8)
    elif data["id"].endswith("SNR"):
        target.append(9)
    elif data["id"].endswith("QSOs-010"):
        target.append(10)
    elif data["id"].endswith("QSOs-101"):
        target.append(11)
    elif data["id"].endswith("QSOs-201"):
        target.append(12)
    elif data["id"].endswith("QSOs-301"):
        target.append(13)
    elif data["id"].endswith("QSOs-401"):
        target.append(14)
    elif data["id"].endswith("QSOs-hz"):
        target.append(13)
    else:
        target.append(15)
      
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
lda = LDA(n_components=6)
lda.fit(XX, m)

XX_lda = lda.transform(XX)

target_names = ["Halo PNe", "DisK PN", "Model PNe", "Munari Symbiotics", "Symbiotics from IPHAS", "Symbiotics in NGC 55", "C. Buil Symbiotics", "CVs", "HII region in NGC 55", "SN Remanents", "QSOs (0.01<z<1.0)", "QSOs (1.01<z<2.0)", "QSOs (2.01<z<3.0)", "QSOs (3.01<z<4.0)", "QSOs (4.01<z<5.0)", "SFGs"]


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
fig = plt.figure(figsize=(11, 8))
ax1 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax1.scatter(XX_lda[m == i, 0], XX_lda[m == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax1.legend(loc='upper right', fontsize='small', scatterpoints=1,  **lgd_kws)
#plt.title('PCA of IRIS dataset')
ax1.set_xlim(-20.9, 20.9)
ax1.set_ylim(-7.0, 7.5)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
plt.xlabel('LD{}'.format(0+1), fontsize=16)
plt.ylabel('LD{}'.format(1+1), fontsize=16)
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax1.minorticks_on()
ax1.grid()
plt.tight_layout()
plt.savefig("plot1-lda-splus.pdf" )#, bbox_extra_artists=(lgd,), bbox_inches='tight')


fig = plt.figure(figsize=(12, 7))
ax2 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax2.scatter(XX_lda[m == i, 0], XX_lda[m == i, 2], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax2.legend(loc='lower right', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('LD{}'.format(0+1), fontsize=16)
plt.ylabel('LD{}'.format(2+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax2.set_xlim(-20.3, 20.3)
ax2.set_ylim(-6.0, 6.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
ax2.minorticks_on()
ax2.grid()
plt.tight_layout()
plt.savefig("plot2-lda-splus.pdf") #, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure(figsize=(12, 7))
ax3 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax3.scatter(XX_lda[m == i, 0], XX_lda[m == i, 3], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax3.legend(loc='upper left', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('LD{}'.format(0+1), fontsize=16)
plt.ylabel('LD{}'.format(3+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax3.set_xlim(-20.3, 20.3)
ax3.set_ylim(-3.0, 4.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
ax3.minorticks_on()
ax3.grid()
plt.tight_layout()
plt.savefig("plot3-lda-splus.jpg") #, bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure(figsize=(12, 7))
ax4 = fig.add_subplot(111)
colors = ['black', 'yellow',  'green', 'red', 'red', 'red', 'red', 'purple','gray', 'black', 'sage', 'salmon', 'goldenrod', 'royalblue', 'mediumaquamarine', 'cyan' ]

marker = ['o', 'o', '*', 's', '^', 'D', 'o', 's', 'D', '.', 'o', '*', '^', 'D', 's', '<' ]

lw = 0.5 #2

for color, i, marker, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], marker, target_names):
    ax4.scatter(XX_lda[m == i, 0], XX_lda[m == i, 4], color=color, alpha=.8, lw=lw,
                label=target_name, marker = marker,  cmap=plt.cm.spectral)
ax4.legend(loc='lower right', fontsize='small', scatterpoints=1, **lgd_kws)
#plt.title('PCA of IRIS dataset')
plt.xlabel('LD{}'.format(0+1), fontsize=16)
plt.ylabel('LD{}'.format(4+1), fontsize=16)
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax4.set_xlim(-20.3, 20.3)
ax4.set_ylim(-2.0, 1.7)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
ax4.minorticks_on()
ax4.grid()
plt.tight_layout()
plt.savefig("plot4-lda-splus.pdf") #, bbox_extra_artists=(lgd,), bbox_inches='tight')
