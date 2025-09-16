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

label=[]
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
    elif data["id"].endswith("sys-raman"):
        target.append(3)
    elif data["id"].endswith("extr-SySt-raman"):
        target.append(3)
    elif data["id"].endswith("extr-SySt"):
        target.append(3)
    elif data["id"].endswith("ngc185-raman"):
        target.append(3)
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
    #label
    if data['id'].endswith("-1-HPNe"):
        label.append("DdDm-1")
    if data['id'].endswith("SLOAN-HPNe"):
         label.append("H4-1")
    if data['id'].endswith("1359559-HPNe"):
         label.append("PNG 135.9+55.9")
    if data['id'].startswith("ngc"):
         label.append("NGC 2242")
    elif data['id'].startswith("mwc"):
            label.append("MWC 574")
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
pca = PCA(n_components=6)
pca.fit(XX)

XX_pca = pca.transform(XX)

#porcentages
#print(pca.explained_variance_ratio_)

#Lista with the objects
n=10
pc1, pc2 = [[] for _ in range(n)], [[] for _ in range(n)]

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

pc1[0].append(XX_pca[m == 0, 0])
pc2[0].append(XX_pca[m == 0, 1])
pc1[1].append(XX_pca[m == 1, 0])
pc2[1].append(XX_pca[m == 1, 1])
pc1[2].append(XX_pca[m == 3, 0])
pc2[2].append(XX_pca[m == 3, 1])
pc1[3].append(XX_pca[m == 4, 0])
pc2[3].append(XX_pca[m == 4, 1])
pc1[4].append(XX_pca[m == 7, 0])
pc2[4].append(XX_pca[m == 7, 1])
pc1[5].append(XX_pca[m == 10, 0])
pc2[5].append(XX_pca[m == 10, 1])
pc1[6].append(XX_pca[m == 11, 0])
pc2[6].append(XX_pca[m == 11, 1])
pc1[7].append(XX_pca[m == 12, 0])
pc2[7].append(XX_pca[m == 12, 1])
pc1[8].append(XX_pca[m == 13, 0])
pc2[8].append(XX_pca[m == 13, 1])
pc1[9].append(XX_pca[m == 8, 0])
pc2[9].append(XX_pca[m == 8, 1])


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(-20.0, 31.0)
ax1.set_ylim(-5.6, 4.5)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=20) 
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r'PC1', fontsize= 22)
plt.ylabel(r'PC2', fontsize= 22)

ax1.scatter(pc1[0], pc2[0],  c= "green", alpha=0.3, s=46, marker='*', label='CLOUDY modelled HPNe')
ax1.scatter(100, 100,  c= "black", alpha=1.0, s=60, marker='o', label='Obs. HPNe')
ax1.scatter(100, 100,  c= "gray", alpha=0.8, s=40, marker='D', label='Obs. HII regions in NGC 55')
ax1.scatter(pc1[4], pc2[4],  c= "purple", alpha=0.8, s=40, marker='o', label='SDSS CVs')
ax1.scatter(pc1[5], pc2[5],  c= "mediumaquamarine", alpha=0.8, s=40, marker='s', label='SDSS QSOs (1.3<z<1.4)')
ax1.scatter(pc1[6], pc2[6],  c= "royalblue", alpha=0.8, s=40, marker='D', label='SDSS QSOs (2.4<z<2.6)')
ax1.scatter(pc1[7], pc2[7],  c= "goldenrod", alpha=0.8, s=40, marker='^', label='SDSS QSOs (3.2<z<3.4)')
ax1.scatter(pc1[8], pc2[8],  c= "cyan", alpha=0.8, s=40, marker='^', label='SDSS SFGs')
ax1.scatter(pc1[2], pc2[2],  c= "red", alpha=0.8, s=40, marker='s', label='Obs SySts')
ax1.scatter(pc1[3], pc2[3],  c= "red", alpha=0.8, s=40, marker='^', label='IPHAS SySts')
ax1.scatter(pc1[9], pc2[9],  c= "gray", alpha=0.8, s=40, marker='D')
ax1.scatter(pc1[1], pc2[1],  c= "black", alpha=1.0, s=60, marker='o')
ax1.minorticks_on()

# for label_, x, y in zip(np.array(label), pc1[1], pc2[1]):
#     ax1.annotate(label_, (x, y), size=10.5, 
#                    xytext=(55, 10.0), textcoords='offset points', ha='right', va='bottom', weight='bold',)
    
#ax2.grid(which='minor')#, lw=0.3)
ax1.legend(scatterpoints=1, ncol=2, fontsize=12.8, loc='lower center', **lgd_kws)
ax1.grid()
#lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#plt.savefig('luis-JPLUS-Viironen.pdf')#,  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('Fig1-SPLUS-PC1-PC2-v3.pdf')
plt.clf()
sys.exit()
