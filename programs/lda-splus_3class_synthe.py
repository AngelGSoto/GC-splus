'''
Linear discrimant analysis (S-PLUS)-- Plot LDA (at least the implementation in sklearn) can produce at most k-1 components (where k is number of classes). So if you are dealing with binary classification - you'll end up with only 1 dimension.
'''
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
from imblearn.over_sampling import SMOTE
from astropy.table import Table
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings("ignore")

label=[]
label_dr1=[]
X = []
target = []
clas = []

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
        data = OrderedDict((k, v) for k, v in sorted(data.items(), key=lambda x: x[0]))
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
    # elif data["id"].endswith("DPNe"):
    #     target.append(2)
    elif data["id"].endswith("sys"):
        target.append(1)
    elif data["id"].endswith("extr-SySt"):
        target.append(1)
    elif data["id"].endswith("ngc185"):
        target.append(1)
    elif data["id"].endswith("SySt-ic10"):
        target.append(1)
    elif data["id"].endswith("sys-IPHAS"):
        target.append(1)
    else:
        target.append(2)

    clas.append(data["F348_U"])
    clas.append(data["F378"])
    clas.append(data["F395"])
    clas.append(data["F410"])
    clas.append(data["F430"])
    clas.append(data["F480_G"])
    clas.append(data["F515"])
    clas.append(data["F625_R"])
    clas.append(data["F660"])
    clas.append(data["F766_I"])
    clas.append(data["F861"])
    clas.append(data["F911_Z"])
    if data["id"].endswith("E00_300"):
        clas.append(0)
    elif data["id"].endswith("E01_300"):
        clas.append(0)
    elif data["id"].endswith("E02_300"):
        clas.append(0)
    elif data["id"].endswith("E00_600"):
        clas.append(0)
    elif data["id"].endswith("E01_600"):
        clas.append(0)
    elif data["id"].endswith("E02_600"):
        clas.append(0)
    elif data["id"].endswith("E00_100"):
        clas.append(0)
    elif data["id"].endswith("E01_100"):
        clas.append(0)
    elif data["id"].endswith("E02_100"):
        clas.append(0)
    elif data["id"].endswith("HPNe"):
        clas.append(1)
    # elif data["id"].endswith("DPNe"):
    #     clas.append(2)
    elif data["id"].endswith("sys"):
        clas.append(2)
    elif data["id"].endswith("extr-SySt"):
        clas.append(2)
    elif data["id"].endswith("ngc185"):
        clas.append(2)
    elif data["id"].endswith("SySt-ic10"):
        clas.append(2)
    elif data["id"].endswith("sys-IPHAS"):
        clas.append(2)
    else:
        clas.append(3)
        
#print(X.shape)
XX = np.array(X).reshape(shape)
target_ = np.array(target).reshape(shape1)
clas_ = np.array(clas).reshape(shape1)
print("Data shape:", XX.shape)
m = []
m1 = []

XX = clean_nan_inf(XX)

#Create target to classify the kind of object
target_ = clean_nan_inf(target_)
clas_ = clean_nan_inf(clas_)

#XX = np.array(XX[np.logical_not(np.isnan(XX), np.isinf(XX))])
#target_ = np.array(target_[np.logical_not(np.isnan(target_), np.isinf(target_))])

for i in target_:
    m.append(i[12])

m = np.array(m)

for ii in clas_:
    m1.append(ii[12])

m1 = np.array(m1)

print(m.shape)
print(len(target))
print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#Standarized the  S-PLUS photometric data
sc = StandardScaler() 
XX = sc.fit_transform(XX)

#Balancing the data will to better classification models. We will try balancing our data using SMOTE.
# sm = SMOTE(random_state = 33) #33
# XX_new, y_new = sm.fit_sample(XX, m.ravel())
#XX_new = StandardScaler().fit_transform(XX_new)
#create the LDA for J-PLUS photometric system

lda = LDA(n_components=2, store_covariance=True)
lda.fit(XX, m)

XX_lda = lda.transform(XX)

n=4  #11
ld1, ld2, ld3 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

ld1[0].append(XX_lda[m1 == 0, 0])
ld2[0].append(XX_lda[m1 == 0, 1])
ld1[1].append(XX_lda[m1 == 1, 0])
ld2[1].append(XX_lda[m1 == 1, 1])
ld1[2].append(XX_lda[m1 == 2, 0])
ld2[2].append(XX_lda[m1 == 2, 1])
ld1[3].append(XX_lda[m1 == 3, 0])
ld2[3].append(XX_lda[m1 == 3, 1])

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(-8.2, 5.7)
# ax1.set_ylim(-2.5, 1.5)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax1.set_xlim(-8, 10)
#ax1.set_ylim(-10, 10)
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
ax1.scatter(ld1[1], ld2[1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=120.0, label='hPNe')
ax1.scatter(ld1[2], ld2[2],  c= "red", alpha=0.8, s=90, marker='s', edgecolor='black', zorder=3.0, label='SySt')
ax1.scatter(ld1[3], ld2[3],   color= sns.xkcd_rgb["pale yellow"], alpha=0.9, s=60, marker='o', label=r'H$\alpha$ emitters')
# pal1 = sns.dark_palette("yellow", as_cmap=True)
# for ax, ay in zip(ld1[3], ld2[3]):
#     sns.kdeplot(ax, ay, cmap=pal1);

# plt.text(0.62, 0.78, 'Other Halpha emitters',
#          transform=ax1.transAxes, fontsize=13.8)

ax1.legend(scatterpoints=1, ncol=2, fontsize=19.8, loc='upper left', **lgd_kws)
ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
pltfile = 'Fig1-SPLUS18-LD1-LD2.jpg'
save_path = 'Plots-splus/'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()

