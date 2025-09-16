'''
Principal component analysis (S-PLUS internal DR2)---Predicting
'''
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
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
from imblearn.over_sampling import SMOTE
from astropy.table import Table
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

label=[]
label_dr1=[]
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
    # elif data["id"].endswith("DR1jplus"):
    #     target.append(8)
    # elif data["id"].endswith("DR1jplusHash"):
    #     target.append(9)
    # elif data["id"].endswith("DR1SplusWDs"):
    #     target.append(6)
    else:
        target.append(2)
            
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
sc = StandardScaler()
# ##################################################################
# #Accuracy
# X_train, X_test, y_train, y_test = train_test_split(XX, m)
# X_train = sc.fit_transform(X_train) 
# X_test = sc.transform(X_test)

# pca_sco = PCA(n_components = 3) 
  
# X_train_pca = pca_sco.fit_transform(X_train)

# X_test = pca_sco.transform(X_test) 

# classifier = LogisticRegression(solver='lbfgs', max_iter = 12000, multi_class='multinomial')
# classifier.fit(X_train_pca, y_train)

# # Predicting the test set result using  
# # predict function under LogisticRegression  
# test_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_test, test_pred)
# print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, y_test))
# print('Precision score for Testing Dataset  = ', precision_score(test_pred, y_test, average="micro"))
# print('Confusion matrix = ', cm)

# # Acuracy
# X_train1, X_test1, y_train1, y_test1 = train_test_split(XX, m)
# X_train1 = sc.fit_transform(X_train1) 
# X_test1 = sc.transform(X_test1)

# pca_sco1 = PCA(n_components = 3) 
  
# X_train_pca1 = pca_sco1.fit_transform(X_train1)

# sm1 = SMOTE(random_state = 33) #33
# X_train_new1, y_train_new1 = sm1.fit_sample(X_train_pca1, y_train1.ravel())

# X_test1 = pca_sco1.transform(X_test1) 

# classifier1 = LogisticRegression(solver='lbfgs', max_iter = 12000, multi_class='multinomial')
# classifier1.fit(X_train_new1, y_train_new1)

# # Predicting the test set result using  
# # predict function under LogisticRegression  
# test_pred1 = classifier1.predict(X_test1)

# cm = confusion_matrix(y_test1, test_pred1)
# print('Accuracy score for Testing Dataset SMOTE= ', accuracy_score(test_pred1, y_test1))
# print('Precision score for Testing Dataset SMOTE = ', precision_score(test_pred1,  y_test1, average="macro"))
# print('Confusion matrix = ', cm)
######################################################################################3

XX = sc.fit_transform(XX)

# #Balancing the data will to better classification models. We will try balancing our data using SMOTE.
# sm = SMOTE(random_state = 33) #33
# XX_new, y_new = sm.fit_sample(XX, m.ravel())

#apllying pca
pca = PCA(n_components=None)
pca.fit(XX)
variance = pca.explained_variance_ratio_

pca = PCA(n_components=3)
XX_pca = pca.fit_transform(XX)

#Balancing the data will to better classification models. We will try balancing our data using SMOTE.
sm = SMOTE(random_state = 33) #33
XX_pca_new, y_new = sm.fit_sample(XX_pca, m.ravel())

#porcentages
print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

#Lista with the objects
n=3
pc1, pc2, pc3 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]
##################################################
####Acuracy & presicion
##################################################
X_train, X_test, y_train, y_test = train_test_split(XX_pca, m)

classifier = LogisticRegression(solver='lbfgs', max_iter = 12000, multi_class='multinomial')
classifier.fit(X_train, y_train.ravel())

# Predicting the test set result using  
# predict function under LogisticRegression  
test_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, test_pred)
print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, y_test))
print('Precision score for Testing Dataset  = ', precision_score(test_pred, y_test, average="macro"))
print('Confusion matrix = ', cm)

#SMOTE
sm1 = SMOTE(random_state = 33)
X_train_new, y_train_new = sm1.fit_sample(X_train, y_train.ravel())

# fit the model
classifier.fit(X_train_new, y_train_new)

# prediction for Training data
train_pred_sm = classifier.predict(X_train_new)

# prediction for Testing data
test_pred_sm = classifier.predict(X_test)

print('Accuracy score for Testing Dataset SMOTE = ', accuracy_score(test_pred_sm, y_test))
print('Precision score for Testing Dataset SMOTE= ', precision_score(test_pred_sm, y_test,  average="macro"))


#################################################
pc1[0].append(XX_pca_new[y_new == 0, 0])
pc2[0].append(XX_pca_new[y_new == 0, 1])
pc3[0].append(XX_pca_new[y_new == 0, 2])
pc1[1].append(XX_pca_new[y_new == 1, 0])
pc2[1].append(XX_pca_new[y_new == 1, 1])
pc3[1].append(XX_pca_new[y_new == 1, 2])
pc1[2].append(XX_pca_new[y_new == 2, 0])
pc2[2].append(XX_pca_new[y_new == 2, 1])
pc3[2].append(XX_pca_new[y_new == 2, 2])

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(-8.2, 5.7)
# ax1.set_ylim(-2.5, 1.5)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.set_xlim(-10, 10)
#ax1.set_ylim(-10, 10)
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
ax1.scatter(pc1[1], pc2[1],  c= "red", alpha=0.8, s=70, marker='s', zorder=3.0, label='SySt')
ax1.scatter(pc1[2], pc2[2],   color= sns.xkcd_rgb["pale yellow"], alpha=0.9, s=60, marker='o', label=r'H$\alpha$ emitters')
# pal1 = sns.dark_palette("yellow", as_cmap=True)
# for ax, ay in zip(pc1[2], pc2[2]):
#     sns.kdeplot(ax, ay, cmap=pal1);

#ax1.legend(scatterpoints=1, ncol=2, fontsize=19.8, loc='upper left', **lgd_kws)
ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
pltfile = 'Fig1-SPLUS18-PC1-PC2_SMOTE.jpg'
save_path = 'Plots-splus/'
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
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_xlim(-8.2, 5.7)
# ax2.set_ylim(-1.1, 1.0)
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
ax2.scatter(pc1[1], pc3[1],  c= "red", alpha=0.8, s=70, marker='s',  zorder=3.0, label='SySt')
ax2.scatter(pc1[2], pc3[2],   color= sns.xkcd_rgb["pale yellow"], alpha=0.9, s=60, marker='o', label='Halpha emitters')
# pal2 = sns.dark_palette("yellow", as_cmap=True)
# for bx, by in zip(pc1[2], pc3[2]):
#     sns.kdeplot(bx, by, cmap=pal2);

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
#ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper left', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#pltfile = 'Fig2-JPLUS-PC1-PC3-veri.pdf'
pltfile = 'Fig2-JPLUS18-PC1-PC3_SMOTE.jpg'
file_save = os.path.join(save_path, pltfile)
plt.savefig(file_save)
plt.clf()
