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
from imblearn.over_sampling import SMOTE

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
    # if data['id'].endswith("6129-DR1jplus"):
    #     label_dr1.append("J-PLUS object")
    # if data['id'].endswith("4019-DR1jplus"):
    #     label_dr1.append("LEDA 2790884")
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

#Standarized the  S-PLUS photometric data
sc = StandardScaler() 
XX = sc.fit_transform(XX)

#Balancing the data will to better classification models. We will try balancing our data using SMOTE.
sm = SMOTE(random_state = 33) #33
XX_new, y_new = sm.fit_sample(XX, m.ravel())
#XX_new = StandardScaler().fit_transform(XX_new)
#create the LDA for J-PLUS photometric system
lda = LDA(n_components=6)
lda.fit(XX_new, y_new)

XX_lda = lda.transform(XX_new)

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
    if da['id'].endswith("3014-DR1jplusHash"):
        label_dr1.append("Kn J1857.7+3931")
    if da['id'].endswith("5598-DR1jplusHash"):
        label_dr1.append("Jacoby")
    if da['id'].endswith("6034-DR1jplusHash"):
        label_dr1.append("TK 1")
    if da['id'].endswith("36135-DR1jplusHash"):
        label_dr1.append("f r-1")
    if da['id'].endswith("6129-DR1jplus"):
        label_dr1.append("J-PLUS object")
    if da['id'].endswith("4019-DR1jplus"):
        label_dr1.append("LEDA 2790884")
    if da['id'].endswith("12636-DR1jplus"):
        label_dr1.append("LEDA 101538")
    elif da['id'].startswith("18242-DR1jplus"):
        label_dr1.append("PN Sp 4-1")
    # if da['id'].startswith("PNG135-hPN-SVD"):
    #     label_dr1.append("PNG 135")
    # elif da['id'].startswith("H-41-hPNe-SVD"):
    #     label_dr1.append("H 4-1")
    

shape_new = (len(file_list1), 12)
XX_test = np.array(X_new).reshape(shape_new)
XX_test = sc.transform(XX_test)
#XX_test = lda.transform(XX_test)

y_pred = lda.predict(XX_test)
y_prob = lda.predict_proba(XX_test)
y_pred_dec = lda.decision_function(XX_test)

for a, b, c, d in zip(label_dr1, y_pred, y_prob, y_pred_dec):
    print(a, b, c)
# print(XX_new)
# print(y_pred)



