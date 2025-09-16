'''
Linear discrimant analysis (S-PLUS)--predicting
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
from astropy.table import Table

label=[]
label_dr1=[]
X = []
target = []

pattern =  "../*-spectros/*-SPLUS18-magnitude.json"

def clean_nan_inf(M):
    mask_nan = np.sum(np.isnan(M), 1) > 0
    mask_inf = np.sum(np.isinf(M), 1) > 0
    lines_to_discard = np.logical_xor(mask_nan, mask_inf)
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
    #elif data["id"].endswith("sys-Ext"):
        #target.append(5)
    #elif data["id"].endswith("survey"):
        #target.append(6)
    elif data["id"].endswith("CV"):
        target.append(2)
    elif data["id"].endswith("ExtHII"):
        target.append(3)
    elif data["id"].endswith("galaxy"):
        target.append(3)
    # elif data["id"].endswith("SNR"):
    #     target.append(9)
    elif data["id"].endswith("QSOs-13"):
        target.append(4)
    elif data["id"].endswith("QSOs-24"):
        target.append(4)
    elif data["id"].endswith("QSOs-32"):
        target.append(4)
    elif data["id"].endswith("YSOs"):
        target.append(5)
    # elif data["id"].endswith("DR1jplus"):
    #     target.append(8)
    # elif data["id"].endswith("DR1jplusHash"):
    #     target.append(9)
    # elif data["id"].endswith("DR1SplusWDs"):
    #     target.append(6)
    else:
        target.append(6)
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
####Predicting   #############################################################
##############################################################################
X_new = []
File_name = input('Input file name:')
tab = Table.read(File_name, format="ascii.tab")
#tab = Table.read("TAP_DR1SPLUS_HA_r_03.tab", format="ascii.tab")
#tab = Table.read("SPLUS_HYDRA_master_catalogue_iDR2_december_2019-auto-PN.tab", format="ascii.tab")
for da in tab:
    X_new.append(da["ujava_auto"])
    X_new.append(da["f378_auto"])
    X_new.append(da["f395_auto"])
    X_new.append(da["f410_auto"])
    X_new.append(da["f430_auto"])
    X_new.append(da["g_auto"])
    X_new.append(da["f515_auto"])
    X_new.append(da["r_auto"])
    X_new.append(da["f660_auto"])
    X_new.append(da["i_auto"])
    X_new.append(da["f861_auto"])
    X_new.append(da["z_auto"])

shape_new = (len(tab["z_auto"]), 12)
XX_test = np.array(X_new).reshape(shape_new)

XX_test = sc.transform(XX_test)
#XX_test = lda.transform(XX_test)

y_pred = lda.predict(XX_test)
y_prob = lda.predict_proba(XX_test)
y_pred_dec = lda.decision_function(XX_test)

print(len(y_pred))
# for a, b in zip(y_pred, y_prob):
#     print(a, b)

#creating table files with each class selected
label_pro=("P(PN)", "P(SySt)", "P(CV)", "P(H II)", "P(QSO)", "P(YSO)", "P(SFG)")

for label_Pro, label_Nu in zip(label_pro, range(7)):
    tab[label_Pro] = y_prob[:,label_Nu]

def select(clas, name_file):
    m = y_pred==clas
    return tab[m].write(name_file, format='ascii.tab', overwrite=True)

select(0.0, "PN-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(1.0, "SySt-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(2.0, "CV-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(3.0, "HII-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(4.0, "QSO-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(5.0, "YSO-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
select(6.0, "SFG-ld-{}-candidate.tab".format(File_name.split('r_')[-1].split('.tab')[0]))
