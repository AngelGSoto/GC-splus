'''
Principal component analysis (S-PLUS internal DR2)---Predicting
'''
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from matplotlib.ticker import FormatStrFormatter
from imblearn.over_sampling import SMOTE
from astropy.table import Table

label=[]
label_dr1=[]
X = []
target = []

pattern =  "../../*-spectros/*-SPLUS18-magnitude.json"

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

########################################################################################################
#Predicting           ##################################################################################
########################################################################################################
X_new = []
File_name = input('Input file name:')
tab = Table.read(File_name, format="ascii.tab")
#tab = Table.read("TAP_DR1SPLUS_HA_r_03.tab", format="ascii.tab")
for da in tab:
    X_new.append(da["uJAVA_auto"])
    X_new.append(da["F378_auto"])
    X_new.append(da["F395_auto"])
    X_new.append(da["F410_auto"])
    X_new.append(da["F430_auto"])
    X_new.append(da["g_auto"])
    X_new.append(da["F515_auto"])
    X_new.append(da["r_auto"])
    X_new.append(da["F660_auto"])
    X_new.append(da["i_auto"])
    X_new.append(da["F861_auto"])
    X_new.append(da["z_auto"])

shape_new = (len(tab["z_auto"]), 12)
XX_test = np.array(X_new).reshape(shape_new)
print(XX_test)
XX_test = sc.transform(XX_test)
XX_test = pca.transform(XX_test)

#Fitting Logistic Regression to the Training set
classifier = LogisticRegression(solver='lbfgs', max_iter = 12000, multi_class='multinomial')
classifier.fit(XX_pca_new, y_new)
y_pred = classifier.predict(XX_test)
y_prob = classifier.predict_proba(XX_test)
print(y_pred)

print(len(y_pred))
# for a, b in zip(y_pred, y_prob):
#     print(a, b)

#creating table files with each class selected
label_pro=("P(PN)", "P(SySt)", "P(all else)")

for label_Pro, label_Nu in zip(label_pro, range(3)):
    tab[label_Pro] = y_prob[:,label_Nu]

def select(clas, name_file):
    m = y_pred==clas
    return tab[m].write(name_file, format='ascii.tab', overwrite=True)

select(0.0, "PN-pc-{}.tab".format(File_name.split('.tab')[0]))
select(1.0, "SySt-pc-{}.tab".format(File_name.split('.tab')[0]))
sys.exit()
m0 = y_prob[:,0] >= 0.8
m1 = y_prob[:,1] >= 0.8
tab[m0].write("PN-pc-{}.tab".format(File_name.split('2_')[-1].split('.tab')[0]), format='ascii.tab', overwrite=True)
tab[m1].write("SySt-pc-{}.tab".format(File_name.split('2_')[-1].split('.tab')[0]), format='ascii.tab', overwrite=True)
