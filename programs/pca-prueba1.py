'''
Principal component analysis
'''
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json


X = []
target = []

pattern =  "*-spectros/*-JPLUS13-magnitude.json"

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
    X.append(data["F348"])
    X.append(data["F378"])
    X.append(data["F395"])
    X.append(data["F410"])
    X.append(data["F430"])
    X.append(data["F480_g_sdss"])
    X.append(data["F515"])
    X.append(data["F610_r_sdss"])
    X.append(data["F660"])
    X.append(data["F760_i_sdss"])
    X.append(data["F861"])
    X.append(data["F910_z_sdss"])
    target.append(data["F348"])
    target.append(data["F378"])
    target.append(data["F395"])
    target.append(data["F410"])
    target.append(data["F430"])
    target.append(data["F480_g_sdss"])
    target.append(data["F515"])
    target.append(data["F610_r_sdss"])
    target.append(data["F660"])
    target.append(data["F760_i_sdss"])
    target.append(data["F861"])
    target.append(data["F910_z_sdss"])
    if data["id"].endswith("HPNe"):
        target.append(0)
    elif data["id"].endswith("sys"):
        target.append(1)
    elif data["id"].endswith("CV"):
        target.append(2)
    else:
        target.append(3)
    
  
XX = np.array(X).reshape(shape)
target_ = np.array(target).reshape(shape1)
print("Data shape:", XX.shape)

m = []
XX = clean_nan_inf(XX)
target_ = clean_nan_inf(target_)
for i in target_:
    m.append(i[12])
m = np.array(m)

print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

pca = PCA(n_components=6)
pca.fit(XX)

XX_pca = pca.transform(XX)

target_names = ["a", "b", "c", "d"]

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'purple']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    plt.scatter(XX_pca[m == i, 0], XX_pca[m == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.show()

# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# for name, label in [('Halo PNe', 0), ('SySt', 1), ('Otros', 2)]:
#     ax.text3D(XX_pca[m == label, 0].mean(),
#               XX_pca[m == label, 1].mean() + 1.5,
#               XX_pca[m == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# y = np.choose(m, [1, 2, 0]).astype(np.float)
# ax.scatter(XX_pca[:, 0], XX_pca[:, 1], XX_pca[:, 2], c=y, cmap=plt.cm.spectral)

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

# plt.show()


#fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, .95, 1])  #rect=[0, 0, .95, 1], elev=elev, azim=azim)

#pcs_to_plot = (0, 1, 2)  # Careful!! 0 corresponds to PC1
# # # pcs_to_plot = (3, 4, 5)  # Careful!! 0 corresponds to PC1

# xx = XX_pca[:, pcs_to_plot[0]]
# yy = XX_pca[:, pcs_to_plot[1]]
# zz = XX_pca[:, pcs_to_plot[2]]

# mask0 =np.array([m == 0, 0 ])
# mask1 =np.array([m == 1, 1 ])
# mask2 =np.array([m == 2, 2 ])


# fig = plt.figure(figsize=(7, 6))
# ax1 = fig.add_subplot(111)
# ax1.scatter(xx[mask0], yy[mask0],  color="black", marker='.')
# ax1.scatter(xx[mask1], yy[mask1],  color="red", marker='.')
# ax1.scatter(xx[mask2], yy[mask2],  color="black", marker='.')
# plt.xlabel('PC{}'.format(pcs_to_plot[0]+1))
# plt.ylabel('PC{}'.format(pcs_to_plot[1]+1))
# plt.grid()
# plt.savefig("PCA1-JPLUS.pdf")
# plt.show()

# xx = XX_pca[:, pcs_to_plot[0]]
# yy = XX_pca[:, pcs_to_plot[1]]
# zz = XX_pca[:, pcs_to_plot[2]]
# ax.scatter(xx, yy, zz, color=(.1, .9, .3), marker='.')
# plt.xlabel('PC{}'.format(pcs_to_plot[0]+1))
# plt.ylabel('PC{}'.format(pcs_to_plot[1]+1))
# ax.set_zlabel('PC{}'.format(pcs_to_plot[2]+1))

# plt.savefig("luis-PCA3D-JPLUS.pdf")

