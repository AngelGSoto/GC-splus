'''
Make color-color diagram for a photometryc system available
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


pattern = "*-spectros/*-JPLUS13-magnitude.json"
file_list = glob.glob(pattern)

def filter_mag(e, f1, f2, f3):
    col, col0 = [], []
    if data['id'].endswith(e):
        filter1 = data[f1]
        filter2 = data[f2]
        filter3 = data[f3]
        diff = filter1 - filter2
        diff0 = filter1 - filter3
        col.append(diff)
        col0.append(diff0)
    
    return col, col0

def plot_mag(f1, f2, f3):
    x, y = filter_mag("HPNe", f1, f2, f3)
    x1, y1 = filter_mag("CV",  f1, f2, f3)
    x2, y2 = filter_mag("E00", f1, f2, f3)
    x3, y3 = filter_mag("E01",  f1, f2, f3)
    x4, y4 = filter_mag("E02",  f1, f2, f3)
    x5, y5 = filter_mag("-DPNe", f1, f2, f3)
    x6, y6 = filter_mag("QSOs-hz",  f1, f2, f3)
    x7, y7 = filter_mag("QSOs-010",  f1, f2, f3)
    x8, y8 = filter_mag("QSOs-101",  f1, f2, f3)
    x9, y9 = filter_mag("QSOs-201", f1, f2, f3)
    x10, y10 = filter_mag("QSOs-301", f1, f2, f3)
    x11, y11 = filter_mag("QSOs-401", f1, f2, f3)
    x12, y12 = filter_mag("-SFGs",  f1, f2, f3)
    x13, y13 = filter_mag("-sys",  f1, f2, f3)
    x14, y14 = filter_mag("-sys-IPHAS",  f1, f2, f3) 
    x15, y15 = filter_mag("-ExtHII",  f1, f2, f3)
    x16, y16 = filter_mag("-sys-Ext", f1, f2, f3)
    x17, y17 = filter_mag("-survey",  f1, f2, f3)
    x18, y18 = filter_mag("-SNR", f1, f2, f3)
    for a, b in zip(x, y):
        d_644.append(a)
        d_768.append(b)
    for a, b in zip(x1, y1):
        d_644_c.append(a)
        d_768_c.append(b)
    for a, b in zip(x2, y2):
        model_644.append(a)
        model_768.append(b)
    for a, b in zip(x3, y3):
        model_644.append(a)
        model_768.append(b)
    for a, b in zip(x4, y4):
        model_644.append(a)
        model_768.append(b)
    for a, b in zip(x5, y5):
        d_644_CNP.append(a)
        d_768_CNP.append(b)
    for a, b in zip(x6, y6):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x7, y7):
        d_644_Q010.append(a)
        d_768_Q010.append(b)
    for a, b in zip(x8, y8):
        d_644_Q101.append(a)
        d_768_Q101.append(b)
    for a, b in zip(x9, y9):
        d_644_Q201.append(a)
        d_768_Q201.append(b)
    for a, b in zip(x10, y10):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x11, y11):
        d_644_Q401.append(a)
        d_768_Q401.append(b)
    for a, b in zip(x12, y12):
        d_644_SFGs.append(a)
        d_768_SFGs.append(b)
    for a, b in zip(x13, y13):
        d_644_sys.append(a)
        d_768_sys.append(b)
    for a, b in zip(x14, y14):
        d_644_sys_IPHAS.append(a)
        d_768_sys_IPHAS.append(b)
    for a, b in zip(x15, y15):
        d_644_ExtHII.append(a)
        d_768_ExtHII.append(b)
    for a, b in zip(x16, y16):
        d_644_Extsys.append(a)
        d_768_Extsys.append(b)
    for a, b in zip(x17, y17):
        d_644_sysurvey.append(a)
        d_768_sysurvey.append(b)
    for a, b in zip(x18, y18):
        d_644_SN.append(a)
        d_768_SN.append(b)
d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
model_644, model_768 = [], []
d_644_Qz, d_768_Qz = [], []
d_644_cAlh, d_768_cAlh = [], []
d_644_Q010, d_768_Q010 = [], []
d_644_Q101, d_768_Q101 = [], []
d_644_Q201, d_768_Q201 = [], []
d_644_Q401, d_768_Q401 = [], []
d_644_SFGs, d_768_SFGs = [], []
d_644_sys, d_768_sys = [], []
d_644_sys_IPHAS, d_768_sys_IPHAS = [], []
d_644_ExtHII, d_768_ExtHII = [], []
d_644_Extsys, d_768_Extsys = [], []
d_644_sysurvey, d_768_sysurvey = [], []
d_644_SN, d_768_SN = [], []
d_644_jplus, d_768_jplus = [], []
d_644_jplus1, d_768_jplus1 = [], []
label = []
label_HII = []
label_sys = []
label_j =[]

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith("1-HPNe"):
            label.append(data['id'].split("-H")[0])
        elif data['id'].endswith("SLOAN-HPNe"):
            label.append("H4-1")
        elif data['id'].endswith("1359559-HPNe"):
            label.append("PNG 135.9+55.9")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        if data['id'].endswith("-ExtHII"):
            label_HII.append(data['id'].split("b")[0])
        if data['id'].endswith("-sys"):
            label_sys.append(data['id'].split("-sys")[0])
        plot_mag("F610_r_sdss", "F660", "F760_i_sdss")


#Include de magnitudes from JPLUS images
pattern1 = "JPLUS-data/*-phot/*6pix.json"
file_list1 = glob.glob(pattern1)
print(file_list1)
for file_name1 in file_list1:
    with open(file_name1) as f1:
        data1 = json.load(f1)
        label_j.append(data1["id"])
        fil1 = data1["J0625_rSDSS"]
        fil2 = data1["J0660"] 
        fil3 = data1["J0766_iSDSS"] 
        diff = fil1 - fil2
        diff0 = fil1 - fil3
        d_644_jplus.append(diff)
        d_768_jplus.append(diff0)

# pattern1 = "JPLUS-data/PNG135coadded-phot/*6pix.json"
# file_list1 = glob.glob(pattern1)
# for file_name1 in file_list1:
#     with open(file_name1) as f1:
#         data1 = json.load(f1)
#         fil1 = data1["rSDSS"] 
#         fil2 = data1["J0660"]
#         fil3 = data1["iSDSS"]
#         diff = fil1 - fil2
#         diff0 = fil1 - fil3
        #d_644_jplus1.append(diff)
        #d_768_jplus1.append(diff0)

# pattern2 = "JPLUS-data/*phot/*mean.json"
# file_list2 = glob.glob(pattern2)
# for file_name2 in file_list2:
#     with open(file_name2) as f2:
#         data2 = json.load(f2)
#         file1 = data2["rSDSS"] + 1.09
#         file2 = data2["J0660"] + 1.34
#         file3 = data2["iSDSS"] + 1.25
#         diff = file1 - file2
#         diff0 = file1 - file3
        # d_644_jplus.append(diff)
#         d_768_jplus.append(diff0)
# print(d_644_jplus, d_768_jplus)        
     
#Sketch from Viironen        
x = [-0.98, -0.9, -0.57, -0.17, 0.0, 0.30, 0.60, 0.72, 0.89, 0.89, 0.68, 0.59, 0.54, 0.43, 0.30, 0.18, -0.10, -0.28, 
     -0.41, -0.57, -0.85, -1.00, -0.98 ]
y = [2.90, 3.07, 3.39, 3.45, 3.44, 3.40, 3.30, 3.21, 3.08, 2.90, 2.60, 2.41, 2.10, 1.93, 1.83, 1.80, 1.93, 2.03, 
     2.28, 2.41, 2.6, 2.6, 2.90]
circle = plt.Circle((0.63, 1.1), 0.21999999999999997, color='k', alpha=0.1)

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-2.5,xmax=2.0)
ax1.set_ylim(ymin=-1.0,ymax=3.5)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
#ax1.fill(x, y, color= 'k', alpha=0.1)
#ax1.add_artist(circle)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('r  -  (6799-8650)', size = 16)
plt.ylabel('r -  (6524-6675)', size = 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.fill(model_768, model_644,  ec='gray', fc='red', alpha=0.4)
ax1.scatter(d_768_Q401, d_644_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax1.scatter(d_768_Q201, d_644_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax1.scatter(d_768_Q101, d_644_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax1.scatter(d_768_Q010, d_644_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs')
ax1.scatter(d_768_sys, d_644_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax1.scatter(d_768_Extsys, d_644_Extsys,  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax1.scatter(d_768_sys_IPHAS, d_644_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(d_768_sysurvey, d_644_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
ax1.scatter(d_768_SN, d_644_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
ax1.scatter(d_768_jplus, d_644_jplus,  c= "black", alpha=0.8, marker='s', s=35, label='HPNe from JPLUS survey')
#ax1.scatter(d_768_jplus1, d_644_jplus1,  c= "black", alpha=0.8, marker='s', s=35)
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
#ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
           #transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',)
for label_, x, y in zip(label_j, d_768_jplus, d_644_jplus):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(4, 4), textcoords='offset points', ha='left', va='bottom',)
    

#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

# for label_, x, y in zip(label_HII, d_768_ExtHII, d_644_ExtHII):
#     ax1.annotate(label_, (x, y), alpha=0.9, size=8,
#                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)

# plt.annotate(
#     '', xy=(d_768_L2d0[0]+0.3, d_644_L2d0[0]+0.3), xycoords='data',
#     xytext=(d_768_L2d02[0]+0.3, d_644_L2d02[0]+0.3), textcoords='data',
#     arrowprops={'arrowstyle': '<-'})
#     #arrowprops=dict(arrowstyle="<-",
#                         #))
# plt.annotate(
#     '', xy=(d_768_L2d0[0]+0.35, d_644_L2d0[0]+0.35), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')


# for label_, x, y in zip(label_sys, d_768_sys, d_644_sys):
#     ax1.annotate(label_, (x, y), alpha=0.9, size=8,
#                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)


#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=2, fontsize=5.8, loc='lower left', **lgd_kws)
#ax1.grid()
lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPLUS-Viironen-filk.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
