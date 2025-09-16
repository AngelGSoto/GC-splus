'''
Make color-color diagram for photometryc system available
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


pattern = "*-spectros/*-JPAS15-magnitude.json"
file_list = glob.glob(pattern)

def filter_mag(e, s, f1, f2, f3):
    col, col0 = [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            diff = filter1 - filter2
            diff0 = filter1 - filter3
            col.append(diff)
            col0.append(diff0)
    
    return col, col0

def plot_mag(f1, f2, f3):
    x, y = filter_mag("HPNe", "", f1, f2, f3)
    x1, y1 = filter_mag("catB", "",  f1, f2, f3)
    x2, y2 = filter_mag("E02", "DdDm1_L2", f1, f2, f3)
    x3, y3 = filter_mag("E02", "DdDm1_L3", f1, f2, f3)
    x4, y4 = filter_mag("E02", "DdDm1_L4", f1, f2, f3)
    x5, y5 = filter_mag("E02", "DdDm1_L5", f1, f2, f3)
    x6, y6 = filter_mag("E02", "N2242_L2", f1, f2, f3)
    x7, y7 = filter_mag("E02", "N2242_L3", f1, f2, f3)
    x8, y8 = filter_mag("E02", "N2242_L4", f1, f2, f3)
    x9, y9 = filter_mag("E02", "N2242_L5", f1, f2, f3)
    x10, y10 = filter_mag("E02", "K648_L2", f1, f2, f3)
    x11, y11 = filter_mag("E02", "K648_L3", f1, f2, f3)
    x12, y12 = filter_mag("E02", "K648_L4", f1, f2, f3)
    x13, y13 = filter_mag("E02", "K648_L5", f1, f2, f3)
    x14, y14 = filter_mag("E02", "BB1_L2", f1, f2, f3)
    x15, y15 = filter_mag("E02", "BB1_L3", f1, f2, f3)
    x16, y16 = filter_mag("E02", "BB1_L4", f1, f2, f3)
    x17, y17 = filter_mag("E02", "BB1_L5", f1, f2, f3)
    x18, y18 = filter_mag("E02", "Typ_L2", f1, f2, f3)
    x19, y19 = filter_mag("E02", "Typ_L3", f1, f2, f3)
    x20, y20 = filter_mag("E02", "Typ_L4", f1, f2, f3)
    x21, y21 = filter_mag("E02", "Typ_L5", f1, f2, f3)
    x22, y22 = filter_mag("-DPNe", "",  f1, f2, f3)
    x23, y23 = filter_mag("QSOs-hz", "", f1, f2, f3)
    x24, y24 = filter_mag("QSOs-010", "", f1, f2, f3)
    x25, y25 = filter_mag("QSOs-101", "", f1, f2, f3)
    x26, y26 = filter_mag("QSOs-201", "", f1, f2, f3)
    x27, y27 = filter_mag("QSOs-301", "",  f1, f2, f3)
    x28, y28 = filter_mag("QSOs-401", "",  f1, f2, f3)
    x29, y29 = filter_mag("-SFGs", "", f1, f2, f3)
    x30, y30 = filter_mag("-sys", "", f1, f2, f3)
    x31, y31 = filter_mag("-sys-IPHAS", "", f1, f2, f3) 
    x32, y32 = filter_mag("-ExtHII", "", f1, f2, f3)
    for a, b in zip(x, y):
        d_644.append(a)
        d_768.append(b)
    for a, b in zip(x1, y1):
        d_644_c.append(a)
        d_768_c.append(b)
    for a, b in zip(x2, y2):
        d_644_L2d.append(a)
        d_768_L2d.append(b)
    for a, b in zip(x3, y3):
        d_644_L3d.append(a)
        d_768_L3d.append(b)
    for a, b in zip(x4, y4):
        d_644_L4d.append(a)
        d_768_L4d.append(b)
    for a, b in zip(x5, y5):
        d_644_L5d.append(a)
        d_768_L5d.append(b)
    for a, b in zip(x6, y6):
        d_644_L2N.append(a)
        d_768_L2N.append(b)
    for a, b in zip(x7, y7):
        d_644_L3N.append(a)
        d_768_L3N.append(b)
    for a, b in zip(x8, y8):
        d_644_L4N.append(a)
        d_768_L4N.append(b)
    for a, b in zip(x9, y9):
        d_644_L5N.append(a)
        d_768_L5N.append(b)
    for a, b in zip(x10, y10):
        d_644_L2k.append(a)
        d_768_L2k.append(b)
    for a, b in zip(x11, y11):
        d_644_L3k.append(a)
        d_768_L3k.append(b)
    for a, b in zip(x12, y12):
        d_644_L4k.append(a)
        d_768_L4k.append(b)
    for a, b in zip(x13, y13):
        d_644_L5k.append(a)
        d_768_L5k.append(b)
    for a, b in zip(x14, y14):
        d_644_L2B.append(a)
        d_768_L2B.append(b)
    for a, b in zip(x15, y15):
        d_644_L3B.append(a)
        d_768_L3B.append(b)
    for a, b in zip(x16, y16):
        d_644_L4B.append(a)
        d_768_L4B.append(b)
    for a, b in zip(x17, y17):
        d_644_L5B.append(a)
        d_768_L5B.append(b)
    for a, b in zip(x18, y18):
        d_644_L2T.append(a)
        d_768_L2T.append(b)
    for a, b in zip(x19, y19):
        d_644_L3T.append(a)
        d_768_L3T.append(b)
    for a, b in zip(x20, y20):
        d_644_L4T.append(a)
        d_768_L4T.append(b)
    for a, b in zip(x21, y21):
        d_644_L5T.append(a)
        d_768_L5T.append(b)
    for a, b in zip(x22, y22):
        d_644_CNP.append(a)
        d_768_CNP.append(b)
    for a, b in zip(x23, y23):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x24, y24):
        d_644_Q010.append(a)
        d_768_Q010.append(b)
    for a, b in zip(x25, y25):
        d_644_Q101.append(a)
        d_768_Q101.append(b)
    for a, b in zip(x26, y26):
        d_644_Q201.append(a)
        d_768_Q201.append(b)
    for a, b in zip(x27, y27):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x28, y28):
        d_644_Q401.append(a)
        d_768_Q401.append(b)
    for a, b in zip(x29, y29):
        d_644_SFGs.append(a)
        d_768_SFGs.append(b)
    for a, b in zip(x30, y30):
        d_644_sys.append(a)
        d_768_sys.append(b)
    for a, b in zip(x31, y31):
        d_644_sys_IPHAS.append(a)
        d_768_sys_IPHAS.append(b)
    for a, b in zip(x32, y32):
        d_644_ExtHII.append(a)
        d_768_ExtHII.append(b)


d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
d_644_L2d, d_768_L2d = [], []
d_644_L3d, d_768_L3d = [], []
d_644_L4d, d_768_L4d = [], []
d_644_L5d, d_768_L5d = [], []
d_644_L2N, d_768_L2N = [], []
d_644_L3N, d_768_L3N = [], []
d_644_L4N, d_768_L4N = [], []
d_644_L5N, d_768_L5N = [], []
d_644_L2k, d_768_L2k = [], []
d_644_L3k, d_768_L3k = [], []
d_644_L4k, d_768_L4k = [], []
d_644_L5k, d_768_L5k = [], []
d_644_L2B, d_768_L2B = [], []
d_644_L3B, d_768_L3B = [], []
d_644_L4B, d_768_L4B = [], []
d_644_L5B, d_768_L5B = [], []
d_644_L2T, d_768_L2T = [], []
d_644_L3T, d_768_L3T = [], []
d_644_L4T, d_768_L4T = [], []
d_644_L5T, d_768_L5T = [], []
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

label = []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith("1-HPNe"):
            label.append(data['id'].split("-H")[0])
        elif data['id'].endswith("SLOAN-HPNe"):
            label.append("H4-1")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        plot_mag("Jv0915_6199", "Jv0915_6600", "Jv0915_7700")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-1.7,xmax=2.0)
ax1.set_ylim(ymin=-1.8,ymax=6.0)
ax1.set_xlim(xmin=-0.8,xmax=1.4)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('Jv0915_6199 - Jv0915_7700', size = 12)
plt.ylabel('Jv0915_6199 - Jv0915_6600', size = 12)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d, d_644_L2d,  c= "orange", alpha=0.8, marker='s', label='BB dddm1 L2')
ax1.scatter(d_768_L3d, d_644_L3d,  c= "orange", alpha=0.8, marker='D', label='BB dddm1 L3')
ax1.scatter(d_768_L4d, d_644_L4d,  c= "orange", alpha=0.8, marker='^', label='BB dddm1 L4')
ax1.scatter(d_768_L5d, d_644_L5d,  c= "orange", alpha=0.8, marker='*', label='BB dddm1 L5')
ax1.scatter(d_768_L2N, d_644_L2N,  c= "green", alpha=0.8, marker='s', label='BB N2242 L2')
ax1.scatter(d_768_L3N, d_644_L3N,  c= "green", alpha=0.8, marker='D', label='BB N2242 L3')
ax1.scatter(d_768_L4N, d_644_L4N,  c= "green", alpha=0.8, marker='^', label='BB N2242 L4')
ax1.scatter(d_768_L5N, d_644_L5N,  c= "green", alpha=0.8, marker='*', label='BB N2242 L5')
ax1.scatter(d_768_L2k, d_644_L2k,  c= "brown", alpha=0.8, marker='s', label='BB K648 L2')
ax1.scatter(d_768_L3k, d_644_L3k,  c= "brown", alpha=0.8, marker='D', label='BB K648 L3')
ax1.scatter(d_768_L4k, d_644_L4k,  c= "brown", alpha=0.8, marker='^', label='BB K648 L4')
ax1.scatter(d_768_L5k, d_644_L5k,  c= "brown", alpha=0.8, marker='*', label='BB K648 L5')
ax1.scatter(d_768_L2B, d_644_L2B,  c= "cyan", alpha=0.8, marker='s', label='BB BB1 L2')
ax1.scatter(d_768_L3B, d_644_L3B,  c= "cyan", alpha=0.8, marker='D', label='BB BB1 L3')
ax1.scatter(d_768_L4B, d_644_L4B,  c= "cyan", alpha=0.8, marker='^', label='BB BB1 L4')
ax1.scatter(d_768_L5B, d_644_L5B,  c= "cyan", alpha=0.8, marker='*', label='BB BB1 L5')
ax1.scatter(d_768_L2T, d_644_L2T,  c= "magenta", alpha=0.8, marker='s', label='BB Typ L2')
ax1.scatter(d_768_L3T, d_644_L3T,  c= "magenta", alpha=0.8, marker='D', label='BB Typ L3')
ax1.scatter(d_768_L4T, d_644_L4T,  c= "magenta", alpha=0.8, marker='^', label='BB Typ L4')
ax1.scatter(d_768_L5T, d_644_L5T,  c= "magenta", alpha=0.8, marker='*', label='BB Typ L5')
ax1.scatter(d_768_Q401, d_644_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax1.scatter(d_768_Q201, d_644_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax1.scatter(d_768_Q101, d_644_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax1.scatter(d_768_Q010, d_644_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs')
ax1.scatter(d_768_sys, d_644_sys,  c= "red", alpha=0.8, marker='s', label='Symbiotics')
ax1.scatter(d_768_sys_IPHAS, d_644_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label='Extragalactic HII')
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
ax1.text(0.05, 0.95, 'Extinction, E0.2',
           transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

# ax1.text(0.05, 0.95, 'Extinction, E02',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)


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
ax1.legend(scatterpoints=1, ncol=2, fontsize=6.0, **lgd_kws)
#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPAS-E02.pdf')

d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
d_644_L2d, d_768_L2d = [], []
d_644_L3d, d_768_L3d = [], []
d_644_L4d, d_768_L4d = [], []
d_644_L5d, d_768_L5d = [], []
d_644_L2N, d_768_L2N = [], []
d_644_L3N, d_768_L3N = [], []
d_644_L4N, d_768_L4N = [], []
d_644_L5N, d_768_L5N = [], []
d_644_L2k, d_768_L2k = [], []
d_644_L3k, d_768_L3k = [], []
d_644_L4k, d_768_L4k = [], []
d_644_L5k, d_768_L5k = [], []
d_644_L2B, d_768_L2B = [], []
d_644_L3B, d_768_L3B = [], []
d_644_L4B, d_768_L4B = [], []
d_644_L5B, d_768_L5B = [], []
d_644_L2T, d_768_L2T = [], []
d_644_L3T, d_768_L3T = [], []
d_644_L4T, d_768_L4T = [], []
d_644_L5T, d_768_L5T = [], []
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

label = []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith("1-HPNe"):
            label.append(data['id'].split("-H")[0])
        elif data['id'].endswith("SLOAN-HPNe"):
            label.append("H4-1")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        plot_mag("Jv0915_6600", "Jv0915_5001", "Jv0915_6199")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-1.7,xmax=2.0)
ax1.set_ylim(ymin=-4.5,ymax=2.0)
ax1.set_xlim(xmin=-6.0,xmax=2.5)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('Jv0915_6600 - Jv0915_6199', size = 12)
plt.ylabel('Jv0915_6600 - Jv0915_5001', size = 12)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d, d_644_L2d,  c= "orange", alpha=0.8, marker='s', label='BB dddm1 L2')
ax1.scatter(d_768_L3d, d_644_L3d,  c= "orange", alpha=0.8, marker='D', label='BB dddm1 L3')
ax1.scatter(d_768_L4d, d_644_L4d,  c= "orange", alpha=0.8, marker='^', label='BB dddm1 L4')
ax1.scatter(d_768_L5d, d_644_L5d,  c= "orange", alpha=0.8, marker='*', label='BB dddm1 L5')
ax1.scatter(d_768_L2N, d_644_L2N,  c= "green", alpha=0.8, marker='s', label='BB N2242 L2')
ax1.scatter(d_768_L3N, d_644_L3N,  c= "green", alpha=0.8, marker='D', label='BB N2242 L3')
ax1.scatter(d_768_L4N, d_644_L4N,  c= "green", alpha=0.8, marker='^', label='BB N2242 L4')
ax1.scatter(d_768_L5N, d_644_L5N,  c= "green", alpha=0.8, marker='*', label='BB N2242 L5')
ax1.scatter(d_768_L2k, d_644_L2k,  c= "brown", alpha=0.8, marker='s', label='BB K648 L2')
ax1.scatter(d_768_L3k, d_644_L3k,  c= "brown", alpha=0.8, marker='D', label='BB K648 L3')
ax1.scatter(d_768_L4k, d_644_L4k,  c= "brown", alpha=0.8, marker='^', label='BB K648 L4')
ax1.scatter(d_768_L5k, d_644_L5k,  c= "brown", alpha=0.8, marker='*', label='BB K648 L5')
ax1.scatter(d_768_L2B, d_644_L2B,  c= "cyan", alpha=0.8, marker='s', label='BB BB1 L2')
ax1.scatter(d_768_L3B, d_644_L3B,  c= "cyan", alpha=0.8, marker='D', label='BB BB1 L3')
ax1.scatter(d_768_L4B, d_644_L4B,  c= "cyan", alpha=0.8, marker='^', label='BB BB1 L4')
ax1.scatter(d_768_L5B, d_644_L5B,  c= "cyan", alpha=0.8, marker='*', label='BB BB1 L5')
ax1.scatter(d_768_L2T, d_644_L2T,  c= "magenta", alpha=0.8, marker='s', label='BB Typ L2')
ax1.scatter(d_768_L3T, d_644_L3T,  c= "magenta", alpha=0.8, marker='D', label='BB Typ L3')
ax1.scatter(d_768_L4T, d_644_L4T,  c= "magenta", alpha=0.8, marker='^', label='BB Typ L4')
ax1.scatter(d_768_L5T, d_644_L5T,  c= "magenta", alpha=0.8, marker='*', label='BB Typ L5')
ax1.scatter(d_768_Q401, d_644_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax1.scatter(d_768_Q201, d_644_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax1.scatter(d_768_Q101, d_644_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax1.scatter(d_768_Q010, d_644_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs')
ax1.scatter(d_768_sys, d_644_sys,  c= "red", alpha=0.8, marker='s', label='Symbiotics')
ax1.scatter(d_768_sys_IPHAS, d_644_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label='Extragalactic HII')
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
ax1.text(0.05, 0.95, 'Extinction, E0.2',
           transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)


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
ax1.legend(scatterpoints=1, ncol=3, fontsize=5.0, loc='lower left', **lgd_kws)
#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPAS-Jv0915_6600-E02.pdf')
