'''
Make color-color diagram to particular filters with diagram of Teresa
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import six

pattern = "*-spectros/*-alh-magnitude.json"

file_list = glob.glob(pattern)

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
d_644_QS, d_768_QS = [], []
d_644_Qz, d_768_Qz = [], []
d_644_sim, d_768_sim = [], []
d_644_st, d_768_st = [], []
d_644_cAlh, d_768_cAlh = [], []
d_644_Q010, d_768_Q010 = [], []
d_644_Q101, d_768_Q101 = [], []
d_644_Q201, d_768_Q201 = [], []
d_644_Q401, d_768_Q401 = [], []


label = []
can_alh = []

def filter_mag(p):
    col, col0 = [], []
    if data['id'].endswith(p) or data['id'].startswith(p):
        F613 = data['F_613_3']
        F644 = data['F_644_3']
        F768 = data['F_768_3']
        diff_644 = F613 - F644
        diff_768 = F613 - F768
        col.append(diff_644)
        col0.append(diff_768)
    
    return col, col0
     
for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith("1-HPNe"):
            label.append(data['id'].split("-H")[0])
        elif data['id'].endswith("SLOANHPNe"):
            label.append("H4-1")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        x, y = filter_mag("HPNe")
        x1, y1 = filter_mag("catB")
        x2, y2 = filter_mag("DdDm1_L2")
        x3, y3 = filter_mag("DdDm1_L3")
        x4, y4 = filter_mag("DdDm1_L4")
        x5, y5 = filter_mag("DdDm1_L5")
        x6, y6 = filter_mag("N2242_L2")
        x7, y7 = filter_mag("N2242_L3")
        x8, y8 = filter_mag("N2242_L4")
        x9, y9 = filter_mag("N2242_L5")
        x10, y10 = filter_mag("K648_L2")
        x11, y11 = filter_mag("K648_L3")
        x12, y12 = filter_mag("K648_L4")
        x13, y13 = filter_mag("K648_L5")
        x14, y14 = filter_mag("BB1_L2")
        x15, y15 = filter_mag("BB1_L3")
        x16, y16 = filter_mag("BB1_L4")
        x17, y17 = filter_mag("BB1_L5")
        x18, y18 = filter_mag("Typ_L2")
        x19, y19 = filter_mag("Typ_L3")
        x20, y20 = filter_mag("Typ_L4")
        x21, y21 = filter_mag("Typ_L5")
        x22, y22 = filter_mag("-C-PNe")
        x23, y23 = filter_mag("QSOs-hz")
        x24, y24 = filter_mag("QSOs-010")
        x25, y25 = filter_mag("QSOs-101")
        x26, y26 = filter_mag("QSOs-201")
        x27, y27 = filter_mag("QSOs-401")
      
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
            d_644_Q401.append(a)
            d_768_Q401.append(b)

f = open('mag_simbioticas_Munari', 'r')

header1 = f.readline()
header2 = f.readline()
header3 = f.readline() 
header4 = f.readline()
header5 = f.readline()
header6 = f.readline() 
header7 = f.readline()
header8 = f.readline()
header9 = f.readline() 
header10 = f.readline()
header11= f.readline()
header12= f.readline() 

for line in f:
    line = line.strip()
    columns = line.split()
    mag1 = float(columns[7])
    mag2 = float(columns[8])
    mag3 = float(columns[12])
    d_mag1 = mag1 - mag2
    d_mag2 = mag1 - mag3
    d_644_sim.append(d_mag1)
    d_768_sim.append(d_mag2)

f = open('mag_star_forming_galaxiesSDSS', 'r')
header1 = f.readline()

for line in f:
    line = line.strip()
    columns = line.split()
    mag1 = float(columns[7])
    mag2 = float(columns[8])
    mag3 = float(columns[12])
    d_mag1 = mag1 - mag2
    d_mag2 = mag1 - mag3
    d_644_st.append(d_mag1)
    d_768_st.append(d_mag2)

f = open('magalh_swireQSO', 'r')

header1 = f.readline()
header2 = f.readline()
header3 = f.readline() 
header4 = f.readline()
header5 = f.readline()
header6 = f.readline() 
header7 = f.readline()
header8 = f.readline()
header9 = f.readline() 
header10 = f.readline()
header11= f.readline()
header12= f.readline() 
header13 = f.readline()
header14 = f.readline()
header15 = f.readline() 
header16= f.readline()
header17 = f.readline()
header18 = f.readline() 
header19 = f.readline()
header20 = f.readline()
header21 = f.readline() 
header22 = f.readline()
header23 = f.readline()
header24 = f.readline() 
header26 = f.readline()
header27 = f.readline()
header28 = f.readline() 

for line in f:
    line = line.strip()
    columns = line.split()
    mag1 = float(columns[7])
    mag2 = float(columns[8])
    mag3 = float(columns[12])
    d_mag1 = mag1 - mag2
    d_mag2 = mag1 - mag3
    d_644_QS.append(d_mag1)
    d_768_QS.append(d_mag2)

f = open('Alhambra-candidates/list-candidates.txt', 'r')
header1 = f.readline()
header2 = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    mag1 = float(columns[23])
    mag2 = float(columns[25])
    mag3 = float(columns[33])
    d_mag1 = mag1 - mag2
    d_mag2 = mag1 - mag3
    d_644_cAlh.append(d_mag1)
    d_768_cAlh.append(d_mag2)
    can_alh.append(columns[0].split('0')[-1])

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark") #context="talk")
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-0.6,xmax=1.4)
ax1.set_ylim(ymin=-1.5,ymax=5.0)
plt.xlabel('F613W - F768W', size = 12)
plt.ylabel('F613W - F644W', size = 12)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.5, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d, d_644_L2d,  c= "orange", alpha=0.8, marker='s', label='BB DdDm-1 L2')
ax1.scatter(d_768_L3d, d_644_L3d,  c= "orange", alpha=0.8, marker='D', label='BB DdDm-1 L3')
ax1.scatter(d_768_L4d, d_644_L4d,  c= "orange", alpha=0.8, marker='^', label='BB DdDm-1 L4')
ax1.scatter(d_768_L5d, d_644_L5d,  c= "orange", alpha=0.8, marker='*', label='BB DdDm-1 L5')
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
#ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='s', label='QSOs (3.06<z<3.29)')
#ax1.scatter(d_768_QS, d_644_QS,  c= "royalblue", alpha=0.8, marker='D', label='QSOs')
ax1.scatter(d_768_Q401, d_644_Q401,  c= "royalblue", alpha=0.8, marker='s',  label='QSOs (4.01<z<4.11)')
ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.06<z<3.29)')
ax1.scatter(d_768_Q201, d_644_Q201,  c= "royalblue", alpha=0.8, marker='^',  label='QSOs (2.01<z<2.11)')
ax1.scatter(d_768_Q101, d_644_Q101,  c= "royalblue", alpha=0.8, marker='*',  label='QSOs (1.01<z<1.11)')
ax1.scatter(d_768_Q010, d_644_Q010,  c= "royalblue", alpha=0.8, marker='o',  label='QSOs (0.01<z<0.11)')
ax1.scatter(d_768_sim, d_644_sim,  c= "red", alpha=0.8, marker='s', label='Symbiotics')
ax1.scatter(d_768_st, d_644_st,  c= "linen", alpha=0.3, marker='^', label='SF galaxies')
ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA candidates')
ax1.text(0.05, 0.95, 'Extinction, E0.1',
           transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(-3,3), textcoords='offset points', ha='left', va='bottom',)
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
ax1.text(0.05, 0.95, 'Extinction, E0.1',
           transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(3,-8), textcoords='offset points', ha='left', va='bottom',)
ax1.minorticks_on()
ax1.grid(which='minor', lw=0.3)
ax1.legend(scatterpoints=1, ncol=2, fontsize='x-small', **lgd_kws)
#ax1.grid()
sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('col-ALHAMBRA.pdf')
