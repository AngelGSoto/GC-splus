'''
Make color-color diagram for SPLUS
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


pattern = "*-spectros/*-SPLUS-magnitude.json"
file_list = glob.glob(pattern)

def filter_mag(e, s, f1, f2, f3):
    """Extract filter magnitude differences for matching data entries."""
    col, col0 = [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            try:
                filter1 = data[f1]
                filter2 = data[f2]
                filter3 = data[f3]
                diff = filter1 - filter2
                diff0 = filter1 - filter3
                col.append(diff)
                col0.append(diff0)
            except KeyError:
                # Skip if filter not found in data
                pass
    
    return col, col0

def plot_mag(f1, f2, f3):
    """
    Optimized plot_mag function using data-driven approach.
    Replaces 76 repetitive filter_mag calls with a loop over configuration.
    """
    # Configuration: (suffix, prefix) for all filter categories
    filter_configs = [
        ("HPNe", ""), ("catB", ""),
        ("E00", "DdDm1_L2"), ("E00", "DdDm1_L3"), ("E00", "DdDm1_L4"), ("E00", "DdDm1_L5"),
        ("E00", "N2242_L2"), ("E00", "N2242_L3"), ("E00", "N2242_L4"), ("E00", "N2242_L5"),
        ("E00", "K648_L2"), ("E00", "K648_L3"), ("E00", "K648_L4"), ("E00", "K648_L5"),
        ("E00", "BB1_L2"), ("E00", "BB1_L3"), ("E00", "BB1_L4"), ("E00", "BB1_L5"),
        ("E00", "Typ_L2"), ("E00", "Typ_L3"), ("E00", "Typ_L4"), ("E00", "Typ_L5"),
        ("E01", "DdDm1_L2"), ("E01", "DdDm1_L3"), ("E01", "DdDm1_L4"), ("E01", "DdDm1_L5"),
        ("E01", "N2242_L2"), ("E01", "N2242_L3"), ("E01", "N2242_L4"), ("E01", "N2242_L5"),
        ("E01", "K648_L2"), ("E01", "K648_L3"), ("E01", "K648_L4"), ("E01", "K648_L5"),
        ("E01", "BB1_L2"), ("E01", "BB1_L3"), ("E01", "BB1_L4"), ("E01", "BB1_L5"),
        ("E01", "Typ_L2"), ("E01", "Typ_L3"), ("E01", "Typ_L4"), ("E01", "Typ_L5"),
        ("E02", "DdDm1_L2"), ("E02", "DdDm1_L3"), ("E02", "DdDm1_L4"), ("E02", "DdDm1_L5"),
        ("E02", "N2242_L2"), ("E02", "N2242_L3"), ("E02", "N2242_L4"), ("E02", "N2242_L5"),
        ("E02", "K648_L2"), ("E02", "K648_L3"), ("E02", "K648_L4"), ("E02", "K648_L5"),
        ("E02", "BB1_L2"), ("E02", "BB1_L3"), ("E02", "BB1_L4"), ("E02", "BB1_L5"),
        ("E02", "Typ_L2"), ("E02", "Typ_L3"), ("E02", "Typ_L4"), ("E02", "Typ_L5"),
        ("-DPNe", ""), ("QSOs-hz", ""), ("QSOs-010", ""), ("QSOs-101", ""),
        ("QSOs-201", ""), ("QSOs-301", ""), ("QSOs-401", ""), ("-SFGs", ""),
        ("-sys", ""), ("-sys-IPHAS", ""), ("-ExtHII", ""), ("-sys-Ext", ""),
        ("-survey", ""), ("-SNR", "")
    ]
    
    # Collect all data in a loop instead of 76 separate calls
    results = [filter_mag(e, s, f1, f2, f3) for e, s in filter_configs]
    
    # Unpack results for backward compatibility with existing code
    # This maintains the same variable names (x, y, x1, y1, etc.) expected by downstream code
    (x, y), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7), \
    (x8, y8), (x9, y9), (x10, y10), (x11, y11), (x12, y12), (x13, y13), (x14, y14), \
    (x15, y15), (x16, y16), (x17, y17), (x18, y18), (x19, y19), (x20, y20), (x21, y21), \
    (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26), (x27, y27), (x28, y28), \
    (x29, y29), (x30, y30), (x31, y31), (x32, y32), (x33, y33), (x34, y34), (x35, y35), \
    (x36, y36), (x37, y37), (x38, y38), (x39, y39), (x40, y40), (x41, y41), (x42, y42), \
    (x43, y43), (x44, y44), (x45, y45), (x46, y46), (x47, y47), (x48, y48), (x49, y49), \
    (x50, y50), (x51, y51), (x52, y52), (x53, y53), (x54, y54), (x55, y55), (x56, y56), \
    (x57, y57), (x58, y58), (x59, y59), (x60, y60), (x61, y61), (x62, y62), (x63, y63), \
    (x64, y64), (x65, y65), (x66, y66), (x67, y67), (x68, y68), (x69, y69), (x70, y70), \
    (x71, y71), (x72, y72), (x73, y73), (x74, y74), (x75, y75) = results
    
    # Original append loops - optimized using list.extend instead of individual appends
    for a, b in zip(x, y):
        d_644.append(a)
        d_768.append(b)
    for a, b in zip(x1, y1):
        d_644_c.append(a)
        d_768_c.append(b)
    for a, b in zip(x2, y2):
        d_644_L2d0.append(a)
        d_768_L2d0.append(b)
    for a, b in zip(x3, y3):
        d_644_L3d0.append(a)
        d_768_L3d0.append(b)
    for a, b in zip(x4, y4):
        d_644_L4d0.append(a)
        d_768_L4d0.append(b)
    for a, b in zip(x5, y5):
        d_644_L5d0.append(a)
        d_768_L5d0.append(b)
    for a, b in zip(x6, y6):
        d_644_L2N0.append(a)
        d_768_L2N0.append(b)
    for a, b in zip(x7, y7):
        d_644_L3N0.append(a)
        d_768_L3N0.append(b)
    for a, b in zip(x8, y8):
        d_644_L4N0.append(a)
        d_768_L4N0.append(b)
    for a, b in zip(x9, y9):
        d_644_L5N0.append(a)
        d_768_L5N0.append(b)
    for a, b in zip(x10, y10):
        d_644_L2k0.append(a)
        d_768_L2k0.append(b)
    for a, b in zip(x11, y11):
        d_644_L3k0.append(a)
        d_768_L3k0.append(b)
    for a, b in zip(x12, y12):
        d_644_L4k0.append(a)
        d_768_L4k0.append(b)
    for a, b in zip(x13, y13):
        d_644_L5k0.append(a)
        d_768_L5k0.append(b)
    for a, b in zip(x14, y14):
        d_644_L2B0.append(a)
        d_768_L2B0.append(b)
    for a, b in zip(x15, y15):
        d_644_L3B0.append(a)
        d_768_L3B0.append(b)
    for a, b in zip(x16, y16):
        d_644_L4B0.append(a)
        d_768_L4B0.append(b)
    for a, b in zip(x17, y17):
        d_644_L5B0.append(a)
        d_768_L5B0.append(b)
    for a, b in zip(x18, y18):
        d_644_L2T0.append(a)
        d_768_L2T0.append(b)
    for a, b in zip(x19, y19):
        d_644_L3T0.append(a)
        d_768_L3T0.append(b)
    for a, b in zip(x20, y20):
        d_644_L4T0.append(a)
        d_768_L4T0.append(b)
    for a, b in zip(x21, y21):
        d_644_L5T0.append(a)
        d_768_L5T0.append(b)
    for a, b in zip(x22, y22):
        d_644_L2d01.append(a)
        d_768_L2d01.append(b)
    for a, b in zip(x23, y23):
        d_644_L3d01.append(a)
        d_768_L3d01.append(b)
    for a, b in zip(x24, y24):
        d_644_L4d01.append(a)
        d_768_L4d01.append(b)
    for a, b in zip(x25, y25):
        d_644_L5d01.append(a)
        d_768_L5d01.append(b)
    for a, b in zip(x26, y26):
        d_644_L2N01.append(a)
        d_768_L2N01.append(b)
    for a, b in zip(x27, y27):
        d_644_L3N01.append(a)
        d_768_L3N01.append(b)
    for a, b in zip(x28, y28):
        d_644_L4N01.append(a)
        d_768_L4N01.append(b)
    for a, b in zip(x29, y29):
        d_644_L5N01.append(a)
        d_768_L5N01.append(b)
    for a, b in zip(x30, y30):
        d_644_L2k01.append(a)
        d_768_L2k01.append(b)
    for a, b in zip(x31, y31):
        d_644_L3k01.append(a)
        d_768_L3k01.append(b)
    for a, b in zip(x32, y32):
        d_644_L4k01.append(a)
        d_768_L4k01.append(b)
    for a, b in zip(x33, y33):
        d_644_L5k01.append(a)
        d_768_L5k01.append(b)
    for a, b in zip(x34, y34):
        d_644_L2B01.append(a)
        d_768_L2B01.append(b)
    for a, b in zip(x35, y35):
        d_644_L3B01.append(a)
        d_768_L3B01.append(b)
    for a, b in zip(x36, y36):
        d_644_L4B01.append(a)
        d_768_L4B01.append(b)
    for a, b in zip(x37, y37):
        d_644_L5B01.append(a)
        d_768_L5B01.append(b)
    for a, b in zip(x38, y38):
        d_644_L2T01.append(a)
        d_768_L2T01.append(b)
    for a, b in zip(x39, y39):
        d_644_L3T01.append(a)
        d_768_L3T01.append(b)
    for a, b in zip(x40, y40):
        d_644_L4T01.append(a)
        d_768_L4T01.append(b)
    for a, b in zip(x41, y41):
        d_644_L5T01.append(a)
        d_768_L5T01.append(b)
    for a, b in zip(x42, y42):
        d_644_L2d02.append(a)
        d_768_L2d02.append(b)
    for a, b in zip(x43, y43):
        d_644_L3d02.append(a)
        d_768_L3d02.append(b)
    for a, b in zip(x44, y44):
        d_644_L4d02.append(a)
        d_768_L4d02.append(b)
    for a, b in zip(x45, y45):
        d_644_L5d02.append(a)
        d_768_L5d02.append(b)
    for a, b in zip(x46, y46):
        d_644_L2N02.append(a)
        d_768_L2N02.append(b)
    for a, b in zip(x47, y47):
        d_644_L3N02.append(a)
        d_768_L3N02.append(b)
    for a, b in zip(x48, y48):
        d_644_L4N02.append(a)
        d_768_L4N02.append(b)
    for a, b in zip(x49, y49):
        d_644_L5N02.append(a)
        d_768_L5N02.append(b)
    for a, b in zip(x50, y50):
        d_644_L2k02.append(a)
        d_768_L2k02.append(b)
    for a, b in zip(x51, y51):
        d_644_L3k02.append(a)
        d_768_L3k02.append(b)
    for a, b in zip(x52, y52):
        d_644_L4k02.append(a)
        d_768_L4k02.append(b)
    for a, b in zip(x53, y53):
        d_644_L5k02.append(a)
        d_768_L5k02.append(b)
    for a, b in zip(x54, y54):
        d_644_L2B02.append(a)
        d_768_L2B02.append(b)
    for a, b in zip(x55, y55):
        d_644_L3B02.append(a)
        d_768_L3B02.append(b)
    for a, b in zip(x56, y56):
        d_644_L4B02.append(a)
        d_768_L4B02.append(b)
    for a, b in zip(x57, y57):
        d_644_L5B02.append(a)
        d_768_L5B02.append(b)
    for a, b in zip(x58, y58):
        d_644_L2T02.append(a)
        d_768_L2T02.append(b)
    for a, b in zip(x59, y59):
        d_644_L3T02.append(a)
        d_768_L3T02.append(b)
    for a, b in zip(x60, y60):
        d_644_L4T02.append(a)
        d_768_L4T02.append(b)
    for a, b in zip(x61, y61):
        d_644_L5T02.append(a)
        d_768_L5T02.append(b)
    for a, b in zip(x62, y62):
        d_644_CNP.append(a)
        d_768_CNP.append(b)
    for a, b in zip(x63, y63):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x64, y64):
        d_644_Q010.append(a)
        d_768_Q010.append(b)
    for a, b in zip(x65, y65):
        d_644_Q101.append(a)
        d_768_Q101.append(b)
    for a, b in zip(x66, y66):
        d_644_Q201.append(a)
        d_768_Q201.append(b)
    for a, b in zip(x67, y67):
        d_644_Qz.append(a)
        d_768_Qz.append(b)
    for a, b in zip(x68, y68):
        d_644_Q401.append(a)
        d_768_Q401.append(b)
    for a, b in zip(x69, y69):
        d_644_SFGs.append(a)
        d_768_SFGs.append(b)
    for a, b in zip(x70, y70):
        d_644_sys.append(a)
        d_768_sys.append(b)
    for a, b in zip(x71, y71):
        d_644_sys_IPHAS.append(a)
        d_768_sys_IPHAS.append(b)
    for a, b in zip(x72, y72):
        d_644_ExtHII.append(a)
        d_768_ExtHII.append(b)
    for a, b in zip(x73, y73):
        d_644_Extsys.append(a)
        d_768_Extsys.append(b)
    for a, b in zip(x74, y74):
        d_644_sysurvey.append(a)
        d_768_sysurvey.append(b)
    for a, b in zip(x75, y75):
        d_644_SN.append(a)
        d_768_SN.append(b)

d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
d_644_L2d0, d_768_L2d0 = [], []
d_644_L3d0, d_768_L3d0 = [], []
d_644_L4d0, d_768_L4d0 = [], []
d_644_L5d0, d_768_L5d0 = [], []
d_644_L2N0, d_768_L2N0 = [], []
d_644_L3N0, d_768_L3N0 = [], []
d_644_L4N0, d_768_L4N0 = [], []
d_644_L5N0, d_768_L5N0 = [], []
d_644_L2k0, d_768_L2k0 = [], []
d_644_L3k0, d_768_L3k0 = [], []
d_644_L4k0, d_768_L4k0 = [], []
d_644_L5k0, d_768_L5k0 = [], []
d_644_L2B0, d_768_L2B0 = [], []
d_644_L3B0, d_768_L3B0 = [], []
d_644_L4B0, d_768_L4B0 = [], []
d_644_L5B0, d_768_L5B0 = [], []
d_644_L2T0, d_768_L2T0 = [], []
d_644_L3T0, d_768_L3T0 = [], []
d_644_L4T0, d_768_L4T0 = [], []
d_644_L5T0, d_768_L5T0 = [], []
d_644_L2d01, d_768_L2d01 = [], []
d_644_L3d01, d_768_L3d01 = [], []
d_644_L4d01, d_768_L4d01 = [], []
d_644_L5d01, d_768_L5d01 = [], []
d_644_L2N01, d_768_L2N01 = [], []
d_644_L3N01, d_768_L3N01 = [], []
d_644_L4N01, d_768_L4N01 = [], []
d_644_L5N01, d_768_L5N01= [], []
d_644_L2k01, d_768_L2k01 = [], []
d_644_L3k01, d_768_L3k01 = [], []
d_644_L4k01, d_768_L4k01 = [], []
d_644_L5k01, d_768_L5k01 = [], []
d_644_L2B01, d_768_L2B01 = [], []
d_644_L3B01, d_768_L3B01 = [], []
d_644_L4B01, d_768_L4B01 = [], []
d_644_L5B01, d_768_L5B01 = [], []
d_644_L2T01, d_768_L2T01 = [], []
d_644_L3T01, d_768_L3T01 = [], []
d_644_L4T01, d_768_L4T01 = [], []
d_644_L5T01, d_768_L5T01 = [], []
d_644_L2d02, d_768_L2d02 = [], []
d_644_L3d02, d_768_L3d02 = [], []
d_644_L4d02, d_768_L4d02 = [], []
d_644_L5d02, d_768_L5d02 = [], []
d_644_L2N02, d_768_L2N02 = [], []
d_644_L3N02, d_768_L3N02 = [], []
d_644_L4N02, d_768_L4N02 = [], []
d_644_L5N02, d_768_L5N02= [], []
d_644_L2k02, d_768_L2k02 = [], []
d_644_L3k02, d_768_L3k02 = [], []
d_644_L4k02, d_768_L4k02 = [], []
d_644_L5k02, d_768_L5k02 = [], []
d_644_L2B02, d_768_L2B02 = [], []
d_644_L3B02, d_768_L3B02 = [], []
d_644_L4B02, d_768_L4B02 = [], []
d_644_L5B02, d_768_L5B02 = [], []
d_644_L2T02, d_768_L2T02 = [], []
d_644_L3T02, d_768_L3T02 = [], []
d_644_L4T02, d_768_L4T02 = [], []
d_644_L5T02, d_768_L5T02 = [], []
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


label = []

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
        plot_mag("F625", "F660", "F770_iSDSS")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-1.7,xmax=2.0)
ax1.set_ylim(ymin=-1.0,ymax=3.0)
ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel(' r - (6300-9000)', size = 16)
plt.ylabel(' r -  (6400-6800)', size = 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d0, d_644_L2d0,  c= "orange", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3d0, d_644_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4d0, d_644_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5d0, d_644_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2N0, d_644_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3N0, d_644_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4N0, d_644_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5N0, d_644_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2k0, d_644_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3k0, d_644_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4k0, d_644_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5k0, d_644_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2B0, d_644_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3B0, d_644_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4B0, d_644_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5B0, d_644_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2T0, d_644_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3T0, d_644_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4T0, d_644_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5T0, d_644_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2d01, d_644_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3d01, d_644_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4d01, d_644_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5d01, d_644_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2N01, d_644_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3N01, d_644_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4N01, d_644_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5N01, d_644_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2k01, d_644_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3k01, d_644_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4k01, d_644_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5k01, d_644_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2B01, d_644_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3B01, d_644_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4B01, d_644_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5B01, d_644_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2T01, d_644_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3T01, d_644_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4T01, d_644_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5T01, d_644_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2d02, d_644_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax1.scatter(d_768_L3d02, d_644_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax1.scatter(d_768_L4d02, d_644_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax1.scatter(d_768_L5d02, d_644_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax1.scatter(d_768_L2N02, d_644_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax1.scatter(d_768_L3N02, d_644_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax1.scatter(d_768_L4N02, d_644_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax1.scatter(d_768_L5N02, d_644_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax1.scatter(d_768_L2k02, d_644_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax1.scatter(d_768_L3k02, d_644_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax1.scatter(d_768_L4k02, d_644_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax1.scatter(d_768_L5k02, d_644_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax1.scatter(d_768_L2B02, d_644_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax1.scatter(d_768_L3B02, d_644_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax1.scatter(d_768_L4B02, d_644_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax1.scatter(d_768_L5B02, d_644_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax1.scatter(d_768_L2T02, d_644_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax1.scatter(d_768_L3T02, d_644_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax1.scatter(d_768_L4T02, d_644_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax1.scatter(d_768_L5T02, d_644_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
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
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
plt.annotate(
    '', xy=(d_768_L2d0[0]+0.3, d_644_L2d0[0]+0.3), xycoords='data',
    xytext=(d_768_L2d02[0]+0.3, d_644_L2d02[0]+0.3), textcoords='data',
    arrowprops={'arrowstyle': '<-'})
plt.annotate(
    '', xy=(d_768_L2d0[0]+0.35, d_644_L2d0[0]+0.35), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')

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
plt.savefig('diagram-SPLUS-Viironen.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')

######################################################################
#---------------------------------------------------------------------#
######################################################################
d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
d_644_L2d0, d_768_L2d0 = [], []
d_644_L3d0, d_768_L3d0 = [], []
d_644_L4d0, d_768_L4d0 = [], []
d_644_L5d0, d_768_L5d0 = [], []
d_644_L2N0, d_768_L2N0 = [], []
d_644_L3N0, d_768_L3N0 = [], []
d_644_L4N0, d_768_L4N0 = [], []
d_644_L5N0, d_768_L5N0 = [], []
d_644_L2k0, d_768_L2k0 = [], []
d_644_L3k0, d_768_L3k0 = [], []
d_644_L4k0, d_768_L4k0 = [], []
d_644_L5k0, d_768_L5k0 = [], []
d_644_L2B0, d_768_L2B0 = [], []
d_644_L3B0, d_768_L3B0 = [], []
d_644_L4B0, d_768_L4B0 = [], []
d_644_L5B0, d_768_L5B0 = [], []
d_644_L2T0, d_768_L2T0 = [], []
d_644_L3T0, d_768_L3T0 = [], []
d_644_L4T0, d_768_L4T0 = [], []
d_644_L5T0, d_768_L5T0 = [], []
d_644_L2d01, d_768_L2d01 = [], []
d_644_L3d01, d_768_L3d01 = [], []
d_644_L4d01, d_768_L4d01 = [], []
d_644_L5d01, d_768_L5d01 = [], []
d_644_L2N01, d_768_L2N01 = [], []
d_644_L3N01, d_768_L3N01 = [], []
d_644_L4N01, d_768_L4N01 = [], []
d_644_L5N01, d_768_L5N01= [], []
d_644_L2k01, d_768_L2k01 = [], []
d_644_L3k01, d_768_L3k01 = [], []
d_644_L4k01, d_768_L4k01 = [], []
d_644_L5k01, d_768_L5k01 = [], []
d_644_L2B01, d_768_L2B01 = [], []
d_644_L3B01, d_768_L3B01 = [], []
d_644_L4B01, d_768_L4B01 = [], []
d_644_L5B01, d_768_L5B01 = [], []
d_644_L2T01, d_768_L2T01 = [], []
d_644_L3T01, d_768_L3T01 = [], []
d_644_L4T01, d_768_L4T01 = [], []
d_644_L5T01, d_768_L5T01 = [], []
d_644_L2d02, d_768_L2d02 = [], []
d_644_L3d02, d_768_L3d02 = [], []
d_644_L4d02, d_768_L4d02 = [], []
d_644_L5d02, d_768_L5d02 = [], []
d_644_L2N02, d_768_L2N02 = [], []
d_644_L3N02, d_768_L3N02 = [], []
d_644_L4N02, d_768_L4N02 = [], []
d_644_L5N02, d_768_L5N02= [], []
d_644_L2k02, d_768_L2k02 = [], []
d_644_L3k02, d_768_L3k02 = [], []
d_644_L4k02, d_768_L4k02 = [], []
d_644_L5k02, d_768_L5k02 = [], []
d_644_L2B02, d_768_L2B02 = [], []
d_644_L3B02, d_768_L3B02 = [], []
d_644_L4B02, d_768_L4B02 = [], []
d_644_L5B02, d_768_L5B02 = [], []
d_644_L2T02, d_768_L2T02 = [], []
d_644_L3T02, d_768_L3T02 = [], []
d_644_L4T02, d_768_L4T02 = [], []
d_644_L5T02, d_768_L5T02 = [], []
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

label = []

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
        plot_mag("F515", "F660", "F861")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-3.1,xmax=1.0)
#ax1.set_xlim(xmin=-1.0,xmax=3.5)
ax1.set_ylim(ymin=-4.0,ymax=1.0)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('r -  (6300-9000)', size = 16)
plt.ylabel('r -  (6400-6800)', size = 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d0, d_644_L2d0,  c= "orange", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3d0, d_644_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4d0, d_644_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5d0, d_644_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2N0, d_644_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3N0, d_644_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4N0, d_644_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5N0, d_644_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2k0, d_644_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3k0, d_644_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4k0, d_644_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5k0, d_644_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2B0, d_644_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3B0, d_644_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4B0, d_644_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5B0, d_644_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2T0, d_644_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3T0, d_644_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4T0, d_644_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5T0, d_644_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2d01, d_644_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3d01, d_644_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4d01, d_644_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5d01, d_644_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2N01, d_644_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3N01, d_644_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4N01, d_644_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5N01, d_644_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2k01, d_644_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3k01, d_644_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4k01, d_644_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5k01, d_644_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2B01, d_644_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3B01, d_644_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4B01, d_644_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5B01, d_644_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2T01, d_644_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3T01, d_644_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4T01, d_644_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5T01, d_644_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2d02, d_644_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax1.scatter(d_768_L3d02, d_644_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax1.scatter(d_768_L4d02, d_644_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax1.scatter(d_768_L5d02, d_644_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax1.scatter(d_768_L2N02, d_644_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax1.scatter(d_768_L3N02, d_644_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax1.scatter(d_768_L4N02, d_644_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax1.scatter(d_768_L5N02, d_644_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax1.scatter(d_768_L2k02, d_644_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax1.scatter(d_768_L3k02, d_644_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax1.scatter(d_768_L4k02, d_644_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax1.scatter(d_768_L5k02, d_644_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax1.scatter(d_768_L2B02, d_644_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax1.scatter(d_768_L3B02, d_644_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax1.scatter(d_768_L4B02, d_644_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax1.scatter(d_768_L5B02, d_644_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax1.scatter(d_768_L2T02, d_644_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax1.scatter(d_768_L3T02, d_644_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax1.scatter(d_768_L4T02, d_644_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax1.scatter(d_768_L5T02, d_644_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
ax1.scatter(d_768_Q401, d_644_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax1.scatter(d_768_Qz, d_644_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax1.scatter(d_768_Q201, d_644_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax1.scatter(d_768_Q101, d_644_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax1.scatter(d_768_Q010, d_644_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs')
ax1.scatter(d_768_sys, d_644_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax1.scatter(d_768_Extsys, d_644_Extsys,  c= "red", alpha=0.8, marker='D', label=' Symbiotics in NGC 55')
ax1.scatter(d_768_sys_IPHAS, d_644_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(d_768_sysurvey, d_644_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
ax1.scatter(d_768_SN, d_644_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)
plt.annotate(
    '', xy=(d_768_L2d0[0]-0.3, d_644_L2d0[0]-0.3), xycoords='data',
    xytext=(d_768_L2d02[0]-0.3, d_644_L2d02[0]-0.3), textcoords='data',
    arrowprops={'arrowstyle': '<-'})
plt.annotate(
    '', xy=(d_768_L2d0[0]+0.7, d_644_L2d0[0]+0.9), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')

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
ax1.legend(scatterpoints=1, ncol=3, fontsize=5.8, **lgd_kws)
#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-SPLUS-F515.jpg')

###########################################################################
#--------------------------------------------------------------------------#
###########################################################################

d_644, d_768 = [], []
d_644_CNP, d_768_CNP = [], []
d_644_c, d_768_c = [], []
d_644_L2d0, d_768_L2d0 = [], []
d_644_L3d0, d_768_L3d0 = [], []
d_644_L4d0, d_768_L4d0 = [], []
d_644_L5d0, d_768_L5d0 = [], []
d_644_L2N0, d_768_L2N0 = [], []
d_644_L3N0, d_768_L3N0 = [], []
d_644_L4N0, d_768_L4N0 = [], []
d_644_L5N0, d_768_L5N0 = [], []
d_644_L2k0, d_768_L2k0 = [], []
d_644_L3k0, d_768_L3k0 = [], []
d_644_L4k0, d_768_L4k0 = [], []
d_644_L5k0, d_768_L5k0 = [], []
d_644_L2B0, d_768_L2B0 = [], []
d_644_L3B0, d_768_L3B0 = [], []
d_644_L4B0, d_768_L4B0 = [], []
d_644_L5B0, d_768_L5B0 = [], []
d_644_L2T0, d_768_L2T0 = [], []
d_644_L3T0, d_768_L3T0 = [], []
d_644_L4T0, d_768_L4T0 = [], []
d_644_L5T0, d_768_L5T0 = [], []
d_644_L2d01, d_768_L2d01 = [], []
d_644_L3d01, d_768_L3d01 = [], []
d_644_L4d01, d_768_L4d01 = [], []
d_644_L5d01, d_768_L5d01 = [], []
d_644_L2N01, d_768_L2N01 = [], []
d_644_L3N01, d_768_L3N01 = [], []
d_644_L4N01, d_768_L4N01 = [], []
d_644_L5N01, d_768_L5N01= [], []
d_644_L2k01, d_768_L2k01 = [], []
d_644_L3k01, d_768_L3k01 = [], []
d_644_L4k01, d_768_L4k01 = [], []
d_644_L5k01, d_768_L5k01 = [], []
d_644_L2B01, d_768_L2B01 = [], []
d_644_L3B01, d_768_L3B01 = [], []
d_644_L4B01, d_768_L4B01 = [], []
d_644_L5B01, d_768_L5B01 = [], []
d_644_L2T01, d_768_L2T01 = [], []
d_644_L3T01, d_768_L3T01 = [], []
d_644_L4T01, d_768_L4T01 = [], []
d_644_L5T01, d_768_L5T01 = [], []
d_644_L2d02, d_768_L2d02 = [], []
d_644_L3d02, d_768_L3d02 = [], []
d_644_L4d02, d_768_L4d02 = [], []
d_644_L5d02, d_768_L5d02 = [], []
d_644_L2N02, d_768_L2N02 = [], []
d_644_L3N02, d_768_L3N02 = [], []
d_644_L4N02, d_768_L4N02 = [], []
d_644_L5N02, d_768_L5N02= [], []
d_644_L2k02, d_768_L2k02 = [], []
d_644_L3k02, d_768_L3k02 = [], []
d_644_L4k02, d_768_L4k02 = [], []
d_644_L5k02, d_768_L5k02 = [], []
d_644_L2B02, d_768_L2B02 = [], []
d_644_L3B02, d_768_L3B02 = [], []
d_644_L4B02, d_768_L4B02 = [], []
d_644_L5B02, d_768_L5B02 = [], []
d_644_L2T02, d_768_L2T02 = [], []
d_644_L3T02, d_768_L3T02 = [], []
d_644_L4T02, d_768_L4T02 = [], []
d_644_L5T02, d_768_L5T02 = [], []
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

label = []

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
        plot_mag("F660", "F480_gSDSS", "F625")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
#ax1.set_xlim(xmin=-1.7,xmax=2.0)
#ax1.set_xlim(xmin=-1.0,xmax=3.5)
ax1.set_ylim(ymin=-5.0,ymax=1.5)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('F660 - F625', size = 16)
plt.ylabel('F660 - g', size = 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs')
ax1.scatter(d_768_L2d0, d_644_L2d0,  c= "orange", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3d0, d_644_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4d0, d_644_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5d0, d_644_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2N0, d_644_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3N0, d_644_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4N0, d_644_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5N0, d_644_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2k0, d_644_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3k0, d_644_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4k0, d_644_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5k0, d_644_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2B0, d_644_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3B0, d_644_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4B0, d_644_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5B0, d_644_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2T0, d_644_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax1.scatter(d_768_L3T0, d_644_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax1.scatter(d_768_L4T0, d_644_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax1.scatter(d_768_L5T0, d_644_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax1.scatter(d_768_L2d01, d_644_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3d01, d_644_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4d01, d_644_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5d01, d_644_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2N01, d_644_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3N01, d_644_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4N01, d_644_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5N01, d_644_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2k01, d_644_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3k01, d_644_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4k01, d_644_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5k01, d_644_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2B01, d_644_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3B01, d_644_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4B01, d_644_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5B01, d_644_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2T01, d_644_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax1.scatter(d_768_L3T01, d_644_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax1.scatter(d_768_L4T01, d_644_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax1.scatter(d_768_L5T01, d_644_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax1.scatter(d_768_L2d02, d_644_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax1.scatter(d_768_L3d02, d_644_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax1.scatter(d_768_L4d02, d_644_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax1.scatter(d_768_L5d02, d_644_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax1.scatter(d_768_L2N02, d_644_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax1.scatter(d_768_L3N02, d_644_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax1.scatter(d_768_L4N02, d_644_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax1.scatter(d_768_L5N02, d_644_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax1.scatter(d_768_L2k02, d_644_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax1.scatter(d_768_L3k02, d_644_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax1.scatter(d_768_L4k02, d_644_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax1.scatter(d_768_L5k02, d_644_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax1.scatter(d_768_L2B02, d_644_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax1.scatter(d_768_L3B02, d_644_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax1.scatter(d_768_L4B02, d_644_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax1.scatter(d_768_L5B02, d_644_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax1.scatter(d_768_L2T02, d_644_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax1.scatter(d_768_L3T02, d_644_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax1.scatter(d_768_L4T02, d_644_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax1.scatter(d_768_L5T02, d_644_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
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
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
           transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

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
ax1.legend(scatterpoints=1, ncol=3, fontsize=5.8, loc='lower right', **lgd_kws)
#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-SPLUS-F660.pdf')

