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
    x1, y1 = filter_mag("CV", "", f1, f2, f3)
    x2, y2 = filter_mag("E00", "DdDm1_L2", f1, f2, f3)
    x3, y3 = filter_mag("E00", "DdDm1_L3", f1, f2, f3)
    x4, y4 = filter_mag("E00", "DdDm1_L4", f1, f2, f3)
    x5, y5 = filter_mag("E00", "DdDm1_L5", f1, f2, f3)
    x6, y6 = filter_mag("E00", "N2242_L2", f1, f2, f3)
    x7, y7 = filter_mag("E00", "N2242_L3", f1, f2, f3)
    x8, y8 = filter_mag("E00", "N2242_L4", f1, f2, f3)
    x9, y9 = filter_mag("E00", "N2242_L5", f1, f2, f3)
    x10, y10 = filter_mag("E00", "K648_L2", f1, f2, f3)
    x11, y11 = filter_mag("E00", "K648_L3", f1, f2, f3)
    x12, y12 = filter_mag("E00", "K648_L4", f1, f2, f3)
    x13, y13 = filter_mag("E00", "K648_L5", f1, f2, f3)
    x14, y14 = filter_mag("E00", "BB1_L2", f1, f2, f3)
    x15, y15 = filter_mag("E00", "BB1_L3", f1, f2, f3)
    x16, y16 = filter_mag("E00", "BB1_L4", f1, f2, f3)
    x17, y17 = filter_mag("E00", "BB1_L5", f1, f2, f3)
    x18, y18 = filter_mag("E00", "Typ_L2", f1, f2, f3)
    x19, y19 = filter_mag("E00", "Typ_L3", f1, f2, f3)
    x20, y20 = filter_mag("E00", "Typ_L4", f1, f2, f3)
    x21, y21 = filter_mag("E00", "Typ_L5", f1, f2, f3)
    x22, y22 = filter_mag("E01", "DdDm1_L2", f1, f2, f3)
    x23, y23 = filter_mag("E01", "DdDm1_L3", f1, f2, f3)
    x24, y24 = filter_mag("E01", "DdDm1_L4", f1, f2, f3)
    x25, y25 = filter_mag("E01", "DdDm1_L5", f1, f2, f3)
    x26, y26 = filter_mag("E01", "N2242_L2", f1, f2, f3)
    x27, y27 = filter_mag("E01", "N2242_L3", f1, f2, f3)
    x28, y28 = filter_mag("E01", "N2242_L4", f1, f2, f3)
    x29, y29 = filter_mag("E01", "N2242_L5", f1, f2, f3)
    x30, y30 = filter_mag("E01", "K648_L2", f1, f2, f3)
    x31, y31 = filter_mag("E01", "K648_L3", f1, f2, f3)
    x32, y32 = filter_mag("E01", "K648_L4", f1, f2, f3)
    x33, y33 = filter_mag("E01", "K648_L5", f1, f2, f3)
    x34, y34 = filter_mag("E01", "BB1_L2", f1, f2, f3)
    x35, y35 = filter_mag("E01", "BB1_L3", f1, f2, f3)
    x36, y36 = filter_mag("E01", "BB1_L4", f1, f2, f3)
    x37, y37 = filter_mag("E01", "BB1_L5", f1, f2, f3)
    x38, y38 = filter_mag("E01", "Typ_L2", f1, f2, f3)
    x39, y39 = filter_mag("E01", "Typ_L3", f1, f2, f3)
    x40, y40 = filter_mag("E01", "Typ_L4", f1, f2, f3)
    x41, y41 = filter_mag("E01", "Typ_L5", f1, f2, f3)
    x42, y42 = filter_mag("E02", "DdDm1_L2", f1, f2, f3)
    x43, y43 = filter_mag("E02", "DdDm1_L3", f1, f2, f3)
    x44, y44 = filter_mag("E02", "DdDm1_L4", f1, f2, f3)
    x45, y45 = filter_mag("E02", "DdDm1_L5", f1, f2, f3)
    x46, y46 = filter_mag("E02", "N2242_L2", f1, f2, f3)
    x47, y47 = filter_mag("E02", "N2242_L3", f1, f2, f3)
    x48, y48 = filter_mag("E02", "N2242_L4", f1, f2, f3)
    x49, y49 = filter_mag("E02", "N2242_L5", f1, f2, f3)
    x50, y50 = filter_mag("E02", "K648_L2", f1, f2, f3)
    x51, y51 = filter_mag("E02", "K648_L3", f1, f2, f3)
    x52, y52 = filter_mag("E02", "K648_L4", f1, f2, f3)
    x53, y53 = filter_mag("E02", "K648_L5", f1, f2, f3)
    x54, y54 = filter_mag("E02", "BB1_L2", f1, f2, f3)
    x55, y55 = filter_mag("E02", "BB1_L3", f1, f2, f3)
    x56, y56 = filter_mag("E02", "BB1_L4", f1, f2, f3)
    x57, y57 = filter_mag("E02", "BB1_L5", f1, f2, f3)
    x58, y58 = filter_mag("E02", "Typ_L2", f1, f2, f3)
    x59, y59 = filter_mag("E02", "Typ_L3", f1, f2, f3)
    x60, y60 = filter_mag("E02", "Typ_L4", f1, f2, f3)
    x61, y61 = filter_mag("E02", "Typ_L5", f1, f2, f3)
    x62, y62 = filter_mag("-DPNe", "", f1, f2, f3)
    x63, y63 = filter_mag("QSOs-hz", "", f1, f2, f3)
    x64, y64 = filter_mag("QSOs-010", "",  f1, f2, f3)
    x65, y65 = filter_mag("QSOs-101", "", f1, f2, f3)
    x66, y66 = filter_mag("QSOs-201", "", f1, f2, f3)
    x67, y67 = filter_mag("QSOs-301", "", f1, f2, f3)
    x68, y68 = filter_mag("QSOs-401", "", f1, f2, f3)
    x69, y69 = filter_mag("-SFGs", "", f1, f2, f3)
    x70, y70 = filter_mag("-sys", "", f1, f2, f3)
    x71, y71 = filter_mag("-sys-IPHAS", "", f1, f2, f3) 
    x72, y72 = filter_mag("-ExtHII", "", f1, f2, f3)
    x73, y73 = filter_mag("-sys-Ext", '', f1, f2, f3)
    x74, y74 = filter_mag("-survey", '', f1, f2, f3)
    x75, y75 = filter_mag("-SNR", '', f1, f2, f3)
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
label_HII = []
label_sys = []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("Jv0915_6199", "Jv0915_6600", "Jv0915_7700")
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
        else:
            None

print(label)
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
ax1.set_xlim(xmin=-1.5,xmax=2.0)
ax1.set_ylim(ymin=-2.5,ymax=5.7)
plt.tick_params(axis='x', labelsize=15) 
#plt.tick_params(axis='y', labelsize=15)
#ax1.fill(x, y, color= 'k', alpha=0.1)
#ax1.add_artist(circle)
# ax1.set_xlim(xmin=-0.8,xmax=2.5)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('r - (7590-7810)', fontsize= 16)
plt.ylabel(' r - (6495-6705)', fontsize= 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN from SDSS')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs from SDSS')
ax1.scatter(d_768_L2d0, d_644_L2d0,  c= "orange", alpha=0.8, marker='s', s=5 )
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
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs from SDSS')
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
            xytext=(3, 3), textcoords='offset points', ha='left', va='bottom',)

    
plt.annotate(
    '', xy=(d_768_L2d0[0]-0.91, d_644_L2d0[0]-0.91), xycoords='data',
    xytext=(d_768_L2d02[0]-0.91, d_644_L2d02[0]-0.91), textcoords='data',
    arrowprops={'arrowstyle': '->'})
plt.annotate(
    'Extinction', xy=(d_768_L2d0[0]-1.2, d_644_L2d0[0]-1.1), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')
# ax1.arrow(d_768_L2d0[0], d_644_L2d0[0], d_768_L2d02[0], d_644_L2d02[0], fc="k", ec="k", head_width=0.05, head_length=0.1 )
# ax1.plot()
#ax1.quiver(d_768_L2d0[0], d_644_L2d0[0], d_768_L2d02[0], d_644_L2d02[0], angles='xy',scale_units='xy',scale=1)
# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, d_768_ExtHII, d_644_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)

#for label_, x, y in zip(label_sys, d_768_sys, d_644_sys):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
# ax1.legend(scatterpoints=1, ncol=3, fontsize=6.0, **lgd_kws)
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.1, box.height])
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), **lgd_kws)

#este es el de la leyenda
lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)


#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPAS-Viironen.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

#################################################################################################
#------------------------------------------------------------------------------------------------
#################################################################################################

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
label_HII = []
label_sys = []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        plot_mag("Jv0915_6600", "Jv0915_5001", "Jv0915_6199")
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
       

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-5.3,xmax=2.0)
# ax1.set_ylim(ymin=-4.5,ymax=2.0)
# ax1.set_xlim(xmin=-6.0,xmax=2.5)
###ax1.set_ylim(ymin=-4.0,ymax=1.5)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
plt.xlabel('(6495-6705) - r', size = 16)
plt.ylabel('(6495-6705) - (4895-5110)', size = 16)
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
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label=' HII region in NGC 55')
ax1.scatter(d_768_SN, d_644_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

plt.annotate(
    '', xy=(d_768_L2d0[0]+1, d_644_L2d0[0]+1), xycoords='data',
    xytext=(d_768_L2d02[0]+1, d_644_L2d02[0]+1), textcoords='data',
    arrowprops={'arrowstyle': '<-'})
plt.annotate(
    '', xy=(d_768_L2d0[0]+0.9, d_644_L2d0[0]+1.01), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')

# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, d_768_ExtHII, d_644_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#for label_, x, y in zip(label_sys, d_768_sys, d_644_sys):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax1.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=3, fontsize=5.0, loc='lower left', **lgd_kws)
#ax1.grid()
lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPAS-Jv0915_6600.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')

########################################################
#-------------------------------------------------------#
#######################################################

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
label_HII = []
label_sys = []

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
        plot_mag("Jv0915_5799", "Jv0915_6600", "Jv0915_8300")

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
ax1.set_xlim(xmin=-1.0,xmax=3.0)
# ax1.set_xlim(xmin=-1.1,xmax=4.0)
# ax1.set_ylim(ymin=-1.5,ymax=4.0)
ax1.set_ylim(ymin=-1.3,ymax=4.5)
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
plt.xlabel('(5690-5915) - (8190-8410)', size = 16)
plt.ylabel('(5690-5915) - (6495-6705)', size = 16)
ax1.scatter(d_768, d_644, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(d_768_CNP, d_644_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN from SDSS')
ax1.scatter(d_768_c, d_644_c, c='purple', alpha=0.8, label='CVs from SDSS')
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
ax1.scatter(d_768_SFGs, d_644_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs from SDSS')
ax1.scatter(d_768_sys, d_644_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax1.scatter(d_768_Extsys, d_644_Extsys,  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax1.scatter(d_768_sys_IPHAS, d_644_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(d_768_sysurvey, d_644_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics ')
ax1.scatter(d_768_ExtHII, d_644_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
#ax1.scatter(d_768_cAlh, d_644_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
ax1.scatter(d_768_SN, d_644_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.arrow(1, 1, 1, 1, head_width=0.07, head_length=0.5, fc='k', ec='k')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, d_768, d_644):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)


plt.annotate(
    '', xy=(d_768_L2d0[0]-0.5, d_644_L2d0[0]-0.5), xycoords='data',
    xytext=(d_768_L2d02[0]-0.5, d_644_L2d02[0]-0.5), textcoords='data',
    arrowprops={'arrowstyle': '<-'})
plt.annotate(
    '', xy=(d_768_L2d0[0]-0.6, d_644_L2d0[0]-0.35), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')
# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, d_768_cAlh, d_644_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, d_768_ExtHII, d_644_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom;',)

#for Z, x, y in zip(z, d_768_Qz, d_644_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)
#for label_, x, y in zip(label_sys, d_768_sys, d_644_sys):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax1.minorticks_on()
lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid(which='minor')#, lw=0.3)
#ax1.legend(scatterpoints=1, ncol=3, fontsize=5.0, **lgd_kws)
#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax1.grid()
ax1.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('diagram-JPAS-v0915_5799.pdf',  bbox_extra_artists=(lgd,), bbox_inches='tight')
