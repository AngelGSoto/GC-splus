'''
Find the magnited of each source and estimate the colour
'''
from __future__ import print_function
import numpy as np
import glob
import json


def filter_mag(p, f1, f2, f3):
    col, col0 = [], []
    if data['id'].endswith(p) or data['id'].startswith(p):
        filter1 = data[f1]
        filter2 = data[f2]
        filter3 = data[f3]
        diff = filter1 - filter2
        diff0 = filter1 - filter3
        col.append(diff)
        col0.append(diff0)
    
    return col, col0

def find_mag(f1, f2, f3):
    x, y = filter_mag("HPNe", f1, f2, f3)
    x1, y1 = filter_mag("catB", f1, f2, f3)
    x2, y2 = filter_mag("DdDm1_L2", f1, f2, f3)
    x3, y3 = filter_mag("DdDm1_L3", f1, f2, f3)
    x4, y4 = filter_mag("DdDm1_L4", f1, f2, f3)
    x5, y5 = filter_mag("DdDm1_L5", f1, f2, f3)
    x6, y6 = filter_mag("N2242_L2", f1, f2, f3)
    x7, y7 = filter_mag("N2242_L3", f1, f2, f3)
    x8, y8 = filter_mag("N2242_L4", f1, f2, f3)
    x9, y9 = filter_mag("N2242_L5", f1, f2, f3)
    x10, y10 = filter_mag("K648_L2", f1, f2, f3)
    x11, y11 = filter_mag("K648_L3", f1, f2, f3)
    x12, y12 = filter_mag("K648_L4", f1, f2, f3)
    x13, y13 = filter_mag("K648_L5", f1, f2, f3)
    x14, y14 = filter_mag("BB1_L2", f1, f2, f3)
    x15, y15 = filter_mag("BB1_L3", f1, f2, f3)
    x16, y16 = filter_mag("BB1_L4", f1, f2, f3)
    x17, y17 = filter_mag("BB1_L5", f1, f2, f3)
    x18, y18 = filter_mag("Typ_L2", f1, f2, f3)
    x19, y19 = filter_mag("Typ_L3", f1, f2, f3)
    x20, y20 = filter_mag("Typ_L4", f1, f2, f3)
    x21, y21 = filter_mag("Typ_L5", f1, f2, f3)
    x22, y22 = filter_mag("-C-PNe", f1, f2, f3)
    x23, y23 = filter_mag("QSOs-hz", f1, f2, f3)
    x24, y24 = filter_mag("QSOs-010", f1, f2, f3)
    x25, y25 = filter_mag("QSOs-101", f1, f2, f3)
    x26, y26 = filter_mag("QSOs-201", f1, f2, f3)
    x27, y27 = filter_mag("QSOs-301", f1, f2, f3)
    x28, y28 = filter_mag("QSOs-401", f1, f2, f3)
    x29, y29 = filter_mag("-SFGs", f1, f2, f3)
    x30, y30 = filter_mag("-sys", f1, f2, f3)
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
      
