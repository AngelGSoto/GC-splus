'''
Make inices diagram for photometryc system available
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns


pattern = "*-spectros/*-JPAS15-magnitude.json"
file_list = glob.glob(pattern)

def filter_mag(e, s, f1, f2, f3, f4, f5, f6, f7):
    col, col0, col1 = [], [], []
    if data['id'].endswith(e):
        if data['id'].startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            filter4 = data[f4]
            filter5 = data[f5]
            filter6 = data[f6]
            filter7 = data[f7]
            A1 = ((filter2-filter3) - (filter3-filter4))/filter5 + (filter6-filter7)
            B1 = ((filter1-filter2) - (filter2-filter3))/filter5
            C1 = filter3-filter4
            col.append(A1)
            col0.append(B1)
            col1.append(C1)
    
    return col, col0, col1

def plot_mag(f1, f2, f3, f4, f5, f6, f7):
    x, y, z = filter_mag("HPNe", "", f1, f2, f3, f4, f5, f6, f7)
    x1, y1, z1 = filter_mag("CV", "", f1, f2, f3, f4, f5, f6, f7)
    x2, y2, z2 = filter_mag("E00", "DdDm1_L2", f1, f2, f3, f4, f5, f6, f7)
    x3, y3, z3 = filter_mag("E00", "DdDm1_L3", f1, f2, f3, f4, f5, f6, f7)
    x4, y4, z4 = filter_mag("E00", "DdDm1_L4", f1, f2, f3, f4, f5, f6, f7)
    x5, y5, z5 = filter_mag("E00", "DdDm1_L5", f1, f2, f3, f4, f5, f6, f7)
    x6, y6, z6 = filter_mag("E00", "N2242_L2", f1, f2, f3, f4, f5, f6, f7)
    x7, y7, z7 = filter_mag("E00", "N2242_L3", f1, f2, f3, f4, f5, f6, f7)
    x8, y8, z8 = filter_mag("E00", "N2242_L4", f1, f2, f3, f4, f5, f6, f7)
    x9, y9, z9 = filter_mag("E00", "N2242_L5", f1, f2, f3, f4, f5, f6, f7)
    x10, y10, z10 = filter_mag("E00", "K648_L2", f1, f2, f3, f4, f5, f6, f7)
    x11, y11, z11 = filter_mag("E00", "K648_L3", f1, f2, f3, f4, f5, f6, f7)
    x12, y12, z12 = filter_mag("E00", "K648_L4", f1, f2, f3, f4, f5, f6, f7)
    x13, y13, z13 = filter_mag("E00", "K648_L5", f1, f2, f3, f4, f5, f6, f7)
    x14, y14, z14 = filter_mag("E00", "BB1_L2", f1, f2, f3, f4, f5, f6, f7)
    x15, y15, z15 = filter_mag("E00", "BB1_L3", f1, f2, f3, f4, f5, f6, f7)
    x16, y16, z16 = filter_mag("E00", "BB1_L4", f1, f2, f3, f4, f5, f6, f7)
    x17, y17, z17 = filter_mag("E00", "BB1_L5", f1, f2, f3, f4, f5, f6, f7)
    x18, y18, z18 = filter_mag("E00", "Typ_L2", f1, f2, f3, f4, f5, f6, f7)
    x19, y19, z19 = filter_mag("E00", "Typ_L3", f1, f2, f3, f4, f5, f6, f7)
    x20, y20, z20 = filter_mag("E00", "Typ_L4", f1, f2, f3, f4, f5, f6, f7)
    x21, y21, z21 = filter_mag("E00", "Typ_L5", f1, f2, f3, f4, f5, f6, f7)
    x22, y22, z22 = filter_mag("E01", "DdDm1_L2", f1, f2, f3, f4, f5, f6, f7)
    x23, y23, z23 = filter_mag("E01", "DdDm1_L3", f1, f2, f3, f4, f5, f6, f7)
    x24, y24, z24 = filter_mag("E01", "DdDm1_L4", f1, f2, f3, f4, f5, f6, f7)
    x25, y25, z25 = filter_mag("E01", "DdDm1_L5", f1, f2, f3, f4, f5, f6, f7)
    x26, y26, z26 = filter_mag("E01", "N2242_L2", f1, f2, f3, f4, f5, f6, f7)
    x27, y27, z27 = filter_mag("E01", "N2242_L3", f1, f2, f3, f4, f5, f6, f7)
    x28, y28, z28 = filter_mag("E01", "N2242_L4", f1, f2, f3, f4, f5, f6, f7)
    x29, y29, z29 = filter_mag("E01", "N2242_L5", f1, f2, f3, f4, f5, f6, f7)
    x30, y30, z30 = filter_mag("E01", "K648_L2", f1, f2, f3, f4, f5, f6, f7)
    x31, y31, z31 = filter_mag("E01", "K648_L3", f1, f2, f3, f4, f5, f6, f7)
    x32, y32, z32 = filter_mag("E01", "K648_L4", f1, f2, f3, f4, f5, f6, f7)
    x33, y33, z33 = filter_mag("E01", "K648_L5", f1, f2, f3, f4, f5, f6, f7)
    x34, y34, z34 = filter_mag("E01", "BB1_L2", f1, f2, f3, f4, f5, f6, f7)
    x35, y35, z35 = filter_mag("E01", "BB1_L3", f1, f2, f3, f4, f5, f6, f7)
    x36, y36, z36 = filter_mag("E01", "BB1_L4", f1, f2, f3, f4, f5, f6, f7)
    x37, y37, z37 = filter_mag("E01", "BB1_L5", f1, f2, f3, f4, f5, f6, f7)
    x38, y38, z38 = filter_mag("E01", "Typ_L2", f1, f2, f3, f4, f5, f6, f7)
    x39, y39, z39 = filter_mag("E01", "Typ_L3", f1, f2, f3, f4, f5, f6, f7)
    x40, y40, z40 = filter_mag("E01", "Typ_L4", f1, f2, f3, f4, f5, f6, f7)
    x41, y41, z41 = filter_mag("E01", "Typ_L5", f1, f2, f3, f4, f5, f6, f7)
    x42, y42, z42 = filter_mag("E02", "DdDm1_L2", f1, f2, f3, f4, f5, f6, f7)
    x43, y43, z43 = filter_mag("E02", "DdDm1_L3", f1, f2, f3, f4, f5, f6, f7)
    x44, y44, z44 = filter_mag("E02", "DdDm1_L4", f1, f2, f3, f4, f5, f6, f7)
    x45, y45, z45 = filter_mag("E02", "DdDm1_L5", f1, f2, f3, f4, f5, f6, f7)
    x46, y46, z46 = filter_mag("E02", "N2242_L2", f1, f2, f3, f4, f5, f6, f7)
    x47, y47, z47 = filter_mag("E02", "N2242_L3", f1, f2, f3, f4, f5, f6, f7)
    x48, y48, z48 = filter_mag("E02", "N2242_L4", f1, f2, f3, f4, f5, f6, f7)
    x49, y49, z49 = filter_mag("E02", "N2242_L5", f1, f2, f3, f4, f5, f6, f7)
    x50, y50, z50 = filter_mag("E02", "K648_L2", f1, f2, f3, f4, f5, f6, f7)
    x51, y51, z51 = filter_mag("E02", "K648_L3", f1, f2, f3, f4, f5, f6, f7)
    x52, y52, z52 = filter_mag("E02", "K648_L4", f1, f2, f3, f4, f5, f6, f7)
    x53, y53, z53 = filter_mag("E02", "K648_L5", f1, f2, f3, f4, f5, f6, f7)
    x54, y54, z54 = filter_mag("E02", "BB1_L2", f1, f2, f3, f4, f5, f6, f7)
    x55, y55, z55 = filter_mag("E02", "BB1_L3", f1, f2, f3, f4, f5, f6, f7)
    x56, y56, z56 = filter_mag("E02", "BB1_L4", f1, f2, f3, f4, f5, f6, f7)
    x57, y57, z57 = filter_mag("E02", "BB1_L5", f1, f2, f3, f4, f5, f6, f7)
    x58, y58, z58 = filter_mag("E02", "Typ_L2", f1, f2, f3, f4, f5, f6, f7)
    x59, y59, z59 = filter_mag("E02", "Typ_L3", f1, f2, f3, f4, f5, f6, f7)
    x60, y60, z60 = filter_mag("E02", "Typ_L4", f1, f2, f3, f4, f5, f6, f7)
    x61, y61, z61 = filter_mag("E02", "Typ_L5", f1, f2, f3, f4, f5, f6, f7)
    x62, y62, z62 = filter_mag("-DPNe", "", f1, f2, f3, f4, f5, f6, f7)
    x63, y63, z63 = filter_mag("QSOs-hz", "", f1, f2, f3, f4, f5, f6, f7)
    x64, y64, z64 = filter_mag("QSOs-010", "",  f1, f2, f3, f4, f5, f6, f7)
    x65, y65, z65 = filter_mag("QSOs-101", "", f1, f2, f3, f4, f5, f6, f7)
    x66, y66, z66 = filter_mag("QSOs-201", "", f1, f2, f3, f4, f5, f6, f7)
    x67, y67, z67 = filter_mag("QSOs-301", "", f1, f2, f3, f4, f5, f6, f7)
    x68, y68, z68 = filter_mag("QSOs-401", "", f1, f2, f3, f4, f5, f6, f7)
    x69, y69, z69 = filter_mag("-SFGs", "", f1, f2, f3, f4, f5, f6, f7)
    x70, y70, z70 = filter_mag("-sys", "", f1, f2, f3, f4, f5, f6, f7)
    x71, y71, z71 = filter_mag("-sys-IPHAS", "", f1, f2, f3, f4, f5, f6, f7) 
    x72, y72, z72 = filter_mag("-ExtHII", "", f1, f2, f3, f4, f5, f6, f7)
    x73, y73, z73 = filter_mag("-sys-Ext", "", f1, f2, f3, f4, f5, f6, f7)
    x74, y74, z74 = filter_mag("-survey", "", f1, f2, f3, f4, f5, f6, f7)
    x75, y75, z75 = filter_mag("-SNR", "", f1, f2, f3, f4, f5, f6, f7)
    for a, b, c in zip(x, y, z):
        A1.append(a)
        B1.append(b)
        C1.append(c)
    for a, b, c in zip(x1, y1, z1):
        A1_c.append(a)
        B1_c.append(b)
        C1_c.append(c)
    for a, b, c in zip(x2, y2, z2):
        A1_L2d0.append(a)
        B1_L2d0.append(b)
        C1_L2d0.append(c)
    for a, b, c in zip(x3, y3, z3):
        A1_L3d0.append(a)
        B1_L3d0.append(b)
        C1_L3d0.append(c)
    for a, b, c in zip(x4, y4, z4):
        A1_L4d0.append(a)
        B1_L4d0.append(b)
        C1_L4d0.append(c)
    for a, b, c in zip(x5, y5, z5):
        A1_L5d0.append(a)
        B1_L5d0.append(b)
        C1_L5d0.append(c)
    for a, b, c in zip(x6, y6, z6):
        A1_L2N0.append(a)
        B1_L2N0.append(b)
        C1_L2N0.append(c)
    for a, b, c in zip(x7, y7, z7):
        A1_L3N0.append(a)
        B1_L3N0.append(b)
        C1_L3N0.append(c)
    for a, b, c in zip(x8, y8, z8):
        A1_L4N0.append(a)
        B1_L4N0.append(b)
        C1_L4N0.append(c)
    for a, b, c in zip(x9, y9, z9):
        A1_L5N0.append(a)
        B1_L5N0.append(b)
        C1_L5N0.append(c)
    for a, b, c in zip(x10, y10, z10):
        A1_L2k0.append(a)
        B1_L2k0.append(b)
        C1_L2k0.append(c)
    for a, b, c in zip(x11, y11, z11):
        A1_L3k0.append(a)
        B1_L3k0.append(b)
        C1_L3k0.append(c)
    for a, b, c in zip(x12, y12, z12):
        A1_L4k0.append(a)
        B1_L4k0.append(b)
        C1_L4k0.append(c)
    for a, b, c in zip(x13, y13, z13):
        A1_L5k0.append(a)
        B1_L5k0.append(b)
        C1_L5k0.append(c)
    for a, b, c in zip(x14, y14, z14):
        A1_L2B0.append(a)
        B1_L2B0.append(b)
        C1_L2B0.append(c)
    for a, b, c in zip(x15, y15, z15):
        A1_L3B0.append(a)
        B1_L3B0.append(b)
        C1_L3B0.append(c)
    for a, b, c in zip(x16, y16, z16):
        A1_L4B0.append(a)
        B1_L4B0.append(b)
        C1_L4B0.append(c)
    for a, b, c in zip(x17, y17, z17):
        A1_L5B0.append(a)
        B1_L5B0.append(b)
        C1_L5B0.append(c)
    for a, b, c in zip(x18, y18, z18):
        A1_L2T0.append(a)
        B1_L2T0.append(b)
        C1_L2T0.append(c)
    for a, b, c in zip(x19, y19, z19):
        A1_L3T0.append(a)
        B1_L3T0.append(b)
        C1_L3T0.append(c)
    for a, b, c in zip(x20, y20, z20):
        A1_L4T0.append(a)
        B1_L4T0.append(b)
        C1_L4T0.append(c)
    for a, b, c in zip(x21, y21, z21):
        A1_L5T0.append(a)
        B1_L5T0.append(b)
        C1_L5T0.append(c)
    for a, b, c in zip(x22, y22, z22):
        A1_L2d01.append(a)
        B1_L2d01.append(b)
        C1_L2d01.append(c)
    for a, b, c in zip(x23, y23, z23):
        A1_L3d01.append(a)
        B1_L3d01.append(b)
        C1_L3d01.append(c)
    for a, b, c in zip(x24, y24, z24):
        A1_L4d01.append(a)
        B1_L4d01.append(b)
        C1_L4d01.append(c)
    for a, b, c in zip(x25, y25, z25):
        A1_L5d01.append(a)
        B1_L5d01.append(b)
        C1_L5d01.append(c)
    for a, b, c in zip(x26, y26, z26):
        A1_L2N01.append(a)
        B1_L2N01.append(b)
        C1_L2N01.append(c)
    for a, b, c in zip(x27, y27, z27):
        A1_L3N01.append(a)
        B1_L3N01.append(b)
        C1_L3N01.append(c)
    for a, b, c in zip(x28, y28, z28):
        A1_L4N01.append(a)
        B1_L4N01.append(b)
        C1_L4N01.append(c)
    for a, b, c in zip(x29, y29, z29):
        A1_L5N01.append(a)
        B1_L5N01.append(b)
        C1_L5N01.append(c)
    for a, b, c in zip(x30, y30, z30):
        A1_L2k01.append(a)
        B1_L2k01.append(b)
        C1_L2k01.append(c)
    for a, b, c in zip(x31, y31, z31):
        A1_L3k01.append(a)
        B1_L3k01.append(b)
        C1_L3k01.append(c)
    for a, b, c in zip(x32, y32, z32):
        A1_L4k01.append(a)
        B1_L4k01.append(b)
        C1_L4k01.append(c)
    for a, b, c in zip(x33, y33, z33):
        A1_L5k01.append(a)
        B1_L5k01.append(b)
        C1_L5k01.append(c)
    for a, b, c in zip(x34, y34, z34):
        A1_L2B01.append(a)
        B1_L2B01.append(b)
        C1_L2B01.append(c)
    for a, b, c in zip(x35, y35, z35):
        A1_L3B01.append(a)
        B1_L3B01.append(b)
        C1_L3B01.append(c)
    for a, b, c in zip(x36, y36, z36):
        A1_L4B01.append(a)
        B1_L4B01.append(b)
        C1_L4B01.append(c)
    for a, b, c in zip(x37, y37, z37):
        A1_L5B01.append(a)
        B1_L5B01.append(b)
        C1_L5B01.append(c)
    for a, b, c in zip(x38, y38, z38):
        A1_L2T01.append(a)
        B1_L2T01.append(b)
        C1_L2T01.append(c)
    for a, b, c in zip(x39, y39, z39):
        A1_L3T01.append(a)
        B1_L3T01.append(b)
        C1_L3T01.append(c)
    for a, b, c in zip(x40, y40, z40):
        A1_L4T01.append(a)
        B1_L4T01.append(b)
        C1_L4T01.append(c)
    for a, b, c in zip(x41, y41, z41):
        A1_L5T01.append(a)
        B1_L5T01.append(b)
        C1_L5T01.append(c)
    for a, b, c in zip(x42, y42, z42):
        A1_L2d02.append(a)
        B1_L2d02.append(b)
        C1_L2d02.append(c)
    for a, b, c in zip(x43, y43, z43):
        A1_L3d02.append(a)
        B1_L3d02.append(b)
        C1_L3d02.append(c)
    for a, b, c in zip(x44, y44, z44):
        A1_L4d02.append(a)
        B1_L4d02.append(b)
        C1_L4d02.append(c)
    for a, b, c in zip(x45, y45, z45):
        A1_L5d02.append(a)
        B1_L5d02.append(b)
        C1_L5d02.append(c)
    for a, b, c in zip(x46, y46, z46):
        A1_L2N02.append(a)
        B1_L2N02.append(b)
        C1_L2N02.append(c)
    for a, b, c in zip(x47, y47, z47):
        A1_L3N02.append(a)
        B1_L3N02.append(b)
        C1_L3N02.append(c)
    for a, b, c in zip(x48, y48, z48):
        A1_L4N02.append(a)
        B1_L4N02.append(b)
        C1_L4N02.append(c)
    for a, b, c in zip(x49, y49, z49):
        A1_L5N02.append(a)
        B1_L5N02.append(b)
        C1_L5N02.append(c)
    for a, b, c in zip(x50, y50, z50):
        A1_L2k02.append(a)
        B1_L2k02.append(b)
        C1_L2k02.append(c)
    for a, b, c in zip(x51, y51, z51):
        A1_L3k02.append(a)
        B1_L3k02.append(b)
        C1_L3k02.append(c)
    for a, b, c in zip(x52, y52, z52):
        A1_L4k02.append(a)
        B1_L4k02.append(b)
        C1_L4k02.append(c)
    for a, b, c in zip(x53, y53, z53):
        A1_L5k02.append(a)
        B1_L5k02.append(b)
        C1_L5k02.append(c)
    for a, b, c in zip(x54, y54, z54):
        A1_L2B02.append(a)
        B1_L2B02.append(b)
        C1_L2B02.append(c)
    for a, b, c in zip(x55, y55, z55):
        A1_L3B02.append(a)
        B1_L3B02.append(b)
        C1_L3B02.append(c)
    for a, b, c in zip(x56, y56, z56):
        A1_L4B02.append(a)
        B1_L4B02.append(b)
        C1_L4B02.append(c)
    for a, b, c in zip(x57, y57, z57):
        A1_L5B02.append(a)
        B1_L5B02.append(b)
        C1_L5B02.append(c)
    for a, b, c in zip(x58, y58, z58):
        A1_L2T02.append(a)
        B1_L2T02.append(b)
        C1_L2T02.append(c)
    for a, b, c in zip(x59, y59, z59):
        A1_L3T02.append(a)
        B1_L3T02.append(b)
        C1_L3T02.append(c)
    for a, b, c in zip(x60, y60, z60):
        A1_L4T02.append(a)
        B1_L4T02.append(b)
        C1_L4T02.append(c)
    for a, b, c in zip(x61, y61, z61):
        A1_L5T02.append(a)
        B1_L5T02.append(b)
        C1_L5T02.append(c)
    for a, b, c in zip(x62, y62, z62):
        A1_CNP.append(a)
        B1_CNP.append(b)
        C1_CNP.append(c)
    for a, b, c in zip(x63, y63, z63):
        A1_Qz.append(a)
        B1_Qz.append(b)
        C1_Qz.append(c)
    for a, b, c in zip(x64, y64, z64):
        A1_Q010.append(a)
        B1_Q010.append(b)
        C1_Q010.append(c)
    for a, b, c in zip(x65, y65, z65):
        A1_Q101.append(a)
        B1_Q101.append(b)
        C1_Q101.append(c)
    for a, b, c in zip(x66, y66, z66):
        A1_Q201.append(a)
        B1_Q201.append(b)
        C1_Q201.append(c)
    for a, b, c in zip(x67, y67, z67):
        A1_Qz.append(a)
        B1_Qz.append(b)
        C1_Qz.append(c)
    for a, b, c in zip(x68, y68, z68):
        A1_Q401.append(a)
        B1_Q401.append(b)
        C1_Q401.append(c)
    for a, b, c in zip(x69, y69, z69):
        A1_SFGs.append(a)
        B1_SFGs.append(b)
        C1_SFGs.append(c)
    for a, b, c in zip(x70, y70, z70):
        A1_sys.append(a)
        B1_sys.append(b)
        C1_sys.append(c)
    for a, b, c in zip(x71, y71, z71):
        A1_sys_IPHAS.append(a)
        B1_sys_IPHAS.append(b)
        C1_sys_IPHAS.append(c)
    for a, b, c in zip(x72, y72, z72):
        A1_ExtHII.append(a)
        B1_ExtHII.append(b)
        C1_ExtHII.append(c)
    for a, b, c in zip(x73, y73, z73):
        A1_Extsys.append(a)
        B1_Extsys.append(b)
        C1_Extsys.append(c)
    for a, b, c in zip(x74, y74, z74):
        A1_sysurvey.append(a)
        B1_sysurvey.append(b)
        C1_sysurvey.append(c)
    for a, b, c in zip(x75, y75, z75):
        A1_SN.append(a)
        B1_SN.append(b)
        C1_SN.append(c)

A1, B1, C1 = [], [], []
A1_CNP, B1_CNP, C1_CNP = [], [], []
A1_c, B1_c, C1_c = [], [], []
A1_L2d0, B1_L2d0, C1_L2d0 = [], [], []
A1_L3d0, B1_L3d0, C1_L3d0 = [], [], []
A1_L4d0, B1_L4d0, C1_L4d0 = [], [], []
A1_L5d0, B1_L5d0, C1_L5d0 = [], [], []
A1_L2N0, B1_L2N0, C1_L2N0 = [], [], []
A1_L3N0, B1_L3N0, C1_L3N0 = [], [], []
A1_L4N0, B1_L4N0, C1_L4N0 = [], [], []
A1_L5N0, B1_L5N0, C1_L5N0 = [], [], []
A1_L2k0, B1_L2k0, C1_L2k0 = [], [], []
A1_L3k0, B1_L3k0, C1_L3k0 = [], [], []
A1_L4k0, B1_L4k0, C1_L4k0 = [], [], []
A1_L5k0, B1_L5k0, C1_L5k0 = [], [], []
A1_L2B0, B1_L2B0, C1_L2B0 = [], [], []
A1_L3B0, B1_L3B0, C1_L3B0 = [], [], []
A1_L4B0, B1_L4B0, C1_L4B0 = [], [], []
A1_L5B0, B1_L5B0, C1_L5B0 = [], [], []
A1_L2T0, B1_L2T0, C1_L2T0 = [], [], []
A1_L3T0, B1_L3T0, C1_L3T0 = [], [], []
A1_L4T0, B1_L4T0, C1_L4T0 = [], [], []
A1_L5T0, B1_L5T0, C1_L5T0 = [], [], []
A1_L2d01, B1_L2d01, C1_L2d01 = [], [], []
A1_L3d01, B1_L3d01, C1_L3d01 = [], [], []
A1_L4d01, B1_L4d01, C1_L4d01 = [], [], []
A1_L5d01, B1_L5d01, C1_L5d01 = [], [], []
A1_L2N01, B1_L2N01, C1_L2N01 = [], [], []
A1_L3N01, B1_L3N01, C1_L3N01 = [], [], []
A1_L4N01, B1_L4N01, C1_L4N01 = [], [], []
A1_L5N01, B1_L5N01, C1_L5N01= [], [], []
A1_L2k01, B1_L2k01, C1_L2k01 = [], [], []
A1_L3k01, B1_L3k01, C1_L3k01 = [], [], []
A1_L4k01, B1_L4k01, C1_L4k01 = [], [], []
A1_L5k01, B1_L5k01, C1_L5k01 = [], [], []
A1_L2B01, B1_L2B01, C1_L2B01 = [], [], []
A1_L3B01, B1_L3B01, C1_L3B01 = [], [], []
A1_L4B01, B1_L4B01, C1_L4B01 = [], [], []
A1_L5B01, B1_L5B01, C1_L5B01 = [], [], []
A1_L2T01, B1_L2T01, C1_L2T01 = [], [], []
A1_L3T01, B1_L3T01, C1_L3T01 = [], [], []
A1_L4T01, B1_L4T01, C1_L4T01 = [], [], []
A1_L5T01, B1_L5T01, C1_L5T01 = [], [], []
A1_L2d02, B1_L2d02, C1_L2d02 = [], [], []
A1_L3d02, B1_L3d02, C1_L3d02 = [], [], []
A1_L4d02, B1_L4d02, C1_L4d02 = [], [], []
A1_L5d02, B1_L5d02, C1_L5d02 = [], [], []
A1_L2N02, B1_L2N02, C1_L2N02 = [], [], []
A1_L3N02, B1_L3N02, C1_L3N02 = [], [], []
A1_L4N02, B1_L4N02, C1_L4N02 = [], [], []
A1_L5N02, B1_L5N02, C1_L5N02 = [], [], []
A1_L2k02, B1_L2k02, C1_L2k02 = [], [], []
A1_L3k02, B1_L3k02, C1_L3k02 = [], [], []
A1_L4k02, B1_L4k02, C1_L4k02 = [], [], []
A1_L5k02, B1_L5k02, C1_L5k02 = [], [], []
A1_L2B02, B1_L2B02, C1_L2B02 = [], [], []
A1_L3B02, B1_L3B02, C1_L3B02 = [], [], []
A1_L4B02, B1_L4B02, C1_L4B02 = [], [], []
A1_L5B02, B1_L5B02, C1_L5B02 = [], [], []
A1_L2T02, B1_L2T02, C1_L2T02 = [], [], []
A1_L3T02, B1_L3T02, C1_L3T02 = [], [], []
A1_L4T02, B1_L4T02, C1_L4T02 = [], [], []
A1_L5T02, B1_L5T02, C1_L5T02 = [], [], []
A1_Qz, B1_Qz, C1_Qz = [], [], []
A1_cAlh, B1_cAlh, C1_cAlh = [], [], []
A1_Q010, B1_Q010, C1_Q010 = [], [], []
A1_Q101, B1_Q101, C1_Q101 = [], [], []
A1_Q201, B1_Q201, C1_Q201 = [], [], []
A1_Q401, B1_Q401, C1_Q401 = [], [], []
A1_SFGs, B1_SFGs, C1_SFGs = [], [], []
A1_sys, B1_sys, C1_sys = [], [], []
A1_sys_IPHAS, B1_sys_IPHAS, C1_sys_IPHAS = [], [], []
A1_ExtHII, B1_ExtHII, C1_ExtHII = [], [], []
A1_Extsys, B1_Extsys, C1_Extsys = [], [], []
A1_sysurvey, B1_sysurvey, C1_sysurvey = [], [], []
A1_SN, B1_SN, C1_SN = [], [], []

label = []
label_HII = []
label_sys = []

for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
        if data['id'].endswith("1-HPNe"):
            label.append(data['id'].split("-H")[0])
        elif data['id'].endswith("1359559-HPNe"):
            label.append("PNG 135.9+55.9")
        elif data['id'].endswith("SLOAN-HPNe"):
            label.append("H4-1")
        elif data['id'].startswith("ngc"):
            label.append("NGC 2242")
        elif data['id'].startswith("mwc"):
            label.append("MWC 574")
        if data['id'].endswith("-ExtHII"):
            label_HII.append(data['id'].split("b")[0])
        elif data['id'].endswith("-sys"):
            label_sys.append(data['id'].split("-sys")[0])
        plot_mag("Jv0915_3495", "Jv0915_4100", "Jv0915_4701", "Jv0915_5500", 
                                 "Jv0915_6600", "Jv0915_9000", "Jv0915_9669")


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(7, 6))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(xmin=-1.5,xmax=2.0)
# ax1.set_ylim(ymin=-2.5,ymax=5.7)
plt.tick_params(axis='x', labelsize=15) 
#plt.tick_params(axis='y', labelsize=15)
#ax1.fill(x, y, color= 'k', alpha=0.1)
#ax1.add_artist(circle)
# ax1.set_xlim(xmin=-0.8,xmax=2.5)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('B1', fontsize= 16)
plt.ylabel('A1', fontsize= 16)
ax1.scatter(B1, A1, c='black', alpha=0.8, s=35, label='Halo PNe')
ax1.scatter(B1_CNP, A1_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN from SDSS')
ax1.scatter(B1_c, A1_c, c='purple', alpha=0.8, label='CVs from SDSS')
ax1.scatter(B1_L2d0, A1_L2d0,  c= "orange", alpha=0.8, marker='s', s=5 )
ax1.scatter(B1_L3d0, A1_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax1.scatter(B1_L4d0, A1_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax1.scatter(B1_L5d0, A1_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax1.scatter(B1_L2N0, A1_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax1.scatter(B1_L3N0, A1_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax1.scatter(B1_L4N0, A1_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax1.scatter(B1_L5N0, A1_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax1.scatter(B1_L2k0, A1_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax1.scatter(B1_L3k0, A1_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax1.scatter(B1_L4k0, A1_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax1.scatter(B1_L5k0, A1_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax1.scatter(B1_L2B0, A1_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax1.scatter(B1_L3B0, A1_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax1.scatter(B1_L4B0, A1_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax1.scatter(B1_L5B0, A1_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax1.scatter(B1_L2T0, A1_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax1.scatter(B1_L3T0, A1_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax1.scatter(B1_L4T0, A1_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax1.scatter(B1_L5T0, A1_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax1.scatter(B1_L2d01, A1_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax1.scatter(B1_L3d01, A1_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax1.scatter(B1_L4d01, A1_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax1.scatter(B1_L5d01, A1_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax1.scatter(B1_L2N01, A1_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax1.scatter(B1_L3N01, A1_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax1.scatter(B1_L4N01, A1_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax1.scatter(B1_L5N01, A1_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax1.scatter(B1_L2k01, A1_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax1.scatter(B1_L3k01, A1_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax1.scatter(B1_L4k01, A1_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax1.scatter(B1_L5k01, A1_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax1.scatter(B1_L2B01, A1_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax1.scatter(B1_L3B01, A1_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax1.scatter(B1_L4B01, A1_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax1.scatter(B1_L5B01, A1_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax1.scatter(B1_L2T01, A1_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax1.scatter(B1_L3T01, A1_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax1.scatter(B1_L4T01, A1_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax1.scatter(B1_L5T01, A1_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax1.scatter(B1_L2d02, A1_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax1.scatter(B1_L3d02, A1_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax1.scatter(B1_L4d02, A1_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax1.scatter(B1_L5d02, A1_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax1.scatter(B1_L2N02, A1_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax1.scatter(B1_L3N02, A1_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax1.scatter(B1_L4N02, A1_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax1.scatter(B1_L5N02, A1_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax1.scatter(B1_L2k02, A1_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax1.scatter(B1_L3k02, A1_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax1.scatter(B1_L4k02, A1_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax1.scatter(B1_L5k02, A1_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax1.scatter(B1_L2B02, A1_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax1.scatter(B1_L3B02, A1_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax1.scatter(B1_L4B02, A1_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax1.scatter(B1_L5B02, A1_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax1.scatter(B1_L2T02, A1_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax1.scatter(B1_L3T02, A1_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax1.scatter(B1_L4T02, A1_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax1.scatter(B1_L5T02, A1_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
ax1.scatter(B1_Q401, A1_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax1.scatter(B1_Qz, A1_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax1.scatter(B1_Q201, A1_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax1.scatter(B1_Q101, A1_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax1.scatter(B1_Q010, A1_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax1.scatter(B1_SFGs, A1_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs from SDSS')
ax1.scatter(B1_sys, A1_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax1.scatter(B1_Extsys, A1_Extsys,  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax1.scatter(B1_sys_IPHAS, A1_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax1.scatter(B1_sysurvey, A1_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Buil Symbiotics')
ax1.scatter(B1_ExtHII, A1_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
ax1.scatter(B1_SN, A1_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.scatter(B1_cAlh, A1_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, B1, A1):
    ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(3, 3), textcoords='offset points', ha='left', va='bottom',)

plt.annotate(
    '', xy=(B1_L2d0[0]-0.91, A1_L2d0[0]-0.91), xycoords='data',
    xytext=(B1_L2d02[0]-0.91, A1_L2d02[0]-0.91), textcoords='data',
    arrowprops={'arrowstyle': '->'})
plt.annotate(
    'Extinction', xy=(B1_L2d0[0]-1.2, A1_L2d0[0]-1.1), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')
# ax1.arrow(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], fc="k", ec="k", head_width=0.05, head_length=0.1 )
# ax1.plot()
#ax1.quiver(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], angles='xy',scale_units='xy',scale=1)
# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, B1_cAlh, A1_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, B1_ExtHII, A1_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, B1_Qz, A1_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)

#for label_, x, y in zip(label_sys, B1_sys, A1_sys):
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
plt.savefig('indices-jpas.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


fig = plt.figure(figsize=(7, 6))
ax2 = fig.add_subplot(111)
ax2.set_xlim(xmin=-2.4,xmax=3.0)
ax2.set_ylim(ymin=-1.6,ymax=1.5)
plt.tick_params(axis='x', labelsize=15) 
#plt.tick_params(axis='y', labelsize=15)
#ax1.fill(x, y, color= 'k', alpha=0.1)
#ax1.add_artist(circle)
# ax1.set_xlim(xmin=-0.8,xmax=2.5)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('C1', fontsize= 16)
plt.ylabel('A1', fontsize= 16)
ax2.scatter(C1, A1, c='black', alpha=0.8, s=35, label='Halo PNe')
ax2.scatter(C1_CNP, A1_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN from SDSS')
ax2.scatter(C1_c, A1_c, c='purple', alpha=0.8, label='CVs from SDSS')
ax2.scatter(C1_L2d0, A1_L2d0,  c= "orange", alpha=0.8, marker='s', s=5 )
ax2.scatter(C1_L3d0, A1_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax2.scatter(C1_L4d0, A1_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax2.scatter(C1_L5d0, A1_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax2.scatter(C1_L2N0, A1_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax2.scatter(C1_L3N0, A1_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax2.scatter(C1_L4N0, A1_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax2.scatter(C1_L5N0, A1_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax2.scatter(C1_L2k0, A1_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax2.scatter(C1_L3k0, A1_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax2.scatter(C1_L4k0, A1_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax2.scatter(C1_L5k0, A1_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax2.scatter(C1_L2B0, A1_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax2.scatter(C1_L3B0, A1_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax2.scatter(C1_L4B0, A1_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax2.scatter(C1_L5B0, A1_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax2.scatter(C1_L2T0, A1_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax2.scatter(C1_L3T0, A1_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax2.scatter(C1_L4T0, A1_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax2.scatter(C1_L5T0, A1_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax2.scatter(C1_L2d01, A1_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax2.scatter(C1_L3d01, A1_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax2.scatter(C1_L4d01, A1_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax2.scatter(C1_L5d01, A1_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax2.scatter(C1_L2N01, A1_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax2.scatter(C1_L3N01, A1_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax2.scatter(C1_L4N01, A1_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax2.scatter(C1_L5N01, A1_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax2.scatter(C1_L2k01, A1_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax2.scatter(C1_L3k01, A1_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax2.scatter(C1_L4k01, A1_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax2.scatter(C1_L5k01, A1_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax2.scatter(C1_L2B01, A1_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax2.scatter(C1_L3B01, A1_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax2.scatter(C1_L4B01, A1_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax2.scatter(C1_L5B01, A1_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax2.scatter(C1_L2T01, A1_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax2.scatter(C1_L3T01, A1_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax2.scatter(C1_L4T01, A1_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax2.scatter(C1_L5T01, A1_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax2.scatter(C1_L2d02, A1_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax2.scatter(C1_L3d02, A1_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax2.scatter(C1_L4d02, A1_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax2.scatter(C1_L5d02, A1_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax2.scatter(C1_L2N02, A1_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax2.scatter(C1_L3N02, A1_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax2.scatter(C1_L4N02, A1_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax2.scatter(C1_L5N02, A1_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax2.scatter(C1_L2k02, A1_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax2.scatter(C1_L3k02, A1_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax2.scatter(C1_L4k02, A1_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax2.scatter(C1_L5k02, A1_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax2.scatter(C1_L2B02, A1_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax2.scatter(C1_L3B02, A1_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax2.scatter(C1_L4B02, A1_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax2.scatter(C1_L5B02, A1_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax2.scatter(C1_L2T02, A1_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax2.scatter(C1_L3T02, A1_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax2.scatter(C1_L4T02, A1_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax2.scatter(C1_L5T02, A1_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
ax2.scatter(C1_Q401, A1_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax2.scatter(C1_Qz, A1_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax2.scatter(C1_Q201, A1_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax2.scatter(C1_Q101, A1_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax2.scatter(C1_Q010, A1_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax2.scatter(C1_SFGs, A1_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs from SDSS')
ax2.scatter(C1_sys, A1_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax2.scatter(C1_Extsys, A1_Extsys,  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax2.scatter(C1_sys_IPHAS, A1_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax2.scatter(C1_sysurvey, A1_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Cuil Symbiotics')
ax2.scatter(C1_ExtHII, A1_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
ax2.scatter(C1_SN, A1_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.scatter(C1_cAlh, A1_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, C1, A1):
    ax2.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(3, 3), textcoords='offset points', ha='left', va='bottom',)

plt.annotate(
    '', xy=(C1_L2d0[0]-0.91, A1_L2d0[0]-0.91), xycoords='data',
    xytext=(C1_L2d02[0]-0.91, A1_L2d02[0]-0.91), textcoords='data',
    arrowprops={'arrowstyle': '->'})
plt.annotate(
    'Extinction', xy=(C1_L2d0[0]-1.2, A1_L2d0[0]-1.1), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')
# ax1.arrow(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], fc="k", ec="k", head_width=0.05, head_length=0.1 )
# ax1.plot()
#ax1.quiver(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], angles='xy',scale_units='xy',scale=1)
# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, B1_cAlh, A1_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, B1_ExtHII, A1_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, B1_Qz, A1_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)

#for label_, x, y in zip(label_sys, B1_sys, A1_sys):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax2.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
# ax1.legend(scatterpoints=1, ncol=3, fontsize=6.0, **lgd_kws)
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.1, box.height])
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), **lgd_kws)

#este es el de la leyenda
lgd = ax2.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid()
ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('indices1-jpas.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')    
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111)
ax3.set_xlim(xmin=-2.3,xmax=2.5)
ax3.set_ylim(ymin=-0.7,ymax=0.8)
plt.tick_params(axis='x', labelsize=15) 
#plt.tick_params(axis='y', labelsize=15)
#ax1.fill(x, y, color= 'k', alpha=0.1)
#ax1.add_artist(circle)
# ax1.set_xlim(xmin=-0.8,xmax=2.5)
#ax1.set_ylim(ymin=-1.8,ymax=5.0)
plt.xlabel('C1', fontsize= 16)
plt.ylabel('A1', fontsize= 16)
ax3.scatter(C1, A1, c='black', alpha=0.8, s=35, label='Halo PNe')
ax3.scatter(C1_CNP, B1_CNP,  c= "yellow", alpha=0.8, marker='o', label='Disk PN from SDSS')
ax3.scatter(C1_c, B1_c, c='purple', alpha=0.8, label='CVs from SDSS')
ax3.scatter(C1_L2d0, B1_L2d0,  c= "orange", alpha=0.8, marker='s', s=5 )
ax3.scatter(C1_L3d0, B1_L3d0,  c= "orange", alpha=0.8, marker='D', s=5)
ax3.scatter(C1_L4d0, B1_L4d0,  c= "orange", alpha=0.8, marker='^', s=5)
ax3.scatter(C1_L5d0, B1_L5d0,  c= "orange", alpha=0.8, marker='*', s=5)
ax3.scatter(C1_L2N0, B1_L2N0,  c= "green", alpha=0.8, marker='s', s=5)
ax3.scatter(C1_L3N0, B1_L3N0,  c= "green", alpha=0.8, marker='D', s=5)
ax3.scatter(C1_L4N0, B1_L4N0,  c= "green", alpha=0.8, marker='^', s=5)
ax3.scatter(C1_L5N0, B1_L5N0,  c= "green", alpha=0.8, marker='*', s=5)
ax3.scatter(C1_L2k0, B1_L2k0,  c= "brown", alpha=0.8, marker='s', s=5)
ax3.scatter(C1_L3k0, B1_L3k0,  c= "brown", alpha=0.8, marker='D', s=5)
ax3.scatter(C1_L4k0, B1_L4k0,  c= "brown", alpha=0.8, marker='^', s=5)
ax3.scatter(C1_L5k0, B1_L5k0,  c= "brown", alpha=0.8, marker='*', s=5)
ax3.scatter(C1_L2B0, B1_L2B0,  c= "cyan", alpha=0.8, marker='s', s=5)
ax3.scatter(C1_L3B0, B1_L3B0,  c= "cyan", alpha=0.8, marker='D', s=5)
ax3.scatter(C1_L4B0, B1_L4B0,  c= "cyan", alpha=0.8, marker='^', s=5)
ax3.scatter(C1_L5B0, B1_L5B0,  c= "cyan", alpha=0.8, marker='*', s=5)
ax3.scatter(C1_L2T0, B1_L2T0,  c= "magenta", alpha=0.8, marker='s', s=5)
ax3.scatter(C1_L3T0, B1_L3T0,  c= "magenta", alpha=0.8, marker='D', s=5)
ax3.scatter(C1_L4T0, B1_L4T0,  c= "magenta", alpha=0.8, marker='^', s=5)
ax3.scatter(C1_L5T0, B1_L5T0,  c= "magenta", alpha=0.8, marker='*', s=5)
ax3.scatter(C1_L2d01, B1_L2d01,  c= "orange", alpha=0.8, s=11,  marker='s')
ax3.scatter(C1_L3d01, B1_L3d01,  c= "orange", alpha=0.8, s=11,  marker='D')
ax3.scatter(C1_L4d01, B1_L4d01,  c= "orange", alpha=0.8, s=11,  marker='^')
ax3.scatter(C1_L5d01, B1_L5d01,  c= "orange", alpha=0.8, s=11,  marker='*')
ax3.scatter(C1_L2N01, B1_L2N01,  c= "green", alpha=0.8, s=11,  marker='s')
ax3.scatter(C1_L3N01, B1_L3N01,  c= "green", alpha=0.8, s=11,  marker='D')
ax3.scatter(C1_L4N01, B1_L4N01,  c= "green", alpha=0.8, s=11,  marker='^')
ax3.scatter(C1_L5N01, B1_L5N01,  c= "green", alpha=0.8, s=11,  marker='*')
ax3.scatter(C1_L2k01, B1_L2k01,  c= "brown", alpha=0.8, s=11,  marker='s')
ax3.scatter(C1_L3k01, B1_L3k01,  c= "brown", alpha=0.8, s=11,  marker='D')
ax3.scatter(C1_L4k01, B1_L4k01,  c= "brown", alpha=0.8, s=11,  marker='^')
ax3.scatter(C1_L5k01, B1_L5k01,  c= "brown", alpha=0.8, s=11,  marker='*')
ax3.scatter(C1_L2B01, B1_L2B01,  c= "cyan", alpha=0.8, s=11,  marker='s')
ax3.scatter(C1_L3B01, B1_L3B01,  c= "cyan", alpha=0.8, s=11,  marker='D')
ax3.scatter(C1_L4B01, B1_L4B01,  c= "cyan", alpha=0.8, s=11,  marker='^')
ax3.scatter(C1_L5B01, B1_L5B01,  c= "cyan", alpha=0.8, s=11,  marker='*')
ax3.scatter(C1_L2T01, B1_L2T01,  c= "magenta", alpha=0.8, s=11,  marker='s')
ax3.scatter(C1_L3T01, B1_L3T01,  c= "magenta", alpha=0.8, s=11,  marker='D')
ax3.scatter(C1_L4T01, B1_L4T01,  c= "magenta", alpha=0.8, s=11,  marker='^')
ax3.scatter(C1_L5T01, B1_L5T01,  c= "magenta", alpha=0.8, s=11,  marker='*')
ax3.scatter(C1_L2d02, B1_L2d02,  c= "orange", alpha=0.8, s=28,   marker='s', label='BB dddm1 L2')
ax3.scatter(C1_L3d02, B1_L3d02,  c= "orange", alpha=0.8, s=28,  marker='D', label='BB dddm1 L3')
ax3.scatter(C1_L4d02, B1_L4d02,  c= "orange", alpha=0.8, s=28,  marker='^', label='BB dddm1 L4')
ax3.scatter(C1_L5d02, B1_L5d02,  c= "orange", alpha=0.8, s=28,  marker='*', label='BB dddm1 L5')
ax3.scatter(C1_L2N02, B1_L2N02,  c= "green", alpha=0.8, s=28,  marker='s', label='BB N2242 L2')
ax3.scatter(C1_L3N02, B1_L3N02,  c= "green", alpha=0.8, s=28,  marker='D', label='BB N2242 L3')
ax3.scatter(C1_L4N02, B1_L4N02,  c= "green", alpha=0.8, s=28,  marker='^', label='BB N2242 L4')
ax3.scatter(C1_L5N02, B1_L5N02,  c= "green", alpha=0.8, s=28,  marker='*', label='BB N2242 L5')
ax3.scatter(C1_L2k02, B1_L2k02,  c= "brown", alpha=0.8, s=28,  marker='s', label='BB K648 L2')
ax3.scatter(C1_L3k02, B1_L3k02,  c= "brown", alpha=0.8, s=28,  marker='D', label='BB K648 L3')
ax3.scatter(C1_L4k02, B1_L4k02,  c= "brown", alpha=0.8, s=28,  marker='^', label='BB K648 L4')
ax3.scatter(C1_L5k02, B1_L5k02,  c= "brown", alpha=0.8, s=28,  marker='*', label='BB K648 L5')
ax3.scatter(C1_L2B02, B1_L2B02,  c= "cyan", alpha=0.8, s=28,  marker='s', label='BB BB1 L2')
ax3.scatter(C1_L3B02, B1_L3B02,  c= "cyan", alpha=0.8, s=28,  marker='D', label='BB BB1 L3')
ax3.scatter(C1_L4B02, B1_L4B02,  c= "cyan", alpha=0.8, s=28,  marker='^', label='BB BB1 L4')
ax3.scatter(C1_L5B02, B1_L5B02,  c= "cyan", alpha=0.8, s=28,  marker='*', label='BB BB1 L5')
ax3.scatter(C1_L2T02, B1_L2T02,  c= "magenta", alpha=0.8, s=28,  marker='s', label='BB Typ L2')
ax3.scatter(C1_L3T02, B1_L3T02,  c= "magenta", alpha=0.8, s=28,  marker='D', label='BB Typ L3')
ax3.scatter(C1_L4T02, B1_L4T02,  c= "magenta", alpha=0.8, s=28,  marker='^', label='BB Typ L4')
ax3.scatter(C1_L5T02, B1_L5T02,  c= "magenta", alpha=0.8, s=28,  marker='*',  label='BB Typ L5')
ax3.scatter(C1_Q401, B1_Q401,  c= "mediumaquamarine" , alpha=0.8, marker='s',  label='QSOs (4.01<z<5.0)')
ax3.scatter(C1_Qz, B1_Qz,  c= "royalblue", alpha=0.8, marker='D',  label='QSOs (3.01<z<4.0)')
ax3.scatter(C1_Q201, B1_Q201,  c= "goldenrod", alpha=0.8, marker='^',  label='QSOs (2.01<z<3.0)')
ax3.scatter(C1_Q101, B1_Q101,  c= "salmon", alpha=0.8, marker='*',  label='QSOs (1.01<z<2.0)')
ax3.scatter(C1_Q010, B1_Q010,  c= "sage", alpha=0.8, marker='o',  label='QSOs (0.01<z<1.0)')
ax3.scatter(C1_SFGs, B1_SFGs,  c= "white", alpha=0.3, marker='^', label='SFGs from SDSS')
ax3.scatter(C1_sys, B1_sys,  c= "red", alpha=0.8, marker='s', label='Munari Symbiotics')
ax3.scatter(C1_Extsys, B1_Extsys,  c= "red", alpha=0.8, marker='D', label='Symbiotics in NGC 55')
ax3.scatter(C1_sys_IPHAS, B1_sys_IPHAS,  c= "red", alpha=0.8, marker='^', label='Symbiotics from IPHAS')
ax3.scatter(C1_sysurvey, B1_sysurvey,  c= "red", alpha=0.8, marker='o', label='C. Cuil Symbiotics')
ax3.scatter(C1_ExtHII, B1_ExtHII,  c= "gray", alpha=0.8, marker='D', label='HII region in NGC 55')
ax3.scatter(C1_SN, B1_SN,  c= "black", alpha=0.8, marker='.', label='SN Remanents')
#ax1.scatter(C1_cAlh, A1_cAlh,  c= "greenyellow", alpha=0.8, marker='D', label='ALHAMBRA Candidates')
# ax1.text(0.05, 0.95, 'Symbol size of the models indicates extinction, E',
#            transform=ax1.transAxes, fontsize='x-small')
for label_, x, y in zip(label, C1, B1):
    ax3.annotate(label_, (x, y), alpha=0.9, size=8,
                   xytext=(3, 3), textcoords='offset points', ha='left', va='bottom',)

plt.annotate(
    '', xy=(C1_L2d0[0]-0.91, B1_L2d0[0]-0.91), xycoords='data',
    xytext=(C1_L2d02[0]-0.91, B1_L2d02[0]-0.91), textcoords='data',
    arrowprops={'arrowstyle': '->'})
plt.annotate(
    'Extinction', xy=(C1_L2d0[0]-1.2, B1_L2d0[0]-1.1), xycoords='data',
    xytext=(5, 0), textcoords='offset points', fontsize='x-small')
# ax1.arrow(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], fc="k", ec="k", head_width=0.05, head_length=0.1 )
# ax1.plot()
#ax1.quiver(B1_L2d0[0], A1_L2d0[0], B1_L2d02[0], A1_L2d02[0], angles='xy',scale_units='xy',scale=1)
# ax1.text(0.05, 0.95, 'Extinction, E0.1',
#            transform=ax1.transAxes, fontsize='x-small')
#for label_, x, y in zip(can_alh, B1_cAlh, A1_cAlh):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(3,-10), textcoords='offset points', ha='left', va='bottom',)

#for label_, x, y in zip(label_HII, B1_ExtHII, A1_ExtHII):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#for Z, x, y in zip(z, B1_Qz, A1_Qz):
    #ax1.annotate("{:.3f}".format(Z), (x, y), fontsize='x-small',
                       #xytext=(5,-5), textcoords='offset points', ha='left', bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.5}, alpha=0.7)

#for label_, x, y in zip(label_sys, B1_sys, A1_sys):
    #ax1.annotate(label_, (x, y), alpha=0.9, size=8,
                   #xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',)

#ax1.set_title(" ".join([cmd_args.source]))
#ax1.grid(True)
#ax1.annotate('Higher z(3.288)', xy=(0.08749580383300781, 0.181182861328125), xytext=(-0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
#ax1.annotate('Lower z(3.065)', xy=(0.3957328796386719, 0.1367034912109375), xytext=(0.5, -0.58),
             #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
ax3.minorticks_on()
#ax1.grid(which='minor')#, lw=0.3)
# ax1.legend(scatterpoints=1, ncol=3, fontsize=6.0, **lgd_kws)
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.1, box.height])
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), **lgd_kws)

#este es el de la leyenda
lgd = ax3.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax1.grid()
ax3.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.savefig('indices2-jpas.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


