'''
To make SEDs of JPLUS objets
'''
from __future__ import print_function
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

parser = argparse.ArgumentParser(
    description="""Estimate magnitudes of a spectrum for any photometryc system""")

parser.add_argument("source", type=str,
                    default="H4-1-mag-sloan",
                    help="Name of source, taken the prefix ")

parser.add_argument("--fitfile", type=str,
                    default="spec-2241-54156-0108_H41_SLOAN-HPNe-JPLUS13-magnitude",
                    help="File with convolved magnitudes")


parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
regionfile = cmd_args.source + ".dat"

# Get any convolved magnitudes

objectfile = cmd_args.fitfile + ".json"




filters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
name_filters = []
mag = []
err = []
filters_sdss = ['6', '8', '10', '12']
filters_conv = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']


# sdss_H4 = [16.12, 16.32, 17.99, 18.11] #h4-1    from SLOAN catag
# sdss_PNG135 = [ 18.19, 18.73, 18.98] #PNG 135
                         
pattern = "*.apdat"
file_list = glob.glob(pattern)

def load_file(X):
    conv = []
    with open(X) as f:
         data = json.load(f)
         for k in sorted(data.keys()):
             if k.startswith('F'):
                 imagename=k
                 if np.isfinite(data[imagename]):
                     conv.append(data[imagename])
    return conv   


for file_name in sorted(file_list):
    x = np.loadtxt(file_name,  delimiter = None, converters = None, skiprows = 0, 
                                       usecols = None, unpack = False, ndmin = 0,)
    mag.append(x[7]) #h4-1
    err.append(x[12]) #h4-1
    #mag.append(x[7]) #PNG 135
    name_filters.append(file_name.split('p9_J')[-1].split('_s')[0])
# for a, b, c in zip(filters, name_filters, mag):
#     print(a, b, c)
    #print(file_name.split('-1_')[-1].split('_s')[0])


# Convolved for medirec = "Halo-PNe-spectros/"
#file_list = glob.glob(pattern)

#with open("spec-0953-52411-0160_PNG_1359559-HPNe-JPLUS13-magnitude.json") as f: # PNG 135

# with open("spec-2241-54156-0108_H41_SLOAN-HPNe-JPLUS13-magnitude.json") as f:    #H4 1
#     data = json.load(f)

#     for k in sorted(data.keys()):
#         if k.startswith('F'):
#             imagename=k
#             if np.isfinite(data[imagename]):
#                 conv.append(data[imagename])

sdss = np.loadtxt(regionfile)

conv= load_file(objectfile)

# magnitude calibrate using sloan
m_cal = np.array(mag)               #H4 1
#m_cal = 7.60 + 0.54*np.array(mag)               # PNG 135
# propagation error
#err_cal = 0.86*abs(np.array(err))               # H4 1
err_cal = 0.54*abs(np.array(err))               # PNG 135

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': True}
sns.set(style="dark")

if cmd_args.savefig:
    plotfile = regionfile.replace(".dat", 
                    "-maginst-conv.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlim(xmin=0.0,xmax=13.2)
    #ax1.set_ylim(ymin=14,ymax=23)     #H4 1
    ax1.set_ylim(ymin=16.1,ymax=23)  #PNG135
#ax1.set_xlabel(r'$\lbda$')
    ax1.set_xlabel(r'Filter')
    ax1.set_ylabel(r'Magnitude')
    ax1.plot(filters, mag, 'ko-', label="JPLUS images (instrumental)")
    (_, caps, _) = ax1.errorbar(filters, mag,  yerr = err, marker='o', fmt='ko-', markersize=8)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax1.plot(filters_sdss, sdss[:,0], 'bo-', label="SDSS catalog")
    (_, caps1, _) = ax1.errorbar(filters_sdss, sdss[:,0], yerr=sdss[:,1], marker='o', fmt='bo-')
    for cap in caps1:
        cap.set_markeredgewidth(1)
    ax1.plot(filters_conv, conv, 'ro-', label="Convolved spectrum (SDSS)")
    ax1.plot(filters, m_cal, 'go-', label="Calibra by fit: y = 1.26+0.86*x") #H4 1
    (_, caps2, _) = ax1.errorbar(filters, m_cal, yerr=err_cal, marker='o', fmt='go-')
    for cap in caps2:
        cap.set_markeredgewidth(1)
         
    #ax1.plot(filters, m_cal, 'go-', label="mag cal by fit: y = 7.60 + 0.54x") #PNG
#ax1.set_title(" ".join([cmd_args.source]))
#ax1.plot(Wavelengthh, Fluxx, 'k-')
#ax1.grid(True)
    ax1.set_xticks(np.arange(len(filters)+1))                     # put all value of the list
    plt.grid()
#sns.despine(bottom=True)
#ax1.minorticks_on()
#ax1.grid(which='minor', lw=0.3)
#plt.tight_layout()
#plt.margins(0.06)
#plt.subplots_adjust(bottom=0.17)
# plt.annotate(
#     '1:uJAVA, 2:F0378, 3:J0395, 4:J0410, 5:J0430, 6:gSDSS, 7:J0515, 8:rSDSS, 9:J0660, 10:iSDSS, 11:J0861, 12:zSDSS', xy=(filters, mag), xycoords='data',
#     xytext=(5, 0), textcoords='offset points', fontsize='x-small')

    ax1.text(0.8, 16.2, '1:uJAVA, 2:F0378, 3:J0395, 4:J0410, 5:J0430, 6:gSDSS, 7:J0515, 8:rSDSS, 9:J0660, 10:iSDSS, 11:J0861, 12:zSDSS', style='italic',
         bbox={'facecolor':'white', 'alpha':0.8, 'pad':10}, fontsize=7) ## 14.2 for H4 1 nad 16.2 for PNG 135

    lgd = ax1.legend()

    plt.savefig(plotfile)
#plt.savefig("PNG135-plot-mag-jplusimages.pdf")



