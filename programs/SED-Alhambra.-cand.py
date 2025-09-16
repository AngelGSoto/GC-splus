'''
Make SEDs of ALHAMBRA candidates
'''
from __future__ import print_function
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

filters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
mag = []

parser = argparse.ArgumentParser(
    description="""Estimate Alhambra magnitudes of a spectrum""")

parser.add_argument("source", type=str,
                    default="mag-1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--savefig", action="store_true",
                    help="Save a figure showing the magnitude")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

cmd_args = parser.parse_args()
regionfile = cmd_args.source + "-mag.txt"

f = open(regionfile, 'r')
for line in f:
    line = line.strip()
    columns = line.split()
    mag0 = columns[0]
    mag1 = columns[7]
    mag2 = columns[9]
    mag3 = columns[11]
    mag4 = columns[13]
    mag5 = columns[15]
    mag6 = columns[17]
    mag7 = columns[19]
    mag8 = columns[21]
    mag9 = columns[23]
    mag10 = columns[25]
    mag11 = columns[27]
    mag12 = columns[29]
    mag13 = columns[31]
    mag14 = columns[33]
    mag15 = columns[35]
    mag16 = columns[37]
    mag17 = columns[39]
    mag18 = columns[41]
    mag19 = columns[43]
    mag20 = columns[45]
    mag.append(mag1)
    mag.append(mag2) 
    mag.append(mag3) 
    mag.append(mag4)
    mag.append(mag5) 
    mag.append(mag6) 
    mag.append(mag7) 
    mag.append(mag8) 
    mag.append(mag9) 
    mag.append(mag10)
    mag.append(mag11)
    mag.append(mag12)
    mag.append(mag13)
    mag.append(mag14)
    mag.append(mag15) 
    mag.append(mag16) 
    mag.append(mag17) 
    mag.append(mag18) 
    mag.append(mag19) 
    mag.append(mag20) 
   
sns.set(style="dark") #context="talk")
if cmd_args.savefig:
    plotfile = regionfile.replace("-mag.txt", "-SED.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    #ax1.set_xlim(xmin=0.0,xmax=20)
    #ax1.set_ylim(ymin=22,ymax=26)
    #ax1.set_xlabel(r'$\lambda$')
    ax1.set_xlabel(r'Filter')
    ax1.set_ylabel(r'Magnitude')
    ax1.plot(filters, mag, 'ko-')
    ax1.set_title(" ".join([cmd_args.source]))
    #ax1.plot(Wavelengthh, Fluxx, 'k-')
    #ax1.grid(True)
    sns.despine(bottom=True)
    ax1.minorticks_on()
    ax1.grid(which='minor', lw=0.3)
    plt.tight_layout()
    plt.margins(0.06)
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(plotfile)
