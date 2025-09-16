'''
Read the file.fits and write a ASCII file with wavelenght and flux by Using Pyraf 
'''
from __future__ import print_function
import glob
import numpy as np
import argparse
from pyraf import iraf

parser = argparse.ArgumentParser(
    description="""Write wave and flux of a spectrum""")

parser.add_argument("source", type=str,
                    default="DDDM1",
                    help="Name of source, taken the prefix ")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + ".fit"

iraf.wspectext(regionfile, regionfile.replace(".fit", ".dat"), 
               header=False)

if args.debug:
    print("Spectrum converted to ascii:", regionfile)
