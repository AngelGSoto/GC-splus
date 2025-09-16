'''
Cuting images fits
Based in pyFIST.py and extract-image.py from Henney program

'''
from __future__ import print_function
import numpy as np
import json
import os
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u 
import argparse
import sys


parser = argparse.ArgumentParser(
    description="""Cut images from fits files""")

parser.add_argument("source", type=str,
                    default="H-10b.0016",
                    help="Name of source (prefix for files) ")

parser.add_argument("--debug", action="store_true",
                    help="Print out verbose debugging info about each line in region file")

args = parser.parse_args()
regionfile = args.source + "_swp"

hdu = fits.open(regionfile)
crop_coords_unit=u.degree
position = "position.reg"
ra, dec = [], []

f = open(position, 'r')
header1 = f.readline()
header2 = f.readline()
header3 = f.readline()
for line in f:
    line = line.strip()
    columns = line.split()
    coor = line.split("(")[-1].split("\"")[0]
    ra1, dec1 = coor.split(",")[0:2]
    crop_c = coord.SkyCoord(ra1, dec1, unit=(u.hourangle, u.degree))
    
   
#locc = sys.argv[1:]
# ra = input('Enter RA: ')
# dec = input('Enter DEC: ')
# ra = args.ra
# dec = args.dec

    w = wcs.WCS(hdu[0].header)
    print(w)
#crop_coords = np.array(w.wcs_pix2world(hdu[0].data.shape[0]/2., 
				       #hdu[0].data.shape[1]/2., 0))
  
#crop_c = coord.SkyCoord(crop_coords[0], crop_coords[1], unit=u.degree)

#crop_radius=input('Enter Radius: ')
    crop_radius = 6.90*u.arcsec
    pix_scale = 0.0996*u.arcsec

    crop_c_pix = w.wcs_world2pix(crop_c.ra.degree, crop_c.dec.degree, 0)
    crop_radius_pixels = crop_radius.to(u.arcsec) / pix_scale.to(u.arcsec)
   
    x1 = int(np.clip(crop_c_pix[0]-crop_radius_pixels, 0, hdu[0].data.shape[0]-1))
    x2 = int(np.clip(crop_c_pix[0]+crop_radius_pixels, 0, hdu[0].data.shape[0]-1))
    y1 = int(np.clip(crop_c_pix[1]-crop_radius_pixels, 0, hdu[0].data.shape[1]-1))
    y2 = int(np.clip(crop_c_pix[1]+crop_radius_pixels, 0, hdu[0].data.shape[1]-1))
    

    hdu[0].data = hdu[0].data[y1:y2, x1:x2]
    
    hdu[0].header['CRPIX1'] -= x1
    hdu[0].header['CRPIX2'] -= y1
    # hdu[0].header['CRVAL1'] = crop_c.ra.degree
    # hdu[0].header['CRVAL2'] = crop_c.dec.degree
    w = WCS(hdu[0].header)

################### 
#Save the new file#
###################
    outfile = regionfile.replace("_swp", "_swp-crop.fits")
    new_hdu = fits.PrimaryHDU(hdu[0].data, header=hdu[0].header)
    new_hdu.writeto(outfile, output_verify="fix", clobber=True)
