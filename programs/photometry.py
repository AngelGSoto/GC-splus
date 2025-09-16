'''
Making 
'''
import astropy.io.fits as fits
from astropy import wcs
import numpy as np
from pyraf import iraf
from iraf import phot
from phot import aperphot

img='../JPLUS-data/NGC2242/1500054-NGC2242_iSDSS_swp.fits'  #path to the image
RA = 98.530632435
DEC = +44.777164556
hdulist = fits.open(img)
w = wcs.WCS(hdulist['PRIMARY'].header)
world = np.array([[RA, DEC]])
pix = w.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
print "Pixel Coordinates: ", pix[0,0], pix[0,1]

#call aperture function
observation=aperphot(img, timekey=None, pos=[pix[0,0], pix[0,1]], dap=[4,8,12], resamp=2, retfull=False)

# Print outputs
print "Aperture flux:", observation.phot
print "Background:   ", observation.bg
