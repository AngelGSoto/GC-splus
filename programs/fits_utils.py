'''
Utility functions for cutting FITS images.
This module contains shared functions for cropping FITS files.
'''
from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u


def read_position_file(position_file):
    """
    Read coordinates from a position.reg file.
    
    Parameters:
    -----------
    position_file : str
        Path to the position file
        
    Returns:
    --------
    list : List of SkyCoord objects
    """
    coords = []
    with open(position_file, 'r') as f:
        # Skip header lines
        f.readline()
        f.readline()
        f.readline()
        
        for line in f:
            line = line.strip()
            coor = line.split("(")[-1].split("\"")[0]
            ra1, dec1 = coor.split(",")[0:2]
            crop_c = coord.SkyCoord(ra1, dec1, unit=(u.hourangle, u.degree))
            coords.append(crop_c)
    
    return coords


def crop_fits_image(hdu, crop_coord, crop_radius, pix_scale, update_crval=False):
    """
    Crop a FITS image around a specific coordinate.
    
    Parameters:
    -----------
    hdu : astropy.io.fits.HDUList
        FITS HDU list
    crop_coord : astropy.coordinates.SkyCoord
        Sky coordinate to center the crop
    crop_radius : astropy.units.Quantity
        Radius of the crop in angular units
    pix_scale : astropy.units.Quantity
        Pixel scale in angular units per pixel
    update_crval : bool
        If True, update CRVAL1/CRVAL2 to the crop center
        
    Returns:
    --------
    astropy.io.fits.PrimaryHDU : Cropped HDU
    """
    w = wcs.WCS(hdu[0].header)
    
    # Convert crop center to pixel coordinates
    crop_c_pix = np.array(w.wcs_world2pix(crop_coord.ra.degree, crop_coord.dec.degree, 0))
    crop_radius_pixels = crop_radius.to(u.arcsec) / pix_scale.to(u.arcsec)
    
    # Calculate crop boundaries
    x1 = int(np.clip(crop_c_pix[0] - crop_radius_pixels, 0, hdu[0].data.shape[0] - 1))
    x2 = int(np.clip(crop_c_pix[0] + crop_radius_pixels, 0, hdu[0].data.shape[0] - 1))
    y1 = int(np.clip(crop_c_pix[1] - crop_radius_pixels, 0, hdu[0].data.shape[1] - 1))
    y2 = int(np.clip(crop_c_pix[1] + crop_radius_pixels, 0, hdu[0].data.shape[1] - 1))
    
    # Crop the data
    cropped_data = hdu[0].data[y1:y2, x1:x2]
    
    # Update header
    new_header = hdu[0].header.copy()
    if update_crval:
        # Set CRPIX to the center of the cropped image
        new_header['CRPIX1'] = (cropped_data.shape[0] - 0.5) / 2.
        new_header['CRPIX2'] = (cropped_data.shape[1] - 0.5) / 2.
        # Set CRVAL to the crop center
        new_header['CRVAL1'] = crop_coord.ra.degree
        new_header['CRVAL2'] = crop_coord.dec.degree
    else:
        # Adjust CRPIX by the crop offset
        new_header['CRPIX1'] -= x1
        new_header['CRPIX2'] -= y1
    
    # Create new HDU
    new_hdu = fits.PrimaryHDU(cropped_data, header=new_header)
    
    return new_hdu


def save_cropped_fits(hdu, input_filename, output_suffix):
    """
    Save a cropped FITS file.
    
    Parameters:
    -----------
    hdu : astropy.io.fits.PrimaryHDU
        HDU to save
    input_filename : str
        Input filename to derive output name from
    output_suffix : str
        Suffix to use for output file (will replace input extension/suffix)
    """
    outfile = input_filename.replace(".fits", output_suffix).replace("_swp", output_suffix.split('.')[0])
    hdu.writeto(outfile, output_verify="fix", overwrite=True)
    return outfile
