#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected.py - ESTRELLAS DE REFERENCIA (SIN aperture correction)
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAMETER = 3  # 3 arcsec para estrellas de referencia

def extract_instrumental_magnitudes_uncorrected(image_path, reference_positions, field_name, filter_name):
    """
    Extrae magnitudes instrumentales SIN APERTURE CORRECTION para estrellas de referencia
    """
    try:
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            
            if data is None:
                logging.error(f"No se encontraron datos en {image_path}")
                return pd.DataFrame()
        
        pixscale = header.get('PIXSCALE', 0.55)
        
        # 1. Verificación mínima de fondo (conservadora para SPLUS)
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        if abs(median) < 0.1 * std:  # Fondo ya restado
            data_corrected = data
            bkg_rms = std
            logging.info(f"{filter_name}: Usando imagen con fondo ya restado")
        else:
            # Corrección mínima si es necesario
            data_corrected, bkg_rms = apply_minimal_background_correction(data, filter_name)
        
        # 2. Fotometría SIN aperture correction
        aperture_radius_px = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_px)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        # Fotometría básica
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        # Restar fondo local
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # 3. CALCULAR factores de aperture correction (pero NO aplicarlos)
        aperture_corrections = calculate_aperture_correction_factors(
            data_corrected, header, reference_positions, filter_name)
        
        # 4. Magnitudes instrumentales SIN CORREGIR (para zero points)
        min_flux = 1e-10
        mag_inst_uncorrected = -2.5 * np.log10(np.maximum(flux, min_flux))
        
        # Error estimation
        flux_err = np.sqrt(np.abs(flux) + (apertures.area * bkg_rms**2))
        snr = np.where(flux_err > 0, flux / flux_err, 0.0)
        
        # 5. Resultados - magnitudes SIN corregir + factores de corrección
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            f'flux_uncorrected_{filter_name}': flux,  # Flux sin corregir
            f'flux_err_{filter_name}': flux_err,
            f'mag_inst_uncorrected_{filter_name}': mag_inst_uncorrected,  # Mag sin corregir
            f'snr_{filter_name}': snr,
            # Factores de corrección para usar en cúmulos
            f'ap_corr_3_{filter_name}': aperture_corrections.get(3, 1.0),
            f'ap_corr_4_{filter_name}': aperture_corrections.get(4, 1.0),
            f'ap_corr_5_{filter_name}': aperture_corrections.get(5, 1.0),
            f'ap_corr_6_{filter_name}': aperture_corrections.get(6, 1.0),
        })
        
        # Añadir coordenadas
        try:
            wcs = WCS(header)
            ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
        except:
            results['ra'] = np.nan
            results['dec'] = np.nan
        
        logging.info(f"{filter_name}: {len(results)} estrellas, mag promedio (sin corregir) = {mag_inst_uncorrected.mean():.2f}")
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        return pd.DataFrame()

def calculate_aperture_correction_factors(data, header, star_positions, filter_name, max_stars=50):
    """
    Calcula factores de aperture correction para usar en cúmulos
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        
        # Seleccionar estrellas brillantes para el cálculo
        if len(star_positions) > max_stars:
            temp_aperture = CircularAperture(star_positions, r=5/pixscale)
            temp_phot = aperture_photometry(data, temp_aperture)
            fluxes = temp_phot['aperture_sum'].data
            bright_indices = np.argsort(fluxes)[-max_stars:]
            star_positions = star_positions[bright_indices]
        
        # Curva de crecimiento
        aperture_diameters = np.array([1, 2, 3, 4, 5, 6, 8, 12, 20, 30, 50])
        aperture_radii_px = (aperture_diameters / 2) / pixscale
        
        growth_fluxes = []
        for radius in aperture_radii_px:
            apertures = CircularAperture(star_positions, r=radius)
            phot_table = aperture_photometry(data, apertures)
            growth_fluxes.append(phot_table['aperture_sum'].data)
        
        growth_fluxes = np.array(growth_fluxes)
        total_fluxes = growth_fluxes[-1]
        valid_mask = total_fluxes > 0
        
        if np.sum(valid_mask) < 5:
            return {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
        
        normalized_curves = growth_fluxes[:, valid_mask] / total_fluxes[valid_mask]
        median_growth = np.median(normalized_curves, axis=1)
        
        # Factores para diferentes aperturas
        corrections = {}
        aperture_sizes = [3, 4, 5, 6]
        
        for ap_size in aperture_sizes:
            target_radius_px = (ap_size / 2) / pixscale
            idx = np.argmin(np.abs(aperture_radii_px - target_radius_px))
            fraction = median_growth[idx]
            
            if fraction > 0 and fraction <= 1:
                corrections[ap_size] = 1.0 / fraction
            else:
                corrections[ap_size] = 1.0
        
        logging.info(f"{filter_name}: Factores de corrección: {corrections}")
        return corrections
        
    except Exception as e:
        logging.error(f"Error calculando factores de corrección: {e}")
        return {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}

def apply_minimal_background_correction(data, filter_name):
    """Aplica corrección mínima de fondo si es necesario"""
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        if std < 2.0:
            return data, std
        
        # Corrección conservadora
        mask = data > median + 20 * std
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(data, 
                          box_size=200,
                          filter_size=5,
                          sigma_clip=sigma_clip,
                          bkg_estimator=MedianBackground(),
                          mask=dilated_mask,
                          exclude_percentile=30)
        
        data_corrected = data - bkg.background
        return data_corrected, bkg.background_rms_median
        
    except Exception as e:
        logging.warning(f"Corrección de fondo falló: {e}")
        return data, np.std(data)

# [Resto del script similar al anterior...]
