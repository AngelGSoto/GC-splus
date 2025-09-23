#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected.py - Versión corregida que aplica las mismas correcciones
de fondo y apertura que se usan para los cúmulos globulares.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std, sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
BASE_DATA_DIR = "."

def model_background_residuals(data, background_box_size=100):
    """Modela y resta residuos de fondo"""
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        mask = data > median + 5 * std
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((15, 15)))
        
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(data, 
                          box_size=background_box_size, 
                          filter_size=5,
                          sigma_clip=sigma_clip, 
                          bkg_estimator=MedianBackground(), 
                          mask=dilated_mask,
                          exclude_percentile=10)
        
        data_corrected = data - bkg.background
        bkg_rms = bkg.background_rms_median
        logging.info(f"Background residuals modeled with box_size={background_box_size}")
        return data_corrected, bkg_rms
        
    except Exception as e:
        logging.warning(f"Background residual modeling failed: {e}")
        return data, np.nanstd(data)

def calculate_splus_official_aperture_correction(data, header, star_positions, filter_name):
    """Calcula corrección de apertura según metodología S-PLUS"""
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        aperture_diameters = np.linspace(1, 50, 32)
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
            logging.warning(f"Insufficient valid stars for aperture correction")
            return 0.0, 1.0, 0
        
        normalized_curves = growth_fluxes[:, valid_mask] / total_fluxes[valid_mask]
        mean_growth = np.median(normalized_curves, axis=1)
        
        aperture_3arcsec_idx = np.argmin(np.abs(aperture_diameters - 3))
        fraction_in_3arcsec = mean_growth[aperture_3arcsec_idx]
        
        if fraction_in_3arcsec <= 0:
            return 0.0, 1.0, 0
        
        aperture_correction_mag = -2.5 * np.log10(fraction_in_3arcsec)
        aperture_correction_flux = 1.0 / fraction_in_3arcsec
        
        n_stars = np.sum(valid_mask)
        logging.info(f"{filter_name}: Aperture correction = {aperture_correction_mag:.3f} mag")
        
        return aperture_correction_mag, aperture_correction_flux, n_stars
        
    except Exception as e:
        logging.error(f"Error in aperture correction: {e}")
        return 0.0, 1.0, 0

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia corregido"""
    try:
        ref_catalog = pd.read_csv(catalog_path)
        
        with fits.open(image_path) as hdul:
            wcs = None
            data_shape = None
            
            for i, hdu in enumerate(hdul):
                try:
                    if hasattr(hdu, 'header') and hasattr(hdu, 'data') and hdu.data is not None:
                        temp_wcs = WCS(hdu.header)
                        wcs = temp_wcs
                        data_shape = hdu.data.shape
                        header = hdu.header
                        logging.info(f"Using HDU {i} with shape: {data_shape}")
                        break
                except:
                    continue
            
            if wcs is None:
                logging.error(f"Could not find valid WCS in {image_path}")
                return np.array([]), pd.DataFrame()
        
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
        
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        
        if data_shape is not None:
            margin_pixels = int(25 / 0.55)
            valid_mask = valid_mask & (x > margin_pixels) & (x < data_shape[1] - margin_pixels)
            valid_mask = valid_mask & (y > margin_pixels) & (y < data_shape[0] - margin_pixels)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Found {len(positions)} valid positions")
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error getting reference positions: {e}")
        return np.array([]), pd.DataFrame()

def extract_stars_corrected(image_path, reference_positions, aperture_diameter=3):
    """Extrae estrellas con metodología corregida"""
    try:
        with fits.open(image_path) as hdul:
            data = None
            header = None
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            
            if data is None:
                logging.error(f"No data found in {image_path}")
                return pd.DataFrame()
        
        pixscale = header.get('PIXSCALE', 0.55)
        data_corrected, bkg_rms = model_background_residuals(data)
        
        try:
            wcs = WCS(header)
            has_wcs = True
        except:
            has_wcs = False
        
        positions = reference_positions
        sources = pd.DataFrame({'xcentroid': positions[:, 0], 'ycentroid': positions[:, 1]})
        
        # Calcular corrección de apertura
        ap_corr_mag, ap_corr_flux, n_stars_ap = calculate_splus_official_aperture_correction(
            data_corrected, header, positions, os.path.basename(image_path))
        
        # Fotometría
        aperture_radius_pixels = (aperture_diameter / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(positions, r=aperture_radius_pixels)
        annulus = CircularAnnulus(positions, r_in=ann_in_px, r_out=ann_out_px)
        
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        flux_corrected = flux * ap_corr_flux
        
        # Cálculo de errores
        flux_err = np.sqrt(np.abs(flux_corrected) + (apertures.area * bkg_rms**2))
        
        min_flux = 1e-10
        mag_inst_uncorrected = -2.5 * np.log10(np.maximum(flux, min_flux))
        mag_inst_corrected = -2.5 * np.log10(np.maximum(flux_corrected, min_flux))
        snr = np.where(flux_err > 0, flux_corrected / flux_err, 0.0)
        
        # Resultados
        sources['flux_adu_uncorrected'] = flux
        sources['flux_adu'] = flux_corrected
        sources['flux_err'] = flux_err
        sources['mag_inst_uncorrected'] = mag_inst_uncorrected
        sources['mag_inst'] = mag_inst_corrected
        sources['snr'] = snr
        sources['ap_correction_mag'] = ap_corr_mag
        sources['ap_correction_flux'] = ap_corr_flux
        sources['n_stars_ap_corr'] = n_stars_ap
        
        if has_wcs:
            ra, dec = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
            sources['ra'] = ra
            sources['dec'] = dec
        else:
            sources['ra'] = np.nan
            sources['dec'] = np.nan
            
        return sources[['ra', 'dec', 'flux_adu_uncorrected', 'flux_adu', 'flux_err', 
                       'mag_inst_uncorrected', 'mag_inst', 'snr', 
                       'ap_correction_mag', 'ap_correction_flux', 'n_stars_ap_corr']]
        
    except Exception as e:
        logging.error(f"Error in star extraction: {e}")
        return pd.DataFrame()

def process_field_corrected(field_name):
    """Procesa un campo completo"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO: {field_name}")
    logging.info(f"{'='*60}")
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Field directory not found: {field_dir}")
        return False
    
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_splus_method.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Input catalog not found: {input_catalog}")
        return False
    
    band_catalogs = {}
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = os.path.join(field_dir, f"{field_name}_{band}.fits.fz")
        if not os.path.exists(image_file):
            image_file = os.path.join(field_dir, f"{field_name}_{band}.fits")
            if not os.path.exists(image_file):
                logging.warning(f"Image {field_name}_{band} not found")
                continue
        
        if reference_positions is None:
            reference_positions, ref_catalog = get_reference_positions_from_catalog(
                input_catalog, image_file)
            if len(reference_positions) == 0:
                logging.error("No valid reference positions found")
                return False
        
        cat = extract_stars_corrected(image_file, reference_positions)
        if len(cat) == 0:
            logging.warning(f"{band}: No stars extracted")
            continue
            
        band_catalogs[band] = cat
        logging.info(f"{band}: {len(cat)} stars extracted")

    if not band_catalogs:
        logging.error("No stars extracted from any band")
        return False

    # Combinar resultados
    combined_catalog = ref_catalog.copy()
    for band, cat in band_catalogs.items():
        if len(cat) != len(combined_catalog):
            min_len = min(len(cat), len(combined_catalog))
            cat = cat.iloc[:min_len]
            combined_catalog = combined_catalog.iloc[:min_len]
        
        combined_catalog[f'mag_inst_uncorrected_{band}'] = cat['mag_inst_uncorrected'].values
        combined_catalog[f'mag_inst_{band}'] = cat['mag_inst'].values
        combined_catalog[f'flux_adu_{band}'] = cat['flux_adu'].values
        combined_catalog[f'flux_err_{band}'] = cat['flux_err'].values
        combined_catalog[f'snr_{band}'] = cat['snr'].values
        combined_catalog[f'ap_corr_mag_{band}'] = cat['ap_correction_mag'].values
        combined_catalog[f'ap_corr_flux_{band}'] = cat['ap_correction_flux'].values
    
    combined_catalog.to_csv(output_catalog, index=False)
    logging.info(f"Results saved to: {output_catalog}")
    
    # Estadísticas
    for band in band_catalogs.keys():
        mag_col = f'mag_inst_{band}'
        if mag_col in combined_catalog.columns:
            valid_mags = combined_catalog[mag_col][combined_catalog[mag_col] < 50]
            if len(valid_mags) > 0:
                mean_mag = valid_mags.mean()
                logging.info(f"{band}: Mean mag = {mean_mag:.2f} (n={len(valid_mags)})")
    
    return True

def main():
    test_fields = ['CenA01']  # Probar con uno primero
    
    for field in test_fields:
        success = process_field_corrected(field)
        if not success:
            logging.error(f"Error processing {field}")
        time.sleep(2)
    
    logging.info("Processing completed")

if __name__ == '__main__':
    main()
