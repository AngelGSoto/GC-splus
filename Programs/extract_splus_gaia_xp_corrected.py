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
from astropy.coordinates import SkyCoord
from astropy.units import u
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
from astroquery.gaia import Gaia
from scipy.spatial import KDTree
from gaiaxpy import calibrate
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --------------------------- CONFIGURACIÓN ---------------------------
SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
FIELD_RADIUS_DEG = 0.3
MIN_CAL_STARS = 10
BASE_DATA_DIR = "."

# --------------------------- FUNCIONES MEJORADAS ---------------------------

def model_background_residuals(data, background_box_size=100):
    """
    Modela y resta residuos de fondo a gran escala (igual que en el script de cúmulos globulares).
    """
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Crear máscara para excluir objetos brillantes
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
        logging.warning(f"Background residual modeling failed: {e}. Using original image.")
        return data, np.nanstd(data)

def compute_empirical_aperture_correction(data, header, positions, pixscale, 
                                         small_radius_arcsec=1.5, large_radius_arcsec=6.0):
    """
    Calcula corrección de apertura empírica usando las mismas estrellas de referencia.
    """
    try:
        small_r_px = small_radius_arcsec / pixscale
        large_r_px = large_radius_arcsec / pixscale

        # Crear aperturas
        small_ap = CircularAperture(positions, r=small_r_px)
        large_ap = CircularAperture(positions, r=large_r_px)

        # Fotometría
        small_tab = aperture_photometry(data, small_ap)
        large_tab = aperture_photometry(data, large_ap)

        flux_small = small_tab['aperture_sum']
        flux_large = large_tab['aperture_sum']
        
        # Filtrar mediciones buenas
        mask_good = (flux_small > 0) & (flux_large > 0) & (flux_large > flux_small)
        
        if mask_good.sum() < 5:
            logging.warning("Insufficient good stars for aperture correction measurement.")
            return 1.0, 0.0, int(mask_good.sum())

        # Calcular corrección
        corr = (flux_large[mask_good] / flux_small[mask_good])
        median_corr = np.median(corr)
        corr_std = np.std(corr)
        
        return float(median_corr), float(corr_std), int(mask_good.sum())
        
    except Exception as e:
        logging.error(f"Error in aperture correction calculation: {e}")
        return 1.0, 0.0, 0

def extract_stars_corrected(image_path, reference_positions=None, threshold_sigma=3, 
                           background_box_size=100, aperture_diameter=3):
    """
    Extrae estrellas aplicando las mismas correcciones que para los cúmulos globulares.
    """
    try:
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                logging.error(f"No data found in {image_path}")
                return pd.DataFrame()
        
        # Obtener parámetros del header
        pixscale = header.get('PIXSCALE', 0.55)
        gain = float(header.get('GAIN', header.get('CCDGAIN', 1.0)))
        rdnoise = float(header.get('RDNOISE', 0.0))
        saturate_level = float(header.get('SATURATE', np.inf))
        fwhm_arcsec = header.get('FWHMMEAN', 1.25)
        
        logging.info(f"Processing {os.path.basename(image_path)} with PIXSCALE={pixscale}")
        
        # Aplicar corrección de fondo (igual que para cúmulos globulares)
        data_corrected, bkg_rms = model_background_residuals(data, background_box_size)
        
        # Intentar obtener WCS
        try:
            wcs = WCS(header)
            has_wcs = True
        except:
            logging.warning(f"Could not interpret WCS in {image_path}")
            has_wcs = False
        
        # Si no se proporcionan posiciones, detectar estrellas
        if reference_positions is None:
            fwhm_pixels = fwhm_arcsec / pixscale
            std = mad_std(data_corrected)
            daofind = DAOStarFinder(fwhm=fwhm_pixels, threshold=threshold_sigma*std)
            sources = daofind(data_corrected)
            
            if sources is None:
                logging.warning(f"No stars detected in {image_path}")
                return pd.DataFrame()
                
            sources = sources.to_pandas()
            positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        else:
            # Usar las posiciones de referencia proporcionadas
            positions = reference_positions
            sources = pd.DataFrame({'xcentroid': positions[:, 0], 'ycentroid': positions[:, 1]})
        
        # Fotometría de apertura con el mismo método que para cúmulos globulares
        aperture_radius_pixels = (aperture_diameter / 2.0) / pixscale
        
        # Añadir anillo de fondo (igual que en el script de cúmulos)
        ann_in_px = 6.0 / pixscale
        ann_out_px = 9.0 / pixscale
        
        apertures = CircularAperture(positions, r=aperture_radius_pixels)
        annulus = CircularAnnulus(positions, r_in=ann_in_px, r_out=ann_out_px)
        
        # Fotometría
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        # Calcular fondo por fuente
        annulus_area = annulus.area
        bkg_mean_per_source = bkg_phot_table['aperture_sum'] / annulus_area
        
        # Flux en ADU (sin normalizar por EXPTIME)
        flux = phot_table['aperture_sum'] - (bkg_mean_per_source * apertures.area)
        
        # Calcular corrección de apertura empírica
        ap_corr, ap_corr_std, n_stars = compute_empirical_aperture_correction(
            data_corrected, header, positions, pixscale)
        
        # Aplicar corrección de apertura
        flux_corrected = flux * ap_corr
        
        # Calcular errores (igual que en el script de cúmulos)
        bkg_rms_e = bkg_rms * gain if not np.isnan(bkg_rms) else 0.0
        flux_e = np.abs(flux_corrected) * gain
        aperture_area = apertures.area
        var_src = flux_e
        var_bkg = aperture_area * (bkg_rms_e ** 2)
        var_read = aperture_area * (rdnoise ** 2)
        flux_err_e = np.sqrt(np.abs(var_src) + var_bkg + var_read)
        flux_err = flux_err_e / gain
        
        # Magnitud instrumental corregida
        min_flux = 1e-10
        mag_inst = -2.5 * np.log10(np.maximum(flux_corrected, min_flux))
        
        # SNR
        snr = np.where(flux_err > 0, flux_corrected / flux_err, 0.0)
        
        # Preparar resultados
        sources['flux_adu'] = flux_corrected
        sources['flux_err'] = flux_err
        sources['mag_inst'] = mag_inst
        sources['snr'] = snr
        sources['ap_correction'] = ap_corr
        sources['ap_correction_std'] = ap_corr_std
        sources['n_stars_ap_corr'] = n_stars
        
        # Coordenadas astronómicas si hay WCS
        if has_wcs:
            coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources['ra'] = coords.ra.deg
            sources['dec'] = coords.dec.deg
        else:
            sources['ra'] = np.nan
            sources['dec'] = np.nan
            
        # Información adicional
        sources['fwhm'] = fwhm_arcsec
        sources['aperture_diameter'] = aperture_diameter
        sources['pixscale'] = pixscale
        sources['gain'] = gain
        sources['rdnoise'] = rdnoise
        
        return sources[['ra', 'dec', 'flux_adu', 'flux_err', 'mag_inst', 'snr', 
                       'ap_correction', 'ap_correction_std', 'n_stars_ap_corr',
                       'fwhm', 'aperture_diameter', 'pixscale', 'gain', 'rdnoise']]
        
    except Exception as e:
        logging.error(f"Error in corrected star extraction from {image_path}: {e}")
        return pd.DataFrame()

def get_reference_positions_from_catalog(catalog_path, image_path):
    """
    Obtiene las posiciones en píxeles de las estrellas de referencia usando WCS.
    """
    try:
        # Leer catálogo de referencia
        ref_catalog = pd.read_csv(catalog_path)
        
        # Leer imagen para obtener WCS
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    header = hdu.header
                    break
            
            wcs = WCS(header)
        
        # Convertir coordenadas a píxeles
        coords = SkyCoord(ra=ref_catalog['ra'].values*u.deg, 
                         dec=ref_catalog['dec'].values*u.deg)
        x, y = wcs.world_to_pixel(coords)
        
        # Filtrar posiciones válidas
        valid_mask = np.isfinite(x) & np.isfinite(y)
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        
        logging.info(f"Found {len(positions)} valid reference positions")
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error getting reference positions: {e}")
        return np.array([]), pd.DataFrame()

def process_field_corrected(field_name, base_dir=BASE_DATA_DIR, aperture_diameter=3):
    """
    Procesa un campo aplicando las correcciones de fondo y apertura.
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO CAMPO CON CORRECCIONES: {field_name}")
    logging.info(f"{'='*60}")
    
    field_dir = os.path.join(base_dir, field_name)
    if not os.path.exists(field_dir):
        logging.error(f"Field directory not found: {field_dir}")
        return False
    
    # Archivos de entrada y salida
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_corrected.csv'
    spectra_dir = f'gaia_spectra_{field_name}'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Input catalog not found: {input_catalog}")
        return False
    
    # Procesar cada filtro
    band_catalogs = {}
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = os.path.join(field_dir, f"{field_name}_{band}.fits.fz")
        if not os.path.exists(image_file):
            image_file = os.path.join(field_dir, f"{field_name}_{band}.fits")
            if not os.path.exists(image_file):
                logging.warning(f"Image {field_name}_{band} not found. Skipping...")
                continue
        
        # Obtener posiciones de referencia (solo una vez)
        if reference_positions is None:
            reference_positions, ref_catalog = get_reference_positions_from_catalog(
                input_catalog, image_file)
            
            if len(reference_positions) == 0:
                logging.error("No valid reference positions found")
                return False
        
        # Extraer estrellas con correcciones
        cat = extract_stars_corrected(image_file, reference_positions, 
                                     aperture_diameter=aperture_diameter)
        
        if len(cat) == 0:
            logging.warning(f"{band}: No stars extracted.")
            continue
            
        band_catalogs[band] = cat
        logging.info(f"{band}: {len(cat)} stars extracted with corrections.")

    if not band_catalogs:
        logging.error("No stars extracted from any band.")
        return False

    # Combinar resultados
    logging.info("Combining results from all bands...")
    
    # Usar el catálogo de referencia como base
    combined_catalog = ref_catalog.copy()
    
    for band, cat in band_catalogs.items():
        # Añadir columnas de este filtro
        combined_catalog[f'mag_inst_{band}'] = cat['mag_inst'].values
        combined_catalog[f'flux_adu_{band}'] = cat['flux_adu'].values
        combined_catalog[f'flux_err_{band}'] = cat['flux_err'].values
        combined_catalog[f'snr_{band}'] = cat['snr'].values
        combined_catalog[f'ap_corr_{band}'] = cat['ap_correction'].values
        combined_catalog[f'ap_corr_std_{band}'] = cat['ap_correction_std'].values
        
        # Información instrumental (solo una vez por banda)
        if f'fwhm_{band}' not in combined_catalog.columns:
            combined_catalog[f'fwhm_{band}'] = cat['fwhm'].values[0]
            combined_catalog[f'pixscale_{band}'] = cat['pixscale'].values[0]
            combined_catalog[f'gain_{band}'] = cat['gain'].values[0]
            combined_catalog[f'rdnoise_{band}'] = cat['rdnoise'].values[0]
    
    # Guardar resultados
    combined_catalog.to_csv(output_catalog, index=False)
    logging.info(f"Corrected catalog saved to: {output_catalog}")
    logging.info(f"Total reference stars processed: {len(combined_catalog)}")
    
    # Estadísticas
    for band in band_catalogs.keys():
        mag_col = f'mag_inst_{band}'
        if mag_col in combined_catalog.columns:
            valid_mags = combined_catalog[mag_col][combined_catalog[mag_col] < 50]
            mean_mag = valid_mags.mean()
            std_mag = valid_mags.std()
            logging.info(f"{band}: Mean mag = {mean_mag:.2f} ± {std_mag:.2f} (n={len(valid_mags)})")
    
    return True

def main():
    # Campos a procesar
    fields = [
        'CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 
        'CenA07', 'CenA08', 'CenA09', 'CenA10', 'CenA11', 'CenA12',
        'CenA13', 'CenA14', 'CenA15', 'CenA16', 'CenA17', 'CenA18',
        'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24'
    ]
    
    # Procesar cada campo
    for field in fields:
        success = process_field_corrected(field, aperture_diameter=3)
        if not success:
            logging.error(f"Error processing field {field}. Continuing with next...")
        
        # Pausa entre campos
        time.sleep(10)
    
    logging.info("\nProcessing of all fields completed.")
    
if __name__ == '__main__':
    main()
