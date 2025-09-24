#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected.py - Versión simplificada para calcular magnitudes 
instrumentales de referencia solo para 3 arcsec (corregidas por apertura) para zero points.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, MedianBackground
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAMETER = 3  # Solo 3 arcsec para zero points

def model_background_residuals(data, field_name, filter_name):
    """
    Background treatment IDÉNTICO al usado en cúmulos globulares
    """
    try:
        # Estadísticas básicas de la imagen
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Para imágenes S-PLUS ya restadas, ser conservador
        if std < 1.0:
            logging.info(f"Imagen plana (std={std:.3f}). Saltando modelado de fondo.")
            return data, std
        
        # Crear máscara para objetos brillantes
        mask = data > median + 10 * std
        
        # Dilatar máscara
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5)))
        
        # PARÁMETROS IDÉNTICOS a cúmulos globulares
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(data, 
                          box_size=50,    # ← MISMO que en cúmulos globulares
                          filter_size=3,  # ← MISMO que en cúmulos globulares
                          sigma_clip=sigma_clip, 
                          bkg_estimator=MedianBackground(), 
                          mask=dilated_mask,
                          exclude_percentile=20)
        
        # Solo restar si el modelo de fondo tiene estructura significativa
        bkg_range = np.max(bkg.background) - np.min(bkg.background)
        if bkg_range < 3 * std:
            data_corrected = data - np.median(bkg.background)
        else:
            data_corrected = data - bkg.background
        
        bkg_rms = bkg.background_rms_median
        logging.info(f"Fondo tratado: std_original={std:.3f}")
        return data_corrected, bkg_rms
        
    except Exception as e:
        logging.warning(f"Tratamiento de fondo falló: {e}. Usando imagen original.")
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        return data, std

def calculate_aperture_correction_3arcsec(data, header, star_positions, filter_name):
    """
    Calcula corrección de apertura específicamente para 3 arcsec
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        
        # Definir aperturas para curva de crecimiento
        aperture_diameters = np.linspace(1, 50, 32)
        aperture_radii_px = (aperture_diameters / 2) / pixscale
        
        growth_fluxes = []
        for radius in aperture_radii_px:
            apertures = CircularAperture(star_positions, r=radius)
            phot_table = aperture_photometry(data, apertures)
            growth_fluxes.append(phot_table['aperture_sum'].data)
        
        growth_fluxes = np.array(growth_fluxes)
        total_fluxes = growth_fluxes[-1]  # Flujo en apertura grande (50 arcsec)
        valid_mask = total_fluxes > 0
        
        if np.sum(valid_mask) < 5:
            logging.warning(f"Estrellas insuficientes para corrección de apertura")
            return 0.0, 1.0, 0
        
        # Curva de crecimiento normalizada
        normalized_curves = growth_fluxes[:, valid_mask] / total_fluxes[valid_mask]
        mean_growth = np.median(normalized_curves, axis=1)
        
        # Encontrar la fracción para 3 arcsec
        target_idx = np.argmin(np.abs(aperture_diameters - 3.0))
        fraction_in_3arcsec = mean_growth[target_idx]
        
        if fraction_in_3arcsec <= 0:
            return 0.0, 1.0, 0
        
        aperture_correction_mag = -2.5 * np.log10(fraction_in_3arcsec)
        aperture_correction_flux = 1.0 / fraction_in_3arcsec
        
        n_stars = np.sum(valid_mask)
        logging.info(f"{filter_name}: Corrección apertura 3arcsec = {aperture_correction_mag:.3f} mag (n={n_stars})")
        
        return aperture_correction_mag, aperture_correction_flux, n_stars
        
    except Exception as e:
        logging.error(f"Error en corrección de apertura: {e}")
        return 0.0, 1.0, 0

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia desde el catálogo"""
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
                        break
                except:
                    continue
            
            if wcs is None:
                logging.error(f"No se pudo encontrar WCS válido en {image_path}")
                return np.array([]), pd.DataFrame()
        
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
        
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        
        if data_shape is not None:
            # Margen para evitar bordes
            margin_pixels = 50
            valid_mask = valid_mask & (x > margin_pixels) & (x < data_shape[1] - margin_pixels)
            valid_mask = valid_mask & (y > margin_pixels) & (y < data_shape[0] - margin_pixels)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Encontradas {len(positions)} posiciones válidas de {len(ref_catalog)} estrellas en catálogo")
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error obteniendo posiciones de referencia: {e}")
        return np.array([]), pd.DataFrame()

def extract_instrumental_magnitudes(image_path, reference_positions, field_name, filter_name):
    """
    Extrae magnitudes instrumentales para 3 arcsec con correcciones aplicadas
    """
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
                logging.error(f"No se encontraron datos en {image_path}")
                return pd.DataFrame()
        
        pixscale = header.get('PIXSCALE', 0.55)
        
        # 1. APLICAR CORRECCIÓN DE FONDO (igual que en cúmulos globulares)
        data_corrected, bkg_rms = model_background_residuals(data, field_name, filter_name)
        
        # 2. CALCULAR CORRECCIÓN DE APERTURA para 3 arcsec
        ap_corr_mag, ap_corr_flux, n_stars_ap = calculate_aperture_correction_3arcsec(
            data_corrected, header, reference_positions, filter_name)
        
        # 3. FOTOMETRÍA con apertura de 3 arcsec
        aperture_radius_pixels = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale  # Anillo de fondo igual que en GCs
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_pixels)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        # Fotometría
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        # Restar fondo
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux_uncorrected = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # Aplicar corrección de apertura
        flux_corrected = flux_uncorrected * ap_corr_flux
        
        # Cálculo de errores
        flux_err = np.sqrt(np.abs(flux_corrected) + (apertures.area * bkg_rms**2))
        
        # Magnitudes instrumentales corregidas
        min_flux = 1e-10
        mag_inst_uncorrected = -2.5 * np.log10(np.maximum(flux_uncorrected, min_flux))
        mag_inst_corrected = -2.5 * np.log10(np.maximum(flux_corrected, min_flux))
        
        snr = np.where(flux_err > 0, flux_corrected / flux_err, 0.0)
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            f'flux_uncorrected_{filter_name}': flux_uncorrected,
            f'flux_corrected_{filter_name}': flux_corrected,
            f'flux_err_{filter_name}': flux_err,
            f'mag_inst_uncorrected_{filter_name}': mag_inst_uncorrected,
            f'mag_inst_corrected_{filter_name}': mag_inst_corrected,  # ← ESTA ES LA IMPORTANTE
            f'snr_{filter_name}': snr,
            f'ap_corr_mag_{filter_name}': ap_corr_mag,
            f'ap_corr_flux_{filter_name}': ap_corr_flux,
            f'n_stars_ap_corr_{filter_name}': n_stars_ap
        })
        
        # Añadir coordenadas si es posible
        try:
            wcs = WCS(header)
            ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
        except:
            results['ra'] = np.nan
            results['dec'] = np.nan
            logging.warning("No se pudieron calcular coordenadas WCS")
        
        logging.info(f"{filter_name}: {len(results)} estrellas procesadas, mag promedio = {mag_inst_corrected.mean():.2f}")
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        return pd.DataFrame()

def process_field_corrected(field_name):
    """Procesa un campo completo para todas las bandas SPLUS"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO: {field_name}")
    logging.info(f"{'='*60}")
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Directorio de campo no encontrado: {field_dir}")
        return False
    
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_splus_method.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Catálogo de entrada no encontrado: {input_catalog}")
        return False
    
    # Obtener posiciones de referencia (usamos la primera imagen disponible)
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = os.path.join(field_dir, f"{field_name}_{band}.fits.fz")
        if not os.path.exists(image_file):
            image_file = os.path.join(field_dir, f"{field_name}_{band}.fits")
            if not os.path.exists(image_file):
                continue
        
        reference_positions, ref_catalog = get_reference_positions_from_catalog(input_catalog, image_file)
        if len(reference_positions) > 0:
            logging.info(f"Usando {band} para obtener posiciones de referencia")
            break
    
    if reference_positions is None or len(reference_positions) == 0:
        logging.error("No se pudieron obtener posiciones de referencia válidas")
        return False
    
    # Procesar cada banda
    band_results = {}
    for band in SPLUS_FILTERS:
        image_file = os.path.join(field_dir, f"{field_name}_{band}.fits.fz")
        if not os.path.exists(image_file):
            image_file = os.path.join(field_dir, f"{field_name}_{band}.fits")
            if not os.path.exists(image_file):
                logging.warning(f"Imagen {field_name}_{band} no encontrada")
                continue
        
        results = extract_instrumental_magnitudes(image_file, reference_positions, field_name, band)
        if len(results) > 0:
            band_results[band] = results
        else:
            logging.warning(f"{band}: No se pudieron extraer magnitudes")
    
    if not band_results:
        logging.error("No se procesó ninguna banda correctamente")
        return False
    
    # Combinar resultados con el catálogo original
    combined_catalog = ref_catalog.copy()
    
    for band, results in band_results.items():
        if len(results) == len(combined_catalog):
            # Añadir todas las columnas de esta banda
            for col in results.columns:
                if col not in ['ra', 'dec']:  # Evitar duplicar coordenadas
                    combined_catalog[col] = results[col].values
        else:
            logging.warning(f"{band}: Número de estrellas no coincide ({len(results)} vs {len(combined_catalog)})")
    
    # Guardar resultados
    combined_catalog.to_csv(output_catalog, index=False)
    logging.info(f"Resultados guardados en: {output_catalog}")
    
    # Reportar estadísticas
    for band in SPLUS_FILTERS:
        mag_col = f'mag_inst_corrected_{band}'
        if mag_col in combined_catalog.columns:
            valid_mags = combined_catalog[mag_col][combined_catalog[mag_col] < 50]
            if len(valid_mags) > 0:
                mean_mag = valid_mags.mean()
                std_mag = valid_mags.std()
                logging.info(f"{band}: Mag corregida = {mean_mag:.2f} ± {std_mag:.2f} (n={len(valid_mags)})")
    
    return True

def main():
    """Función principal"""
    # Campos a procesar (empezar con uno para prueba)
    #test_fields = ['CenA01']
    all_fields = [
        'CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 
        'CenA07', 'CenA08', 'CenA09', 'CenA10', 'CenA11', 'CenA12',
        'CenA13', 'CenA14', 'CenA15', 'CenA16', 'CenA17', 'CenA18',
        'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24'
    ]
    
    # Procesar solo campos de prueba primero
    fields_to_process = all_fields
    
    start_time = time.time()
    
    for i, field in enumerate(fields_to_process):
        logging.info(f"Procesando campo {i+1}/{len(fields_to_process)}: {field}")
        success = process_field_corrected(field)
        if not success:
            logging.error(f"Error procesando {field}")
        logging.info(f"Campo {field} completado\n")
    
    total_time = time.time() - start_time
    logging.info(f"Procesamiento completado en {total_time/60:.1f} minutos")

if __name__ == '__main__':
    main()
