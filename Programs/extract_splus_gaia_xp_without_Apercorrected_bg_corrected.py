#!/usr/bin/env python3
"""
extract_splus_gaia_xp_without_Apercorrected_bg_corrected.py
VERSIÓN SIN CORRECCIONES DE APERTURA - COMPLETA
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from scipy.ndimage import gaussian_filter
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAM = 3.0  # Apertura de 3" para fotometría

# ========== FUNCIONES AUXILIARES ==========

def find_valid_image_hdu(fits_file):
    """Encuentra el HDU válido en el archivo FITS"""
    try:
        with fits.open(fits_file) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.header.get('NAXIS', 0) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.is_celestial:
                            return hdu, i, wcs
                    except Exception:
                        continue
        return None, None, None
    except Exception as e:
        logging.error(f"Error abriendo {fits_file}: {e}")
        return None, None, None

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia desde el catálogo"""
    try:
        ref_catalog = pd.read_csv(catalog_path)
        hdu, _, wcs = find_valid_image_hdu(image_path)
        if wcs is None:
            logging.error(f"No WCS válido en {image_path}")
            return np.array([]), pd.DataFrame()
        
        data_shape = hdu.data.shape
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        
        try:
            x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
        except Exception as e:
            logging.warning(f"all_world2pix falló: {e}")
            try:
                coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
                x, y = wcs.world_to_pixel(coords)
            except Exception as e2:
                logging.error(f"Conversión falló: {e2}")
                return np.array([]), pd.DataFrame()
        
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        if data_shape is not None:
            margin = 100
            valid_mask &= (x > margin) & (x < data_shape[1] - margin)
            valid_mask &= (y > margin) & (y < data_shape[0] - margin)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Encontradas {len(positions)} posiciones válidas")
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error obteniendo posiciones: {e}")
        return np.array([]), pd.DataFrame()

def find_image_file(field_dir, field_name, filter_name):
    """Encuentra el archivo de imagen para un filtro dado"""
    for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
        path = os.path.join(field_dir, ext)
        if os.path.exists(path):
            return path
    return None


def detect_galaxy_structure(data, filter_name):
    """
    Versión mejorada para detectar estructura galáctica residual
    """
    try:
        from photutils.background import Background2D, MedianBackground
        from astropy.stats import SigmaClip
        
        # Usar Background2D para modelar variaciones a gran escala
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        
        # Box size más pequeño para capturar estructura galáctica
        box_size = (100, 100)  # píxeles
        bkg = Background2D(data, box_size, 
                          filter_size=(3, 3),
                          sigma_clip=sigma_clip, 
                          bkg_estimator=bkg_estimator)
        
        # Calcular residuos estructurados
        residual = data - bkg.background
        residual_std = np.std(residual)
        
        # Umbral más conservador para detección
        has_significant_structure = np.max(np.abs(residual)) > 5 * residual_std
        
        if has_significant_structure:
            # Encontrar centro de la estructura máxima
            height, width = data.shape
            y, x = np.ogrid[:height, :width]
            
            # Usar el pico de residuo para centrar
            max_residual_pos = np.unravel_index(np.argmax(np.abs(residual)), data.shape)
            center_y, center_x = max_residual_pos
            
            # Radio estimado basado en la extensión de residuos significativos
            significant_mask = np.abs(residual) > 3 * residual_std
            if np.any(significant_mask):
                y_pos, x_pos = np.where(significant_mask)
                distances = np.sqrt((x_pos - center_x)**2 + (y_pos - center_y)**2)
                structure_radius = np.percentile(distances, 90)  # 90th percentile
            else:
                structure_radius = min(center_x, center_y, 
                                     width - center_x, height - center_y) * 0.7
            
            logging.info(f"{filter_name}: Estructura significativa detectada - centro ({center_x:.1f}, {center_y:.1f}), radio {structure_radius:.1f} pix")
            return True, center_x, center_y, structure_radius
        else:
            logging.info(f"{filter_name}: Sin estructura residual significativa")
            return False, data.shape[1] // 2, data.shape[0] // 2, 0
            
    except Exception as e:
        logging.warning(f"Error en detección mejorada: {e}")
        return False, data.shape[1] // 2, data.shape[0] // 2, 0

def apply_unsharp_mask_selective(data, positions, center_x, center_y, structure_radius, filter_name):
    """
    Versión más conservadora del unsharp mask
    """
    try:
        # Calcular distancias al centro de estructura
        distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
        
        # Solo estrellas dentro del radio de estructura + margen
        margin = 50  # píxeles de margen
        near_structure = distances < (structure_radius + margin)
        n_near = np.sum(near_structure)
        
        if n_near == 0:
            return data, False
        
        logging.info(f"{filter_name}: Aplicando unsharp mask conservador a {n_near}/{len(positions)} estrellas")
        
        # Parámetros más conservadores
        data_clean = data.copy()
        data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Sigma más pequeño para preservar estructuras estelares
        sigma = max(8.0, structure_radius / 30.0)  # Más conservador
        smoothed = gaussian_filter(data_clean, sigma=sigma)
        
        # Sustracción más suave
        data_unsharp = data_clean - 0.3 * smoothed  # Reducido de 0.7 a 0.3
        
        # Aplicar solo en regiones cerca de estructura galáctica
        mask = np.zeros_like(data, dtype=bool)
        y_idx, x_idx = np.ogrid[:data.shape[0], :data.shape[1]]
        structure_mask = (x_idx - center_x)**2 + (y_idx - center_y)**2 < (structure_radius + margin)**2
        
        result = data.copy()
        result[structure_mask] = data_unsharp[structure_mask]
        
        logging.info(f"{filter_name}: Unsharp mask aplicado (sigma={sigma:.1f}, factor=0.3)")
        return result, True
        
    except Exception as e:
        logging.error(f"Error en unsharp mask conservador: {e}")
        return data, False

# ========== FUNCIÓN PRINCIPAL MODIFICADA ==========

def extract_instrumental_mags_no_aper_corr(image_path, positions, ref_catalog, field_name, filter_name):
    """
    Extrae magnitudes instrumentales SIN correcciones de apertura
    """
    try:
        hdu, _, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        # Analizar header para diagnóstico
        exptime = header.get('EFECTIME', header.get('EXPTIME', 1.0))
        if exptime <= 0:
            exptime = 1.0
        pixscale = header.get('PIXSCALE', 0.55)
        
        logging.info(f"{filter_name}: EXPTIME={exptime:.1f}s, PIXSCALE={pixscale:.2f}\"")
        
        # Estadísticas de la imagen
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0)
        logging.info(f"{filter_name}: Estadísticas - Media={mean_val:.6f}, Mediana={median_val:.6f}, Std={std_val:.6f} ADU")
        
        # Detectar estructura galáctica
        has_structure, center_x, center_y, structure_radius = detect_galaxy_structure(data, filter_name)
        
        # Aplicar unsharp mask selectivo si hay estructura
        used_unsharp = False
        if has_structure and structure_radius > 0:
            data_processed, used_unsharp = apply_unsharp_mask_selective(
                data, positions, center_x, center_y, structure_radius, filter_name
            )
        else:
            data_processed = data.copy()
        
        # ✅ MODIFICACIÓN CRÍTICA: SIN CORRECCIÓN DE APERTURA
        aperture_correction = 0.0  # No aplicar corrección
        
        # Fotometría en apertura de 3"
        aperture_radius = (APERTURE_DIAM / 2.0) / pixscale
        aperture = CircularAperture(positions, r=aperture_radius)
        
        # Fotometría directa
        phot_table = aperture_photometry(data_processed, aperture)
        flux_adus = phot_table['aperture_sum']
        
        # Estimación de error
        flux_err_adus = np.sqrt(np.abs(flux_adus) + 0.1)
        
        # ✅ Fórmula SIN corrección de apertura: minstr = −2.5 log10(flux_adus)
        min_flux = 1e-10
        mag_inst = -2.5 * np.log10(np.maximum(flux_adus, min_flux))
        
        # Calcular errores
        mag_err = (2.5 / np.log(10)) * (flux_err_adus / np.maximum(flux_adus, min_flux))
        snr = np.where(flux_err_adus > 0, flux_adus / flux_err_adus, 0.0)
        
        # Resultados
        results = {
            'x_pix': positions[:, 0],
            'y_pix': positions[:, 1],
            # Flujos en ADUs
            f'flux_adus_3.0_{filter_name}': flux_adus,
            f'flux_err_adus_3.0_{filter_name}': flux_err_adus,
            # Magnitudes instrumentales SIN CORREGIR
            f'mag_inst_3.0_{filter_name}': mag_inst,
            f'mag_inst_total_{filter_name}': mag_inst,
            f'mag_err_3.0_{filter_name}': mag_err,
            f'snr_3.0_{filter_name}': snr,
            f'aper_corr_3.0_{filter_name}': aperture_correction,  # Siempre 0.0
            f'pixscale_{filter_name}': pixscale,
            f'exptime_{filter_name}': exptime,
            f'has_galaxy_structure_{filter_name}': has_structure,
            f'used_unsharp_{filter_name}': used_unsharp,
        }
        
        # Coordenadas
        try:
            ra, dec = wcs.all_pix2world(positions[:, 0], positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
        except:
            if 'ra' in ref_catalog.columns and 'dec' in ref_catalog.columns:
                results['ra'] = ref_catalog['ra'].values
                results['dec'] = ref_catalog['dec'].values
        
        df = pd.DataFrame(results)
        
        # Estadísticas de magnitudes
        valid_mags = mag_inst[np.isfinite(mag_inst)]
        if len(valid_mags) > 0:
            min_mag = np.min(valid_mags)
            max_mag = np.max(valid_mags)
            median_mag = np.median(valid_mags)
            
            method = "unsharp" if used_unsharp else "normal"
            logging.info(f"{filter_name}: Magnitudes en rango [{min_mag:.2f}, {max_mag:.2f}], mediana={median_mag:.2f}, método: {method}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error en {filter_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_field_no_aper_corr(field_name):
    """Procesa un campo SIN correcciones de apertura"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO {field_name} (SIN CORRECCIONES DE APERTURA)")
    logging.info(f"{'='*60}")
    
    field_dir = field_name
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_no_aper_corr.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Catálogo no encontrado: {input_catalog}")
        return False
    
    # Encontrar posiciones
    positions, ref_catalog = None, None
    for band in SPLUS_FILTERS:
        img = find_image_file(field_dir, field_name, band)
        if img:
            positions, ref_catalog = get_reference_positions_from_catalog(input_catalog, img)
            if len(positions) > 10:
                break
    
    if positions is None or len(positions) == 0:
        logging.error("No se obtuvieron posiciones válidas")
        return False
    
    # Procesar cada filtro
    all_results = {}
    for band in SPLUS_FILTERS:
        img = find_image_file(field_dir, field_name, band)
        if not img:
            logging.warning(f"Imagen no encontrada para {band}")
            continue
            
        logging.info(f"Procesando {band}...")
        df = extract_instrumental_mags_no_aper_corr(img, positions, ref_catalog, field_name, band)
        if not df.empty:
            all_results[band] = df
    
    if not all_results:
        logging.error("No se procesó ningún filtro")
        return False
    
    # Combinar resultados
    combined = ref_catalog.copy()
    for band, df in all_results.items():
        if len(df) == len(combined):
            for col in df.columns:
                if col not in ['ra', 'dec']:
                    combined[col] = df[col].values
    
    # Guardar resultados
    combined.to_csv(output_catalog, index=False)
    logging.info(f"✅ Resultados SIN correcciones de apertura guardados en: {output_catalog}")
    
    # Estadísticas finales
    logging.info("ESTADÍSTICAS FINALES (SIN CORRECCIONES DE APERTURA):")
    for band in SPLUS_FILTERS:
        mag_col = f'mag_inst_total_{band}'
        if mag_col in combined.columns:
            mags = combined[mag_col].dropna()
            if len(mags) > 0:
                min_mag = mags.min()
                max_mag = mags.max()
                median_mag = mags.median()
                
                logging.info(f"{band}: [{min_mag:.2f}, {max_mag:.2f}], mediana={median_mag:.2f}")
    
    return True

def main():
    """Función principal"""
    test_fields = ['CenA01', 'CenA02']
    
    successful_fields = []
    for field in test_fields:
        if process_field_no_aper_corr(field):
            successful_fields.append(field)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESAMIENTO COMPLETADO - SIN CORRECCIONES DE APERTURA")
    logging.info(f"✅ Magnitudes instrumentales puras (sin correcciones)")
    logging.info(f"✅ Unsharp mask para estructuras galácticas se mantiene")
    logging.info(f"✅ SIN problemas de correcciones de apertura sobreestimadas")
    logging.info(f"Campos exitosos: {successful_fields}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    main()
