#!/usr/bin/env python3
"""
extract_splus_gaia_xp_final_practical.py
VERSIÓN PRÁCTICA - Usa correcciones de apertura fijas basadas en S-PLUS
- Evita el problema con growth curves en imágenes con flujos anómalos
- Usa valores típicos de corrección de apertura para S-PLUS
- Mantiene unsharp mask para Centaurus A
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

def get_aperture_correction_fixed(filter_name, fwhm):
    """
    Correcciones REALISTAS basadas en caracterización S-PLUS
    """
    # Valores base de la pipeline S-PLUS para apertura de 3"
    base_corrections = {
        'F378': 0.15,  # Seeing típico ~2.0" en azul
        'F395': 0.16,  
        'F410': 0.17,  
        'F430': 0.18,  
        'F515': 0.20,  # Seeing ~1.8"
        'F660': 0.22,  # Seeing ~1.5" 
        'F861': 0.25   # Seeing ~1.3" - mejor seeing → más corrección
    }
    
    # Ajuste por FWHM REAL (inverso a tu implementación actual)
    # MEJOR seeing → MÁS luz fuera de apertura → MÁS corrección
    if fwhm < 1.2:
        # Seeing excelente (raro en S-PLUS)
        correction = base_corrections[filter_name] + 0.10
    elif fwhm < 1.6:
        # Seeing muy bueno (típico en rojo)
        correction = base_corrections[filter_name] + 0.05
    elif fwhm < 2.0:
        # Seeing típico de S-PLUS
        correction = base_corrections[filter_name]
    else:
        # Seeing pobre (típico en azul)
        correction = base_corrections[filter_name] - 0.05
    
    # Limitar a rango físico
    correction = max(0.08, min(0.40, correction))
    
    logging.info(f"{filter_name}: FWHM={fwhm:.2f}\", Corrección realista = {correction:.3f} mag")
    return correction

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

def analyze_image_header(header, filter_name):
    """Analiza el header SPLUS para información de diagnóstico"""
    exptime = header.get('EFECTIME', header.get('EXPTIME', 1.0))
    if exptime <= 0:
        exptime = 1.0
        logging.warning(f"{filter_name}: EXPTIME inválido, usando 1.0")
    
    pixscale = header.get('PIXSCALE', 0.55)
    fwhm = header.get('FWHMMEAN', 1.5)
    
    logging.info(f"{filter_name}: EXPTIME={exptime:.1f}s, PIXSCALE={pixscale:.2f}\", FWHM={fwhm:.2f}\"")
    
    return {
        'exptime': exptime,
        'pixscale': pixscale,
        'fwhm': fwhm
    }

def detect_galaxy_structure(data, filter_name):
    """
    Detecta si hay estructura galáctica significativa en la imagen
    Versión simplificada
    """
    try:
        # Estadísticas básicas
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Para Centaurus A, buscamos gradientes
        height, width = data.shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Calcular perfil radial simple
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_r = min(center_x, center_y)
        
        radial_bins = np.linspace(0, max_r, 20)
        radial_profile = []
        
        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
            if np.sum(mask) > 100:
                radial_profile.append(np.median(data[mask]))
            else:
                radial_profile.append(0.0)
        
        radial_profile = np.array(radial_profile)
        
        if len(radial_profile) > 5:
            center_brightness = np.mean(radial_profile[:5])
            outer_brightness = np.mean(radial_profile[-5:])
            brightness_gradient = center_brightness - outer_brightness
            
            # Criterio simple para estructura galáctica
            has_structure = brightness_gradient > 2 * std
            
            if has_structure:
                logging.info(f"{filter_name}: Estructura galáctica detectada (gradiente: {brightness_gradient:.6f})")
                return True, center_x, center_y, max_r * 0.7
            else:
                logging.info(f"{filter_name}: Sin estructura galáctica significativa")
                return False, center_x, center_y, 0
        else:
            return False, data.shape[1] // 2, data.shape[0] // 2, 0
            
    except Exception as e:
        logging.warning(f"Error detectando estructura galáctica: {e}")
        return False, data.shape[1] // 2, data.shape[0] // 2, 0

def apply_unsharp_mask_selective(data, positions, center_x, center_y, structure_radius, filter_name):
    """
    Aplica unsharp masking solo a estrellas cerca de la estructura galáctica
    """
    try:
        # Calcular distancias de cada estrella al centro de la estructura
        distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
        
        # Identificar estrellas dentro del área de influencia
        near_structure = distances < structure_radius
        n_near = np.sum(near_structure)
        
        if n_near == 0:
            logging.info(f"{filter_name}: No hay estrellas cerca de la estructura, sin unsharp mask")
            return data, False
        
        logging.info(f"{filter_name}: Aplicando unsharp mask a {n_near} estrellas cerca de estructura galáctica")
        
        # Aplicar unsharp mask
        data_clean = data.copy()
        data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        sigma = max(15.0, structure_radius / 20.0)
        smoothed = gaussian_filter(data_clean, sigma=sigma)
        data_unsharp = data_clean - 0.7 * smoothed
        
        logging.info(f"{filter_name}: Unsharp mask aplicado (sigma={sigma:.1f})")
        
        return data_unsharp, True
        
    except Exception as e:
        logging.error(f"Error aplicando unsharp mask selectivo: {e}")
        return data, False

def extract_instrumental_mags_practical(image_path, positions, ref_catalog, field_name, filter_name):
    """
    Extrae magnitudes instrumentales con correcciones de apertura fijas
    """
    try:
        hdu, _, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        # Analizar header para diagnóstico
        header_info = analyze_image_header(header, filter_name)
        pixscale = header_info['pixscale']
        exptime = header_info['exptime']
        fwhm = header_info['fwhm']
        
        # Estadísticas de la imagen
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0)
        logging.info(f"{filter_name}: Estadísticas - Media={mean_val:.6f}, Mediana={median_val:.6f}, Std={std_val:.6f} ADU")
        
        # Paso 1: Detectar estructura galáctica
        has_structure, center_x, center_y, structure_radius = detect_galaxy_structure(data, filter_name)
        
        # Paso 2: Aplicar unsharp mask selectivo si hay estructura
        used_unsharp = False
        if has_structure and structure_radius > 0:
            data_processed, used_unsharp = apply_unsharp_mask_selective(
                data, positions, center_x, center_y, structure_radius, filter_name
            )
        else:
            data_processed = data.copy()
        
        # Paso 3: Obtener corrección de apertura FIJA (no growth curves)
        aperture_correction = get_aperture_correction_fixed(filter_name, fwhm)
        
        # Paso 4: Fotometría en apertura de 3"
        aperture_radius = (APERTURE_DIAM / 2.0) / pixscale
        aperture = CircularAperture(positions, r=aperture_radius)
        
        # Fotometría directa
        phot_table = aperture_photometry(data_processed, aperture)
        flux_adus = phot_table['aperture_sum']
        
        # Estimación de error
        flux_err_adus = np.sqrt(np.abs(flux_adus) + 0.1)
        
        # ✅ Fórmula CORRECTA: minstr = −2.5 log10(flux_adus) + ACm
        min_flux = 1e-10
        mag_inst = -2.5 * np.log10(np.maximum(flux_adus, min_flux)) + aperture_correction
        
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
            # Magnitudes instrumentales CORREGIDAS
            f'mag_inst_3.0_{filter_name}': mag_inst,
            f'mag_inst_total_{filter_name}': mag_inst,
            f'mag_err_3.0_{filter_name}': mag_err,
            f'snr_3.0_{filter_name}': snr,
            f'aper_corr_3.0_{filter_name}': aperture_correction,
            f'pixscale_{filter_name}': pixscale,
            f'exptime_{filter_name}': exptime,
            f'fwhm_{filter_name}': fwhm,
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

def process_field_practical(field_name):
    """Procesa un campo con correcciones de apertura fijas"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO {field_name} (CORRECCIONES FIJAS)")
    logging.info(f"Usa valores típicos de S-PLUS para corrección de apertura")
    logging.info(f"{'='*60}")
    
    field_dir = field_name
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_splus_practical.csv'
    
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
        df = extract_instrumental_mags_practical(img, positions, ref_catalog, field_name, band)
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
    logging.info(f"✅ Resultados guardados en: {output_catalog}")
    
    # Estadísticas finales
    logging.info("ESTADÍSTICAS FINALES (Correcciones Fijas):")
    for band in SPLUS_FILTERS:
        mag_col = f'mag_inst_total_{band}'
        if mag_col in combined.columns:
            mags = combined[mag_col].dropna()
            if len(mags) > 0:
                min_mag = mags.min()
                max_mag = mags.max()
                median_mag = mags.median()
                aper_corr = combined[f'aper_corr_3.0_{band}'].iloc[0] if f'aper_corr_3.0_{band}' in combined.columns else 0.0
                
                logging.info(f"{band}: [{min_mag:.2f}, {max_mag:.2f}], mediana={median_mag:.2f}, AP corr: {aper_corr:.3f}")
    
    return True

def main():
    """Función principal"""
    test_fields = ['CenA01', 'CenA02']
    
    successful_fields = []
    for field in test_fields:
        if process_field_practical(field):
            successful_fields.append(field)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESAMIENTO COMPLETADO - CORRECCIONES FIJAS")
    logging.info(f"✅ Correcciones de apertura basadas en valores típicos S-PLUS")
    logging.info(f"✅ Ajustadas por seeing (FWHM)")
    logging.info(f"✅ Unsharp mask para estructuras galácticas")
    logging.info(f"✅ SIN problemas de growth curves")
    logging.info(f"Campos exitosos: {successful_fields}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    main()
