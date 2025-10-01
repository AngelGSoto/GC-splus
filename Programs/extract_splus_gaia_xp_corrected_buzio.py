#!/usr/bin/env python3
"""
extract_splus_gaia_xp_SIMPLE_3arcsec.py
MÉTODO SIMPLE 3" PARA ESTRELLAS DE REFERENCIA - ÓPTIMO PARA CALIBRACIÓN
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAM = 3.0  # ✅ 3" ÓPTIMO PARA ESTRELLAS

# ========== FUNCIONES SIMPLIFICADAS ==========

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

def perform_robust_photometry_3arcsec(data, positions, filter_name, pixel_scale=0.55):
    """
    Fotometría ROBUSTA para estrellas de referencia
    - Apertura: 3" diameter (1.5" radius)  
    - Sky annulus: 6-9" (estándar robusto)
    - Estimación de errores mejorada
    """
    try:
        # APERTURAS ESTÁNDAR PARA ESTRELLAS - 3" DIÁMETRO
        aperture_radius = (APERTURE_DIAM / 2.0) / pixel_scale  # 1.5" radius
        annulus_inner = (6.0 / 2.0) / pixel_scale  # 3" inner radius
        annulus_outer = (9.0 / 2.0) / pixel_scale  # 4.5" outer radius
        
        apertures = CircularAperture(positions, r=aperture_radius)
        annulus = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)
        
        # Fotometría
        phot_table = aperture_photometry(data, apertures)
        bkg_table = aperture_photometry(data, annulus)
        
        raw_flux = phot_table['aperture_sum'].data
        bkg_flux = bkg_table['aperture_sum'].data
        
        # Cálculo de áreas
        aperture_area = apertures.area
        annulus_area = annulus.area
        
        # Estimación ROBUSTA del fondo para cada fuente
        bkg_mean_per_pixel = np.zeros(len(positions))
        bkg_std_per_pixel = np.zeros(len(positions))
        
        for i, pos in enumerate(positions):
            try:
                # Crear máscara para el anillo de esta fuente
                mask = annulus.to_mask(method='center')[i]
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                
                if len(annulus_data_1d) > 10:
                    # Estadísticas robustas del fondo local
                    mean, median, std = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)
                    bkg_mean_per_pixel[i] = median
                    bkg_std_per_pixel[i] = std
                else:
                    # Fallback simple
                    bkg_mean_per_pixel[i] = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                    bkg_std_per_pixel[i] = np.std(annulus_data_1d) if len(annulus_data_1d) > 0 else 1.0
            except:
                # Fallback global
                bkg_mean_per_pixel[i] = bkg_flux[i] / annulus_area
                bkg_std_per_pixel[i] = 1.0
                
        # Flujo neto
        net_flux = raw_flux - (bkg_mean_per_pixel * aperture_area)
        
        # Estimación de errores ROBUSTA
        net_flux_err = np.sqrt(np.abs(net_flux) + (aperture_area * bkg_std_per_pixel**2))
        snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
        
        logging.info(f"{filter_name}: Fotometría 3\" - {len(positions)} estrellas, fondo medio: {np.median(bkg_mean_per_pixel):.4f} ADU/pix")
        return net_flux, net_flux_err, snr, aperture_radius
        
    except Exception as e:
        logging.error(f"Error en fotometría 3\": {e}")
        n = len(positions)
        return np.zeros(n), np.full(n, 99.0), np.zeros(n), (APERTURE_DIAM/2.0)/pixel_scale

# ========== FUNCIÓN PRINCIPAL ==========

def extract_instrumental_mags_3arcsec(image_path, positions, ref_catalog, field_name, filter_name):
    """
    Extrae magnitudes instrumentales con apertura 3" para estrellas
    - ÓPTIMO para capturar toda la luz estelar
    - SIN procesamiento complejo (no necesario para estrellas)
    - ROBUSTO para calibración
    """
    try:
        hdu, _, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        # Analizar header
        exptime = header.get('EFECTIME', header.get('EXPTIME', 1.0))
        if exptime <= 0:
            exptime = 1.0
        pixscale = header.get('PIXSCALE', 0.55)
        
        logging.info(f"{filter_name}: EXPTIME={exptime:.1f}s, PIXSCALE={pixscale:.2f}\"")
        
        # Estadísticas de la imagen
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0)
        logging.info(f"{filter_name}: Estadísticas - Media={mean_val:.6f}, Mediana={median_val:.6f}, Std={std_val:.6f} ADU")
        
        # ✅ USAR IMAGEN DIRECTAMENTE - SIN PROCESAMIENTO PARA ESTRELLAS
        # Las estrellas no necesitan unsharp mask porque:
        # 1. Están aisladas del fondo galáctico
        # 2. Son fuentes puntuales bien definidas
        # 3. El procesamiento adicional introduce incertidumbre en la calibración
        data_processed = data.copy()
        
        # ✅ FOTOMETRÍA ROBUSTA 3"
        flux, flux_err, snr, aperture_radius = perform_robust_photometry_3arcsec(
            data_processed, positions, filter_name, pixscale)
        
        # ✅ SIN CORRECCIÓN DE APERTURA PARA ESTRELLAS
        # Razón: Queremos magnitudes instrumentales PURAS para calibración
        # Las correcciones de apertura introducen dependencias adicionales
        aperture_correction = 0.0
        
        # Magnitudes instrumentales
        min_flux = 1e-10
        valid_flux_mask = (flux > min_flux) & np.isfinite(flux)
        
        mag_inst = np.where(valid_flux_mask, -2.5 * np.log10(flux), 99.0)
        mag_err = np.where(valid_flux_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
        # Validación adicional
        reasonable_mag = (mag_inst >= 10.0) & (mag_inst <= 30.0)
        reasonable_err = (mag_err >= 0.0) & (mag_err <= 5.0)
        final_mask = valid_flux_mask & reasonable_mag & reasonable_err
        
        mag_inst = np.where(final_mask, mag_inst, 99.0)
        mag_err = np.where(final_mask, mag_err, 99.0)
        snr = np.where(final_mask, snr, 0.0)
        
        # Resultados
        results = {
            'x_pix': positions[:, 0],
            'y_pix': positions[:, 1],
            # Flujos en ADUs (3")
            f'flux_adus_{APERTURE_DIAM:.1f}_{filter_name}': flux,
            f'flux_err_adus_{APERTURE_DIAM:.1f}_{filter_name}': flux_err,
            # Magnitudes instrumentales (3")
            f'mag_inst_{APERTURE_DIAM:.1f}_{filter_name}': mag_inst,
            f'mag_inst_total_{filter_name}': mag_inst,  # Para compatibilidad
            f'mag_err_{APERTURE_DIAM:.1f}_{filter_name}': mag_err,
            f'snr_{APERTURE_DIAM:.1f}_{filter_name}': snr,
            f'aper_corr_{APERTURE_DIAM:.1f}_{filter_name}': aperture_correction,  # 0.0
            f'pixscale_{filter_name}': pixscale,
            f'exptime_{filter_name}': exptime,
            f'aperture_diam_{filter_name}': APERTURE_DIAM,  # 3.0
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
        valid_mags = mag_inst[np.isfinite(mag_inst) & (mag_inst < 99.0)]
        if len(valid_mags) > 0:
            min_mag = np.min(valid_mags)
            max_mag = np.max(valid_mags)
            median_mag = np.median(valid_mags)
            n_valid = len(valid_mags)
            
            logging.info(f"{filter_name}: Magnitudes 3\" - {n_valid}/{len(positions)} válidas, "
                        f"rango [{min_mag:.2f}, {max_mag:.2f}], mediana={median_mag:.2f}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error en {filter_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_field_3arcsec(field_name):
    """Procesa un campo con método 3\" para estrellas"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO {field_name} (MÉTODO 3\" PARA ESTRELLAS)")
    logging.info(f"{'='*60}")
    
    field_dir = field_name
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_3arcsec.csv'
    
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
        df = extract_instrumental_mags_3arcsec(img, positions, ref_catalog, field_name, band)
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
    logging.info(f"✅ Resultados 3\" guardados en: {output_catalog}")
    
    # Estadísticas finales
    logging.info("ESTADÍSTICAS FINALES (MÉTODO 3\"):")
    for band in SPLUS_FILTERS:
        mag_col = f'mag_inst_total_{band}'
        if mag_col in combined.columns:
            mags = combined[mag_col][combined[mag_col] < 99.0]
            if len(mags) > 0:
                min_mag = mags.min()
                max_mag = mags.max()
                median_mag = mags.median()
                n_valid = len(mags)
                
                logging.info(f"{band}: {n_valid} válidas, [{min_mag:.2f}, {max_mag:.2f}], mediana={median_mag:.2f}")
    
    return True

def main():
    """Función principal"""
    test_fields = ['CenA01', 'CenA02']
    
    successful_fields = []
    for field in test_fields:
        if process_field_3arcsec(field):
            successful_fields.append(field)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESAMIENTO COMPLETADO - MÉTODO 3\" PARA ESTRELLAS")
    logging.info(f"✅ Apertura: 3\" diameter - ÓPTIMA para estrellas")
    logging.info(f"✅ Sin unsharp mask - No necesario para estrellas aisladas")
    logging.info(f"✅ Sin corrección de apertura - Magnitudes instrumentales puras")
    logging.info(f"✅ Fotometría robusta con estimación local de fondo")
    logging.info(f"✅ Ideal para calibración de zero points")
    logging.info(f"Campos exitosos: {successful_fields}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    main()
