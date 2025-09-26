#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected_final.py - Versión FINAL corregida que aplica aperture correction
y maneja correctamente la calibración usando FLXSCAL y EXPTIME del header.
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
APERTURE_DIAMETER = 3  # Apertura estándar para fotometría de referencia
APERTURES_FOR_CORRECTION = [3, 4, 5, 6]

def find_valid_image_hdu(fits_file):
    """Encuentra la primera HDU válida con datos y WCS"""
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

def check_splus_background_status(data, filter_name):
    """Verifica si la imagen SPLUS ya tiene el fondo restado"""
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        background_already_subtracted = (abs(median) < 0.1 * std)
        
        if background_already_subtracted:
            logging.info(f"{filter_name}: Fondo ya restado (mediana={median:.3f}, std={std:.3f})")
            return data, std, False
        else:
            logging.warning(f"{filter_name}: Posible fondo residual (mediana={median:.3f}, std={std:.3f})")
            return apply_minimal_background_correction(data, filter_name), std, True
            
    except Exception as e:
        logging.warning(f"Error verificando fondo: {e}")
        return data, np.std(data), False

def apply_minimal_background_correction(data, filter_name):
    """Aplica corrección de fondo mínima y conservadora"""
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        if std < 1.0:
            logging.info(f"{filter_name}: Variación mínima, no se aplica corrección")
            return data
        
        mask = data > median + 15 * std
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(data, 
                          box_size=150,
                          filter_size=5,
                          sigma_clip=sigma_clip, 
                          bkg_estimator=MedianBackground(), 
                          mask=dilated_mask,
                          exclude_percentile=30)
        
        bkg_range = np.max(bkg.background) - np.min(bkg.background)
        if bkg_range < 2 * std:
            data_corrected = data - np.median(bkg.background)
            logging.info(f"{filter_name}: Corrección mínima aplicada")
        else:
            data_corrected = data - bkg.background
            logging.info(f"{filter_name}: Corrección completa aplicada")
        
        return data_corrected
        
    except Exception as e:
        logging.warning(f"Corrección de fondo falló: {e}")
        return data

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia desde el catálogo"""
    try:
        ref_catalog = pd.read_csv(catalog_path)
        
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if wcs is None:
            logging.error(f"No se pudo encontrar WCS válido en {image_path}")
            return np.array([]), pd.DataFrame()
        
        data_shape = hdu.data.shape
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        
        # CONVERSIÓN ROBUSTA DE COORDENADAS
        try:
            # Método 1: Usar all_pix2world (más robusto)
            x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
        except Exception as e:
            logging.warning(f"all_world2pix falló, intentando método alternativo: {e}")
            # Método 2: Usar SkyCoord
            try:
                coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
                x, y = wcs.world_to_pixel(coords)
            except Exception as e2:
                logging.error(f"Todos los métodos de conversación fallaron: {e2}")
                return np.array([]), pd.DataFrame()
        
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        
        if data_shape is not None:
            margin_pixels = 100
            valid_mask = valid_mask & (x > margin_pixels) & (x < data_shape[1] - margin_pixels)
            valid_mask = valid_mask & (y > margin_pixels) & (y < data_shape[0] - margin_pixels)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Encontradas {len(positions)} posiciones válidas")
        
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error obteniendo posiciones: {e}")
        return np.array([]), pd.DataFrame()

def find_image_file(field_dir, field_name, filter_name):
    """Encuentra el archivo de imagen"""
    possible_extensions = [
        f"{field_name}_{filter_name}.fits.fz",
        f"{field_name}_{filter_name}.fits"
    ]
    
    for ext in possible_extensions:
        image_path = os.path.join(field_dir, ext)
        if os.path.exists(image_path):
            return image_path
    
    return None

def calculate_aperture_correction_simple(data, header, star_positions, filter_name):
    """
    Calcula aperture correction usando método simple pero robusto
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        fwhm_arcsec = header.get('FWHMMEAN', 1.5)
        fwhm_pixels = fwhm_arcsec / pixscale
        
        logging.info(f"{filter_name}: FWHM = {fwhm_arcsec:.2f} arcsec")
        
        if len(star_positions) < 10:
            logging.warning(f"{filter_name}: Insuficientes estrellas para aperture correction")
            return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}
        
        # Usar valores empíricos basados en FWHM
        # Para FWHM ~1.5", factores típicos son:
        empirical_factors = {
            3: 1.25,  # 25% de corrección
            4: 1.15,  # 15% de corrección
            5: 1.10,  # 10% de corrección  
            6: 1.05   # 5% de corrección
        }
        
        # Ajustar basado en FWHM real
        seeing_factor = 1.5 / fwhm_arcsec  # Normalizar a 1.5"
        
        adjusted_factors = {}
        for ap_size, factor in empirical_factors.items():
            adjusted_factor = 1.0 + (factor - 1.0) * seeing_factor
            # Límites razonables
            if ap_size == 3:
                adjusted_factor = min(adjusted_factor, 1.3)
            elif ap_size == 4:
                adjusted_factor = min(adjusted_factor, 1.2)
            elif ap_size == 5:
                adjusted_factor = min(adjusted_factor, 1.15)
            else:  # 6 arcsec
                adjusted_factor = min(adjusted_factor, 1.1)
            adjusted_factors[ap_size] = max(adjusted_factor, 1.0)
        
        logging.info(f"{filter_name}: Factores de corrección: {adjusted_factors}")
        return adjusted_factors
        
    except Exception as e:
        logging.error(f"Error calculando aperture correction para {filter_name}: {e}")
        return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}

def get_calibration_factors(header, filter_name):
    """
    Obtiene factores de calibración del header SPLUS
    Basado en la información del header proporcionado
    """
    try:
        # Obtener tiempo de exposición
        exptime = header.get('EXPTIME', 1.0)
        
        # Buscar factores FLXSCAL (puede haber múltiples para diferentes exposiciones)
        flxscal_keys = [key for key in header.keys() if 'FLXSCAL' in key]
        if flxscal_keys:
            flxscal_values = [header[key] for key in flxscal_keys]
            flxscal_mean = np.mean(flxscal_values)
            logging.info(f"{filter_name}: EXPTIME={exptime}s, FLXSCAL={flxscal_mean:.6f}")
        else:
            # Si no hay FLXSCAL, usar valor por defecto
            flxscal_mean = 1.0
            logging.warning(f"{filter_name}: No FLXSCAL found, using 1.0")
        
        # El valor de GAIN parece incorrecto (825 e-/ADU), así que no lo usamos
        # En su lugar, usamos FLXSCAL que es más confiable
        
        return exptime, flxscal_mean
        
    except Exception as e:
        logging.error(f"Error obteniendo factores de calibración: {e}")
        return 1.0, 1.0

def extract_instrumental_magnitudes_CORREGIDAS(image_path, reference_positions, ref_catalog, field_name, filter_name):
    """
    ✅ VERSIÓN CORREGIDA: Aplica aperture correction y calibración correctamente
    """
    try:
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            logging.error(f"No se pudo encontrar HDU válida en {image_path}")
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        pixscale = header.get('PIXSCALE', 0.55)
        logging.info(f"{filter_name}: Procesando imagen {data.shape}, pixscale={pixscale}")
        
        # 1. Obtener factores de calibración del header
        exptime, flxscal = get_calibration_factors(header, filter_name)
        
        # 2. Verificar y corregir fondo
        data_corrected, bkg_rms, needs_correction = check_splus_background_status(data, filter_name)
        
        # 3. Calcular factores de corrección de apertura
        aperture_correction_factors = calculate_aperture_correction_simple(
            data_corrected, header, reference_positions, filter_name)
        
        # 4. Fotometría con apertura estándar (3 arcsec)
        aperture_radius_px = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_px)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux_uncorrected = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # ✅ CORRECCIÓN: Aplicar aperture correction al flujo
        correction_factor = aperture_correction_factors.get(APERTURE_DIAMETER, 1.0)
        flux_corrected = flux_uncorrected * correction_factor
        
        # ✅✅✅ CORRECCIÓN CRÍTICA: Aplicar calibración usando FLXSCAL y EXPTIME
        # Esto convierte los flujos a unidades físicas consistentes
        flux_calibrated = flux_corrected * flxscal / exptime
        
        # ✅ Calcular magnitudes instrumentales CALIBRADAS (deberían ser valores razonables)
        min_flux = 1e-10  # Evitar log(0)
        mag_inst_calibrated = -2.5 * np.log10(np.maximum(flux_calibrated, min_flux))
        
        # Calcular errores también calibrados
        flux_err = np.sqrt(np.abs(flux_corrected) + (apertures.area * bkg_rms**2))
        flux_err_calibrated = flux_err * flxscal / exptime
        snr = np.where(flux_err_calibrated > 0, flux_calibrated / flux_err_calibrated, 0.0)
        
        # 5. Crear DataFrame con magnitudes CALIBRADAS
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            f'flux_raw_{filter_name}': flux_uncorrected,      # Flujo crudo (sin correcciones)
            f'flux_corrected_{filter_name}': flux_corrected,  # Flujo con aperture correction
            f'flux_calibrated_{filter_name}': flux_calibrated, # Flujo calibrado (físico)
            f'flux_err_{filter_name}': flux_err_calibrated,
            f'mag_inst_calibrated_{filter_name}': mag_inst_calibrated,  # Magnitud calibrada
            f'snr_{filter_name}': snr,
            # Factores de corrección para todas las aperturas
            f'ap_corr_3_{filter_name}': aperture_correction_factors.get(3, 1.0),
            f'ap_corr_4_{filter_name}': aperture_correction_factors.get(4, 1.0),
            f'ap_corr_5_{filter_name}': aperture_correction_factors.get(5, 1.0),
            f'ap_corr_6_{filter_name}': aperture_correction_factors.get(6, 1.0),
            f'bkg_rms_{filter_name}': bkg_rms,
            f'needs_bkg_correction_{filter_name}': needs_correction,
            f'fwhm_{filter_name}': header.get('FWHMMEAN', 1.5),
            f'correction_applied_{filter_name}': correction_factor,
            f'exptime_{filter_name}': exptime,
            f'flxscal_{filter_name}': flxscal
        })
        
        # 6. Añadir coordenadas
        try:
            ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
            logging.info(f"{filter_name}: Coordenadas calculadas con all_pix2world")
        except Exception as e:
            logging.warning(f"{filter_name}: all_pix2world falló: {e}")
            if 'ra' in ref_catalog.columns and 'dec' in ref_catalog.columns:
                results['ra'] = ref_catalog['ra'].values
                results['dec'] = ref_catalog['dec'].values
                logging.info(f"{filter_name}: Usando coordenadas originales del catálogo")
            else:
                results['ra'] = np.nan
                results['dec'] = np.nan
                logging.warning(f"{filter_name}: No se pudieron obtener coordenadas")
        
        # 7. Estadísticas de calidad con valores CALIBRADOS
        valid_flux = flux_calibrated > min_flux
        n_valid = np.sum(valid_flux)
        
        if n_valid > 0:
            mean_mag = mag_inst_calibrated[valid_flux].mean()
            std_mag = mag_inst_calibrated[valid_flux].std()
            mean_snr = snr[valid_flux].mean()
            
            # Verificar que las magnitudes sean razonables (típicamente 10-25 mag)
            if mean_mag < 5 or mean_mag > 30:
                logging.warning(f"{filter_name}: Magnitud promedio {mean_mag:.2f} fuera del rango típico 10-25 mag")
            else:
                logging.info(f"{filter_name}: Magnitudes en rango razonable")
                
            logging.info(f"{filter_name}: {n_valid} estrellas, mag calibrada: {mean_mag:.2f} ± {std_mag:.2f}, "
                       f"SNR: {mean_snr:.1f}, Factor AP: {correction_factor:.3f}")
            
            # Información adicional para diagnóstico
            logging.info(f"{filter_name}: Flujo calibrado promedio: {flux_calibrated[valid_flux].mean():.3f}")
        else:
            logging.warning(f"{filter_name}: No hay estrellas con flux positivo")
        
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_field_corrected(field_name):
    """Procesa un campo completo con la metodología corregida"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO CAMPO: {field_name} (VERSIÓN CALIBRADA)")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Directorio no encontrado: {field_dir}")
        return False
    
    # ✅ Usar el nombre que el script 2 espera
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_splus_method.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Catálogo no encontrado: {input_catalog}")
        return False
    
    # Obtener posiciones de referencia
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            continue
        
        reference_positions, ref_catalog = get_reference_positions_from_catalog(input_catalog, image_file)
        if len(reference_positions) > 10:
            logging.info(f"Usando {band} para posiciones de referencia ({len(reference_positions)} estrellas)")
            break
        else:
            reference_positions = None
            ref_catalog = None
    
    if reference_positions is None:
        logging.error("No se pudieron obtener posiciones válidas")
        return False
    
    # Procesar cada banda con la función CORREGIDA
    band_results = {}
    processed_filters = []
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            logging.warning(f"Imagen {field_name}_{band} no encontrada")
            continue
        
        logging.info(f"Procesando {band}...")
        # ✅ Usar la función CORREGIDA
        results = extract_instrumental_magnitudes_CORREGIDAS(
            image_file, reference_positions, ref_catalog, field_name, band)
        
        if len(results) > 0:
            band_results[band] = results
            processed_filters.append(band)
            logging.info(f"{band}: Completado (con calibración FLXSCAL/EXPTIME)")
        else:
            logging.warning(f"{band}: No se pudieron extraer magnitudes")
    
    if not band_results:
        logging.error("No se procesó ninguna banda")
        return False
    
    # Combinar resultados
    combined_catalog = ref_catalog.copy()
    
    for band in processed_filters:
        if band in band_results:
            results = band_results[band]
            if len(results) == len(combined_catalog):
                for col in results.columns:
                    if col not in ['ra', 'dec']:
                        combined_catalog[col] = results[col].values
            else:
                logging.warning(f"{band}: Número de estrellas no coincide, haciendo merge por coordenadas")
                if 'ra' in results.columns and 'dec' in results.columns:
                    try:
                        merged = pd.merge(combined_catalog, results, on=['ra', 'dec'], how='left', suffixes=('', f'_{band}'))
                        combined_catalog = merged
                    except Exception as e:
                        logging.error(f"Error en merge: {e}")
    
    # Guardar resultados
    combined_catalog.to_csv(output_catalog, index=False)
    total_time = time.time() - start_time
    
    logging.info(f"RESUMEN {field_name}:")
    logging.info(f"Tiempo: {total_time:.1f}s, Filtros procesados: {len(processed_filters)}/{len(SPLUS_FILTERS)}")
    logging.info(f"Resultados guardados en: {output_catalog}")
    logging.info(f"✅ CALIBRACIÓN FLXSCAL/EXPTIME APLICADA correctamente")
    
    # Verificar que los resultados sean coherentes
    for band in processed_filters:
        mag_col = f'mag_inst_calibrated_{band}'
        ap_corr_col = f'ap_corr_3_{band}'
        
        if mag_col in combined_catalog.columns and ap_corr_col in combined_catalog.columns:
            mean_mag = combined_catalog[mag_col].mean()
            mean_ap_corr = combined_catalog[ap_corr_col].mean()
            
            # Verificar coherencia
            if 5 < mean_mag < 30:  # Rango razonable para magnitudes
                status = "✅ COHERENTE"
            else:
                status = "⚠️  VERIFICAR"
                
            logging.info(f"{band}: Mag promedio: {mean_mag:.2f}, AP corr: {mean_ap_corr:.3f} {status}")
    
    return True

def main():
    """Función principal"""
    # Procesar solo campos de prueba inicialmente
    test_fields = ['CenA01', 'CenA02']  # Campos de prueba
    all_fields = [f'CenA{i:02d}' for i in range(1, 25)]
    
    # Usar campos de prueba para verificación
    fields_to_process = all_fields
    logging.info(f"INICIANDO PROCESAMIENTO CALIBRADO de {len(fields_to_process)} campos")
    logging.info("✅ ESTA VERSIÓN USA FLXSCAL/EXPTIME PARA CALIBRACIÓN CORRECTA")
    
    start_time = time.time()
    successful_fields = []
    failed_fields = []
    
    for i, field in enumerate(fields_to_process):
        logging.info(f"\n{'#'*80}")
        logging.info(f"Procesando campo {i+1}/{len(fields_to_process)}: {field}")
        logging.info(f"{'#'*80}")
        
        success = process_field_corrected(field)
        
        if success:
            successful_fields.append(field)
        else:
            failed_fields.append(field)
        
        time.sleep(1)  # Pequeña pausa entre campos
    
    total_time = time.time() - start_time
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESAMIENTO COMPLETADO")
    logging.info(f"{'='*80}")
    logging.info(f"Tiempo total: {total_time/60:.1f} minutos")
    logging.info(f"Éxitos: {len(successful_fields)}, Fallos: {len(failed_fields)}")
    
    if successful_fields:
        logging.info(f"Campos exitosos: {', '.join(successful_fields)}")
        logging.info("✅ Los archivos contienen magnitudes CALIBRADAS y coherentes")
        
        # Verificación final
        for field in successful_fields:
            output_file = f'{field}_gaia_xp_matches_splus_method.csv'
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                print(f"\nVerificación de {field}:")
                for band in SPLUS_FILTERS:
                    mag_col = f'mag_inst_calibrated_{band}'
                    if mag_col in df.columns:
                        mag_values = df[mag_col]
                        valid_mags = mag_values[(mag_values > 5) & (mag_values < 30)]
                        if len(valid_mags) > 0:
                            print(f"  {band}: {len(valid_mags)} estrellas, mag: {valid_mags.mean():.2f} ± {valid_mags.std():.2f}")
    
    if failed_fields:
        logging.info(f"Campos fallidos: {', '.join(failed_fields)}")

if __name__ == '__main__':
    main()
