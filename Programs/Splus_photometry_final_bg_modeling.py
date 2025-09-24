#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected.py - Extracción de magnitudes instrumentales de estrellas de referencia
MAGNITUDES SIN APERTURE CORRECTION (para cálculo de zero points) + Factores de corrección para cúmulos
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
from photutils.segmentation import detect_sources
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAMETER = 3  # 3 arcsec para estrellas de referencia (sin corregir)
APERTURES_FOR_CORRECTION = [3, 4, 5, 6]  # Aperturas para calcular factores de corrección

def find_valid_image_hdu(fits_file):
    """Encuentra la primera HDU válida con datos y WCS en un archivo FITS"""
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
    """
    Verifica si la imagen SPLUS ya tiene el fondo restado
    Returns: (data_corrected, bkg_rms, needs_correction)
    """
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Criterio para imágenes SPLUS: mediana cercana a cero indica fondo ya restado
        background_already_subtracted = (abs(median) < 0.1 * std)
        
        if background_already_subtracted:
            logging.info(f"{filter_name}: Fondo ya restado (mediana={median:.3f}, std={std:.3f})")
            return data, std, False
        else:
            logging.warning(f"{filter_name}: Posible fondo residual (mediana={median:.3f}, std={std:.3f})")
            # Aplicar corrección mínima
            return apply_minimal_background_correction(data, filter_name), std, True
            
    except Exception as e:
        logging.warning(f"Error verificando fondo: {e}. Usando imagen original.")
        return data, np.std(data), False

def apply_minimal_background_correction(data, filter_name):
    """
    Aplica corrección de fondo mínima y conservadora
    """
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Solo corregir si hay variación significativa
        if std < 1.0:
            logging.info(f"{filter_name}: Variación mínima, no se aplica corrección de fondo")
            return data
        
        # Crear máscara para objetos muy brillantes (umbral alto)
        mask = data > median + 15 * std
        
        # Dilatación mínima
        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        
        # Usar boxes grandes para modelar solo variaciones de gran escala
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(data, 
                          box_size=150,  # Boxes grandes
                          filter_size=5,  # Suavizado moderado
                          sigma_clip=sigma_clip, 
                          bkg_estimator=MedianBackground(), 
                          mask=dilated_mask,
                          exclude_percentile=30)  # Excluir más pixels
        
        # Solo restar si el modelo muestra estructura significativa
        bkg_range = np.max(bkg.background) - np.min(bkg.background)
        if bkg_range < 2 * std:
            data_corrected = data - np.median(bkg.background)  # Solo mediana
            logging.info(f"{filter_name}: Corrección mínima aplicada (solo mediana)")
        else:
            data_corrected = data - bkg.background
            logging.info(f"{filter_name}: Corrección completa aplicada")
        
        return data_corrected
        
    except Exception as e:
        logging.warning(f"Corrección de fondo falló: {e}. Usando imagen original.")
        return data

def calculate_aperture_correction_factors(data, header, star_positions, filter_name, max_stars=50):
    """
    Calcula factores de aperture correction para diferentes tamaños de apertura
    Estos factores se usarán en el Script 2 para corregir los cúmulos
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        
        # Seleccionar subconjunto de estrellas brillantes para eficiencia
        if len(star_positions) > max_stars:
            # Calcular flujos aproximados para ranking
            temp_aperture = CircularAperture(star_positions, r=5/pixscale)
            temp_phot = aperture_photometry(data, temp_aperture)
            fluxes = temp_phot['aperture_sum'].data
            
            # Seleccionar estrellas brillantes pero no saturadas
            valid_fluxes = fluxes[fluxes > 0]
            if len(valid_fluxes) > 10:
                flux_threshold = np.percentile(valid_fluxes, 70)
                bright_indices = np.where(fluxes >= flux_threshold)[0]
                if len(bright_indices) > max_stars:
                    bright_indices = bright_indices[:max_stars]
                star_positions = star_positions[bright_indices]
                logging.info(f"{filter_name}: Usando {len(star_positions)} estrellas brillantes para cálculo de corrección")
        
        # Definir aperturas para curva de crecimiento
        aperture_diameters = np.array([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50])
        aperture_radii_px = (aperture_diameters / 2) / pixscale
        
        # Medir curvas de crecimiento
        growth_fluxes = []
        for radius in aperture_radii_px:
            apertures = CircularAperture(star_positions, r=radius)
            phot_table = aperture_photometry(data, apertures)
            growth_fluxes.append(phot_table['aperture_sum'].data)
        
        growth_fluxes = np.array(growth_fluxes)
        total_fluxes = growth_fluxes[-1]  # Flujo en apertura más grande (50 arcsec)
        
        # Filtrar estrellas con flujos válidos
        valid_mask = (total_fluxes > 0) & (total_fluxes < np.percentile(total_fluxes[total_fluxes > 0], 95))
        
        if np.sum(valid_mask) < 5:
            logging.warning(f"{filter_name}: Estrellas insuficientes para cálculo de corrección")
            return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}
        
        # Curvas de crecimiento normalizadas
        growth_fluxes_valid = growth_fluxes[:, valid_mask]
        total_fluxes_valid = total_fluxes[valid_mask]
        normalized_curves = growth_fluxes_valid / total_fluxes_valid
        
        # Curva de crecimiento mediana (robusta)
        median_growth = np.median(normalized_curves, axis=1)
        
        # Calcular factores de corrección para cada apertura
        correction_factors = {}
        
        for aperture_size in APERTURES_FOR_CORRECTION:
            target_radius_px = (aperture_size / 2) / pixscale
            idx = np.argmin(np.abs(aperture_radii_px - target_radius_px))
            fraction_in_aperture = median_growth[idx]
            
            if fraction_in_aperture > 0 and fraction_in_aperture <= 1:
                correction_factor = 1.0 / fraction_in_aperture
                # Aplicar límites razonables
                correction_factor = np.clip(correction_factor, 1.0, 5.0)
            else:
                correction_factor = 1.0
                logging.warning(f"{filter_name}: Fracción inválida para {aperture_size}arcsec: {fraction_in_aperture}")
            
            correction_factors[aperture_size] = correction_factor
        
        n_stars = np.sum(valid_mask)
        logging.info(f"{filter_name}: Factores de corrección calculados con {n_stars} estrellas: {correction_factors}")
        
        return correction_factors
        
    except Exception as e:
        logging.error(f"Error calculando factores de corrección para {filter_name}: {e}")
        return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia desde el catálogo usando WCS"""
    try:
        ref_catalog = pd.read_csv(catalog_path)
        
        # Encontrar HDU válida
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if wcs is None:
            logging.error(f"No se pudo encontrar WCS válido en {image_path}")
            return np.array([]), pd.DataFrame()
        
        data_shape = hdu.data.shape
        
        # Convertir coordenadas a píxeles
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        
        # Usar SkyCoord para mejor precisión
        coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
        x, y = wcs.world_to_pixel(coords)
        
        # Filtrar posiciones válidas
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        
        if data_shape is not None:
            # Margen conservador para evitar bordes
            margin_pixels = 100
            valid_mask = valid_mask & (x > margin_pixels) & (x < data_shape[1] - margin_pixels)
            valid_mask = valid_mask & (y > margin_pixels) & (y < data_shape[0] - margin_pixels)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Encontradas {len(positions)} posiciones válidas de {len(ref_catalog)} estrellas")
        
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error obteniendo posiciones de referencia: {e}")
        return np.array([]), pd.DataFrame()

def find_image_file(field_dir, field_name, filter_name):
    """Encuentra el archivo de imagen con diferentes extensiones posibles"""
    possible_extensions = [
        f"{field_name}_{filter_name}.fits.fz",
        f"{field_name}_{filter_name}.fits",
        f"{field_name}_{filter_name}.fz"
    ]
    
    for ext in possible_extensions:
        image_path = os.path.join(field_dir, ext)
        if os.path.exists(image_path):
            return image_path
    
    return None

def extract_instrumental_magnitudes_uncorrected(image_path, reference_positions, field_name, filter_name):
    """
    Extrae magnitudes instrumentales SIN APERTURE CORRECTION para estrellas de referencia
    Estas magnitudes se usarán para calcular zero points
    """
    try:
        # Encontrar HDU válida
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            logging.error(f"No se pudo encontrar HDU válida en {image_path}")
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        pixscale = header.get('PIXSCALE', 0.55)
        logging.info(f"{filter_name}: Procesando imagen {data.shape}, pixscale={pixscale}")
        
        # 1. VERIFICAR Y CORREGIR FONDO SI ES NECESARIO
        data_corrected, bkg_rms, needs_correction = check_splus_background_status(data, filter_name)
        
        # 2. CALCULAR FACTORES DE CORRECCIÓN DE APERTURA (para usar en cúmulos)
        aperture_correction_factors = calculate_aperture_correction_factors(
            data_corrected, header, reference_positions, filter_name)
        
        # 3. FOTOMETRÍA PARA ESTRELLAS DE REFERENCIA (SIN CORREGIR)
        aperture_radius_px = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_px)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        # Fotometría básica
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        # Restar fondo local
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux_uncorrected = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # 4. MAGNITUDES INSTRUMENTALES SIN CORREGIR (para zero points)
        min_flux = 1e-10
        mag_inst_uncorrected = -2.5 * np.log10(np.maximum(flux_uncorrected, min_flux))
        
        # Cálculo de errores
        flux_err = np.sqrt(np.abs(flux_uncorrected) + (apertures.area * bkg_rms**2))
        snr = np.where(flux_err > 0, flux_uncorrected / flux_err, 0.0)
        
        # 5. CREAR DATAFRAME DE RESULTADOS
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            # Flux y magnitudes SIN corregir (para zero points)
            f'flux_uncorrected_{filter_name}': flux_uncorrected,
            f'flux_err_{filter_name}': flux_err,
            f'mag_inst_uncorrected_{filter_name}': mag_inst_uncorrected,
            f'snr_{filter_name}': snr,
            # Factores de corrección para usar en cúmulos (Script 2)
            f'ap_corr_3_{filter_name}': aperture_correction_factors.get(3, 1.0),
            f'ap_corr_4_{filter_name}': aperture_correction_factors.get(4, 1.0),
            f'ap_corr_5_{filter_name}': aperture_correction_factors.get(5, 1.0),
            f'ap_corr_6_{filter_name}': aperture_correction_factors.get(6, 1.0),
            # Información adicional
            f'bkg_rms_{filter_name}': bkg_rms,
            f'needs_bkg_correction_{filter_name}': needs_correction
        })
        
        # Añadir coordenadas celestiales
        try:
            ra, dec = wcs.pixel_to_world(reference_positions[:, 0], reference_positions[:, 1])
            results['ra'] = ra.deg
            results['dec'] = dec.deg
        except Exception as e:
            logging.warning(f"{filter_name}: No se pudieron calcular coordenadas WCS: {e}")
            # Intentar método alternativo
            try:
                ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
                results['ra'] = ra
                results['dec'] = dec
            except:
                results['ra'] = np.nan
                results['dec'] = np.nan
        
        # Estadísticas de calidad
        valid_flux = flux_uncorrected > 0
        n_valid = np.sum(valid_flux)
        
        if n_valid > 0:
            mean_mag = mag_inst_uncorrected[valid_flux].mean()
            std_mag = mag_inst_uncorrected[valid_flux].std()
            mean_snr = snr[valid_flux].mean()
            
            logging.info(f"{filter_name}: {n_valid}/{len(results)} estrellas válidas, "
                       f"mag = {mean_mag:.2f} ± {std_mag:.2f}, SNR = {mean_snr:.2f}")
        else:
            logging.warning(f"{filter_name}: No hay estrellas con flux positivo")
        
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        return pd.DataFrame()

def process_field_corrected(field_name):
    """Procesa un campo completo para todas las bandas SPLUS"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO CAMPO: {field_name}")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Directorio de campo no encontrado: {field_dir}")
        return False
    
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_splus_method.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Catálogo de entrada no encontrado: {input_catalog}")
        return False
    
    # OBTENER POSICIONES DE REFERENCIA (usar primera imagen disponible)
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            continue
        
        reference_positions, ref_catalog = get_reference_positions_from_catalog(input_catalog, image_file)
        if len(reference_positions) > 10:  # Mínimo 10 estrellas
            logging.info(f"Usando {band} para obtener posiciones de referencia ({len(reference_positions)} estrellas)")
            break
        else:
            logging.warning(f"{band}: Solo {len(reference_positions)} estrellas encontradas, probando siguiente filtro")
            reference_positions = None
            ref_catalog = None
    
    if reference_positions is None or len(reference_positions) == 0:
        logging.error("No se pudieron obtener posiciones de referencia válidas")
        return False
    
    # PROCESAR CADA BANDA
    band_results = {}
    processed_filters = []
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            logging.warning(f"Imagen {field_name}_{band} no encontrada")
            continue
        
        logging.info(f"Procesando {band}...")
        filter_start_time = time.time()
        
        results = extract_instrumental_magnitudes_uncorrected(image_file, reference_positions, field_name, band)
        
        filter_processing_time = time.time() - filter_start_time
        
        if len(results) > 0:
            band_results[band] = results
            processed_filters.append(band)
            logging.info(f"{band}: Completado en {filter_processing_time:.1f} segundos")
        else:
            logging.warning(f"{band}: No se pudieron extraer magnitudes")
    
    if not band_results:
        logging.error("No se procesó ninguna banda correctamente")
        return False
    
    # COMBINAR RESULTADOS CON EL CATÁLOGO ORIGINAL
    combined_catalog = ref_catalog.copy()
    
    for band in processed_filters:
        if band in band_results:
            results = band_results[band]
            
            # Verificar que coincidan las dimensiones
            if len(results) == len(combined_catalog):
                # Añadir todas las columnas de esta banda
                for col in results.columns:
                    if col not in ['ra', 'dec']:  # Evitar duplicar coordenadas
                        combined_catalog[col] = results[col].values
            else:
                logging.warning(f"{band}: Número de estrellas no coincide ({len(results)} vs {len(combined_catalog)})")
                # Intentar merge por coordenadas
                try:
                    if 'ra' in results.columns and 'dec' in results.columns:
                        # Verificar que las coordenadas sean compatibles
                        coord_tolerance = 1e-5  # Grados
                        
                        # Crear columnas de identificación por coordenadas
                        results['coord_id'] = results['ra'].round(5).astype(str) + '_' + results['dec'].round(5).astype(str)
                        combined_catalog['coord_id'] = combined_catalog['ra'].round(5).astype(str) + '_' + combined_catalog['dec'].round(5).astype(str)
                        
                        # Merge por coordenadas
                        merged = pd.merge(combined_catalog, results, on='coord_id', how='left', suffixes=('', f'_{band}'))
                        combined_catalog = merged
                        
                        # Limpiar columnas temporales
                        if 'coord_id' in combined_catalog.columns:
                            combined_catalog = combined_catalog.drop('coord_id', axis=1)
                        if f'coord_id_{band}' in combined_catalog.columns:
                            combined_catalog = combined_catalog.drop(f'coord_id_{band}', axis=1)
                            
                    else:
                        logging.error(f"{band}: No se pueden mergear por coordenadas - columnas faltantes")
                except Exception as e:
                    logging.error(f"{band}: Error en merge por coordenadas: {e}")
    
    # GUARDAR RESULTADOS
    combined_catalog.to_csv(output_catalog, index=False)
    
    total_time = time.time() - start_time
    
    # REPORTAR ESTADÍSTICAS FINALES
    logging.info(f"\nRESUMEN PARA {field_name}:")
    logging.info(f"Tiempo total de procesamiento: {total_time:.1f} segundos")
    logging.info(f"Filtros procesados: {len(processed_filters)}/{len(SPLUS_FILTERS)}")
    logging.info(f"Estrellas procesadas: {len(combined_catalog)}")
    
    for band in processed_filters:
        mag_col = f'mag_inst_uncorrected_{band}'
        if mag_col in combined_catalog.columns:
            valid_mask = combined_catalog[mag_col] < 50
            valid_mags = combined_catalog.loc[valid_mask, mag_col]
            if len(valid_mags) > 0:
                mean_mag = valid_mags.mean()
                std_mag = valid_mags.std()
                logging.info(f"  {band}: {len(valid_mags)} estrellas, mag = {mean_mag:.2f} ± {std_mag:.2f}")
    
    logging.info(f"Resultados guardados en: {output_catalog}")
    return True

def main():
    """Función principal"""
    # Campos a procesar
    all_fields = [
        'CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 
        'CenA07', 'CenA08', 'CenA09', 'CenA10', 'CenA11', 'CenA12',
        'CenA13', 'CenA14', 'CenA15', 'CenA16', 'CenA17', 'CenA18',
        'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24'
    ]
    
    # Para pruebas, procesar solo algunos campos
    test_fields = ['CenA01', 'CenA02']  # Campos de prueba
    fields_to_process = all_fields  # Cambiar a test_fields para pruebas
    
    logging.info(f"INICIANDO PROCESAMIENTO DE {len(fields_to_process)} CAMPOS")
    logging.info(f"Filtros SPLUS: {', '.join(SPLUS_FILTERS)}")
    logging.info(f"Apertura para estrellas referencia: {APERTURE_DIAMETER} arcsec (SIN corregir)")
    logging.info(f"Factores de corrección calculados para: {APERTURES_FOR_CORRECTION} arcsec")
    
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
            logging.info(f"✓ {field} procesado exitosamente")
        else:
            failed_fields.append(field)
            logging.error(f"✗ {field} falló")
        
        # Pequeña pausa entre campos
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    # REPORTE FINAL
    logging.info(f"\n{'='*80}")
    logging.info("PROCESAMIENTO COMPLETADO")
    logging.info(f"{'='*80}")
    logging.info(f"Tiempo total: {total_time/60:.1f} minutos")
    logging.info(f"Campos exitosos: {len(successful_fields)}/{len(fields_to_process)}")
    logging.info(f"Campos fallidos: {len(failed_fields)}")
    
    if successful_fields:
        logging.info(f"Campos exitosos: {', '.join(successful_fields)}")
    
    if failed_fields:
        logging.info(f"Campos fallidos: {', '.join(failed_fields)}")
        
    if successful_fields:
        logging.info(f"\nARCHIVOS GENERADOS:")
        for field in successful_fields:
            output_file = f"{field}_gaia_xp_matches_splus_method.csv"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024 / 1024  # MB
                logging.info(f"  {output_file} ({file_size:.1f} MB)")
    
    logging.info(f"\nUSO DE LOS RESULTADOS:")
    logging.info("1. Magnitudes 'mag_inst_uncorrected_*': Usar para calcular zero points")
    logging.info("2. Factores 'ap_corr_*_*': Usar en Script 2 para corregir cúmulos")
    logging.info("3. Las magnitudes de referencia NO están corregidas por apertura")

if __name__ == '__main__':
    main()
