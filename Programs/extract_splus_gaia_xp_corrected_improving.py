#!/usr/bin/env python3
"""
extract_splus_gaia_xp_corrected_improving.py - Versión corregida con manejo robusto de coordenadas
y método híbrido de aperture correction.
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
from scipy.optimize import curve_fit
import warnings
from scipy import ndimage
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAMETER = 3
APERTURES_FOR_CORRECTION = [3, 4, 5, 6]


# ================
# NUEVAS FUNCIONES DE APERTURE CORRECTION (Versión B)
# ================

def calculate_aperture_correction_robust(data, header, star_positions, filter_name, max_stars=30):
    """
    Calcula aperture correction usando método más robusto y físico
    Basado en el crecimiento esperado del PSF en lugar de normalización a apertura grande
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        fwhm_arcsec = header.get('FWHMMEAN', 1.5)  # FWHM en arcsec del header
        fwhm_pixels = fwhm_arcsec / pixscale
        
        logging.info(f"{filter_name}: FWHM = {fwhm_arcsec:.2f} arcsec ({fwhm_pixels:.1f} pix)")
        
        # Seleccionar estrellas bien aisladas y no saturadas
        if len(star_positions) > max_stars:
            # Medir flujos en apertura pequeña para selección
            small_aperture = CircularAperture(star_positions, r=2*fwhm_pixels)  # 2×FWHM
            phot_table = aperture_photometry(data, small_aperture)
            fluxes = phot_table['aperture_sum'].data
            
            # Seleccionar estrellas con flujo moderado (no saturadas, buena SNR)
            valid_fluxes = fluxes[fluxes > 0]
            if len(valid_fluxes) > 10:
                low_percentile = np.percentile(valid_fluxes, 20)
                high_percentile = np.percentile(valid_fluxes, 80)
                good_flux_mask = (fluxes >= low_percentile) & (fluxes <= high_percentile)
                star_positions = star_positions[good_flux_mask]
                logging.info(f"{filter_name}: Seleccionadas {len(star_positions)} estrellas con flujo moderado")
        
        if len(star_positions) < 5:
            logging.warning(f"{filter_name}: Insuficientes estrellas para aperture correction")
            return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}
        
        # Aperturas para curva de crecimiento (más conservadoras)
        aperture_diameters = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12])  # Hasta 12 arcsec máximo
        aperture_radii_px = (aperture_diameters / 2) / pixscale
        
        growth_fluxes = []
        
        for radius in aperture_radii_px:
            apertures = CircularAperture(star_positions, r=radius)
            phot_table = aperture_photometry(data, apertures)
            growth_fluxes.append(phot_table['aperture_sum'].data)
        
        growth_fluxes = np.array(growth_fluxes)
        
        # En lugar de normalizar a apertura grande, usar modelo de crecimiento teórico
        correction_factors = {}
        
        for aperture_size in APERTURES_FOR_CORRECTION:
            target_radius_arcsec = aperture_size / 2
            target_radius_px = target_radius_arcsec / pixscale
            
            # Encontrar índice más cercano en nuestras mediciones
            idx = np.argmin(np.abs(aperture_radii_px - target_radius_px))
            measured_radius = aperture_radii_px[idx]
            measured_diameter = aperture_diameters[idx]
            
            # Para cada estrella, encontrar la mejor apertura de referencia (4×FWHM)
            reference_radius_px = min(4 * fwhm_pixels, 6 / pixscale)  # Máximo 6 arcsec
            ref_idx = np.argmin(np.abs(aperture_radii_px - reference_radius_px))
            
            # Calcular relación flujo_apertura / flujo_referencia
            fluxes_target = growth_fluxes[idx]
            fluxes_reference = growth_fluxes[ref_idx]
            
            valid_mask = (fluxes_reference > 0) & (fluxes_target > 0)
            
            if np.sum(valid_mask) < 5:
                logging.warning(f"{filter_name}: No hay suficientes medidas válidas para {aperture_size} arcsec")
                correction_factors[aperture_size] = 1.0
                continue
            
            ratios = fluxes_reference[valid_mask] / fluxes_target[valid_mask]
            
            # Usar mediana robusta en lugar de media
            median_ratio = np.median(ratios)
            mad = np.median(np.abs(ratios - median_ratio))
            
            # Filtrar outliers
            good_ratios = ratios[np.abs(ratios - median_ratio) < 3 * mad]
            
            if len(good_ratios) < 5:
                correction_factor = 1.0
            else:
                correction_factor = np.median(good_ratios)
            
            # Aplicar límites físicos razonables
            if aperture_size <= 4:
                correction_factor = np.clip(correction_factor, 1.0, 2.0)
            else:
                correction_factor = np.clip(correction_factor, 1.0, 1.5)
            
            correction_factors[aperture_size] = correction_factor
            
            logging.info(f"{filter_name}: Apertura {aperture_size} arcsec -> Factor: {correction_factor:.3f} "
                       f"(basado en {len(good_ratios)} estrellas)")
        
        logging.info(f"{filter_name}: Factores finales: {correction_factors}")
        return correction_factors
        
    except Exception as e:
        logging.error(f"Error en aperture correction robusto para {filter_name}: {e}")
        return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}


def calculate_aperture_correction_using_psf_model(data, header, star_positions, filter_name):
    """
    Método alternativo: usar modelo teórico de PSF para calcular corrección
    """
    try:
        pixscale = header.get('PIXSCALE', 0.55)
        fwhm_arcsec = header.get('FWHMMEAN', 1.5)
        fwhm_pixels = fwhm_arcsec / pixscale
        
        # Parámetros típicos de PSF para S-PLUS
        # Para un perfil gaussiano, la fracción de luz en una apertura circular es:
        # F(r) = 1 - exp(-r² / (2σ²)), donde σ = FWHM / (2√(2ln2))
        sigma_pixels = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))
        
        correction_factors = {}
        
        for aperture_size in APERTURES_FOR_CORRECTION:
            aperture_radius_arcsec = aperture_size / 2
            aperture_radius_px = aperture_radius_arcsec / pixscale
            
            # Fracción de luz en apertura teórica para PSF gaussiana
            fraction_in_aperture = 1 - np.exp(-0.5 * (aperture_radius_px / sigma_pixels)**2)
            
            # Apertura de referencia (infinito para teoría)
            fraction_in_reference = 1.0
            
            if fraction_in_aperture > 0:
                correction_factor = fraction_in_reference / fraction_in_aperture
            else:
                correction_factor = 1.0
            
            # Ajustar empíricamente basado en datos reales de S-PLUS
            # Factores típicamente entre 1.0-1.3 para aperturas de 3-6 arcsec con FWHM ~1.5"
            if aperture_size == 3:
                correction_factor = min(correction_factor, 1.3)
            elif aperture_size == 4:
                correction_factor = min(correction_factor, 1.2)
            elif aperture_size == 5:
                correction_factor = min(correction_factor, 1.15)
            else:  # 6 arcsec
                correction_factor = min(correction_factor, 1.1)
            
            correction_factors[aperture_size] = correction_factor
        
        logging.info(f"{filter_name}: Factores teóricos (PSF): {correction_factors}")
        return correction_factors
        
    except Exception as e:
        logging.error(f"Error en método teórico para {filter_name}: {e}")
        return {ap: 1.0 for ap in APERTURES_FOR_CORRECTION}


def calculate_empirical_aperture_correction(data, header, star_positions, filter_name):
    """
    Método empírico conservador basado en valores típicos de S-PLUS
    """
    # Valores empíricos típicos para S-PLUS con FWHM ~1.5 arcsec
    empirical_corrections = {
        3: 1.25,  # 25% de corrección para 3 arcsec
        4: 1.15,  # 15% para 4 arcsec  
        5: 1.10,  # 10% para 5 arcsec
        6: 1.05   # 5% para 6 arcsec
    }
    
    # Ajustar basado en FWHM real
    try:
        fwhm_arcsec = header.get('FWHMMEAN', 1.5)
        # Si el seeing es peor, menos corrección necesaria
        seeing_factor = 1.5 / fwhm_arcsec  # Normalizar a 1.5"
        
        adjusted_corrections = {}
        for ap_size, factor in empirical_corrections.items():
            # Ajustar factor basado en calidad de seeing
            adjusted_factor = 1.0 + (factor - 1.0) * seeing_factor
            adjusted_corrections[ap_size] = min(adjusted_factor, 1.3)  # Límite superior
            
        logging.info(f"{filter_name}: Factores empíricos ajustados: {adjusted_corrections}")
        return adjusted_corrections
        
    except:
        logging.info(f"{filter_name}: Usando factores empíricos por defecto: {empirical_corrections}")
        return empirical_corrections


def hybrid_aperture_correction(data, header, star_positions, filter_name):
    """
    Método híbrido: combina métodos robusto, teórico y empírico
    """
    # Método 1: Robustecido
    robust_factors = calculate_aperture_correction_robust(data, header, star_positions, filter_name)
    
    # Método 2: Teórico
    theoretical_factors = calculate_aperture_correction_using_psf_model(data, header, star_positions, filter_name)
    
    # Método 3: Empírico
    empirical_factors = calculate_empirical_aperture_correction(data, header, star_positions, filter_name)
    
    # Combinar ponderadamente
    final_factors = {}
    
    for aperture_size in APERTURES_FOR_CORRECTION:
        factors = []
        weights = []
        
        # Método robusto (peso 2 si tiene datos buenos)
        robust_factor = robust_factors.get(aperture_size, 1.0)
        if robust_factor != 1.0 and 1.0 < robust_factor < 1.5:
            factors.append(robust_factor)
            weights.append(2.0)
        
        # Método teórico (peso 1)
        theoretical_factor = theoretical_factors.get(aperture_size, 1.0)
        factors.append(theoretical_factor)
        weights.append(1.0)
        
        # Método empírico (peso 1.5)
        empirical_factor = empirical_factors.get(aperture_size, 1.0)
        factors.append(empirical_factor)
        weights.append(1.5)
        
        # Promedio ponderado
        if factors:
            final_factor = np.average(factors, weights=weights)
        else:
            final_factor = empirical_factor  # Fallback a empírico
        
        # Límites conservadores
        if aperture_size == 3:
            final_factor = min(final_factor, 1.3)
        elif aperture_size == 4:
            final_factor = min(final_factor, 1.2)
        elif aperture_size == 5:
            final_factor = min(final_factor, 1.15)
        else:  # 6 arcsec
            final_factor = min(final_factor, 1.1)
        
        final_factors[aperture_size] = final_factor
    
    logging.info(f"{filter_name}: Factores híbridos finales: {final_factors}")
    return final_factors


# ================
# FUNCIONES EXISTENTES DE TU SCRIPT (Versión A)
# ================

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
                logging.error(f"Todos los métodos de conversión fallaron: {e2}")
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


def extract_instrumental_magnitudes_uncorrected(image_path, reference_positions, ref_catalog, field_name, filter_name):
    """Extrae magnitudes instrumentales SIN aperture correction"""
    try:
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            logging.error(f"No se pudo encontrar HDU válida en {image_path}")
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        pixscale = header.get('PIXSCALE', 0.55)
        logging.info(f"{filter_name}: Procesando imagen {data.shape}, pixscale={pixscale}")
        
        # 1. Verificar y corregir fondo
        data_corrected, bkg_rms, needs_correction = check_splus_background_status(data, filter_name)
        
        # 2. Calcular factores de corrección (¡USANDO EL NUEVO MÉTODO!)
        aperture_correction_factors = hybrid_aperture_correction(
            data_corrected, header, reference_positions, filter_name)
        
        # 3. Fotometría SIN corrección
        aperture_radius_px = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_px)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        phot_table = aperture_photometry(data_corrected, apertures)
        bkg_phot_table = aperture_photometry(data_corrected, annulus)
        
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux_uncorrected = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # 4. Magnitudes instrumentales SIN corregir
        min_flux = 1e-10
        mag_inst_uncorrected = -2.5 * np.log10(np.maximum(flux_uncorrected, min_flux))
        
        flux_err = np.sqrt(np.abs(flux_uncorrected) + (apertures.area * bkg_rms**2))
        snr = np.where(flux_err > 0, flux_uncorrected / flux_err, 0.0)
        
        # 5. Crear DataFrame
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            f'flux_uncorrected_{filter_name}': flux_uncorrected,
            f'flux_err_{filter_name}': flux_err,
            f'mag_inst_uncorrected_{filter_name}': mag_inst_uncorrected,
            f'snr_{filter_name}': snr,
            f'ap_corr_3_{filter_name}': aperture_correction_factors.get(3, 1.0),
            f'ap_corr_4_{filter_name}': aperture_correction_factors.get(4, 1.0),
            f'ap_corr_5_{filter_name}': aperture_correction_factors.get(5, 1.0),
            f'ap_corr_6_{filter_name}': aperture_correction_factors.get(6, 1.0),
            f'bkg_rms_{filter_name}': bkg_rms,
            f'needs_bkg_correction_{filter_name}': needs_correction,
            f'fwhm_{filter_name}': header.get('FWHMMEAN', 1.5)
        })
        
        # 6. AÑADIR COORDENADAS - VERSIÓN CORREGIDA
        try:
            # Método más robusto: usar all_pix2world
            ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
            logging.info(f"{filter_name}: Coordenadas calculadas con all_pix2world")
        except Exception as e:
            logging.warning(f"{filter_name}: all_pix2world falló: {e}")
            # Usar coordenadas originales del catálogo
            if 'ra' in ref_catalog.columns and 'dec' in ref_catalog.columns:
                results['ra'] = ref_catalog['ra'].values
                results['dec'] = ref_catalog['dec'].values
                logging.info(f"{filter_name}: Usando coordenadas originales del catálogo")
            else:
                results['ra'] = np.nan
                results['dec'] = np.nan
                logging.warning(f"{filter_name}: No se pudieron obtener coordenadas")
        
        # Estadísticas de calidad
        valid_flux = flux_uncorrected > 0
        n_valid = np.sum(valid_flux)
        
        if n_valid > 0:
            mean_mag = mag_inst_uncorrected[valid_flux].mean()
            std_mag = mag_inst_uncorrected[valid_flux].std()
            mean_snr = snr[valid_flux].mean()
            logging.info(f"{filter_name}: {n_valid}/{len(results)} estrellas válidas")
        else:
            logging.warning(f"{filter_name}: No hay estrellas con flux positivo")
        
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        return pd.DataFrame()


def process_field_corrected(field_name):
    """Procesa un campo completo"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO CAMPO: {field_name}")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Directorio no encontrado: {field_dir}")
        return False
    
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
    
    # Procesar cada banda
    band_results = {}
    processed_filters = []
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            logging.warning(f"Imagen {field_name}_{band} no encontrada")
            continue
        
        logging.info(f"Procesando {band}...")
        results = extract_instrumental_magnitudes_uncorrected(image_file, reference_positions, ref_catalog, field_name, band)
        
        if len(results) > 0:
            band_results[band] = results
            processed_filters.append(band)
            logging.info(f"{band}: Completado")
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
                logging.warning(f"{band}: Número de estrellas no coincide")
                # Merge por coordenadas si es posible
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
    logging.info(f"Tiempo: {total_time:.1f}s, Filtros: {len(processed_filters)}/{len(SPLUS_FILTERS)}")
    logging.info(f"Resultados guardados en: {output_catalog}")
    
    return True


def main():
    """Función principal"""
    all_fields = [f'CenA{i:02d}' for i in range(1, 25)]
    test_fields = ['CenA01', 'CenA02']
    fields_to_process = all_fields  # Cambiar a test_fields para pruebas
    
    logging.info(f"INICIANDO PROCESAMIENTO DE {len(fields_to_process)} CAMPOS")
    
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
        
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESAMIENTO COMPLETADO")
    logging.info(f"{'='*80}")
    logging.info(f"Tiempo total: {total_time/60:.1f} minutos")
    logging.info(f"Éxitos: {len(successful_fields)}, Fallos: {len(failed_fields)}")
    
    if successful_fields:
        logging.info(f"Campos exitosos: {', '.join(successful_fields)}")


if __name__ == '__main__':
    main()
