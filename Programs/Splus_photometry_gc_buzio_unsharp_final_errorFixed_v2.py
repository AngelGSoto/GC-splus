#!/usr/bin/env python3
"""
Splus_photometry_gc_scientific_v18_corrected_aperture.py
VERSIÓN CORREGIDA: Corrección de problemas en aperture correction y metodología
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder
from astropy.stats import SigmaClip, sigma_clipped_stats, mad_std
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm
import warnings
import os
import logging
import time
import traceback
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_photometry_v18_corrected.log'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings('ignore')

class SPLUSPhotometryConfig:
    """Configuración CORREGIDA basada en análisis de problemas"""
    def __init__(self):
        self.pixel_scale = 0.55
        # Aperturas principales - más conservadoras
        self.aperture_diams = [2.0, 3.0]  
        self.reference_aperture_diam = 6.0
        self.annulus_inner = 4.0
        self.annulus_outer = 6.0
        self.margin = 50
        self.min_reference_stars = 10  # Aumentado para mayor robustez
        self.quality_snr_threshold = 5
        self.max_aperture_correction = 0.8  # Reducido
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        
        # Parámetros para resta de galaxia
        self.median_box_size = 25
        self.gaussian_sigma = 5
        
        # Configuración de diagnóstico
        self.save_diagnostic_images = True
        self.diagnostic_dir = "aperture_correction_diagnostics"
        
        # Parámetros de crecimiento optimizados
        self.growth_curve_radii = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        self.plateau_threshold = 0.02  # Más conservador

config = SPLUSPhotometryConfig()

def calculate_aperture_correction_corrected(reference_positions, data, header, target_aperture=2.0, reference_aperture=6.0):
    """
    Cálculo CORREGIDO de la corrección de apertura
    PROBLEMA IDENTIFICADO: El signo y aplicación de la corrección estaban incorrectos
    """
    try:
        if len(reference_positions) < config.min_reference_stars:
            logging.warning(f"Not enough reference stars ({len(reference_positions)}) for reliable aperture correction")
            return 0.0, {}  # Devolver 0 en lugar de corrección potencialmente errónea
        
        header_info = extract_header_information(header)
        pixel_scale = header_info['pixel_scale']
        
        # Convertir aperturas a píxeles (radio)
        target_radius = target_aperture / 2.0 / pixel_scale
        reference_radius = reference_aperture / 2.0 / pixel_scale
        
        corrections = []
        valid_stars = 0
        
        for pos in reference_positions:
            try:
                # Verificar que la posición esté lejos de los bordes para ambas aperturas
                if (pos[0] < reference_radius or pos[0] >= data.shape[1] - reference_radius or
                    pos[1] < reference_radius or pos[1] >= data.shape[0] - reference_radius):
                    continue
                
                # Crear aperturas
                target_aperture_obj = CircularAperture([pos], r=target_radius)
                reference_aperture_obj = CircularAperture([pos], r=reference_radius)
                
                # Fotometría
                target_phot = aperture_photometry(data, target_aperture_obj)
                reference_phot = aperture_photometry(data, reference_aperture_obj)
                
                target_flux = target_phot['aperture_sum'].data[0]
                reference_flux = reference_phot['aperture_sum'].data[0]
                
                # Validar flujos
                if (target_flux > 0 and reference_flux > 0 and 
                    np.isfinite(target_flux) and np.isfinite(reference_flux) and
                    reference_flux > target_flux):  # La apertura grande debe tener más flujo
                    
                    # CORRECCIÓN CLAVE: Cálculo correcto del offset
                    # Si reference_flux > target_flux, entonces necesitamos añadir flujo
                    # a la apertura pequeña para igualar a la grande
                    flux_ratio = reference_flux / target_flux
                    
                    # Validar ratio físicamente posible
                    if 1.0 < flux_ratio < 10.0:  # Límites razonables
                        # La corrección en magnitudes es positiva (hace la magnitud más débil)
                        correction = 2.5 * np.log10(flux_ratio)
                        corrections.append(correction)
                        valid_stars += 1
                        
            except Exception as e:
                continue
        
        if len(corrections) < 3:
            logging.warning(f"Only {len(corrections)} valid corrections found")
            return 0.0, {'n_stars': len(corrections), 'status': 'insufficient'}
        
        # Usar mediana robusta
        corrections = np.array(corrections)
        median_correction = np.median(corrections)
        mad_correction = mad_std(corrections)
        
        # Filtrar outliers usando MAD
        filtered_corrections = corrections[
            (corrections >= median_correction - 2 * mad_correction) & 
            (corrections <= median_correction + 2 * mad_correction)
        ]
        
        if len(filtered_corrections) > 0:
            final_correction = np.median(filtered_corrections)
        else:
            final_correction = median_correction
        
        # Validación física de la corrección
        if final_correction > config.max_aperture_correction:
            logging.warning(f"Aperture correction {final_correction:.3f} exceeds maximum allowed, using {config.max_aperture_correction}")
            final_correction = config.max_aperture_correction
        
        diagnostics = {
            'n_stars_total': len(reference_positions),
            'n_stars_valid': valid_stars,
            'n_corrections': len(corrections),
            'median_correction': final_correction,
            'mad_correction': mad_correction,
            'target_aperture': target_aperture,
            'reference_aperture': reference_aperture,
            'status': 'success'
        }
        
        logging.info(f"✅ CORRECTED aperture correction: {final_correction:.3f} mag "
                   f"(based on {len(corrections)} stars, {target_aperture}\" → {reference_aperture}\")")
        
        return final_correction, diagnostics
        
    except Exception as e:
        logging.error(f"Aperture correction calculation failed: {e}")
        return 0.0, {'status': 'error', 'error': str(e)}

def apply_photometry_corrections_corrected(net_flux, net_flux_err, zero_point, aperture_correction):
    """
    Aplicación CORREGIDA de las correcciones fotométricas
    PROBLEMA IDENTIFICADO: El signo de la corrección de apertura estaba invertido
    """
    try:
        # Validar flujos
        valid_flux = (net_flux > 1e-10) & (net_flux_err > 0) & np.isfinite(net_flux) & np.isfinite(net_flux_err)
        
        # Magnitud instrumental
        mag_inst = np.where(valid_flux, -2.5 * np.log10(net_flux), 99.0)
        
        # CORRECCIÓN CLAVE: La corrección de apertura se SUMA (no resta)
        # porque compensa la luz perdida fuera de la apertura pequeña
        mag_corrected = np.where(valid_flux, mag_inst + zero_point + aperture_correction, 99.0)
        
        # Error en magnitud (fórmula estándar)
        mag_err = np.where(valid_flux, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
        
        # SNR
        snr = np.where(valid_flux, net_flux / net_flux_err, 0.0)
        
        return mag_corrected, mag_err, snr
        
    except Exception as e:
        logging.error(f"Error applying photometry corrections: {e}")
        n = len(net_flux) if hasattr(net_flux, '__len__') else 1
        return np.full(n, 99.0), np.full(n, 99.0), np.full(n, 0.0)

def validate_against_taylor_photometry(splus_mags, taylor_mags, filter_mapping, source_ids):
    """
    Validación en tiempo real contra fotometría Taylor
    """
    try:
        validation_results = {}
        
        for splus_filter, taylor_filter in filter_mapping.items():
            splus_col = f'MAG_{splus_filter}_2'  # Usar apertura de 2"
            
            if splus_col in splus_mags.columns and taylor_filter in taylor_mags.columns:
                # Combinar datos
                merged = pd.merge(
                    splus_mags[['T17ID', splus_col]],
                    taylor_mags[['T17ID', taylor_filter]],
                    on='T17ID',
                    how='inner'
                )
                
                # Filtrar valores válidos
                valid_mask = (
                    (merged[splus_col] < 50) & 
                    (merged[taylor_filter] < 50) &
                    np.isfinite(merged[splus_col]) & 
                    np.isfinite(merged[taylor_filter])
                )
                
                valid_data = merged[valid_mask]
                
                if len(valid_data) > 5:
                    differences = valid_data[splus_col] - valid_data[taylor_filter]
                    
                    validation_results[splus_filter] = {
                        'taylor_filter': taylor_filter,
                        'n_sources': len(valid_data),
                        'mean_diff': np.mean(differences),
                        'median_diff': np.median(differences),
                        'std_diff': np.std(differences),
                        'mad_diff': mad_std(differences)
                    }
        
        return validation_results
        
    except Exception as e:
        logging.warning(f"Taylor validation failed: {e}")
        return {}

def extract_header_information(header):
    """Extrae información crítica del header S-PLUS (sin cambios)"""
    info = {
        'pixel_scale': header.get('PIXSCALE', 0.55),
        'seeing_fwhm': header.get('FWHMMEAN', 1.8),
        'exptime': header.get('EXPTIME', header.get('TEXPOSED', 870.0)),
        'gain': header.get('GAIN', 825.35),
        'saturation': header.get('SATURATE', 221.7),
        'airmass': header.get('AIRMASS', 1.1),
        'filter': header.get('FILTER', header.get('BAND', 'Unknown')),
        'field': header.get('FIELD', 'Unknown'),
        'mjd_obs': header.get('MJD-OBS', 0.0),
        'ncombine': header.get('NCOMBINE', 1)
    }
    
    return info

def process_single_filter_corrected(args):
    """Procesamiento CORREGIDO para un solo filtro"""
    try:
        (field_name, filter_name, valid_positions, valid_indices, 
         zeropoints, taylor_catalog, debug) = args
        
        logging.info(f"🔬 {field_name} {filter_name}: Starting CORRECTED processing")
        
        # Cargar imagen (código existente)
        def find_splus_file(field_name, filter_name, file_type='image'):
            patterns = {
                'image': [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"],
                'weight': [f"{field_name}_{filter_name}.weight.fits.fz", 
                          f"{field_name}_{filter_name}.weight.fits"]
            }
            for pattern in patterns[file_type]:
                path = os.path.join(field_name, pattern)
                if os.path.exists(path):
                    return path
            return None
        
        image_path = find_splus_file(field_name, filter_name, 'image')
        if not image_path:
            return None, filter_name
        
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data_original = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                return None, filter_name
        
        # Cargar weight map (código existente)
        weight_path = find_splus_file(field_name, filter_name, 'weight')
        error_map = None
        if weight_path:
            # Usar tu función existente load_weight_map_splus_corrected
            error_map = load_weight_map_splus_corrected(weight_path, data_original.shape, header, data_original)

        if error_map is None:
            # Fallback
            data_abs = np.abs(data_original)
            median_val = np.median(data_abs)
            gain = header.get('GAIN', 825.35)
            read_noise = header.get('RDNOISE', 5.0)
            error_map = np.sqrt(np.maximum(data_abs, 0) / gain + read_noise**2)
        
        # Aplicar resta de galaxia (código existente)
        data_residual, galaxy_background, _ = subtract_galaxy_background(
            data_original,
            median_box_size=config.median_box_size,
            gaussian_sigma=config.gaussian_sigma
        )
        
        # Detección de estrellas de referencia (código existente)
        reference_stars = detect_reference_stars_daofind_corrected(data_residual, error_map, header)
        
        # =============================================================================
        # NUEVA SECCIÓN: CÁLCULO CORREGIDO DE CORRECCIÓN DE APERTURA
        # =============================================================================
        
        aperture_corrections = {}
        aperture_diagnostics = {}
        
        for aperture_diam in config.aperture_diams:
            correction, diagnostics = calculate_aperture_correction_corrected(
                reference_stars, 
                data_residual, 
                header,
                target_aperture=aperture_diam,
                reference_aperture=config.reference_aperture_diam
            )
            
            aperture_corrections[aperture_diam] = correction
            aperture_diagnostics[aperture_diam] = diagnostics
        
        # =============================================================================
        # FOTOMETRÍA CORREGIDA
        # =============================================================================
        
        results = {'indices': valid_indices}
        pixel_scale = extract_header_information(header)['pixel_scale']
        zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
        
        for aperture_diam in config.aperture_diams:
            aperture_radius = (aperture_diam / 2) / pixel_scale
            annulus_inner = (config.annulus_inner / 2) / pixel_scale
            annulus_outer = (config.annulus_outer / 2) / pixel_scale
            
            # Filtrar posiciones válidas
            valid_for_photometry_mask = []
            for pos in valid_positions:
                if (pos[0] >= aperture_radius and pos[0] < data_residual.shape[1] - aperture_radius and
                    pos[1] >= aperture_radius and pos[1] < data_residual.shape[0] - aperture_radius):
                    valid_for_photometry_mask.append(True)
                else:
                    valid_for_photometry_mask.append(False)
            
            valid_for_photometry_mask = np.array(valid_for_photometry_mask)
            if np.sum(valid_for_photometry_mask) == 0:
                n_sources = len(valid_indices)
                prefix = f"{filter_name}_{aperture_diam:.0f}"
                results[f'FLUX_{prefix}'] = np.full(n_sources, 0.0)
                results[f'FLUXERR_{prefix}'] = np.full(n_sources, 99.0)
                results[f'MAG_{prefix}'] = np.full(n_sources, 99.0)
                results[f'MAGERR_{prefix}'] = np.full(n_sources, 99.0)
                results[f'SNR_{prefix}'] = np.full(n_sources, 0.0)
                results[f'AP_CORR_{prefix}'] = np.full(n_sources, aperture_corrections[aperture_diam])
                continue
            
            filtered_positions = valid_positions[valid_for_photometry_mask]
            
            # Fotometría (usar tu función existente calculate_optimized_photometry_errors)
            net_flux, net_flux_err, bkg_medians, bkg_errors = calculate_optimized_photometry_errors(
                data_residual, error_map, filtered_positions, 
                aperture_radius, annulus_inner, annulus_outer
            )
            
            # APLICACIÓN CORREGIDA de correcciones
            mag, mag_err, snr = apply_photometry_corrections_corrected(
                net_flux, net_flux_err, zero_point, aperture_corrections[aperture_diam]
            )
            
            # Crear arrays completos
            n_total = len(valid_indices)
            full_flux = np.full(n_total, 0.0)
            full_flux_err = np.full(n_total, 99.0)
            full_mag = np.full(n_total, 99.0)
            full_mag_err = np.full(n_total, 99.0)
            full_snr = np.full(n_total, 0.0)
            
            full_flux[valid_for_photometry_mask] = net_flux
            full_flux_err[valid_for_photometry_mask] = net_flux_err
            full_mag[valid_for_photometry_mask] = mag
            full_mag_err[valid_for_photometry_mask] = mag_err
            full_snr[valid_for_photometry_mask] = snr
            
            prefix = f"{filter_name}_{aperture_diam:.0f}"
            results[f'FLUX_{prefix}'] = full_flux
            results[f'FLUXERR_{prefix}'] = full_flux_err
            results[f'MAG_{prefix}'] = full_mag
            results[f'MAGERR_{prefix}'] = full_mag_err
            results[f'SNR_{prefix}'] = full_snr
            results[f'AP_CORR_{prefix}'] = np.full(n_total, aperture_corrections[aperture_diam])
        
        # =============================================================================
        # VALIDACIÓN EN TIEMPO REAL
        # =============================================================================
        
        # Crear DataFrame temporal para validación
        temp_df = pd.DataFrame({
            'T17ID': valid_indices,
            f'MAG_{filter_name}_2': results[f'MAG_{filter_name}_2']
        })
        
        filter_mapping = {
            'F378': 'umag', 'F395': 'umag', 'F410': 'gmag', 
            'F430': 'gmag', 'F515': 'gmag', 'F660': 'rmag', 'F861': 'imag'
        }
        
        validation = validate_against_taylor_photometry(
            temp_df, taylor_catalog, {filter_name: filter_mapping[filter_name]}, valid_indices
        )
        
        if validation:
            for filt, stats in validation.items():
                logging.info(f"📊 {field_name} {filter_name} vs Taylor: "
                           f"Δmedian = {stats['median_diff']:.3f}, "
                           f"MAD = {stats['mad_diff']:.3f}, "
                           f"n = {stats['n_sources']}")
        
        valid_measurements = np.sum(results[f'SNR_{filter_name}_2'] > 0)
        logging.info(f"✅ {field_name} {filter_name}: CORRECTED processing completed - "
                   f"{valid_measurements} valid measurements")
        
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {field_name} {filter_name}: CORRECTED processing failed: {e}")
        traceback.print_exc()
        return None, filter_name

# =============================================================================
# FUNCIONES DE APOYO (mantener tus versiones existentes)
# =============================================================================

def load_weight_map_splus_corrected(weight_path, data_shape, header, data_original):
    """Tu función existente - mantener igual"""
    # ... (tu código existente)

def subtract_galaxy_background(data, median_box_size=25, gaussian_sigma=5):
    """Tu función existente - mantener igual"""
    # ... (tu código existente)

def detect_reference_stars_daofind_corrected(data, error_map, header, nstars=30):
    """Tu función existente - mantener igual"""
    # ... (tu código existente)

def calculate_optimized_photometry_errors(data, error_map, positions, aperture_radius, annulus_inner, annulus_outer):
    """Tu función existente - mantener igual"""
    # ... (tu código existente)

def analyze_growth_curves_realistic(positions, data, error_map, header, output_dir="growth_curve"):
    """Tu función existente - mantener igual"""
    # ... (tu código existente)

class SPLUSGCScientificPhotometryCorrected:
    """Pipeline principal corregido"""
    
    def __init__(self, catalog_path, zeropoints_file, taylor_catalog_path, debug=False):
        # Inicialización similar a tu clase existente
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        
        # Cargar catálogo de Taylor para validación
        self.taylor_catalog = pd.read_csv(taylor_catalog_path)
        
        # ... (resto de tu inicialización existente)
        
        logging.info("🎯 INITIALIZED CORRECTED S-PLUS PHOTOMETRY PIPELINE v18")
        logging.info("   - APERTURE CORRECTION: Fixed sign and calculation")
        logging.info("   - VALIDATION: Real-time comparison with Taylor photometry")
        logging.info("   - METHODOLOGY: Consistent approach for all sources")
    
    def process_field_corrected(self, field_name):
        """Procesamiento corregido para un campo"""
        # ... (adaptar tu método process_field_optimized existente)
        
        # En la llamada a process_single_filter_corrected, pasar taylor_catalog
        args = (
            field_name, 
            filt, 
            valid_positions, 
            valid_indices,
            self.zeropoints,
            self.taylor_catalog,  # Nuevo parámetro
            self.debug
        )
        
        result, filter_name = process_single_filter_corrected(args)
        
        # ... (resto del procesamiento)

def main():
    """Función principal corregida"""
    logging.info("=" * 80)
    logging.info("🎯 S-PLUS GLOBULAR CLUSTER PHOTOMETRY v18 - CORRECTED APERTURE")
    logging.info("   CORRECCIONES: Signo y cálculo de corrección de apertura")
    logging.info("   VALIDACIÓN: Comparación en tiempo real con Taylor")
    logging.info("   METODOLOGÍA: Consistente entre estrellas y cúmulos")
    logging.info("=" * 80)
    
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_3arcsec.csv'
    taylor_catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'  # O tu catálogo de Taylor
    
    # Inicializar pipeline corregido
    photometry = SPLUSGCScientificPhotometryCorrected(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        taylor_catalog_path=taylor_catalog_path,
        debug=True
    )
    
    # Procesar campos (similar a tu main existente)
    # ...

if __name__ == "__main__":
    main()
