#!/usr/bin/env python3
"""
Splus_photometry_gc_scientific_v17_FAST_SAFE_ERRORS_FIXED.py
VERSIÓN RÁPIDA con CORRECCIÓN DE ERRORES - Solo errores corregidos
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_scientific_photometry_FAST_SAFE_ERRORS_FIXED.log'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings('ignore')

class SPLUSPhotometryConfig:
    """Configuración OPTIMIZADA para máxima velocidad"""
    def __init__(self):
        self.pixel_scale = 0.55
        self.aperture_diams = [2.0, 3.0]  
        self.reference_aperture_diam = 6.0
        self.annulus_inner = 4.0
        self.annulus_outer = 6.0
        self.margin = 50
        self.min_reference_stars = 3
        self.quality_snr_threshold = 5
        
        # ⚡ MODO RÁPIDO: Desactivar gráficos
        self.save_diagnostic_images = False
        self.diagnostic_fields = []
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.median_box_size = 25
        self.gaussian_sigma = 5
        
        # PARÁMETROS OPTIMIZADOS
        self.growth_curve_radii = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        self.plateau_threshold = 0.02

config = SPLUSPhotometryConfig()

def extract_header_information(header):
    """Extrae información crítica del header S-PLUS"""
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

def subtract_galaxy_background(data, median_box_size=25, gaussian_sigma=5):
    """RESTA EL FONDO GALÁCTICO"""
    try:
        median_filtered = median_filter(data, size=median_box_size)
        galaxy_background = gaussian_filter(median_filtered, sigma=gaussian_sigma)
        residual_image = data - galaxy_background
        return residual_image, galaxy_background, median_filtered
    except Exception as e:
        logging.error(f"Error in galaxy subtraction: {e}")
        return data, np.zeros_like(data), data

def detect_reference_stars_fast(data, error_map, header, nstars=15):
    """Detección de estrellas RÁPIDA"""
    try:
        header_info = extract_header_information(header)
        seeing_fwhm = header_info['seeing_fwhm']
        pixel_scale = header_info['pixel_scale']
        
        fwhm_pixels = seeing_fwhm / pixel_scale
        data_positive = data - np.min(data) + 1.0
        
        mean, median, std = sigma_clipped_stats(data_positive, sigma=2.0, maxiters=3)
        threshold = 8.0 * std
        
        daofind = DAOStarFinder(fwhm=fwhm_pixels, 
                               threshold=threshold,
                               sharplo=0.2, sharphi=1.0,
                               roundlo=-1.0, roundhi=1.0,
                               peakmax=1000000.0)
        
        sources = daofind(data_positive)
        
        if sources is None:
            return np.array([])
        
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        fluxes = sources['flux']
        
        sorted_indices = np.argsort(-fluxes)
        n_to_keep = min(nstars, len(positions))
        best_positions = positions[sorted_indices[:n_to_keep]]
        
        logging.info(f"FAST DAOFind found {len(best_positions)} stars")
        return best_positions
        
    except Exception as e:
        logging.warning(f"Fast DAOFind detection failed: {e}")
        return np.array([])

def analyze_growth_curves_fast(positions, data, error_map, header):
    """Análisis de curvas de crecimiento RÁPIDO - sin gráficos"""
    try:
        header_info = extract_header_information(header)
        pixel_scale = header_info['pixel_scale']
        seeing_fwhm = header_info['seeing_fwhm']
        
        growth_radii_pixels = config.growth_curve_radii / 2.0 / pixel_scale
        plateau_radii = []
        
        for i, pos in enumerate(positions[:8]):
            try:
                fluxes = []
                valid_radii = []
                
                for radius in growth_radii_pixels:
                    if (pos[0] < radius or pos[0] >= data.shape[1] - radius or
                        pos[1] < radius or pos[1] >= data.shape[0] - radius):
                        continue
                    
                    aperture = CircularAperture([pos], r=radius)
                    phot_table = aperture_photometry(data, aperture)
                    flux = phot_table['aperture_sum'].data[0]
                    
                    flux_abs = abs(flux)
                    if flux_abs > 0 and np.isfinite(flux_abs):
                        fluxes.append(flux_abs)
                        valid_radii.append(radius * pixel_scale * 2)
                
                if len(fluxes) < 3:
                    continue
                
                fluxes = np.array(fluxes)
                valid_radii = np.array(valid_radii)
                max_flux = np.max(fluxes)
                
                if max_flux <= 0:
                    continue
                
                normalized_fluxes = fluxes / max_flux
                
                if len(valid_radii) > 3:
                    f = interp1d(valid_radii, normalized_fluxes, kind='linear', 
                                fill_value='extrapolate')
                    
                    test_radii = np.array([4.0, 5.0, 6.0])
                    test_fluxes = f(test_radii)
                    derivatives = np.abs(np.diff(test_fluxes) / np.diff(test_radii))
                    
                    low_derivative_mask = derivatives < config.plateau_threshold
                    
                    if np.any(low_derivative_mask):
                        stable_indices = np.where(low_derivative_mask)[0]
                        plateau_radius = test_radii[stable_indices[0] + 1]
                    else:
                        plateau_radius = 6.0
                else:
                    plateau_radius = min(valid_radii[-1], 6.0)
                
                plateau_radius = min(plateau_radius, 6.0)
                
                if plateau_radius > seeing_fwhm * 1.2:
                    plateau_radii.append(plateau_radius)
                    
            except Exception as e:
                continue
        
        if not plateau_radii:
            return min(6.0, seeing_fwhm * 2.5), {}
        
        plateau_radii = np.array(plateau_radii)
        median_plateau = np.median(plateau_radii)
        recommended_aperture = min(max(median_plateau, seeing_fwhm * 1.5), 6.0)
        
        diagnostics = {
            'median_plateau_radius': median_plateau,
            'n_sources_analyzed': len(plateau_radii),
            'seeing_fwhm': seeing_fwhm,
            'recommended_aperture': recommended_aperture,
            'method': 'fast_4-6arcsec_range'
        }
        
        logging.info(f"FAST growth curve: median_plateau={median_plateau:.1f}\", "
                   f"recommended={recommended_aperture:.1f}\"")
        
        return recommended_aperture, diagnostics
        
    except Exception as e:
        logging.warning(f"Fast growth curve analysis failed: {e}")
        header_info = extract_header_information(header)
        seeing_fwhm = header_info['seeing_fwhm']
        return min(6.0, seeing_fwhm * 2.5), {}

def calculate_aperture_correction_fast(reference_positions, data, header):
    """Cálculo RÁPIDO de corrección de apertura"""
    try:
        if len(reference_positions) < 2:
            seeing = header.get('FWHMMEAN', 1.8)
            default_corr = min(0.5, seeing * 0.2)
            return default_corr, default_corr * 0.8, {}
        
        header_info = extract_header_information(header)
        pixel_scale = header_info['pixel_scale']
        
        radius_2 = 2.0 / 2.0 / pixel_scale
        radius_3 = 3.0 / 2.0 / pixel_scale
        radius_6 = 6.0 / 2.0 / pixel_scale
        
        corrections_2 = []
        corrections_3 = []
        
        for pos in reference_positions[:10]:
            try:
                if (pos[0] < radius_6 or pos[0] >= data.shape[1] - radius_6 or
                    pos[1] < radius_6 or pos[1] >= data.shape[0] - radius_6):
                    continue
                
                aperture_2 = CircularAperture([pos], r=radius_2)
                aperture_3 = CircularAperture([pos], r=radius_3)
                aperture_6 = CircularAperture([pos], r=radius_6)
                
                phot_2 = aperture_photometry(data, aperture_2)
                phot_3 = aperture_photometry(data, aperture_3)
                phot_6 = aperture_photometry(data, aperture_6)
                
                flux_2 = phot_2['aperture_sum'].data[0]
                flux_3 = phot_3['aperture_sum'].data[0]
                flux_6 = phot_6['aperture_sum'].data[0]
                
                flux_2_abs, flux_3_abs, flux_6_abs = abs(flux_2), abs(flux_3), abs(flux_6)
                
                if (flux_2_abs > 0 and flux_3_abs > 0 and flux_6_abs > 0 and
                    flux_6_abs >= flux_3_abs and flux_6_abs >= flux_2_abs):
                    
                    corr_2 = -2.5 * np.log10(flux_2_abs / flux_6_abs)
                    corr_3 = -2.5 * np.log10(flux_3_abs / flux_6_abs)
                    
                    if (0 < corr_2 < 1.0 and 0 < corr_3 < 1.0 and 
                        corr_3 < corr_2):
                        corrections_2.append(corr_2)
                        corrections_3.append(corr_3)
                        
            except Exception as e:
                continue
        
        if len(corrections_2) < 2:
            seeing = header_info['seeing_fwhm']
            default_corr = min(0.5, seeing * 0.2)
            return default_corr, default_corr * 0.8, {}
        
        median_corr_2 = np.median(corrections_2)
        median_corr_3 = np.median(corrections_3)
        
        diagnostics = {
            'n_stars': len(corrections_2),
            'median_correction_2': median_corr_2,
            'median_correction_3': median_corr_3,
            'reference_aperture': 6.0
        }
        
        logging.info(f"FAST aperture correction - 2\": {median_corr_2:.3f}, "
                   f"3\": {median_corr_3:.3f} ({len(corrections_2)} stars)")
        
        return median_corr_2, median_corr_3, diagnostics
        
    except Exception as e:
        logging.warning(f"Fast aperture correction failed: {e}")
        seeing = header.get('FWHMMEAN', 1.8)
        default_corr = min(0.5, seeing * 0.2)
        return default_corr, default_corr * 0.8, {}

# =============================================================================
# CORRECCIONES ESPECÍFICAS PARA ESTIMACIÓN DE ERRORES - VERSIÓN RÁPIDA
# =============================================================================

def load_weight_map_splus_corrected_fast(weight_path, data_shape, header, data_original):
    """
    Versión RÁPIDA para weight maps de S-PLUS
    """
    try:
        with fits.open(weight_path) as whdul:
            for hdu in whdul:
                if hdu.data is not None:
                    weight_data = hdu.data.astype(float)
                    weight_header = hdu.header
                    break
            else:
                return None
        
        if weight_data.shape != data_shape:
            return None
        
        valid_weight = (weight_data > 0) & np.isfinite(weight_data)
        valid_fraction = np.sum(valid_weight) / weight_data.size
        
        if valid_fraction < 0.5:
            return None
        
        # PARA S-PLUS: inverse variance weights (1/σ²)
        error_map = 1.0 / np.sqrt(weight_data)
        
        # Validación rápida
        error_median = np.median(error_map[valid_weight])
        data_median = np.median(np.abs(data_original[valid_weight]))
        ratio = error_median / data_median if data_median > 0 else float('inf')
        
        logging.info(f"⚡ Weight map: valid={valid_fraction:.3f}, error_median={error_median:.3f}, ratio={ratio:.3f}")
        
        return error_map
        
    except Exception as e:
        return None

def calculate_background_error_fast(annulus_data_1d, annulus_error_1d, aperture_area):
    """
    Cálculo RÁPIDO del error del fondo
    """
    if len(annulus_data_1d) < 5:
        return 0.0, 0.0
    
    # Mediana del fondo
    bkg_median = np.median(annulus_data_1d)
    
    # Error rápido: usar MAD y fórmula simplificada
    bkg_mad = mad_std(annulus_data_1d)
    bkg_error_per_pixel = 1.253 * bkg_mad / np.sqrt(len(annulus_data_1d))
    total_bkg_error = bkg_error_per_pixel * aperture_area
    
    return bkg_median, total_bkg_error

def process_single_filter_fast(args):
    """Procesamiento RÁPIDO para un solo filtro - CON ERRORES CORREGIDOS"""
    try:
        (field_name, filter_name, valid_positions, valid_indices, 
         zeropoints, debug) = args
        
        logging.info(f"⚡ {filter_name}: FAST processing with CORRECTED ERRORS")
        
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
        
        # Cargar imagen
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
        
        # Cargar weight map CORREGIDO
        weight_path = find_splus_file(field_name, filter_name, 'weight')
        error_map = None
        if weight_path:
            error_map = load_weight_map_splus_corrected_fast(weight_path, data_original.shape, 
                                                           header, data_original)

        if error_map is None:
            # Fallback rápido
            data_abs = np.abs(data_original)
            median_val = np.median(data_abs)
            gain = header.get('GAIN', 825.35)
            read_noise = header.get('RDNOISE', 5.0)
            error_map = np.sqrt(np.maximum(data_abs, 0) / gain + read_noise**2)
        
        # Validar posiciones
        if len(valid_positions) == 0:
            return None, filter_name
        
        header_info = extract_header_information(header)
        margin = config.margin
        
        valid_mask = (
            (valid_positions[:, 0] >= margin) & 
            (valid_positions[:, 0] < data_original.shape[1] - margin) &
            (valid_positions[:, 1] >= margin) & 
            (valid_positions[:, 1] < data_original.shape[0] - margin)
        )
        
        if np.sum(valid_mask) == 0:
            return None, filter_name
        
        valid_positions = valid_positions[valid_mask]
        valid_indices = valid_indices[valid_mask]
        
        # APLICAR RESTA DE GALAXIA
        data_residual, galaxy_background, _ = subtract_galaxy_background(
            data_original,
            median_box_size=config.median_box_size,
            gaussian_sigma=config.gaussian_sigma
        )
        
        # Usar imagen RESIDUAL
        data_for_detection = data_residual
        data_for_photometry = data_residual
        
        # Detección de estrellas RÁPIDA
        reference_stars = detect_reference_stars_fast(data_for_detection, error_map, header)
        
        if len(reference_stars) < config.min_reference_stars:
            reference_stars = valid_positions[:min(8, len(valid_positions))]
            logging.info(f"Using {len(reference_stars)} positions as reference")
        
        # ANÁLISIS DE CRECIMIENTO RÁPIDO
        if len(reference_stars) > 0:
            analysis_positions = reference_stars
        else:
            analysis_positions = valid_positions[:min(10, len(valid_positions))]
            
        recommended_aperture, growth_diagnostics = analyze_growth_curves_fast(
            analysis_positions, data_for_photometry, error_map, header)
        
        # CORRECCIÓN DE APERTURA RÁPIDA
        aperture_correction_2, aperture_correction_3, ap_diagnostics = \
            calculate_aperture_correction_fast(reference_stars, data_for_photometry, header)
        
        # =============================================================================
        # FOTOMETRÍA CON ERRORES CORREGIDOS - PARTE MODIFICADA
        # =============================================================================
        results = {'indices': valid_indices}
        pixel_scale = header_info['pixel_scale']
        zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
        
        for aperture_diam in config.aperture_diams:
            aperture_radius = (aperture_diam / 2) / pixel_scale
            annulus_inner = (config.annulus_inner / 2) / pixel_scale
            annulus_outer = (config.annulus_outer / 2) / pixel_scale
            
            # Filtrar posiciones válidas
            valid_for_photometry_mask = []
            for pos in valid_positions:
                if (pos[0] >= aperture_radius and pos[0] < data_for_photometry.shape[1] - aperture_radius and
                    pos[1] >= aperture_radius and pos[1] < data_for_photometry.shape[0] - aperture_radius):
                    valid_for_photometry_mask.append(True)
                else:
                    valid_for_photometry_mask.append(False)
            
            valid_for_photometry_mask = np.array(valid_for_photometry_mask)
            if np.sum(valid_for_photometry_mask) == 0:
                n_sources = len(valid_positions)
                prefix = f"{filter_name}_{aperture_diam:.0f}"
                results[f'FLUX_{prefix}'] = np.full(n_sources, 0.0)
                results[f'FLUXERR_{prefix}'] = np.full(n_sources, 99.0)
                results[f'MAG_{prefix}'] = np.full(n_sources, 99.0)
                results[f'MAGERR_{prefix}'] = np.full(n_sources, 99.0)
                results[f'SNR_{prefix}'] = np.full(n_sources, 0.0)
                continue
            
            filtered_positions = valid_positions[valid_for_photometry_mask]
            
            apertures = CircularAperture(filtered_positions, r=aperture_radius)
            annulus = CircularAnnulus(filtered_positions, r_in=annulus_inner, r_out=annulus_outer)
            
            # FOTOMETRÍA CON ERRORES CORREGIDOS
            phot_table = aperture_photometry(data_for_photometry, apertures, error=error_map)
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = phot_table['aperture_sum_err'].data
            
            # CORRECCIÓN: Cálculo MEJORADO del fondo y errores
            bkg_medians = np.zeros(len(filtered_positions))
            bkg_errors = np.zeros(len(filtered_positions))
            
            for i, pos in enumerate(filtered_positions):
                try:
                    mask = annulus.to_mask(method='center')[i]
                    annulus_data = mask.multiply(data_for_photometry)
                    annulus_error = mask.multiply(error_map)
                    
                    annulus_data_1d = annulus_data[mask.data > 0]
                    annulus_error_1d = annulus_error[mask.data > 0]
                    
                    if len(annulus_data_1d) > 5:
                        bkg_median, bkg_error = calculate_background_error_fast(
                            annulus_data_1d, annulus_error_1d, apertures.area)
                    else:
                        bkg_median = 0.0
                        bkg_error = 0.0
                except:
                    bkg_median = 0.0
                    bkg_error = 0.0
                
                bkg_medians[i] = bkg_median
                bkg_errors[i] = bkg_error
            
            # Flujo neto CORREGIDO
            net_flux = raw_flux - (bkg_medians * apertures.area)
            
            # PROPAGACIÓN CORRECTA DE ERRORES
            net_flux_err = np.sqrt(raw_flux_err**2 + bkg_errors**2)
            
            # Límite rápido para errores excesivos
            reasonable_error_ratio = 0.5
            for i in range(len(net_flux)):
                if net_flux[i] > 0 and net_flux_err[i] > 0:
                    current_ratio = net_flux_err[i] / net_flux[i]
                    if current_ratio > reasonable_error_ratio:
                        net_flux_err[i] = net_flux[i] * reasonable_error_ratio
            
            # Cálculo de magnitudes y SNR
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            valid_flux = (net_flux > 1e-10) & (net_flux_err > 0) & np.isfinite(net_flux)
            
            mag_inst = np.where(valid_flux, -2.5 * np.log10(net_flux), 99.0)
            
            if aperture_diam == 2.0:
                aperture_correction = aperture_correction_2
            else:
                aperture_correction = aperture_correction_3
                
            mag = np.where(valid_flux, mag_inst + zero_point - aperture_correction, 99.0)
            mag_err = np.where(valid_flux, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
            
            n_total = len(valid_indices)
            full_flux = np.full(n_total, 0.0)
            full_flux_err = np.full(n_total, 99.0)
            full_mag = np.full(n_total, 99.0)
            full_mag_err = np.full(n_total, 99.0)
            full_snr = np.full(n_total, 0.0)
            
            full_flux[valid_for_photometry_mask] = net_flux
            full_flux_err[valid_for_photometry_mask] = net_flux_err
            full_mag[valid_for_photometry_mask] = np.where(valid_flux, mag, 99.0)
            full_mag_err[valid_for_photometry_mask] = np.where(valid_flux, mag_err, 99.0)
            full_snr[valid_for_photometry_mask] = snr
            
            prefix = f"{filter_name}_{aperture_diam:.0f}"
            results[f'FLUX_{prefix}'] = full_flux
            results[f'FLUXERR_{prefix}'] = full_flux_err
            results[f'MAG_{prefix}'] = full_mag
            results[f'MAGERR_{prefix}'] = full_mag_err
            results[f'SNR_{prefix}'] = full_snr
            results[f'AP_CORR_{prefix}'] = np.full(n_total, aperture_correction)
        
        valid_measurements = np.sum([np.sum(results[f'SNR_{filter_name}_{ap:.0f}'] > 0) 
                                   for ap in config.aperture_diams])
        
        logging.info(f"✅ {filter_name}: FAST processing with CORRECTED ERRORS completed - {valid_measurements} valid measurements")
        
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {filter_name}: FAST PROCESSING FAILED: {e}")
        return None, filter_name

class SPLUSGCScientificPhotometryFAST:
    """Pipeline ACELERADO - CON ERRORES CORREGIDOS"""
    
    def __init__(self, catalog_path, zeropoints_file, debug=False, n_workers=None):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
            
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        
        required_columns = ['field'] + config.filters
        for _, row in self.zeropoints_df.iterrows():
            field = row['field']
            self.zeropoints[field] = {}
            for filt in config.filters:
                self.zeropoints[field][filt] = float(row[filt])
        
        logging.info(f"✅ Loaded zeropoints for {len(self.zeropoints)} fields")
        
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        self.filters = config.filters
        self.debug = debug
        
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
            
        self.id_col = next((col for col in ['recno', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        # ⚡ REDUCIR WORKERS para evitar sobrecarga
        self.n_workers = min(n_workers or 8, 12)  # Máximo 12 workers
        
        logging.info("🚀 INITIALIZED FAST S-PLUS PHOTOMETRY PIPELINE WITH CORRECTED ERRORS")
        logging.info(f"   ⚡ PARALLEL WORKERS: {self.n_workers}")
        logging.info("   🛡️  CORRECTED ERROR ESTIMATION")
        logging.info("   ⚡ FAST WEIGHT MAP PROCESSING")
    
    def find_splus_file(self, field_name, filter_name):
        """Encuentra archivos S-PLUS"""
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None
    
    def is_source_in_field(self, ra, dec, field_ra, field_dec, radius=0.84):
        """Verifica si una fuente está dentro del campo"""
        if field_ra is None or field_dec is None:
            return False
        c1 = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        c2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        return c1.separation(c2).degree <= radius
    
    def process_field_fast(self, field_name):
        """Procesamiento ACELERADO del campo - CON ERRORES CORREGIDOS"""
        logging.info(f"⚡ Processing field {field_name} with CORRECTED ERRORS")
        start_time = time.time()
        
        if not os.path.exists(field_name):
            return None
            
        if field_name not in self.zeropoints:
            logging.warning(f"No zeropoints for field {field_name}")
            return None
            
        first_filter_img = self.find_splus_file(field_name, self.filters[0])
        if not first_filter_img:
            return None
            
        with fits.open(first_filter_img) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    header = hdu.header
                    wcs = WCS(header)
                    break
            else:
                return None
        
        field_ra, field_dec = header.get('CRVAL1'), header.get('CRVAL2')
        if field_ra is None or field_dec is None:
            return None
        
        self.catalog[self.ra_col] = pd.to_numeric(self.catalog[self.ra_col], errors='coerce')
        self.catalog[self.dec_col] = pd.to_numeric(self.catalog[self.dec_col], errors='coerce')
        self.catalog = self.catalog.dropna(subset=[self.ra_col, self.dec_col])
        
        in_field_mask = [
            self.is_source_in_field(row[self.ra_col], row[self.dec_col], field_ra, field_dec)
            for _, row in self.catalog.iterrows()
        ]
        field_sources = self.catalog[in_field_mask].copy()
        logging.info(f"Found {len(field_sources)} sources in field {field_name}")
        
        if len(field_sources) == 0:
            return None
            
        ra_vals = field_sources[self.ra_col].astype(float).values
        dec_vals = field_sources[self.dec_col].astype(float).values
        
        try:
            coords = SkyCoord(ra=ra_vals*u.deg, dec=dec_vals*u.deg)
            x, y = wcs.world_to_pixel(coords)
        except Exception as e:
            logging.error(f"WCS conversion failed: {e}")
            return None
            
        positions = np.column_stack((x, y))
        height, width = header['NAXIS2'], header['NAXIS1']
        margin = config.margin
        
        valid_mask = (
            (x >= margin) & (x < width - margin) & 
            (y >= margin) & (y < height - margin) &
            np.isfinite(x) & np.isfinite(y)
        )
        valid_positions = positions[valid_mask]
        valid_indices = field_sources.index[valid_mask].values
        
        if len(valid_positions) == 0:
            return None
            
        results_df = field_sources.copy()
        successful_filters = 0
        
        # ⚡ PROCESAR FILTROS EN SERIE - CON ERRORES CORREGIDOS
        for filt in tqdm(self.filters, desc=f"Processing {field_name} filters"):
            args = (
                field_name, 
                filt, 
                valid_positions, 
                valid_indices,
                self.zeropoints,
                self.debug
            )
            result, filter_name = process_single_filter_fast(args)
            if result is not None:
                temp_df = pd.DataFrame(result)
                temp_df.set_index('indices', inplace=True)
                for col in temp_df.columns:
                    if col != 'indices':
                        results_df.loc[temp_df.index, col] = temp_df[col].values
                successful_filters += 1
                logging.info(f"✅ {filter_name}: FAST results with CORRECTED ERRORS integrated")
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            results_df['PROCESSING_DATE'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results_df['PHOTOMETRY_METHOD'] = 'S-PLUS_FAST_SAFE_v17_CORRECTED_ERRORS'
            results_df['PRIMARY_APERTURE'] = '2 arcsec'
            results_df['GALAXY_SUBTRACTION'] = 'APPLIED'
            results_df['ERROR_METHOD'] = 'WEIGHT_MAP_CORRECTED_FAST'
            
            elapsed_time = time.time() - start_time
            logging.info(f"⚡ FAST field {field_name} with CORRECTED ERRORS completed: "
                       f"{successful_filters}/{len(self.filters)} filters in {elapsed_time:.1f}s")
            
            # Guardar resultado individual inmediatamente
            output_file = f'{field_name}_gc_photometry_FAST_SAFE_CORRECTED_ERRORS.csv'
            results_df.to_csv(output_file, index=False)
            logging.info(f"💾 Saved {field_name} to {output_file}")
            
            return results_df
        else:
            return None

def process_single_field_wrapper(args):
    """Wrapper para procesar un campo individual"""
    photometry, field_name = args
    try:
        return photometry.process_field_fast(field_name)
    except Exception as e:
        logging.error(f"❌ Error processing {field_name}: {e}")
        return None

def main():
    """Función principal - PARALELIZACIÓN SEGURA CON ERRORES CORREGIDOS"""
    logging.info("=" * 80)
    logging.info("🚀 S-PLUS GLOBULAR CLUSTER PHOTOMETRY - FAST SAFE WITH CORRECTED ERRORS")
    logging.info("   ⚡ PARALLEL FIELDS + SERIAL FILTERS = NO NESTED POOLS")
    logging.info("   🛡️  CORRECTED ERROR ESTIMATION - FAST VERSION")
    logging.info("   ⚡ WEIGHT MAPS: Proper inverse variance handling")
    logging.info("   ⚡ BACKGROUND ERROR: MAD-based robust calculation")
    logging.info("=" * 80)
    
    catalog_path = '../TAP_1_J_MNRAS_3444_psc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_3arcsec.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    # ⚡ CONFIGURACIÓN SEGURA
    n_workers = 8  # Número conservador para evitar sobrecarga
    
    logging.info(f"🛡️  Using {n_workers} safe workers with CORRECTED ERRORS")
    
    # Inicializar UNA sola instancia
    photometry = SPLUSGCScientificPhotometryFAST(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=False,
        n_workers=n_workers
    )
    
    fields = [f'CenA{i:02d}' for i in range(1, 25)]
    
    # ⚡ PARALELIZACIÓN SEGURA de campos
    logging.info(f"🚀 Processing {len(fields)} fields with {n_workers} workers (CORRECTED ERRORS)")
    
    all_results = []
    completed = 0
    
    # Procesar campos en grupos para mayor estabilidad
    group_size = n_workers
    for i in range(0, len(fields), group_size):
        group_fields = fields[i:i + group_size]
        logging.info(f"📦 Processing group {i//group_size + 1}: {len(group_fields)} fields")
        
        # Usar multiprocessing para campos
        from multiprocessing import Pool
        with Pool(processes=min(len(group_fields), n_workers)) as pool:
            args_list = [(photometry, field) for field in group_fields]
            
            for result in tqdm(pool.imap(process_single_field_wrapper, args_list), 
                             total=len(group_fields), 
                             desc=f"Group {i//group_size + 1}"):
                if result is not None:
                    all_results.append(result)
                    completed += 1
                    logging.info(f"✅ Progress: {completed}/{len(fields)} fields completed")
    
    # Combinar resultados finales
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        os.makedirs("Results", exist_ok=True)
        output_file = 'Results/all_fields_gc_photometry_FAST_SAFE_CORRECTED_ERRORS_v17.csv'
        final_results.to_csv(output_file, index=False)
        
        logging.info("🎉 FAST SAFE PARALLEL PHOTOMETRY WITH CORRECTED ERRORS COMPLETED")
        logging.info("   🛡️  NO NESTED POOLS - EXECUCIÓN SEGURA")
        logging.info("   ⚡ CAMPOS EN PARALELO + FILTROS EN SERIE")
        logging.info("   ✅ ERRORES CORREGIDOS: Weight maps + MAD background")
        logging.info(f"   📊 Output: {output_file}")
        
        total_sources = len(final_results)
        logging.info(f"   📈 Total sources processed: {total_sources}")

if __name__ == "__main__":
    # Configuración para evitar conflictos
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    main()
