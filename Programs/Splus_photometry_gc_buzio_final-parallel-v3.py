#!/usr/bin/env python3
"""
Splus_photometry_gc_buzio_final-parallel-v4.py
VERSIÓN CORREGIDA - Mejoras en análisis de fondo y corrección de apertura
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip, sigma_clipped_stats, sigma_clip
from scipy.ndimage import median_filter, gaussian_filter
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import os
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

# Configuración de estilo para plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def exponential_growth_curve(r, a, b, c):
    """Curva de crecimiento exponencial para fitting"""
    return a * (1 - np.exp(-b * r)) + c

def calculate_seeing_from_psf(header):
    """Estimar seeing desde header o usar valor por defecto"""
    fwhm = header.get('FWHMMEAN', header.get('FWHM', header.get('SEEING', 1.8)))
    return float(fwhm)

def analyze_background_variability_improved(data, box_size=50):
    """Analizar variabilidad del fondo mejorado - maneja valores extremos"""
    try:
        # Primero, hacer un sigma clipping agresivo para eliminar valores extremos
        data_clipped = sigma_clip(data, sigma=5, maxiters=3)
        valid_data = data_clipped.data[~data_clipped.mask]
        
        if len(valid_data) == 0:
            return True, 1.0, None
        
        # Usar percentiles para ser robusto a outliers
        background_std = np.std(valid_data)
        background_median = np.median(valid_data)
        
        # Si hay valores extremos, usar percentiles en lugar de std
        p16, p84 = np.percentile(valid_data, [16, 84])
        robust_std = (p84 - p16) / 2
        
        variability_ratio = robust_std / background_median if background_median > 0 else 0
        
        # Umbral más conservador
        needs_unsharp = variability_ratio > 0.02 and background_median > 0.1
        
        return needs_unsharp, variability_ratio, None
        
    except Exception as e:
        logging.warning(f"Improved background analysis failed: {e}")
        return True, 1.0, None

def create_optimized_unsharp_mask(data, median_box_size=25, gaussian_sigma=5):
    """Unsharp masking optimizado con manejo de errores"""
    try:
        # Hacer una copia y normalizar para evitar problemas numéricos
        data_normalized = data.copy()
        data_median = np.median(data_normalized)
        data_std = np.std(data_normalized)
        
        # Clip valores extremos
        clip_min = data_median - 5 * data_std
        clip_max = data_median + 5 * data_std
        data_normalized = np.clip(data_normalized, clip_min, clip_max)
        
        # Aplicar filtros
        median_filtered = median_filter(data_normalized, size=median_box_size)
        gaussian_smoothed = gaussian_filter(median_filtered, sigma=gaussian_sigma)
        unsharp_mask = data_normalized - gaussian_smoothed
        
        return unsharp_mask, median_box_size, gaussian_sigma
        
    except Exception as e:
        logging.error(f"Error in optimized unsharp masking: {e}")
        return data, median_box_size, gaussian_sigma

def save_unsharp_diagnostic(original_data, unsharp_data, field_name, filter_name, output_dir="unsharp_mask"):
    """Guardar diagnóstico de unsharp masking"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Usar percentiles para scaling robusto
        vmin_o, vmax_o = np.percentile(original_data, [5, 95])
        vmin_u, vmax_u = np.percentile(unsharp_data, [5, 95])
        
        # Original
        im0 = axes[0,0].imshow(original_data, vmin=vmin_o, vmax=vmax_o, cmap='gray', origin='lower')
        axes[0,0].set_title(f'Original: {field_name} {filter_name}')
        plt.colorbar(im0, ax=axes[0,0])
        
        # Unsharp masked
        im1 = axes[0,1].imshow(unsharp_data, vmin=vmin_u, vmax=vmax_u, cmap='gray', origin='lower')
        axes[0,1].set_title(f'Unsharp Masked: {field_name} {filter_name}')
        plt.colorbar(im1, ax=axes[0,1])
        
        # Residual
        residual = original_data - unsharp_data
        vmin_r, vmax_r = np.percentile(residual, [5, 95])
        im2 = axes[1,0].imshow(residual, vmin=vmin_r, vmax=vmax_r, cmap='RdBu_r', origin='lower')
        axes[1,0].set_title(f'Residual (Original - Unsharp)')
        plt.colorbar(im2, ax=axes[1,0])
        
        # Histograma
        axes[1,1].hist(original_data.flatten(), bins=100, alpha=0.7, label='Original', density=True, range=(vmin_o, vmax_o))
        axes[1,1].hist(unsharp_data.flatten(), bins=100, alpha=0.7, label='Unsharp', density=True, range=(vmin_u, vmax_u))
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].set_title('Pixel Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{field_name}_{filter_name}_unsharp_diagnostic.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Error saving unsharp diagnostic: {e}")

def find_high_snr_reference_stars_fixed(positions, data, snr_threshold=30, min_separation=50):
    """Versión corregida para encontrar estrellas de referencia"""
    try:
        # Verificar que positions sea un array 2D
        if len(positions) == 0:
            return np.array([])
            
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 2)
        
        test_aperture_radius = 4.0  # arcsec
        pixel_scale = 0.55
        radius_pixels = test_aperture_radius / pixel_scale
        
        snr_values = []
        valid_positions = []
        
        for pos in positions:
            try:
                # Verificar que la posición esté dentro de la imagen
                y, x = int(pos[1]), int(pos[0])
                if (x < radius_pixels or x >= data.shape[1] - radius_pixels or 
                    y < radius_pixels or y >= data.shape[0] - radius_pixels):
                    continue
                
                aperture = CircularAperture([pos], r=radius_pixels)
                annulus = CircularAnnulus([pos], r_in=radius_pixels*1.5, r_out=radius_pixels*2.0)
                
                phot_table = aperture_photometry(data, aperture)
                raw_flux = phot_table['aperture_sum'].data[0]
                
                # Estimación robusta de fondo
                try:
                    mask = annulus.to_mask(method='center')[0]
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    
                    if len(annulus_data_1d) > 10:
                        _, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d, sigma=2.5, maxiters=3)
                    else:
                        bkg_median = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                        bkg_std = np.std(annulus_data_1d) if len(annulus_data_1d) > 0 else 1.0
                except:
                    bkg_median = 0.0
                    bkg_std = 1.0
                
                net_flux = max(0, raw_flux - (bkg_median * aperture.area))
                snr = net_flux / np.sqrt(net_flux + aperture.area * bkg_std**2) if net_flux > 0 else 0
                
                snr_values.append(snr)
                valid_positions.append(pos)
                
            except Exception as e:
                continue
        
        if not snr_values:
            return np.array([])
            
        snr_values = np.array(snr_values)
        valid_positions = np.array(valid_positions)
        
        # Seleccionar estrellas con buen SNR
        high_snr_mask = snr_values > snr_threshold
        high_snr_positions = valid_positions[high_snr_mask]
        high_snr_values = snr_values[high_snr_mask]
        
        if len(high_snr_positions) == 0:
            logging.warning(f"No high-SNR reference stars found (SNR > {snr_threshold})")
            return np.array([])
        
        # Filtrar por separación
        try:
            from scipy.spatial import KDTree
            tree = KDTree(high_snr_positions)
            isolated_mask = np.ones(len(high_snr_positions), dtype=bool)
            
            for i, pos in enumerate(high_snr_positions):
                if len(high_snr_positions) > 1:
                    distances, indices = tree.query(pos, k=min(3, len(high_snr_positions)))
                    if len(distances) > 1 and np.min(distances[1:]) < min_separation:
                        isolated_mask[i] = False
            
            isolated_positions = high_snr_positions[isolated_mask]
            isolated_snr = high_snr_values[isolated_mask]
            
            # Ordenar por SNR y tomar las mejores
            if len(isolated_positions) > 0:
                sorted_indices = np.argsort(-isolated_snr)
                best_positions = isolated_positions[sorted_indices[:min(15, len(isolated_positions))]]
                logging.info(f"Found {len(best_positions)} high-SNR reference stars (SNR > {snr_threshold})")
                return best_positions
                
        except Exception as e:
            logging.warning(f"KDTree failed, using all high-SNR stars: {e}")
            return high_snr_positions[:min(10, len(high_snr_positions))]
        
        return np.array([])
        
    except Exception as e:
        logging.error(f"Error in find_high_snr_reference_stars: {e}")
        return np.array([])

def calculate_physical_aperture_correction_improved(reference_positions, data, filter_name, pixel_scale, seeing, debug=False):
    """
    ✅ CORRECCIÓN DE APERTURA MEJORADA - Más robusta
    """
    try:
        if len(reference_positions) < 2:  # Reducido el mínimo requerido
            logging.warning(f"{filter_name}: Insufficient reference stars ({len(reference_positions)})")
            # Fallback basado en seeing y filtro
            base_corrections = {
                'F378': (-0.25, -0.12), 'F395': (-0.22, -0.10), 'F410': (-0.20, -0.09),
                'F430': (-0.18, -0.08), 'F515': (-0.15, -0.07), 'F660': (-0.12, -0.06), 
                'F861': (-0.10, -0.05)
            }
            default_2, default_3 = base_corrections.get(filter_name, (-0.15, -0.07))
            # Ajustar por seeing
            seeing_factor = seeing / 1.5
            correction_2 = default_2 * seeing_factor
            correction_3 = default_3 * seeing_factor
            return correction_2, correction_3, {}
        
        # Aperturas para curva de crecimiento (en arcsec)
        aperture_diams = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
        aperture_radii = aperture_diams / 2
        
        all_growth_curves = []
        successful_stars = 0
        
        for i, pos in enumerate(reference_positions):
            try:
                fluxes = []
                
                for ap_radius in aperture_radii:
                    radius_pixels = ap_radius / pixel_scale
                    
                    # Verificar que la apertura esté dentro de la imagen
                    y, x = int(pos[1]), int(pos[0])
                    if (x < radius_pixels or x >= data.shape[1] - radius_pixels or 
                        y < radius_pixels or y >= data.shape[0] - radius_pixels):
                        fluxes.append(np.nan)
                        continue
                    
                    aperture = CircularAperture([pos], r=radius_pixels)
                    annulus = CircularAnnulus([pos], r_in=radius_pixels*1.5, r_out=radius_pixels*2.0)
                    
                    # Fotometría
                    phot_table = aperture_photometry(data, aperture)
                    raw_flux = phot_table['aperture_sum'].data[0]
                    
                    # Estimación robusta de fondo
                    try:
                        mask = annulus.to_mask(method='center')[0]
                        annulus_data = mask.multiply(data)
                        annulus_data_1d = annulus_data[mask.data > 0]
                        
                        if len(annulus_data_1d) > 5:
                            _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d, sigma=2.5, maxiters=3)
                        else:
                            bkg_median = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                    except:
                        bkg_median = 0.0
                    
                    net_flux = raw_flux - (bkg_median * aperture.area)
                    
                    if net_flux > 1e-10:
                        fluxes.append(net_flux)
                    else:
                        fluxes.append(np.nan)
                
                # Verificar curva de crecimiento válida
                valid_fluxes = np.array([f for f in fluxes if not np.isnan(f)])
                if len(valid_fluxes) >= 4 and np.all(np.diff(valid_fluxes) > -1e-10):  # Permitir pequeñas disminuciones
                    # Normalizar por flujo en apertura más grande
                    total_flux = valid_fluxes[-1]
                    if total_flux > 0:
                        normalized_fluxes = valid_fluxes / total_flux
                        all_growth_curves.append(normalized_fluxes)
                        successful_stars += 1
                        
            except Exception as e:
                if debug:
                    logging.debug(f"Star {i} failed in aperture correction: {e}")
                continue
        
        logging.info(f"{filter_name}: Successful growth curves: {successful_stars}/{len(reference_positions)}")
        
        if successful_stars < 2:
            logging.warning(f"{filter_name}: Not enough valid growth curves, using fallback")
            base_corrections = {
                'F378': (-0.25, -0.12), 'F395': (-0.22, -0.10), 'F410': (-0.20, -0.09),
                'F430': (-0.18, -0.08), 'F515': (-0.15, -0.07), 'F660': (-0.12, -0.06), 
                'F861': (-0.10, -0.05)
            }
            default_2, default_3 = base_corrections.get(filter_name, (-0.15, -0.07))
            seeing_factor = seeing / 1.5
            return default_2 * seeing_factor, default_3 * seeing_factor, {}
        
        # Promediar curvas de crecimiento
        min_length = min(len(curve) for curve in all_growth_curves)
        truncated_curves = [curve[:min_length] for curve in all_growth_curves]
        mean_growth_curve = np.mean(truncated_curves, axis=0)
        std_growth_curve = np.std(truncated_curves, axis=0)
        
        # Usar las aperturas correspondientes
        used_radii = aperture_radii[:min_length]
        
        # Método simplificado: usar la diferencia entre aperturas pequeñas y grandes
        if len(mean_growth_curve) >= 3:
            flux_2 = mean_growth_curve[0]  # 2" diameter (1" radius)
            flux_3 = mean_growth_curve[1]  # 3" diameter (1.5" radius)  
            flux_ref = mean_growth_curve[-1]  # Largest aperture
            
            if flux_2 > 0 and flux_ref > 0:
                correction_2 = -2.5 * np.log10(flux_2 / flux_ref)
            else:
                correction_2 = -0.15
                
            if flux_3 > 0 and flux_ref > 0:
                correction_3 = -2.5 * np.log10(flux_3 / flux_ref)
            else:
                correction_3 = -0.08
        else:
            # Fallback final
            correction_2 = -0.15
            correction_3 = -0.08
        
        # Limitar correcciones a valores físicos
        correction_2 = max(-0.8, min(-0.01, correction_2))
        correction_3 = max(-0.5, min(-0.01, correction_3))
        
        diagnostics = {
            'successful_stars': successful_stars,
            'mean_growth_curve': mean_growth_curve,
            'std_growth_curve': std_growth_curve,
            'aperture_radii': used_radii,
            'seeing': seeing
        }
        
        logging.info(f"{filter_name}: Aperture corrections - 2\"={correction_2:.3f}, 3\"={correction_3:.3f}")
        return correction_2, correction_3, diagnostics
            
    except Exception as e:
        logging.error(f"{filter_name}: Aperture correction failed: {e}")
        # Fallback final
        return -0.15, -0.08, {}

def process_single_filter_corrected(args):
    """
    ✅ VERSIÓN CORREGIDA con mejor manejo de errores
    """
    try:
        (field_name, filter_name, valid_positions, valid_indices, 
         zeropoints, pixel_scale, debug, global_seeing) = args
        
        logging.info(f"Starting {filter_name} processing")
        
        def find_image_file(field_name, filter_name):
            for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
                path = os.path.join(field_name, ext)
                if os.path.exists(path):
                    return path
            return None

        # Cargar imagen
        image_path = find_image_file(field_name, filter_name)
        if not image_path:
            logging.warning(f"{filter_name}: Image not found")
            return None, filter_name
        
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data_original = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                return None, filter_name

        # Estimar seeing
        seeing = calculate_seeing_from_psf(header)
        
        # ✅ ANÁLISIS DE FONDO MEJORADO
        needs_unsharp, variability_ratio, _ = analyze_background_variability_improved(data_original)
        
        if needs_unsharp:
            data_processed, median_size, gauss_sigma = create_optimized_unsharp_mask(data_original)
            logging.info(f"✅ {filter_name}: Unsharp masking applied (variability: {variability_ratio:.3f})")
            
            if debug:
                save_unsharp_diagnostic(data_original, data_processed, field_name, filter_name)
        else:
            data_processed = data_original
            logging.info(f"✅ {filter_name}: No unsharp masking needed (variability: {variability_ratio:.3f})")

        # ✅ ENCONTRAR ESTRELLAS DE REFERENCIA CORREGIDO
        reference_stars = find_high_snr_reference_stars_fixed(valid_positions, data_processed)
        
        # ✅ CORRECCIÓN DE APERTURA MEJORADA
        aperture_correction_2, aperture_correction_3, ap_diagnostics = calculate_physical_aperture_correction_improved(
            reference_stars, data_processed, filter_name, pixel_scale, seeing, debug)

        # ✅ FOTOMETRÍA ROBUSTA
        def perform_robust_photometry(data, positions, apertures, annulus):
            """Fotometría robusta con manejo de errores"""
            try:
                phot_table = aperture_photometry(data, apertures)
                raw_flux = phot_table['aperture_sum'].data
                
                bkg_median_per_source = np.zeros(len(positions))
                
                for i, pos in enumerate(positions):
                    try:
                        mask = annulus.to_mask(method='center')[i]
                        annulus_data = mask.multiply(data)
                        annulus_data_1d = annulus_data[mask.data > 0]
                        
                        if len(annulus_data_1d) > 5:
                            _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d, sigma=2.5, maxiters=3)
                        else:
                            bkg_median = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                        bkg_median_per_source[i] = bkg_median
                    except:
                        bkg_median_per_source[i] = 0.0
                
                net_flux = raw_flux - (bkg_median_per_source * apertures.area)
                
                # Estimación simple de errores
                net_flux_err = np.sqrt(np.abs(net_flux) + (apertures.area * 0.1**2))
                
                return net_flux, net_flux_err, bkg_median_per_source
                
            except Exception as e:
                logging.error(f"Error in robust photometry: {e}")
                n = len(positions)
                return np.zeros(n), np.full(n, 99.0), np.zeros(n)

        # Realizar fotometría
        aperture_diams = [2.0, 3.0]
        results = {'indices': valid_indices}
        
        for aperture_diam in aperture_diams:
            aperture_radius = (aperture_diam / 2) / pixel_scale
            annulus_inner = (3.0 / 2) / pixel_scale
            annulus_outer = (4.0 / 2) / pixel_scale
            
            apertures = CircularAperture(valid_positions, r=aperture_radius)
            annulus = CircularAnnulus(valid_positions, r_in=annulus_inner, r_out=annulus_outer)
            
            net_flux, net_flux_err, background_levels = perform_robust_photometry(
                data_processed, valid_positions, apertures, annulus)
            
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            
            zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
            min_flux = 1e-10
            valid_mask = (net_flux > min_flux) & (net_flux_err > 0) & np.isfinite(net_flux)
            
            mag_inst = np.where(valid_mask, -2.5 * np.log10(net_flux), 99.0)
            
            # Aplicar corrección de apertura
            if aperture_diam == 2.0:
                aperture_correction = aperture_correction_2
            else:
                aperture_correction = aperture_correction_3
                
            mag = np.where(valid_mask, mag_inst + zero_point + aperture_correction, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
            
            # Filtros de calidad
            reasonable = (mag >= 10.0) & (mag <= 30.0) & (mag_err <= 2.0) & (snr > 0.1)
            final_mask = valid_mask & reasonable
            
            mag = np.where(final_mask, mag, 99.0)
            mag_err = np.where(final_mask, mag_err, 99.0)
            snr = np.where(final_mask, snr, 0.0)
            
            prefix = f"{filter_name}_{aperture_diam:.0f}"
            results[f'FLUX_{prefix}'] = net_flux
            results[f'FLUXERR_{prefix}'] = net_flux_err
            results[f'MAG_{prefix}'] = mag
            results[f'MAGERR_{prefix}'] = mag_err
            results[f'SNR_{prefix}'] = snr
            results[f'AP_CORR_{prefix}'] = np.full(len(net_flux), aperture_correction)
            results[f'QUALITY_{prefix}'] = final_mask.astype(int)
        
        valid_count = np.sum(final_mask)
        logging.info(f"✅ {filter_name}: Completed - {valid_count} valid sources, "
                   f"AP corr: 2\"={aperture_correction_2:.3f}, 3\"={aperture_correction_3:.3f}")
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {filter_name}: Processing failed - {e}")
        traceback.print_exc()
        return None, filter_name

# [La clase SPLUSGCPhotometryImproved se mantiene igual hasta el método process_field_improved]
# Solo cambiamos la llamada a process_single_filter_corrected

class SPLUSGCPhotometryImproved:
    def __init__(self, catalog_path, zeropoints_file, debug=False, n_workers=None):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Cargar zero points 
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        for _, row in self.zeropoints_df.iterrows():
            field = row['field']
            self.zeropoints[field] = {}
            for filt in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                self.zeropoints[field][filt] = row[filt]
        
        # Cargar catálogo
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        # CONFIGURACIÓN MEJORADA
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55
        self.debug = debug
        self.n_workers = n_workers or 1
        
        # Mapeo de columnas
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        logging.info(f"Using IMPROVED method with physical aperture corrections")

    def get_field_center_from_header(self, field_name, filter_name='F660'):
        """Obtener centro del campo desde header"""
        files = [
            os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz"),
            os.path.join(field_name, f"{field_name}_{filter_name}.fits")
        ]
        for f in files:
            if os.path.exists(f):
                try:
                    with fits.open(f) as hdul:
                        for hdu in hdul:
                            if hdu.data is not None:
                                ra = hdu.header.get('CRVAL1')
                                dec = hdu.header.get('CRVAL2')
                                if ra and dec:
                                    return float(ra), float(dec)
                except:
                    continue
        return None, None

    def is_source_in_field(self, ra, dec, field_ra, field_dec, radius=0.84):
        """Verificar si fuente está dentro del campo"""
        if field_ra is None or field_dec is None:
            return False
        c1 = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        c2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        return c1.separation(c2).degree <= radius

    def process_field_improved(self, field_name):
        """Procesar campo completo - VERSIÓN MEJORADA"""
        logging.info(f"🎯 Processing field {field_name} (IMPROVED METHOD - {self.n_workers} workers)")
        
        start_time = time.time()
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist")
            return None
        
        # Obtener centro del campo
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}")
            return None
        
        # Filtrar fuentes en el campo
        self.catalog[self.ra_col] = pd.to_numeric(self.catalog[self.ra_col], errors='coerce')
        self.catalog[self.dec_col] = pd.to_numeric(self.catalog[self.dec_col], errors='coerce')
        self.catalog = self.catalog.dropna(subset=[self.ra_col, self.dec_col])
        
        in_field_mask = [
            self.is_source_in_field(row[self.ra_col], row[self.dec_col], field_ra, field_dec)
            for _, row in self.catalog.iterrows()
        ]
        field_sources = self.catalog[in_field_mask].copy()
        logging.info(f"Found {len(field_sources)} GC sources in field {field_name}")
        
        if len(field_sources) == 0:
            return None
        
        # Obtener posiciones en píxeles
        first_filter_img = self.find_image_file(field_name, self.filters[0])
        if not first_filter_img:
            logging.error(f"Cannot find any image for field {field_name}")
            return None

        with fits.open(first_filter_img) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    header = hdu.header
                    wcs = WCS(header)
                    break
            else:
                return None

        # Convertir coordenadas a píxeles
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

        # Validar posiciones dentro de imagen
        margin = 50
        valid_mask = (
            (x >= margin) & (x < width - margin) & 
            (y >= margin) & (y < height - margin) &
            np.isfinite(x) & np.isfinite(y)
        )
        valid_positions = positions[valid_mask]
        valid_indices = field_sources.index[valid_mask].values

        if len(valid_positions) == 0:
            logging.warning(f"No valid positions in field {field_name}")
            return None

        logging.info(f"Valid positions for photometry: {len(valid_positions)}")

        # Obtener seeing global del campo
        global_seeing = calculate_seeing_from_psf(header)

        # ✅ PROCESAMIENTO MEJORADO
        results_df = field_sources.copy()
        successful_filters = 0
        
        # Preparar argumentos
        args_list = []
        for filt in self.filters:
            args = (
                field_name, 
                filt, 
                valid_positions, 
                valid_indices,
                self.zeropoints,
                self.pixel_scale,
                self.debug,
                global_seeing
            )
            args_list.append(args)
        
        # Procesamiento paralelo o serial
        if self.n_workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [executor.submit(process_single_filter_corrected, args) for args in args_list]
                    
                    for future in as_completed(futures):
                        try:
                            result, filter_name = future.result(timeout=600)  # Aumentado timeout
                            if result is not None:
                                self._integrate_results(results_df, result, filter_name)
                                successful_filters += 1
                                logging.info(f"✅ {filter_name}: Results integrated")
                        except Exception as e:
                            logging.error(f"❌ Error in future: {e}")
            except Exception as e:
                logging.error(f"Parallel failed, using serial: {e}")
                successful_filters = self._process_serial_improved(args_list, results_df)
        else:
            successful_filters = self._process_serial_improved(args_list, results_df)
        
        elapsed_time = time.time() - start_time
        logging.info(f"🎯 Field {field_name} completed: {successful_filters}/{len(self.filters)} filters "
                   f"in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            return results_df
        else:
            return None

    def _process_serial_improved(self, args_list, results_df):
        """Procesamiento serial mejorado"""
        successful_filters = 0
        for args in tqdm(args_list, desc="Processing filters"):
            result, filter_name = process_single_filter_corrected(args)
            if result is not None:
                self._integrate_results(results_df, result, filter_name)
                successful_filters += 1
        return successful_filters

    def _integrate_results(self, results_df, result, filter_name):
        """Integrar resultados"""
        temp_df = pd.DataFrame(result)
        temp_df.set_index('indices', inplace=True)
        
        for col in temp_df.columns:
            if col != 'indices':
                results_df.loc[temp_df.index, col] = temp_df[col].values

    def find_image_file(self, field_name, filter_name):
        """Buscar archivos de imagen"""
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None

def main():
    """Función principal"""
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_3arcsec.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    # Configuración
    test_mode = True
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    # Crear instancia mejorada
    photometry = SPLUSGCPhotometryImproved(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True,
        n_workers=4  # Reducido para mayor estabilidad
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_improved(field)
        if results is not None and len(results) > 0:
            all_results.append(results)
            
            output_file = f'{field}_gc_photometry_improved_v4.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results to {output_file}")
            
            # Estadísticas detalladas del campo
            logging.info(f"📊 Field {field} detailed statistics:")
            for filter_name in photometry.filters:
                for aperture in ['2', '3']:
                    mag_col = f'MAG_{filter_name}_{aperture}'
                    err_col = f'MAGERR_{filter_name}_{aperture}'
                    snr_col = f'SNR_{filter_name}_{aperture}'
                    ap_corr_col = f'AP_CORR_{filter_name}_{aperture}'
                    
                    if mag_col in results.columns:
                        valid_mask = results[mag_col] < 99.0
                        valid_count = np.sum(valid_mask)
                        
                        if valid_count > 0:
                            valid_mags = results[mag_col][valid_mask]
                            valid_errors = results[err_col][valid_mask]
                            valid_snr = results[snr_col][valid_mask]
                            ap_corr = results[ap_corr_col].iloc[0] if ap_corr_col in results.columns else 0.0
                            
                            median_mag = np.median(valid_mags)
                            median_error = np.median(valid_errors)
                            median_snr = np.median(valid_snr)
                            
                            logging.info(f"  {filter_name} {aperture}\": {valid_count} valid, "
                                       f"mag={median_mag:.2f}±{median_error:.2f}, "
                                       f"SNR={median_snr:.1f}, AP corr={ap_corr:.3f}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_improved_v4.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        # Análisis de calidad final
        logging.info("\n📊 ANÁLISIS DE CALIDAD FINAL (IMPROVED METHOD v4):")
        for filter_name in photometry.filters:
            for aperture in ['2', '3']:
                mag_col = f'MAG_{filter_name}_{aperture}'
                err_col = f'MAGERR_{filter_name}_{aperture}'
                
                if mag_col in final_results.columns:
                    valid_mask = final_results[mag_col] < 99.0
                    valid_count = np.sum(valid_mask)
                    
                    if valid_count > 0:
                        valid_mags = final_results[mag_col][valid_mask]
                        valid_errors = final_results[err_col][valid_mask]
                        
                        completeness = valid_count / len(final_results) * 100
                        median_mag = np.median(valid_mags)
                        median_error = np.median(valid_errors)
                        low_error_ratio = np.sum(valid_errors < 0.1) / valid_count * 100
                        
                        logging.info(f"  {filter_name} {aperture}\": {valid_count} ({completeness:.1f}%) "
                                   f"median={median_mag:.2f}±{median_error:.3f}, "
                                   f"errors<0.1: {low_error_ratio:.1f}%")

if __name__ == "__main__":
    # Configuración segura para multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
