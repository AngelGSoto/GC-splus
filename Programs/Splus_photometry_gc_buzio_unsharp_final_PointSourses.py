#!/usr/bin/env python3
"""
Splus_photometry_gc_scientific_v17_PointSource.py
VERSIÓN CORREGIDA: Resta de galaxia para fotometría precisa de cúmulos globulares
- Apertura principal: 2 arcsec (mejor coherencia con Taylor)
- Meseta realista: 4-6 arcsec (no 8 arcsec)
- Parámetros optimizados basados en análisis comparativo
- RESTA DE GALAXIA: Aplicada a todos los campos para fotometría de GCs
- DIAGNÓSTICO: Imágenes guardadas para F660 y F861 en campos seleccionados
- CORRECCIONES APLICADAS: Cálculo científico preciso de fondo y errores
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
import seaborn as sns
from scipy.spatial import KDTree
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage
import concurrent.futures  # AGREGADO para paralelización

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_scientific_photometry_v17_optimized_PointSource.log'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings('ignore')

class SPLUSPhotometryConfig:
    """Configuración OPTIMIZADA basada en resultados empíricos"""
    def __init__(self):
        self.pixel_scale = 0.55
        # Apertura PRINCIPAL de 2" basada en análisis comparativo
        self.aperture_diams = [2.0, 3.0]  
        self.reference_aperture_diam = 6.0  # Reducido de 8.0 a 6.0 (más realista)
        self.annulus_inner = 4.0  # Más conservador
        self.annulus_outer = 6.0
        self.margin = 50
        self.min_reference_stars = 5
        self.quality_snr_threshold = 5
        self.max_aperture_correction = 1.0
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        
        # PARÁMETROS PARA RESTA DE GALAXIA (Buzzeo et al. 2022)
        self.median_box_size = 25  # 25x25 pixels median box
        self.gaussian_sigma = 5    # Gaussian smoothing with σ=5 pixels
        
        # CONFIGURACIÓN DIAGNÓSTICO
        self.save_diagnostic_images = True
        self.diagnostic_dir = "galaxy_subtraction_diagnostics"
        self.diagnostic_fields = ['CenA01', 'CenA11', 'CenA12']  # Campos para diagnóstico
        self.diagnostic_filters = ['F660', 'F861']  # Filtros para diagnóstico
        
        # PARÁMETROS OPTIMIZADOS: crecimiento hasta 6" (no 10")
        self.growth_curve_radii = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        self.plateau_threshold = 0.01  # Más estricto

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
    
    logging.info(f"Header info: seeing={info['seeing_fwhm']:.3f}\", "
                 f"pixel_scale={info['pixel_scale']:.3f}\"/pix, "
                 f"exptime={info['exptime']:.1f}s, airmass={info['airmass']:.3f}")
    
    return info

def subtract_galaxy_background(data, median_box_size=25, gaussian_sigma=5):
    """
    RESTA EL FONDO GALÁCTICO para aislar cúmulos globulares y estrellas
    Método CORRECTO: data - smoothed (como en Buzzeo et al. 2022)
    """
    try:
        # Paso 1: Aplicar filtro de mediana para eliminar fuentes puntuales
        median_filtered = median_filter(data, size=median_box_size)
        
        # Paso 2: Aplicar suavizado gaussiano para obtener el fondo galáctico
        galaxy_background = gaussian_filter(median_filtered, sigma=gaussian_sigma)
        
        # Paso 3: CORRECTO - Restar el fondo galáctico de la imagen original
        residual_image = data - galaxy_background
        
        logging.info(f"✅ GALAXY SUBTRACTION: "
                   f"median_box={median_box_size}, gaussian_sigma={gaussian_sigma}")
        
        return residual_image, galaxy_background, median_filtered
        
    except Exception as e:
        logging.error(f"Error in galaxy subtraction: {e}")
        return data, np.zeros_like(data), data

def save_diagnostic_images(original_data, residual_data, galaxy_background, header, field_name, filter_name):
    """
    Guarda imágenes de diagnóstico: original, residual y fondo galáctico
    """
    try:
        # Verificar si debemos guardar diagnóstico para este campo y filtro
        if (not config.save_diagnostic_images or 
            field_name not in config.diagnostic_fields or
            filter_name not in config.diagnostic_filters):
            return
        
        # Crear directorio de diagnóstico
        diagnostic_dir = Path(config.diagnostic_dir) / field_name
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Guardar imagen ORIGINAL
        original_path = diagnostic_dir / f"{field_name}_{filter_name}_original.fits"
        fits.writeto(original_path, original_data, header, overwrite=True)
        
        # 2. Guardar imagen RESIDUAL (después de restar galaxia)
        residual_path = diagnostic_dir / f"{field_name}_{filter_name}_residual.fits"
        
        # Actualizar header con información de procesamiento
        new_header = header.copy()
        new_header['HISTORY'] = f'Processed by SPLUS pipeline v17 - Galaxy Subtraction'
        new_header['HISTORY'] = f'Median box: {config.median_box_size}'
        new_header['HISTORY'] = f'Gaussian sigma: {config.gaussian_sigma}'
        new_header['PROCTYPE'] = 'residual'
        
        fits.writeto(residual_path, residual_data, new_header, overwrite=True)
        
        # 3. Guardar fondo galáctico
        background_path = diagnostic_dir / f"{field_name}_{filter_name}_background.fits"
        bg_header = header.copy()
        bg_header['PROCTYPE'] = 'galaxy_background'
        fits.writeto(background_path, galaxy_background, bg_header, overwrite=True)
        
        # 4. Crear plot comparativo
        create_comparison_plot(original_data, residual_data, galaxy_background, field_name, filter_name, diagnostic_dir)
        
        logging.info(f"📁 GALAXY SUBTRACTION DIAGNOSTIC: Images saved for {field_name} {filter_name}")
        logging.info(f"   Original: {original_path}")
        logging.info(f"   Residual: {residual_path}")
        logging.info(f"   Background: {background_path}")
        
    except Exception as e:
        logging.error(f"Error saving galaxy subtraction diagnostic images: {e}")

def create_comparison_plot(original, residual, background, field_name, filter_name, output_dir):
    """
    Crea un plot comparativo entre original, residual y fondo
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        fig.suptitle(f'Galaxy Subtraction - {field_name} {filter_name}\n'
                    f'Median box: {config.median_box_size}, Gaussian σ: {config.gaussian_sigma}', 
                    fontsize=12, fontweight='bold')
        
        # Usar percentiles para escalado consistente
        vmin, vmax = np.percentile(original, [5, 95])
        
        # 1. Imagen original
        im1 = axes[0, 0].imshow(original, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title('Original Image')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # 2. Fondo galáctico
        im2 = axes[0, 1].imshow(background, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title('Galaxy Background')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # 3. Imagen residual (original - fondo)
        im3 = axes[0, 2].imshow(residual, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title('Residual (Original - Background)')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # 4. Histograma de valores originales
        axes[1, 0].hist(original.flatten(), bins=100, alpha=0.7, color='blue', label='Original')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Original Image Histogram')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Histograma de valores residuales
        axes[1, 1].hist(residual.flatten(), bins=100, alpha=0.7, color='red', label='Residual')
        axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Residual Image Histogram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Histograma de fondos
        axes[1, 2].hist(background.flatten(), bins=100, alpha=0.7, color='green', label='Background')
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Background Histogram')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / f"{field_name}_{filter_name}_galaxy_subtraction_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 GALAXY SUBTRACTION DIAGNOSTIC: Comparison plot saved: {plot_path}")
        
    except Exception as e:
        logging.warning(f"Could not create galaxy subtraction comparison plot: {e}")

def detect_reference_stars_daofind_corrected(data, error_map, header, nstars=30):
    """
    Detección de estrellas usando DAOFind - VERSIÓN CORREGIDA
    """
    try:
        header_info = extract_header_information(header)
        seeing_fwhm = header_info['seeing_fwhm']
        pixel_scale = header_info['pixel_scale']
        
        fwhm_pixels = seeing_fwhm / pixel_scale
        
        # Para S-PLUS, ajustar el análisis
        data_positive = data - np.min(data) + 1.0
        
        # Estimar el fondo
        mean, median, std = sigma_clipped_stats(data_positive, sigma=3.0)
        
        # Umbral adaptativo
        threshold = 5.0 * std
        
        # Encontrar estrellas con DAOFind - SIN parámetros problemáticos
        daofind = DAOStarFinder(fwhm=fwhm_pixels, 
                               threshold=threshold,
                               sharplo=0.2, sharphi=1.0,
                               roundlo=-1.0, roundhi=1.0)  # Parámetros corregidos
        
        sources = daofind(data_positive)
        
        if sources is None:
            logging.warning("DAOFind no detectó ninguna estrella")
            return np.array([])
        
        # Verificar columnas disponibles
        available_columns = sources.colnames
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        fluxes = sources['flux']
        
        # Filtrar por calidad si las columnas están disponibles
        if 'sharpness' in available_columns:
            sharpness = sources['sharpness']
            good_sharpness = (sharpness > 0.2) & (sharpness < 1.0)
        else:
            good_sharpness = np.ones_like(fluxes, dtype=bool)
            
        if 'roundness1' in available_columns:
            roundness = sources['roundness1']
            good_roundness = np.abs(roundness) < 1.0
        elif 'roundness' in available_columns:
            roundness = sources['roundness']
            good_roundness = np.abs(roundness) < 1.0
        else:
            good_roundness = np.ones_like(fluxes, dtype=bool)
        
        # Filtrar por SNR
        snr_values = fluxes / std
        good_snr = snr_values > config.quality_snr_threshold
        
        # Combinar criterios de calidad
        quality_mask = good_sharpness & good_roundness & good_snr
        
        if np.sum(quality_mask) == 0:
            logging.warning(f"No stars passed quality filters")
            # Usar solo criterio SNR si no hay otras columnas
            quality_mask = good_snr
        
        if np.sum(quality_mask) == 0:
            logging.warning(f"No stars with SNR > {config.quality_snr_threshold}")
            return np.array([])
        
        positions = positions[quality_mask]
        fluxes = fluxes[quality_mask]
        snr_values = snr_values[quality_mask]
        
        # Ordenar por flujo y tomar las mejores
        sorted_indices = np.argsort(-fluxes)
        n_to_keep = min(nstars, len(positions))
        best_positions = positions[sorted_indices[:n_to_keep]]
        
        logging.info(f"DAOFind found {len(best_positions)} quality stars "
                   f"(SNR range: {np.min(snr_values):.1f}-{np.max(snr_values):.1f})")
        
        return best_positions
        
    except Exception as e:
        logging.error(f"Error in corrected DAOFind detection: {e}")
        # Fallback: búsqueda simple por umbral
        try:
            header_info = extract_header_information(header)
            data_positive = data - np.min(data) + 1.0
            mean, median, std = sigma_clipped_stats(data_positive, sigma=3.0)
            
            # Encontrar píxeles brillantes
            threshold = 5.0 * std
            bright_pixels = np.where(data_positive > threshold)
            
            if len(bright_pixels[0]) == 0:
                return np.array([])
                
            # Agrupar píxeles brillantes
            labeled, num_features = ndimage.label(data_positive > threshold)
            centers = ndimage.center_of_mass(data_positive, labeled, range(1, num_features+1))
            
            positions = np.array(centers)[:, ::-1]  # Convertir a (x, y)
            
            # Tomar las posiciones más brillantes
            fluxes = []
            for pos in positions:
                y, x = int(pos[1]), int(pos[0])
                if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                    fluxes.append(data_positive[y, x])
                else:
                    fluxes.append(0)
            
            fluxes = np.array(fluxes)
            if len(fluxes) > 0:
                sorted_indices = np.argsort(-fluxes)
                n_to_keep = min(nstars, len(positions))
                best_positions = positions[sorted_indices[:n_to_keep]]
                logging.info(f"Fallback found {len(best_positions)} bright sources")
                return best_positions
            else:
                return np.array([])
                
        except Exception as e2:
            logging.error(f"Fallback detection also failed: {e2}")
            return np.array([])

def analyze_growth_curves_realistic(positions, data, error_map, header, output_dir="growth_curve"):
    """
    Análisis de curvas de crecimiento OPTIMIZADO - MÁXIMO 6 ARCSEC
    Basado en observación empírica de meseta entre 4-6 arcsec
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        header_info = extract_header_information(header)
        pixel_scale = header_info['pixel_scale']
        seeing_fwhm = header_info['seeing_fwhm']
        
        growth_radii_pixels = config.growth_curve_radii / 2.0 / pixel_scale
        
        growth_data = []
        plateau_radii = []
        
        for i, pos in enumerate(positions):
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
                    
                    # Usar valor absoluto para S-PLUS
                    flux_abs = abs(flux)
                    
                    if flux_abs > 0 and np.isfinite(flux_abs):
                        fluxes.append(flux_abs)
                        valid_radii.append(radius * pixel_scale * 2)  # Diámetro en arcsec
                
                if len(fluxes) < 5:
                    continue
                
                fluxes = np.array(fluxes)
                valid_radii = np.array(valid_radii)
                
                # Normalizar flujos
                max_flux = np.max(fluxes)
                if max_flux <= 0:
                    continue
                
                normalized_fluxes = fluxes / max_flux
                
                # ESTRATEGIA MEJORADA: buscar meseta en rango 4-6 arcsec
                if len(valid_radii) > 4:
                    f = interp1d(valid_radii, normalized_fluxes, kind='quadratic', 
                                fill_value='extrapolate')
                    radii_dense = np.linspace(valid_radii[0], valid_radii[-1], 100)
                    fluxes_dense = f(radii_dense)
                    
                    # Calcular derivada suavizada
                    derivatives = np.abs(np.diff(fluxes_dense) / np.diff(radii_dense))
                    
                    # Buscar estabilización en rango 4-6 arcsec
                    target_range_mask = (radii_dense[:-1] >= 4.0) & (radii_dense[:-1] <= 6.0)
                    derivatives_in_range = derivatives[target_range_mask]
                    radii_in_range = radii_dense[:-1][target_range_mask]
                    
                    if len(derivatives_in_range) > 0:
                        # Encontrar donde la derivada cae por debajo del umbral en el rango objetivo
                        low_derivative_mask = derivatives_in_range < config.plateau_threshold
                        
                        if np.any(low_derivative_mask):
                            # Tomar el primer radio donde la derivada es baja
                            stable_indices = np.where(low_derivative_mask)[0]
                            plateau_radius = radii_in_range[stable_indices[0]]
                        else:
                            # Si no se encuentra, usar 6" como máximo
                            plateau_radius = 6.0
                    else:
                        plateau_radius = 6.0
                else:
                    plateau_radius = min(valid_radii[-1], 6.0)
                
                # Validación física - LÍMITE MÁXIMO 6"
                plateau_radius = min(plateau_radius, 6.0)
                
                if plateau_radius > seeing_fwhm * 1.2:  # Debe ser mayor que el seeing
                    growth_data.append({
                        'position': pos,
                        'radii': valid_radii,
                        'fluxes': fluxes,
                        'normalized_fluxes': normalized_fluxes,
                        'plateau_radius': plateau_radius
                    })
                    plateau_radii.append(plateau_radius)
                    
            except Exception as e:
                continue
        
        if not growth_data:
            logging.warning("No valid growth curves found")
            return min(6.0, seeing_fwhm * 2.5), {}  # Límite 6"
        
        # Estadísticas robustas
        plateau_radii = np.array(plateau_radii)
        median_plateau = np.median(plateau_radii)
        mad_plateau = mad_std(plateau_radii)
        
        # Filtrar valores atípicos
        valid_plateaus = plateau_radii[(plateau_radii >= median_plateau - 2*mad_plateau) & 
                                     (plateau_radii <= median_plateau + 2*mad_plateau)]
        
        if len(valid_plateaus) > 0:
            final_plateau = np.median(valid_plateaus)
        else:
            final_plateau = median_plateau
        
        # Límites prácticos - MÁXIMO 6" (optimizado basado en resultados)
        recommended_aperture = min(max(final_plateau, seeing_fwhm * 1.5), 6.0)
        
        # Diagnóstico visual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Curvas de crecimiento
        for i, gd in enumerate(growth_data[:15]):
            color = plt.cm.viridis(i / min(15, len(growth_data)))
            ax1.plot(gd['radii'], gd['normalized_fluxes'], color=color, alpha=0.6, linewidth=1)
        
        ax1.axvline(recommended_aperture, color='red', linestyle='--', linewidth=2,
                   label=f'Recommended: {recommended_aperture:.1f}"')
        ax1.axvline(seeing_fwhm, color='orange', linestyle=':', linewidth=2,
                   label=f'Seeing: {seeing_fwhm:.1f}"')
        ax1.axvspan(4.0, 6.0, alpha=0.2, color='green', label='Target range 4-6"')
        ax1.set_xlabel('Aperture Diameter (arcsec)')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title(f'Realistic Growth Curves - {header_info["field"]} {header_info["filter"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma
        ax2.hist(plateau_radii, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(recommended_aperture, color='red', linestyle='--', linewidth=2,
                   label=f'Recommended: {recommended_aperture:.1f}"')
        ax2.axvspan(4.0, 6.0, alpha=0.2, color='green', label='Target range')
        ax2.set_xlabel('Plateau Radius (arcsec)')
        ax2.set_ylabel('Count')
        ax2.set_title('Plateau Radius Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        field_name = header_info['field']
        filter_name = header_info['filter']
        plot_path = os.path.join(output_dir, f'{field_name}_{filter_name}_growth_analysis_realistic.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        diagnostics = {
            'median_plateau_radius': final_plateau,
            'n_sources_analyzed': len(growth_data),
            'seeing_fwhm': seeing_fwhm,
            'recommended_aperture': recommended_aperture,
            'method': 'realistic_4-6arcsec_range'
        }
        
        logging.info(f"REALISTIC growth curve: median_plateau={final_plateau:.1f}\", "
                   f"recommended={recommended_aperture:.1f}\", "
                   f"based on {len(growth_data)} sources (max 6\")")
        
        return recommended_aperture, diagnostics
        
    except Exception as e:
        logging.error(f"Realistic growth curve analysis failed: {e}")
        header_info = extract_header_information(header)
        seeing_fwhm = header_info['seeing_fwhm']
        return min(6.0, seeing_fwhm * 2.5), {}

def calculate_aperture_correction_robust(reference_positions, data, header):
    """
    Cálculo ROBUSTO de corrección de apertura usando 6" como referencia
    """
    try:
        if len(reference_positions) < 3:
            seeing = header.get('FWHMMEAN', 1.8)
            default_corr = min(0.5, seeing * 0.2)
            logging.info(f"Not enough reference stars ({len(reference_positions)}), using default correction: {default_corr:.3f}")
            return default_corr, default_corr * 0.8, {}
        
        header_info = extract_header_information(header)
        pixel_scale = header_info['pixel_scale']
        
        radius_2 = 2.0 / 2.0 / pixel_scale
        radius_3 = 3.0 / 2.0 / pixel_scale
        radius_6 = 6.0 / 2.0 / pixel_scale  # REFERENCIA OPTIMIZADA: 6" no 8"
        
        corrections_2 = []
        corrections_3 = []
        
        for pos in reference_positions:
            try:
                # Verificar bordes con radio de 6"
                if (pos[0] < radius_6 or pos[0] >= data.shape[1] - radius_6 or
                    pos[1] < radius_6 or pos[1] >= data.shape[0] - radius_6):
                    continue
                
                # Medir flujos
                aperture_2 = CircularAperture([pos], r=radius_2)
                aperture_3 = CircularAperture([pos], r=radius_3)
                aperture_6 = CircularAperture([pos], r=radius_6)
                
                phot_2 = aperture_photometry(data, aperture_2)
                phot_3 = aperture_photometry(data, aperture_3)
                phot_6 = aperture_photometry(data, aperture_6)
                
                flux_2 = phot_2['aperture_sum'].data[0]
                flux_3 = phot_3['aperture_sum'].data[0]
                flux_6 = phot_6['aperture_sum'].data[0]
                
                # Usar valores absolutos para S-PLUS
                flux_2_abs, flux_3_abs, flux_6_abs = abs(flux_2), abs(flux_3), abs(flux_6)
                
                if (flux_2_abs > 0 and flux_3_abs > 0 and flux_6_abs > 0 and
                    flux_6_abs >= flux_3_abs and flux_6_abs >= flux_2_abs):
                    
                    corr_2 = -2.5 * np.log10(flux_2_abs / flux_6_abs)
                    corr_3 = -2.5 * np.log10(flux_3_abs / flux_6_abs)
                    
                    # Validaciones físicas más estrictas
                    if (0 < corr_2 < 1.0 and 0 < corr_3 < 1.0 and 
                        corr_3 < corr_2 and abs(corr_2 - corr_3) > 0.1):
                        corrections_2.append(corr_2)
                        corrections_3.append(corr_3)
                        
            except Exception as e:
                continue
        
        if len(corrections_2) < 3:
            seeing = header_info['seeing_fwhm']
            default_corr = min(0.5, seeing * 0.2)
            logging.warning(f"Not enough valid corrections ({len(corrections_2)}), using default: {default_corr:.3f}")
            return default_corr, default_corr * 0.8, {}
        
        # Usar mediana robusta
        median_corr_2 = np.median(corrections_2)
        median_corr_3 = np.median(corrections_3)
        std_corr_2 = np.std(corrections_2)
        std_corr_3 = np.std(corrections_3)
        
        # Filtrar outliers basado en MAD
        mad_2 = mad_std(corrections_2)
        mad_3 = mad_std(corrections_3)
        
        filtered_2 = [c for c in corrections_2 if abs(c - median_corr_2) < 2 * mad_2]
        filtered_3 = [c for c in corrections_3 if abs(c - median_corr_3) < 2 * mad_3]
        
        if len(filtered_2) > 0 and len(filtered_3) > 0:
            final_corr_2 = np.median(filtered_2)
            final_corr_3 = np.median(filtered_3)
        else:
            final_corr_2 = median_corr_2
            final_corr_3 = median_corr_3
        
        diagnostics = {
            'n_stars': len(corrections_2),
            'median_correction_2': final_corr_2,
            'median_correction_3': final_corr_3,
            'std_correction_2': std_corr_2,
            'std_correction_3': std_corr_3,
            'n_filtered_2': len(filtered_2),
            'n_filtered_3': len(filtered_3),
            'reference_aperture': 6.0  # Especificar que usamos 6"
        }
        
        logging.info(f"OPTIMIZED aperture correction - 2\": {final_corr_2:.3f} ± {std_corr_2:.3f}, "
                   f"3\": {final_corr_3:.3f} ± {std_corr_3:.3f} "
                   f"({len(corrections_2)} stars, {len(filtered_2)} after filtering, ref=6\")")
        
        return final_corr_2, final_corr_3, diagnostics
        
    except Exception as e:
        logging.error(f"Robust aperture correction failed: {e}")
        seeing = header.get('FWHMMEAN', 1.8)
        default_corr = min(0.5, seeing * 0.2)
        return default_corr, default_corr * 0.8, {}

def validate_aperture_coherence(photometry_results, taylor_catalog, aperture_diam=2.0):
    """
    Valida la coherencia entre fotometría S-PLUS y Taylor
    Basado en resultados empíricos que muestran mejor coherencia con 2"
    """
    try:
        coherence_metrics = {}
        
        # Mapeo de filtros S-PLUS a Taylor
        filter_mapping = {
            'F410': 'gmag',
            'F430': 'gmag', 
            'F515': 'gmag',
            'F660': 'rmag',
            'F861': 'imag',
            'F378': 'umag',
            'F395': 'umag'
        }
        
        for splus_filter, taylor_filter in filter_mapping.items():
            mag_col = f'MAG_{splus_filter}_{aperture_diam:.0f}'
            
            if mag_col in photometry_results.columns and taylor_filter in taylor_catalog.columns:
                # Combinar datos
                merged_data = pd.merge(
                    photometry_results[['recno', mag_col]], 
                    taylor_catalog[['recno', taylor_filter]],
                    on='recno', 
                    how='inner'
                )
                
                if len(merged_data) > 10:  # Mínimo para análisis estadístico
                    valid_mask = (merged_data[mag_col] < 90) & (merged_data[taylor_filter] < 90)
                    valid_data = merged_data[valid_mask]
                    
                    if len(valid_data) > 10:
                        differences = valid_data[mag_col] - valid_data[taylor_filter]
                        
                        coherence_metrics[splus_filter] = {
                            'taylor_filter': taylor_filter,
                            'n_sources': len(valid_data),
                            'mean_diff': np.mean(differences),
                            'median_diff': np.median(differences),
                            'std_diff': np.std(differences),
                            'mad_diff': mad_std(differences)
                        }
        
        # Análisis general
        if coherence_metrics:
            all_medians = [metrics['median_diff'] for metrics in coherence_metrics.values()]
            all_mads = [metrics['mad_diff'] for metrics in coherence_metrics.values()]
            
            overall_metrics = {
                'mean_median_diff': np.mean(all_medians),
                'mean_mad': np.mean(all_mads),
                'aperture_used': aperture_diam,
                'n_filters': len(coherence_metrics)
            }
            
            logging.info(f"APERTURE COHERENCE ANALYSIS - {aperture_diam}\": "
                       f"mean_Δ_median={overall_metrics['mean_median_diff']:.3f}, "
                       f"mean_MAD={overall_metrics['mean_mad']:.3f}")
            
            return coherence_metrics, overall_metrics
        
        return coherence_metrics, {}
        
    except Exception as e:
        logging.error(f"Aperture coherence validation failed: {e}")
        return {}, {}

def load_weight_map_splus(weight_path, data_shape):
    """Carga mapa de peso para S-PLUS"""
    try:
        with fits.open(weight_path) as whdul:
            for whdu in whdul:
                if whdu.data is not None:
                    weight_data = whdu.data.astype(float)
                    break
            else:
                return None
        
        if weight_data.shape != data_shape:
            return None
        
        valid_weight = (weight_data > 0) & np.isfinite(weight_data)
        if np.sum(valid_weight) / weight_data.size < 0.5:
            return None
        
        error_map = np.full_like(weight_data, np.median(np.abs(weight_data)))
        error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
        
        return error_map
        
    except Exception as e:
        logging.error(f"Error loading weight map: {e}")
        return None

def robust_sigma_clipped_stats(data, sigma=3.0, maxiters=5):
    """Estadísticas robustas para S-PLUS"""
    try:
        data_abs = np.abs(data)
        mean, median, std = sigma_clipped_stats(data_abs, sigma=sigma, maxiters=maxiters)
        return mean, median, std
    except:
        clean_data = data_abs[np.isfinite(data_abs)]
        if len(clean_data) > 0:
            return np.mean(clean_data), np.median(clean_data), np.std(clean_data)
        else:
            return 0.0, 0.0, 1.0

def process_single_filter_splus_optimized(args):
    """Procesamiento OPTIMIZADO para S-PLUS con resta de galaxia y correcciones científicas"""
    try:
        (field_name, filter_name, valid_positions, valid_indices, 
         zeropoints, debug) = args
        
        logging.info(f"🔬 {filter_name}: Starting SCIENTIFIC S-PLUS processing with corrections")
        
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
        
        # Cargar weight map
        weight_path = find_splus_file(field_name, filter_name, 'weight')
        error_map = None
        if weight_path:
            error_map = load_weight_map_splus(weight_path, data_original.shape)
        
        if error_map is None:
            data_abs = np.abs(data_original)
            median_val = np.median(data_abs)
            error_map = np.sqrt(data_abs + median_val * 0.1)
        
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
        
        # APLICAR RESTA DE GALAXIA A TODOS LOS CAMPOS
        data_residual, galaxy_background, _ = subtract_galaxy_background(
            data_original,
            median_box_size=config.median_box_size,
            gaussian_sigma=config.gaussian_sigma
        )
        
        logging.info(f"✅ {filter_name}: GALAXY SUBTRACTION applied to {field_name}")
        
        # GUARDAR IMÁGENES DE DIAGNÓSTICO para F660 y F861 en campos seleccionados
        save_diagnostic_images(data_original, data_residual, galaxy_background, header, field_name, filter_name)
        
        # Para DETECCIÓN y FOTOMETRÍA usar imagen RESIDUAL (después de restar galaxia)
        data_for_detection = data_residual
        data_for_photometry = data_residual
        
        logging.info(f"📊 {filter_name}: Using RESIDUAL data for GC detection and photometry")
        
        # Detección de estrellas CORREGIDA (usar data_for_detection residual)
        reference_stars = detect_reference_stars_daofind_corrected(data_for_detection, error_map, header)
        
        if len(reference_stars) < config.min_reference_stars:
            # Fallback: usar GCs brillantes para referencia
            fluxes = []
            test_radius = header_info['seeing_fwhm'] / header_info['pixel_scale']
            for pos in valid_positions[:20]:
                try:
                    if (pos[0] >= test_radius and pos[0] < data_for_detection.shape[1] - test_radius and
                        pos[1] >= test_radius and pos[1] < data_for_detection.shape[0] - test_radius):
                        aperture = CircularAperture([pos], r=test_radius)
                        phot = aperture_photometry(data_for_detection, aperture)
                        flux = abs(phot['aperture_sum'].data[0])
                        fluxes.append(flux)
                    else:
                        fluxes.append(0)
                except:
                    fluxes.append(0)
            
            if len(fluxes) > 0 and np.max(fluxes) > 0:
                bright_indices = np.argsort(-np.array(fluxes))[:min(10, len(fluxes))]
                reference_stars = valid_positions[:20][bright_indices]
                logging.info(f"Using {len(reference_stars)} bright sources as reference")
            else:
                logging.warning("No bright sources found for reference")
        
        # ANÁLISIS DE CRECIMIENTO OPTIMIZADO (4-6 arcsec) - Usar datos RESIDUALES
        if len(reference_stars) > 0:
            analysis_positions = reference_stars
        else:
            analysis_positions = valid_positions[:min(15, len(valid_positions))]
            
        recommended_aperture, growth_diagnostics = analyze_growth_curves_realistic(
            analysis_positions, data_for_photometry, error_map, header)
        
        # CORRECCIÓN DE APERTURA OPTIMIZADA (referencia 6") - Usar datos RESIDUALES
        aperture_correction_2, aperture_correction_3, ap_diagnostics = \
            calculate_aperture_correction_robust(reference_stars, data_for_photometry, header)
        
        # Fotometría de GCs - Usar datos RESIDUALES (data_for_photometry) con CORRECCIONES CIENTÍFICAS
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
            
            # FOTOMETRÍA CON DATOS RESIDUALES Y CORRECCIONES CIENTÍFICAS
            phot_table = aperture_photometry(data_for_photometry, apertures, error=error_map)
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = phot_table['aperture_sum_err'].data
            
            # CORRECCIÓN: Cálculo científico preciso del fondo y errores
            bkg_medians = np.zeros(len(filtered_positions))
            bkg_stds = np.zeros(len(filtered_positions))
            
            for i, pos in enumerate(filtered_positions):
                try:
                    mask = annulus.to_mask(method='center')[i]
                    annulus_data = mask.multiply(data_for_photometry)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    
                    if len(annulus_data_1d) > 5:
                        # CORRECCIÓN: Usar valores REALES, no absolutos
                        bkg_median = np.median(annulus_data_1d)
                        bkg_std = np.std(annulus_data_1d)
                    else:
                        bkg_median = 0.0
                        bkg_std = 0.0
                except:
                    bkg_median = 0.0
                    bkg_std = 0.0
                
                bkg_medians[i] = bkg_median
                bkg_stds[i] = bkg_std
            
            # CORRECCIÓN: Flujo neto sin valor absoluto
            net_flux = raw_flux - (bkg_medians * apertures.area)
            
            # CORRECCIÓN: Error más realista usando desviación estándar
            net_flux_err = np.sqrt(raw_flux_err**2 + (bkg_stds**2 * apertures.area))
            
            # Cálculo de magnitudes (solo para flujos positivos y válidos)
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            valid_flux = (net_flux > 1e-10) & (net_flux_err > 0) & np.isfinite(net_flux)
            
            mag_inst = np.where(valid_flux, -2.5 * np.log10(net_flux), 99.0)
            
            # Aplicar corrección OPTIMIZADA
            if aperture_diam == 2.0:
                aperture_correction = aperture_correction_2
            else:
                aperture_correction = aperture_correction_3
                
            mag = np.where(valid_flux, mag_inst + zero_point - aperture_correction, 99.0)
            mag_err = np.where(valid_flux, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
            
            # Crear arrays completos
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
        
        logging.info(f"✅ {filter_name}: SCIENTIFIC processing completed - "
                   f"{valid_measurements} valid measurements with corrected background estimation")
        
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {filter_name}: SCIENTIFIC PROCESSING FAILED: {e}")
        traceback.print_exc()
        return None, filter_name

class SPLUSGCScientificPhotometryOptimized:
    """Pipeline OPTIMIZADO basado en resultados empíricos"""
    
    def __init__(self, catalog_path, zeropoints_file, debug=False):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
            
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        
        required_columns = ['field'] + config.filters
        missing_columns = [col for col in required_columns if col not in self.zeropoints_df.columns]
        
        if missing_columns:
            available_columns = list(self.zeropoints_df.columns)
            logging.error(f"Zeropoints file missing columns: {missing_columns}")
            raise ValueError(f"Zeropoints file missing columns: {missing_columns}")
        
        for _, row in self.zeropoints_df.iterrows():
            field = row['field']
            self.zeropoints[field] = {}
            for filt in config.filters:
                self.zeropoints[field][filt] = float(row[filt])
        
        logging.info(f"✅ Loaded zeropoints for {len(self.zeropoints)} fields from {zeropoints_file}")
        
        sample_fields = list(self.zeropoints.keys())[:2]
        for field in sample_fields:
            logging.info(f"  {field}: " + ", ".join([f"{filt}={self.zeropoints[field][filt]:.3f}" 
                                                   for filt in config.filters[:3]]) + "...")
                
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
            
        logging.info("🎯 INITIALIZED SCIENTIFIC S-PLUS PHOTOMETRY PIPELINE v17")
        logging.info("   - APERTURE PRINCIPAL: 2 arcsec (mejor coherencia con Taylor)")
        logging.info("   - MESETA REALISTA: 4-6 arcsec (no 8 arcsec)")
        logging.info("   - CORRECCIÓN REFERENCIA: 6 arcsec")
        logging.info("   - GALAXY SUBTRACTION: Aplicada a todos los campos para fotometría de GCs")
        logging.info("   - CORRECCIONES CIENTÍFICAS: Background estimation sin np.abs()")
        logging.info("   - ERRORES REALISTAS: Usando desviación estándar del fondo")
        logging.info("   - DIAGNÓSTICO: Imágenes guardadas para F660/F861 en CenA01, CenA11, CenA12")
        logging.info("   - PARÁMETROS BASADOS EN ANÁLISIS EMPÍRICO")
    
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
    
    def process_field_optimized(self, field_name):
        """Procesamiento OPTIMIZADO basado en resultados empíricos"""
        logging.info(f"🎯 Processing field {field_name} with SCIENTIFIC CORRECTIONS")
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
        # ⚡ MODIFICACIÓN: Paralelización de filtros con ThreadPoolExecutor
        args_list = []
        for filt in self.filters:
            args = (
                field_name, 
                filt, 
                valid_positions, 
                valid_indices,
                self.zeropoints,
                self.debug
            )
            args_list.append(args)
        
        # Usar ThreadPoolExecutor para procesar filtros en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.filters)) as executor:
            future_to_filter = {executor.submit(process_single_filter_splus_optimized, arg): arg[1] for arg in args_list}
            for future in concurrent.futures.as_completed(future_to_filter):
                filt = future_to_filter[future]
                try:
                    result, filter_name = future.result()
                    if result is not None:
                        temp_df = pd.DataFrame(result)
                        temp_df.set_index('indices', inplace=True)
                        for col in temp_df.columns:
                            if col != 'indices':
                                results_df.loc[temp_df.index, col] = temp_df[col].values
                        successful_filters += 1
                        logging.info(f"✅ {filter_name}: SCIENTIFIC results integrated (parallel)")
                except Exception as e:
                    logging.error(f"❌ Error processing {filt}: {e}")
                    continue
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            results_df['PROCESSING_DATE'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results_df['PHOTOMETRY_METHOD'] = 'S-PLUS_SCIENTIFIC_v17_CORRECTED'
            results_df['PRIMARY_APERTURE'] = '2 arcsec'
            results_df['GALAXY_SUBTRACTION'] = 'APPLIED_FOR_PHOTOMETRY'
            results_df['BACKGROUND_ESTIMATION'] = 'SCIENTIFIC_CORRECTED'
            
            # Validación de coherencia
            try:
                coherence_metrics, overall_metrics = validate_aperture_coherence(
                    results_df, self.original_catalog, aperture_diam=2.0)
                
                if overall_metrics:
                    results_df['COHERENCE_MEDIAN_DIFF'] = overall_metrics['mean_median_diff']
                    results_df['COHERENCE_MAD'] = overall_metrics['mean_mad']
                    logging.info(f"📊 Field {field_name} coherence: Δ_median={overall_metrics['mean_median_diff']:.3f}, "
                               f"MAD={overall_metrics['mean_mad']:.3f}")
            except Exception as e:
                logging.warning(f"Coherence validation failed for {field_name}: {e}")
            
            elapsed_time = time.time() - start_time
            logging.info(f"🎯 SCIENTIFIC field {field_name} completed: "
                       f"{successful_filters}/{len(self.filters)} filters in {elapsed_time:.1f}s")
            return results_df
        else:
            return None

def main():
    """Función principal OPTIMIZADA"""
    logging.info("=" * 80)
    logging.info("🎯 S-PLUS GLOBULAR CLUSTER SCIENTIFIC PHOTOMETRY v17 CORRECTED")
    logging.info("   BASADO EN RESULTADOS EMPÍRICOS: 2 arcsec + 4-6 arcsec meseta")
    logging.info("   GALAXY SUBTRACTION: Aplicada a todos los campos para fotometría de GCs")
    logging.info("   CORRECCIONES CIENTÍFICAS: Background estimation sin np.abs()")
    logging.info("   ERRORES REALISTAS: Usando desviación estándar del fondo")
    logging.info("   FOTOMETRÍA: Realizada en imágenes residuales (original - fondo galáctico)")
    logging.info("   DIAGNÓSTICO: Imágenes guardadas para F660/F861 en CenA01, CenA11, CenA12")
    logging.info("=" * 80)
    
    catalog_path = '../TAP_1_J_MNRAS_3444_psc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_3arcsec.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    zp_df = pd.read_csv(zeropoints_file)
    logging.info(f"📊 Zeropoints file: {len(zp_df)} fields")
    
    # If you want run the test mode change it for True
    test_mode = False
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    photometry = SPLUSGCScientificPhotometryOptimized(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing S-PLUS fields"):
        results = photometry.process_field_optimized(field)
        if results is not None and len(results) > 0:
            all_results.append(results)
            output_file = f'{field}_gc_photometry_scientific_v17_PointSource.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} SCIENTIFIC results to {output_file}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        os.makedirs("Results", exist_ok=True)
        output_file = 'Results/all_fields_gc_photometry_scientific_v17_PointSource.csv'
        final_results.to_csv(output_file, index=False)
        
        logging.info("🎉 SCIENTIFIC S-PLUS PHOTOMETRY COMPLETED SUCCESSFULLY")
        logging.info("   ✅ APERTURE PRINCIPAL: 2 arcsec (mejor coherencia con Taylor)")
        logging.info("   ✅ MESETA REALISTA: 4-6 arcsec (basado en observación empírica)")
        logging.info("   ✅ CORRECCIÓN REFERENCIA: 6 arcsec (no 8 arcsec)")
        logging.info("   ✅ GALAXY SUBTRACTION: Aplicada a todos los campos")
        logging.info("   ✅ CORRECCIONES CIENTÍFICAS: Background estimation sin np.abs()")
        logging.info("   ✅ ERRORES REALISTAS: Usando desviación estándar del fondo")
        logging.info("   ✅ FOTOMETRÍA: Realizada en imágenes residuales (original - fondo)")
        logging.info("   ✅ DIAGNÓSTICO: Imágenes guardadas en carpeta 'galaxy_subtraction_diagnostics'")
        logging.info("   ✅ PARÁMETROS OPTIMIZADOS BASADOS EN ANÁLISIS COMPARATIVO")
        logging.info(f"   📊 Final catalog: {output_file}")

if __name__ == "__main__":
    main()
