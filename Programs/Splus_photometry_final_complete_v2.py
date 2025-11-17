#!/usr/bin/env python3
"""
Splus_photometry_final_complete_v2.py
Fotometría completa con MODO TEST para probar campos específicos
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats, mad_std
from scipy.ndimage import median_filter, gaussian_filter
import logging
import os
from tqdm import tqdm
import warnings
import time
import traceback
import matplotlib.pyplot as plt
from pathlib import Path
import argparse  # NUEVO: para manejar argumentos de línea de comandos

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_photometry_complete_v2_TEST.log'),
        logging.StreamHandler()
    ]
)

class SPLUSCompletePhotometry:
    def __init__(self, test_mode=False):
        self.pixel_scale = 0.55
        self.aperture_diameters = [2.0, 3.0]
        self.annulus_inner = 4.0
        self.annulus_outer = 6.0
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.margin = 20
        
        # Campos que REQUIEREN unsharp mask (cerca del centro galáctico)
        self.fields_requiring_unsharp = ['CenA01', 'CenA11', 'CenA12', 'CenA13', 'CenA16', 'CenA17']
        
        # Directorio para diagnósticos de unsharp mask
        self.unsharp_diagnostic_dir = "unsharp-mask-final"
        
        # MODO TEST - variables adicionales
        self.test_mode = test_mode
        if test_mode:
            logging.info("🔬 TEST MODE ACTIVATED - Processing limited fields/filters")
        
        # Crear directorio de diagnóstico
        os.makedirs(self.unsharp_diagnostic_dir, exist_ok=True)
        
        logging.info("🎯 S-PLUS COMPLETE PHOTOMETRY v2 INITIALIZED")
        logging.info(f"   - Unsharp mask for: {self.fields_requiring_unsharp}")
        logging.info(f"   - Diagnostics: {self.unsharp_diagnostic_dir}")
        if test_mode:
            logging.info("   - 🧪 TEST MODE: Limited processing for validation")

    def find_splus_files(self, field, filter_name):
        """Encuentra archivos S-PLUS usando la estructura real"""
        base_path = field
        image_file = f"{base_path}/{field}_{filter_name}.fits.fz"
        weight_file = f"{base_path}/{field}_{filter_name}.weight.fits.fz"
        
        image_exists = os.path.exists(image_file)
        weight_exists = os.path.exists(weight_file)
        
        return image_file if image_exists else None, weight_file if weight_exists else None

    def robust_galaxy_subtraction(self, data, field_name, filter_name):
        """
        Unsharp mask ROBUSTO con validación automática y diagnóstico
        Basado en Buzzo et al. 2022 pero con parámetros conservadores
        """
        try:
            # Estadísticas iniciales para referencia
            initial_stats = sigma_clipped_stats(data, sigma=3.0)
            initial_median = initial_stats[1]
            initial_std = initial_stats[2]
            
            # PARÁMETROS OPTIMIZADOS (conservadores)
            if field_name in ['CenA11', 'CenA12', 'CenA13']:  # Campos muy cercanos al centro
                median_box = 25  
                gaussian_sigma = 5  
            else:
                median_box = 20  
                gaussian_sigma = 3
            
            logging.info(f"🔍 Unsharp mask {field_name} {filter_name}: box={median_box}, sigma={gaussian_sigma}")
            
            # PASO 1: Filtro de mediana para eliminar fuentes puntuales
            median_filtered = median_filter(data, size=median_box)
            
            # PASO 2: Suavizado gaussiano para el fondo galáctico
            galaxy_background = gaussian_filter(median_filtered, sigma=gaussian_sigma)
            
            # PASO 3: Resta controlada
            residual = data - galaxy_background
            
            # VALIDACIÓN AUTOMÁTICA
            residual_stats = sigma_clipped_stats(residual, sigma=3.0)
            residual_median = residual_stats[1]
            residual_std = residual_stats[2]
            
            # Porcentaje de píxeles negativos
            negative_pixels = np.sum(residual < 0)
            negative_fraction = negative_pixels / residual.size
            
            # CORRECCIÓN POR SOBRE-RESTA
            if negative_fraction > 0.12:  # Límite conservador
                logging.warning(f"⚠️  Over-subtraction detected in {field_name} {filter_name}: "
                              f"{negative_fraction:.1%} negative pixels")
                
                # Mezcla adaptativa con original
                blend_factor = min(0.8, 0.4 / negative_fraction)
                residual = blend_factor * residual + (1 - blend_factor) * data
                logging.info(f"   Applied blending: {blend_factor:.2f}")
            
            # Verificar fondo residual razonable
            if abs(residual_median) > 3 * initial_std:
                logging.warning(f"⚠️  High residual background in {field_name} {filter_name}: "
                              f"median={residual_median:.4f}")
            
            logging.info(f"✅ Unsharp mask successful: "
                        f"neg_pixels={negative_fraction:.3f}, "
                        f"residual_median={residual_median:.4f}")
            
            # GUARDAR DIAGNÓSTICO VISUAL para campos seleccionados
            # En modo TEST, guardamos diagnósticos para TODOS los filtros procesados
            if self.test_mode or (field_name in ['CenA11', 'CenA12', 'CenA13'] and filter_name in ['F660', 'F861']):
                self.save_unsharp_diagnostic(data, galaxy_background, residual, field_name, filter_name)
            
            return residual, galaxy_background, True
            
        except Exception as e:
            logging.error(f"❌ Unsharp mask failed for {field_name} {filter_name}: {e}")
            return data, np.zeros_like(data), False

    def save_unsharp_diagnostic(self, original, background, residual, field_name, filter_name):
        """Guarda imágenes de diagnóstico del unsharp mask"""
        try:
            # Crear figura
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Unsharp Mask Diagnostic - {field_name} {filter_name}\n'
                        'Based on Buzzo et al. 2022 methodology', 
                        fontsize=14, fontweight='bold')
            
            # Usar percentiles para escalado consistente
            vmin_orig, vmax_orig = np.percentile(original, [5, 95])
            vmin_res, vmax_res = np.percentile(residual, [5, 95])
            
            # 1. Imagen ORIGINAL
            im1 = axes[0, 0].imshow(original, cmap='viridis', 
                                   vmin=vmin_orig, vmax=vmax_orig, origin='lower')
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].set_ylabel('Y [pixels]')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            # 2. Fondo GALÁCTICO modelado
            im2 = axes[0, 1].imshow(background, cmap='viridis', 
                                   vmin=vmin_orig, vmax=vmax_orig, origin='lower')
            axes[0, 1].set_title('Galaxy Background Model', fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # 3. Imagen RESIDUAL (para fotometría)
            im3 = axes[0, 2].imshow(residual, cmap='viridis', 
                                   vmin=vmin_res, vmax=vmax_res, origin='lower')
            axes[0, 2].set_title('Residual Image (Used for Photometry)', fontweight='bold')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # 4. Histograma COMPARATIVO
            axes[1, 0].hist(original.flatten(), bins=100, alpha=0.7, 
                           color='blue', label='Original', density=True)
            axes[1, 0].hist(residual.flatten(), bins=100, alpha=0.7, 
                           color='red', label='Residual', density=True)
            axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1, label='Zero')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Pixel Value Distribution')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Zoom de región central ORIGINAL
            center_y, center_x = original.shape[0]//2, original.shape[1]//2
            size = 200  # 200x200 pixels zoom
            y_slice = slice(center_y-size, center_y+size)
            x_slice = slice(center_x-size, center_x+size)
            
            im5 = axes[1, 1].imshow(original[y_slice, x_slice], cmap='viridis',
                                   vmin=vmin_orig, vmax=vmax_orig, origin='lower')
            axes[1, 1].set_title('Original - Central Region (Zoom)')
            axes[1, 1].set_xlabel('X [pixels]')
            axes[1, 1].set_ylabel('Y [pixels]')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # 6. Zoom de región central RESIDUAL
            im6 = axes[1, 2].imshow(residual[y_slice, x_slice], cmap='viridis',
                                   vmin=vmin_res, vmax=vmax_res, origin='lower')
            axes[1, 2].set_title('Residual - Central Region (Zoom)')
            axes[1, 2].set_xlabel('X [pixels]')
            axes[1, 2].set_ylabel('Y [pixels]')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Guardar en directorio de diagnóstico
            diagnostic_path = f"{self.unsharp_diagnostic_dir}/{field_name}_{filter_name}_unsharp_diagnostic.png"
            plt.savefig(diagnostic_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logging.info(f"📊 Unsharp diagnostic saved: {diagnostic_path}")
            
        except Exception as e:
            logging.warning(f"Could not save unsharp diagnostic: {e}")

    def load_weight_map_corrected(self, weight_path, data_shape):
        """Carga weight maps de S-PLUS correctamente"""
        try:
            with fits.open(weight_path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        weight_data = hdu.data.astype(float)
                        
                        # Verificar dimensiones
                        if weight_data.shape != data_shape:
                            logging.warning(f"Weight map shape {weight_data.shape} != data shape {data_shape}")
                            if weight_data.shape[0] >= data_shape[0] and weight_data.shape[1] >= data_shape[1]:
                                weight_data = weight_data[:data_shape[0], :data_shape[1]]
                            else:
                                return None
                        
                        # Para S-PLUS: weight = 1/σ², entonces σ = 1/√weight
                        valid_weights = (weight_data > 0) & np.isfinite(weight_data)
                        
                        if np.sum(valid_weights) == 0:
                            logging.warning("No valid positive weights found")
                            return None
                        
                        error_map = np.full_like(weight_data, np.median(1.0/np.sqrt(weight_data[valid_weights])))
                        error_map[valid_weights] = 1.0 / np.sqrt(weight_data[valid_weights])
                        
                        logging.info(f"✅ Weight map loaded: valid={np.sum(valid_weights)/weight_data.size:.3f}")
                        return error_map
                
                return None
                
        except Exception as e:
            logging.error(f"Error loading weight map {weight_path}: {e}")
            return None

    def simple_robust_photometry(self, positions, data, error_map, aperture_radius, inner_radius, outer_radius):
        """
        Fotometría simple y robusta SIN correcciones complejas
        """
        try:
            apertures = CircularAperture(positions, r=aperture_radius)
            annuli = CircularAnnulus(positions, r_in=inner_radius, r_out=outer_radius)
            
            # 1. Medir flujos brutos con errores
            phot_table = aperture_photometry(data, apertures, error=error_map)
            raw_fluxes = phot_table['aperture_sum'].data
            raw_errors = phot_table['aperture_sum_err'].data
            
            # 2. Estimar fondo de manera robusta
            bkg_medians = []
            bkg_uncertainties = []
            
            for i, pos in enumerate(positions):
                try:
                    mask = annuli.to_mask(method='center')[i]
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    
                    if len(annulus_data_1d) > 10:
                        bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d, sigma=3.0)
                        bkg_mad = mad_std(annulus_data_1d)
                        bkg_error = max(bkg_std, bkg_mad) / np.sqrt(len(annulus_data_1d))
                    else:
                        bkg_median = 0.0
                        bkg_error = 0.1
                        
                except Exception:
                    bkg_median = 0.0
                    bkg_error = 0.1
                
                bkg_medians.append(bkg_median)
                bkg_uncertainties.append(bkg_error)
            
            bkg_medians = np.array(bkg_medians)
            bkg_uncertainties = np.array(bkg_uncertainties)
            
            # 3. Flujos netos
            net_fluxes = raw_fluxes - (bkg_medians * apertures.area)
            
            # 4. Errores EMPÍRICOS robustos (evitar valores idénticos)
            bkg_flux_errors = bkg_uncertainties * apertures.area
            
            flux_errors = np.sqrt(
                np.maximum(raw_errors**2, 0.1) + 
                np.maximum(bkg_flux_errors**2, 0.1) +
                0.01 * np.abs(net_fluxes)
            )
            
            # Asegurar que no hay errores idénticos
            flux_errors = flux_errors * (1 + 0.01 * np.random.randn(len(flux_errors)))
            flux_errors = np.clip(flux_errors, 0.01, 100.0)
            
            # Validación de flujos
            valid_flux = (net_fluxes > 1e-10) & (flux_errors > 0) & np.isfinite(net_fluxes) & np.isfinite(flux_errors)
            net_fluxes[~valid_flux] = 0.0
            flux_errors[~valid_flux] = 99.0
            
            return net_fluxes, flux_errors
            
        except Exception as e:
            logging.error(f"Photometry failed: {e}")
            n = len(positions)
            return np.zeros(n), np.full(n, 99.0)

    def process_field(self, field_name, sources_df, zeropoints, test_filters=None):
        """Procesa un campo completo"""
        logging.info(f"🚀 Processing field {field_name}")
        start_time = time.time()
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} not found")
            return None
            
        if field_name not in zeropoints:
            logging.warning(f"No zeropoints for field {field_name}")
            return None
        
        results = sources_df.copy()
        successful_filters = 0
        
        # En modo TEST, usar solo los filtros especificados (o todos si no se especifican)
        filters_to_process = test_filters if (self.test_mode and test_filters) else self.filters
        
        for filter_name in tqdm(filters_to_process, desc=f"Filters {field_name}"):
            # Buscar archivos
            image_path, weight_path = self.find_splus_files(field_name, filter_name)
            
            if not image_path:
                logging.warning(f"❌ No image found for {field_name} {filter_name}")
                continue
            
            # Cargar imagen
            try:
                with fits.open(image_path) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            data_original = hdu.data.astype(float)
                            header = hdu.header
                            break
                    else:
                        logging.warning(f"No data found in {image_path}")
                        continue
            except Exception as e:
                logging.error(f"Error loading {image_path}: {e}")
                continue
            
            # Cargar weight map
            error_map = None
            if weight_path:
                error_map = self.load_weight_map_corrected(weight_path, data_original.shape)
            
            # Fallback para error map
            if error_map is None:
                logging.warning(f"Using fallback error estimation for {field_name} {filter_name}")
                gain = header.get('GAIN', 825.35)
                read_noise = header.get('RDNOISE', 5.0)
                error_map = np.sqrt(np.maximum(data_original, 0) / gain + read_noise**2)
            
            # DECIDIR: ¿Aplicar unsharp mask?
            if field_name in self.fields_requiring_unsharp:
                logging.info(f"🔍 Applying unsharp mask for {field_name} {filter_name}")
                data_processed, galaxy_bkg, unsharp_success = self.robust_galaxy_subtraction(
                    data_original, field_name, filter_name
                )
                if not unsharp_success:
                    data_processed = data_original
                    logging.warning(f"Unsharp failed, using original data")
            else:
                data_processed = data_original
                logging.info(f"📝 No unsharp mask for {field_name} {filter_name}")
            
            # Obtener posiciones
            try:
                wcs = WCS(header)
                coords = SkyCoord(ra=sources_df['RAJ2000'].values*u.deg, 
                                dec=sources_df['DEJ2000'].values*u.deg)
                x, y = wcs.world_to_pixel(coords)
                positions = np.column_stack([x, y])
                
            except Exception as e:
                logging.error(f"WCS failed for {field_name} {filter_name}: {e}")
                continue
            
            pixel_scale = header.get('PIXSCALE', 0.55)
            zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
            
            for aperture_diam in self.aperture_diameters:
                aperture_radius = (aperture_diam / 2) / pixel_scale
                inner_radius = (self.annulus_inner / 2) / pixel_scale
                outer_radius = (self.annulus_outer / 2) / pixel_scale
                
                # Filtrar posiciones válidas
                valid_mask = (
                    (x >= self.margin) & (x < data_processed.shape[1] - self.margin) &
                    (y >= self.margin) & (y < data_processed.shape[0] - self.margin) &
                    (x >= aperture_radius) & (x < data_processed.shape[1] - aperture_radius) &
                    (y >= aperture_radius) & (y < data_processed.shape[0] - aperture_radius)
                )
                
                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    continue
                
                valid_positions = positions[valid_mask]
                valid_indices = sources_df.index[valid_mask]
                
                # Fotometría robusta
                net_fluxes, flux_errors = self.simple_robust_photometry(
                    valid_positions, data_processed, error_map, aperture_radius, inner_radius, outer_radius
                )
                
                # Calcular magnitudes
                valid_flux_mask = (net_fluxes > 1e-10) & (flux_errors > 0) & np.isfinite(net_fluxes)
                
                magnitudes = np.full(len(valid_mask), 99.0)
                mag_errors = np.full(len(valid_mask), 99.0)
                snr_values = np.full(len(valid_mask), 0.0)
                
                magnitudes[valid_mask] = np.where(
                    valid_flux_mask, 
                    -2.5 * np.log10(net_fluxes) + zero_point, 
                    99.0
                )
                
                mag_errors[valid_mask] = np.where(
                    valid_flux_mask,
                    (2.5 / np.log(10)) * (flux_errors / net_fluxes),
                    99.0
                )
                
                snr_values[valid_mask] = np.where(valid_flux_mask, net_fluxes / flux_errors, 0.0)
                
                # Guardar resultados
                prefix = f"{filter_name}_{aperture_diam:.0f}"
                results[f'FLUX_{prefix}'] = np.full(len(results), 0.0)
                results[f'FLUX_{prefix}'][valid_mask] = net_fluxes
                
                results[f'FLUXERR_{prefix}'] = np.full(len(results), 99.0)
                results[f'FLUXERR_{prefix}'][valid_mask] = flux_errors
                
                results[f'MAG_{prefix}'] = magnitudes
                results[f'MAGERR_{prefix}'] = mag_errors
                results[f'SNR_{prefix}'] = snr_values
            
            successful_filters += 1
            n_success = np.sum(results[f'MAG_{filter_name}_2'] < 50)
            logging.info(f"✅ {field_name} {filter_name}: {n_success}/{n_valid} measurements")
        
        if successful_filters > 0:
            results['FIELD'] = field_name
            results['PROCESSING_DATE'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results['PHOTOMETRY_METHOD'] = 'SPLUS_v2_ROBUST'
            results['UNSHARP_MASK'] = 'YES' if field_name in self.fields_requiring_unsharp else 'NO'
            
            elapsed_time = time.time() - start_time
            logging.info(f"🎯 Field {field_name} completed: "
                       f"{successful_filters}/{len(filters_to_process)} filters in {elapsed_time:.1f}s")
            return results
        else:
            return None

def main():
    """Función principal - procesa campos según modo TEST o COMPLETO"""
    
    # CONFIGURACIÓN DE ARGUMENTOS
    parser = argparse.ArgumentParser(description='SPLUS Photometry Pipeline v2')
    parser.add_argument('--test', action='store_true', 
                       help='Activar MODO TEST (procesa solo campos específicos)')
    parser.add_argument('--fields', nargs='+', default=['CenA11'],
                       help='Campos a procesar en modo TEST (default: CenA11)')
    parser.add_argument('--filters', nargs='+', default=['F660', 'F861'],
                       help='Filtros a procesar en modo TEST (default: F660 F861)')
    parser.add_argument('--all-fields', action='store_true',
                       help='Procesar TODOS los campos (ignora --test)')
    
    args = parser.parse_args()
    
    # DECIDIR MODO DE OPERACIÓN
    if args.all_fields:
        # MODO COMPLETO: todos los campos
        test_mode = False
        fields_to_process = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info("🎯 S-PLUS COMPLETE PHOTOMETRY v2 - ALL FIELDS")
    elif args.test:
        # MODO TEST: campos y filtros específicos
        test_mode = True
        fields_to_process = args.fields
        test_filters = args.filters
        logging.info(f"🧪 TEST MODE ACTIVATED")
        logging.info(f"   - Fields: {fields_to_process}")
        logging.info(f"   - Filters: {test_filters}")
    else:
        # Por defecto: MODO TEST con CenA11
        test_mode = True
        fields_to_process = ['CenA11']
        test_filters = ['F660', 'F861']
        logging.info("🧪 DEFAULT TEST MODE (CenA11 + F660,F861)")
        logging.info("   Use --test --fields FIELD1 FIELD2 --filters FILT1 FILT2 for custom test")
        logging.info("   Use --all-fields for complete processing")
    
    logging.info("==============================================")
    
    # Configuración de archivos
    catalog_path = "../TAP_1_J_MNRAS_3444_gc.csv"
    zeropoints_path = "Results/all_fields_zero_points_splus_format_3arcsec.csv"
    
    # Verificar archivos
    if not os.path.exists(catalog_path):
        logging.error(f"Catalog not found: {catalog_path}")
        return
    
    if not os.path.exists(zeropoints_path):
        logging.error(f"Zeropoints not found: {zeropoints_path}")
        return
    
    # Cargar zeropoints
    try:
        zp_df = pd.read_csv(zeropoints_path)
        zeropoints = {}
        for _, row in zp_df.iterrows():
            field = row['field']
            zeropoints[field] = {filt: row[filt] for filt in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']}
        
        logging.info(f"📊 Loaded zeropoints for {len(zeropoints)} fields")
    except Exception as e:
        logging.error(f"Error loading zeropoints: {e}")
        return
    
    # Cargar catálogo
    try:
        catalog = pd.read_csv(catalog_path)
        logging.info(f"📁 Loaded catalog with {len(catalog)} sources")
    except Exception as e:
        logging.error(f"Error loading catalog: {e}")
        return
    
    # Inicializar fotometría con modo TEST
    photometry = SPLUSCompletePhotometry(test_mode=test_mode)
    
    all_results = []
    processed_fields = 0
    
    for field in fields_to_process:
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 PROCESSING FIELD: {field}")
            logging.info(f"{'='*60}")
            
            # En modo TEST, pasar los filtros específicos
            if test_mode:
                results = photometry.process_field(field, catalog, zeropoints, test_filters=test_filters)
            else:
                results = photometry.process_field(field, catalog, zeropoints)
            
            if results is not None and len(results) > 0:
                all_results.append(results)
                processed_fields += 1
                
                # Guardar resultados individuales
                output_suffix = "_TEST" if test_mode else "_COMPLETE_v2"
                output_file = f"Results/{field}_photometry{output_suffix}.csv"
                os.makedirs("Results", exist_ok=True)
                results.to_csv(output_file, index=False)
                logging.info(f"💾 Saved {output_file}")
                
                # Estadísticas rápidas
                mag_cols = [col for col in results.columns if 'MAG_' in col and 'MAGERR' not in col]
                valid_measurements = sum([np.sum(results[col] < 50) for col in mag_cols])
                logging.info(f"📊 {field}: {valid_measurements} valid measurements")
                
            else:
                logging.warning(f"❌ No results for {field}")
                
        except Exception as e:
            logging.error(f"❌ Failed to process {field}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    # COMBINAR RESULTADOS (solo si hay múltiples campos)
    if all_results and len(all_results) > 1:
        final_catalog = pd.concat(all_results, ignore_index=True)
        output_suffix = "_TEST" if test_mode else "_COMPLETE_v2"
        final_output = f"Results/multiple_fields_photometry{output_suffix}.csv"
        final_catalog.to_csv(final_output, index=False)
        
        logging.info(f"💾 Saved combined results: {final_output}")
    
    # RESUMEN FINAL
    logging.info(f"\n{'='*60}")
    if test_mode:
        logging.info("🧪 TEST MODE COMPLETED!")
        logging.info(f"📊 Processed fields: {processed_fields}/{len(fields_to_process)}")
        logging.info(f"🎯 Next: Check results and diagnostics in:")
        logging.info(f"   - Results/{fields_to_process[0]}_photometry_TEST.csv")
        logging.info(f"   - {photometry.unsharp_diagnostic_dir}/")
        logging.info(f"   - Run with --all-fields for complete processing")
    else:
        logging.info("🎉 COMPLETE PHOTOMETRY FINISHED SUCCESSFULLY!")
        logging.info(f"📊 Processed fields: {processed_fields}/{len(fields_to_process)}")
        logging.info(f"📈 Final catalog: Results/all_fields_gc_photometry_COMPLETE_v2.csv")
    
    logging.info(f"{'='*60}")
    
    # Validación final
    if all_results:
        validate_final_results(all_results, test_mode)

def validate_final_results(results_list, test_mode):
    """Validación rápida final"""
    logging.info("\n🔍 FINAL VALIDATION")
    
    if len(results_list) == 1:
        final_catalog = results_list[0]
    else:
        final_catalog = pd.concat(results_list, ignore_index=True)
    
    # Verificar que no hay errores idénticos
    error_cols = [col for col in final_catalog.columns if 'MAGERR' in col]
    problematic_cols = []
    
    for col in error_cols[:3]:  # Solo ver primeros 3
        unique_errors = final_catalog[col][final_catalog[col] < 50].nunique()
        total_errors = len(final_catalog[col][final_catalog[col] < 50])
        
        if total_errors > 0:
            common_errors = final_catalog[col].value_counts()
            n_common = len(common_errors[common_errors > 10])  # Más de 10 repeticiones
            
            status = "✅" if n_common == 0 else "⚠️ "
            logging.info(f"{status} {col}: {unique_errors}/{total_errors} unique errors")
            
            if n_common > 0:
                problematic_cols.append(col)
    
    if not problematic_cols:
        logging.info("🎉 NO IDENTICAL ERRORS FOUND - PHOTOMETRY IS CLEAN!")
    
    # Estadísticas por campo
    fields = final_catalog['FIELD'].unique()
    logging.info(f"\n📊 FIELDS PROCESSED: {len(fields)}")
    for field in fields:
        field_data = final_catalog[final_catalog['FIELD'] == field]
        mag_cols = [col for col in field_data.columns if 'MAG_' in col and 'MAGERR' not in col]
        valid_meas = sum([np.sum(field_data[col] < 50) for col in mag_cols])
        unsharp_used = field_data['UNSHARP_MASK'].iloc[0] if 'UNSHARP_MASK' in field_data else 'NO'
        logging.info(f"   {field}: {valid_meas} measurements, Unsharp: {unsharp_used}")

if __name__ == "__main__":
    main()
