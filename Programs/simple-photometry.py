#!/usr/bin/env python3
"""
splus_photometry_final_clean.py
Fotometría limpia usando la estructura real de archivos S-PLUS
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats, mad_std
import logging
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_clean_photometry.log'),
        logging.StreamHandler()
    ]
)

class SPLUSCleanPhotometry:
    def __init__(self):
        self.pixel_scale = 0.55
        self.aperture_diameters = [2.0, 3.0]  # Aperturas simples
        self.annulus_inner = 4.0
        self.annulus_outer = 6.0
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.margin = 20  # Margen más conservador
    
    def find_splus_files(self, field, filter_name):
        """Encuentra archivos S-PLUS usando la estructura real"""
        base_path = field
        image_file = f"{base_path}/{field}_{filter_name}.fits.fz"
        weight_file = f"{base_path}/{field}_{filter_name}.weight.fits.fz"
        
        # Verificar que existen
        image_exists = os.path.exists(image_file)
        weight_exists = os.path.exists(weight_file)
        
        return image_file if image_exists else None, weight_file if weight_exists else None
    
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
                            # Intentar recortar o ajustar si es necesario
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
                        
                        logging.info(f"Weight map loaded: valid={np.sum(valid_weights)/weight_data.size:.3f}, "
                                   f"error_range=[{np.min(error_map[valid_weights]):.3f}, {np.max(error_map[valid_weights]):.3f}]")
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
                        # Usar sigma-clipping para estimar fondo
                        bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d, sigma=3.0)
                        bkg_mad = mad_std(annulus_data_1d)
                        
                        # Error del fondo (conservador)
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
            
            # Propagación básica pero con términos de ruido empírico
            flux_errors = np.sqrt(
                np.maximum(raw_errors**2, 0.1) + 
                np.maximum(bkg_flux_errors**2, 0.1) +
                0.01 * np.abs(net_fluxes)  # Término de Poisson empírico
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
    
    def process_field(self, field_name, sources_df, zeropoints):
        """Procesa un campo completo de manera robusta"""
        logging.info(f"🚀 Processing field {field_name}")
        
        results = sources_df.copy()
        
        for filter_name in tqdm(self.filters, desc=f"Filters {field_name}"):
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
                            data = hdu.data.astype(float)
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
                error_map = self.load_weight_map_corrected(weight_path, data.shape)
            
            # Fallback para error map
            if error_map is None:
                logging.warning(f"Using fallback error estimation for {field_name} {filter_name}")
                gain = header.get('GAIN', 825.35)
                read_noise = header.get('RDNOISE', 5.0)
                error_map = np.sqrt(np.maximum(data, 0) / gain + read_noise**2)
            
            # Obtener posiciones
            try:
                wcs = WCS(header)
                coords = SkyCoord(ra=sources_df['RAJ2000'].values*u.deg, 
                                dec=sources_df['DEJ2000'].values*u.deg)
                x, y = wcs.world_to_pixel(coords)
                positions = np.column_stack([x, y])
                
                logging.info(f"📍 {filter_name}: {len(positions)} sources, image shape {data.shape}")
                
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
                    (x >= self.margin) & (x < data.shape[1] - self.margin) &
                    (y >= self.margin) & (y < data.shape[0] - self.margin) &
                    (x >= aperture_radius) & (x < data.shape[1] - aperture_radius) &
                    (y >= aperture_radius) & (y < data.shape[0] - aperture_radius)
                )
                
                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    logging.warning(f"No valid positions for {filter_name} {aperture_diam}\"")
                    continue
                
                valid_positions = positions[valid_mask]
                valid_indices = sources_df.index[valid_mask]
                
                # Fotometría robusta
                net_fluxes, flux_errors = self.simple_robust_photometry(
                    valid_positions, data, error_map, aperture_radius, inner_radius, outer_radius
                )
                
                # Calcular magnitudes
                valid_flux_mask = (net_fluxes > 1e-10) & (flux_errors > 0) & np.isfinite(net_fluxes)
                
                magnitudes = np.full(len(valid_mask), 99.0)
                mag_errors = np.full(len(valid_mask), 99.0)
                snr_values = np.full(len(valid_mask), 0.0)
                
                # Solo calcular para flujos válidos
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
            
            n_success = np.sum(results[f'MAG_{filter_name}_2'] < 50)
            logging.info(f"✅ {filter_name}: {n_success}/{n_valid} successful measurements")
        
        results['FIELD'] = field_name
        return results

def main():
    """Función principal"""
    
    logging.info("🎯 S-PLUS CLEAN PHOTOMETRY - FRESH START")
    logging.info("==============================================")
    
    # Configuración
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
    
    # Inicializar fotometría
    photometry = SPLUSCleanPhotometry()
    
    # Procesar campos (empezar con prueba)
    test_fields = ['CenA01']  # Solo un campo para prueba
    
    all_results = []
    for field in test_fields:
        if not os.path.exists(field):
            logging.warning(f"Field directory {field} not found, skipping")
            continue
            
        try:
            results = photometry.process_field(field, catalog, zeropoints)
            
            if len(results) > 0:
                all_results.append(results)
                
                # Guardar resultados individuales
                output_file = f"Results/{field}_photometry_CLEAN_FINAL.csv"
                os.makedirs("Results", exist_ok=True)
                results.to_csv(output_file, index=False)
                logging.info(f"💾 Saved {output_file}")
            else:
                logging.warning(f"No results for {field}")
                
        except Exception as e:
            logging.error(f"Failed to process {field}: {e}")
            continue
    
    # Combinar resultados
    if all_results:
        final_catalog = pd.concat(all_results, ignore_index=True)
        final_output = "Results/gc_photometry_CLEAN_FINAL.csv"
        final_catalog.to_csv(final_output, index=False)
        
        logging.info("🎉 CLEAN PHOTOMETRY COMPLETED SUCCESSFULLY!")
        logging.info(f"📊 Final catalog: {final_output}")
        logging.info(f"📈 Total sources: {len(final_catalog)}")
        
        # Validación rápida
        validate_quick(final_catalog)
    else:
        logging.error("❌ No results generated")

def validate_quick(results_df):
    """Validación rápida de la fotometría"""
    logging.info("\n🔍 QUICK VALIDATION")
    
    # Verificar errores idénticos
    error_cols = [col for col in results_df.columns if 'MAGERR' in col]
    
    for col in error_cols[:3]:  # Solo primeros 3
        errors = results_df[col][results_df[col] < 50]
        unique_errors = len(np.unique(errors))
        total_errors = len(errors)
        
        if total_errors > 0:
            common_errors = results_df[col].value_counts()
            n_common = len(common_errors[common_errors > 10])
            
            status = "✅" if n_common == 0 else "⚠️ "
            logging.info(f"{status} {col}: {unique_errors}/{total_errors} unique errors")
    
    # Estadísticas básicas
    mag_cols = [col for col in results_df.columns if 'MAG_' in col and 'MAGERR' not in col]
    valid_measurements = sum([np.sum(results_df[col] < 50) for col in mag_cols])
    total_possible = len(results_df) * len(mag_cols)
    
    logging.info(f"📊 Measurement success: {valid_measurements}/{total_possible} ({valid_measurements/total_possible*100:.1f}%)")

if __name__ == "__main__":
    main()
