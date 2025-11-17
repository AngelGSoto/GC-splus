#!/usr/bin/env python3
"""
Splus_photometry_corrected_final.py
Fotometría CORREGIDA para todos los campos CenA01-CenA24
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
import argparse

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('splus_photometry_corrected_final.log'),
        logging.StreamHandler()
    ]
)

class SPLUSPhotometryCorrectedFinal:
    def __init__(self, test_mode=False, apply_unsharp_to_all=True):
        self.pixel_scale = 0.55
        self.aperture_diameters = [2.0, 3.0]
        self.annulus_inner = 4.0
        self.annulus_outer = 6.0
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.margin = 20
        
        # Tratamiento consistente
        self.apply_unsharp_to_all = apply_unsharp_to_all
        
        # Lista completa de campos
        self.all_cenA_fields = [f'CenA{i:02d}' for i in range(1, 25)]
        
        # Directorios
        self.unsharp_diagnostic_dir = "unsharp-mask-corrected"
        self.results_dir = "Results_Corrected"
        
        # MODO TEST
        self.test_mode = test_mode
        
        # Crear directorios
        os.makedirs(self.unsharp_diagnostic_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logging.info("🎯 S-PLUS PHOTOMETRY CORRECTED FINAL INITIALIZED")
        if self.apply_unsharp_to_all:
            logging.info("   - APPLYING IDENTICAL UNSHARP TO ALL 24 FIELDS")
        else:
            logging.info("   - USING ORIGINAL DATA FOR ALL FIELDS")

    def find_splus_files(self, field, filter_name):
        """Encuentra archivos S-PLUS"""
        base_path = field
        image_file = f"{base_path}/{field}_{filter_name}.fits.fz"
        weight_file = f"{base_path}/{field}_{filter_name}.weight.fits.fz"
        
        return (image_file if os.path.exists(image_file) else None,
                weight_file if os.path.exists(weight_file) else None)

    def get_image_bounds(self, header):
        """Obtiene los límites RA/DEC de la imagen usando WCS"""
        try:
            wcs = WCS(header)
            width = header.get('NAXIS1', 0)
            height = header.get('NAXIS2', 0)
            
            # Esquinas de la imagen en pixels
            corners_pix = np.array([
                [0, 0], 
                [width-1, 0], 
                [0, height-1], 
                [width-1, height-1]
            ])
            
            # Convertir a coordenadas mundiales
            corners_world = wcs.pixel_to_world(corners_pix[:, 0], corners_pix[:, 1])
            
            ra_values = [coord.ra.deg for coord in corners_world]
            dec_values = [coord.dec.deg for coord in corners_world]
            
            ra_min, ra_max = min(ra_values), max(ra_values)
            dec_min, dec_max = min(dec_values), max(dec_values)
            
            return ra_min, ra_max, dec_min, dec_max, wcs
            
        except Exception as e:
            logging.error(f"Error getting image bounds: {e}")
            return None, None, None, None, None

    def filter_sources_in_field(self, sources_df, ra_min, ra_max, dec_min, dec_max):
        """Filtra fuentes que están dentro del campo actual"""
        # Margen de 0.1 grados para seguridad
        margin = 0.1
        mask = (
            (sources_df['RAJ2000'] >= ra_min - margin) & 
            (sources_df['RAJ2000'] <= ra_max + margin) & 
            (sources_df['DEJ2000'] >= dec_min - margin) & 
            (sources_df['DEJ2000'] <= dec_max + margin)
        )
        
        filtered_df = sources_df[mask].copy()
        logging.info(f"📊 Field contains {len(filtered_df)} of {len(sources_df)} total sources")
        
        return filtered_df

    def simple_unsharp_mask(self, data, field_name, filter_name):
        """Unsharp mask idéntico al paper"""
        try:
            median_box = 25  
            gaussian_sigma = 5
            
            logging.info(f"🔍 Applying unsharp to {field_name} {filter_name}")
            
            median_filtered = median_filter(data, size=median_box)
            galaxy_background = gaussian_filter(median_filtered, sigma=gaussian_sigma)
            residual = data - galaxy_background
            
            # Estadísticas informativas
            residual_stats = sigma_clipped_stats(residual, sigma=3.0)
            negative_fraction = np.sum(residual < 0) / residual.size
            
            logging.info(f"✅ Unsharp successful: neg_pixels={negative_fraction:.3f}")
            
            # Guardar diagnóstico para algunos campos
            if self.test_mode or field_name in ['CenA01', 'CenA11', 'CenA12']:
                self.save_diagnostic(data, galaxy_background, residual, field_name, filter_name)
            
            return residual, galaxy_background, True
            
        except Exception as e:
            logging.error(f"❌ Unsharp failed for {field_name} {filter_name}: {e}")
            return data, np.zeros_like(data), False

    def save_diagnostic(self, original, background, residual, field_name, filter_name):
        """Guarda diagnóstico del unsharp mask"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            vmin_orig, vmax_orig = np.percentile(original, [1, 99])
            
            im1 = axes[0].imshow(original, cmap='gray', origin='lower',
                               vmin=vmin_orig, vmax=vmax_orig)
            axes[0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(background, cmap='gray', origin='lower',
                               vmin=vmin_orig, vmax=vmax_orig)
            axes[1].set_title('Galaxy Background')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(residual, cmap='RdGy_r', origin='lower',
                               vmin=-np.percentile(np.abs(residual), 99),
                               vmax=np.percentile(np.abs(residual), 99))
            axes[2].set_title('Residual Image')
            plt.colorbar(im3, ax=axes[2])
            
            fig.suptitle(f'Unsharp Mask - {field_name} {filter_name}')
            plt.tight_layout()
            
            diagnostic_path = f"{self.unsharp_diagnostic_dir}/{field_name}_{filter_name}.png"
            plt.savefig(diagnostic_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.info(f"📊 Diagnostic saved: {diagnostic_path}")
            
        except Exception as e:
            logging.warning(f"Could not save diagnostic: {e}")

    def load_weight_map(self, weight_path, data_shape):
        """Carga weight map corregido"""
        try:
            with fits.open(weight_path) as whdul:
                for hdu in whdul:
                    if hdu.data is not None:
                        weight_data = hdu.data.astype(float)
                        break
                else:
                    return None
            
            if weight_data.shape != data_shape:
                return None
            
            valid_weight = (weight_data > 0) & np.isfinite(weight_data)
            if np.sum(valid_weight) / weight_data.size < 0.5:
                return None
            
            return 1.0 / np.sqrt(weight_data)
            
        except Exception as e:
            logging.error(f"Error loading weight map: {e}")
            return None

    def calculate_photometry(self, data, error_map, positions, aperture_radius, inner_radius, outer_radius):
        """Calcula fotometría para un conjunto de posiciones"""
        apertures = CircularAperture(positions, r=aperture_radius)
        annulus = CircularAnnulus(positions, r_in=inner_radius, r_out=outer_radius)
        
        # Fotometría básica
        phot_table = aperture_photometry(data, apertures, error=error_map)
        raw_flux = phot_table['aperture_sum'].data
        raw_flux_err = phot_table['aperture_sum_err'].data
        
        bkg_medians = []
        bkg_errors = []
        
        # Calcular fondo para cada posición
        for i, pos in enumerate(positions):
            try:
                mask = annulus.to_mask(method='center')[i]
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                
                if len(annulus_data_1d) > 10:
                    bkg_median = np.median(annulus_data_1d)
                    bkg_mad = mad_std(annulus_data_1d)
                    bkg_error = 1.253 * bkg_mad / np.sqrt(len(annulus_data_1d)) * apertures.area
                else:
                    bkg_median = 0.0
                    bkg_error = 0.0
                    
            except Exception:
                bkg_median = 0.0
                bkg_error = 0.0
            
            bkg_medians.append(bkg_median)
            bkg_errors.append(bkg_error)
        
        bkg_medians = np.array(bkg_medians)
        bkg_errors = np.array(bkg_errors)
        
        # Calcular flujo neto y errores
        net_flux = raw_flux - (bkg_medians * apertures.area)
        net_flux_err = np.sqrt(raw_flux_err**2 + bkg_errors**2)
        
        # Limitar error máximo razonable
        reasonable_ratio = 0.5
        for i in range(len(net_flux)):
            if net_flux[i] > 0 and net_flux_err[i] > 0:
                if net_flux_err[i] / net_flux[i] > reasonable_ratio:
                    net_flux_err[i] = net_flux[i] * reasonable_ratio
        
        return net_flux, net_flux_err, bkg_medians

    def process_field(self, field_name, sources_df, zeropoints, test_filters=None):
        """Procesa un campo completo - VERSIÓN CORREGIDA"""
        logging.info(f"🚀 Processing field {field_name}")
        start_time = time.time()
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} not found")
            return None
            
        if field_name not in zeropoints:
            logging.warning(f"No zeropoints for field {field_name}")
            return None
        
        # Paso 1: Obtener límites del campo usando una imagen de referencia
        sample_filter = test_filters[0] if (self.test_mode and test_filters) else self.filters[0]
        image_path, _ = self.find_splus_files(field_name, sample_filter)
        
        if not image_path:
            logging.warning(f"No image found for {field_name}")
            return None
        
        try:
            with fits.open(image_path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        header = hdu.header
                        break
                else:
                    logging.warning(f"No data in {image_path}")
                    return None
        except Exception as e:
            logging.error(f"Error loading {image_path}: {e}")
            return None
        
        # Paso 2: Filtrar fuentes que están en este campo
        ra_min, ra_max, dec_min, dec_max, wcs = self.get_image_bounds(header)
        if wcs is None:
            logging.error(f"Could not get WCS for {field_name}")
            return None
        
        field_sources = self.filter_sources_in_field(sources_df, ra_min, ra_max, dec_min, dec_max)
        
        if len(field_sources) == 0:
            logging.warning(f"No sources in field {field_name}")
            return None
        
        # Paso 3: Inicializar DataFrame SOLO con fuentes de este campo
        results = field_sources.copy()
        results['FIELD'] = field_name
        
        # Inicializar columnas de resultados
        filters_to_process = test_filters if (self.test_mode and test_filters) else self.filters
        
        for filter_name in filters_to_process:
            for aperture_diam in self.aperture_diameters:
                prefix = f"{filter_name}_{aperture_diam:.0f}"
                results[f'FLUX_{prefix}'] = 0.0
                results[f'FLUXERR_{prefix}'] = 99.0
                results[f'MAG_{prefix}'] = 99.0
                results[f'MAGERR_{prefix}'] = 99.0
                results[f'SNR_{prefix}'] = 0.0
        
        successful_filters = 0
        
        # Paso 4: Procesar cada filtro
        for filter_name in tqdm(filters_to_process, desc=f"Filters {field_name}"):
            image_path, weight_path = self.find_splus_files(field_name, filter_name)
            
            if not image_path:
                logging.warning(f"No image for {field_name} {filter_name}")
                continue
            
            try:
                with fits.open(image_path) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            data_original = hdu.data.astype(float)
                            header = hdu.header
                            break
                    else:
                        continue
            except Exception as e:
                logging.error(f"Error loading {image_path}: {e}")
                continue
            
            # Cargar error map
            error_map = None
            if weight_path:
                error_map = self.load_weight_map(weight_path, data_original.shape)
            
            if error_map is None:
                gain = header.get('GAIN', 825.35)
                read_noise = header.get('RDNOISE', 5.0)
                error_map = np.sqrt(np.maximum(data_original, 0) / gain + read_noise**2)
            
            # Aplicar unsharp si está configurado
            if self.apply_unsharp_to_all:
                data_processed, _, unsharp_ok = self.simple_unsharp_mask(
                    data_original, field_name, filter_name
                )
                if not unsharp_ok:
                    data_processed = data_original
            else:
                data_processed = data_original
            
            # Obtener posiciones SOLO de las fuentes en este campo
            try:
                wcs = WCS(header)
                coords = SkyCoord(ra=field_sources['RAJ2000'].values*u.deg, 
                                dec=field_sources['DEJ2000'].values*u.deg)
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
                
                # Validar posiciones
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
                valid_indices = field_sources.index[valid_mask]
                
                # Fotometría
                net_fluxes, flux_errors, bkg_medians = self.calculate_photometry(
                    data_processed, error_map, valid_positions, aperture_radius, inner_radius, outer_radius
                )
                
                # Calcular magnitudes
                valid_flux_mask = (net_fluxes > 1e-10) & (flux_errors > 0) & np.isfinite(net_fluxes)
                
                magnitudes = np.where(valid_flux_mask, -2.5 * np.log10(net_fluxes) + zero_point, 99.0)
                mag_errors = np.where(valid_flux_mask, (2.5 / np.log(10)) * (flux_errors / net_fluxes), 99.0)
                snr_values = np.where(valid_flux_mask, net_fluxes / flux_errors, 0.0)
                
                # Asignar resultados SOLO a fuentes válidas
                prefix = f"{filter_name}_{aperture_diam:.0f}"
                results.loc[valid_indices, f'FLUX_{prefix}'] = net_fluxes
                results.loc[valid_indices, f'FLUXERR_{prefix}'] = flux_errors
                results.loc[valid_indices, f'MAG_{prefix}'] = magnitudes
                results.loc[valid_indices, f'MAGERR_{prefix}'] = mag_errors
                results.loc[valid_indices, f'SNR_{prefix}'] = snr_values
            
            successful_filters += 1
            
            # Estadísticas del filtro
            if f'MAG_{filter_name}_2' in results.columns:
                n_success = np.sum(results[f'MAG_{filter_name}_2'] < 50)
                logging.info(f"✅ {field_name} {filter_name}: {n_success}/{n_valid} valid measurements")
        
        if successful_filters > 0:
            results['PROCESSING_DATE'] = time.strftime('%Y-%m-%d %H:%M:%S')
            results['UNSHARP_APPLIED'] = 'YES' if self.apply_unsharp_to_all else 'NO'
            
            elapsed_time = time.time() - start_time
            logging.info(f"🎯 Field {field_name} completed: {successful_filters} filters, {len(results)} sources, {elapsed_time:.1f}s")
            return results
        else:
            return None

def main():
    parser = argparse.ArgumentParser(description='SPLUS Photometry - Corrected Final Version')
    parser.add_argument('--test', action='store_true', help='Modo TEST')
    parser.add_argument('--fields', nargs='+', default=['CenA11'])
    parser.add_argument('--filters', nargs='+', default=['F660', 'F861'])
    parser.add_argument('--all-fields', action='store_true', help='Procesar todos los campos')
    parser.add_argument('--no-unsharp', action='store_true', help='No aplicar unsharp')
    
    args = parser.parse_args()
    
    # Configuración
    if args.all_fields:
        test_mode = False
        fields_to_process = [f'CenA{i:02d}' for i in range(1, 25)]
    elif args.test:
        test_mode = True
        fields_to_process = args.fields
        test_filters = args.filters
    else:
        test_mode = True
        fields_to_process = ['CenA11']
        test_filters = ['F660', 'F861']
    
    apply_unsharp = not args.no_unsharp
    
    logging.info("==============================================")
    logging.info("🎯 S-PLUS PHOTOMETRY CORRECTED FINAL VERSION")
    logging.info(f"   Processing {len(fields_to_process)} fields")
    logging.info(f"   Unsharp mask: {apply_unsharp}")
    logging.info("==============================================")
    
    # Cargar datos
    catalog_path = "../TAP_1_J_MNRAS_3444_gc.csv"
    zeropoints_path = "Results/all_fields_zero_points_splus_format_3arcsec.csv"
    
    if not os.path.exists(catalog_path):
        logging.error(f"Catalog not found: {catalog_path}")
        return
    
    if not os.path.exists(zeropoints_path):
        logging.error(f"Zeropoints not found: {zeropoints_path}")
        return
    
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
    
    try:
        catalog = pd.read_csv(catalog_path)
        logging.info(f"📁 Loaded catalog with {len(catalog)} sources")
    except Exception as e:
        logging.error(f"Error loading catalog: {e}")
        return
    
    # Procesar campos
    photometry = SPLUSPhotometryCorrectedFinal(
        test_mode=test_mode, 
        apply_unsharp_to_all=apply_unsharp
    )
    
    all_results = []
    processed_fields = 0
    
    for field in fields_to_process:
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 PROCESSING: {field}")
            logging.info(f"{'='*60}")
            
            if test_mode:
                results = photometry.process_field(field, catalog, zeropoints, test_filters=test_filters)
            else:
                results = photometry.process_field(field, catalog, zeropoints)
            
            if results is not None:
                all_results.append(results)
                processed_fields += 1
                
                # Guardar individual
                suffix = "_TEST" if test_mode else "_COMPLETE"
                if not apply_unsharp:
                    suffix += "_no_unsharp"
                    
                output_file = f"{photometry.results_dir}/{field}_photometry{suffix}.csv"
                results.to_csv(output_file, index=False)
                logging.info(f"💾 Saved {output_file} ({len(results)} sources)")
                
            else:
                logging.warning(f"❌ No results for {field}")
                
        except Exception as e:
            logging.error(f"❌ Failed to process {field}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    # Combinar resultados
    if all_results:
        final_catalog = pd.concat(all_results, ignore_index=True)
        
        suffix = "_TEST" if test_mode else "_COMPLETE"
        if not apply_unsharp:
            suffix += "_no_unsharp"
            
        final_output = f"{photometry.results_dir}/all_fields_photometry{suffix}.csv"
        final_catalog.to_csv(final_output, index=False)
        
        # Estadísticas finales
        total_sources = len(final_catalog)
        unique_sources = len(final_catalog.drop_duplicates(subset=['RAJ2000', 'DEJ2000']))
        
        logging.info(f"\n{'='*60}")
        logging.info("🎉 PHOTOMETRY COMPLETED SUCCESSFULLY!")
        logging.info(f"📊 Processed: {processed_fields}/{len(fields_to_process)} fields")
        logging.info(f"📈 Final catalog: {total_sources} entries, {unique_sources} unique sources")
        logging.info(f"💾 Results in: {photometry.results_dir}/")
        logging.info(f"{'='*60}")
    
    else:
        logging.error("❌ No results generated!")

if __name__ == "__main__":
    main()
