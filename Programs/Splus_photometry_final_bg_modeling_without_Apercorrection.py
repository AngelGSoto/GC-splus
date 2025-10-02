#!/usr/bin/env python3
"""
Splus_photometry_buzzo_method_PARALLEL_WEIGHTS.py
VERSIÓN PARALELIZADA CON WEIGHT MAPS Y MÉTODO BUZZO CORREGIDO
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm
import warnings
import os
import logging
from scipy.interpolate import interp1d
import time
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSGCPhotometryBuzzoParallel:
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
        
        # Cargar catálogo de cúmulos globulares
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        # ✅ CONFIGURACIÓN BUZZO 2022
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55
        self.aperture_diam_photometry = 2.0  # ✅ 2" para fotometría (como Buzzo)
        self.aperture_diam_reference = 6.0   # ✅ 6" como referencia (como Buzzo)
        self.debug = debug
        self.n_workers = n_workers or min(len(self.filters), mp.cpu_count() - 1)
        
        # Parámetros para DETECCIÓN (no para fotometría)
        self.median_box_size = 25
        self.gaussian_sigma = 5
        
        # Mapeo de columnas del catálogo
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        logging.info(f"Using Buzzo et al. 2022 methodology with {self.n_workers} parallel workers")

    def find_weight_file(self, field_name, filter_name):
        """Buscar archivos de peso"""
        patterns = [
            f"{field_name}_{filter_name}.weight.fits.fz",
            f"{field_name}_{filter_name}.weight.fits",
            f"{field_name}_{filter_name}_weight.fits.fz",
            f"{field_name}_{filter_name}_weight.fits"
        ]
        for p in patterns:
            path = os.path.join(field_name, p)
            if os.path.exists(path):
                return path
        return None

    def load_and_validate_weight_map(self, weight_path, data_shape):
        """Cargar y validar mapa de pesos"""
        try:
            with fits.open(weight_path) as whdul:
                for whdu in whdul:
                    if whdu.data is not None:
                        weight_data = whdu.data.astype(float)
                        break
                else:
                    return None
            
            if weight_data.shape != data_shape:
                logging.warning(f"Weight map shape {weight_data.shape} doesn't match data shape {data_shape}")
                return None
            
            valid_weight = weight_data > 0
            if np.sum(valid_weight) / weight_data.size < 0.5:
                logging.warning("Less than 50% of weight map is valid")
                return None
            
            # ✅ CALCULAR ERROR MAP CORRECTAMENTE
            error_map = np.full_like(weight_data, np.inf)  # Inf para píxeles inválidos
            error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
            
            logging.info(f"Weight map loaded: valid pixels = {np.sum(valid_weight)/weight_data.size*100:.1f}%")
            return error_map
            
        except Exception as e:
            logging.error(f"Error loading weight map {weight_path}: {e}")
            return None

    def create_detection_image(self, field_name):
        """Crear imagen de detección combinando g, r, i, z como Buzzo"""
        try:
            detection_filters = ['F515', 'F660', 'F861']  # g, r, i aproximados
            
            combined_image = None
            valid_filters = 0
            
            for filter_name in detection_filters:
                image_path = self.find_image_file(field_name, filter_name)
                if image_path and os.path.exists(image_path):
                    with fits.open(image_path) as hdul:
                        for hdu in hdul:
                            if hdu.data is not None:
                                data = hdu.data.astype(float)
                                if combined_image is None:
                                    combined_image = np.zeros_like(data)
                                combined_image += data
                                valid_filters += 1
                                break
            
            if valid_filters == 0:
                logging.warning("Could not create detection image, using first available filter")
                for filter_name in self.filters:
                    image_path = self.find_image_file(field_name, filter_name)
                    if image_path:
                        with fits.open(image_path) as hdul:
                            for hdu in hdul:
                                if hdu.data is not None:
                                    return hdu.data.astype(float)
                return None
            
            # Normalizar
            combined_image = combined_image / valid_filters
            
            # ✅ Aplicar unsharp masking SOLO para detección
            median_filtered = median_filter(combined_image, size=self.median_box_size)
            gaussian_smoothed = gaussian_filter(median_filtered, sigma=self.gaussian_sigma)
            detection_image = combined_image - gaussian_smoothed
            
            logging.info(f"Created detection image from {valid_filters} filters")
            return detection_image
            
        except Exception as e:
            logging.error(f"Error creating detection image: {e}")
            return None

    def find_isolated_gcs(self, positions, image_shape, n_isolated=5):
        """Encontrar GCs aislados para corrección de apertura"""
        try:
            height, width = image_shape
            margin = 0.15  # 15% del borde
            
            center_x, center_y = width / 2, height / 2
            distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
            
            max_distance = np.max(distances)
            isolated_mask = distances > (1 - margin) * max_distance
            
            isolated_positions = positions[isolated_mask]
            
            # Ordenar por distancia (más lejanos primero)
            if len(isolated_positions) > n_isolated:
                isolated_distances = distances[isolated_mask]
                sorted_indices = np.argsort(-isolated_distances)
                isolated_positions = isolated_positions[sorted_indices[:n_isolated]]
            
            logging.info(f"Selected {len(isolated_positions)} isolated GCs for aperture correction")
            return isolated_positions
            
        except Exception as e:
            logging.error(f"Error finding isolated GCs: {e}")
            return positions[:min(n_isolated, len(positions))]

    def calculate_aperture_correction_buzzo(self, isolated_positions, data_original, filter_name, error_map=None):
        """
        ✅ IMPLEMENTACIÓN EXACTA DE BUZZO 2022 CON WEIGHT MAPS
        """
        try:
            if len(isolated_positions) < 3:
                logging.warning(f"{filter_name}: Not enough isolated GCs for aperture correction")
                return 0.0
            
            aperture_corrections = []
            aperture_diams = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0])
            
            for i, pos in enumerate(isolated_positions):
                try:
                    magnitudes = []
                    
                    for ap_diam in aperture_diams:
                        ap_radius = (ap_diam / 2) / self.pixel_scale
                        bkg_inner = (3.0 / 2) / self.pixel_scale
                        bkg_outer = (4.0 / 2) / self.pixel_scale
                        
                        aperture = CircularAperture([pos], r=ap_radius)
                        annulus = CircularAnnulus([pos], r_in=bkg_inner, r_out=bkg_outer)
                        
                        # Fotometría en imagen ORIGINAL con error propagation
                        flux, flux_err = self.perform_photometry_with_errors(
                            data_original, [pos], aperture, annulus, error_map)
                        
                        if flux[0] > 1e-10 and flux_err[0] > 0:
                            mag = -2.5 * np.log10(flux[0])
                            magnitudes.append(mag)
                        else:
                            magnitudes.append(np.nan)
                    
                    # Encontrar plateau alrededor de 6" como Buzzo
                    valid_indices = ~np.isnan(magnitudes)
                    if np.sum(valid_indices) >= 5:
                        valid_diams = aperture_diams[valid_indices]
                        valid_mags = np.array(magnitudes)[valid_indices]
                        
                        # Identificar radio de plateau (≈6")
                        plateau_radius = 6.0
                        idx_2arcsec = np.argmin(np.abs(valid_diams - 2.0))
                        idx_plateau = np.argmin(np.abs(valid_diams - plateau_radius))
                        
                        if (idx_2arcsec < len(valid_mags) and 
                            idx_plateau < len(valid_mags)):
                            
                            mag_2arcsec = valid_mags[idx_2arcsec]
                            mag_plateau = valid_mags[idx_plateau]
                            
                            # ✅ FÓRMULA CORREGIDA: 2" es más débil, corrección POSITIVA
                            correction = mag_plateau - mag_2arcsec
                            
                            if 0.0 <= correction <= 1.5:  # Rango físico razonable
                                aperture_corrections.append(correction)
                                if self.debug:
                                    logging.debug(f"GC {i}: Δm = {correction:.3f} mag")
                                
                except Exception as e:
                    logging.debug(f"Error in GC {i} growth curve: {e}")
                    continue
            
            if len(aperture_corrections) >= 2:
                final_correction = np.median(aperture_corrections)
                logging.info(f"{filter_name}: Aperture correction = {final_correction:.3f} mag "
                           f"(based on {len(aperture_corrections)} GCs)")
                return final_correction
            else:
                logging.warning(f"{filter_name}: Could not determine reliable aperture correction")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error in aperture correction: {e}")
            return 0.0

    def perform_photometry_with_errors(self, data, positions, apertures, annulus, error_map=None):
        """
        ✅ FOTOMETRÍA CON PROPAGACIÓN CORRECTA DE ERRORES USANDO WEIGHT MAPS
        """
        try:
            # Fotometría básica
            phot_table = aperture_photometry(data, apertures)
            raw_flux = phot_table['aperture_sum'].data
            
            # Estimación robusta de fondo
            bkg_median_per_source = np.zeros(len(positions))
            bkg_std_per_source = np.zeros(len(positions))
            
            for i, pos in enumerate(positions):
                try:
                    mask = annulus.to_mask(method='center')[i]
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    
                    if len(annulus_data_1d) > 10:
                        _, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)
                        bkg_median_per_source[i] = bkg_median
                        bkg_std_per_source[i] = bkg_std
                    else:
                        bkg_median_per_source[i] = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                        bkg_std_per_source[i] = np.std(annulus_data_1d) if len(annulus_data_1d) > 0 else 1.0
                except:
                    bkg_median_per_source[i] = 0.0
                    bkg_std_per_source[i] = 1.0
            
            # Flujo neto
            net_flux = raw_flux - (bkg_median_per_source * apertures.area)
            
            # ✅ PROPAGACIÓN DE ERRORES MEJORADA
            if error_map is not None:
                # Error desde weight map
                error_flux_from_weights = np.zeros(len(positions))
                for i, pos in enumerate(positions):
                    try:
                        aperture_mask = apertures.to_mask(method='center')[i]
                        error_in_aperture = error_map * aperture_mask.data
                        # Suma en cuadratura de los errores en la apertura
                        error_flux_from_weights[i] = np.sqrt(np.sum(error_in_aperture**2))
                    except:
                        error_flux_from_weights[i] = np.sqrt(apertures.area) * np.median(error_map[error_map < np.inf])
                
                # Error del fondo
                bkg_error = apertures.area * (bkg_std_per_source**2)
                
                # Error total
                net_flux_err = np.sqrt(error_flux_from_weights**2 + bkg_error)
            else:
                # Error estándar (Poisson + fondo)
                net_flux_err = np.sqrt(np.abs(net_flux) + apertures.area * (bkg_std_per_source**2))
            
            return net_flux, net_flux_err
            
        except Exception as e:
            logging.error(f"Error in photometry with errors: {e}")
            n = len(positions)
            return np.zeros(n), np.full(n, 99.0)

    def process_single_filter(self, args):
        """
        ✅ FUNCIÓN PARA PROCESAMIENTO PARALELO DE UN FILTRO
        """
        field_name, filter_name, valid_positions, valid_indices, isolated_positions = args
        
        try:
            logging.info(f"Starting {filter_name} processing")
            
            # Cargar imagen y weight map
            image_path = self.find_image_file(field_name, filter_name)
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
            
            # Cargar weight map
            weight_path = self.find_weight_file(field_name, filter_name)
            error_map = None
            if weight_path:
                error_map = self.load_and_validate_weight_map(weight_path, data_original.shape)
            else:
                logging.warning(f"{filter_name}: No weight map found")
            
            # Calcular corrección de apertura
            aperture_correction = self.calculate_aperture_correction_buzzo(
                isolated_positions, data_original, filter_name, error_map)
            
            # Preparar aperturas para fotometría
            aperture_radius = (self.aperture_diam_photometry / 2) / self.pixel_scale
            annulus_inner = (3.0 / 2) / self.pixel_scale
            annulus_outer = (4.0 / 2) / self.pixel_scale
            
            apertures = CircularAperture(valid_positions, r=aperture_radius)
            annulus = CircularAnnulus(valid_positions, r_in=annulus_inner, r_out=annulus_outer)
            
            # Fotometría con errores
            flux, flux_err = self.perform_photometry_with_errors(
                data_original, valid_positions, apertures, annulus, error_map)
            
            # Calcular SNR
            snr = np.where(flux_err > 0, flux / flux_err, 0.0)
            
            # Calcular magnitudes
            zero_point = self.zeropoints.get(field_name, {}).get(filter_name, 0.0)
            min_flux = 1e-10
            valid_mask = (flux > min_flux) & (flux_err > 0) & np.isfinite(flux) & np.isfinite(flux_err)
            
            mag_inst = np.where(valid_mask, -2.5 * np.log10(flux), 99.0)
            mag = np.where(valid_mask, mag_inst + zero_point + aperture_correction, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
            
            # Validación final
            reasonable_mag = (mag >= 10.0) & (mag <= 30.0)
            reasonable_err = (mag_err >= 0.0) & (mag_err <= 5.0)
            final_mask = valid_mask & reasonable_mag & reasonable_err
            
            mag = np.where(final_mask, mag, 99.0)
            mag_err = np.where(final_mask, mag_err, 99.0)
            snr = np.where(final_mask, snr, 0.0)
            
            # Preparar resultados
            results = {
                'indices': valid_indices,
                f'FLUX_{filter_name}_2': flux,
                f'FLUXERR_{filter_name}_2': flux_err,
                f'MAG_{filter_name}_2': mag,
                f'MAGERR_{filter_name}_2': mag_err,
                f'SNR_{filter_name}_2': snr,
                f'AP_CORR_{filter_name}_2': np.full(len(flux), aperture_correction),
                f'QUALITY_{filter_name}_2': final_mask.astype(int)
            }
            
            logging.info(f"✅ {filter_name}: Completed - {np.sum(final_mask)}/{len(flux)} valid measurements")
            return results, filter_name
            
        except Exception as e:
            logging.error(f"❌ {filter_name}: Processing failed - {e}")
            return None, filter_name

    def find_image_file(self, field_name, filter_name):
        """Buscar archivos de imagen"""
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None

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

    def process_field_parallel(self, field_name):
        """Procesar campo completo en PARALELO"""
        logging.info(f"🎯 Processing field {field_name} (PARALLEL METHOD - {self.n_workers} workers)")
        
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

        # ✅ CREAR IMAGEN DE DETECCIÓN Y ENCONTRAR GCs AISLADOS
        detection_image = self.create_detection_image(field_name)
        if detection_image is not None:
            isolated_positions = self.find_isolated_gcs(valid_positions, detection_image.shape)
        else:
            isolated_positions = valid_positions[:5]  # Fallback
            logging.warning("Using fallback isolated positions")

        # ✅ PROCESAMIENTO PARALELO DE FILTROS
        results_df = field_sources.copy()
        successful_filters = 0
        
        # Preparar argumentos para paralelización
        args_list = [(field_name, filt, valid_positions, valid_indices, isolated_positions) 
                    for filt in self.filters]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_filter = {
                executor.submit(self.process_single_filter, args): args[1] 
                for args in args_list
            }
            
            for future in as_completed(future_to_filter):
                filter_name = future_to_filter[future]
                try:
                    result, filt_name = future.result()
                    if result is not None:
                        # Combinar resultados
                        temp_df = pd.DataFrame(result)
                        temp_df.set_index('indices', inplace=True)
                        
                        for col in temp_df.columns:
                            results_df.loc[temp_df.index, col] = temp_df[col].values
                        
                        successful_filters += 1
                        logging.info(f"✅ {filter_name}: Results integrated successfully")
                    else:
                        logging.warning(f"❌ {filter_name}: No results to integrate")
                        
                except Exception as e:
                    logging.error(f"❌ {filter_name}: Parallel processing failed - {e}")
        
        # Estadísticas finales
        elapsed_time = time.time() - start_time
        logging.info(f"🎯 Field {field_name} completed: {successful_filters}/{len(self.filters)} filters "
                   f"in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            return results_df
        else:
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
    
    # Crear instancia paralelizada
    photometry = SPLUSGCPhotometryBuzzoParallel(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True,
        n_workers=4  # Ajustar según tu CPU
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_parallel(field)
        if results is not None and len(results) > 0:
            all_results.append(results)
            
            output_file = f'{field}_gc_photometry_parallel.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results to {output_file}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_parallel_weight_corrected.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        # Estadísticas finales
        logging.info("\n📊 ESTADÍSTICAS FINALES (PARALLEL + WEIGHT MAPS):")
        total_sources = len(final_results)
        measured_sources = len(final_results[final_results['FIELD'].notna()])
        logging.info(f"Total sources in catalog: {len(photometry.original_catalog)}")
        logging.info(f"Measured sources: {measured_sources}")
        
        # Verificar calidad por filtro
        for filter_name in photometry.filters:
            mag_col = f'MAG_{filter_name}_2'
            if mag_col in final_results.columns:
                valid_mags = final_results[mag_col][final_results[mag_col] < 99.0]
                if len(valid_mags) > 0:
                    median_mag = np.median(valid_mags)
                    std_mag = np.std(valid_mags)
                    logging.info(f"  {filter_name}: {len(valid_mags)} valid, median={median_mag:.2f} ± {std_mag:.2f}")

if __name__ == "__main__":
    # Configuración para multiprocessing en Windows
    mp.freeze_support()
    main()
