#!/usr/bin/env python3
"""
Splus_photometry_buzzo_method_20_ISOLATED_GCs.py
VERSIÓN CON 20 GCs AISLADOS PARA CORRECCIÓN DE APERTURA MÁS ROBUSTA
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

def process_single_filter_20_isolated(args):
    """
    ✅ VERSIÓN CON 20 GCs AISLADOS PARA CORRECCIÓN DE APERTURA
    """
    # Desempaquetar argumentos
    (field_name, filter_name, valid_positions, valid_indices, 
     isolated_positions, zeropoints, pixel_scale, debug, 
     median_box_size, gaussian_sigma) = args
    
    try:
        logging.info(f"Starting {filter_name} processing")
        
        def find_image_file(field_name, filter_name):
            for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
                path = os.path.join(field_name, ext)
                if os.path.exists(path):
                    return path
            return None

        def find_weight_file(field_name, filter_name):
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

        def load_and_validate_weight_map(weight_path, data_shape):
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
                
                valid_weight = weight_data > 0
                if np.sum(valid_weight) / weight_data.size < 0.5:
                    return None
                
                error_map = np.full_like(weight_data, np.inf)
                error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
                return error_map
                
            except Exception as e:
                logging.error(f"Error loading weight map: {e}")
                return None

        def create_unsharp_mask(data, median_box_size=25, gaussian_sigma=5):
            """Unsharp masking optimizado"""
            try:
                median_filtered = median_filter(data, size=median_box_size)
                gaussian_smoothed = gaussian_filter(median_filtered, sigma=gaussian_sigma)
                unsharp_mask = data - gaussian_smoothed
                return unsharp_mask
            except Exception as e:
                logging.error(f"Error in unsharp masking: {e}")
                return data

        def calculate_aperture_correction_20_isolated(isolated_positions, data_unsharp, filter_name, 
                                                    pixel_scale, debug=False):
            """
            ✅ CORRECCIÓN DE APERTURA CON 20 GCs AISLADOS - MÁS ROBUSTA
            """
            try:
                # ✅ VALORES POR DEFECTO FÍSICOS
                default_corrections = {
                    'F378': (-0.25, -0.12),   # UV - más afectado por seeing
                    'F395': (-0.22, -0.10),   # UV 
                    'F410': (-0.20, -0.09),   # Azul
                    'F430': (-0.18, -0.08),   # Azul
                    'F515': (-0.15, -0.07),   # Verde
                    'F660': (-0.12, -0.06),   # Rojo  
                    'F861': (-0.10, -0.05)    # IR
                }
                
                # ✅ REQUERIR MÍNIMO 10 GCs AISLADOS PARA CÁLCULO
                if len(isolated_positions) < 10:
                    logging.warning(f"{filter_name}: Not enough isolated GCs ({len(isolated_positions)}), using default corrections")
                    return default_corrections.get(filter_name, (-0.15, -0.07))
                
                aperture_corrections_2 = []
                aperture_corrections_3 = []
                aperture_diams = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
                
                successful_gcs = 0
                
                for i, pos in enumerate(isolated_positions):
                    try:
                        fluxes = []  # Trabajar con flujos es más estable
                        valid_measurements = 0
                        
                        for ap_diam in aperture_diams:
                            ap_radius = (ap_diam / 2) / pixel_scale
                            bkg_inner = (3.0 / 2) / pixel_scale
                            bkg_outer = (4.0 / 2) / pixel_scale
                            
                            aperture = CircularAperture([pos], r=ap_radius)
                            annulus = CircularAnnulus([pos], r_in=bkg_inner, r_out=bkg_outer)
                            
                            # Fotometría
                            phot_table = aperture_photometry(data_unsharp, aperture)
                            raw_flux = phot_table['aperture_sum'].data[0]
                            
                            # Estimación de fondo robusta
                            try:
                                mask = annulus.to_mask(method='center')[0]
                                annulus_data = mask.multiply(data_unsharp)
                                annulus_data_1d = annulus_data[mask.data > 0]
                                
                                if len(annulus_data_1d) > 5:
                                    _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d, sigma=2.5, maxiters=3)
                                else:
                                    bkg_median = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                            except:
                                bkg_median = 0.0
                            
                            flux_net = raw_flux - (bkg_median * aperture.area)
                            
                            if flux_net > 1e-10:
                                fluxes.append(flux_net)
                                valid_measurements += 1
                            else:
                                fluxes.append(np.nan)
                        
                        # ✅ CRITERIO MÁS ESTRICTO CON MÁS GCs
                        if valid_measurements >= 4:
                            valid_indices = ~np.isnan(fluxes)
                            valid_diams = aperture_diams[valid_indices]
                            valid_fluxes = np.array(fluxes)[valid_indices]
                            
                            # Verificar comportamiento físico
                            if (valid_fluxes[0] < valid_fluxes[-1] and  # 2" < 6" en flujo
                                np.all(np.diff(valid_fluxes) > 0)):     # Monótonamente creciente
                                
                                try:
                                    # Convertir a magnitudes
                                    mag_2arcsec = -2.5 * np.log10(valid_fluxes[0])  # 2"
                                    mag_3arcsec = -2.5 * np.log10(valid_fluxes[1])  # 3" 
                                    mag_6arcsec = -2.5 * np.log10(valid_fluxes[-1]) # 6"
                                    
                                    # ✅ VERIFICACIONES FÍSICAS ESTRICTAS
                                    if (mag_2arcsec > mag_3arcsec > mag_6arcsec and  # Magnitudes decrecientes
                                        mag_2arcsec - mag_6arcsec < 0.8 and          # Diferencia máxima física
                                        mag_3arcsec - mag_6arcsec < 0.5):            # Diferencia máxima física
                                        
                                        correction_2 = mag_6arcsec - mag_2arcsec  # Negativa
                                        correction_3 = mag_6arcsec - mag_3arcsec  # Negativa
                                        
                                        # ✅ RANGOS FÍSICOS REALISTAS
                                        if (-0.6 <= correction_2 <= -0.05 and 
                                            -0.4 <= correction_3 <= -0.02):
                                            
                                            aperture_corrections_2.append(correction_2)
                                            aperture_corrections_3.append(correction_3)
                                            successful_gcs += 1
                                            
                                            if debug and successful_gcs <= 5:  # Log solo primeros 5 para no saturar
                                                logging.debug(f"GC {i}: Δm_2 = {correction_2:.3f}, Δm_3 = {correction_3:.3f}")
                                    
                                except Exception as e:
                                    if debug and i < 5:  # Log solo primeros errores
                                        logging.debug(f"Magnitude conversion failed for GC {i}: {e}")
                                    continue
                                    
                    except Exception as e:
                        if debug and i < 5:  # Log solo primeros errores
                            logging.debug(f"Error in GC {i}: {e}")
                        continue
                
                logging.info(f"{filter_name}: Valid GCs for aperture correction: {successful_gcs}/{len(isolated_positions)}")
                
                # ✅ CON 20 GCs, PODEMOS SER MÁS ESTRICTOS
                min_valid_gcs = max(5, len(isolated_positions) // 4)  # Mínimo 5 o 25% de los GCs
                
                if len(aperture_corrections_2) >= min_valid_gcs:
                    final_correction_2 = np.median(aperture_corrections_2)
                    final_correction_3 = np.median(aperture_corrections_3)
                    
                    # Calcular dispersión
                    std_correction_2 = np.std(aperture_corrections_2)
                    std_correction_3 = np.std(aperture_corrections_3)
                    
                    # ✅ VERIFICACIÓN DE CALIDAD CON MÁS GCs
                    if (abs(final_correction_2) > 0.8 or abs(final_correction_3) > 0.5 or
                        std_correction_2 > 0.2 or std_correction_3 > 0.15):  # Dispersión más estricta
                        logging.warning(f"{filter_name}: Corrections too large/inconsistent (std_2={std_correction_2:.3f}, std_3={std_correction_3:.3f}), using defaults")
                        return default_corrections.get(filter_name, (-0.15, -0.07))
                    
                    logging.info(f"{filter_name}: Aperture correction 2\" = {final_correction_2:.3f} ± {std_correction_2:.3f} mag, "
                               f"3\" = {final_correction_3:.3f} ± {std_correction_3:.3f} mag (based on {len(aperture_corrections_2)} GCs)")
                    return final_correction_2, final_correction_3
                else:
                    logging.warning(f"{filter_name}: Only {len(aperture_corrections_2)} valid GCs (min required: {min_valid_gcs}), using default corrections")
                    return default_corrections.get(filter_name, (-0.15, -0.07))
                    
            except Exception as e:
                logging.error(f"Error in aperture correction for {filter_name}: {e}")
                return default_corrections.get(filter_name, (-0.15, -0.07))

        def perform_photometry_robust(data, positions, apertures, annulus, error_map=None):
            """Fotometría robusta"""
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
                
                if error_map is not None:
                    error_flux_from_weights = np.zeros(len(positions))
                    for i, pos in enumerate(positions):
                        try:
                            aperture_mask = apertures.to_mask(method='center')[i]
                            error_in_aperture = error_map * aperture_mask.data
                            error_flux_from_weights[i] = np.sqrt(np.sum(error_in_aperture**2))
                        except:
                            valid_errors = error_map[error_map < np.inf]
                            if len(valid_errors) > 0:
                                error_flux_from_weights[i] = np.sqrt(apertures.area) * np.median(valid_errors)
                            else:
                                error_flux_from_weights[i] = np.sqrt(apertures.area)
                    
                    net_flux_err = np.sqrt(error_flux_from_weights**2 + (apertures.area * 0.1**2))
                else:
                    net_flux_err = np.sqrt(np.abs(net_flux) + (apertures.area * 0.1**2))
                
                return net_flux, net_flux_err
                
            except Exception as e:
                logging.error(f"Error in robust photometry: {e}")
                n = len(positions)
                return np.zeros(n), np.full(n, 99.0)

        # Cargar imagen
        image_path = find_image_file(field_name, filter_name)
        if not image_path:
            logging.warning(f"{filter_name}: Image not found")
            return None, filter_name
        
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data_original = hdu.data.astype(float)
                    break
            else:
                return None, filter_name
        
        # Cargar weight map
        weight_path = find_weight_file(field_name, filter_name)
        error_map = None
        if weight_path and debug:
            error_map = load_and_validate_weight_map(weight_path, data_original.shape)

        # ✅ 1. APLICAR UNSHARP MASKING
        data_unsharp = create_unsharp_mask(data_original, median_box_size, gaussian_sigma)
        logging.info(f"✅ {filter_name}: Unsharp masking applied")

        # ✅ 2. CALCULAR CORRECCIÓN DE APERTURA CON 20 GCs AISLADOS
        aperture_correction_2, aperture_correction_3 = calculate_aperture_correction_20_isolated(
            isolated_positions, data_unsharp, filter_name, pixel_scale, debug)

        # ✅ 3. FOTOMETRÍA
        aperture_diams = [2.0, 3.0]
        results = {'indices': valid_indices}
        
        for aperture_diam in aperture_diams:
            aperture_radius = (aperture_diam / 2) / pixel_scale
            annulus_inner = (3.0 / 2) / pixel_scale
            annulus_outer = (4.0 / 2) / pixel_scale
            
            apertures = CircularAperture(valid_positions, r=aperture_radius)
            annulus = CircularAnnulus(valid_positions, r_in=annulus_inner, r_out=annulus_outer)
            
            net_flux, net_flux_err = perform_photometry_robust(
                data_unsharp, valid_positions, apertures, annulus, error_map)
            
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            
            zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
            min_flux = 1e-10
            valid_mask = (net_flux > min_flux) & (net_flux_err > 0) & np.isfinite(net_flux)
            
            mag_inst = np.where(valid_mask, -2.5 * np.log10(net_flux), 99.0)
            
            if aperture_diam == 2.0:
                aperture_correction = aperture_correction_2
            else:
                aperture_correction = aperture_correction_3
                
            mag = np.where(valid_mask, mag_inst + zero_point + aperture_correction, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
            
            reasonable = (mag >= 10.0) & (mag <= 30.0) & (mag_err <= 5.0)
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
        
        logging.info(f"✅ {filter_name}: Completed - {np.sum(final_mask)} valid, "
                   f"AP corr: 2\"={aperture_correction_2:.3f}, 3\"={aperture_correction_3:.3f}")
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {filter_name}: Processing failed - {e}")
        traceback.print_exc()
        return None, filter_name

class SPLUSGCPhotometryBuzzo20Isolated:
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
        
        # CONFIGURACIÓN
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55
        self.debug = debug
        self.n_workers = n_workers or 1
        
        # Parámetros unsharp masking
        self.median_box_size = 25
        self.gaussian_sigma = 5
        
        # Mapeo de columnas
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        logging.info(f"Using 20 ISOLATED GCs method for robust aperture correction")

    def find_isolated_gcs(self, positions, image_shape, n_isolated=20):
        """
        ✅ ENCONTRAR 20 GCs AISLADOS (O MÁXIMO POSIBLE)
        """
        try:
            height, width = image_shape
            
            # ✅ ESTRATEGIA PARA MAXIMIZAR GCs AISLADOS:
            # 1. Primero buscar en regiones muy externas (20% del borde)
            margin_outer = 0.20
            center_x, center_y = width / 2, height / 2
            distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
            max_distance = np.max(distances)
            
            isolated_mask_outer = distances > (1 - margin_outer) * max_distance
            isolated_positions_outer = positions[isolated_mask_outer]
            
            # 2. Si no hay suficientes, relajar criterio a 15%
            if len(isolated_positions_outer) < n_isolated:
                margin_middle = 0.15
                isolated_mask_middle = distances > (1 - margin_middle) * max_distance
                isolated_positions_middle = positions[isolated_mask_middle]
                
                # Combinar posiciones únicas
                all_isolated = np.unique(np.vstack([isolated_positions_outer, isolated_positions_middle]), axis=0)
            else:
                all_isolated = isolated_positions_outer
            
            # 3. Si todavía no hay suficientes, usar 10%
            if len(all_isolated) < n_isolated:
                margin_inner = 0.10
                isolated_mask_inner = distances > (1 - margin_inner) * max_distance
                isolated_positions_inner = positions[isolated_mask_inner]
                all_isolated = np.unique(np.vstack([all_isolated, isolated_positions_inner]), axis=0)
            
            # 4. Ordenar por distancia (más lejanos primero) y tomar hasta n_isolated
            if len(all_isolated) > 0:
                isolated_distances = np.sqrt((all_isolated[:, 0] - center_x)**2 + (all_isolated[:, 1] - center_y)**2)
                sorted_indices = np.argsort(-isolated_distances)
                isolated_positions = all_isolated[sorted_indices[:min(n_isolated, len(all_isolated))]]
            else:
                isolated_positions = np.array([])
            
            logging.info(f"Selected {len(isolated_positions)} isolated GCs for aperture correction (target: {n_isolated})")
            return isolated_positions
            
        except Exception as e:
            logging.error(f"Error finding isolated GCs: {e}")
            return positions[:min(n_isolated, len(positions))]

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

    def process_field_20_isolated(self, field_name):
        """Procesar campo completo - CON 20 GCs AISLADOS"""
        logging.info(f"🎯 Processing field {field_name} (20 ISOLATED GCs METHOD - {self.n_workers} workers)")
        
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

        # ✅ IDENTIFICAR 20 GCs AISLADOS
        isolated_positions = self.find_isolated_gcs(valid_positions, (height, width), n_isolated=20)

        # ✅ PROCESAMIENTO CON 20 GCs AISLADOS
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
                isolated_positions,
                self.zeropoints,
                self.pixel_scale,
                self.debug,
                self.median_box_size,
                self.gaussian_sigma
            )
            args_list.append(args)
        
        # Procesamiento paralelo o serial
        if self.n_workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [executor.submit(process_single_filter_20_isolated, args) for args in args_list]
                    
                    for future in as_completed(futures):
                        try:
                            result, filter_name = future.result(timeout=300)
                            if result is not None:
                                self._integrate_results(results_df, result, filter_name)
                                successful_filters += 1
                                logging.info(f"✅ {filter_name}: Results integrated")
                        except Exception as e:
                            logging.error(f"❌ Error in future: {e}")
            except Exception as e:
                logging.error(f"Parallel failed, using serial: {e}")
                successful_filters = self._process_serial(args_list, results_df)
        else:
            successful_filters = self._process_serial(args_list, results_df)
        
        elapsed_time = time.time() - start_time
        logging.info(f"🎯 Field {field_name} completed: {successful_filters}/{len(self.filters)} filters "
                   f"in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            return results_df
        else:
            return None

    def _process_serial(self, args_list, results_df):
        """Procesamiento serial"""
        successful_filters = 0
        for args in tqdm(args_list, desc="Processing filters"):
            result, filter_name = process_single_filter_20_isolated(args)
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
    
    # Crear instancia con 20 GCs aislados
    photometry = SPLUSGCPhotometryBuzzo20Isolated(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True,
        n_workers=7
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_20_isolated(field)
        if results is not None and len(results) > 0:
            all_results.append(results)
            
            output_file = f'{field}_gc_photometry_20_isolated.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results to {output_file}")
            
            # Estadísticas del campo
            logging.info(f"📊 Field {field} statistics:")
            for filter_name in photometry.filters:
                for aperture in ['2', '3']:
                    mag_col = f'MAG_{filter_name}_{aperture}'
                    ap_corr_col = f'AP_CORR_{filter_name}_{aperture}'
                    if mag_col in results.columns and ap_corr_col in results.columns:
                        valid_count = len(results[results[mag_col] < 99.0])
                        if len(results) > 0:
                            ap_corr_values = results[ap_corr_col].unique()
                            ap_corr = ap_corr_values[0] if len(ap_corr_values) > 0 else 0.0
                            logging.info(f"  {filter_name} {aperture}\": {valid_count} valid, AP corr={ap_corr:.3f}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_20_isolated.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        # Estadísticas finales
        logging.info("\n📊 ESTADÍSTICAS FINALES (20 ISOLATED GCs METHOD):")
        for filter_name in photometry.filters:
            for aperture in ['2', '3']:
                mag_col = f'MAG_{filter_name}_{aperture}'
                if mag_col in final_results.columns:
                    valid_mags = final_results[mag_col][final_results[mag_col] < 99.0]
                    if len(valid_mags) > 0:
                        median_mag = np.median(valid_mags)
                        std_mag = np.std(valid_mags)
                        logging.info(f"  {filter_name} {aperture}\": {len(valid_mags)} valid, "
                                   f"median={median_mag:.2f} ± {std_mag:.2f}")

if __name__ == "__main__":
    # Configuración segura para multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
