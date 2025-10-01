#!/usr/bin/env python3
"""
Splus_photometry_final_coherent_FIXED_v4.py
VERSIÓN CORREGIDA - Usa correcciones de apertura GUARDADAS del Script 1 y unsharp mask idéntico
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
from tqdm import tqdm
import warnings
import os
import logging
from scipy import ndimage
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSGCPhotometryCoherent:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # ✅ Cargar zero points COHERENTE con Script 2
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
        logging.info(f"Loaded GC catalog with {len(self.catalog)} sources")
        
        # Configuración COHERENTE con Script 1
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel
        self.apertures = [3, 4, 5, 6]  # diámetros en arcsec
        self.debug = debug
        self.debug_filter = debug_filter
        
        # Mapeo de columnas del catálogo de Taylor
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        logging.info(f"Using columns: RA={self.ra_col}, DEC={self.dec_col}, ID={self.id_col}")

    def load_aperture_corrections_from_script1(self, field_name):
        """
        ✅ CARGA CORRECCIONES DEL SCRIPT 1 - desde el archivo CSV
        El Script 1 GUARDA las correcciones en: 'aper_corr_3.0_{filter_name}'
        """
        try:
            # ✅ ARCHIVO REAL del Script 1
            ref_catalog_path = f"{field_name}_gaia_xp_matches_splus_practical.csv"
            
            if not os.path.exists(ref_catalog_path):
                logging.warning(f"❌ No aperture correction file found: {ref_catalog_path}")
                return self._get_default_corrections()
            
            ref_df = pd.read_csv(ref_catalog_path)
            logging.info(f"✅ Loaded aperture corrections from: {ref_catalog_path}")
            
            # ✅ VERIFICAR COLUMNAS DE CORRECCIÓN
            available_columns = ref_df.columns.tolist()
            ac_columns = [col for col in available_columns if 'aper_corr' in col]
            logging.info(f"Available aperture correction columns: {ac_columns}")
            
            # ✅ CARGAR CORRECCIONES PARA CADA FILTRO
            corrections = {}
            for filter_name in self.filters:
                # ✅ COLUMNA EXACTA que usa el Script 1
                ac_col = f'aper_corr_3.0_{filter_name}'
                
                if ac_col in ref_df.columns:
                    # Filtrar valores válidos (rango físico)
                    valid_ac = ref_df[
                        (ref_df[ac_col] >= 0.0) & 
                        (ref_df[ac_col] <= 2.0)
                    ][ac_col]
                    
                    if len(valid_ac) > 0:
                        # Usar la mediana de todas las correcciones válidas
                        ac_value = float(valid_ac.median())
                        n_stars = len(valid_ac)
                        logging.info(f"✅ {filter_name}: Aperture correction = {ac_value:.3f} mag (from {n_stars} stars)")
                    else:
                        ac_value = 0.0
                        logging.warning(f"❌ {filter_name}: No valid aperture corrections")
                else:
                    ac_value = 0.0
                    logging.warning(f"❌ {filter_name}: Aperture correction column '{ac_col}' not found")
                
                # ✅ PARA TODAS LAS APERTURAS, usar la misma corrección de 3"
                # Esto es una aproximación coherente con el Script 1
                filter_corrections = {}
                for aperture_size in self.apertures:
                    filter_corrections[aperture_size] = ac_value
                
                corrections[filter_name] = filter_corrections
            
            return corrections
            
        except Exception as e:
            logging.error(f"Error loading aperture corrections from Script 1: {e}")
            return self._get_default_corrections()

    def _get_default_corrections(self):
        """Correcciones por defecto - COHERENTE con Script 1"""
        default = {}
        # ✅ VALORES BASE del Script 1 como fallback
        default_corrections = {
            'F378': 0.15, 'F395': 0.16, 'F410': 0.17, 
            'F430': 0.18, 'F515': 0.20, 'F660': 0.22, 'F861': 0.25
        }
        
        for filt in self.filters:
            default[filt] = {}
            base_corr = default_corrections.get(filt, 0.20)
            for aperture_size in self.apertures:
                default[filt][aperture_size] = base_corr
        
        logging.warning("⚠️ Using DEFAULT aperture corrections (Script 1 base values)")
        return default

    def detect_galaxy_structure_identical(self, data, filter_name):
        """
        ✅ DETECCIÓN IDÉNTICA al Script 1
        """
        try:
            # Estadísticas básicas (igual al Script 1)
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            height, width = data.shape
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            
            # ✅ ANÁLISIS RADIAL IDÉNTICO al Script 1
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_r = min(center_x, center_y)
            
            radial_bins = np.linspace(0, max_r, 20)  # ✅ 20 bins como Script 1
            radial_profile = []
            
            for i in range(len(radial_bins) - 1):
                mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
                if np.sum(mask) > 100:  # ✅ Mismo umbral que Script 1
                    radial_profile.append(np.median(data[mask]))
                else:
                    radial_profile.append(0.0)
            
            radial_profile = np.array(radial_profile)
            
            if len(radial_profile) > 5:
                # ✅ MISMO CÁLCULO que Script 1
                center_brightness = np.mean(radial_profile[:5])
                outer_brightness = np.mean(radial_profile[-5:])
                brightness_gradient = center_brightness - outer_brightness
                
                # ✅ MISMO CRITERIO que Script 1
                has_structure = brightness_gradient > 2 * std
                
                if has_structure:
                    logging.info(f"{filter_name}: Galaxy structure detected (gradient: {brightness_gradient:.6f})")
                    return True, center_x, center_y, max_r * 0.7  # ✅ 70% como Script 1
                else:
                    logging.info(f"{filter_name}: No significant galaxy structure")
                    return False, center_x, center_y, 0
            else:
                return False, data.shape[1] // 2, data.shape[0] // 2, 0
                
        except Exception as e:
            logging.warning(f"Error in galaxy structure detection: {e}")
            return False, data.shape[1] // 2, data.shape[0] // 2, 0

    def apply_unsharp_mask_identical(self, data, positions, center_x, center_y, structure_radius, filter_name):
        """
        ✅ UNSHARP MASK IDÉNTICO al Script 1
        """
        try:
            # ✅ CALCULAR DISTANCIAS COMO EN SCRIPT 1
            distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
            near_structure = distances < structure_radius
            n_near = np.sum(near_structure)
            
            if n_near == 0:
                logging.info(f"{filter_name}: No stars near structure, no unsharp mask")
                return data, False
            
            logging.info(f"{filter_name}: Applying unsharp mask to {n_near} stars near galaxy structure")
            
            # ✅ PROCESAMIENTO IDÉNTICO al Script 1
            data_clean = data.copy()
            data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ✅ MISMO CÁLCULO DE SIGMA: max(15.0, structure_radius / 20.0)
            sigma = max(15.0, structure_radius / 20.0)
            smoothed = gaussian_filter(data_clean, sigma=sigma)
            
            # ✅ MISMA FÓRMULA: data - 0.7 * smoothed
            data_unsharp = data_clean - 0.7 * smoothed
            
            logging.info(f"{filter_name}: Unsharp mask applied (sigma={sigma:.1f})")
            return data_unsharp, True
            
        except Exception as e:
            logging.error(f"Error in unsharp mask: {e}")
            return data, False

    def process_image_with_galaxy_correction_identical(self, data, positions, filter_name):
        """
        ✅ PROCESAMIENTO IDÉNTICO al Script 1
        """
        # ✅ DETECCIÓN IDÉNTICA
        has_structure, center_x, center_y, structure_radius = self.detect_galaxy_structure_identical(data, filter_name)
        
        use_unsharp = False
        data_processed = data.copy()
        
        if has_structure and structure_radius > 0:
            # ✅ UNSHARP MASK IDÉNTICO
            data_processed, use_unsharp = self.apply_unsharp_mask_identical(
                data, positions, center_x, center_y, structure_radius, filter_name)
        else:
            logging.info(f"{filter_name}: No galaxy structure, normal processing")
        
        return data_processed, use_unsharp, has_structure

    def apply_background_correction_coherent(self, data, filter_name):
        """Corrección de fondo - COHERENTE con Script 1"""
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            # Si la mediana es muy cercana a cero, asumir que el fondo ya está restado
            if abs(median) < 0.05 * std:
                logging.info(f"{filter_name}: Background already subtracted (median={median:.6f})")
                return data, std
            
            # ✅ PARÁMETROS COHERENTES
            box_size = int(50.0 / self.pixel_scale)  # 50 arcsec
            box_size = max(25, min(box_size, min(data.shape) // 10))
            
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            
            # Máscara para excluir fuentes brillantes
            threshold = median + 5 * std
            mask = data > threshold
            if np.sum(mask) > 0:
                dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            else:
                dilated_mask = None
            
            bkg = Background2D(
                data, 
                box_size=box_size, 
                filter_size=3,
                sigma_clip=sigma_clip, 
                bkg_estimator=bkg_estimator,
                mask=dilated_mask,
                exclude_percentile=10
            )
            
            data_corr = data - bkg.background
            bkg_rms = np.std(bkg.background)
            
            logging.info(f"{filter_name}: Background corrected (box={box_size}, RMS={bkg_rms:.6f})")
            return data_corr, bkg_rms
            
        except Exception as e:
            logging.warning(f"Background correction failed: {e}")
            return data, np.std(data)

    def find_image_file(self, field_name, filter_name):
        """Buscar archivos de imagen - COHERENTE con Script 1"""
        patterns = [
            f"{field_name}/{field_name}_{filter_name}.fits.fz",
            f"{field_name}/{field_name}_{filter_name}.fits",
            f"{field_name}_{filter_name}.fits.fz", 
            f"{field_name}_{filter_name}.fits"
        ]
        for path in patterns:
            if os.path.exists(path):
                return path
        return None

    def apply_aperture_correction_coherent(self, flux, flux_err, aperture_correction_mag):
        """
        ✅ CORRECCIÓN DE APERTURA COHERENTE con Script 1
        Fórmula IDÉNTICA: flux_total = flux_aper * 10^(0.4 * ACm)
        """
        if aperture_correction_mag == 0.0:
            return flux, flux_err
        
        # ✅ FÓRMULA IDÉNTICA al Script 1
        flux_correction_factor = 10**(0.4 * aperture_correction_mag)
        flux_corrected = flux * flux_correction_factor
        flux_err_corrected = flux_err * flux_correction_factor
        
        logging.debug(f"Aperture correction: AC={aperture_correction_mag:.3f}, factor={flux_correction_factor:.3f}")
        return flux_corrected, flux_err_corrected

    def calculate_magnitudes_coherent(self, flux, flux_err, zero_point):
        """
        ✅ CÁLCULO DE MAGNITUDES COHERENTE con Script 1
        """
        min_flux = 1e-10
        valid_mask = (flux > min_flux) & (flux_err > 0) & np.isfinite(flux) & np.isfinite(flux_err)
        
        # ✅ FÓRMULA: mag = -2.5 * log10(flux) + ZP (IDÉNTICA)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(valid_mask, -2.5 * np.log10(flux) + zero_point, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
        # Filtrar magnitudes fuera de rango físico
        mag = np.where((mag >= 10.0) & (mag <= 30.0), mag, 99.0)
        
        return mag, mag_err

    def process_field_identical(self, field_name):
        """Procesar campo completo - IDÉNTICO a Script 1"""
        logging.info(f"🎯 Processing field {field_name}")
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist")
            return None
        
        # ✅ CARGAR CORRECCIONES DEL SCRIPT 1
        aperture_corrections = self.load_aperture_corrections_from_script1(field_name)
        
        # Usar todas las fuentes para prueba
        field_sources = self.catalog.copy()
        logging.info(f"Testing with {len(field_sources)} GC sources in field {field_name}")
        
        if len(field_sources) == 0:
            return None
        
        results = field_sources.copy()
        
        # Procesar cada filtro
        for filter_name in self.filters:
            logging.info(f"  Processing filter {filter_name}")
            image_path = self.find_image_file(field_name, filter_name)
            if not image_path:
                logging.warning(f"    Image not found for {field_name} {filter_name}")
                continue
            
            try:
                # Cargar imagen
                with fits.open(image_path) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            data = hdu.data.astype(float)
                            header = hdu.header
                            break
                    else:
                        continue
                
                # Obtener WCS y convertir coordenadas
                wcs = WCS(header)
                ra_vals = field_sources[self.ra_col].astype(float).values
                dec_vals = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_vals*u.deg, dec=dec_vals*u.deg)
                x, y = wcs.world_to_pixel(coords)
                
                positions = np.column_stack((x, y))
                height, width = data.shape
                
                # Filtrar posiciones válidas
                margin = 50
                valid_mask = (
                    (x >= margin) & (x < width - margin) & 
                    (y >= margin) & (y < height - margin) &
                    np.isfinite(x) & np.isfinite(y)
                )
                valid_positions = positions[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_positions) == 0:
                    logging.warning(f"No valid positions for {filter_name}")
                    continue
                
                # ✅ PROCESAMIENTO IDÉNTICO al Script 1
                data_processed, used_unsharp, has_structure = self.process_image_with_galaxy_correction_identical(
                    data, valid_positions, filter_name)
                
                # ✅ CORRECCIÓN DE FONDO (adicional para cúmulos)
                data_corrected, bkg_rms = self.apply_background_correction_coherent(data_processed, filter_name)
                
                # ✅ OBTENER ZERO POINT COHERENTE
                zero_point = self.zeropoints.get(field_name, {}).get(filter_name, 0.0)
                if zero_point == 0.0:
                    logging.warning(f"No zero point for {field_name} {filter_name}")
                
                # ✅ OBTENER CORRECCIONES DE APERTURA DEL SCRIPT 1
                filter_corrections = aperture_corrections.get(filter_name, {})
                
                # Procesar cada apertura
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    annulus_inner_px = (6 / 2) / self.pixel_scale
                    annulus_outer_px = (9 / 2) / self.pixel_scale
                    
                    # Fotometría simple
                    flux, flux_err, snr, _ = self.perform_photometry_simple(
                        data_corrected, np.full_like(data_corrected, bkg_rms), 
                        valid_positions, aperture_radius_px, annulus_inner_px, annulus_outer_px)
                    
                    # ✅ APLICAR CORRECCIÓN DE APERTURA DEL SCRIPT 1
                    aperture_correction_mag = filter_corrections.get(aperture_size, 0.0)
                    flux_corr, flux_err_corr = self.apply_aperture_correction_coherent(
                        flux, flux_err, aperture_correction_mag)
                    
                    # ✅ CÁLCULO DE MAGNITUDES COHERENTE
                    mag, mag_err = self.calculate_magnitudes_coherent(flux_corr, flux_err_corr, zero_point)
                    
                    # Guardar resultados
                    for i, idx in enumerate(valid_indices):
                        prefix = f"{filter_name}_{aperture_size}"
                        results.loc[results.index[idx], f'FLUX_{prefix}'] = flux_corr[i]
                        results.loc[results.index[idx], f'FLUXERR_{prefix}'] = flux_err_corr[i]
                        results.loc[results.index[idx], f'MAG_{prefix}'] = mag[i]
                        results.loc[results.index[idx], f'MAGERR_{prefix}'] = mag_err[i]
                        results.loc[results.index[idx], f'SNR_{prefix}'] = snr[i]
                        results.loc[results.index[idx], f'AP_CORR_{prefix}'] = aperture_correction_mag
                        results.loc[results.index[idx], f'USED_UNSHARP_{filter_name}'] = used_unsharp
                        results.loc[results.index[idx], f'HAS_STRUCTURE_{filter_name}'] = has_structure
                
                logging.info(f"    ✅ Completed {filter_name}")
                
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                continue
        
        return results

    # Mantener función de fotometría sin cambios
    def perform_photometry_simple(self, data, error_map, positions, aperture_radius_px, annulus_inner_px, annulus_outer_px):
        """Fotometría simple"""
        try:
            apertures = CircularAperture(positions, r=aperture_radius_px)
            annulus = CircularAnnulus(positions, r_in=annulus_inner_px, r_out=annulus_outer_px)
            
            phot_table = aperture_photometry(data, apertures, error=error_map)
            bkg_table = aperture_photometry(data, annulus, error=error_map)
            
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = phot_table['aperture_sum_err'].data
            bkg_flux = bkg_table['aperture_sum'].data
            bkg_flux_err = bkg_table['aperture_sum_err'].data
            
            aperture_area = apertures.area
            annulus_area = annulus.area
            
            bkg_mean = bkg_flux / annulus_area
            bkg_mean_err = bkg_flux_err / annulus_area
            
            net_flux = raw_flux - (bkg_mean * aperture_area)
            net_flux_err = np.sqrt(raw_flux_err**2 + (aperture_area * bkg_mean_err)**2)
            
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            
            return net_flux, net_flux_err, snr, aperture_area
            
        except Exception as e:
            logging.error(f"Photometry error: {e}")
            n = len(positions)
            return np.zeros(n), np.zeros(n), np.zeros(n), np.pi * aperture_radius_px**2

def main():
    """Función principal"""
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_compatible.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    # Modo de prueba
    test_mode = True
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    photometry = SPLUSGCPhotometryCoherent(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_identical(field)
        if results is not None and len(results) > 0:
            results['FIELD'] = field
            all_results.append(results)
            
            output_file = f'{field}_gc_photometry_identical.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_identical.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")

if __name__ == "__main__":
    main()
