#!/usr/bin/env python3
"""
Splus_photometry_final_NO_APER_CORR_IDENTICAL.py
VERSIÓN TOTALMENTE COHERENTE con Script 1 mejorado
ELIMINADA la corrección de fondo adicional que no existe en Script 1
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

class SPLUSGCPhotometryIdentical:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Cargar zero points del Script 2 SIN CORRECCIONES
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
        
        # Configuración IDÉNTICA con Script 1
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

    def get_zero_aperture_corrections(self):
        """
        ✅ CORRECCIONES CERO - IDÉNTICO a Script 1 modificado
        """
        zero_corrections = {}
        for filter_name in self.filters:
            zero_corrections[filter_name] = {ap: 0.0 for ap in self.apertures}
        
        logging.info("✅ Using ZERO aperture corrections (identical to modified Script 1)")
        return zero_corrections

    def detect_galaxy_structure_identical(self, data, filter_name):
        """
        ✅ DETECCIÓN MEJORADA IDÉNTICA al Script 1 - Background2D con residuos
        """
        try:
            from photutils.background import Background2D, MedianBackground
            from astropy.stats import SigmaClip
            
            # Usar Background2D para modelar variaciones a gran escala
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            
            # Box size más pequeño para capturar estructura galáctica
            box_size = (100, 100)  # píxeles
            bkg = Background2D(data, box_size, 
                              filter_size=(3, 3),
                              sigma_clip=sigma_clip, 
                              bkg_estimator=bkg_estimator)
            
            # Calcular residuos estructurados
            residual = data - bkg.background
            residual_std = np.std(residual)
            
            # Umbral más conservador para detección
            has_significant_structure = np.max(np.abs(residual)) > 5 * residual_std
            
            if has_significant_structure:
                # Encontrar centro de la estructura máxima
                height, width = data.shape
                y, x = np.ogrid[:height, :width]
                
                # Usar el pico de residuo para centrar
                max_residual_pos = np.unravel_index(np.argmax(np.abs(residual)), data.shape)
                center_y, center_x = max_residual_pos
                
                # Radio estimado basado en la extensión de residuos significativos
                significant_mask = np.abs(residual) > 3 * residual_std
                if np.any(significant_mask):
                    y_pos, x_pos = np.where(significant_mask)
                    distances = np.sqrt((x_pos - center_x)**2 + (y_pos - center_y)**2)
                    structure_radius = np.percentile(distances, 90)  # 90th percentile
                else:
                    structure_radius = min(center_x, center_y, 
                                         width - center_x, height - center_y) * 0.7
                
                logging.info(f"{filter_name}: Estructura significativa detectada - centro ({center_x:.1f}, {center_y:.1f}), radio {structure_radius:.1f} pix")
                return True, center_x, center_y, structure_radius
            else:
                logging.info(f"{filter_name}: Sin estructura residual significativa")
                return False, data.shape[1] // 2, data.shape[0] // 2, 0
                
        except Exception as e:
            logging.warning(f"Error en detección mejorada: {e}")
            return False, data.shape[1] // 2, data.shape[0] // 2, 0

    def apply_unsharp_mask_selective_identical(self, data, positions, center_x, center_y, structure_radius, filter_name):
        """
        ✅ UNSHARP MASK CONSERVADOR IDÉNTICO al Script 1
        """
        try:
            # Calcular distancias al centro de estructura
            distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
            
            # ✅ Solo fuentes dentro del radio de estructura + margen (50 píxeles)
            margin = 50  # píxeles de margen
            near_structure = distances < (structure_radius + margin)
            n_near = np.sum(near_structure)
            
            if n_near == 0:
                return data, False
            
            logging.info(f"{filter_name}: Aplicando unsharp mask conservador a {n_near}/{len(positions)} fuentes")
            
            # ✅ Parámetros conservadores IDÉNTICOS
            data_clean = data.copy()
            data_clean = np.nan_to_num(data_clean, nan=0.0, posinf=0.0, neginf=0.0)
            
            # ✅ Sigma más pequeño: max(8.0, structure_radius / 30.0)
            sigma = max(8.0, structure_radius / 30.0)  # Más conservador
            smoothed = gaussian_filter(data_clean, sigma=sigma)
            
            # ✅ Sustracción más suave: 0.3 (reducido de 0.7)
            data_unsharp = data_clean - 0.3 * smoothed
            
            # ✅ Aplicar solo en regiones cerca de estructura galáctica
            mask = np.zeros_like(data, dtype=bool)
            y_idx, x_idx = np.ogrid[:data.shape[0], :data.shape[1]]
            structure_mask = (x_idx - center_x)**2 + (y_idx - center_y)**2 < (structure_radius + margin)**2
            
            result = data.copy()
            result[structure_mask] = data_unsharp[structure_mask]
            
            logging.info(f"{filter_name}: Unsharp mask aplicado (sigma={sigma:.1f}, factor=0.3)")
            return result, True
            
        except Exception as e:
            logging.error(f"Error en unsharp mask conservador: {e}")
            return data, False

    def process_image_with_galaxy_correction_identical(self, data, positions, filter_name):
        """
        ✅ PROCESAMIENTO TOTALMENTE IDÉNTICO al Script 1 mejorado
        SOLO unsharp mask si hay estructura - SIN Background2D adicional
        """
        # ✅ DETECCIÓN MEJORADA con Background2D (solo para detección)
        has_structure, center_x, center_y, structure_radius = self.detect_galaxy_structure_identical(data, filter_name)
        
        use_unsharp = False
        data_processed = data.copy()
        
        if has_structure and structure_radius > 0:
            # ✅ UNSHARP MASK CONSERVADOR (solo corrección aplicada)
            data_processed, use_unsharp = self.apply_unsharp_mask_selective_identical(
                data, positions, center_x, center_y, structure_radius, filter_name)
        else:
            logging.info(f"{filter_name}: Sin estructura galáctica, procesamiento normal")
        
        return data_processed, use_unsharp, has_structure

    def find_image_file(self, field_name, filter_name):
        """Buscar archivos de imagen - IDÉNTICO a Script 1"""
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None

    def apply_no_aperture_correction(self, flux, flux_err):
        """
        ✅ NO APLICA CORRECCIÓN DE APERTURA
        Idéntico a Script 1 modificado
        """
        return flux, flux_err

    def calculate_magnitudes_identical(self, flux, flux_err, zero_point):
        """
        ✅ CÁLCULO DE MAGNITUDES IDÉNTICO al Script 1
        Fórmula: mag = -2.5 * log10(flux) + ZP
        """
        min_flux = 1e-10
        valid_mask = (flux > min_flux) & (flux_err > 0) & np.isfinite(flux) & np.isfinite(flux_err)
        
        # ✅ FÓRMULA IDÉNTICA
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(valid_mask, -2.5 * np.log10(flux) + zero_point, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
        # Filtrar magnitudes fuera de rango físico
        mag = np.where((mag >= 10.0) & (mag <= 30.0), mag, 99.0)
        
        return mag, mag_err

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

    def process_field_identical(self, field_name):
        """Procesar campo completo - TOTALMENTE IDÉNTICO a Script 1 mejorado"""
        logging.info(f"🎯 Processing field {field_name} (IDENTICAL TO IMPROVED SCRIPT 1)")
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist")
            return None
        
        # Obtener centro del campo
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}")
            return None
        
        # ✅ USAR CORRECCIONES CERO (idéntico a Script 1 modificado)
        aperture_corrections = self.get_zero_aperture_corrections()
        
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
                
                # Convertir coordenadas a píxeles
                ra_vals = field_sources[self.ra_col].astype(float).values
                dec_vals = field_sources[self.dec_col].astype(float).values
                try:
                    wcs = WCS(header)
                    coords = SkyCoord(ra=ra_vals*u.deg, dec=dec_vals*u.deg)
                    x, y = wcs.world_to_pixel(coords)
                except:
                    logging.warning("Using catalog coordinates as pixels (no WCS)")
                    x, y = ra_vals, dec_vals
                
                positions = np.column_stack((x, y))
                height, width = data.shape
                margin = (max(self.apertures) / self.pixel_scale) * 2
                valid_mask = (
                    (x >= margin) & (x < width - margin) & 
                    (y >= margin) & (y < height - margin) &
                    np.isfinite(x) & np.isfinite(y)
                )
                valid_positions = positions[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_positions) == 0:
                    continue
                
                # ✅ PROCESAMIENTO TOTALMENTE IDÉNTICO al Script 1 mejorado
                # SOLO unsharp mask si hay estructura - SIN Background2D adicional
                data_processed, used_unsharp, has_structure = self.process_image_with_galaxy_correction_identical(
                    data, valid_positions, filter_name)
                
                # ✅ USAR DIRECTAMENTE la imagen procesada (COHERENTE con Script 1)
                # NO aplicar Background2D adicional que no existe en Script 1
                final_data = data_processed
                
                # ✅ Estimar RMS del fondo para cálculo de errores
                mean_val, median_val, std_val = sigma_clipped_stats(final_data, sigma=3.0)
                bkg_rms = std_val
                
                # ✅ OBTENER ZERO POINT COHERENTE
                zero_point = self.zeropoints.get(field_name, {}).get(filter_name, 0.0)
                if zero_point == 0.0:
                    logging.warning(f"No zero point for {field_name} {filter_name}")
                
                # Procesar cada apertura
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    annulus_inner_px = (6 / 2) / self.pixel_scale
                    annulus_outer_px = (9 / 2) / self.pixel_scale
                    
                    # Fotometría simple sobre final_data (sin correcciones adicionales)
                    flux, flux_err, snr, _ = self.perform_photometry_simple(
                        final_data, np.full_like(final_data, bkg_rms), 
                        valid_positions, aperture_radius_px, annulus_inner_px, annulus_outer_px)
                    
                    # ✅ NO APLICAR CORRECCIÓN DE APERTURA (idéntico a Script 1)
                    flux_corr, flux_err_corr = self.apply_no_aperture_correction(flux, flux_err)
                    
                    # ✅ CÁLCULO DE MAGNITUDES IDÉNTICO
                    mag, mag_err = self.calculate_magnitudes_identical(flux_corr, flux_err_corr, zero_point)
                    
                    # Guardar resultados
                    for i, idx in enumerate(valid_indices):
                        source_idx = results.index[idx]
                        prefix = f"{filter_name}_{aperture_size}"
                        results.loc[source_idx, f'FLUX_{prefix}'] = flux_corr[i]
                        results.loc[source_idx, f'FLUXERR_{prefix}'] = flux_err_corr[i]
                        results.loc[source_idx, f'MAG_{prefix}'] = mag[i]
                        results.loc[source_idx, f'MAGERR_{prefix}'] = mag_err[i]
                        results.loc[source_idx, f'SNR_{prefix}'] = snr[i]
                        results.loc[source_idx, f'AP_CORR_{prefix}'] = 0.0  # ✅ Siempre 0.0
                        results.loc[source_idx, f'USED_UNSHARP_{filter_name}'] = used_unsharp
                        results.loc[source_idx, f'HAS_STRUCTURE_{filter_name}'] = has_structure
                
                logging.info(f"    ✅ Completed {filter_name}")
                
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                continue
        
        return results

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
        """Cargar mapa de pesos"""
        try:
            with fits.open(weight_path) as whdul:
                for whdu in whdul:
                    if whdu.data is not None:
                        weight_data = whdu.data.astype(float)
                        break
                else:
                    raise ValueError("No data in weight file")
            
            if weight_data.shape != data_shape:
                return None
            
            valid_weight = weight_data > 0
            if np.sum(valid_weight) / weight_data.size < 0.5:
                return None
            
            error_map = np.full_like(weight_data, np.nan)
            error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
            return error_map
        except:
            return None

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
    # ✅ USAR ZERO POINTS SIN CORRECCIONES
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_no_aper_corr.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    # Modo de prueba
    test_mode = True
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    photometry = SPLUSGCPhotometryIdentical(
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
            
            output_file = f'{field}_gc_photometry_identical_gc.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_identical_gc.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        # Estadísticas finales
        logging.info("\n📊 ESTADÍSTICAS FINALES:")
        total_sources = len(final_results)
        measured_sources = len(final_results[final_results['FIELD'].notna()])
        logging.info(f"Fuentes totales en catálogo: {len(photometry.original_catalog)}")
        logging.info(f"Fuentes medidas: {measured_sources}")
        logging.info(f"Fuentes con campo asignado: {total_sources}")

if __name__ == "__main__":
    main()
