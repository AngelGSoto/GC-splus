#!/usr/bin/env python3
"""
Splus_photometry_final_bg_modeling_aper_correction_CORREGIDO.py - Fotometría de cúmulos globulares con aperture correction correcto
Versión corregida para trabajar con las nuevas magnitudes calibradas del script 1.
CON SOPORTE COMPLETO PARA LAS NUEVAS MAGNITUDES CALIBRADAS Y FACTORES DE CORRECCIÓN.
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
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSGCPhotometryCorrected:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
        """
        Inicializa la clase de fotometría para cúmulos globulares
        """
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
        
        # Configuración
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel para SPLUS
        self.apertures = [3, 4, 5, 6]  # diámetros en arcsec
        self.debug = debug
        self.debug_filter = debug_filter
        
        # Mapeo de columnas del catálogo
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RAJ2000/DEJ2000 or RA/DEC or ra/dec columns")
        
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain T17ID or ID or id column")
        
        logging.info(f"Using columns: RA={self.ra_col}, DEC={self.dec_col}, ID={self.id_col}")

    def validate_aperture_correction_factors(self, correction_factors, filter_name):
        """
        Valida que los factores de corrección sean razonables
        """
        reasonable_ranges = {
            3: (1.0, 1.3),   # 0-30% de corrección para 3 arcsec
            4: (1.0, 1.2),   # 0-20% para 4 arcsec
            5: (1.0, 1.15),  # 0-15% para 5 arcsec
            6: (1.0, 1.1)    # 0-10% para 6 arcsec
        }
        
        validated_factors = {}
        issues = []
        
        for aperture_size, factor in correction_factors.items():
            min_val, max_val = reasonable_ranges.get(aperture_size, (1.0, 2.0))
            
            if factor < min_val or factor > max_val:
                issues.append(f"Aperture {aperture_size}arcsec: factor {factor:.3f} fuera de rango razonable [{min_val:.1f}-{max_val:.1f}]")
                # Forzar a valor razonable
                if factor > max_val:
                    validated_factors[aperture_size] = max_val
                else:
                    validated_factors[aperture_size] = min_val
            else:
                validated_factors[aperture_size] = factor
        
        if issues:
            logging.warning(f"{filter_name}: Problemas con factores de corrección:")
            for issue in issues:
                logging.warning(f"  {issue}")
        
        return validated_factors

    def load_aperture_correction_factors(self, field_name):
        """
        Carga y VALIDA los factores de aperture correction del archivo generado por el script 1
        """
        try:
            # ✅ Asegurarse de que usa el archivo correcto
            ref_catalog_path = f"{field_name}_gaia_xp_matches_splus_method.csv"
            logging.info(f"Buscando factores de aperture correction en: {ref_catalog_path}")
            
            if not os.path.exists(ref_catalog_path):
                logging.error(f"❌ ARCHIVO NO ENCONTRADO: {ref_catalog_path}")
                logging.error("❌ Ejecuta primero el Script 1 para generar los factores de corrección")
                return self._get_default_corrections()
            
            ref_df = pd.read_csv(ref_catalog_path)
            
            # ✅ VERIFICAR que el archivo contiene las columnas correctas
            sample_filter = self.filters[0]
            sample_column = f'ap_corr_3_{sample_filter}'
            
            if sample_column not in ref_df.columns:
                logging.error(f"❌ Columna {sample_column} no encontrada en {ref_catalog_path}")
                logging.error("❌ El Script 1 no generó los factores correctamente")
                return self._get_default_corrections()
            
            corrections = {}
            
            for filter_name in self.filters:
                filter_corrections = {}
                for aperture_size in self.apertures:
                    col_name = f'ap_corr_{aperture_size}_{filter_name}'
                    if col_name in ref_df.columns:
                        # Filtrar factores razonables
                        valid_factors = ref_df[
                            (ref_df[col_name] >= 0.5) & 
                            (ref_df[col_name] <= 3.0)
                        ][col_name]
                        
                        if len(valid_factors) > 0:
                            factor = float(valid_factors.median())
                            filter_corrections[aperture_size] = factor
                        else:
                            filter_corrections[aperture_size] = 1.0
                            logging.warning(f"{filter_name}: No hay factores válidos para {aperture_size}arcsec")
                    else:
                        filter_corrections[aperture_size] = 1.0
                        logging.warning(f"{filter_name}: Columna {col_name} no encontrada")
                
                # VALIDAR los factores
                validated_factors = self.validate_aperture_correction_factors(filter_corrections, filter_name)
                corrections[filter_name] = validated_factors
                
                # ✅ VERIFICAR que los factores no sean 1.0 (indicaría problema)
                if all(factor == 1.0 for factor in validated_factors.values()):
                    logging.warning(f"⚠️  {filter_name}: Todos los factores son 1.0 - verificar Script 1")
                else:
                    logging.info(f"✅ {filter_name}: Factores validados: {validated_factors}")
            
            return corrections
            
        except Exception as e:
            logging.error(f"Error loading aperture corrections: {e}")
            traceback.print_exc()
            return self._get_default_corrections()
    
    def _get_default_corrections(self):
        """Devuelve factores de corrección por defecto (sin corrección)"""
        default_corrections = {}
        for filter_name in self.filters:
            default_corrections[filter_name] = {ap: 1.0 for ap in self.apertures}
        logging.warning("⚠️  Usando factores de corrección por defecto (1.0)")
        return default_corrections

    def check_splus_background_status(self, data, filter_name):
        """
        Verifica si la imagen SPLUS ya tiene el fondo restado
        """
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            # En imágenes SPLUS procesadas, la mediana debería ser cercana a cero
            background_already_subtracted = (abs(median) < 0.1 * std)
            
            if background_already_subtracted:
                logging.info(f"{filter_name}: Background already subtracted (median={median:.3f}, std={std:.3f})")
                return False, std
            else:
                logging.warning(f"{filter_name}: Possible residual background (median={median:.3f}, std={std:.3f})")
                return True, std
                
        except Exception as e:
            logging.warning(f"Background check failed: {e}")
            return False, np.std(data)

    def apply_conservative_background_correction(self, data, filter_name):
        """
        Aplica corrección de fondo conservadora solo si es absolutamente necesario
        """
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            # Solo corregir si hay variación significativa
            if std < 2.0:
                logging.info(f"{filter_name}: Minimal variation, skipping background correction")
                return data, std
            
            # Máscara conservadora para fuentes brillantes
            mask = data > median + 15 * std
            
            # Dilatación mínima
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            
            # Boxes grandes para variaciones de gran escala
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=200,
                              filter_size=5,
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=30)
            
            # Solo restar si el modelo muestra estructura significativa
            bkg_range = np.max(bkg.background) - np.min(bkg.background)
            if bkg_range < 2 * std:
                data_corrected = data - np.median(bkg.background)
                logging.info(f"{filter_name}: Applied minimal background correction")
            else:
                data_corrected = data - bkg.background
                logging.info(f"{filter_name}: Applied full background correction")
            
            return data_corrected, bkg.background_rms_median
            
        except Exception as e:
            logging.warning(f"Background correction failed: {e}")
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            return data, std

    def get_field_center_from_header(self, field_name, filter_name='F660'):
        """
        Obtiene el centro del campo desde el header FITS
        """
        possible_files = [
            os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz"),
            os.path.join(field_name, f"{field_name}_{filter_name}.fits")
        ]
        
        for sci_file in possible_files:
            if os.path.exists(sci_file):
                try:
                    with fits.open(sci_file) as hdul:
                        for hdu in hdul:
                            if hdu.data is not None and hasattr(hdu, 'header'):
                                header = hdu.header
                                ra_center = header.get('CRVAL1')
                                dec_center = header.get('CRVAL2')
                                if ra_center is not None and dec_center is not None:
                                    return float(ra_center), float(dec_center)
                except Exception as e:
                    logging.warning(f"Error reading {sci_file}: {e}")
                    continue
        
        logging.warning(f"Could not determine field center for {field_name}")
        return None, None

    def is_source_in_field(self, source_ra, source_dec, field_ra, field_dec, field_radius_deg=0.84):
        """
        Verifica si una fuente está dentro del campo
        """
        if field_ra is None or field_dec is None:
            return False
            
        coord1 = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
        coord2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        separation = coord1.separation(coord2).degree
        return separation <= field_radius_deg

    def find_image_file(self, field_name, filter_name):
        """Encuentra el archivo de imagen para un campo y filtro dados"""
        possible_extensions = [
            f"{field_name}_{filter_name}.fits.fz",
            f"{field_name}_{filter_name}.fits"
        ]
        
        for ext in possible_extensions:
            image_path = os.path.join(field_name, ext)
            if os.path.exists(image_path):
                return image_path
        
        return None

    def find_weight_file(self, field_name, filter_name):
        """Encuentra el archivo de peso para un campo y filtro dados"""
        possible_patterns = [
            f"{field_name}_{filter_name}.weight.fits.fz",
            f"{field_name}_{filter_name}.weight.fits",
            f"{field_name}_{filter_name}_weight.fits.fz", 
            f"{field_name}_{filter_name}_weight.fits",
            f"weight/{field_name}_{filter_name}.weight.fits"
        ]
        
        for pattern in possible_patterns:
            weight_path = os.path.join(field_name, pattern)
            if os.path.exists(weight_path):
                logging.info(f"Found weight file: {weight_path}")
                return weight_path
        
        logging.warning(f"Weight file not found for {field_name} {filter_name}")
        return None

    def validate_weight_map(self, weight_data):
        """Valida que el weight map sea razonable"""
        if weight_data is None:
            return False, "Weight data is None"
        
        if weight_data.size == 0:
            return False, "Weight data is empty"
        
        if np.all(weight_data <= 0):
            return False, "All weight values are <= 0"
        
        if np.any(np.isnan(weight_data)):
            return False, "Weight map contains NaN values"
        
        if np.any(np.isinf(weight_data)):
            return False, "Weight map contains Inf values"
        
        valid_fraction = np.sum(weight_data > 0) / weight_data.size
        if valid_fraction < 0.5:
            return False, f"Less than 50% valid weights ({valid_fraction:.1%})"
        
        return True, "Valid weight map"

    def load_and_validate_weight_map(self, weight_path, data_shape):
        """Carga y valida el weight map"""
        try:
            with fits.open(weight_path) as whdul:
                for whdu in whdul:
                    if whdu.data is not None:
                        weight_data = whdu.data.astype(float)
                        break
                else:
                    raise ValueError("No data found in weight file")
            
            # Validar forma
            if weight_data.shape != data_shape:
                logging.warning(f"Weight map shape {weight_data.shape} doesn't match data shape {data_shape}")
                return None
            
            # Validar contenido
            is_valid, message = self.validate_weight_map(weight_data)
            if not is_valid:
                logging.warning(f"Invalid weight map: {message}")
                return None
            
            # Calcular error map (relación física correcta: error = 1/sqrt(weight))
            valid_weight = weight_data > 0
            error_map = np.full_like(weight_data, np.nan)
            error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
            
            # Manejar valores no finitos
            if np.any(~np.isfinite(error_map)):
                finite_errors = error_map[np.isfinite(error_map)]
                if len(finite_errors) > 0:
                    max_error = np.nanmax(finite_errors)
                    error_map[~np.isfinite(error_map)] = max_error
                    logging.warning("Replaced non-finite values in error map")
                else:
                    raise ValueError("No finite values in error map")
            
            logging.info(f"Successfully loaded weight map with {np.sum(valid_weight)} valid pixels")
            return error_map
            
        except Exception as e:
            logging.warning(f"Error loading weight map {weight_path}: {e}")
            return None

    def perform_photometry_with_aperture_correction(self, data, error_map, positions, 
                                                   aperture_radius_px, annulus_inner_px, 
                                                   annulus_outer_px, correction_factor):
        """
        Fotometría con errores calculados correctamente usando Photutils y weight maps.
        """
        try:
            # Crear aperturas
            apertures = CircularAperture(positions, r=aperture_radius_px)
            annulus = CircularAnnulus(positions, r_in=annulus_inner_px, r_out=annulus_outer_px)
            
            # Fotometría con propagación de errores usando el error_map
            phot_table = aperture_photometry(data, apertures, error=error_map)
            bkg_phot_table = aperture_photometry(data, annulus, error=error_map)
            
            # Obtener flujos y errores calculados por Photutils
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = phot_table['aperture_sum_err'].data
            bkg_flux = bkg_phot_table['aperture_sum'].data
            bkg_flux_err = bkg_phot_table['aperture_sum_err'].data
            
            # Calcular áreas
            aperture_area = apertures.area
            annulus_area = annulus.area
            
            # Fondo medio por píxel y su error
            bkg_mean = bkg_flux / annulus_area
            bkg_mean_err = bkg_flux_err / annulus_area
            
            # Flujo neto (bruto - fondo) y su error (propagación cuadrática)
            net_flux = raw_flux - bkg_mean * aperture_area
            net_flux_err = np.sqrt(raw_flux_err**2 + (aperture_area * bkg_mean_err)**2)
            
            # Aplicar corrección de apertura al flujo neto y su error
            flux_corrected = net_flux * correction_factor
            flux_err_corrected = net_flux_err * correction_factor
            
            # Calcular SNR
            snr = np.where(flux_err_corrected > 0, flux_corrected / flux_err_corrected, 0.0)
            
            return flux_corrected, flux_err_corrected, snr, aperture_area
            
        except Exception as e:
            logging.error(f"Error en fotometría con Photutils: {e}")
            traceback.print_exc()
            # Fallback simple en caso de error
            return self._simple_photometry_fallback(data, positions, aperture_radius_px, 
                                                  annulus_inner_px, annulus_outer_px, 
                                                  correction_factor)

    def _simple_photometry_fallback(self, data, positions, aperture_radius_px, 
                                  annulus_inner_px, annulus_outer_px, correction_factor):
        """Método de respaldo simple para casos de error"""
        try:
            apertures = CircularAperture(positions, r=aperture_radius_px)
            annulus = CircularAnnulus(positions, r_in=annulus_inner_px, r_out=annulus_outer_px)
            
            # Fotometría básica sin errores detallados
            phot_table = aperture_photometry(data, apertures)
            bkg_phot_table = aperture_photometry(data, annulus)
            
            # Cálculo simple de fondo y flujo neto
            bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
            net_flux = phot_table['aperture_sum'] - bkg_mean * apertures.area
            
            # Aplicar corrección
            flux_corrected = net_flux * correction_factor
            
            # Estimación simple de error (Poisson)
            flux_err = np.sqrt(np.abs(net_flux)) * correction_factor
            snr = np.where(flux_err > 0, flux_corrected / flux_err, 0.0)
            
            logging.warning("Using fallback photometry method (simple error estimation)")
            return flux_corrected, flux_err, snr, apertures.area
            
        except Exception as e:
            logging.error(f"Error in fallback photometry: {e}")
            # Último recurso: retornar arrays de ceros
            n_positions = len(positions)
            return (np.zeros(n_positions), np.zeros(n_positions), 
                    np.zeros(n_positions), apertures.area)

    def calculate_magnitudes_with_validation(self, flux, flux_err, zero_point, filter_name, aperture_size):
        """
        Calcula magnitudes con validación adicional - RANGO AJUSTADO para magnitudes calibradas
        """
        # Validar flujos antes de calcular magnitudes
        valid_flux_mask = (flux > 0) & (flux < 1e6) & (flux_err > 0) & np.isfinite(flux) & np.isfinite(flux_err)
        
        # Calcular magnitudes solo para flujos válidos
        mag = np.where(valid_flux_mask, zero_point - 2.5 * np.log10(flux), 99.0)
        mag_err = np.where(valid_flux_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
        # ✅ RANGO AJUSTADO: Ahora acepta magnitudes más brillantes (5-30) debido a la calibración
        reasonable_mag_range = (5.0, 30.0)  # Antes era (10.0, 30.0)
        mag = np.where((mag >= reasonable_mag_range[0]) & (mag <= reasonable_mag_range[1]), mag, 99.0)
        
        # Log para diagnóstico
        n_valid = np.sum(valid_flux_mask)
        if n_valid > 0:
            valid_mags = mag[valid_flux_mask]
            mean_mag = np.mean(valid_mags)
            std_mag = np.std(valid_mags)
            logging.debug(f"{filter_name}_{aperture_size}: {n_valid} fuentes válidas, mag: {mean_mag:.2f} ± {std_mag:.2f}")
        else:
            logging.warning(f"{filter_name}_{aperture_size}: No hay fuentes con flujos válidos")
        
        return mag, mag_err

    def save_debug_image(self, data, positions, aperture_radius_px, field_name, filter_name):
        """
        Guarda imagen de diagnóstico para debug
        """
        if len(positions) == 0:
            return
            
        try:
            # Encontrar fuente cercana al centro
            center_y, center_x = data.shape[0]//2, data.shape[1]//2
            distances = np.sqrt((positions[:,0]-center_x)**2 + (positions[:,1]-center_y)**2)
            idx = np.argmin(distances)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Imagen original
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            ax1.imshow(data, origin='lower', cmap='gray', vmin=median-std, vmax=median+3*std)
            ax1.set_title(f'{field_name} {filter_name} - Image')
            
            # Aperturas
            aperture = CircularAperture(positions[idx], r=aperture_radius_px)
            aperture.plot(ax=ax1, color='red', lw=2, label=f'Aperture ({aperture_radius_px*self.pixel_scale*2:.1f} arcsec)')
            ax1.legend()
            
            # Zoom alrededor de la fuente
            x, y = positions[idx]
            size = 100
            y_min = max(0, int(y - size))
            y_max = min(data.shape[0], int(y + size))
            x_min = max(0, int(x - size))
            x_max = min(data.shape[1], int(x + size))
            
            if y_max > y_min and x_max > x_min:
                cutout = data[y_min:y_max, x_min:x_max]
                ax2.imshow(cutout, origin='lower', cmap='gray', 
                          vmin=median-std, vmax=median+3*std)
                aperture_zoom = CircularAperture([x-x_min, y-y_min], r=aperture_radius_px)
                aperture_zoom.plot(ax=ax2, color='red', lw=2)
                ax2.set_title('Zoom on source')
            
            plt.tight_layout()
            plt.savefig(f'{field_name}_{filter_name}_photometry_debug.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved debug image for {field_name} {filter_name}")
            
        except Exception as e:
            logging.warning(f"Could not save debug image: {e}")

    def compare_with_taylor_catalog(self, results, field_name):
        """
        Compara resultados con el catálogo de Taylor para validación
        """
        try:
            # Asumiendo que el catálogo original tiene magnitudes de Taylor
            taylor_mag_columns = ['gmag', 'rmag', 'imag']
            
            for taylor_col in taylor_mag_columns:
                if taylor_col in results.columns:
                    # Comparar con magnitudes S-PLUS equivalentes
                    splus_filter = self.get_splus_equivalent_filter(taylor_col)
                    if splus_filter:
                        mag_col = f'MAG_{splus_filter}_4'  # Usar apertura de 4 arcsec
                        if mag_col in results.columns:
                            # Filtrar fuentes válidas
                            valid_mask = (results[mag_col] < 50) & (results[taylor_col] < 50)
                            if np.sum(valid_mask) > 10:
                                diff = results.loc[valid_mask, mag_col] - results.loc[valid_mask, taylor_col]
                                mean_diff = diff.mean()
                                std_diff = diff.std()
                                logging.info(f"Comparación con Taylor {taylor_col}->{splus_filter}: Δmag = {mean_diff:.3f} ± {std_diff:.3f} (n={np.sum(valid_mask)})")
            
        except Exception as e:
            logging.warning(f"Error en comparación con Taylor: {e}")

    def get_splus_equivalent_filter(self, taylor_filter):
        """Mapeo aproximado entre filtros Taylor y S-PLUS"""
        mapping = {
            'gmag': 'F515',
            'rmag': 'F660', 
            'imag': 'F861'
        }
        return mapping.get(taylor_filter, None)

    def verify_photometry_coherence(self, field_name, filter_name, aperture_size, flux_values, mag_values, zero_point):
        """
        Verifica la coherencia entre flujos y magnitudes después de la calibración
        """
        try:
            # Verificar que la conversión flujo→magnitud sea coherente
            valid_mask = (flux_values > 0) & (mag_values < 50)
            
            if np.sum(valid_mask) > 10:
                expected_mags = zero_point - 2.5 * np.log10(flux_values[valid_mask])
                actual_mags = mag_values[valid_mask]
                
                differences = actual_mags - expected_mags
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                
                if abs(mean_diff) > 0.1:  # Diferencia mayor a 0.1 mag
                    logging.warning(f"{field_name} {filter_name}: Incoherencia en conversión flujo→mag: {mean_diff:.3f} ± {std_diff:.3f}")
                else:
                    logging.debug(f"{field_name} {filter_name}: Conversión coherente (±{std_diff:.3f} mag)")
                    
        except Exception as e:
            logging.debug(f"Verificación de coherencia falló: {e}")

    def process_field(self, field_name):
        """
        Procesa un campo completo de cúmulos globulares
        """
        logging.info(f"Processing field {field_name}")
        
        # Verificar que existe el directorio del campo
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist. Skipping.")
            return None
        
        # Obtener centro del campo
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None
        
        # Cargar y VALIDAR factores de corrección
        aperture_corrections = self.load_aperture_correction_factors(field_name)
        
        # Filtrar fuentes en el campo
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
        
        # Inicializar DataFrame de resultados
        results = field_sources.copy()
        
        # Procesar cada filtro
        for filter_name in self.filters:
            logging.info(f"  Processing filter {filter_name}")
            
            # Encontrar archivo de imagen
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
                        logging.warning(f"No data found in {image_path}")
                        continue
                
                # Cargar y validar weight map
                error_map = None
                weight_path = self.find_weight_file(field_name, filter_name)
                if weight_path:
                    error_map = self.load_and_validate_weight_map(weight_path, data.shape)
                
                # Si no hay weight map válido, usar estimación basada en fondo
                if error_map is None:
                    logging.info(f"    Using background-based error estimation for {filter_name}")
                    # Verificar y corregir fondo si es necesario
                    needs_correction, original_std = self.check_splus_background_status(data, filter_name)
                    if needs_correction:
                        data_corrected, bkg_rms = self.apply_conservative_background_correction(data, filter_name)
                    else:
                        data_corrected, bkg_rms = data, original_std
                    
                    error_map = np.full_like(data_corrected, bkg_rms)
                else:
                    # Si tenemos weight map, solo corregir fondo pero mantener el error_map
                    needs_correction, original_std = self.check_splus_background_status(data, filter_name)
                    if needs_correction:
                        data_corrected, _ = self.apply_conservative_background_correction(data, filter_name)
                    else:
                        data_corrected = data
                
                # Obtener WCS
                try:
                    wcs = WCS(header)
                except:
                    logging.warning(f"Could not create WCS for {image_path}")
                    wcs = None
                
                # Convertir coordenadas a píxeles
                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                
                if wcs is not None:
                    coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                    x, y = wcs.world_to_pixel(coords)
                else:
                    # Fallback: asumir que las coordenadas ya están en píxeles
                    logging.warning("Using catalog coordinates as pixel coordinates (no WCS)")
                    x, y = ra_values, dec_values
                
                positions = np.column_stack((x, y))
                
                # Filtrar posiciones válidas (lejos de bordes)
                height, width = data_corrected.shape
                margin = (max(self.apertures) / self.pixel_scale) * 2
                
                valid_mask = (
                    (x >= margin) & (x < width - margin) & 
                    (y >= margin) & (y < height - margin) &
                    np.isfinite(x) & np.isfinite(y)
                )
                
                valid_positions = positions[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_positions) == 0:
                    logging.warning(f"No valid positions for photometry in {field_name} {filter_name}")
                    continue
                
                logging.info(f"    {len(valid_positions)} valid positions for photometry")
                
                # Obtener zero point para este campo y filtro
                zero_point = self.zeropoints.get(field_name, {}).get(filter_name)
                if zero_point is None:
                    logging.warning(f"No zero point found for {field_name} {filter_name}")
                    zero_point = 0.0
                
                # Obtener factores de corrección para este filtro
                filter_corrections = aperture_corrections.get(filter_name, {ap: 1.0 for ap in self.apertures})
                
                # Procesar cada tamaño de apertura
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    annulus_inner_px = (6 / 2) / self.pixel_scale
                    annulus_outer_px = (9 / 2) / self.pixel_scale
                    
                    # Obtener factor de corrección
                    correction_factor = filter_corrections.get(aperture_size, 1.0)
                    
                    # Validar factor de corrección
                    if correction_factor < 0.5 or correction_factor > 3.0:
                        logging.warning(f"{filter_name}_{aperture_size}: Factor inválido {correction_factor:.3f}, usando 1.0")
                        correction_factor = 1.0
                    
                    # Realizar fotometría con aplicación CORRECTA de la corrección
                    flux_corrected, flux_err_corrected, snr, aperture_area = self.perform_photometry_with_aperture_correction(
                        data_corrected, error_map, valid_positions, aperture_radius_px,
                        annulus_inner_px, annulus_outer_px, correction_factor)
                    
                    # Calcular magnitudes con validación
                    mag, mag_err = self.calculate_magnitudes_with_validation(
                        flux_corrected, flux_err_corrected, zero_point, filter_name, aperture_size)
                    
                    # Verificar coherencia
                    self.verify_photometry_coherence(field_name, filter_name, aperture_size, 
                                                   flux_corrected, mag, zero_point)
                    
                    # Guardar resultados para cada fuente válida
                    for i, idx in enumerate(valid_indices):
                        source_idx = results.index[idx]
                        prefix = f"{filter_name}_{aperture_size}"
                        
                        results.loc[source_idx, f'X_{prefix}'] = valid_positions[i, 0]
                        results.loc[source_idx, f'Y_{prefix}'] = valid_positions[i, 1]
                        results.loc[source_idx, f'FLUX_CORR_{prefix}'] = flux_corrected[i]
                        results.loc[source_idx, f'FLUXERR_{prefix}'] = flux_err_corrected[i]
                        results.loc[source_idx, f'MAG_{prefix}'] = mag[i]
                        results.loc[source_idx, f'MAGERR_{prefix}'] = mag_err[i]
                        results.loc[source_idx, f'SNR_{prefix}'] = snr[i]
                        results.loc[source_idx, f'AP_CORR_{prefix}'] = correction_factor
                    
                    # Imagen debug
                    if (self.debug and filter_name == self.debug_filter and 
                        aperture_size == 4 and len(valid_positions) > 0):
                        self.save_debug_image(data_corrected, valid_positions, 
                                            aperture_radius_px, field_name, filter_name)
                
                logging.info(f"    Completed {filter_name} with {len(valid_positions)} sources")
                
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        # Comparar con Taylor si está disponible
        self.compare_with_taylor_catalog(results, field_name)
        
        return results

def main():
    """Función principal"""
    # Configuración
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    # ✅ Asegúrate de que esta ruta apunte a los NUEVOS zero points calculados
    zeropoints_file = 'Results/all_fields_zero_points_splus_format.csv'
    
    # Verificar archivos de entrada
    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        exit(1)
    
    if not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
        # Puede que necesites recalcular los zero points primero
        logging.error("❌ Necesitas recalcular los zero points con las nuevas magnitudes del script 1")
        logging.error("💡 Ejecuta: python ../Programs/ZeroPoints_calculations.py")
        exit(1)
    
    # Verificar que los archivos de factores existen
    test_field = 'CenA01'
    factors_file = f"{test_field}_gaia_xp_matches_splus_method.csv"
    if not os.path.exists(factors_file):
        logging.error(f"❌ Archivo de factores no encontrado: {factors_file}")
        logging.error("💡 Ejecuta primero el Script 1: python ../Programs/extract_splus_gaia_xp_corrected_final.py")
        exit(1)
    
    # Modo de operación
    test_mode = True
    test_field = 'CenA01'
    
    if test_mode:
        fields = [test_field]
        logging.info(f"TEST MODE: Processing only field {test_field}")
    else:
        fields = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info(f"FULL MODE: Processing all 24 fields")
    
    try:
        # Inicializar photometry
        photometry = SPLUSGCPhotometryCorrected(
            catalog_path=catalog_path,
            zeropoints_file=zeropoints_file,
            debug=True,
            debug_filter='F660'
        )
        
        all_results = []
        
        # Procesar campos
        for field in tqdm(fields, desc="Processing fields"):
            logging.info(f"\n{'='*60}")
            logging.info(f"PROCESSING FIELD: {field}")
            logging.info(f"{'='*60}")
            
            results = photometry.process_field(field)
            
            if results is not None and len(results) > 0:
                results['FIELD'] = field
                all_results.append(results)
                
                # Guardar resultados individuales del campo
                output_file = f'{field}_gc_photometry_corrected.csv'
                results.to_csv(output_file, index=False, float_format='%.6f')
                logging.info(f"Saved results for {field} to {output_file}")
            else:
                logging.warning(f"No results for field {field}")
        
        # Combinar todos los resultados
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            
            # Rellenar valores NaN apropiadamente
            for filter_name in photometry.filters:
                for aperture_size in photometry.apertures:
                    prefix = f"{filter_name}_{aperture_size}"
                    
                    columns_to_fill = [
                        f'FLUX_CORR_{prefix}', f'FLUXERR_{prefix}',
                        f'MAG_{prefix}', f'MAGERR_{prefix}', f'SNR_{prefix}', f'AP_CORR_{prefix}'
                    ]
                    
                    for col in columns_to_fill:
                        if col in final_results.columns:
                            if 'FLUX' in col and 'ERR' not in col:
                                final_results[col] = final_results[col].fillna(0.0)
                            elif 'MAG' in col or 'ERR' in col:
                                final_results[col] = final_results[col].fillna(99.0)
                            elif 'SNR' in col or 'AP_CORR' in col:
                                final_results[col] = final_results[col].fillna(0.0)
            
            # Crear directorio Results si no existe
            os.makedirs('Results', exist_ok=True)
            
            # Guardar catálogo final
            output_file = 'Results/all_fields_gc_photometry_merged_corrected.csv'
            final_results.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged catalog saved to {output_file}")
            
            # Generar resumen estadístico
            logging.info(f"\n{'='*60}")
            logging.info("PHOTOMETRY SUMMARY")
            logging.info(f"{'='*60}")
            logging.info(f"Total sources in original catalog: {len(photometry.original_catalog)}")
            logging.info(f"Total sources with measurements: {len(final_results)}")
            
            for filter_name in photometry.filters:
                for aperture_size in photometry.apertures:
                    mag_col = f'MAG_{filter_name}_{aperture_size}'
                    if mag_col in final_results.columns:
                        valid_data = final_results[final_results[mag_col] < 50]
                        if len(valid_data) > 0:
                            mean_snr = valid_data[f'SNR_{filter_name}_{aperture_size}'].mean()
                            mean_mag = valid_data[mag_col].mean()
                            mean_ap_corr = valid_data[f'AP_CORR_{filter_name}_{aperture_size}'].mean()
                            n_valid = len(valid_data)
                            
                            # ✅ Indicar si los factores de corrección se están aplicando
                            ap_status = "✅" if mean_ap_corr != 1.0 else "❌"
                            
                            logging.info(f"{filter_name}_{aperture_size}: {n_valid} valid, "
                                       f"Mean SNR: {mean_snr:.1f}, Mean Mag: {mean_mag:.2f}, "
                                       f"Mean AP Corr: {mean_ap_corr:.3f} {ap_status}")
        
        else:
            logging.error("No results were generated for any field")
            
    except Exception as e:
        logging.error(f"Fatal error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
