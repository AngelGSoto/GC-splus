#!/usr/bin/env python3
"""
splus_gc_photometry_corrected.py - Fotometría de cúmulos globulares con aperture correction correcto
Aplica aperture correction solo a los cúmulos, usando factores calculados de estrellas de referencia
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

class SPLUSGCPhotometry:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
        """
        Inicializa la clase de fotometría para cúmulos globulares
        
        Parameters:
        catalog_path: Path al catálogo de cúmulos globulares
        zeropoints_file: Path al archivo de zero points calculados con estrellas SIN corregir
        debug: Modo debug para guardar imágenes
        debug_filter: Filtro para el cual guardar imágenes debug
        """
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Cargar zero points (calculados con magnitudes SIN aperture correction)
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

    def load_aperture_correction_factors(self, field_name):
        """
        Carga los factores de aperture correction del archivo de estrellas de referencia
        
        Parameters:
        field_name: Nombre del campo (ej: 'CenA01')
        
        Returns:
        Dict con factores de corrección por filtro y apertura
        """
        try:
            ref_catalog_path = f"{field_name}_gaia_xp_matches_splus_method.csv"
            if not os.path.exists(ref_catalog_path):
                logging.warning(f"Reference catalog not found: {ref_catalog_path}")
                return self._get_default_corrections()
            
            ref_df = pd.read_csv(ref_catalog_path)
            corrections = {}
            
            for filter_name in self.filters:
                filter_corrections = {}
                for aperture_size in self.apertures:
                    col_name = f'ap_corr_{aperture_size}_{filter_name}'
                    if col_name in ref_df.columns:
                        # Usar la mediana de los factores de todas las estrellas de referencia
                        valid_factors = ref_df[col_name][ref_df[col_name].between(0.5, 5.0)]  # Rango razonable
                        if len(valid_factors) > 0:
                            factor = float(valid_factors.median())
                            filter_corrections[aperture_size] = factor
                        else:
                            filter_corrections[aperture_size] = 1.0
                            logging.warning(f"Invalid factors for {col_name}, using 1.0")
                    else:
                        filter_corrections[aperture_size] = 1.0
                        logging.warning(f"Correction factor column not found: {col_name}")
                
                corrections[filter_name] = filter_corrections
                logging.info(f"{filter_name}: Aperture corrections loaded: {filter_corrections}")
            
            return corrections
            
        except Exception as e:
            logging.error(f"Error loading aperture corrections for {field_name}: {e}")
            return self._get_default_corrections()
    
    def _get_default_corrections(self):
        """Devuelve factores de corrección por defecto (sin corrección)"""
        default_corrections = {}
        for filter_name in self.filters:
            default_corrections[filter_name] = {ap: 1.0 for ap in self.apertures}
        return default_corrections

    def check_splus_background_status(self, data, filter_name):
        """
        Verifica si la imagen SPLUS ya tiene el fondo restado
        
        Returns:
        (needs_correction, bkg_rms)
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
            mask = data > median + 15 * std  # Umbral alto para no enmascarar cúmulos
            
            # Dilatación mínima
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            
            # Boxes grandes para variaciones de gran escala
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=200,  # Muy grande para suavizado
                              filter_size=5,
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=30)
            
            # Solo restar si el modelo muestra estructura significativa
            bkg_range = np.max(bkg.background) - np.min(bkg.background)
            if bkg_range < 2 * std:
                data_corrected = data - np.median(bkg.background)  # Solo restar mediana
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

    def perform_photometry_with_error_propagation(self, data, positions, aperture_radius_px, 
                                                 annulus_inner_px, annulus_outer_px, bkg_rms):
        """
        Realiza fotometría con propagación adecuada de errores
        
        Returns:
        flux, flux_err, snr, aperture_area
        """
        try:
            # Crear aperturas
            apertures = CircularAperture(positions, r=aperture_radius_px)
            annulus = CircularAnnulus(positions, r_in=annulus_inner_px, r_out=annulus_outer_px)
            
            # Áreas para cálculos
            aperture_area = apertures.area
            annulus_area = annulus.area
            
            # Array de errores (dominado por ruido de fondo)
            error_array = np.full_like(data, bkg_rms)
            
            # Fotometría con propagación de errores
            phot_table = aperture_photometry(data, apertures, error=error_array)
            bkg_phot_table = aperture_photometry(data, annulus, error=error_array)
            
            # Extraer valores
            source_flux = phot_table['aperture_sum'].data
            source_flux_err = phot_table['aperture_sum_err'].data
            bkg_flux = bkg_phot_table['aperture_sum'].data
            bkg_flux_err = bkg_phot_table['aperture_sum_err'].data
            
            # Calcular fondo por pixel
            bkg_per_pixel = bkg_flux / annulus_area
            bkg_per_pixel_err = bkg_flux_err / annulus_area
            
            # Restar fondo
            bkg_in_aperture = bkg_per_pixel * aperture_area
            bkg_in_aperture_err = bkg_per_pixel_err * aperture_area
            
            net_flux = source_flux - bkg_in_aperture
            net_flux_err = np.sqrt(source_flux_err**2 + bkg_in_aperture_err**2)
            
            # SNR
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            
            return net_flux, net_flux_err, snr, aperture_area
            
        except Exception as e:
            logging.warning(f"Advanced error propagation failed, using simple method: {e}")
            # Fallback a método simple
            return self._simple_photometry(data, positions, aperture_radius_px, 
                                          annulus_inner_px, annulus_outer_px, bkg_rms)

    def _simple_photometry(self, data, positions, aperture_radius_px, annulus_inner_px, annulus_outer_px, bkg_rms):
        """Método simple de fotometría (fallback)"""
        apertures = CircularAperture(positions, r=aperture_radius_px)
        annulus = CircularAnnulus(positions, r_in=annulus_inner_px, r_out=annulus_outer_px)
        
        phot_table = aperture_photometry(data, apertures)
        bkg_phot_table = aperture_photometry(data, annulus)
        
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        net_flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # Estimación simple de error
        net_flux_err = np.sqrt(np.abs(net_flux) + (apertures.area * bkg_rms**2))
        snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
        
        return net_flux, net_flux_err, snr, apertures.area

    def calculate_magnitudes(self, flux, flux_err, zero_point):
        """
        Calcula magnitudes y errores de manera segura
        """
        # Magnitudes
        mag = np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)
        
        # Errores en magnitud
        mag_err = np.where(flux > 0, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
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
            size = 100  # píxeles alrededor de la fuente
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
        
        # Cargar factores de aperture correction
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
                
                # Obtener WCS
                try:
                    wcs = WCS(header)
                except:
                    logging.warning(f"Could not create WCS for {image_path}")
                    wcs = None
                
                # Verificar y corregir fondo si es necesario
                needs_correction, original_std = self.check_splus_background_status(data, filter_name)
                if needs_correction:
                    data_corrected, bkg_rms = self.apply_conservative_background_correction(data, filter_name)
                else:
                    data_corrected, bkg_rms = data, original_std
                
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
                margin = (max(self.apertures) / self.pixel_scale) * 2  # Margen conservador
                
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
                    
                    # Realizar fotometría
                    flux, flux_err, snr, aperture_area = self.perform_photometry_with_error_propagation(
                        data_corrected, valid_positions, aperture_radius_px,
                        annulus_inner_px, annulus_outer_px, bkg_rms)
                    
                    # Aplicar aperture correction
                    correction_factor = filter_corrections.get(aperture_size, 1.0)
                    flux_corrected = flux * correction_factor
                    flux_err_corrected = flux_err * correction_factor
                    
                    # Calcular magnitudes
                    mag, mag_err = self.calculate_magnitudes(flux_corrected, flux_err_corrected, zero_point)
                    
                    # Guardar resultados para cada fuente válida
                    for i, idx in enumerate(valid_indices):
                        source_idx = results.index[idx]
                        prefix = f"{filter_name}_{aperture_size}"
                        
                        results.loc[source_idx, f'X_{prefix}'] = valid_positions[i, 0]
                        results.loc[source_idx, f'Y_{prefix}'] = valid_positions[i, 1]
                        results.loc[source_idx, f'FLUX_RAW_{prefix}'] = flux[i]
                        results.loc[source_idx, f'FLUX_CORR_{prefix}'] = flux_corrected[i]
                        results.loc[source_idx, f'FLUXERR_{prefix}'] = flux_err_corrected[i]
                        results.loc[source_idx, f'MAG_{prefix}'] = mag[i]
                        results.loc[source_idx, f'MAGERR_{prefix}'] = mag_err[i]
                        results.loc[source_idx, f'SNR_{prefix}'] = snr[i]
                        results.loc[source_idx, f'AP_CORR_{prefix}'] = correction_factor
                    
                    # Imagen debug para una apertura específica
                    if (self.debug and filter_name == self.debug_filter and 
                        aperture_size == 4 and len(valid_positions) > 0):
                        self.save_debug_image(data_corrected, valid_positions, 
                                            aperture_radius_px, field_name, filter_name)
                
                logging.info(f"    Completed {filter_name} with {len(valid_positions)} sources")
                
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        return results

def main():
    """Función principal"""
    # Configuración
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_splus_method.csv'
    
    # Verificar archivos de entrada
    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        exit(1)
    
    if not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
        exit(1)
    
    # Modo de operación
    test_mode = True  # Cambiar a False para procesar todos los campos
    test_field = 'CenA01'
    
    if test_mode:
        fields = [test_field]
        logging.info(f"TEST MODE: Processing only field {test_field}")
    else:
        fields = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info(f"FULL MODE: Processing all 24 fields")
    
    try:
        # Inicializar photometry
        photometry = SPLUSGCPhotometry(
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
                    
                    # Columnas a rellenar
                    columns_to_fill = [
                        f'FLUX_RAW_{prefix}', f'FLUX_CORR_{prefix}', f'FLUXERR_{prefix}',
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
                            
                            logging.info(f"{filter_name}_{aperture_size}: {n_valid} valid, "
                                       f"Mean SNR: {mean_snr:.1f}, Mean Mag: {mean_mag:.2f}, "
                                       f"Mean AP Corr: {mean_ap_corr:.3f}")
        
        else:
            logging.error("No results were generated for any field")
            
    except Exception as e:
        logging.error(f"Fatal error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
