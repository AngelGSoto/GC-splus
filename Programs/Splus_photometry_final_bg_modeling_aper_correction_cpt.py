#!/usr/bin/env python3
"""
SPLUS GC Photometry - Versión corregida para consistencia con zero points
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from tqdm import tqdm
import warnings
import os
import traceback
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy import ndimage

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSPhotometryConsistent:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Cargar zero points
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {
            row['field']: {filt: row[filt] for filt in ['F378','F395','F410','F430','F515','F660','F861']}
            for _, row in self.zeropoints_df.iterrows()
        }
        
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        # Aperturas consistentes con zero points (solo 3 arcsec para magnitudes finales)
        self.apertures = [3, 4, 5, 6]  # Diameters in arcsec
        self.debug = debug
        self.debug_filter = debug_filter
        
        # Identificar columnas del catálogo
        self.ra_col = next((col for col in ['RAJ2000', 'RA'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RAJ2000/DEJ2000 or RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain T17ID or ID column")
        
        # Cache for aperture corrections
        self.aperture_corrections = {}
        self.pixel_scale = 0.55  # Default S-PLUS pixel scale

    def model_background_residuals(self, data, field_name, filter_name):
        """
        IDÉNTICO al usado en el script de zero points
        """
        try:
            # Estadísticas básicas de la imagen
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            # Para imágenes S-PLUS ya restadas, ser conservador
            if std < 1.0:
                logging.info(f"Imagen plana (std={std:.3f}). Saltando modelado de fondo.")
                return data, std
            
            # Crear máscara para objetos brillantes (umbral más alto)
            mask = data > median + 10 * std
            
            # Dilatar máscara moderadamente (5x5 como en zero points)
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5)))
            
            # PARÁMETROS IDÉNTICOS a zero points
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=50,    # ← MISMO que zero points
                              filter_size=3,  # ← MISMO que zero points  
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=20)
            
            # Solo restar si el modelo de fondo tiene estructura significativa
            bkg_range = np.max(bkg.background) - np.min(bkg.background)
            if bkg_range < 3 * std:
                data_corrected = data - np.median(bkg.background)
            else:
                data_corrected = data - bkg.background
            
            bkg_rms = bkg.background_rms_median
            logging.info(f"Fondo tratado: std_original={std:.3f}")
            return data_corrected, bkg_rms
            
        except Exception as e:
            logging.warning(f"Tratamiento de fondo falló: {e}. Usando imagen original.")
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            return data, std

    def calculate_aperture_correction_consistent(self, data, header, star_positions, filter_name, target_aperture_diameter):
        """
        CORRECCIÓN DE APERTURA CONSISTENTE con método de zero points
        """
        try:
            pixscale = header.get('PIXSCALE', self.pixel_scale)
            
            # Definir aperturas para curva de crecimiento (igual que zero points)
            aperture_diameters = np.linspace(1, 50, 32)
            aperture_radii_px = (aperture_diameters / 2) / pixscale
            
            growth_fluxes = []
            for radius in aperture_radii_px:
                apertures = CircularAperture(star_positions, r=radius)
                phot_table = aperture_photometry(data, apertures)
                growth_fluxes.append(phot_table['aperture_sum'].data)
            
            growth_fluxes = np.array(growth_fluxes)
            total_fluxes = growth_fluxes[-1]  # Flujo en apertura grande (50 arcsec)
            valid_mask = total_fluxes > 0
            
            if np.sum(valid_mask) < 5:
                logging.warning(f"Estrellas insuficientes para corrección de apertura")
                return 1.0, 0, 0.0
            
            # Curva de crecimiento normalizada
            normalized_curves = growth_fluxes[:, valid_mask] / total_fluxes[valid_mask]
            mean_growth = np.median(normalized_curves, axis=1)
            
            # Encontrar la fracción para la apertura específica
            target_idx = np.argmin(np.abs(aperture_diameters - target_aperture_diameter))
            fraction_in_target_aperture = mean_growth[target_idx]
            
            if fraction_in_target_aperture <= 0:
                return 1.0, 0, 0.0
            
            aperture_correction_flux = 1.0 / fraction_in_target_aperture
            n_stars = np.sum(valid_mask)
            
            logging.info(f"{filter_name} {target_aperture_diameter}arcsec: Corrección = {aperture_correction_flux:.3f} (n={n_stars})")
            return aperture_correction_flux, n_stars, fraction_in_target_aperture
            
        except Exception as e:
            logging.error(f"Error en corrección de apertura: {e}")
            return 1.0, 0, 0.0

    def load_reference_stars_for_aperture_correction(self, field_name):
        """
        Cargar estrellas de referencia usadas para zero points
        """
        ref_catalog_path = f'{field_name}_gaia_xp_matches_splus_method.csv'
        if not os.path.exists(ref_catalog_path):
            logging.warning(f"No se encontró catálogo de referencia: {ref_catalog_path}")
            return None, None
        
        try:
            ref_catalog = pd.read_csv(ref_catalog_path)
            # Obtener posiciones de las estrellas de referencia
            if 'x_pix' in ref_catalog.columns and 'y_pix' in ref_catalog.columns:
                positions = ref_catalog[['x_pix', 'y_pix']].values
                return positions, ref_catalog
            else:
                logging.warning("Catálogo de referencia no tiene coordenadas pixel")
                return None, None
        except Exception as e:
            logging.error(f"Error cargando estrellas de referencia: {e}")
            return None, None

    def get_field_center_from_header(self, field_name, filter_name='F660'):
        """Obtener centro del campo desde header"""
        possible_files = [
            os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz"),
            os.path.join(field_name, f"{field_name}_{filter_name}.fits")
        ]
        for sci_file in possible_files:
            if os.path.exists(sci_file):
                try:
                    with fits.open(sci_file) as hdul:
                        if len(hdul) > 1 and hdul[1].data is not None:
                            header = hdul[1].header
                        else:
                            header = hdul[0].header
                        ra_center = header.get('CRVAL1')
                        dec_center = header.get('CRVAL2')
                        if ra_center is not None and dec_center is not None:
                            return ra_center, dec_center
                except Exception as e:
                    logging.warning(f"Error reading {sci_file}: {e}")
                    continue
        return None, None

    def is_source_in_field(self, source_ra, source_dec, field_ra, field_dec, field_radius_deg=0.84):
        """Verificar si fuente está en campo"""
        coord1 = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
        coord2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        separation = coord1.separation(coord2).degree
        return separation <= field_radius_deg

    def safe_magnitude_calculation(self, flux, zero_point):
        """Cálculo seguro de magnitudes"""
        return np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)

    def process_field(self, field_name):
        """Procesar campo completo con metodología consistente"""
        logging.info(f"Processing field {field_name}")
        
        # Determinar centro del campo
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None

        # Preparar catálogo
        self.catalog[self.ra_col] = self.catalog[self.ra_col].astype(float)
        self.catalog[self.dec_col] = self.catalog[self.dec_col].astype(float)

        # Filtrar fuentes en el campo
        in_field_mask = [
            self.is_source_in_field(row[self.ra_col], row[self.dec_col], field_ra, field_dec)
            for _, row in self.catalog.iterrows()
        ]
        field_sources = self.catalog[in_field_mask].copy()
        field_sources = field_sources.reset_index(drop=True)
        logging.info(f"Found {len(field_sources)} sources in field {field_name}")
        if len(field_sources) == 0:
            return None

        results = field_sources.copy()

        # Procesar cada filtro
        for filter_name in ['F378','F395','F410','F430','F515','F660','F861']:
            logging.info(f"  Processing filter {filter_name}")
            
            # Encontrar archivo de imagen
            image_path = None
            for ext in ['.fits.fz', '.fits']:
                candidate = os.path.join(field_name, f"{field_name}_{filter_name}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if not image_path:
                logging.warning(f"    No image found for {field_name} {filter_name}. Skipping.")
                continue

            try:
                with fits.open(image_path) as hdul:
                    if len(hdul) > 1 and hdul[1].data is not None:
                        hdu = hdul[1]
                    else:
                        hdu = hdul[0]
                    header = hdu.header
                    data = hdu.data.astype(float)

                # Extraer valores del header
                gain = float(header.get('GAIN', header.get('CCDGAIN', 1.0)))
                rdnoise = float(header.get('RDNOISE', 0.0))
                exptime = float(header.get('EXPTIME', header.get('TEXPOSED', 1.0)))
                pixscale = float(header.get('PIXSCALE', self.pixel_scale))
                saturate_level = float(header.get('SATURATE', np.inf))

                wcs = WCS(header)

                # 1. APLICAR CORRECCIÓN DE FONDO (IDÉNTICA a zero points)
                data_corrected, bkg_rms = self.model_background_residuals(data, field_name, filter_name)

                # 2. CARGAR ESTRELLAS DE REFERENCIA para corrección de apertura
                ref_positions, ref_catalog = self.load_reference_stars_for_aperture_correction(field_name)
                
                # 3. CALCULAR CORRECCIONES DE APERTURA para cada tamaño
                ap_corrections = {}
                if ref_positions is not None and len(ref_positions) > 5:
                    for ap_diam in self.apertures:
                        ap_corr_flux, n_stars, fraction = self.calculate_aperture_correction_consistent(
                            data_corrected, header, ref_positions, filter_name, ap_diam)
                        ap_corrections[ap_diam] = {
                            'factor': ap_corr_flux,
                            'n_stars': n_stars,
                            'fraction': fraction
                        }
                else:
                    logging.warning(f"Usando corrección de apertura por defecto para {field_name} {filter_name}")
                    for ap_diam in self.apertures:
                        ap_corrections[ap_diam] = {'factor': 1.0, 'n_stars': 0, 'fraction': 1.0}

                # Convertir coordenadas mundo a pixel
                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                x, y = wcs.world_to_pixel(coords)
                positions_all = np.column_stack((x, y))

                # Filtrar posiciones cerca de bordes
                valid_positions = []
                valid_indices = []
                height, width = data_corrected.shape
                max_ap_radius_px = (max(self.apertures) / 2.0) / pixscale
                
                for i, (x_pos, y_pos) in enumerate(positions_all):
                    if not np.isfinite(x_pos) or not np.isfinite(y_pos):
                        continue
                    if (x_pos >= max_ap_radius_px and x_pos <= width - max_ap_radius_px and
                        y_pos >= max_ap_radius_px and y_pos <= height - max_ap_ap_radius_px):
                        valid_positions.append([x_pos, y_pos])
                        valid_indices.append(i)

                if len(valid_positions) == 0:
                    logging.warning(f"No valid positions for photometry in {field_name} {filter_name}")
                    continue

                valid_positions = np.array(valid_positions)

                # Fotometría para cada apertura
                for ap_diam in self.apertures:
                    ap_radius_px = (ap_diam / 2.0) / pixscale
                    
                    # Apertura y anillo de fondo (6-9 arcsec, igual que zero points)
                    apertures = CircularAperture(valid_positions, r=ap_radius_px)
                    ann_in_px = 6.0 / pixscale
                    ann_out_px = 9.0 / pixscale
                    annulus = CircularAnnulus(valid_positions, r_in=ann_in_px, r_out=ann_out_px)

                    # Fotometría
                    phot_table = aperture_photometry(data_corrected, apertures)
                    bkg_phot_table = aperture_photometry(data_corrected, annulus)

                    # Calcular fondo y flujo
                    annulus_area = annulus.area
                    bkg_mean_per_source = bkg_phot_table['aperture_sum'] / annulus_area
                    flux_uncorrected = phot_table['aperture_sum'] - (bkg_mean_per_source * apertures.area)

                    # APLICAR CORRECCIÓN DE APERTURA
                    ap_corr_info = ap_corrections[ap_diam]
                    flux_corrected = flux_uncorrected * ap_corr_info['factor']

                    # Cálculo de errores
                    flux_err = np.sqrt(np.abs(flux_corrected) + (apertures.area * bkg_rms**2))

                    # Convertir a magnitudes usando zero point
                    try:
                        zero_point = self.zeropoints[field_name][filter_name]
                    except Exception:
                        logging.warning(f"No zero point for {field_name} {filter_name}")
                        zero_point = None

                    if zero_point is not None:
                        mag = self.safe_magnitude_calculation(flux_corrected, zero_point)
                        mag_err = np.where(flux_corrected > 0,
                                         (2.5 / np.log(10)) * (flux_err / flux_corrected),
                                         99.0)
                    else:
                        mag = np.full(len(flux_corrected), 99.0)
                        mag_err = np.full(len(flux_corrected), 99.0)

                    snr = np.where(flux_err > 0, flux_corrected / flux_err, 0.0)

                    # Guardar resultados
                    for i_local, idx in enumerate(valid_indices):
                        results.loc[results.index[idx], f'FLUX_{filter_name}_{ap_diam}'] = float(flux_corrected[i_local])
                        results.loc[results.index[idx], f'FLUXERR_{filter_name}_{ap_diam}'] = float(flux_err[i_local])
                        results.loc[results.index[idx], f'MAG_{filter_name}_{ap_diam}'] = float(mag[i_local])
                        results.loc[results.index[idx], f'MAGERR_{filter_name}_{ap_diam}'] = float(mag_err[i_local])
                        results.loc[results.index[idx], f'SNR_{filter_name}_{ap_diam}'] = float(snr[i_local])
                        results.loc[results.index[idx], f'APCORR_{filter_name}_{ap_diam}'] = ap_corr_info['factor']
                        results.loc[results.index[idx], f'APCORR_NSTARS_{filter_name}_{ap_diam}'] = ap_corr_info['n_stars']

            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue

        return results


# ---- Main Execution ----
if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'all_fields_zero_points_splus_format_corrected.csv'  # Usar zero points corregidos

    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        exit(1)
    elif not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
        exit(1)

    # Modo prueba
    test_mode = True
    test_field = 'CenA01'

    if test_mode:
        fields = [test_field]
        logging.info(f"Test mode activado. Procesando: {test_field}")
    else:
        fields = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info("Procesando todos los campos")

    try:
        all_results = []
        photometry = SPLUSPhotometryConsistent(catalog_path, zeropoints_file, debug=True, debug_filter='F660')
        
        for field in tqdm(fields, desc="Processing fields"):
            if not os.path.exists(field):
                logging.warning(f"Field directory {field} does not exist. Skipping.")
                continue
                
            results = photometry.process_field(field)
            if results is not None and len(results) > 0:
                results['FIELD'] = field
                all_results.append(results)
                output_file = f'{field}_gc_photometry_consistent.csv'
                results.to_csv(output_file, index=False, float_format='%.6f')
                logging.info(f"Saved results for {field} to {output_file}")
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            
            # Merge con catálogo original
            final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
            merged_catalog = photometry.original_catalog.merge(
                final_results, on='T17ID', how='left', suffixes=('', '_y')
            )
            
            # Limpiar columnas duplicadas
            for col in merged_catalog.columns:
                if col.endswith('_y'):
                    merged_catalog.drop(col, axis=1, inplace=True)
                    
            output_file = 'all_fields_gc_photometry_consistent.csv'
            merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged results saved to {output_file}")

            # Estadísticas
            logging.info(f"Total sources in original catalog: {len(photometry.original_catalog)}")
            logging.info(f"Total sources with measurements: {len(final_results)}")
            
            for filter_name in ['F378','F395','F410','F430','F515','F660','F861']:
                for aperture in [3]:  # Solo reportar para 3 arcsec (usado en zero points)
                    mag_col = f'MAG_{filter_name}_{aperture}'
                    if mag_col in merged_catalog.columns:
                        valid_data = merged_catalog[merged_catalog[mag_col] < 50]
                        if len(valid_data) > 0:
                            mean_snr = valid_data[f'SNR_{filter_name}_{aperture}'].mean()
                            mean_mag = valid_data[mag_col].mean()
                            logging.info(f"{filter_name}_{aperture}: {len(valid_data)} valid, Mean SNR: {mean_snr:.1f}, Mean Mag: {mean_mag:.2f}")

    except Exception as e:
        logging.error(f"Execution error: {e}")
        traceback.print_exc()
