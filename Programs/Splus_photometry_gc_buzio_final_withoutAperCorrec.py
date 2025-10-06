#!/usr/bin/env python3
"""
Splus_photometry_gc_simple_robust.py
VERSIÓN SIMPLIFICADA - Fotometría robusta sin corrección de apertura
MEJORADA: manejo correcto de errores sin weight map, ajustes de calidad
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.stats import SigmaClip, sigma_clipped_stats
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm
import warnings
import os
import logging
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

# Configuración de estilo para plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def calculate_seeing_from_psf(header):
    """Estimar seeing desde header o usar valor por defecto"""
    fwhm = header.get('FWHMMEAN', header.get('FWHM', header.get('SEEING', 1.8)))
    return float(fwhm)

def analyze_background_variability(data):
    """Analizar variabilidad del fondo de forma simple"""
    try:
        p10, p90 = np.percentile(data, [10, 90])
        data_clipped = data[(data >= p10) & (data <= p90)]
        
        if len(data_clipped) == 0:
            return False
        
        background_median = np.median(data_clipped)
        background_std = np.std(data_clipped)
        variability_ratio = background_std / background_median if background_median > 0 else 0
        needs_unsharp = variability_ratio > 0.03 and background_median > 0.05
        
        logging.info(f"Background: median={background_median:.3f}, std={background_std:.3f}, ratio={variability_ratio:.3f}, unsharp={needs_unsharp}")
        return needs_unsharp
        
    except Exception as e:
        logging.warning(f"Background analysis failed: {e}")
        return False

def create_unsharp_mask(data, median_size=25, sigma=5):
    """Unsharp masking simple"""
    try:
        p05, p95 = np.percentile(data, [5, 95])
        data_clipped = np.clip(data, p05, p95)
        median_filtered = median_filter(data_clipped, size=median_size)
        gaussian_smoothed = gaussian_filter(median_filtered, sigma=sigma)
        unsharp_mask = data_clipped - gaussian_smoothed
        return unsharp_mask
    except Exception as e:
        logging.error(f"Unsharp masking failed: {e}")
        return data

def save_unsharp_diagnostic(original, unsharp, field_name, filter_name):
    """Guardar diagnóstico de unsharp masking"""
    try:
        os.makedirs("unsharp_diagnostics", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmin_o, vmax_o = np.percentile(original, [5, 95])
        vmin_u, vmax_u = np.percentile(unsharp, [5, 95])
        axes[0].imshow(original, vmin=vmin_o, vmax=vmax_o, cmap='gray', origin='lower')
        axes[0].set_title(f'Original: {field_name} {filter_name}')
        axes[1].imshow(unsharp, vmin=vmin_u, vmax=vmax_u, cmap='gray', origin='lower')
        axes[1].set_title(f'Unsharp Masked: {field_name} {filter_name}')
        plt.tight_layout()
        plt.savefig(f"unsharp_diagnostics/{field_name}_{filter_name}_unsharp.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Error saving unsharp diagnostic: {e}")

def load_weight_map(weight_path, data_shape):
    """Cargar weight map y convertir a error map"""
    try:
        with fits.open(weight_path) as whdul:
            for whdu in whdul:
                if whdu.data is not None:
                    weight_data = whdu.data.astype(float)
                    break
            else:
                return None
        
        if weight_data.shape != data_shape:
            logging.warning(f"Weight map shape mismatch: {weight_data.shape} vs {data_shape}")
            return None
        
        valid_weights = (weight_data > 0) & np.isfinite(weight_data)
        if np.sum(valid_weights) / weight_data.size < 0.5:
            logging.warning("Less than 50% valid weights")
            return None
        
        error_map = np.full_like(weight_data, np.inf)
        error_map[valid_weights] = 1.0 / np.sqrt(weight_data[valid_weights])
        median_error = np.median(error_map[valid_weights])
        logging.info(f"Weight map loaded: median error = {median_error:.4f}")
        return error_map
        
    except Exception as e:
        logging.error(f"Error loading weight map: {e}")
        return None

def robust_photometry(data, positions, aperture_radius, annulus_inner, annulus_outer, error_map=None):
    """
    Fotometría robusta con soporte para weight maps o errores de Poisson
    """
    try:
        apertures = CircularAperture(positions, r=aperture_radius)
        annulus = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)
        
        # Si hay error_map, lo usamos; si no, no lo pasamos
        if error_map is not None:
            phot_table = aperture_photometry(data, apertures, error=error_map)
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = phot_table['aperture_sum_err'].data
        else:
            phot_table = aperture_photometry(data, apertures)
            raw_flux = phot_table['aperture_sum'].data
            raw_flux_err = None  # Lo calcularemos manualmente
        
        bkg_median = np.zeros(len(positions))
        bkg_std = np.zeros(len(positions))
        
        for i, pos in enumerate(positions):
            try:
                mask = annulus.to_mask(method='center')[i]
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                
                if len(annulus_data_1d) > 10:
                    _, bkg_med, bkg_sig = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)
                else:
                    bkg_med = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                    bkg_sig = np.std(annulus_data_1d) if len(annulus_data_1d) > 0 else 1.0
                
                bkg_median[i] = bkg_med
                bkg_std[i] = bkg_sig
                
            except Exception as e:
                bkg_median[i] = 0.0
                bkg_std[i] = 1.0
        
        net_flux = raw_flux - (bkg_median * apertures.area)
        
        # Cálculo robusto del error
        if error_map is not None:
            # Error ya incluye incertidumbre de fondo en raw_flux_err, pero refinamos con fondo local
            bkg_flux_error = bkg_std * np.sqrt(apertures.area)
            net_flux_err = np.sqrt(raw_flux_err**2 + bkg_flux_error**2)
        else:
            # Estimación de Poisson + fondo
            # Asegurar que net_flux no sea negativo para sqrt
            poisson_term = np.abs(net_flux)  # Aproximación para Poisson
            bkg_term = apertures.area * bkg_std**2
            net_flux_err = np.sqrt(poisson_term + bkg_term)
            # Evitar NaN o inf
            net_flux_err = np.where(np.isfinite(net_flux_err), net_flux_err, np.inf)
        
        return net_flux, net_flux_err, bkg_median
        
    except Exception as e:
        logging.error(f"Photometry failed: {e}")
        n = len(positions)
        return np.zeros(n), np.full(n, np.inf), np.zeros(n)

def process_single_filter_simple(args):
    """
    Procesamiento simple de un filtro sin corrección de apertura
    """
    try:
        (field_name, filter_name, valid_positions, valid_indices, 
         zeropoints, pixel_scale, debug) = args
        
        logging.info(f"Processing {filter_name} (SIMPLE METHOD)")
        
        def find_files(field_name, filter_name):
            base_path = os.path.join(field_name, f"{field_name}_{filter_name}")
            for ext in [".fits.fz", ".fits"]:
                if os.path.exists(base_path + ext):
                    image_path = base_path + ext
                    break
            else:
                return None, None
            weight_path = None
            for pattern in [".weight.fits.fz", ".weight.fits", "_weight.fits.fz", "_weight.fits"]:
                if os.path.exists(base_path + pattern):
                    weight_path = base_path + pattern
                    break
            return image_path, weight_path

        image_path, weight_path = find_files(field_name, filter_name)
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

        error_map = None
        if weight_path:
            error_map = load_weight_map(weight_path, data_original.shape)
            if error_map is not None:
                logging.info(f"✅ {filter_name}: Using weight map for errors")
            else:
                logging.warning(f"{filter_name}: Weight map invalid, using Poisson errors")
        else:
            logging.warning(f"{filter_name}: No weight map found, using Poisson errors")

        needs_unsharp = analyze_background_variability(data_original)
        if needs_unsharp:
            data_processed = create_unsharp_mask(data_original)
            logging.info(f"✅ {filter_name}: Applied unsharp masking")
            if debug:
                save_unsharp_diagnostic(data_original, data_processed, field_name, filter_name)
        else:
            data_processed = data_original
            logging.info(f"✅ {filter_name}: No unsharp masking needed")

        aperture_diams = [2.0, 3.0]
        results = {'indices': valid_indices}
        
        for aperture_diam in aperture_diams:
            aperture_radius = (aperture_diam / 2) / pixel_scale
            annulus_inner = (3.0 / 2) / pixel_scale
            annulus_outer = (4.0 / 2) / pixel_scale
            
            net_flux, net_flux_err, background_levels = robust_photometry(
                data_processed, valid_positions, aperture_radius, annulus_inner, annulus_outer, error_map)
            
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            zero_point = zeropoints.get(field_name, {}).get(filter_name, 0.0)
            
            valid_flux = (net_flux > 1e-10) & (net_flux_err > 0) & np.isfinite(net_flux) & np.isfinite(net_flux_err)
            mag_inst = np.where(valid_flux, -2.5 * np.log10(net_flux), 99.0)
            mag = np.where(valid_flux, mag_inst + zero_point, 99.0)
            mag_err = np.where(valid_flux, (2.5 / np.log(10)) * (net_flux_err / net_flux), 99.0)
            
            # Ajuste: límite inferior más realista (12.0 en lugar de 10.0)
            good_quality = (mag >= 12.0) & (mag <= 30.0) & (mag_err < 1.0) & (snr > 2.0)
            final_mask = valid_flux & good_quality
            
            mag = np.where(final_mask, mag, 99.0)
            mag_err = np.where(final_mask, mag_err, 99.0)
            snr = np.where(final_mask, snr, 0.0)
            
            prefix = f"{filter_name}_{aperture_diam:.0f}"
            results[f'FLUX_{prefix}'] = net_flux
            results[f'FLUXERR_{prefix}'] = net_flux_err
            results[f'MAG_{prefix}'] = mag
            results[f'MAGERR_{prefix}'] = mag_err
            results[f'SNR_{prefix}'] = snr
            results[f'QUALITY_{prefix}'] = final_mask.astype(int)
            results[f'HAS_WEIGHT_{prefix}'] = np.full(len(net_flux), error_map is not None)
            results[f'UNSHARP_APPLIED_{prefix}'] = np.full(len(net_flux), needs_unsharp)
        
        valid_count = np.sum(final_mask)
        median_error = np.median(mag_err[final_mask]) if np.sum(final_mask) > 0 else 0.0
        logging.info(f"✅ {filter_name}: Completed - {valid_count} valid sources, median error={median_error:.3f}")
        return results, filter_name
        
    except Exception as e:
        logging.error(f"❌ {filter_name}: Processing failed - {e}")
        traceback.print_exc()
        return None, filter_name

class SPLUSGCPhotometrySimple:
    def __init__(self, catalog_path, zeropoints_file, debug=False):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        for _, row in self.zeropoints_df.iterrows():
            field = row['field']
            self.zeropoints[field] = {}
            for filt in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                self.zeropoints[field][filt] = row[filt]
        
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55
        self.debug = debug
        
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        logging.info(f"Using SIMPLE ROBUST METHOD without aperture correction")

    def get_field_center_from_header(self, field_name, filter_name='F660'):
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
        if field_ra is None or field_dec is None:
            return False
        c1 = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        c2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        return c1.separation(c2).degree <= radius

    def process_field_simple(self, field_name):
        logging.info(f"🎯 Processing field {field_name} (SIMPLE METHOD)")
        start_time = time.time()
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist")
            return None
        
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}")
            return None
        
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

        results_df = field_sources.copy()
        successful_filters = 0
        
        args_list = []
        for filt in self.filters:
            args = (
                field_name, 
                filt, 
                valid_positions, 
                valid_indices,
                self.zeropoints,
                self.pixel_scale,
                self.debug
            )
            args_list.append(args)
        
        successful_filters = self._process_serial_simple(args_list, results_df)
        
        elapsed_time = time.time() - start_time
        logging.info(f"🎯 Field {field_name} completed: {successful_filters}/{len(self.filters)} filters "
                   f"in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        
        if successful_filters > 0:
            results_df['FIELD'] = field_name
            return results_df
        else:
            return None

    def _process_serial_simple(self, args_list, results_df):
        successful_filters = 0
        for args in tqdm(args_list, desc="Processing filters"):
            try:
                result, filter_name = process_single_filter_simple(args)
                if result is not None:
                    self._integrate_results(results_df, result, filter_name)
                    successful_filters += 1
                    logging.info(f"✅ {filter_name}: Results integrated")
                else:
                    logging.warning(f"❌ {filter_name}: No results")
            except Exception as e:
                logging.error(f"❌ Error processing {args[1]}: {e}")
                continue
        return successful_filters

    def _integrate_results(self, results_df, result, filter_name):
        temp_df = pd.DataFrame(result)
        temp_df.set_index('indices', inplace=True)
        for col in temp_df.columns:
            if col != 'indices':
                results_df.loc[temp_df.index, col] = temp_df[col].values

    def find_image_file(self, field_name, filter_name):
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None

def main():
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_3arcsec.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    test_mode = True
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    photometry = SPLUSGCPhotometrySimple(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_simple(field)
        if results is not None and len(results) > 0:
            all_results.append(results)
            output_file = f'{field}_gc_photometry_simple.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results to {output_file}")
            
            logging.info(f"📊 Field {field} detailed statistics (SIMPLE METHOD):")
            for filter_name in photometry.filters:
                for aperture in ['2', '3']:
                    mag_col = f'MAG_{filter_name}_{aperture}'
                    err_col = f'MAGERR_{filter_name}_{aperture}'
                    snr_col = f'SNR_{filter_name}_{aperture}'
                    weight_col = f'HAS_WEIGHT_{filter_name}_{aperture}'
                    
                    if mag_col in results.columns:
                        valid_mask = results[mag_col] < 99.0
                        valid_count = np.sum(valid_mask)
                        if valid_count > 0:
                            valid_mags = results[mag_col][valid_mask]
                            valid_errors = results[err_col][valid_mask]
                            valid_snr = results[snr_col][valid_mask]
                            has_weight = results[weight_col].iloc[0] if weight_col in results.columns else False
                            median_mag = np.median(valid_mags)
                            median_error = np.median(valid_errors)
                            median_snr = np.median(valid_snr)
                            weight_status = "with weights" if has_weight else "no weights"
                            logging.info(f"  {filter_name} {aperture}\": {valid_count} valid {weight_status}, "
                                       f"mag={median_mag:.2f}±{median_error:.3f}, SNR={median_snr:.1f}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        os.makedirs("Results", exist_ok=True)
        output_file = 'Results/all_fields_gc_photometry_simple.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        logging.info("\n📊 FINAL QUALITY ANALYSIS (SIMPLE METHOD):")
        total_sources = 0
        good_sources = 0
        
        for filter_name in photometry.filters:
            for aperture in ['2', '3']:
                mag_col = f'MAG_{filter_name}_{aperture}'
                err_col = f'MAGERR_{filter_name}_{aperture}'
                weight_col = f'HAS_WEIGHT_{filter_name}_{aperture}'
                
                if mag_col in final_results.columns:
                    valid_mask = final_results[mag_col] < 99.0
                    valid_count = np.sum(valid_mask)
                    total_sources += valid_count
                    if valid_count > 0:
                        valid_errors = final_results[err_col][valid_mask]
                        has_weights = final_results[weight_col].iloc[0] if weight_col in final_results.columns else False
                        completeness = valid_count / len(final_results) * 100
                        median_error = np.median(valid_errors)
                        low_error_ratio = np.sum(valid_errors < 0.1) / valid_count * 100
                        good_sources += np.sum(valid_errors < 0.2)
                        weight_status = "WEIGHTS" if has_weights else "NO WEIGHTS"
                        logging.info(f"  {filter_name} {aperture}\": {valid_count} ({completeness:.1f}%) {weight_status} "
                                   f"median_error={median_error:.3f}, errors<0.1: {low_error_ratio:.1f}%")
        
        good_ratio = good_sources / total_sources * 100 if total_sources > 0 else 0
        logging.info(f"\n📈 SUMMARY: {good_sources}/{total_sources} ({good_ratio:.1f}%) sources with errors < 0.2 mag")

if __name__ == "__main__":
    main()
