#!/usr/bin/env python3
"""
Script actualizado de fotometría de apertura para S-PLUS (adaptado al header que mostraste).
Principales cambios/beneficios:
- No se normaliza por EXPTIME (tus flujos están en ADU).
- Usa GAIN (e-/ADU) para calcular varianzas/errores.
- Calcula corrección de apertura empírica (curve-of-growth) cuando hay suficientes estrellas.
- Estimación de fondo por fuente (annulus local) + modelado de residuales a gran escala.
- Chequeo de bordes usa radios en píxeles (no diámetros por error x2).
- Filtrado por saturación usando SATURATE del header.
- Guarda metadatos útiles (GAIN, EXPTIME, AP_CORR, N_STARS usados).
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
from photutils.detection import DAOStarFinder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy import ndimage

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')


class SPLUSPhotometry:
    def __init__(self, catalog_path, zeropoints_file, background_box_size=100, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        # Espera columnas: field, F378, F395, ...
        self.zeropoints = {
            row['field']: {filt: row[filt] for filt in ['F378','F395','F410','F430','F515','F660','F861']}
            for _, row in self.zeropoints_df.iterrows()
        }
        
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        # Apertures originales (tus valores) se interpretan como DIAMETROS en arcsec.
        # Internamente convertimos a radios en px cuando sea necesario.
        self.apertures = [3, 4, 5, 6]  # DIAMETER in arcsec (mantengo tu convención para compatibilidad)
        self.background_box_size = background_box_size
        self.debug = debug
        self.debug_filter = debug_filter
        
        self.ra_col = next((col for col in ['RAJ2000', 'RA'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RAJ2000/DEJ2000 or RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain T17ID or ID column")
        
        # Cache for aperture corrections
        self.aperture_corrections = {}

    def get_field_center_from_header(self, field_name, filter_name='F660'):
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
        coord1 = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
        coord2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        separation = coord1.separation(coord2).degree
        return separation <= field_radius_deg

    def model_background_residuals(self, data, field_name, filter_name):
        """
        Model and subtract large-scale background residuals (useful for galaxy halos).
        """
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            mask = data > median + 5 * std
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((15, 15)))
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=self.background_box_size, 
                              filter_size=5,
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=10)
            data_corrected = data - bkg.background
            bkg_rms = bkg.background_rms_median

            if self.debug and filter_name == self.debug_filter:
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                vmin = median - 3 * std
                vmax = median + 3 * std
                plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                plt.title('Original (already bkg-subtracted)')
                plt.colorbar()
                plt.subplot(132)
                plt.imshow(bkg.background, origin='lower', cmap='gray')
                plt.title('Modeled Background Residuals')
                plt.colorbar()
                plt.subplot(133)
                plt.imshow(data_corrected, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                plt.title('After Residual Subtraction')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{field_name}_{filter_name}_background_residuals.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved background residuals image for {field_name} {filter_name}")

            logging.info(f"Background residuals modeled with box_size={self.background_box_size}")
            return data_corrected, bkg_rms
        except Exception as e:
            logging.warning(f"Background residual modeling failed: {e}. Using original image.")
            return data, np.nanstd(data)

    def compute_empirical_aperture_correction(self, data, header, mask_positions=None,
                                             small_radius_arcsec=None, large_radius_arcsec=None):
        """
        Compute aperture correction (flux_large / flux_small) using bright isolated stars.
        Returns median_corr, corr_std, n_stars_used
        """
        try:
            pixscale = float(header.get('PIXSCALE', self.pixel_scale))
            fwhm_arcsec = float(header.get('FWHMMEAN', 1.5))
        except Exception:
            pixscale = self.pixel_scale
            fwhm_arcsec = 1.5

        if small_radius_arcsec is None:
            small_radius_arcsec = 1.0 * fwhm_arcsec  # radius
        if large_radius_arcsec is None:
            large_radius_arcsec = 4.0 * fwhm_arcsec  # radius for "total"

        small_r_px = small_radius_arcsec / pixscale
        large_r_px = large_radius_arcsec / pixscale

        # Detect stars unless user provided positions
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        threshold = max(5.0 * std, median + 5.0*std)
        fwhm_pix = max(1.0, fwhm_arcsec / pixscale)
        try:
            daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=threshold)
            sources = daofind(data - median)
        except Exception:
            sources = None

        if sources is None or len(sources) < 6:
            # Fallback: if mask_positions provided, use them, else return None
            if mask_positions is None:
                logging.warning("Pocas o ninguna estrella detectada para corrección empírica.")
                return 1.0, 0.0, 0
            else:
                star_positions = mask_positions
        else:
            star_positions = np.column_stack((sources['xcentroid'], sources['ycentroid']))

        # Build apertures
        small_ap = CircularAperture(star_positions, r=small_r_px)
        large_ap = CircularAperture(star_positions, r=large_r_px)

        small_tab = aperture_photometry(data, small_ap)
        large_tab = aperture_photometry(data, large_ap)

        flux_small = small_tab['aperture_sum']
        flux_large = large_tab['aperture_sum']
        mask_good = (flux_small > 0) & (flux_large > 0)
        if mask_good.sum() < 5:
            logging.warning("Pocas estrellas buenas para medir la corrección de apertura.")
            return 1.0, 0.0, int(mask_good.sum())

        corr = (flux_large[mask_good] / flux_small[mask_good])
        median_corr = np.median(corr)
        corr_std = np.std(corr)
        return float(median_corr), float(corr_std), int(mask_good.sum())

    def calculate_aperture_correction(self, field_name, filter_name, image_header, image_data):
        """
        Wrapper that tries cache -> empirical measurement -> fallback heuristic.
        """
        cache_key = f"{field_name}_{filter_name}"
        if cache_key in self.aperture_corrections:
            return self.aperture_corrections[cache_key]

        # Try empirical measurement
        median_corr, corr_std, nstars = self.compute_empirical_aperture_correction(
            image_data, image_header, mask_positions=None)

        correction_factors = {}
        if nstars >= 5 and median_corr > 0:
            # apply same factor to all aperture diameters (approx)
            for ap_diam in self.apertures:
                correction_factors[ap_diam] = float(median_corr)
            logging.info(f"Aperture correction (empírica) {field_name} {filter_name}: corr={median_corr:.3f} nstars={nstars}")
        else:
            # Fallback: mild empirical heuristic based on FWHM
            fwhm_arcsec = float(image_header.get('FWHMMEAN', 1.5))
            for ap_diam in self.apertures:
                # ap_diam is diameter in arcsec; convert to radius
                ap_radius_arcsec = ap_diam / 2.0
                if fwhm_arcsec < 1.5:
                    if ap_diam <= 4:
                        correction = 1.10 + 0.05 * (ap_diam - 3)
                    else:
                        correction = 1.15 + 0.03 * (ap_diam - 4)
                else:
                    correction = 1.2 + 0.1 * (fwhm_arcsec - 1.0) + 0.05 * (ap_diam - 3)
                correction = min(max(correction, 1.05), 1.8)
                correction_factors[ap_diam] = float(correction)
            logging.info(f"Aperture correction (fallback) {field_name} {filter_name}: used heuristic based on FWHM={fwhm_arcsec:.2f}" )

        # Cache with metadata
        self.aperture_corrections[cache_key] = {
            'factors': correction_factors,
            'empirical': {'median': median_corr, 'std': corr_std, 'nstars': nstars}
        }
        return self.aperture_corrections[cache_key]

    def safe_magnitude_calculation(self, flux, zero_point):
        return np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)

    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        # Determine field center (for selecting sources in catalog)
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None

        # Prepare catalog (convert coords numeric)
        self.catalog[self.ra_col] = self.catalog[self.ra_col].astype(float)
        self.catalog[self.dec_col] = self.catalog[self.dec_col].astype(float)

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

        for filter_name in ['F378','F395','F410','F430','F515','F660','F861']:
            logging.info(f"  Processing filter {filter_name}")
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

                # Extract header values
                gain = float(header.get('GAIN', header.get('CCDGAIN', 1.0)))
                rdnoise = float(header.get('RDNOISE', 0.0))
                exptime = float(header.get('EXPTIME', header.get('TEXPOSED', 1.0)))
                pixscale = float(header.get('PIXSCALE', self.pixel_scale))
                fwhm_arcsec = float(header.get('FWHMMEAN', header.get('FWHMSEXT', 1.5)))
                saturate_level = float(header.get('SATURATE', np.inf))

                wcs = WCS(header)

                # Model and subtract large-scale residuals (useful in galaxy fields)
                data_corrected, bkg_rms = self.model_background_residuals(data, field_name, filter_name)
                # bkg_rms is ADU/pix (median)

                # Pre-calculate aperture corrections (empirical or fallback)
                ap_corr_info = self.calculate_aperture_correction(field_name, filter_name, header, data)
                ap_corr_factors = ap_corr_info['factors']
                ap_corr_meta = ap_corr_info['empirical']

                # World -> pixel for catalog sources
                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                x, y = wcs.world_to_pixel(coords)
                positions_all = np.column_stack((x, y))

                # Filter positions too close to edges using radio en px
                valid_positions = []
                valid_indices = []
                height, width = data_corrected.shape
                max_ap_radius_px = (max(self.apertures) / 2.0) / pixscale
                for i, (x_pos, y_pos) in enumerate(positions_all):
                    if not np.isfinite(x_pos) or not np.isfinite(y_pos):
                        continue
                    if (x_pos >= max_ap_radius_px and x_pos <= width - max_ap_radius_px and
                        y_pos >= max_ap_radius_px and y_pos <= height - max_ap_radius_px):
                        valid_positions.append([x_pos, y_pos])
                        valid_indices.append(i)

                if len(valid_positions) == 0:
                    logging.warning(f"No valid positions for photometry in {field_name} {filter_name}")
                    continue

                valid_positions = np.array(valid_positions)

                # For each aperture diameter (arcsec) compute photometry
                for ap_diam in self.apertures:
                    ap_radius_arcsec = ap_diam / 2.0
                    ap_radius_px = ap_radius_arcsec / pixscale

                    apertures = CircularAperture(valid_positions, r=ap_radius_px)
                    ann_in = max(6.0, 4.0 * (fwhm_arcsec / pixscale)) / pixscale  # safe inner annulus in arcsec then px? keep simple below
                    # We'll set annulus radii in px: use 6-9 arcsec as in tu script but convert
                    ann_in_px = 6.0 / pixscale
                    ann_out_px = 9.0 / pixscale
                    annulus = CircularAnnulus(valid_positions, r_in=ann_in_px, r_out=ann_out_px)

                    phot_table = aperture_photometry(data_corrected, apertures)
                    bkg_phot_table = aperture_photometry(data_corrected, annulus)

                    # background per source (ADU/pix)
                    annulus_area = annulus.area
                    bkg_mean_per_source = bkg_phot_table['aperture_sum'] / annulus_area

                    # flux in ADU (no EXPTIME normalization)
                    flux = phot_table['aperture_sum'] - (bkg_mean_per_source * apertures.area)

                    # Error estimation using GAIN (e-/ADU) and RDNOISE (e-)
                    # Convert to electrons for variances
                    bkg_rms_e = bkg_rms * gain if not np.isnan(bkg_rms) else 0.0
                    flux_e = np.abs(flux) * gain
                    aperture_area = apertures.area
                    var_src = flux_e
                    var_bkg = aperture_area * (bkg_rms_e ** 2)
                    var_read = aperture_area * (rdnoise ** 2)
                    flux_err_e = np.sqrt(np.abs(var_src) + var_bkg + var_read)
                    flux_err = flux_err_e / gain  # back to ADU

                    # Apply aperture correction factor
                    correction_factor = ap_corr_factors.get(ap_diam, 1.0)
                    flux_corrected = flux * correction_factor
                    flux_err_corrected = flux_err * correction_factor

                    # Propagate additional uncertainty if empirical std exists
                    corr_std = ap_corr_meta.get('std', 0.0) if isinstance(ap_corr_meta, dict) else 0.0
                    if corr_std and corr_std > 0:
                        flux_err_corrected = np.sqrt((flux_err_corrected)**2 + ((corr_std * flux_corrected)**2))

                    # Saturation flag
                    saturated = flux > saturate_level

                    # Convert to magnitudes using zero point for ADU
                    try:
                        zero_point = self.zeropoints[field_name][filter_name]
                    except Exception:
                        logging.warning(f"No zero point for field {field_name} filter {filter_name}; skipping mag calc")
                        zero_point = None

                    if zero_point is not None:
                        mag = self.safe_magnitude_calculation(flux_corrected, zero_point)
                        mag_err = np.where(flux_corrected > 0,
                                           (2.5 / np.log(10)) * (flux_err_corrected / flux_corrected),
                                           99.0)
                    else:
                        mag = np.full(len(flux_corrected), 99.0)
                        mag_err = np.full(len(flux_corrected), 99.0)

                    snr = np.where(flux_err_corrected > 0, flux_corrected / flux_err_corrected, 0.0)

                    # Store results in results dataframe using valid_indices mapping
                    for i_local, idx in enumerate(valid_indices):
                        results.loc[results.index[idx], f'X_{filter_name}_{ap_diam}'] = valid_positions[i_local, 0]
                        results.loc[results.index[idx], f'Y_{filter_name}_{ap_diam}'] = valid_positions[i_local, 1]
                        results.loc[results.index[idx], f'FLUX_{filter_name}_{ap_diam}'] = float(flux_corrected[i_local])
                        results.loc[results.index[idx], f'FLUXERR_{filter_name}_{ap_diam}'] = float(flux_err_corrected[i_local])
                        results.loc[results.index[idx], f'MAG_{filter_name}_{ap_diam}'] = float(mag[i_local])
                        results.loc[results.index[idx], f'MAGERR_{filter_name}_{ap_diam}'] = float(mag_err[i_local])
                        results.loc[results.index[idx], f'SNR_{filter_name}_{ap_diam}'] = float(snr[i_local])
                        results.loc[results.index[idx], f'SATUR_{filter_name}_{ap_diam}'] = bool(saturated[i_local])
                        # meta columns
                        results.loc[results.index[idx], f'GAIN_{filter_name}'] = gain
                        results.loc[results.index[idx], f'RDNOISE_{filter_name}'] = rdnoise
                        results.loc[results.index[idx], f'EXPTIME_{filter_name}'] = exptime
                        results.loc[results.index[idx], f'APCORR_{filter_name}'] = float(ap_corr_meta.get('median', 1.0))
                        results.loc[results.index[idx], f'APCORR_STD_{filter_name}'] = float(ap_corr_meta.get('std', 0.0))
                        results.loc[results.index[idx], f'APCORR_NSTARS_{filter_name}'] = int(ap_corr_meta.get('nstars', 0))

                    # Save debug image if requested
                    if self.debug and filter_name == self.debug_filter and ap_diam == 4:
                        self.save_debug_image(data_corrected, valid_positions, ap_radius_px, field_name, filter_name)

            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue

        return results

    def save_debug_image(self, data, positions, aperture_radius_px, field_name, filter_name):
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        distances = np.sqrt((positions[:,0]-center_x)**2 + (positions[:,1]-center_y)**2)
        idx = np.argmin(distances)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        ax.imshow(data, origin='lower', cmap='gray', vmin=median-std, vmax=median+3*std)
        aperture = CircularAperture(positions[idx], r=aperture_radius_px)
        aperture.plot(ax=ax, color='red', lw=1.5, label=f'Aperture ({aperture_radius_px*pixscale*2:.1f} arcsec)')
        ax.set_title(f'{field_name} {filter_name} - Aperture Photometry')
        ax.legend()
        plt.savefig(f'{field_name}_{filter_name}_aperture_photometry.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved debug aperture image for {field_name} {filter_name}")


# ---- Main Execution ----
if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'all_fields_zero_points_splus_format.csv'

    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        exit(1)
    elif not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
        exit(1)

    # ----------- OPCIÓN DE TEST O MODO COMPLETO -----------
    test_mode = True      # Cambia a False para procesar todos los campos
    test_field = 'CenA01' # Campo de prueba (puedes cambiarlo por otro)

    if test_mode:
        fields = [test_field]
        logging.info(f"Test mode activado. Se procesará solo el campo: {test_field}")
    else:
        fields = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info("Procesando todos los campos")
    # ------------------------------------------------------

    try:
        all_results = []
        photometry = SPLUSPhotometry(catalog_path, zeropoints_file,
                                    background_box_size=100, debug=True, debug_filter='F660')
        
        for field in tqdm(fields, desc="Processing fields"):
            if not os.path.exists(field):
                logging.warning(f"Field directory {field} does not exist. Skipping.")
                continue
                
            results = photometry.process_field(field)
            if results is not None and len(results) > 0:
                results['FIELD'] = field
                all_results.append(results)
                output_file = f'{field}_gc_photometry.csv'
                results.to_csv(output_file, index=False, float_format='%.6f')
                logging.info(f"Saved results for {field} to {output_file}")
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            # Fill NaN values with appropriate defaults
            for filter_name in ['F378','F395','F410','F430','F515','F660','F861']:
                for aperture in photometry.apertures:
                    flux_col = f'FLUX_{filter_name}_{aperture}'
                    fluxerr_col = f'FLUXERR_{filter_name}_{aperture}'
                    mag_col = f'MAG_{filter_name}_{aperture}'
                    magerr_col = f'MAGERR_{filter_name}_{aperture}'
                    snr_col = f'SNR_{filter_name}_{aperture}'
                    if flux_col in final_results.columns:
                        final_results[flux_col].fillna(0.0, inplace=True)
                        final_results[fluxerr_col].fillna(99.0, inplace=True)
                        final_results[mag_col].fillna(99.0, inplace=True)
                        final_results[magerr_col].fillna(99.0, inplace=True)
                        final_results[snr_col].fillna(0.0, inplace=True)

            # Merge with original catalog
            final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
            merged_catalog = photometry.original_catalog.merge(
                final_results, on='T17ID', how='left', suffixes=('', '_y')
            )
            # Remove duplicate columns
            for col in merged_catalog.columns:
                if col.endswith('_y'):
                    merged_catalog.drop(col, axis=1, inplace=True)
            output_file = 'all_fields_gc_photometry_merged.csv'
            merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged results saved to {output_file}")

            # Print summary statistics
            logging.info(f"Total sources in original catalog: {len(photometry.original_catalog)}")
            logging.info(f"Total sources with measurements: {len(final_results)}")
            for filter_name in ['F378','F395','F410','F430','F515','F660','F861']:
                for aperture in photometry.apertures:
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
