#!/usr/bin/env python3
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
from matplotlib.colors import LogNorm
import logging
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path, zeropoints_file, background_box_size=100, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {
            row['field']: {filt: row[filt] for filt in ['F378','F395','F410','F430','F515','F660','F861']}
            for _, row in self.zeropoints_df.iterrows()
        }
        
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel
        self.apertures = [3, 4, 5, 6]  # aperture diameters in arcsec
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
                        # Try header CRVALs, and fallback to WCS if needed
                        ra_center = header.get('CRVAL1')
                        dec_center = header.get('CRVAL2')
                        if ra_center is None or dec_center is None:
                            try:
                                wcs = WCS(header)
                                ra_center, dec_center = wcs.wcs.crval
                            except Exception:
                                ra_center, dec_center = None, None
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
        Model and subtract background residuals with a heuristic to avoid double subtraction.
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
            # Heurística: evita restar si el modelo tiene mediana insignificante frente a la mediana de datos
            bkg_median = np.median(bkg.background)
            data_median = np.median(data)
            if np.abs(bkg_median) < 1e-3 * max(1.0, np.abs(data_median)):
                logging.info(f"Background residual median ({bkg_median:.3e}) insignificante vs data median ({data_median:.3e}). No se resta.")
                data_corrected = data.copy()
            else:
                data_corrected = data - bkg.background

            bkg_rms = bkg.background_rms_median

            if self.debug and filter_name == self.debug_filter:
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                vmin = median - 3 * std
                vmax = median + 3 * std
                plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                plt.title('Original (already bkg-subtracted?)')
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
                plt.savefig(f'{field_name}_{filter_name}_background_residuals.png', dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved background residuals image for {field_name} {filter_name}")

            logging.info(f"Background residuals modeled with box_size={self.background_box_size}")
            return data_corrected, bkg_rms

        except Exception as e:
            logging.warning(f"Background residual modeling failed: {e}. Using original image.")
            return data, np.nanstd(data)

    def detect_stars_for_aper_corr(self, data, fwhm_est=3.0, threshold_sigma=5.0, max_candidates=100):
        """
        Detect point sources (stars) to use for aperture correction.
        Returns array of (x, y) in pixel coordinates.
        """
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            daofind = DAOStarFinder(fwhm=fwhm_est, threshold=threshold_sigma*std)
            sources = daofind(data - median)
            if sources is None or len(sources) == 0:
                return np.empty((0,2))
            # Filter out saturated or very bright objects if needed
            # Keep top N by flux
            sources.sort(['flux'])
            sources = sources[::-1]  # descending
            sources = sources[:max_candidates]
            x = sources['xcentroid'].data.astype(float)
            y = sources['ycentroid'].data.astype(float)
            positions = np.column_stack((x, y))
            return positions
        except Exception as e:
            logging.warning(f"Star detection for aperture correction failed: {e}")
            return np.empty((0,2))

    def calculate_aperture_correction_direct(self, field_name, filter_name, science_positions, science_data, large_aperture_size=10):
        """
        Calculate aperture correction (robust) using point sources when available; otherwise
        fall back to using the provided science_positions (but warn).
        large_aperture_size in arcsec (default 10")
        """
        cache_key = f"{field_name}_{filter_name}"
        if cache_key in self.aperture_corrections:
            return self.aperture_corrections[cache_key]

        # Prefer to detect stars in the image for an accurate correction
        star_positions = self.detect_stars_for_aper_corr(science_data, fwhm_est=3.0, threshold_sigma=6.0, max_candidates=200)
        if len(star_positions) >= 5:
            positions_for_growth = star_positions
            logging.info(f"Using {len(star_positions)} detected stars for aperture correction in {field_name} {filter_name}")
        else:
            # fallback: try using the given science_positions, but log a warning
            positions_for_growth = science_positions
            logging.warning(f"Using science objects ({len(science_positions)}) for aperture correction in {field_name} {filter_name} because not enough stars were found.")

        if len(positions_for_growth) < 5:
            logging.warning(f"Not enough objects for aperture correction in {field_name} {filter_name}")
            correction_factors = {ap_size: 1.0 for ap_size in self.apertures}
            self.aperture_corrections[cache_key] = correction_factors
            return correction_factors

        height, width = science_data.shape
        max_ap_px = large_aperture_size / self.pixel_scale
        valid_mask = (
            (positions_for_growth[:, 0] > max_ap_px) &
            (positions_for_growth[:, 0] < width - max_ap_px) &
            (positions_for_growth[:, 1] > max_ap_px) &
            (positions_for_growth[:, 1] < height - max_ap_px)
        )
        valid_positions = positions_for_growth[valid_mask]

        if len(valid_positions) < 3:
            logging.warning("Not enough objects away from edges for aperture correction")
            correction_factors = {ap_size: 1.0 for ap_size in self.apertures}
            self.aperture_corrections[cache_key] = correction_factors
            return correction_factors

        # Build radii (in pixels)
        aperture_radii_px = np.linspace(1.0, large_aperture_size / self.pixel_scale, 20)
        growth_curves = []

        for radius in tqdm(aperture_radii_px, desc=f"Growth curve {filter_name}", leave=False):
            apertures = CircularAperture(valid_positions, r=radius)
            phot_table = aperture_photometry(science_data, apertures)
            growth_curves.append(np.array(phot_table['aperture_sum'].data, dtype=float))

        growth_curves = np.array(growth_curves)  # shape (n_radii, n_positions)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_growth = np.median(growth_curves, axis=1)

        # Save diagnostic plot
        try:
            plt.figure(figsize=(8,5))
            plt.plot(aperture_radii_px * self.pixel_scale, median_growth, marker='o')
            plt.xlabel('Aperture radius (arcsec)')
            plt.ylabel('Median aperture flux (ADU)')
            plt.title(f'Growth curve (median) - {field_name} {filter_name}')
            plt.grid(alpha=0.3)
            plt.savefig(f'{field_name}_{filter_name}_growth_curve_median.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.warning(f"Could not save growth curve plot: {e}")

        total_flux_median = median_growth[-1]
        logging.info(f"{field_name} {filter_name} growth curve: total_flux_median = {total_flux_median:.3e}")
        logging.info(f"median_growth (first/mid/last): {median_growth[0]:.3e}, {median_growth[len(median_growth)//2]:.3e}, {median_growth[-1]:.3e}")

        if total_flux_median <= 0 or np.nanmedian(median_growth) <= 0:
            logging.warning(f"Total median flux <= 0 en {field_name} {filter_name}. Posible sobre-substracción del fondo o unidades inesperadas.")
            correction_factors = {ap_size: 1.0 for ap_size in self.apertures}
            self.aperture_corrections[cache_key] = correction_factors
            return correction_factors

        correction_factors = {}
        for ap_size in self.apertures:
            radius_px = (ap_size / 2.0) / self.pixel_scale
            idx = np.argmin(np.abs(aperture_radii_px - radius_px))
            aperture_flux = median_growth[idx]
            if aperture_flux <= 0:
                logging.warning(f"Aperture flux <=0 para ap {ap_size}\" en {field_name} {filter_name}. Saltando.")
                correction = 1.0
            else:
                correction = total_flux_median / aperture_flux

            if correction > 10:
                logging.warning(f"Large aperture correction ({correction:.1f}) for {field_name} {filter_name} {ap_size}\". Possible problem.")
            correction = max(min(correction, 10.0), 1.0)
            correction_factors[ap_size] = correction
            logging.info(f"Direct aperture correction {field_name} {filter_name} {ap_size}arcsec: {correction_factors[ap_size]:.3f}")

        self.aperture_corrections[cache_key] = correction_factors
        return correction_factors

    def safe_magnitude_calculation(self, flux, zero_point):
        return np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)

    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None

        aperture_corrections = {}
        self.catalog[self.ra_col] = self.catalog[self.ra_col].astype(float)
        self.catalog[self.dec_col] = self.catalog[self.dec_col].astype(float)

        in_field_mask = [
            self.is_source_in_field(row[self.ra_col], row[self.dec_col], field_ra, field_dec)
            for _, row in self.catalog.iterrows()
        ]
        field_sources = self.catalog[in_field_mask].copy()
        logging.info(f"Found {len(field_sources)} sources in field {field_name}")

        if len(field_sources) == 0:
            return None

        results = field_sources.copy()

        for filter_name in self.filters:
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
                        header = hdul[1].header
                        data = hdul[1].data.astype(float)
                    else:
                        header = hdul[0].header
                        data = hdul[0].data.astype(float)

                wcs = WCS(header)
                data_corrected, bkg_rms = self.model_background_residuals(data, field_name, filter_name)

                # Gain / readnoise if available
                gain = header.get('GAIN', header.get('EGAIN', 1.0))
                readnoise = header.get('RDNOISE', header.get('READNOI', 5.0))

                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                x, y = wcs.world_to_pixel(coords)
                positions = np.column_stack((x, y))

                # Remove sources too close to edges
                valid_positions = []
                valid_indices = []
                height, width = data_corrected.shape
                for i, (x_pos, y_pos) in enumerate(positions):
                    max_ap_px = max(self.apertures) / self.pixel_scale
                    if (x_pos > max_ap_px and x_pos < width - max_ap_px and
                        y_pos > max_ap_px and y_pos < height - max_ap_px):
                        valid_positions.append([x_pos, y_pos])
                        valid_indices.append(i)

                if not valid_positions:
                    logging.warning(f"No valid positions for photometry in {field_name} {filter_name}")
                    continue

                valid_positions = np.array(valid_positions)

                # Calculate aperture correction using stars or science objects
                if filter_name not in aperture_corrections:
                    if len(valid_positions) >= 5:
                        aperture_corrections[filter_name] = self.calculate_aperture_correction_direct(
                            field_name, filter_name, valid_positions, data_corrected
                        )
                    else:
                        aperture_corrections[filter_name] = {ap_size: 1.0 for ap_size in self.apertures}
                        logging.warning(f"Not enough objects for aperture correction in {field_name} {filter_name}")

                # Precompute per-source annulus sky using masks (robust)
                # We'll compute annulus values inside the aperture loop because annulus depends on aperture size if desired
                row_labels = field_sources.index[valid_indices]  # labels to assign results safely

                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2.0) / self.pixel_scale
                    apertures = CircularAperture(valid_positions, r=aperture_radius_px)

                    # Define annulus relative to aperture and FWHM estimate
                    # Heuristic: r_in = aperture_radius_px + 4 px; r_out = r_in + 3 px
                    r_in = aperture_radius_px + 4.0
                    r_out = r_in + 3.0
                    annulus = CircularAnnulus(valid_positions, r_in=r_in, r_out=r_out)

                    phot_table = aperture_photometry(data_corrected, apertures)
                    ann_table = aperture_photometry(data_corrected, annulus)

                    # Estimate background per source using annulus pixels (sigma-clipped)
                    bkg_mean_array = []
                    ann_masks = annulus.to_mask(method='center')  # list of masks for each source
                    for j, mask in enumerate(ann_masks):
                        try:
                            ann_data = mask.multiply(data_corrected)
                            ann_data_1d = ann_data[mask.data > 0]
                            if len(ann_data_1d) == 0:
                                mean_sky = 0.0
                                std_sky = bkg_rms
                            else:
                                mean_sky, median_sky, std_sky = sigma_clipped_stats(ann_data_1d, sigma=3.0)
                                mean_sky = median_sky
                            bkg_mean_array.append(mean_sky)
                        except Exception:
                            bkg_mean_array.append(0.0)

                    bkg_mean_array = np.array(bkg_mean_array, dtype=float)
                    aperture_area = apertures.area  # area in pixels

                    flux = np.array(phot_table['aperture_sum'].data, dtype=float) - (bkg_mean_array * aperture_area)

                    # Error estimation including gain and readnoise if possible
                    # Convert to electrons for variance calculation then back to ADU if needed
                    flux_e = flux * gain
                    var_poisson = np.abs(flux_e)  # Poisson from source
                    var_sky = aperture_area * ( (bkg_rms * gain)**2 )
                    var_read = aperture_area * (readnoise**2)
                    var_total = var_poisson + var_sky + var_read
                    # avoid negative due to numerical issues
                    var_total = np.where(var_total > 0, var_total, 0.0)
                    flux_err = np.sqrt(var_total) / (gain if gain != 0 else 1.0)

                    # Apply aperture correction
                    correction_factor = aperture_corrections[filter_name].get(aperture_size, 1.0)
                    flux_corrected = flux * correction_factor
                    flux_err_corrected = flux_err * correction_factor

                    # Calculate magnitudes (use field zero point)
                    zero_point = self.zeropoints.get(field_name, {}).get(filter_name, np.nan)
                    if np.isnan(zero_point):
                        logging.warning(f"Zero point missing for {field_name} {filter_name}. Using 0.0")
                        zero_point = 0.0

                    mag = self.safe_magnitude_calculation(flux_corrected, zero_point)
                    mag_err = np.where(flux_corrected > 0,
                                      (2.5 / np.log(10)) * (flux_err_corrected / flux_corrected),
                                      99.0)
                    snr = np.where(flux_err_corrected > 0, flux_corrected / flux_err_corrected, 0.0)

                    # Store results using the original DataFrame labels (safe assignment)
                    for i_local, label in enumerate(row_labels):
                        results.loc[label, f'X_{filter_name}_{aperture_size}'] = valid_positions[i_local, 0]
                        results.loc[label, f'Y_{filter_name}_{aperture_size}'] = valid_positions[i_local, 1]
                        results.loc[label, f'FLUX_{filter_name}_{aperture_size}'] = float(flux_corrected[i_local])
                        results.loc[label, f'FLUXERR_{filter_name}_{aperture_size}'] = float(flux_err_corrected[i_local])
                        results.loc[label, f'MAG_{filter_name}_{aperture_size}'] = float(mag[i_local])
                        results.loc[label, f'MAGERR_{filter_name}_{aperture_size}'] = float(mag_err[i_local])
                        results.loc[label, f'SNR_{filter_name}_{aperture_size}'] = float(snr[i_local])

                    # Save debug image for one aperture size if requested
                    if self.debug and filter_name == self.debug_filter and aperture_size == 4:
                        self.save_debug_image(data_corrected, valid_positions, aperture_radius_px, field_name, filter_name)

            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue

        return results

    def save_debug_image(self, data, positions, aperture_radius, field_name, filter_name):
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        distances = np.sqrt((positions[:,0]-center_x)**2 + (positions[:,1]-center_y)**2)
        idx = np.argmin(distances)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        ax.imshow(data, origin='lower', cmap='gray', vmin=median-std, vmax=median+3*std)
        aperture = CircularAperture(positions[idx], r=aperture_radius)
        aperture.plot(ax=ax, color='red', lw=1.5, label=f'Aperture ({aperture_radius*self.pixel_scale*2:.1f} arcsec)')
        ax.set_title(f'{field_name} {filter_name} - Aperture Photometry')
        ax.legend()
        plt.savefig(f'{field_name}_{filter_name}_aperture_photometry.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved debug aperture image for {field_name} {filter_name}")

# ---- Main Execution ----
if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'Results/all_fields_zero_points_splus_format_corrected.csv'

    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        exit(1)
    elif not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
        exit(1)

    # Configuration
    test_mode = True
    test_field = 'CenA01'

    if test_mode:
        fields = [test_field]
        logging.info(f"Test mode activated. Processing only field: {test_field}")
    else:
        fields = [f'CenA{i:02d}' for i in range(1, 25)]
        logging.info("Processing all fields")

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
            # Fill NaNs
            for filter_name in photometry.filters:
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

            final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
            merged_catalog = photometry.original_catalog.merge(
                final_results, on='T17ID', how='left', suffixes=('', '_y')
            )
            for col in merged_catalog.columns:
                if col.endswith('_y'):
                    merged_catalog.drop(col, axis=1, inplace=True)
            output_file = 'Results/all_fields_gc_photometry_merged.csv'
            merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged results saved to {output_file}")

            logging.info(f"Total sources in original catalog: {len(photometry.original_catalog)}")
            logging.info(f"Total sources with measurements: {len(final_results)}")
            for filter_name in photometry.filters:
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
