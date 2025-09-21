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
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path, zeropoints_file, background_box_size=50, debug=False, debug_filter='F660'):
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
        self.pixel_scale = 0.55
        self.apertures = [3, 4, 5, 6]
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
    
    def create_source_mask(self, data, snr_threshold=2, npixels=5, dilate_size=11):
        try:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=snr_threshold*std)
            sources = daofind(data - median)
            mask = np.zeros_like(data, dtype=bool)
            if sources is not None and len(sources) > 0:
                source_mask = np.zeros_like(data, dtype=bool)
                for i in range(len(sources)):
                    y, x = int(sources['ycentroid'][i]), int(sources['xcentroid'][i])
                    if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                        source_mask[y, x] = True
                structure = np.ones((dilate_size, dilate_size))
                dilated_mask = ndimage.binary_dilation(source_mask, structure=structure)
                mask = dilated_mask
            return mask
        except Exception as e:
            logging.warning(f"Error creating source mask: {e}")
            return np.zeros_like(data, dtype=bool)
    
    def subtract_background(self, data, field_name, filter_name):
        try:
            mask = self.create_source_mask(data)
            
            # For Centaurus A, we need more aggressive background estimation
            # due to the complex galaxy structure
            sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
            bkg = Background2D(data, 
                              box_size=self.background_box_size, 
                              filter_size=5,  # Increased filter size for smoother background
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=mask,
                              exclude_percentile=10)  # Exclude brightest pixels
            
            data_subtracted = data - bkg.background
            bkg_rms = bkg.background_rms_median  # Use median RMS for better stability
            
            if self.debug and filter_name == self.debug_filter:
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                plt.figure(figsize=(15, 5))
                
                # FIX: Don't use LogNorm with vmin/vmax - use linear scaling instead
                plt.subplot(131)
                plt.imshow(data, origin='lower', cmap='viridis', 
                          vmin=median-3*std, vmax=median+3*std)
                plt.title('Original')
                plt.colorbar()
                
                plt.subplot(132)
                plt.imshow(bkg.background, origin='lower', cmap='viridis')
                plt.title('Estimated Background')
                plt.colorbar()
                
                plt.subplot(133)
                plt.imshow(data_subtracted, origin='lower', cmap='viridis', 
                          vmin=median-3*std, vmax=median+3*std)
                plt.title('Background Subtracted')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(f'{field_name}_{filter_name}_background_subtraction.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved background debug image for {field_name} {filter_name}")
            
            logging.info(f"Background subtracted with box_size={self.background_box_size}")
            return data_subtracted, bkg_rms
            
        except Exception as e:
            logging.warning(f"Background subtraction failed: {e}. Using original image.")
            return data, np.nanstd(data)  # Fallback to simple std estimation

    def calculate_aperture_correction(self, field_name, filter_name, large_aperture_size=15):
        """
        Calculate proper aperture correction using curve growth method
        """
        # Check if we already calculated this correction
        cache_key = f"{field_name}_{filter_name}"
        if cache_key in self.aperture_corrections:
            return self.aperture_corrections[cache_key]
        
        ref_stars_file = f"{field_name}_gaia_xp_matches.csv"
        if not os.path.exists(ref_stars_file):
            logging.warning(f"Reference stars file {ref_stars_file} not found.")
            return {ap_size: 1.0 for ap_size in self.apertures}

        try:
            ref_stars_df = pd.read_csv(ref_stars_file)
            
            # Debug: Check what columns are available
            logging.info(f"Columns in reference file: {ref_stars_df.columns.tolist()}")
            
            # Try to find the correct flux column - note the exact column names from the file
            flux_col = None
            possible_patterns = [
                f'flux_{filter_name}',  # This matches the file: flux_F378, etc.
                f'FLUX_{filter_name}',
                f'flux_{filter_name.lower()}',
                f'FLUX_{filter_name.lower()}',
            ]
            
            for pattern in possible_patterns:
                if pattern in ref_stars_df.columns:
                    flux_col = pattern
                    break
                    
            if flux_col is None:
                logging.warning(f"No flux column found for filter {filter_name}. Available columns: {ref_stars_df.columns.tolist()}")
                return {ap_size: 1.0 for ap_size in self.apertures}
                
            logging.info(f"Using flux column: {flux_col} for filter {filter_name}")

            # More lenient selection criteria
            mask = (
                (ref_stars_df['gaia_ruwe'] < 2.5) &  # More lenient RUWE
                (ref_stars_df['gaia_phot_g_mean_mag'] > 12) & 
                (ref_stars_df['gaia_phot_g_mean_mag'] < 21) &  # Wider magnitude range
                (ref_stars_df[flux_col] > 5)  # Lower flux threshold
            )
            
            good_stars = ref_stars_df[mask].copy()
            logging.info(f"Found {len(good_stars)} good reference stars after filtering")
            
            if len(good_stars) < 3:
                logging.warning(f"Not enough good reference stars for aperture correction. Found {len(good_stars)}")
                # Try to use all stars with minimal criteria
                mask_minimal = (ref_stars_df[flux_col] > 0)
                good_stars = ref_stars_df[mask_minimal].copy()
                logging.info(f"Trying with minimal criteria: {len(good_stars)} stars")
                
                if len(good_stars) < 3:
                    logging.warning("Still not enough stars. Using default correction factors.")
                    return {ap_size: 1.0 for ap_size in self.apertures}
                
        except Exception as e:
            logging.warning(f"Error loading reference stars: {e}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        # Load the image
        image_path = None
        for ext in ['.fits.fz', '.fits']:
            candidate = os.path.join(field_name, f"{field_name}_{filter_name}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if not image_path:
            logging.warning(f"No image found for {field_name} {filter_name}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        try:
            with fits.open(image_path) as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                    header = hdul[1].header
                    data = hdul[1].data.astype(float)
                else:
                    header = hdul[0].header
                    data = hdul[0].data.astype(float)
        except Exception as e:
            logging.warning(f"Error loading image: {e}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        # Subtract background
        data_bg_subtracted, _ = self.subtract_background(data, field_name, filter_name)
        wcs = WCS(header)
        
        # Convert star positions to pixel coordinates
        coords = SkyCoord(ra=good_stars['ra'].values*u.deg, dec=good_stars['dec'].values*u.deg)
        x, y = wcs.world_to_pixel(coords)
        positions = np.column_stack((x, y))
        
        # Remove stars near edges
        height, width = data_bg_subtracted.shape
        max_aperture = large_aperture_size / self.pixel_scale
        valid_mask = (
            (positions[:, 0] > max_aperture) & 
            (positions[:, 0] < width - max_aperture) &
            (positions[:, 1] > max_aperture) & 
            (positions[:, 1] < height - max_aperture)
        )
        positions = positions[valid_mask]
        good_stars = good_stars[valid_mask]
        
        if len(positions) < 5:
            logging.warning(f"Not enough stars away from edges")
            return {ap_size: 1.0 for ap_size in self.apertures}
        
        # Calculate growth curve for each star
        aperture_sizes_px = np.arange(3, 20, 1) / self.pixel_scale
        growth_curves = []
        
        for ap_size in aperture_sizes_px:
            apertures = CircularAperture(positions, r=ap_size)
            phot_table = aperture_photometry(data_bg_subtracted, apertures)
            growth_curves.append(phot_table['aperture_sum'].data)
        
        growth_curves = np.array(growth_curves)
        
        # Normalize growth curves to largest aperture
        norm_curves = growth_curves / growth_curves[-1]
        
        # Fit analytical growth curve model
        def growth_model(r, a, b):
            return 1 - np.exp(-(r/b)**a)
        
        mean_curve = np.mean(norm_curves, axis=1)
        
        try:
            popt, pcov = curve_fit(growth_model, aperture_sizes_px, mean_curve, 
                                  p0=[2.0, 5.0], bounds=([1.0, 2.0], [3.0, 10.0]))
            
            # Calculate correction factors for our apertures
            correction_factors = {}
            large_flux = growth_model(large_aperture_size / self.pixel_scale, *popt)
            
            for ap_size in self.apertures:
                small_flux = growth_model(ap_size / self.pixel_scale / 2, *popt)
                correction_factors[ap_size] = large_flux / small_flux
                logging.info(f"Aperture correction {field_name} {filter_name} {ap_size}arcsec: {correction_factors[ap_size]:.3f}")
                
        except Exception as e:
            logging.warning(f"Curve fitting failed: {e}. Using empirical method.")
            # Fallback: Use empirical method based on flux ratios
            correction_factors = {}
            large_flux = np.mean(growth_curves[-1])
            
            for i, ap_size in enumerate(self.apertures):
                ap_radius_px = (ap_size / 2) / self.pixel_scale
                # Find the closest aperture size in our calculated sizes
                idx = np.argmin(np.abs(aperture_sizes_px - ap_radius_px))
                small_flux = np.mean(growth_curves[idx])
                correction_factors[ap_size] = large_flux / small_flux
                logging.info(f"Empirical aperture correction {field_name} {filter_name} {ap_size}arcsec: {correction_factors[ap_size]:.3f}")
        
        # Cache the result
        self.aperture_corrections[cache_key] = correction_factors
        
        # Plot growth curve for debugging
        if self.debug:
            plt.figure(figsize=(10, 6))
            plt.plot(aperture_sizes_px * self.pixel_scale, mean_curve, 'bo-', label='Mean growth curve')
            try:
                plt.plot(aperture_sizes_px * self.pixel_scale, 
                        growth_model(aperture_sizes_px, *popt), 'r-', label='Fitted model')
            except:
                pass
            plt.xlabel('Aperture radius (arcsec)')
            plt.ylabel('Fraction of total flux')
            plt.title(f'Aperture Growth Curve: {field_name} {filter_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{field_name}_{filter_name}_growth_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        return correction_factors
    
    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None
        
        # Pre-calculate aperture corrections for all filters
        aperture_corrections = {}
        for filter_name in self.filters:
            aperture_corrections[filter_name] = self.calculate_aperture_correction(field_name, filter_name)
        
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
                data_bg_subtracted, bkg_rms = self.subtract_background(data, field_name, filter_name)
                
                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                
                x, y = wcs.world_to_pixel(coords)
                positions = np.column_stack((x, y))
                
                # Remove sources too close to edges
                valid_positions = []
                valid_indices = []
                height, width = data_bg_subtracted.shape
                
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
                
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    apertures = CircularAperture(valid_positions, r=aperture_radius_px)
                    annulus = CircularAnnulus(valid_positions, 
                                            r_in=6/self.pixel_scale, 
                                            r_out=9/self.pixel_scale)
                    
                    # Perform photometry
                    phot_table = aperture_photometry(data_bg_subtracted, apertures)
                    bkg_phot_table = aperture_photometry(data_bg_subtracted, annulus)
                    
                    # Background subtraction
                    bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
                    flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
                    
                    # Error estimation
                    flux_err = np.sqrt(np.abs(flux) + (apertures.area * bkg_rms**2))
                    
                    # Apply aperture correction
                    correction_factor = aperture_corrections[filter_name].get(aperture_size, 1.0)
                    flux_corrected = flux * correction_factor
                    flux_err_corrected = flux_err * correction_factor
                    
                    # Calculate magnitudes
                    zero_point = self.zeropoints[field_name][filter_name]
                    mag = zero_point - 2.5 * np.log10(np.maximum(flux_corrected, 1e-10))
                    mag_err = (2.5 / np.log(10)) * (flux_err_corrected / np.maximum(flux_corrected, 1e-10))
                    snr = flux_corrected / np.maximum(flux_err_corrected, 1e-10)
                    
                    # Handle negative or problematic fluxes
                    bad_flux_mask = (flux_corrected <= 0) | (snr < 1)
                    mag[bad_flux_mask] = 99.0
                    mag_err[bad_flux_mask] = 99.0
                    snr[bad_flux_mask] = 0.0
                    
                    # Store results
                    for i, idx in enumerate(valid_indices):
                        results.loc[results.index[idx], f'X_{filter_name}_{aperture_size}'] = valid_positions[i, 0]
                        results.loc[results.index[idx], f'Y_{filter_name}_{aperture_size}'] = valid_positions[i, 1]
                        results.loc[results.index[idx], f'FLUX_{filter_name}_{aperture_size}'] = flux_corrected[i]
                        results.loc[results.index[idx], f'FLUXERR_{filter_name}_{aperture_size}'] = flux_err_corrected[i]
                        results.loc[results.index[idx], f'MAG_{filter_name}_{aperture_size}'] = mag[i]
                        results.loc[results.index[idx], f'MAGERR_{filter_name}_{aperture_size}'] = mag_err[i]
                        results.loc[results.index[idx], f'SNR_{filter_name}_{aperture_size}'] = snr[i]
                    
                    if self.debug and filter_name == self.debug_filter and aperture_size == 4:
                        self.save_debug_image(data_bg_subtracted, valid_positions, 
                                             aperture_radius_px, field_name, filter_name)
            
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        return results
    
    def save_debug_image(self, data, positions, aperture_radius, field_name, filter_name):
        # Select a representative source near the center
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
                                    background_box_size=50, debug=True, debug_filter='F660')
        
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
            
            # Merge with original catalog
            final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
            merged_catalog = photometry.original_catalog.merge(
                final_results, on='T17ID', how='left', suffixes=('', '_y')
            )
            
            # Remove duplicate columns from merge
            for col in merged_catalog.columns:
                if col.endswith('_y'):
                    merged_catalog.drop(col, axis=1, inplace=True)
            
            # Save final catalog
            output_file = 'all_fields_gc_photometry_merged.csv'
            merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged results saved to {output_file}")
            
            # Print summary statistics
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
