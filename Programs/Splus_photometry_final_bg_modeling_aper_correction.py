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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path, zeropoints_file, debug=False, debug_filter='F660'):
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
        self.debug = debug
        self.debug_filter = debug_filter
        
        self.ra_col = next((col for col in ['RAJ2000', 'RA'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RAJ2000/DEJ2000 or RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain T17ID or ID column")
        
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
        Revised background treatment for S-PLUS images that are already background subtracted
        """
        try:
            # Get basic statistics of the original image
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            
            # For S-PLUS images that are already background subtracted, we should be careful
            # not to over-subtract. Use a more conservative approach.
            
            # Only model residuals if there's significant large-scale structure
            if std < 1.0:  # Very flat image, minimal background residuals
                logging.info(f"Image appears flat (std={std:.3f}). Skipping background modeling.")
                return data, std
            
            # Create a mask for bright objects (more conservative thresholds)
            mask = data > median + 10 * std  # Higher threshold to avoid masking clusters
            
            # Dilate the mask moderately
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((5, 5)))
            
            # Use larger boxes to avoid over-fitting cluster light
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=200,  # Larger boxes for smoother background
                              filter_size=5,
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=20)  # Exclude more pixels
            
            # Only subtract if the background model has significant structure
            bkg_range = np.max(bkg.background) - np.min(bkg.background)
            if bkg_range < 3 * std:  # Background model is relatively flat
                logging.info(f"Background model is flat (range={bkg_range:.3f}). Minimal subtraction applied.")
                data_corrected = data - np.median(bkg.background)  # Just subtract median
            else:
                data_corrected = data - bkg.background
            
            bkg_rms = bkg.background_rms_median
            
            if self.debug and filter_name == self.debug_filter:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(131)
                vmin = median - 2 * std
                vmax = median + 5 * std
                plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                plt.title('Original Image')
                plt.colorbar()
                
                plt.subplot(132)
                plt.imshow(bkg.background, origin='lower', cmap='gray')
                plt.title('Modeled Background')
                plt.colorbar()
                
                plt.subplot(133)
                plt.imshow(data_corrected, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                plt.title('After Background Treatment')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(f'{field_name}_{filter_name}_background_revised.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved revised background image for {field_name} {filter_name}")
            
            logging.info(f"Background treated: original std={std:.3f}, background range={bkg_range:.3f}")
            return data_corrected, bkg_rms
            
        except Exception as e:
            logging.warning(f"Background treatment failed: {e}. Using original image.")
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            return data, std

    def calculate_splus_aperture_correction(self, data, header, source_positions, filter_name):
        """
        Calculate aperture correction using S-PLUS official methodology
        Uses growth curve analysis with 32 apertures from 1-50 arcsec
        """
        try:
            pixscale = header.get('PIXSCALE', self.pixel_scale)
            
            # 32 apertures between 1-50 arcsec (diameter) as in S-PLUS
            aperture_diameters = np.linspace(1, 50, 32)
            aperture_radii_px = (aperture_diameters / 2) / pixscale
            
            # Measure growth curves for each source
            growth_fluxes = []
            
            for radius in aperture_radii_px:
                apertures = CircularAperture(source_positions, r=radius)
                phot_table = aperture_photometry(data, apertures)
                growth_fluxes.append(phot_table['aperture_sum'].data)
            
            growth_fluxes = np.array(growth_fluxes)
            
            # Normalize to largest aperture (50 arcsec)
            total_fluxes = growth_fluxes[-1]
            
            # Filter sources with valid fluxes
            valid_mask = total_fluxes > 0
            if np.sum(valid_mask) < 5:
                logging.warning(f"Insufficient valid sources for aperture correction in {filter_name}")
                return {ap: 1.0 for ap in self.apertures}
            
            growth_fluxes_valid = growth_fluxes[:, valid_mask]
            total_fluxes_valid = total_fluxes[valid_mask]
            
            # Normalized growth curves
            normalized_curves = growth_fluxes_valid / total_fluxes_valid
            
            # Median growth curve (robust)
            median_growth = np.median(normalized_curves, axis=1)
            
            # Calculate correction factors for each aperture size
            correction_factors = {}
            
            for aperture_size in self.apertures:
                aperture_radius_arcsec = aperture_size / 2
                aperture_radius_px = aperture_radius_arcsec / pixscale
                
                # Find closest measured radius
                idx = np.argmin(np.abs(aperture_radii_px - aperture_radius_px))
                fraction_in_aperture = median_growth[idx]
                
                if fraction_in_aperture > 0:
                    correction = 1.0 / fraction_in_aperture
                    # Apply reasonable constraints
                    correction = max(1.0, min(correction, 3.0))
                else:
                    correction = 1.0
                
                correction_factors[aperture_size] = correction
            
            logging.info(f"S-PLUS aperture corrections for {filter_name}: {correction_factors}")
            return correction_factors
            
        except Exception as e:
            logging.error(f"Error in S-PLUS aperture correction for {filter_name}: {e}")
            return {ap: 1.0 for ap in self.apertures}

    def safe_magnitude_calculation(self, flux, zero_point):
        """Calculate magnitudes safely handling non-positive fluxes"""
        return np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)
    
    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None
        
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
                
                # Apply revised background treatment
                data_corrected, bkg_rms = self.model_background_residuals(data, field_name, filter_name)
                
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
                
                # Calculate S-PLUS aperture corrections using science objects
                aperture_corrections = self.calculate_splus_aperture_correction(
                    data_corrected, header, valid_positions, filter_name)
                
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    apertures = CircularAperture(valid_positions, r=aperture_radius_px)
                    annulus = CircularAnnulus(valid_positions, 
                                            r_in=6/self.pixel_scale, 
                                            r_out=9/self.pixel_scale)
                    
                    # Perform photometry
                    phot_table = aperture_photometry(data_corrected, apertures)
                    
                    # Background estimation
                    bkg_phot_table = aperture_photometry(data_corrected, annulus)
                    bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
                    
                    # Calculate flux
                    flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
                    
                    # Error estimation
                    flux_err = np.sqrt(np.abs(flux) + (apertures.area * bkg_rms**2))
                    
                    # Apply aperture correction
                    correction_factor = aperture_corrections.get(aperture_size, 1.0)
                    flux_corrected = flux * correction_factor
                    flux_err_corrected = flux_err * correction_factor
                    
                    # Calculate magnitudes
                    zero_point = self.zeropoints[field_name][filter_name]
                    mag = self.safe_magnitude_calculation(flux_corrected, zero_point)
                    mag_err = np.where(flux_corrected > 0, 
                                      (2.5 / np.log(10)) * (flux_err_corrected / flux_corrected),
                                      99.0)
                    snr = np.where(flux_err_corrected > 0, flux_corrected / flux_err_corrected, 0.0)
                    
                    # Store results
                    for i, idx in enumerate(valid_indices):
                        results.loc[results.index[idx], f'X_{filter_name}_{aperture_size}'] = valid_positions[i, 0]
                        results.loc[results.index[idx], f'Y_{filter_name}_{aperture_size}'] = valid_positions[i, 1]
                        results.loc[results.index[idx], f'FLUX_{filter_name}_{aperture_size}'] = flux_corrected[i]
                        results.loc[results.index[idx], f'FLUXERR_{filter_name}_{aperture_size}'] = flux_err_corrected[i]
                        results.loc[results.index[idx], f'MAG_{filter_name}_{aperture_size}'] = mag[i]
                        results.loc[results.index[idx], f'MAGERR_{filter_name}_{aperture_size}'] = mag_err[i]
                        results.loc[results.index[idx], f'SNR_{filter_name}_{aperture_size}'] = snr[i]
                        results.loc[results.index[idx], f'AP_CORR_{filter_name}_{aperture_size}'] = correction_factor
                    
                    # Debug image for one filter and aperture
                    if self.debug and filter_name == self.debug_filter and aperture_size == 4:
                        self.save_debug_image(data_corrected, valid_positions, 
                                             aperture_radius_px, field_name, filter_name)
            
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        return results
    
    def save_debug_image(self, data, positions, aperture_radius, field_name, filter_name):
        """Save debug image showing apertures"""
        if len(positions) == 0:
            return
            
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        distances = np.sqrt((positions[:,0]-center_x)**2 + (positions[:,1]-center_y)**2)
        idx = np.argmin(distances)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        ax.imshow(data, origin='lower', cmap='gray', vmin=median-std, vmax=median+3*std)
        
        aperture = CircularAperture(positions[idx], r=aperture_radius)
        aperture.plot(ax=ax, color='red', lw=1.5, 
                     label=f'Aperture ({aperture_radius*self.pixel_scale*2:.1f} arcsec)')
        
        ax.set_title(f'{field_name} {filter_name} - Aperture Photometry')
        ax.legend()
        plt.savefig(f'{field_name}_{filter_name}_aperture_photometry.png', 
                   dpi=150, bbox_inches='tight')
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
                                    debug=True, debug_filter='F660')
        
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
            
            # Fill NaN values
            for filter_name in photometry.filters:
                for aperture in photometry.apertures:
                    for col_suffix in ['FLUX', 'FLUXERR', 'MAG', 'MAGERR', 'SNR', 'AP_CORR']:
                        col = f'{col_suffix}_{filter_name}_{aperture}'
                        if col in final_results.columns:
                            if 'FLUX' in col:
                                final_results[col].fillna(0.0, inplace=True)
                            elif 'MAG' in col or 'ERR' in col:
                                final_results[col].fillna(99.0, inplace=True)
                            elif 'AP_CORR' in col:
                                final_results[col].fillna(1.0, inplace=True)
                            else:
                                final_results[col].fillna(0.0, inplace=True)
            
            # Merge with original catalog
            final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
            merged_catalog = photometry.original_catalog.merge(
                final_results, on='T17ID', how='left', suffixes=('', '_y')
            )
            
            # Remove duplicate columns
            for col in merged_catalog.columns:
                if col.endswith('_y'):
                    merged_catalog.drop(col, axis=1, inplace=True)
            
            # Save final catalog
            output_file = 'Results/all_fields_gc_photometry_merged.csv'
            merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
            logging.info(f"Final merged results saved to {output_file}")
            
            # Print summary including aperture corrections
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
                            mean_ap_corr = valid_data[f'AP_CORR_{filter_name}_{aperture}'].mean()
                            logging.info(f"{filter_name}_{aperture}: {len(valid_data)} valid, "
                                       f"Mean SNR: {mean_snr:.1f}, Mean Mag: {mean_mag:.2f}, "
                                       f"Mean AP Corr: {mean_ap_corr:.3f}")
        
    except Exception as e:
        logging.error(f"Execution error: {e}")
        traceback.print_exc()
