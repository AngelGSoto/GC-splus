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
    
    def model_background_residuals(self, data, field_name, filter_name):
        """
        Model and subtract background residuals for Centaurus A
        """
        try:
            # Create a mask for bright objects
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            mask = data > median + 5 * std
            
            # Dilate the mask to cover extended objects
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((15, 15)))
            
            # Use large boxes to model the large-scale background residuals
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, 
                              box_size=self.background_box_size, 
                              filter_size=5,
                              sigma_clip=sigma_clip, 
                              bkg_estimator=MedianBackground(), 
                              mask=dilated_mask,
                              exclude_percentile=10)
            
            # Subtract the modeled background residuals
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

    def calculate_aperture_correction_science_objects(self, field_name, filter_name, science_positions, science_data, large_aperture_size=10):
        """
        Calculate aperture correction using science objects (globular clusters) directly
        """
        cache_key = f"{field_name}_{filter_name}_science"
        if cache_key in self.aperture_corrections:
            return self.aperture_corrections[cache_key]
        
        if len(science_positions) < 5:
            logging.warning(f"Not enough science objects ({len(science_positions)}) for reliable aperture correction")
            return {ap_size: 1.0 for ap_size in self.apertures}
        
        # Filter positions away from edges
        height, width = science_data.shape
        max_ap_px = large_aperture_size / self.pixel_scale
        valid_mask = (
            (science_positions[:, 0] > max_ap_px) & 
            (science_positions[:, 0] < width - max_ap_px) &
            (science_positions[:, 1] > max_ap_px) & 
            (science_positions[:, 1] < height - max_ap_px)
        )
        valid_positions = science_positions[valid_mask]
        
        if len(valid_positions) < 3:
            logging.warning("Not enough science objects away from edges for aperture correction")
            return {ap_size: 1.0 for ap_size in self.apertures}
        
        logging.info(f"Using {len(valid_positions)} science objects for aperture correction in {filter_name}")
        
        # Use conservative aperture radii
        aperture_radii_px = np.linspace(2.0, large_aperture_size/self.pixel_scale, 8)
        growth_fluxes = []
        
        # DEBUG: Check background levels
        mean_bkg = np.mean(science_data)
        median_bkg = np.median(science_data)
        std_bkg = np.std(science_data)
        logging.info(f"Background stats - Mean: {mean_bkg:.3f}, Median: {median_bkg:.3f}, Std: {std_bkg:.3f}")
        
        for radius in aperture_radii_px:
            apertures = CircularAperture(valid_positions, r=radius)
            phot_table = aperture_photometry(science_data, apertures)
            fluxes = phot_table['aperture_sum'].data
            
            # DEBUG: Check flux statistics for this radius
            median_flux = np.median(fluxes)
            mean_flux = np.mean(fluxes)
            logging.info(f"Radius {radius:.1f}px: Median flux = {median_flux:.1f}, Mean flux = {mean_flux:.1f}")
            
            growth_fluxes.append(fluxes)
        
        growth_fluxes = np.array(growth_fluxes)
        
        # Use mean for stability
        mean_growth = np.mean(growth_fluxes, axis=1)
        
        # DEBUG: Check the growth curve
        logging.info("Growth curve values:")
        for i, radius in enumerate(aperture_radii_px):
            logging.info(f"  Radius {radius:.1f}px: {mean_growth[i]:.1f}")
        
        # Check if fluxes are reasonable
        if np.max(mean_growth) < 100:
            logging.warning("Very low fluxes detected. Using fallback aperture correction.")
            return self.calculate_aperture_correction_fallback(filter_name)
        
        # Normalize to largest aperture
        total_flux = mean_growth[-1]
        if total_flux <= 0:
            logging.warning("Invalid total flux for aperture correction")
            return self.calculate_aperture_correction_fallback(filter_name)
        
        fraction_in_aperture = mean_growth / total_flux
        
        # DEBUG: Check fractions
        logging.info("Flux fractions:")
        for i, radius in enumerate(aperture_radii_px):
            logging.info(f"  Radius {radius:.1f}px: {fraction_in_aperture[i]:.3f}")
        
        # Calculate correction factors with conservative approach
        correction_factors = {}
        for ap_size in self.apertures:
            radius_px = (ap_size / 2) / self.pixel_scale
            idx = np.argmin(np.abs(aperture_radii_px - radius_px))
            flux_fraction = fraction_in_aperture[idx]
            
            logging.info(f"Aperture {ap_size}arcsec -> Radius {radius_px:.1f}px -> Fraction {flux_fraction:.3f}")
            
            if flux_fraction > 0.05:
                correction = 1.0 / flux_fraction
                # Conservative constraints
                correction = max(1.0, min(correction, 2.5))
            else:
                correction = 1.0
                logging.warning(f"Very low flux fraction ({flux_fraction:.3f}) for aperture correction")
                
            correction_factors[ap_size] = correction
            logging.info(f"Science-based aperture correction {field_name} {filter_name} {ap_size}arcsec: {correction:.3f}")
        
        # Cache results
        self.aperture_corrections[cache_key] = correction_factors
        
        # Enhanced diagnostic plot
        if self.debug:
            self.plot_aperture_correction_diagnostic(field_name, filter_name, aperture_radii_px, 
                                                   mean_growth, fraction_in_aperture, valid_positions, 
                                                   science_data, correction_factors, large_aperture_size)
        
        return correction_factors

    def calculate_aperture_correction_fallback(self, filter_name):
        """
        Fallback to simple aperture correction based on typical values for extended sources
        """
        # Valores típicos para fuentes extendidas (cúmulos globulares)
        typical_corrections = {
            'F378': {3: 1.4, 4: 1.3, 5: 1.2, 6: 1.1},
            'F395': {3: 1.4, 4: 1.3, 5: 1.2, 6: 1.1},
            'F410': {3: 1.4, 4: 1.3, 5: 1.2, 6: 1.1},
            'F430': {3: 1.4, 4: 1.3, 5: 1.2, 6: 1.1},
            'F515': {3: 1.3, 4: 1.2, 5: 1.1, 6: 1.05},
            'F660': {3: 1.3, 4: 1.2, 5: 1.1, 6: 1.05},
            'F861': {3: 1.3, 4: 1.2, 5: 1.1, 6: 1.05}
        }
        
        corrections = typical_corrections.get(filter_name, {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0})
        logging.info(f"Using fallback aperture corrections for {filter_name}: {corrections}")
        return corrections

    def plot_aperture_correction_diagnostic(self, field_name, filter_name, aperture_radii_px, 
                                          mean_growth, fraction_in_aperture, valid_positions, 
                                          science_data, correction_factors, large_aperture_size):
        """Plot diagnostic information for aperture correction"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(aperture_radii_px * self.pixel_scale, mean_growth, 'bo-', 
                label='Mean growth curve')
        plt.xlabel('Aperture Radius (arcsec)')
        plt.ylabel('Mean Flux (ADU)')
        plt.title(f'Growth Curve: {field_name} {filter_name}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(aperture_radii_px * self.pixel_scale, fraction_in_aperture, 'ro-',
                label='Flux fraction')
        for ap_size in self.apertures:
            radius_arcsec = ap_size / 2
            plt.axvline(radius_arcsec, color='gray', linestyle='--', alpha=0.7)
            plt.text(radius_arcsec, 0.1, f'{ap_size}"', rotation=90, va='bottom')
        plt.xlabel('Aperture Radius (arcsec)')
        plt.ylabel('Fraction of Total Flux')
        plt.title('Flux Fraction')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Show sample apertures on image
        if len(valid_positions) > 0:
            sample_idx = min(5, len(valid_positions) - 1)
            sample_pos = valid_positions[sample_idx]
            cutout_size = int(large_aperture_size / self.pixel_scale) * 3
            y0 = max(0, int(sample_pos[1] - cutout_size))
            y1 = min(science_data.shape[0], int(sample_pos[1] + cutout_size))
            x0 = max(0, int(sample_pos[0] - cutout_size))
            x1 = min(science_data.shape[1], int(sample_pos[0] + cutout_size))
            
            if y1 > y0 and x1 > x0:
                cutout = science_data[y0:y1, x0:x1]
                plt.imshow(cutout, origin='lower', cmap='gray')
                # Plot apertures
                for ap_size in self.apertures:
                    radius_px = (ap_size / 2) / self.pixel_scale
                    circle = plt.Circle((sample_pos[0]-x0, sample_pos[1]-y0), radius_px, 
                                      fill=False, color='red', alpha=0.7, label=f'{ap_size}"')
                    plt.gca().add_patch(circle)
                plt.title('Sample Object with Apertures')
                plt.colorbar()
        
        plt.subplot(2, 2, 4)
        corrections = [correction_factors[ap] for ap in self.apertures]
        plt.bar([str(ap) for ap in self.apertures], corrections)
        plt.xlabel('Aperture Size (arcsec)')
        plt.ylabel('Correction Factor')
        plt.title('Aperture Correction Factors')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{field_name}_{filter_name}_science_aper_correction_diagnostic.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved aperture correction diagnostic plot for {field_name} {filter_name}")

    def safe_magnitude_calculation(self, flux, zero_point):
        """
        Calculate magnitudes safely handling non-positive fluxes
        """
        return np.where(flux > 0, zero_point - 2.5 * np.log10(flux), 99.0)
    
    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None
        
        # We'll calculate aperture corrections per filter during processing
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
                
                # Apply background modeling
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
                
                # Calculate aperture correction using science objects for this filter
                if filter_name not in aperture_corrections:
                    if len(valid_positions) >= 5:
                        aperture_corrections[filter_name] = self.calculate_aperture_correction_science_objects(
                            field_name, filter_name, valid_positions, data_corrected
                        )
                    else:
                        aperture_corrections[filter_name] = self.calculate_aperture_correction_fallback(filter_name)
                        logging.warning(f"Not enough objects for aperture correction in {field_name} {filter_name}")
                
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    apertures = CircularAperture(valid_positions, r=aperture_radius_px)
                    annulus = CircularAnnulus(valid_positions, 
                                            r_in=6/self.pixel_scale, 
                                            r_out=9/self.pixel_scale)
                    
                    # Perform photometry
                    phot_table = aperture_photometry(data_corrected, apertures)
                    
                    # For background estimation, use sigma-clipped stats of the annulus
                    bkg_phot_table = aperture_photometry(data_corrected, annulus)
                    bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
                    
                    # Calculate flux (subtract any remaining background)
                    flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
                    
                    # Error estimation
                    flux_err = np.sqrt(np.abs(flux) + (apertures.area * bkg_rms**2))
                    
                    # Apply aperture correction
                    correction_factor = aperture_corrections[filter_name].get(aperture_size, 1.0)
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
                    
                    if self.debug and filter_name == self.debug_filter and aperture_size == 4:
                        self.save_debug_image(data_corrected, valid_positions, 
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
            output_file = 'Results/all_fields_gc_photometry_merged.csv'
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
