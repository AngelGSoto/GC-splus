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
    def __init__(self, catalog_path, zeropoints_file, background_box_size=50, debug=False, debug_filter='F660'):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Load zeropoints
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {
            row['field']: {filt: row[filt] for filt in ['F378','F395','F410','F430','F515','F660','F861']}
            for _, row in self.zeropoints_df.iterrows()
        }
        
        # Load catalog
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        logging.info(f"Loaded catalog with {len(self.catalog)} sources")
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel
        self.apertures = [3, 4, 5, 6]  # aperture diameters in arcsec
        self.background_box_size = background_box_size
        self.debug = debug
        self.debug_filter = debug_filter
        
        # Column names for RA/DEC (try both)
        self.ra_col = next((col for col in ['RAJ2000', 'RA'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RAJ2000/DEJ2000 or RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain T17ID or ID column")
        
    def get_field_center_from_header(self, field_name, filter_name='F660'):
        # Buscar archivos con las extensiones correctas
        possible_files = [
            os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz"),
            os.path.join(field_name, f"{field_name}_{filter_name}.fits")
        ]
        
        for sci_file in possible_files:
            if os.path.exists(sci_file):
                try:
                    with fits.open(sci_file) as hdul:
                        # Determinar la extensión correcta (0 o 1)
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
            # CORRECCIÓN: DAOStarFinder ya no acepta el parámetro npixels
            daofind = DAOStarFinder(fwhm=3.0, threshold=snr_threshold*std)
            sources = daofind(data - median)
            mask = np.zeros_like(data, dtype=bool)
            
            if sources is not None and len(sources) > 0:
                # Create a binary image for sources
                source_mask = np.zeros_like(data, dtype=bool)
                for i in range(len(sources)):
                    y, x = int(sources['ycentroid'][i]), int(sources['xcentroid'][i])
                    if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                        source_mask[y, x] = True

                # Dilate the source mask
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
            sigma_clip = SigmaClip(sigma=3.0)
            bkg = Background2D(data, box_size=self.background_box_size, filter_size=3,
                               sigma_clip=sigma_clip, bkg_estimator=MedianBackground(), mask=mask)
            data_subtracted = data - bkg.background
            bkg_rms = bkg.background_rms
            
            if self.debug and filter_name == self.debug_filter:
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.imshow(data, origin='lower', cmap='viridis', vmin=median-3*std, vmax=median+3*std)
                plt.title('Original')
                plt.colorbar()
                plt.subplot(132)
                plt.imshow(bkg.background, origin='lower', cmap='viridis')
                plt.title('Estimated Background')
                plt.colorbar()
                plt.subplot(133)
                plt.imshow(data_subtracted, origin='lower', cmap='viridis', vmin=median-3*std, vmax=median+3*std)
                plt.title('Background Subtracted')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f'{field_name}_{filter_name}_background_subtraction.png', dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved background debug image for {field_name} {filter_name}")
            
            logging.info(f"Background subtracted with box_size={self.background_box_size}")
            return data_subtracted, bkg_rms
        except Exception as e:
            logging.warning(f"Background subtraction failed: {e}. Using original image.")
            return data, None

    def calculate_aperture_correction(self, field_name, filter_name, large_aperture_size=10):
        """
        Calculate aperture correction factors using the already-measured reference stars
        from the Gaia XP matches catalog.
        """
        # Load the reference stars catalog for this field
        ref_stars_file = f"{field_name}_gaia_xp_matches.csv"
        if not os.path.exists(ref_stars_file):
            logging.warning(f"Reference stars file {ref_stars_file} not found. Skipping aperture correction.")
            return {ap_size: 1.0 for ap_size in self.apertures}  # Return no correction if file not found

        try:
            ref_stars_df = pd.read_csv(ref_stars_file)
            logging.info(f"Loaded {len(ref_stars_df)} reference stars from {ref_stars_file}")
        except Exception as e:
            logging.warning(f"Error loading reference stars file: {e}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        # Filter stars: good quality, not saturated, not too faint
        # Example filters - adjust based on your data
        good_stars = ref_stars_df[
            (ref_stars_df['gaia_ruwe'] < 1.4) &  # Good astrometric solution
            (ref_stars_df['gaia_phot_g_mean_mag'] > 14) &  # Not too bright (avoid saturation)
            (ref_stars_df['gaia_phot_g_mean_mag'] < 19) &  # Not too faint
            (ref_stars_df[f'flux_{filter_name}'] > 0)  # Positive flux in this filter
        ].copy()
    
        if len(good_stars) < 5:
            logging.warning(f"Not enough good reference stars ({len(good_stars)}) for aperture correction in {field_name} {filter_name}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        # Load the image to get WCS and measure large aperture fluxes
        image_path = None
        for ext in ['.fits.fz', '.fits']:
            candidate = os.path.join(field_name, f"{field_name}_{filter_name}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
    
        if not image_path:
            logging.warning(f"No image found for {field_name} {filter_name}. Skipping aperture correction.")
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
            logging.warning(f"Error loading image for aperture correction: {e}")
            return {ap_size: 1.0 for ap_size in self.apertures}

        # Subtract background
        data_bg_subtracted, bkg_rms = self.subtract_background(data, field_name, filter_name)
    
        # Get WCS for coordinate conversion
        wcs = WCS(header)
    
        # Convert RA/DEC to pixel coordinates
        coords = SkyCoord(ra=good_stars['ra'].values*u.deg, dec=good_stars['dec'].values*u.deg)
        x, y = wcs.world_to_pixel(coords)
        positions = np.column_stack((x, y))
    
        # Filter out stars too close to the edges
        height, width = data_bg_subtracted.shape
        large_radius_px = (large_aperture_size / 2) / self.pixel_scale
        valid_indices = []
    
        for i, (x_pos, y_pos) in enumerate(positions):
            if (x_pos > large_radius_px and x_pos < width - large_radius_px and
                y_pos > large_radius_px and y_pos < height - large_radius_px):
                valid_indices.append(i)
    
        if len(valid_indices) < 5:
            logging.warning(f"Not enough stars away from edges for aperture correction in {field_name} {filter_name}")
            return {ap_size: 1.0 for ap_size in self.apertures}
    
        positions = positions[valid_indices]
        good_stars = good_stars.iloc[valid_indices]
    
        # Measure fluxes in large aperture
        large_radius_px = (large_aperture_size / 2) / self.pixel_scale
        large_apertures = CircularAperture(positions, r=large_radius_px)
        phot_table_large = aperture_photometry(data_bg_subtracted, large_apertures)
        flux_large = phot_table_large['aperture_sum'].data
    
        # Calculate correction factors for each aperture size
        correction_factors = {}
    
        for aperture_size in self.apertures:
            # Get the instrumental flux from the reference catalog
            flux_col = f'flux_{filter_name}'
            flux_small = good_stars[flux_col].values
        
            # Calculate correction factors: flux_large / flux_small
            valid_ratios = []
            for i in range(len(flux_large)):
                if flux_small[i] > 0 and flux_large[i] > 0:  # Avoid division by zero
                    ratio = flux_large[i] / flux_small[i]
                    valid_ratios.append(ratio)
        
            if len(valid_ratios) > 0:
                # Use median to be robust against outliers
                correction_factor = np.median(valid_ratios)
                correction_factors[aperture_size] = correction_factor
                logging.info(f"Aperture correction for {field_name} {filter_name} {aperture_size} arcsec: {correction_factor} (based on {len(valid_ratios)} stars)")
            else:
                logging.warning(f"No valid flux ratios for aperture {aperture_size} in {field_name} {filter_name}")
                correction_factors[aperture_size] = 1.0  # No correction
    
        return correction_factors
    
    def process_field(self, field_name):
        logging.info(f"Processing field {field_name}")
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}. Skipping.")
            return None
        
        # Ensure RA/DEC are float values
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
                    # Determinar la extensión correcta (0 o 1)
                    if len(hdul) > 1 and hdul[1].data is not None:
                        header = hdul[1].header
                        data = hdul[1].data.astype(float)
                    else:
                        header = hdul[0].header
                        data = hdul[0].data.astype(float)
                
                wcs = WCS(header)
                data_bg_subtracted, bkg_rms = self.subtract_background(data, field_name, filter_name)
                
                # Calcular aperture correction para este campo y filtro
                correction_factors = self.calculate_aperture_correction(field_name, filter_name)
                if correction_factors is None:
                    # Si no se pueden calcular, usar factores de 1.0 (sin corrección)
                    correction_factors = {ap_size: 1.0 for ap_size in self.apertures}
                
                # Convert RA/DEC to float to ensure proper coordinate handling
                ra_values = field_sources[self.ra_col].astype(float).values
                dec_values = field_sources[self.dec_col].astype(float).values
                coords = SkyCoord(ra=ra_values*u.deg, dec=dec_values*u.deg)
                
                x, y = wcs.world_to_pixel(coords)
                positions = np.column_stack((x, y))
                
                for aperture_size in self.apertures:
                    aperture_radius_px = (aperture_size / 2) / self.pixel_scale
                    apertures = CircularAperture(positions, r=aperture_radius_px)
                    annulus = CircularAnnulus(positions, r_in=6/self.pixel_scale, r_out=9/self.pixel_scale)
                    
                    phot_table = aperture_photometry(data_bg_subtracted, apertures)
                    bkg_phot_table = aperture_photometry(data_bg_subtracted, annulus)
                    
                    bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
                    flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
                    
                    # ✅ CORRECCIÓN: Usar RMS local en cada posición de fuente
                    if bkg_rms is not None and bkg_rms.ndim == 2:
                        # Extraer RMS local en cada posición (x,y) de las fuentes
                        x_int = np.clip(np.round(positions[:, 0]).astype(int), 0, bkg_rms.shape[1]-1)
                        y_int = np.clip(np.round(positions[:, 1]).astype(int), 0, bkg_rms.shape[0]-1)
                        local_bkg_rms = bkg_rms[y_int, x_int]  # Array de (N,) con RMS en cada fuente
                        flux_err = np.sqrt(np.abs(flux) + (apertures.area * local_bkg_rms**2))
                    elif bkg_rms is not None:
                        # Si bkg_rms es un escalar (poco probable, pero por si acaso)
                        flux_err = np.sqrt(np.abs(flux) + (apertures.area * bkg_rms**2))
                    else:
                        # Fallback: usar std del fondo global
                        _, _, std = sigma_clipped_stats(data_bg_subtracted, sigma=3.0)
                        flux_err = np.sqrt(np.abs(flux) + (apertures.area * std**2))
                    
                    # Aplicar aperture correction
                    flux_corrected = flux * correction_factors[aperture_size]
                    flux_err_corrected = flux_err * correction_factors[aperture_size]
                    
                    zero_point = self.zeropoints[field_name][filter_name]
                    mag = [zero_point - 2.5 * np.log10(f) if f > 0 else 99.0 for f in flux_corrected]
                    mag_err = [(2.5 / np.log(10)) * (fe / f) if f > 0 else 99.0 for f, fe in zip(flux_corrected, flux_err_corrected)]
                    snr = [f / fe if fe > 0 else 0.0 for f, fe in zip(flux_corrected, flux_err_corrected)]
                    
                    # Store in DataFrame
                    results[f'X_{filter_name}_{aperture_size}'] = positions[:, 0]
                    results[f'Y_{filter_name}_{aperture_size}'] = positions[:, 1]
                    results[f'FLUX_{filter_name}_{aperture_size}'] = flux_corrected
                    results[f'FLUXERR_{filter_name}_{aperture_size}'] = flux_err_corrected
                    results[f'MAG_{filter_name}_{aperture_size}'] = mag
                    results[f'MAGERR_{filter_name}_{aperture_size}'] = mag_err
                    results[f'SNR_{filter_name}_{aperture_size}'] = snr
                    
                    # Optional: Add edge flag
                    edge_flag = [
                        (x < aperture_radius_px or x > data.shape[1] - aperture_radius_px or
                         y < aperture_radius_px or y > data.shape[0] - aperture_radius_px)
                        for x, y in positions
                    ]
                    results[f'EDGE_{filter_name}_{aperture_size}'] = edge_flag
                    
                    # Debug image for first source
                    if self.debug and filter_name == self.debug_filter and aperture_size == 4:
                        self.save_debug_image(data_bg_subtracted, positions, aperture_radius_px, 
                                             6/self.pixel_scale, 9/self.pixel_scale, 
                                             field_name, filter_name)
            
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        return results
    
    def save_debug_image(self, data, positions, aperture_radius, annulus_inner, annulus_outer, field_name, filter_name):
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        distances = np.sqrt((positions[:,0]-center_x)**2 + (positions[:,1]-center_y)**2)
        idx = np.argmin(distances)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        ax.imshow(data, origin='lower', cmap='gray', vmin=median-std, vmax=median+3*std)
        
        aperture = CircularAperture(positions[idx], r=aperture_radius)
        inner_annulus = CircularAnnulus(positions[idx], r_in=annulus_inner, r_out=annulus_outer)
        
        aperture.plot(ax=ax, color='red', lw=1.5, label='Aperture')
        inner_annulus.plot(ax=ax, color='blue', lw=1.5, label='Background Annulus')
        
        ax.set_title(f'{field_name} {filter_name} - Aperture Photometry')
        ax.legend()
        plt.savefig(f'{field_name}_{filter_name}_aperture_photometry.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved debug aperture image for {field_name} {filter_name}")

# ---- Main Execution ----

# Actualiza esta lista con todos tus campos
# fields = ['CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 'CenA07', 'CenA08', 
#           'CenA09', 'CenA10', 'CenA11', 'CenA12', 'CenA13', 'CenA14', 'CenA15', 'CenA16',
#           'CenA17', 'CenA18', 'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24']

fields = ['CenA01']

if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'all_fields_zero_points_splus_format.csv'
    
    if not os.path.exists(catalog_path):
        logging.error(f"Catalog file {catalog_path} not found")
        logging.info("Files in current directory:")
        for f in os.listdir('.'):
            logging.info(f"  {f}")
    elif not os.path.exists(zeropoints_file):
        logging.error(f"Zeropoints file {zeropoints_file} not found")
    else:
        try:
            all_results = []
            photometry = SPLUSPhotometry(catalog_path, zeropoints_file,
                                        background_box_size=50, debug=True, debug_filter='F660')
            
            for field in fields:
                # Verificar si el campo existe antes de procesarlo
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
                
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        for col, default in [
                            (f'FLUX_{filter_name}_{aperture}', 0.0),
                            (f'FLUXERR_{filter_name}_{aperture}', 99.0),
                            (f'MAG_{filter_name}_{aperture}', 99.0),
                            (f'MAGERR_{filter_name}_{aperture}', 99.0),
                            (f'SNR_{filter_name}_{aperture}', 0.0)
                        ]:
                            if col in final_results.columns:
                                final_results[col].fillna(default, inplace=True)
                
                final_results.rename(columns={photometry.id_col: 'T17ID'}, inplace=True)
                merged_catalog = photometry.original_catalog.merge(
                    final_results, on='T17ID', how='left', suffixes=('', '_phot')
                )
                
                for col in ['RAJ2000_phot', 'DEJ2000_phot']:
                    if col in merged_catalog.columns:
                        merged_catalog.drop(col, axis=1, inplace=True)
                
                column_order = list(photometry.original_catalog.columns)
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        column_order.extend([
                            f'X_{filter_name}_{aperture}', f'Y_{filter_name}_{aperture}',
                            f'FLUX_{filter_name}_{aperture}', f'FLUXERR_{filter_name}_{aperture}',
                            f'MAG_{filter_name}_{aperture}', f'MAGERR_{filter_name}_{aperture}',
                            f'SNR_{filter_name}_{aperture}', f'EDGE_{filter_name}_{aperture}'
                        ])
                column_order.append('FIELD')
                
                for col in column_order:
                    if col not in merged_catalog.columns:
                        merged_catalog[col] = np.nan
                
                merged_catalog = merged_catalog[column_order]
                output_file = 'all_fields_gc_photometry_merged.csv'
                merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
                logging.info(f"Final merged results saved to {output_file}")
                
                # Quality summary
                logging.info(f"Total sources in original catalog: {len(photometry.original_catalog)}")
                logging.info(f"Total sources with measurements: {len(final_results)}")
                
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        mag_col = f'MAG_{filter_name}_{aperture}'
                        snr_col = f'SNR_{filter_name}_{aperture}'
                        if mag_col in merged_catalog.columns:
                            valid_measurements = (merged_catalog[mag_col] != 99.0).sum()
                            
                            if valid_measurements > 0:
                                valid_data = merged_catalog[merged_catalog[mag_col] != 99.0]
                                mean_snr = valid_data[snr_col].mean()
                                mean_mag = valid_data[mag_col].mean()
                                logging.info(f"{filter_name}_{aperture}: {valid_measurements} valid, "
                                             f"Mean SNR: {mean_snr:.1f}, Mean Mag: {mean_mag:.2f}")
                            else:
                                logging.info(f"{filter_name}_{aperture}: 0 valid measurements")
                        else:
                            logging.info(f"{filter_name}_{aperture}: Column not found in results")
            else:
                logging.warning("No results obtained for any field.")
                output_file = 'all_fields_gc_photometry_merged.csv'
                photometry.original_catalog.to_csv(output_file, index=False)
                logging.info(f"No fields processed. Original catalog saved to {output_file}")
        
        except Exception as e:
            logging.error(f"Execution error: {e}")
            traceback.print_exc()
