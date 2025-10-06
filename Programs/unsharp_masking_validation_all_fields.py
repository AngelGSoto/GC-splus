#!/usr/bin/env python3
"""
buzzeo_galaxy_subtraction_full.py
EXACT IMPLEMENTATION OF BUZZEO ET AL. 2022 METHODOLOGY
Full processing of all filters and fields
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
import warnings
from pathlib import Path
import traceback
from tqdm import tqdm

# Configuración
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class BuzzeoGalaxySubtractionFull:
    def __init__(self, base_path="../anac_data"):
        self.base_path = Path(base_path)
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.fields = [f'CenA{i:02d}' for i in range(1, 25)]  # Todos los campos
        
        # EXACT PARAMETERS FROM BUZZEO ET AL. 2022
        self.median_box_size = 25  # 25x25 pixels median box
        self.gaussian_sigma = 5    # Gaussian smoothing with σ=5 pixels
        
        # Fields that should always get galaxy subtraction
        self.galaxy_center_fields = ['CenA11', 'CenA12', 'CenA13', 'CenA14', 'CenA15',
                                   'CenA16', 'CenA17', 'CenA18', 'CenA19', 'CenA20']
        
        self.output_dir = Path("buzzeo_galaxy_subtraction_full")
        self.output_dir.mkdir(exist_ok=True)
        
        logging.info(f"🎯 Implementing Buzzeo et al. 2022 Galaxy Subtraction - FULL")
        logging.info(f"   Processing {len(self.fields)} fields and {len(self.filters)} filters")

    def find_image_file(self, field, filter_name):
        """Find image file"""
        patterns = [
            f"{field}/{field}_{filter_name}.fits.fz",
            f"{field}/{field}_{filter_name}.fits",
            f"{field}_{filter_name}.fits.fz", 
            f"{field}_{filter_name}.fits"
        ]
        for pattern in patterns:
            path = self.base_path / pattern
            if path.exists():
                return path
        return None

    def apply_buzzeo_galaxy_subtraction(self, data):
        """
        EXACT METHOD FROM BUZZEO ET AL. 2022
        """
        try:
            # Step 1: Apply 25x25 pixel median filter
            median_filtered = median_filter(data, size=self.median_box_size)
            
            # Step 2: Apply Gaussian smoothing with σ=5 pixels
            galaxy_model = gaussian_filter(median_filtered, sigma=self.gaussian_sigma)
            
            # Step 3: Subtract galaxy model to get residual image
            residual_image = data - galaxy_model
            
            return residual_image, galaxy_model, median_filtered
            
        except Exception as e:
            logging.error(f"Error in Buzzeo galaxy subtraction: {e}")
            return data, np.zeros_like(data), data

    def process_field_filter(self, field, filter_name):
        """
        Process single field and filter using Buzzeo methodology
        """
        try:
            image_path = self.find_image_file(field, filter_name)
            if not image_path:
                logging.warning(f"   Image not found for {field} {filter_name}")
                return None
                
            # Read data
            with fits.open(image_path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        original_data = hdu.data.astype(float)
                        header = hdu.header
                        break
                else:
                    logging.warning(f"   No data in {image_path}")
                    return None
            
            header_info = self.extract_header_info(header)
            
            # Apply Buzzeo galaxy subtraction to central galaxy fields
            if field in self.galaxy_center_fields:
                residual_data, galaxy_model, median_filtered = self.apply_buzzeo_galaxy_subtraction(original_data)
                galaxy_subtracted = True
            else:
                residual_data = original_data
                galaxy_model = np.zeros_like(original_data)
                median_filtered = original_data
                galaxy_subtracted = False
            
            # Save residual image for photometry
            residual_path = self.save_residual_image(residual_data, header, field, filter_name, galaxy_subtracted)
            
            # Create verification plot (only for first few to save time)
            plot_path = None
            if field in ['CenA11', 'CenA12', 'CenA13']:  # Only create plots for first 3 fields
                plot_path = self.create_buzzeo_plot(original_data, residual_data, galaxy_model, median_filtered,
                                                  field, filter_name, header_info, galaxy_subtracted)
            
            # Calculate subtraction metrics
            metrics = self.calculate_subtraction_metrics(original_data, residual_data)
            
            return {
                'field': field,
                'filter': filter_name,
                'galaxy_subtracted': galaxy_subtracted,
                'original_image': str(image_path),
                'residual_image': str(residual_path),
                'plot_path': str(plot_path) if plot_path else "Not generated",
                **metrics,
                **header_info
            }
            
        except Exception as e:
            logging.error(f"💥 Error processing {field} {filter_name}: {e}")
            return None

    def extract_header_info(self, header):
        """Extract header information"""
        return {
            'pixel_scale': header.get('PIXSCALE', 0.55),
            'seeing_fwhm': header.get('FWHMMEAN', 1.8),
            'exptime': header.get('EXPTIME', header.get('TEXPOSED', 870.0)),
            'gain': header.get('GAIN', 825.35),
            'filter': header.get('FILTER', header.get('BAND', 'Unknown')),
            'field': header.get('FIELD', 'Unknown')
        }

    def save_residual_image(self, data, header, field, filter_name, galaxy_subtracted):
        """Save residual image for photometry"""
        try:
            hdu_new = fits.PrimaryHDU(data=data, header=header)
            hdu_new.header['HISTORY'] = 'Galaxy subtraction using Buzzeo et al. 2022 method'
            hdu_new.header['HISTORY'] = f'Median box size: {self.median_box_size}'
            hdu_new.header['HISTORY'] = f'Gaussian sigma: {self.gaussian_sigma}'
            hdu_new.header['HISTORY'] = f'Galaxy subtraction applied: {galaxy_subtracted}'
            
            output_dir = self.output_dir / 'residual_images' / field
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f'{field}_{filter_name}_buzzeo_residual.fits'
            hdu_new.writeto(output_path, overwrite=True)
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving residual image: {e}")
            return None

    def calculate_subtraction_metrics(self, original, residual):
        """Calculate galaxy subtraction metrics"""
        try:
            # Calculate reduction in extended structure
            orig_low_freq = gaussian_filter(original, sigma=10)
            resid_low_freq = gaussian_filter(residual, sigma=10)
            
            low_freq_reduction = np.std(resid_low_freq) / np.std(orig_low_freq) if np.std(orig_low_freq) > 0 else 1.0
            
            return {
                'mean_original': np.mean(original),
                'mean_residual': np.mean(residual),
                'std_original': np.std(original),
                'std_residual': np.std(residual),
                'low_frequency_reduction': low_freq_reduction
            }
        except:
            return {}

    def create_buzzeo_plot(self, original, residual, galaxy_model, median_filtered,
                          field, filter_name, header_info, galaxy_subtracted):
        """
        Create verification plot showing Buzzeo method steps
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            title_suffix = " (Galaxy Subtracted)" if galaxy_subtracted else " (No Subtraction)"
            fig.suptitle(f'Buzzeo et al. 2022 - {field} {filter_name}{title_suffix}\n'
                        f'Median box: {self.median_box_size}, Gaussian σ: {self.gaussian_sigma}', 
                        fontsize=12, fontweight='bold')
            
            # Use consistent scaling for comparison
            vmin, vmax = np.percentile(original, [5, 95])
            
            # 1. Original image
            im1 = axes[0, 0].imshow(original, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            
            # 2. Median filtered (25x25)
            im2 = axes[0, 1].imshow(median_filtered, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 1].set_title(f'Median Filtered')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
            
            # 3. Galaxy model (Gaussian smoothed)
            im3 = axes[0, 2].imshow(galaxy_model, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, 2].set_title('Galaxy Model')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
            
            # 4. Residual image (KEY RESULT)
            im4 = axes[1, 0].imshow(residual, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[1, 0].set_title('Residual Image')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
            
            # 5. Difference (what was removed)
            difference = original - residual
            vmin_diff, vmax_diff = np.percentile(difference, [5, 95])
            im5 = axes[1, 1].imshow(difference, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, origin='lower')
            axes[1, 1].set_title('Galaxy Component')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
            
            # 6. Histograms
            axes[1, 2].hist(original.flatten(), bins=100, alpha=0.7, 
                           color='blue', label='Original', log=True, density=True)
            axes[1, 2].hist(residual.flatten(), bins=100, alpha=0.7,
                           color='red', label='Residual', log=True, density=True)
            axes[1, 2].set_xlabel('Pixel Value')
            axes[1, 2].set_ylabel('Density (log)')
            axes[1, 2].set_title('Histogram Comparison')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_dir = self.output_dir / 'verification_plots' / field
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f'{field}_{filter_name}_buzzeo_verification.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logging.error(f"Error creating Buzzeo plot: {e}")
            return None

    def run_full_processing(self, test_mode=False):
        """
        Run full Buzzeo galaxy subtraction processing
        """
        if test_mode:
            fields_to_process = ['CenA11', 'CenA12']  # Solo 2 campos para prueba
            logging.info("🔧 TEST MODE: Processing 2 fields")
        else:
            fields_to_process = self.fields
            logging.info(f"🚀 FULL MODE: Processing {len(fields_to_process)} fields")
        
        results = []
        total_jobs = len(fields_to_process) * len(self.filters)
        
        with tqdm(total=total_jobs, desc="Processing fields/filters") as pbar:
            for field in fields_to_process:
                for filter_name in self.filters:
                    result = self.process_field_filter(field, filter_name)
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        self.generate_full_report(results)
        return results

    def generate_full_report(self, results):
        """Generate full Buzzeo method report"""
        if not results:
            logging.warning("No results to report")
            return
            
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'buzzeo_subtraction_results_full.csv', index=False)
        
        # Summary statistics
        total_processed = len(df)
        galaxy_subtracted = df['galaxy_subtracted'].sum()
        
        print("\n" + "="*70)
        print("🎯 BUZZEO ET AL. 2022 GALAXY SUBTRACTION - FULL REPORT")
        print("="*70)
        print(f"Total images processed: {total_processed}")
        print(f"Galaxy subtraction applied: {galaxy_subtracted} ({galaxy_subtracted/total_processed:.1%})")
        
        if 'low_frequency_reduction' in df.columns:
            avg_reduction = df[df['galaxy_subtracted']]['low_frequency_reduction'].mean()
            print(f"Average low-frequency reduction (galaxy fields): {avg_reduction:.1%}")
        
        # Summary by field
        field_summary = df.groupby('field').agg({
            'galaxy_subtracted': 'sum',
            'low_frequency_reduction': 'mean'
        }).round(3)
        
        print(f"\nField summary:")
        print(field_summary)
        
        print(f"\n📁 Residual images: {self.output_dir / 'residual_images'}")
        print(f"📊 Verification plots: {self.output_dir / 'verification_plots'}")
        print("="*70)

def main():
    processor = BuzzeoGalaxySubtractionFull(base_path="../anac_data")
    
    # Cambiar a test_mode=False para procesar todos los campos
    test_mode = True
    results = processor.run_full_processing(test_mode=test_mode)
    logging.info("✅ BUZZEO METHODOLOGY COMPLETED")

if __name__ == "__main__":
    main()
