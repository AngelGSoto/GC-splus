#!/usr/bin/env python3
"""
Splus_photometry_buzzo_method.py
METODOLOGÍA BUZZO et al. (2021) PARA CÚMULOS GLOBULARES
- Unsharp masking para fondo galáctico
- Apertura 2" + corrección de apertura  
- Anillo de fondo 3-4"
- Cálculo de curvas de crecimiento
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
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm
import warnings
import os
import logging
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
warnings.filterwarnings('ignore')

class SPLUSGCPhotometryBuzzo:
    def __init__(self, catalog_path, zeropoints_file, debug=False):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file {catalog_path} does not exist")
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"Zeropoints file {zeropoints_file} does not exist")
        
        # Cargar zero points 
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
        
        # ✅ CONFIGURACIÓN BUZZO et al. (2021)
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel
        self.aperture_diam = 2.0  # ✅ 2" diámetro (Buzzo et al.)
        self.apertures = [2, 3, 4, 5, 6]  # Incluir 2" como principal
        self.debug = debug
        
        # Parámetros unsharp masking (Buzzo et al.)
        self.median_box_size = 25  # 25x25 pixels
        self.gaussian_sigma = 5    # pixels
        
        # Mapeo de columnas del catálogo
        self.ra_col = next((col for col in ['RAJ2000', 'RA', 'ra'] if col in self.catalog.columns), None)
        self.dec_col = next((col for col in ['DEJ2000', 'DEC', 'dec'] if col in self.catalog.columns), None)
        if not self.ra_col or not self.dec_col:
            raise ValueError("Catalog must contain RA/DEC columns")
        self.id_col = next((col for col in ['T17ID', 'ID', 'id'] if col in self.catalog.columns), None)
        if not self.id_col:
            raise ValueError("Catalog must contain ID column")
        
        logging.info(f"Using Buzzo et al. methodology: 2\" aperture, unsharp masking, aperture correction")

    def create_unsharp_mask(self, data):
        """
        ✅ UNSHARP MASKING según Buzzo et al. (2021)
        - Median filter 25x25 pixels
        - Gaussian smoothing σ=5 pixels
        """
        try:
            # Aplicar filtro mediano
            median_filtered = median_filter(data, size=self.median_box_size)
            
            # Aplicar suavizado gaussiano
            gaussian_smoothed = gaussian_filter(median_filtered, sigma=self.gaussian_sigma)
            
            # Crear unsharp mask restando la versión suavizada
            unsharp_mask = data - gaussian_smoothed
            
            logging.info("Applied unsharp masking (25x25 median + σ=5 gaussian)")
            return unsharp_mask
            
        except Exception as e:
            logging.error(f"Error in unsharp masking: {e}")
            return data

    def find_isolated_gcs(self, positions, image_shape, n_isolated=5):
        """
        ✅ IDENTIFICAR GCs AISLADOS para corrección de apertura
        Según Buzzo et al., usar GCs en regiones externas
        """
        height, width = image_shape
        margin = 0.15  # Usar fuentes en 15% exterior de la imagen
        
        # Calcular distancias al centro
        center_x, center_y = width / 2, height / 2
        distances = np.sqrt((positions[:, 0] - center_x)**2 + (positions[:, 1] - center_y)**2)
        
        # Seleccionar los más alejados del centro
        max_distance = np.max(distances)
        isolated_mask = distances > (1 - margin) * max_distance
        
        isolated_positions = positions[isolated_mask]
        
        # Seleccionar hasta n_isolated
        if len(isolated_positions) > n_isolated:
            # Ordenar por distancia (más lejanos primero)
            isolated_distances = distances[isolated_mask]
            sorted_indices = np.argsort(-isolated_distances)  # Descending
            isolated_positions = isolated_positions[sorted_indices[:n_isolated]]
        
        logging.info(f"Selected {len(isolated_positions)} isolated GCs for aperture correction")
        return isolated_positions

    def calculate_aperture_correction(self, isolated_positions, data, filter_name):
        """
        ✅ CALCULAR CORRECCIÓN DE APERTURA según Buzzo et al.
        - Curvas de crecimiento hasta 30"
        - Plateau alrededor de 6"
        - Corrección = mag(6") - mag(2")
        """
        try:
            aperture_corrections = []
            
            # Aperturas para curva de crecimiento (2" a 30")
            apertures_arcsec = np.arange(2, 31, 0.5)
            reference_aperture = 6.0  # Donde se estabiliza según Buzzo
            
            for pos in isolated_positions:
                growth_curve = []
                
                for ap_diam in apertures_arcsec:
                    ap_radius = (ap_diam / 2) / self.pixel_scale
                    
                    # Anillo de fondo fijo (3-4" como en Buzzo)
                    bkg_inner = (3.0 / 2) / self.pixel_scale
                    bkg_outer = (4.0 / 2) / self.pixel_scale
                    
                    aperture = CircularAperture([pos], r=ap_radius)
                    annulus = CircularAnnulus([pos], r_in=bkg_inner, r_out=bkg_outer)
                    
                    # Fotometría
                    phot = aperture_photometry(data, aperture)
                    bkg_phot = aperture_photometry(data, annulus)
                    
                    # Estimación de fondo con sigma-clipping
                    annulus_mask = annulus.to_mask(method='center')[0]
                    annulus_data = annulus_mask.multiply(data)
                    annulus_data_1d = annulus_data[annulus_mask.data > 0]
                    
                    _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)
                    
                    # Flujo neto y magnitud
                    flux_net = phot['aperture_sum'][0] - (bkg_median * aperture.area)
                    
                    if flux_net > 0:
                        mag = -2.5 * np.log10(flux_net)
                        growth_curve.append((ap_diam, mag))
                
                # Calcular corrección si tenemos suficientes puntos
                if len(growth_curve) > 10:
                    ap_diams, mags = zip(*growth_curve)
                    
                    # Interpolar para encontrar magnitudes específicas
                    interp_func = interp1d(ap_diams, mags, kind='linear', 
                                         fill_value='extrapolate')
                    
                    try:
                        mag_2arcsec = interp_func(2.0)
                        mag_6arcsec = interp_func(reference_aperture)
                        aperture_correction = mag_6arcsec - mag_2arcsec
                        aperture_corrections.append(aperture_correction)
                        
                        if self.debug:
                            logging.debug(f"GC at ({pos[0]:.1f}, {pos[1]:.1f}): "
                                        f"Δm = {mag_6arcsec:.3f} - {mag_2arcsec:.3f} = {aperture_correction:.3f}")
                    except:
                        continue
            
            if aperture_corrections:
                final_correction = np.median(aperture_corrections)
                logging.info(f"{filter_name}: Aperture correction = {final_correction:.3f} mag "
                           f"(based on {len(aperture_corrections)} GCs)")
                return final_correction
            else:
                logging.warning(f"{filter_name}: Could not calculate aperture correction")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error calculating aperture correction: {e}")
            return 0.0

    def perform_buzzo_photometry(self, data, positions, filter_name):
        """
        ✅ FOTOMETRÍA SEGÚN BUZZO et al.
        - Apertura: 2" diámetro
        - Sky annulus: 3-4" 
        - Estimación de fondo local
        """
        try:
            # ✅ APERTURA 2" (Buzzo et al.)
            aperture_radius = (self.aperture_diam / 2) / self.pixel_scale
            annulus_inner = (3.0 / 2) / self.pixel_scale  # 3" inner radius
            annulus_outer = (4.0 / 2) / self.pixel_scale  # 4" outer radius
            
            apertures = CircularAperture(positions, r=aperture_radius)
            annulus = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)
            
            # Fotometría
            phot_table = aperture_photometry(data, apertures)
            bkg_table = aperture_photometry(data, annulus)
            
            raw_flux = phot_table['aperture_sum'].data
            aperture_area = apertures.area
            
            # Estimación robusta del fondo para cada fuente
            bkg_mean_per_pixel = np.zeros(len(positions))
            bkg_std_per_pixel = np.zeros(len(positions))
            
            for i, pos in enumerate(positions):
                try:
                    mask = annulus.to_mask(method='center')[i]
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    
                    if len(annulus_data_1d) > 10:
                        mean, median, std = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)
                        bkg_mean_per_pixel[i] = median
                        bkg_std_per_pixel[i] = std
                    else:
                        bkg_mean_per_pixel[i] = np.median(annulus_data_1d) if len(annulus_data_1d) > 0 else 0.0
                        bkg_std_per_pixel[i] = np.std(annulus_data_1d) if len(annulus_data_1d) > 0 else 1.0
                except:
                    bkg_mean_per_pixel[i] = bkg_table['aperture_sum'][i] / annulus.area
                    bkg_std_per_pixel[i] = 1.0
                    
            # Flujo neto
            net_flux = raw_flux - (bkg_mean_per_pixel * aperture_area)
            
            # Estimación de errores
            net_flux_err = np.sqrt(np.abs(net_flux) + (aperture_area * bkg_std_per_pixel**2))
            snr = np.where(net_flux_err > 0, net_flux / net_flux_err, 0.0)
            
            logging.info(f"{filter_name}: Buzzo photometry 2\" - {len(positions)} sources, "
                        f"background: {np.median(bkg_mean_per_pixel):.4f} ADU/pix")
            return net_flux, net_flux_err, snr, aperture_radius
            
        except Exception as e:
            logging.error(f"Error in Buzzo photometry: {e}")
            n = len(positions)
            return np.zeros(n), np.full(n, 99.0), np.zeros(n), (self.aperture_diam/2)/self.pixel_scale

    def find_image_file(self, field_name, filter_name):
        """Buscar archivos de imagen"""
        for ext in [f"{field_name}_{filter_name}.fits.fz", f"{field_name}_{filter_name}.fits"]:
            path = os.path.join(field_name, ext)
            if os.path.exists(path):
                return path
        return None

    def calculate_magnitudes_buzzo(self, flux, flux_err, zero_point, aperture_correction):
        """
        ✅ CÁLCULO DE MAGNITUDES con corrección de apertura
        mag = -2.5 * log10(flux) + ZP + aperture_correction
        """
        min_flux = 1e-10
        valid_mask = (flux > min_flux) & (flux_err > 0) & np.isfinite(flux) & np.isfinite(flux_err)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mag_inst = np.where(valid_mask, -2.5 * np.log10(flux), 99.0)
            mag = np.where(valid_mask, mag_inst + zero_point + aperture_correction, 99.0)
            mag_err = np.where(valid_mask, (2.5 / np.log(10)) * (flux_err / flux), 99.0)
        
        # Filtrar magnitudes razonables
        reasonable_mag = (mag >= 10.0) & (mag <= 30.0)
        reasonable_err = (mag_err >= 0.0) & (mag_err <= 5.0)
        final_mask = valid_mask & reasonable_mag & reasonable_err
        
        mag = np.where(final_mask, mag, 99.0)
        mag_err = np.where(final_mask, mag_err, 99.0)
        
        return mag, mag_err, final_mask

    def get_field_center_from_header(self, field_name, filter_name='F660'):
        """Obtener centro del campo desde header"""
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
        """Verificar si fuente está dentro del campo"""
        if field_ra is None or field_dec is None:
            return False
        c1 = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        c2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        return c1.separation(c2).degree <= radius

    def process_field_buzzo(self, field_name):
        """Procesar campo completo con metodología Buzzo et al."""
        logging.info(f"🎯 Processing field {field_name} (BUZZO et al. METHODOLOGY)")
        
        if not os.path.exists(field_name):
            logging.warning(f"Field directory {field_name} does not exist")
            return None
        
        # Obtener centro del campo
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            logging.warning(f"Could not get field center for {field_name}")
            return None
        
        # Filtrar fuentes en el campo
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
        
        results = field_sources.copy()
        
        # Procesar cada filtro
        for filter_name in self.filters:
            logging.info(f"  Processing filter {filter_name}")
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
                        continue
                
                # Obtener WCS y convertir coordenadas
                ra_vals = field_sources[self.ra_col].astype(float).values
                dec_vals = field_sources[self.dec_col].astype(float).values
                try:
                    wcs = WCS(header)
                    coords = SkyCoord(ra=ra_vals*u.deg, dec=dec_vals*u.deg)
                    x, y = wcs.world_to_pixel(coords)
                except:
                    logging.warning("Using catalog coordinates as pixels (no WCS)")
                    x, y = ra_vals, dec_vals
                
                positions = np.column_stack((x, y))
                height, width = data.shape
                
                # Validar posiciones dentro de imagen
                margin = 100
                valid_mask = (
                    (x >= margin) & (x < width - margin) & 
                    (y >= margin) & (y < height - margin) &
                    np.isfinite(x) & np.isfinite(y)
                )
                valid_positions = positions[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_positions) == 0:
                    logging.warning(f"    No valid positions for {filter_name}")
                    continue
                
                # ✅ 1. APLICAR UNSHARP MASKING (Buzzo et al.)
                data_unsharp = self.create_unsharp_mask(data)
                
                # ✅ 2. IDENTIFICAR GCs AISLADOS para corrección de apertura
                isolated_positions = self.find_isolated_gcs(valid_positions, data.shape)
                
                # ✅ 3. CALCULAR CORRECCIÓN DE APERTURA
                aperture_correction = self.calculate_aperture_correction(
                    isolated_positions, data_unsharp, filter_name)
                
                # ✅ 4. FOTOMETRÍA BUZZO (2" aperture)
                flux, flux_err, snr, aperture_radius = self.perform_buzzo_photometry(
                    data_unsharp, valid_positions, filter_name)
                
                # ✅ 5. OBTENER ZERO POINT
                zero_point = self.zeropoints.get(field_name, {}).get(filter_name, 0.0)
                if zero_point == 0.0:
                    logging.warning(f"    No zero point for {field_name} {filter_name}")
                
                # ✅ 6. CÁLCULO DE MAGNITUDES CON CORRECCIÓN
                mag, mag_err, final_mask = self.calculate_magnitudes_buzzo(
                    flux, flux_err, zero_point, aperture_correction)
                
                # Aplicar máscara de calidad final
                mag = np.where(final_mask, mag, 99.0)
                mag_err = np.where(final_mask, mag_err, 99.0)
                snr = np.where(final_mask, snr, 0.0)
                
                # Guardar resultados para apertura 2"
                for i, idx in enumerate(valid_indices):
                    source_idx = results.index[idx]
                    prefix = f"{filter_name}_2"
                    results.loc[source_idx, f'FLUX_{prefix}'] = flux[i]
                    results.loc[source_idx, f'FLUXERR_{prefix}'] = flux_err[i]
                    results.loc[source_idx, f'MAG_{prefix}'] = mag[i]
                    results.loc[source_idx, f'MAGERR_{prefix}'] = mag_err[i]
                    results.loc[source_idx, f'SNR_{prefix}'] = snr[i]
                    results.loc[source_idx, f'AP_CORR_{prefix}'] = aperture_correction
                    results.loc[source_idx, f'USED_UNSHARP_{filter_name}'] = True
                
                # Estadísticas
                valid_mags = mag[final_mask]
                if len(valid_mags) > 0:
                    stats = {
                        'min': np.min(valid_mags),
                        'max': np.max(valid_mags),
                        'median': np.median(valid_mags),
                        'std': np.std(valid_mags),
                        'n_valid': len(valid_mags),
                        'total': len(valid_positions)
                    }
                    
                    logging.info(f"    {filter_name}: {stats['n_valid']}/{stats['total']} válidas "
                                f"[{stats['min']:.2f}, {stats['max']:.2f}], "
                                f"mediana={stats['median']:.2f} ± {stats['std']:.2f}, "
                                f"Δap={aperture_correction:.3f}")
                else:
                    logging.warning(f"    {filter_name}: No valid magnitudes")
                
                logging.info(f"    ✅ Completed {filter_name} (Buzzo method)")
                
            except Exception as e:
                logging.error(f"Error processing {field_name} {filter_name}: {e}")
                continue
        
        return results

def main():
    """Función principal"""
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    
    # Usar zero points calibrados con estrellas (script 1)
    zeropoints_file = 'Results//all_fields_zero_points_splus_format_3arcsec.csv'
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if not os.path.exists(zeropoints_file):
        raise FileNotFoundError(f"Zero-points not found: {zeropoints_file}")
    
    # Modo de prueba
    test_mode = True
    fields = ['CenA01'] if test_mode else [f'CenA{i:02d}' for i in range(1, 25)]
    
    photometry = SPLUSGCPhotometryBuzzo(
        catalog_path=catalog_path,
        zeropoints_file=zeropoints_file,
        debug=True
    )
    
    all_results = []
    for field in tqdm(fields, desc="Processing fields"):
        results = photometry.process_field_buzzo(field)
        if results is not None and len(results) > 0:
            results['FIELD'] = field
            all_results.append(results)
            
            output_file = f'{field}_gc_photometry_buzzo_method.csv'
            results.to_csv(output_file, index=False)
            logging.info(f"✅ Saved {field} results to {output_file}")
    
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        output_file = 'Results/all_fields_gc_photometry_buzzo_method.csv'
        final_results.to_csv(output_file, index=False)
        logging.info(f"🎉 Final catalog saved: {output_file}")
        
        # Estadísticas finales
        logging.info("\n📊 ESTADÍSTICAS FINALES (BUZZO METHOD):")
        total_sources = len(final_results)
        measured_sources = len(final_results[final_results['FIELD'].notna()])
        logging.info(f"Fuentes totales en catálogo: {len(photometry.original_catalog)}")
        logging.info(f"Fuentes medidas: {measured_sources}")
        logging.info(f"Metodología: Unsharp masking + 2\" aperture + aperture correction")

if __name__ == "__main__":
    main()
