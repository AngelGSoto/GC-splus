#!/usr/bin/env python3
"""
extract_splus_gaia_xp_diagnostic.py
VERSIÓN DIAGNÓSTICO - Para entender el problema con las growth curves
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
import matplotlib.pyplot as plt
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def find_valid_image_hdu(fits_file):
    """Encuentra el HDU válido en el archivo FITS"""
    try:
        with fits.open(fits_file) as hdul:
            for i, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.header.get('NAXIS', 0) >= 2:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.is_celestial:
                            return hdu, i, wcs
                    except Exception:
                        continue
        return None, None, None
    except Exception as e:
        logging.error(f"Error abriendo {fits_file}: {e}")
        return None, None, None

def diagnostic_growth_curve(field_name, filter_name):
    """Diagnóstico simple de growth curve para una estrella individual"""
    try:
        # Cargar imagen
        image_path = f"{field_name}/{field_name}_{filter_name}.fits.fz"
        hdu, _, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            logging.error("Imagen no encontrada")
            return
        
        data = hdu.data.astype(float)
        pixscale = hdu.header.get('PIXSCALE', 0.55)
        
        # Cargar catálogo de referencia
        catalog_path = f'{field_name}_gaia_xp_matches.csv'
        ref_catalog = pd.read_csv(catalog_path)
        
        # Tomar solo las primeras 5 estrellas para diagnóstico
        positions = []
        for i, row in ref_catalog.head(5).iterrows():
            try:
                x, y = wcs.all_world2pix(row['ra'], row['dec'], 0)
                if np.isfinite(x) and np.isfinite(y) and 100 < x < data.shape[1]-100 and 100 < y < data.shape[0]-100:
                    positions.append([x, y])
            except:
                continue
        
        if not positions:
            logging.error("No se encontraron posiciones válidas")
            return
        
        positions = np.array(positions)
        
        # Aperturas para diagnóstico
        apertures_arcsec = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
        apertures_pixels = [diam/2.0 / pixscale for diam in apertures_arcsec]
        
        logging.info(f"DIAGNÓSTICO {filter_name}: Probando {len(positions)} estrellas con {len(apertures_arcsec)} aperturas")
        
        # Analizar cada estrella individualmente
        for star_idx, pos in enumerate(positions):
            plt.figure(figsize=(12, 8))
            
            # Medir flujos en todas las aperturas para esta estrella
            fluxes = []
            for aperture_radius in apertures_pixels:
                aperture = CircularAperture([pos], r=aperture_radius)
                phot = aperture_photometry(data, aperture)
                flux = phot['aperture_sum'][0]
                fluxes.append(flux)
            
            fluxes = np.array(fluxes)
            
            # Calcular fracción de flujo acumulada
            total_flux = fluxes[-1]  # Flujo en apertura más grande
            flux_fraction = fluxes / total_flux
            
            # Convertir a magnitudes
            mags = -2.5 * np.log10(np.maximum(fluxes, 1e-10))
            
            # Panel 1: Flujo vs Apertura
            plt.subplot(2, 2, 1)
            plt.plot(apertures_arcsec, fluxes, 'bo-', label=f'Star {star_idx+1}')
            plt.axvline(3.0, color='red', linestyle='--', alpha=0.7, label='3 arcsec')
            plt.xlabel('Aperture Diameter (arcsec)')
            plt.ylabel('Flux (ADU)')
            plt.title(f'Flux vs Aperture - {filter_name}')
            plt.grid(True, alpha=0.3)
            
            # Panel 2: Magnitud vs Apertura
            plt.subplot(2, 2, 2)
            plt.plot(apertures_arcsec, mags, 'ro-', label=f'Star {star_idx+1}')
            plt.axvline(3.0, color='red', linestyle='--', alpha=0.7, label='3 arcsec')
            plt.xlabel('Aperture Diameter (arcsec)')
            plt.ylabel('Instrumental Magnitude')
            plt.title('Magnitude vs Aperture')
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            
            # Panel 3: Fracción de flujo
            plt.subplot(2, 2, 3)
            plt.plot(apertures_arcsec, flux_fraction*100, 'go-', label=f'Star {star_idx+1}')
            plt.axvline(3.0, color='red', linestyle='--', alpha=0.7, label='3 arcsec')
            plt.axhline(flux_fraction[apertures_arcsec.index(3.0)]*100, color='red', linestyle=':', alpha=0.7)
            plt.xlabel('Aperture Diameter (arcsec)')
            plt.ylabel('Flux Fraction (%)')
            plt.title(f'Flux Fraction (3" = {flux_fraction[apertures_arcsec.index(3.0)]*100:.1f}%)')
            plt.grid(True, alpha=0.3)
            
            # Panel 4: Imagen de la estrella
            plt.subplot(2, 2, 4)
            x, y = int(pos[0]), int(pos[1])
            cutout_size = 50
            cutout = data[max(0, y-cutout_size):min(data.shape[0], y+cutout_size), 
                         max(0, x-cutout_size):min(data.shape[1], x+cutout_size)]
            
            if cutout.size > 0:
                plt.imshow(cutout, origin='lower', cmap='gray', 
                          vmin=np.percentile(cutout, 5), 
                          vmax=np.percentile(cutout, 95))
                plt.plot(cutout_size, cutout_size, 'r+', markersize=10, markeredgewidth=2)
                plt.title(f'Star cutout (x={x}, y={y})')
                plt.colorbar(label='ADU')
            
            plt.tight_layout()
            plt.savefig(f'diagnostic_{field_name}_{filter_name}_star{star_idx+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Información de diagnóstico en consola
            flux_3arcsec = fluxes[apertures_arcsec.index(3.0)]
            mag_3arcsec = mags[apertures_arcsec.index(3.0)]
            fraction_3arcsec = flux_fraction[apertures_arcsec.index(3.0)]
            
            logging.info(f"Estrella {star_idx+1}:")
            logging.info(f"  Flujo en 3\" = {flux_3arcsec:.1f} ADU")
            logging.info(f"  Flujo total = {total_flux:.1f} ADU")
            logging.info(f"  Fracción en 3\" = {fraction_3arcsec*100:.1f}%")
            logging.info(f"  Magnitud en 3\" = {mag_3arcsec:.2f}")
            
            # Calcular corrección para esta estrella
            if total_flux > flux_3arcsec > 0:
                ap_corr = -2.5 * np.log10(flux_3arcsec / total_flux)
                logging.info(f"  Corrección apertura = {ap_corr:.3f} mag")
            
            logging.info("  " + "-"*40)
        
        # Análisis estadístico rápido
        logging.info(f"ANÁLISIS ESTADÍSTICO {filter_name}:")
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0)
        logging.info(f"  Estadísticas imagen: Media={mean_val:.6f}, Mediana={median_val:.6f}, Std={std_val:.6f} ADU")
        
        # Verificar si hay valores extremos
        extreme_low = np.percentile(data, 1)
        extreme_high = np.percentile(data, 99)
        logging.info(f"  Percentiles: 1% = {extreme_low:.6f}, 99% = {extreme_high:.6f} ADU")
        
    except Exception as e:
        logging.error(f"Error en diagnóstico {filter_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Función principal de diagnóstico"""
    field_name = 'CenA01'
    filters_to_test = ['F378', 'F395']  # Empezar con los problemáticos
    
    for filter_name in filters_to_test:
        logging.info(f"\n{'='*60}")
        logging.info(f"DIAGNÓSTICO PARA {field_name} {filter_name}")
        logging.info(f"{'='*60}")
        diagnostic_growth_curve(field_name, filter_name)

if __name__ == '__main__':
    main()
