#!/usr/bin/env python3
"""
extract_splus_gaia_xp_NO_AC.py - Sin aperture correction, solo método SPLUS básico
Basado estrictamente en: minstr = −2.5 log10(ADUm|3'') + 20
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
APERTURE_DIAMETER = 3.0  # 3 arcsec DIÁMETRO según SPLUS

def find_valid_image_hdu(fits_file):
    """Encuentra la primera HDU válida con datos y WCS"""
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

def get_reference_positions_from_catalog(catalog_path, image_path):
    """Obtiene posiciones de referencia desde el catálogo"""
    try:
        ref_catalog = pd.read_csv(catalog_path)
        
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if wcs is None:
            logging.error(f"No se pudo encontrar WCS válido en {image_path}")
            return np.array([]), pd.DataFrame()
        
        data_shape = hdu.data.shape
        ra_deg = ref_catalog['ra'].values
        dec_deg = ref_catalog['dec'].values
        
        # CONVERSIÓN ROBUSTA DE COORDENADAS
        try:
            x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
        except Exception as e:
            logging.warning(f"all_world2pix falló: {e}")
            try:
                coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
                x, y = wcs.world_to_pixel(coords)
            except Exception as e2:
                logging.error(f"Todos los métodos de conversión fallaron: {e2}")
                return np.array([]), pd.DataFrame()
        
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        
        if data_shape is not None:
            margin_pixels = 100
            valid_mask = valid_mask & (x > margin_pixels) & (x < data_shape[1] - margin_pixels)
            valid_mask = valid_mask & (y > margin_pixels) & (y < data_shape[0] - margin_pixels)
        
        positions = np.column_stack((x[valid_mask], y[valid_mask]))
        logging.info(f"Encontradas {len(positions)} posiciones válidas")
        
        return positions, ref_catalog[valid_mask].copy()
        
    except Exception as e:
        logging.error(f"Error obteniendo posiciones: {e}")
        return np.array([]), pd.DataFrame()

def find_image_file(field_dir, field_name, filter_name):
    """Encuentra el archivo de imagen"""
    possible_extensions = [
        f"{field_name}_{filter_name}.fits.fz",
        f"{field_name}_{filter_name}.fits"
    ]
    
    for ext in possible_extensions:
        image_path = os.path.join(field_dir, ext)
        if os.path.exists(image_path):
            return image_path
    
    return None

def analyze_gaia_magnitudes(catalog_path):
    """Analiza las magnitudes Gaia de las estrellas de referencia"""
    try:
        df = pd.read_csv(catalog_path)
        
        print(f"\n=== ANÁLISIS DE MAGNITUDES GAIA ===")
        print(f"Total de estrellas: {len(df)}")
        
        if 'phot_g_mean_mag' in df.columns:
            g_mags = df['phot_g_mean_mag']
            print(f"Magnitud G: min={g_mags.min():.2f}, mediana={g_mags.median():.2f}, max={g_mags.max():.2f}")
            
            # Distribución por brillo
            bins = [0, 15, 16, 17, 18, 19, 20, 25]
            for i in range(len(bins)-1):
                count = len(g_mags[(g_mags >= bins[i]) & (g_mags < bins[i+1])])
                print(f"  G {bins[i]:2d}-{bins[i+1]:2d}: {count} estrellas")
        
        return True
    except Exception as e:
        logging.error(f"Error analizando magnitudes Gaia: {e}")
        return False

def extract_instrumental_magnitudes_NO_AC(image_path, reference_positions, ref_catalog, field_name, filter_name):
    """
    ✅ VERSIÓN SIMPLIFICADA: Solo fórmula SPLUS básica SIN aperture correction
    minstr = −2.5 log10(ADUm|3'') + 20
    """
    try:
        hdu, hdu_index, wcs = find_valid_image_hdu(image_path)
        if hdu is None:
            logging.error(f"No se pudo encontrar HDU válida en {image_path}")
            return pd.DataFrame()
        
        data = hdu.data.astype(float)
        header = hdu.header
        
        pixscale = header.get('PIXSCALE', 0.55)
        logging.info(f"{filter_name}: Procesando imagen {data.shape}, pixscale={pixscale}")
        
        # 1. Fotometría con apertura de 3 arcsec (DIÁMETRO)
        aperture_radius_px = (APERTURE_DIAMETER / 2.0) / pixscale
        ann_in_px, ann_out_px = 6.0 / pixscale, 9.0 / pixscale
        
        apertures = CircularAperture(reference_positions, r=aperture_radius_px)
        annulus = CircularAnnulus(reference_positions, r_in=ann_in_px, r_out=ann_out_px)
        
        # Fotometría
        phot_table = aperture_photometry(data, apertures)
        bkg_phot_table = aperture_photometry(data, annulus)
        
        # Restar fondo local
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus.area
        flux_3arcsec = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
        
        # 2. ✅ APLICAR FÓRMULA SPLUS EXACTA (SIN APERTURE CORRECTION)
        # minstr = −2.5 log10(ADUm|3'') + 20
        minstr = -2.5 * np.log10(np.maximum(flux_3arcsec, 1e-10)) + 20.0
        
        # 3. Calcular errores
        bkg_rms = np.std(bkg_phot_table['aperture_sum'] / annulus.area)
        flux_err = np.sqrt(np.abs(flux_3arcsec) + (apertures.area * bkg_rms**2))
        mag_err = (2.5 / np.log(10)) * (flux_err / np.maximum(flux_3arcsec, 1e-10))
        snr = np.where(flux_err > 0, flux_3arcsec / flux_err, 0.0)
        
        # 4. Estadísticas de flujo para diagnóstico
        median_flux = np.median(flux_3arcsec[flux_3arcsec > 0])
        mean_flux = np.mean(flux_3arcsec[flux_3arcsec > 0])
        
        # 5. Crear DataFrame
        results = pd.DataFrame({
            'x_pix': reference_positions[:, 0],
            'y_pix': reference_positions[:, 1],
            f'flux_3arcsec_{filter_name}': flux_3arcsec,
            f'flux_err_{filter_name}': flux_err,
            f'mag_inst_splus_{filter_name}': minstr,
            f'mag_err_{filter_name}': mag_err,
            f'snr_{filter_name}': snr,
            f'bkg_rms_{filter_name}': bkg_rms,
            f'fwhm_{filter_name}': header.get('FWHMMEAN', 1.5),
            f'pixscale_{filter_name}': pixscale,
        })
        
        # 6. Añadir coordenadas
        try:
            ra, dec = wcs.all_pix2world(reference_positions[:, 0], reference_positions[:, 1], 0)
            results['ra'] = ra
            results['dec'] = dec
        except Exception as e:
            logging.warning(f"{filter_name}: Error calculando coordenadas: {e}")
            if 'ra' in ref_catalog.columns and 'dec' in ref_catalog.columns:
                results['ra'] = ref_catalog['ra'].values
                results['dec'] = ref_catalog['dec'].values
        
        # 7. Estadísticas de calidad
        valid_mag_mask = (minstr > 10) & (minstr < 30) & np.isfinite(minstr)
        n_valid = np.sum(valid_mag_mask)
        
        if n_valid > 0:
            mean_mag = minstr[valid_mag_mask].mean()
            std_mag = minstr[valid_mag_mask].std()
            mean_snr = snr[valid_mag_mask].mean()
            
            logging.info(f"{filter_name}: {n_valid} estrellas, "
                       f"minstr = {mean_mag:.2f} ± {std_mag:.2f}, "
                       f"SNR = {mean_snr:.1f}, "
                       f"Flujo mediano = {median_flux:.1f} ADU")
            
            # Análisis de rango
            if mean_mag < 16:
                logging.warning(f"{filter_name}: ⚠️  MAGNITUDES MUY BRILLANTES - Verificar estrellas de referencia")
            elif 16 <= mean_mag <= 22:
                logging.info(f"{filter_name}: ✅ Magnitudes en rango esperado")
            else:
                logging.info(f"{filter_name}: ⚠️  Magnitudes tenues")
        
        return results
        
    except Exception as e:
        logging.error(f"Error procesando {filter_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_field_NO_AC(field_name):
    """Procesa un campo completo SIN aperture correction"""
    logging.info(f"\n{'='*60}")
    logging.info(f"PROCESANDO CAMPO: {field_name} (SIN APERTURE CORRECTION)")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    
    field_dir = field_name
    if not os.path.exists(field_dir):
        logging.error(f"Directorio no encontrado: {field_dir}")
        return False
    
    input_catalog = f'{field_name}_gaia_xp_matches.csv'
    output_catalog = f'{field_name}_gaia_xp_matches_no_ac.csv'
    
    if not os.path.exists(input_catalog):
        logging.error(f"Catálogo no encontrado: {input_catalog}")
        return False
    
    # Analizar magnitudes Gaia primero
    analyze_gaia_magnitudes(input_catalog)
    
    # Obtener posiciones de referencia
    reference_positions = None
    ref_catalog = None
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            continue
        
        reference_positions, ref_catalog = get_reference_positions_from_catalog(input_catalog, image_file)
        if len(reference_positions) > 10:
            logging.info(f"Usando {band} para posiciones de referencia ({len(reference_positions)} estrellas)")
            break
        else:
            reference_positions = None
            ref_catalog = None
    
    if reference_positions is None:
        logging.error("No se pudieron obtener posiciones válidas")
        return False
    
    # Procesar cada banda SIN aperture correction
    band_results = {}
    processed_filters = []
    
    for band in SPLUS_FILTERS:
        image_file = find_image_file(field_dir, field_name, band)
        if image_file is None:
            logging.warning(f"Imagen {field_name}_{band} no encontrada")
            continue
        
        logging.info(f"Procesando {band} SIN aperture correction...")
        results = extract_instrumental_magnitudes_NO_AC(
            image_file, reference_positions, ref_catalog, field_name, band)
        
        if len(results) > 0:
            band_results[band] = results
            processed_filters.append(band)
            logging.info(f"{band}: Completado (sin aperture correction)")
        else:
            logging.warning(f"{band}: No se pudieron extraer magnitudes")
    
    if not band_results:
        logging.error("No se procesó ninguna banda")
        return False
    
    # Combinar resultados
    combined_catalog = ref_catalog.copy()
    
    for band in processed_filters:
        if band in band_results:
            results = band_results[band]
            if len(results) == len(combined_catalog):
                for col in results.columns:
                    if col not in ['ra', 'dec']:
                        combined_catalog[col] = results[col].values
            else:
                logging.warning(f"{band}: Número de estrellas no coincide, haciendo merge por coordenadas")
                if 'ra' in results.columns and 'dec' in results.columns:
                    try:
                        merged = pd.merge(combined_catalog, results, on=['ra', 'dec'], how='left', suffixes=('', f'_{band}'))
                        combined_catalog = merged
                    except Exception as e:
                        logging.error(f"Error en merge: {e}")
    
    # Guardar resultados
    combined_catalog.to_csv(output_catalog, index=False)
    total_time = time.time() - start_time
    
    logging.info(f"RESUMEN {field_name}:")
    logging.info(f"Tiempo: {total_time:.1f}s, Filtros procesados: {len(processed_filters)}/{len(SPLUS_FILTERS)}")
    logging.info(f"Resultados guardados en: {output_catalog}")
    logging.info("✅ METODOLOGÍA SPLUS BÁSICA (SIN APERTURE CORRECTION) APLICADA")
    
    return True

def main():
    """Función principal"""
    fields_to_process = ['CenA01', 'CenA02']  # Solo prueba con 2 campos
    
    logging.info("INICIANDO PROCESAMIENTO SIN APERTURE CORRECTION")
    logging.info("Fórmula: minstr = −2.5 log10(ADUm|3'') + 20")
    logging.info("NOTA: No se aplica aperture correction")
    
    start_time = time.time()
    successful_fields = []
    
    for field in fields_to_process:
        success = process_field_NO_AC(field)
        if success:
            successful_fields.append(field)
        time.sleep(1)
    
    total_time = time.time() - start_time
    
    logging.info(f"\n{'='*80}")
    logging.info("PROCESAMIENTO COMPLETADO")
    logging.info(f"{'='*80}")
    logging.info(f"Tiempo total: {total_time/60:.1f} minutos")
    logging.info(f"Campos exitosos: {len(successful_fields)}")
    
    if successful_fields:
        logging.info("✅ Los archivos contienen magnitudes instrumentales SIN aperture correction")
        
        # Verificación final
        for field in successful_fields:
            output_file = f'{field}_gaia_xp_matches_no_ac.csv'
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                print(f"\nVERIFICACIÓN FINAL - {field}:")
                print("=" * 50)
                
                for band in SPLUS_FILTERS:
                    mag_col = f'mag_inst_splus_{band}'
                    if mag_col in df.columns:
                        mag_values = df[mag_col]
                        valid_mags = mag_values[(mag_values > 10) & (mag_values < 25)]
                        
                        if len(valid_mags) > 0:
                            print(f"{band}: {len(valid_mags)} estrellas, "
                                  f"mag: {valid_mags.mean():.2f} ± {valid_mags.std():.2f}")

if __name__ == '__main__':
    main()
