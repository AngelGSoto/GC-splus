#!/usr/bin/env python3

"""
extract_splus_gaia_xp_complete.py - Versión corregida para manejo de errores de Gaia
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from astroquery.gaia import Gaia
from scipy.spatial import KDTree
from gaiaxpy import calibrate
import warnings
warnings.filterwarnings('ignore')

# --------------------------- CONFIGURACIÓN ---------------------------
SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
FIELD_RADIUS_DEG = 0.3  # Radio reducido para consultas más ligeras
MIN_CAL_STARS = 10
BASE_DATA_DIR = "."

# --------------------------- FUNCIONES ---------------------------

def extract_stars(image_path, threshold_sigma=3, max_stars=5000):
    """Extrae estrellas de una imagen S-PLUS y calcula magnitudes instrumentales."""
    try:
        # Astropy puede leer archivos .fits.fz directamente
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                print(f"No se encontraron datos en {image_path}")
                return pd.DataFrame()
        
        # Obtener parámetros importantes del header
        pixscale = header.get('PIXSCALE', 0.55)
        saturate_level = header.get('SATURATE', np.inf)
        
        print(f"Procesando {os.path.basename(image_path)} con PIXSCALE={pixscale}")
        
        # Intentar obtener WCS
        try:
            wcs = WCS(header)
            has_wcs = True
        except:
            print(f"Advertencia: No se pudo interpretar WCS en {image_path}")
            has_wcs = False
        
        # NO RESTAR FONDO - Las imágenes SPLUS ya están corregidas
        data_sub = data
        
        # Detección de estrellas
        fwhm_arcsec = header.get('FWHMMEAN', 1.25)
        fwhm_pixels = fwhm_arcsec / pixscale
        
        std = mad_std(data_sub)
        daofind = DAOStarFinder(fwhm=fwhm_pixels, threshold=threshold_sigma*std)
        sources = daofind(data_sub)
        
        if sources is None:
            print(f"No se detectaron estrellas en {image_path}")
            return pd.DataFrame()
            
        sources = sources.to_pandas()
        
        # Calcular FWHM promedio
        if 'fwhm' in sources.columns and len(sources) > 10:
            avg_fwhm = np.median(sources['fwhm'])
            print(f"FWHM promedio: {avg_fwhm:.2f} píxeles ({avg_fwhm*pixscale:.2f} arcsec)")
        else:
            avg_fwhm = fwhm_pixels
            print(f"Usando FWHM por defecto: {avg_fwhm} píxeles")
        
        # Fotometría de apertura - usar apertura de 3 arcseg (diámetro)
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        aperture_radius_pixels = (3.0 / 2.0) / pixscale
        aperture = CircularAperture(positions, r=aperture_radius_pixels)
        phot_table = aperture_photometry(data_sub, aperture)
        
        # CÁLCULO DE MAGNITUD INSTRUMENTAL
        flux_adu = phot_table['aperture_sum']
        min_flux = 1e-10
        sources['mag_inst'] = -2.5 * np.log10(np.maximum(flux_adu, min_flux))
        sources['flux_adu'] = flux_adu
        
        # Rechazar estrellas saturadas
        max_pixel_values = []
        for _, star in sources.iterrows():
            x, y = int(round(star['xcentroid'])), int(round(star['ycentroid']))
            y_min, y_max = max(0, y-2), min(data_sub.shape[0], y+3)
            x_min, x_max = max(0, x-2), min(data_sub.shape[1], x+3)
            region = data_sub[y_min:y_max, x_min:x_max]
            max_pixel_values.append(np.max(region) if region.size > 0 else 0)
        
        sources['max_pixel_value'] = max_pixel_values
        saturated = sources['max_pixel_value'] > (0.7 * saturate_level)
        if np.sum(saturated) > 0:
            print(f"Rechazando {np.sum(saturated)} estrellas saturadas o casi saturadas")
            sources = sources[~saturated]
        
        # Limitar número de estrellas por brillo
        if len(sources) > max_stars:
            sources = sources.nlargest(max_stars, 'flux_adu')
            print(f"Se seleccionaron las {max_stars} estrellas más brillantes")
        
        # Coordenadas astronómicas si hay WCS
        if has_wcs:
            coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources['ra'] = coords.ra.deg
            sources['dec'] = coords.dec.deg
        else:
            sources['ra'] = np.nan
            sources['dec'] = np.nan
            
        # Añadir información adicional
        sources['fwhm'] = avg_fwhm
        sources['aperture_radius'] = aperture_radius_pixels
            
        return sources[['ra', 'dec', 'flux_adu', 'mag_inst', 'fwhm', 'aperture_radius']]
        
    except Exception as e:
        print(f"Error en extracción de estrellas de {image_path}: {e}")
        return pd.DataFrame()

def sph_to_cart(ra_arr, dec_arr):
    """Convierte coordenadas esféricas (RA, Dec) a coordenadas cartesianas."""
    phi = np.radians(ra_arr)
    theta = np.radians(90.0 - dec_arr)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack([x, y, z]).T

def crossmatch_kdtree(stars, gaia_df, match_radius_arcsec=1.0):
    """Realiza un cruce espacial entre catálogos usando KDTree."""
    if len(stars) == 0 or len(gaia_df) == 0:
        return pd.DataFrame()
        
    stars_ra = stars['ra'].values
    stars_dec = stars['dec'].values
    gaia_ra = gaia_df['ra'].values
    gaia_dec = gaia_df['dec'].values
    
    radius_deg = match_radius_arcsec / 3600.0
    gaia_points = sph_to_cart(gaia_ra, gaia_dec)
    tree = KDTree(gaia_points)
    stars_points = sph_to_cart(stars_ra, stars_dec)
    
    distances, indices = tree.query(stars_points, k=1)
    angular_distances_deg = 2 * np.arcsin(distances / 2) * 180 / np.pi
    mask = angular_distances_deg < radius_deg
    
    matched = stars.iloc[mask].copy()
    matched['gaia_idx'] = indices[mask]
    matched['distance_arcsec'] = angular_distances_deg[mask] * 3600
    
    # Añadir datos de Gaia
    for col in gaia_df.columns:
        if col == 'source_id':
            matched['source_id'] = gaia_df.iloc[indices[mask]][col].values
        elif col not in matched.columns:
            matched[f'gaia_{col}'] = gaia_df.iloc[indices[mask]][col].values
            
    return matched

def query_gaia_sources(ra_center, dec_center, radius_deg=0.3, max_retries=5):
    """Consulta fuentes Gaia DR3 con espectros XP en la región de interés."""
    for attempt in range(max_retries):
        try:
            # Consulta optimizada y más simple
            adql = f"""
            SELECT 
                source_id, ra, dec, 
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, 
                ruwe, parallax, pmra, pmdec
            FROM gaiadr3.gaia_source
            WHERE 
                1=CONTAINS(POINT('ICRS', ra, dec), 
                CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg}))
                AND has_xp_continuous = 'True'
                AND phot_g_mean_mag BETWEEN 14 AND 18
                AND ruwe < 1.4
            """
            
            print(f"Ejecutando consulta ADQL en Gaia (intento {attempt+1}/{max_retries})...")
            
            # Usar launch_job sin el parámetro timeout
            job = Gaia.launch_job(adql, dump_to_file=False, verbose=False)
            r = job.get_results()
            
            print(f"Se encontraron {len(r)} fuentes Gaia con espectros XP")
            return r
            
        except Exception as e:
            print(f"Error en consulta Gaia (intento {attempt+1}): {str(e)[:100]}...")
            
            # Esperar progresivamente más entre intentos
            wait_time = 30 * (attempt + 1)
            print(f"Esperando {wait_time} segundos antes de reintentar...")
            time.sleep(wait_time)
            
    print("No se pudo completar la consulta después de varios intentos")
    return None

def create_complete_catalog(band_catalogs):
    """Crea un catálogo combinado de todas las bandas S-PLUS."""
    if not band_catalogs:
        return pd.DataFrame()
    
    all_stars = pd.DataFrame()
    
    for band, cat in band_catalogs.items():
        if len(cat) == 0:
            continue
            
        band_cat = cat[['ra', 'dec', 'mag_inst', 'flux_adu']].copy()
        band_cat.rename(columns={'mag_inst': f'mag_{band}', 'flux_adu': f'flux_{band}'}, inplace=True)
        
        if len(all_stars) == 0:
            all_stars = band_cat
        else:
            all_points = sph_to_cart(all_stars['ra'].values, all_stars['dec'].values)
            band_points = sph_to_cart(band_cat['ra'].values, band_cat['dec'].values)
            tree = KDTree(band_points)
            distances, indices = tree.query(all_points, k=1)
            
            angular_distances_arcsec = 2 * np.arcsin(distances / 2) * 180 / np.pi * 3600
            mask = angular_distances_arcsec < 1.0
            
            all_stars[f'mag_{band}'] = np.nan
            all_stars[f'flux_{band}'] = np.nan
            if np.sum(mask) > 0:
                all_stars.loc[mask, f'mag_{band}'] = band_cat.iloc[indices[mask]][f'mag_{band}'].values
                all_stars.loc[mask, f'flux_{band}'] = band_cat.iloc[indices[mask]][f'flux_{band}'].values
            
            print(f"{band}: {np.sum(mask)} estrellas coincidentes")
    
    bands_required = [f'mag_{band}' for band in SPLUS_FILTERS]
    complete_mask = all_stars[bands_required].notna().all(axis=1)
    complete_cat = all_stars[complete_mask].copy()
    
    print(f"Estrellas completas (todas las bandas): {len(complete_cat)}")
    return complete_cat

def download_and_convert_spectra(source_ids, output_dir, max_spectra=100):
    """Descarga espectros Gaia XP."""
    os.makedirs(output_dir, exist_ok=True)
    spectra_files = []
    
    source_ids = source_ids[:max_spectra]
    
    try:
        print(f"Descargando {len(source_ids)} espectros...")
        calibrated_spectra, sampling = calibrate(source_ids)
        
        sampling_angstrom = sampling * 10.0
        conversion_factor = 100.0
        
        for i, source_id in enumerate(source_ids):
            flux_w_nm_m2 = calibrated_spectra['flux'][i]
            flux_erg_s_cm2_angstrom = flux_w_nm_m2 * conversion_factor
            
            filename = os.path.join(output_dir, f"gaia_xp_spectrum_{source_id}.dat")
            
            with open(filename, 'w') as f:
                f.write(f"# Gaia XP Spectrum for source_id: {source_id}\n")
                f.write(f"# Wavelength(Angstrom) Flux(erg/s/cm2/Angstrom)\n")
                for wl_ang, flux in zip(sampling_angstrom, flux_erg_s_cm2_angstrom):
                    f.write(f"{wl_ang:.1f} {flux:.6e}\n")
            
            spectra_files.append(filename)
            
            if (i + 1) % 10 == 0:
                print(f"Descargados {i+1}/{len(source_ids)} espectros")
                time.sleep(2)
        
        print(f"Espectros guardados: {len(spectra_files)}")
        return spectra_files
        
    except Exception as e:
        print(f"Error al descargar espectros: {e}")
        return []

def process_field(field_name, base_dir=BASE_DATA_DIR):
    """Procesa un campo completo de SPLUS."""
    print(f"\n{'='*60}")
    print(f"PROCESANDO CAMPO: {field_name}")
    print(f"{'='*60}")
    
    field_dir = os.path.join(base_dir, field_name)
    if not os.path.exists(field_dir):
        print(f"Directorio del campo no encontrado: {field_dir}")
        return False
    
    output_table = f'{field_name}_gaia_xp_matches.csv'
    spectra_dir = f'gaia_spectra_{field_name}'
    
    # Extraer estrellas en cada filtro
    print("Extrayendo estrellas en todas las bandas S-PLUS...")
    band_catalogs = {}
    
    for band in SPLUS_FILTERS:
        image_file = os.path.join(field_dir, f"{field_name}_{band}.fits.fz")
        if not os.path.exists(image_file):
            image_file = os.path.join(field_dir, f"{field_name}_{band}.fits")
            if not os.path.exists(image_file):
                print(f"Imagen {field_name}_{band} no encontrada. Saltando...")
                continue
        
        cat = extract_stars(image_file)
        if len(cat) == 0:
            print(f"{band}: No se extrajeron estrellas.")
            continue
            
        band_catalogs[band] = cat
        print(f"{band}: {len(cat)} estrellas extraídas.")

    if not band_catalogs:
        print("No se pudieron extraer estrellas de ninguna banda.")
        return False

    # Crear catálogo combinado completo
    print("Combinando catálogos de todas las bandas...")
    complete_cat = create_complete_catalog(band_catalogs)
    
    if len(complete_cat) == 0:
        print("No se pudo crear catálogo completo.")
        return False
        
    # Calcular centro del campo
    valid_coords = ~(np.isnan(complete_cat['ra']) | np.isnan(complete_cat['dec']))
    if not any(valid_coords):
        print("No hay coordenadas válidas en el catálogo combinado.")
        return False
        
    ra_c = float(np.nanmedian(complete_cat.loc[valid_coords, 'ra']))
    dec_c = float(np.nanmedian(complete_cat.loc[valid_coords, 'dec']))
    print(f"Centro del campo: RA={ra_c:.6f}, Dec={dec_c:.6f}")

    # Consulta Gaia DR3 con reintentos
    gaia_tbl = query_gaia_sources(ra_c, dec_c, radius_deg=FIELD_RADIUS_DEG)
    if gaia_tbl is None or len(gaia_tbl) == 0:
        print("No se obtuvo catálogo Gaia en la región después de varios intentos.")
        return False
        
    gaia_df = gaia_tbl.to_pandas()
    gaia_df.columns = [col.lower() for col in gaia_df.columns]

    # Cruzar catálogos S-PLUS y Gaia por posición
    matches = crossmatch_kdtree(complete_cat, gaia_df, match_radius_arcsec=1.0)
    print(f"Coincidencias S-PLUS–Gaia dentro de 1\": {len(matches)}")

    if len(matches) == 0:
        print("No hay coincidencias útiles.")
        return False
        
    if len(matches) < MIN_CAL_STARS:
        print(f"Advertencia: Solo {len(matches)} coincidencias (mínimo recomendado: {MIN_CAL_STARS})")

    # Descargar y convertir espectros
    source_ids = matches['source_id'].tolist()
    
    print(f"Descargando espectros para {len(source_ids)} estrellas...")
    spectra_files = download_and_convert_spectra(source_ids, spectra_dir, max_spectra=100)
    
    # Añadir nombres de archivos de espectros al catálogo
    matches['spectrum_file'] = ""
    for i, source_id in enumerate(source_ids[:100]):
        spectrum_file = os.path.join(spectra_dir, f"gaia_xp_spectrum_{source_id}.dat")
        matches.loc[matches['source_id'] == source_id, 'spectrum_file'] = spectrum_file
    
    # Preparar tabla de salida
    output_cols = ['source_id', 'ra', 'dec', 'spectrum_file'] + \
                 [f'mag_{band}' for band in SPLUS_FILTERS] + \
                 [f'flux_{band}' for band in SPLUS_FILTERS] + \
                 ['distance_arcsec',
                  'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag',
                  'gaia_ruwe', 'gaia_parallax', 'gaia_pmra', 'gaia_pmdec']
    
    available_cols = [col for col in output_cols if col in matches.columns]
    matches = matches[available_cols]
    
    # Guardar resultados
    matches.to_csv(output_table, index=False)
    print(f"Tabla final guardada en: {output_table}")
    print(f"Coincidencias encontradas: {len(matches)}")
    print(f"Espectros descargados: {len(spectra_files)}")
    print(f"Directorio de espectros: {spectra_dir}")
    
    return True

# --------------------------- PROCESAMIENTO PRINCIPAL ---------------------------

def main():
    # Campos a procesar
    fields = [
        'CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 
        'CenA07', 'CenA08', 'CenA09', 'CenA10', 'CenA11', 'CenA12',
        'CenA13', 'CenA14', 'CenA15', 'CenA16', 'CenA17', 'CenA18',
        'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24'
    ]
    
    # Procesar cada campo
    for field in fields:
        success = process_field(field)
        if not success:
            print(f"Error procesando campo {field}. Continuando con el siguiente...")
        
        # Pausa entre campos para no sobrecargar el sistema
        time.sleep(30)
    
    print("\nProcesamiento de todos los campos completado.")
    
if __name__ == '__main__':
    main()
