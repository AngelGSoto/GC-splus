#!/usr/bin/env python3

"""
extract_splus_gaia_xp_complete.py - Versión mejorada con correcciones para S-PLUS

Extrae estrellas de imágenes S-PLUS, las cruza con Gaia DR3, descarga espectros XP,
y crea un catálogo completo con información para calibración fotométrica.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry
from astroquery.gaia import Gaia
from scipy.spatial import KDTree
from gaiaxpy import calibrate
import warnings
warnings.filterwarnings('ignore')

# --------------------------- CONFIGURACIÓN ---------------------------
SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
IMAGE_TEMPLATE = 'CenA01_{filter}.fits'
MATCH_RADIUS_ARCSEC = 1.0
FIELD_RADIUS_DEG = 1.0
MIN_CAL_STARS = 5
OUTPUT_TABLE = 'splus_gaia_xp_matches.csv'
SPECTRA_DIR = 'gaia_spectra_cenA01'

# --------------------------- FUNCIONES ---------------------------

def extract_stars(image_file, threshold_sigma=3, max_stars=5000):
    """Extrae estrellas de una imagen S-PLUS y calcula magnitudes instrumentales corregidas."""
    try:
        with fits.open(image_file) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                print(f"No se encontraron datos en {image_file}")
                return pd.DataFrame()
        
        # Obtener parámetros importantes del header
        exptime = header.get('EXPTIME', 1.0)
        gain = header.get('GAIN', 1.0)
        saturate_level = header.get('SATURATE', np.inf)
        pixscale = header.get('PIXSCALE', 0.55)  # arcsec/pixel
        
        print(f"Parámetros para {image_file}: EXPTIME={exptime}, GAIN={gain}, SATURATE={saturate_level}")
        
        # Intentar obtener WCS
        try:
            wcs = WCS(header)
            has_wcs = True
        except:
            print(f"Advertencia: No se pudo interpretar WCS en {image_file}")
            has_wcs = False
        
        # Estimación de fondo mejorada
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, box_size=(50, 50), filter_size=(3, 3), 
                          bkg_estimator=bkg_estimator)
        data_sub = data - bkg.background
        
        # Detección de estrellas - usar FWHM de la imagen si está disponible
        fwhm_arcsec = header.get('FWHMMEAN', 1.25)  # Valor por defecto de 1.25"
        fwhm_pixels = fwhm_arcsec / pixscale
        
        std = mad_std(data_sub)
        daofind = DAOStarFinder(fwhm=fwhm_pixels, threshold=threshold_sigma*std)
        sources = daofind(data_sub)
        
        if sources is None:
            print(f"No se detectaron estrellas en {image_file}")
            return pd.DataFrame()
            
        sources = sources.to_pandas()
        
        # Calcular FWHM promedio de las estrellas detectadas
        if 'fwhm' in sources.columns and len(sources) > 10:
            avg_fwhm = np.median(sources['fwhm'])
            print(f"FWHM promedio en {image_file}: {avg_fwhm:.2f} píxeles ({avg_fwhm*pixscale:.2f} arcsec)")
        else:
            avg_fwhm = fwhm_pixels
            print(f"Usando FWHM por defecto: {avg_fwhm} píxeles")
        
        # Fotometría de apertura - usar 2.5 veces el FWHM
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        aperture_radius = avg_fwhm * 2.5
        aperture = CircularAperture(positions, r=aperture_radius)
        phot_table = aperture_photometry(data_sub, aperture)
        
        # CÁLCULO CORRECTO DE MAGNITUD INSTRUMENTAL
        # Para cálculo de zeropoints, necesitamos el flujo instrumental sin normalizar
        flux_adu = phot_table['aperture_sum']
        
        # La magnitud instrumental se define como: mag_inst = -2.5 * log10(flux_adu)
        # Esto nos dará una magnitud relativa que luego calibraremos con el zeropoint
        min_flux = 1e-10  # Evitar log(0)
        sources['mag_inst'] = -2.5 * np.log10(np.maximum(flux_adu, min_flux))
        
        # También guardamos el flujo en ADU para referencia
        sources['flux_adu'] = flux_adu
        
        # Rechazar estrellas saturadas
        max_pixel_values = []
        for _, star in sources.iterrows():
            x, y = int(round(star['xcentroid'])), int(round(star['ycentroid']))
            # Extraer región 3x3 alrededor del centroide
            region = data_sub[max(0, y-1):y+2, max(0, x-1):x+2]
            max_pixel_values.append(np.max(region) if region.size > 0 else 0)
        
        sources['max_pixel_value'] = max_pixel_values
        saturated = sources['max_pixel_value'] > saturate_level
        if np.sum(saturated) > 0:
            print(f"Rechazando {np.sum(saturated)} estrellas saturadas")
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
        sources['aperture_radius'] = aperture_radius
        sources['exptime'] = exptime
        sources['gain'] = gain
            
        return sources[['ra', 'dec', 'flux_adu', 'mag_inst', 'fwhm', 'aperture_radius']]
        
    except Exception as e:
        print(f"Error en extracción de estrellas de {image_file}: {e}")
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

def query_gaia_sources(ra_center, dec_center, radius_deg=1.0, max_retries=3):
    """Consulta fuentes Gaia DR3 con espectros XP en la región de interés."""
    for attempt in range(max_retries):
        try:
            # Limitar el número de resultados para evitar timeouts
            adql = f"""
            SELECT 
                source_id, ra, dec, 
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, 
                ruwe, parallax, pmra, pmdec
            FROM gaiadr3.gaia_source
            WHERE 
                1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg}))
                AND has_xp_continuous = 'True'
                AND phot_g_mean_mag BETWEEN 12 AND 19
                AND ruwe < 1.4
            """
            
            print(f"Ejecutando consulta ADQL en Gaia (intento {attempt+1}/{max_retries})...")
            job = Gaia.launch_job_async(adql, dump_to_file=False)
            r = job.get_results()
            print(f"Se encontraron {len(r)} fuentes Gaia con espectros XP")
            return r
        except Exception as e:
            print(f"Error en consulta Gaia (intento {attempt+1}): {e}")
            if attempt < max_retries - 1:
                # Esperar antes de reintentar
                time.sleep(10)
            else:
                return None

def create_complete_catalog(band_catalogs):
    """Crea un catálogo combinado de todas las bandas S-PLUS, conservando solo estrellas completas."""
    if not band_catalogs:
        return pd.DataFrame()
    
    # Crear un DataFrame con todas las estrellas de todas las bandas
    all_stars = pd.DataFrame()
    
    for band, cat in band_catalogs.items():
        if len(cat) == 0:
            continue
            
        # Renombrar columnas para esta banda
        band_cat = cat[['ra', 'dec', 'mag_inst', 'flux_adu']].copy()
        band_cat.rename(columns={'mag_inst': f'mag_{band}', 'flux_adu': f'flux_{band}'}, inplace=True)
        
        # Combinar con el catálogo principal
        if len(all_stars) == 0:
            all_stars = band_cat
        else:
            # Cruzar por posición para encontrar estrellas comunes
            all_points = sph_to_cart(all_stars['ra'].values, all_stars['dec'].values)
            band_points = sph_to_cart(band_cat['ra'].values, band_cat['dec'].values)
            tree = KDTree(band_points)
            distances, indices = tree.query(all_points, k=1)
            
            # Convertir distancias to arcsegundos
            angular_distances_arcsec = 2 * np.arcsin(distances / 2) * 180 / np.pi * 3600
            mask = angular_distances_arcsec < MATCH_RADIUS_ARCSEC
            
            # Añadir magnitudes y flujos de esta banda
            all_stars[f'mag_{band}'] = np.nan
            all_stars[f'flux_{band}'] = np.nan
            if np.sum(mask) > 0:
                all_stars.loc[mask, f'mag_{band}'] = band_cat.iloc[indices[mask]][f'mag_{band}'].values
                all_stars.loc[mask, f'flux_{band}'] = band_cat.iloc[indices[mask]][f'flux_{band}'].values
            
            print(f"{band}: {np.sum(mask)} estrellas coincidentes con el catálogo principal")
    
    # Filtrar estrellas que no tienen mediciones en todas las bandas
    bands_required = [f'mag_{band}' for band in SPLUS_FILTERS]
    complete_mask = all_stars[bands_required].notna().all(axis=1)
    complete_cat = all_stars[complete_mask].copy()
    
    print(f"Estrellas completas (todas las bandas): {len(complete_cat)}")
    
    return complete_cat

def download_and_convert_spectra(source_ids, output_dir):
    """Descarga espectros Gaia XP y los convierte al formato correcto."""
    os.makedirs(output_dir, exist_ok=True)
    spectra_files = []
    
    try:
        # Obtener espectros calibrados
        calibrated_spectra, sampling = calibrate(source_ids)
        
        # Convertir sampling de nm a Å (1 nm = 10 Å)
        sampling_angstrom = sampling * 10.0
        
        # Factor de conversión de flujo: de W/nm/m² a erg/s/cm²/Å
        conversion_factor = 100.0
        
        # Procesar cada espectro
        for i, source_id in enumerate(source_ids):
            # Obtener flujo para esta fuente
            flux_w_nm_m2 = calibrated_spectra['flux'][i]
            
            # Convertir flujo a erg/s/cm²/Å
            flux_erg_s_cm2_angstrom = flux_w_nm_m2 * conversion_factor
            
            # Crear nombre de archivo
            filename = os.path.join(output_dir, f"gaia_xp_spectrum_{source_id}.dat")
            
            # Guardar espectro en archivo ASCII de dos columnas
            with open(filename, 'w') as f:
                f.write(f"# Gaia XP Spectrum for source_id: {source_id}\n")
                f.write(f"# Wavelength(Angstrom) Flux(erg/s/cm2/Angstrom)\n")
                for wl_ang, flux in zip(sampling_angstrom, flux_erg_s_cm2_angstrom):
                    f.write(f"{wl_ang:.1f} {flux:.6e}\n")
            
            spectra_files.append(filename)
            print(f"Espectro guardado: {filename}")
        
        return spectra_files
        
    except Exception as e:
        print(f"Error al descargar y convertir espectros: {e}")
        return []

# --------------------------- PROCESAMIENTO PRINCIPAL ---------------------------

def main():
    # Extraer estrellas en cada filtro
    print("Extrayendo estrellas en todas las bandas S-PLUS...")
    band_catalogs = {}
    
    for band in SPLUS_FILTERS:
        image_file = IMAGE_TEMPLATE.format(filter=band)
        if not os.path.exists(image_file):
            print(f"Imagen {image_file} no encontrada. Saltando...")
            continue
            
        cat = extract_stars(image_file)
        if len(cat) == 0:
            print(f"{band}: No se extrajeron estrellas.")
            continue
            
        band_catalogs[band] = cat
        print(f"{band}: {len(cat)} estrellas extraídas.")

    if not band_catalogs:
        print("No se pudieron extraer estrellas de ninguna banda.")
        return

    # Crear catálogo combinado completo
    print("Combinando catálogos de todas las bandas...")
    complete_cat = create_complete_catalog(band_catalogs)
    
    if len(complete_cat) == 0:
        print("No se pudo crear catálogo completo.")
        return
        
    # Calcular centro del campo
    valid_coords = ~(np.isnan(complete_cat['ra']) | np.isnan(complete_cat['dec']))
    if not any(valid_coords):
        print("No hay coordenadas válidas en el catálogo combinado.")
        return
        
    ra_c = float(np.nanmedian(complete_cat.loc[valid_coords, 'ra']))
    dec_c = float(np.nanmedian(complete_cat.loc[valid_coords, 'dec']))
    print(f"Centro del campo: RA={ra_c:.6f}, Dec={dec_c:.6f}")

    # Consulta Gaia DR3 con reintentos
    gaia_tbl = query_gaia_sources(ra_c, dec_c, radius_deg=FIELD_RADIUS_DEG, max_retries=5)
    if gaia_tbl is None or len(gaia_tbl) == 0:
        print("No se obtuvo catálogo Gaia en la región después de varios intentos.")
        return
        
    gaia_df = gaia_tbl.to_pandas()
    gaia_df.columns = [col.lower() for col in gaia_df.columns]

    # Cruzar catálogos S-PLUS y Gaia por posición
    matches = crossmatch_kdtree(complete_cat, gaia_df, match_radius_arcsec=MATCH_RADIUS_ARCSEC)
    print(f"Coincidencias S-PLUS–Gaia dentro de {MATCH_RADIUS_ARCSEC}\": {len(matches)}")

    if len(matches) == 0:
        print("No hay coincidencias útiles.")
        return
        
    if len(matches) < MIN_CAL_STARS:
        print(f"Advertencia: Solo {len(matches)} coincidencias (mínimo recomendado: {MIN_CAL_STARS})")

    # Descargar y convertir espectros
    source_ids = matches['source_id'].tolist()
    
    # Limitar el número de estrellas para evitar sobrecargar el servidor
    max_stars = min(50, len(source_ids))
    source_ids_limited = source_ids[:max_stars]
    
    print(f"Descargando espectros para {len(source_ids_limited)} estrellas...")
    spectra_files = download_and_convert_spectra(source_ids_limited, SPECTRA_DIR)
    
    # Añadir nombres de archivos de espectros al catálogo
    matches['spectrum_file'] = ""
    for i, source_id in enumerate(source_ids_limited):
        spectrum_file = os.path.join(SPECTRA_DIR, f"gaia_xp_spectrum_{source_id}.dat")
        matches.loc[matches['source_id'] == source_id, 'spectrum_file'] = spectrum_file
    
    # Preparar tabla de salida
    output_cols = ['source_id', 'ra', 'dec', 'spectrum_file'] + \
                 [f'mag_{band}' for band in SPLUS_FILTERS] + \
                 [f'flux_{band}' for band in SPLUS_FILTERS] + \
                 ['distance_arcsec',
                  'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag',
                  'gaia_ruwe', 'gaia_parallax', 'gaia_pmra', 'gaia_pmdec']
    
    # Filtrar columnas existentes
    available_cols = [col for col in output_cols if col in matches.columns]
    matches = matches[available_cols]
    
    # Guardar resultados
    matches.to_csv(OUTPUT_TABLE, index=False)
    print(f"Tabla final guardada en: {OUTPUT_TABLE}")
    print(f"Coincidencias encontradas: {len(matches)}")
    print(f"Espectros descargados: {len(spectra_files)}")
    print(f"Directorio de espectros: {SPECTRA_DIR}")
    
    # Información para el cálculo de zeropoints
    print("\n" + "="*50)
    print("INFORMACIÓN PARA CÁLCULO DE ZEROPOINTS:")
    print("="*50)
    print("1. Las magnitudes instrumentales están en la escala: mag_inst = -2.5 * log10(flux_adu)")
    print("2. Los flujos en ADU están en las columnas flux_[band]")
    print("3. Para calcular el zeropoint para cada filtro:")
    print("   a) Calcular magnitudes sintéticas desde los espectros Gaia XP")
    print("   b) Ajustar: mag_synthetic = mag_inst + zeropoint")
    print("   c) El zeropoint es la diferencia mediana: zeropoint = median(mag_synthetic - mag_inst)")
    print("4. Las magnitudes sintéticas se pueden calcular usando:")
    print("   - Las curvas de respuesta de los filtros S-PLUS")
    print("   - Los espectros descargados en", SPECTRA_DIR)

if __name__ == '__main__':
    main()
