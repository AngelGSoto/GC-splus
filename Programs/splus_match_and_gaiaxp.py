#!/usr/bin/env python3
"""
splus_gaiaxp_calibration.py - Calibración fotométrica de campos S-PLUS usando Gaia DR3 y GaiaXPy
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, mad_std, sigma_clip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry
from astroquery.gaia import Gaia
from gaiaxpy import generate, PhotometricSystem
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm

# ---------------------------
#  CONFIGURACIÓN
# ---------------------------
SPLUS_FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
IMAGE_TEMPLATE = 'CenA01_{filter}.fits'
FIELD_RADIUS_DEG = 1.0
MATCH_RAD_ARCSEC = 1.0
MIN_CAL_STARS = 15
OUTPUT_ZP = 'splus_zps_gaiaxpy.csv'

# Mapeo de columnas de JPLUS a S-PLUS
JPLUS_TO_SPLUS = {
    'J0378_mag': 'F378_mag',
    'J0395_mag': 'F395_mag',
    'J0410_mag': 'F410_mag',
    'J0430_mag': 'F430_mag',
    'J0515_mag': 'F515_mag',
    'J0660_mag': 'F660_mag',
    'J0861_mag': 'F861_mag'
}

# ---------------------------
#  FUNCIONES DE EXTRACCIÓN FOTOMÉTRICA
# ---------------------------
def get_image_data_and_header(fits_file):
    with fits.open(fits_file) as hdul:
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                data = hdu.data.astype(float)
                header = hdu.header
                return data, header
    raise ValueError(f"No se encontró ninguna imagen válida en {fits_file}")

def extract_stars(image_file, threshold_sigma=3, fwhm=3.0, max_stars=5000):
    print(f"Extrayendo estrellas de {image_file}...")
    try:
        data, header = get_image_data_and_header(image_file)
        
        # Intentar obtener WCS del header
        try:
            wcs = WCS(header)
            has_wcs = True
        except:
            print("Advertencia: No se pudo crear WCS desde el header.")
            has_wcs = False
            ra_ref = header.get('CRVAL1', 200.0)
            dec_ref = header.get('CRVAL2', -45.0)
            pixel_scale = header.get('CDELT2', 0.55/3600)
        
        # Estimación de fondo
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, box_size=(50, 50), filter_size=(3, 3),
                          bkg_estimator=bkg_estimator)
        data_sub = data - bkg.background
        
        # Detección de estrellas
        std = mad_std(data_sub)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std,
                               sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0)
        sources = daofind(data_sub)
        
        if sources is None:
            print("No se detectaron estrellas")
            return pd.DataFrame()
            
        sources = sources.to_pandas()
        
        # Limitar el número de estrellas si es necesario
        if len(sources) > max_stars:
            sources = sources.sample(max_stars)
        
        # Fotometría de apertura
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        aperture = CircularAperture(positions, r=fwhm*2.0)
        phot_table = aperture_photometry(data_sub, aperture)
        sources['flux_aperture'] = phot_table['aperture_sum']
        
        # Magnitud instrumental
        sources['mag_inst'] = -2.5 * np.log10(sources['flux_aperture'] + 1e-10) + 20.0
        
        # Coordenadas celestiales
        if has_wcs:
            coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources['ra'] = coords.ra.deg
            sources['dec'] = coords.dec.deg
        else:
            # Coordenadas aproximadas si no hay WCS
            height, width = data.shape
            center_x, center_y = width/2, height/2
            sources['ra'] = ra_ref + (sources['xcentroid'] - center_x) * pixel_scale
            sources['dec'] = dec_ref + (sources['ycentroid'] - center_y) * pixel_scale
        
        return sources[['ra', 'dec', 'flux_aperture', 'mag_inst']].dropna()
        
    except Exception as e:
        print(f"Error en extracción de estrellas: {e}")
        return pd.DataFrame()

# ---------------------------
#  FUNCIONES GAIA
# ---------------------------
def query_gaia_adql(ra_center, dec_center, radius_deg=1.0):
    """Consulta Gaia DR3 usando ADQL"""
    radius_deg = float(radius_deg)
    adql = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
           phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error,
           ruwe, parallax, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg}))
      AND has_xp_continuous = 'True'
      AND phot_g_mean_mag BETWEEN 12 AND 20
      AND phot_g_mean_flux_over_error > 20
      AND ruwe < 1.4
    """
    print("Ejecutando ADQL en Gaia (puede tardar)...")
    try:
        job = Gaia.launch_job_async(adql, dump_to_file=False)
        r = job.get_results()
        return r
    except Exception as e:
        print(f"Error en consulta Gaia: {e}")
        return None

def sph_to_cart(ra_arr, dec_arr):
    """Conversión de coordenadas esféricas a cartesianas"""
    phi = np.radians(ra_arr)
    theta = np.radians(90.0 - dec_arr)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack([x, y, z]).T

def crossmatch_kdtree(stars, gaia_df, match_radius_arcsec=1.0):
    """Crossmatch usando KDTree"""
    if len(stars) == 0 or len(gaia_df) == 0:
        return pd.DataFrame()
    
    # Convertir a arrays de numpy
    stars_ra = stars['ra'].values
    stars_dec = stars['dec'].values
    gaia_ra = gaia_df['ra'].values
    gaia_dec = gaia_df['dec'].values
    
    # Convertir radio de arcosegundos a grados
    radius_deg = match_radius_arcsec / 3600.0
    
    # Crear KDTree para las coordenadas de Gaia
    gaia_points = sph_to_cart(gaia_ra, gaia_dec)
    tree = KDTree(gaia_points)
    
    # Convertir coordenadas de estrellas a 3D
    stars_points = sph_to_cart(stars_ra, stars_dec)
    
    # Buscar vecinos más cercanos
    distances, indices = tree.query(stars_points, k=1)
    
    # Convertir distancias a grados
    angular_distances_deg = 2 * np.arcsin(distances / 2) * 180 / np.pi
    angular_distances_arcsec = angular_distances_deg * 3600
    
    # Aplicar el umbral de distancia
    mask = angular_distances_arcsec < match_radius_arcsec
    
    if not np.any(mask):
        return pd.DataFrame()
    
    # Crear DataFrame con los resultados
    matched = stars.iloc[mask].copy()
    matched['gaia_idx'] = indices[mask]
    matched['distance_arcsec'] = angular_distances_arcsec[mask]
    
    # Añadir columnas de Gaia
    for col in gaia_df.columns:
        if col not in matched.columns:
            matched[f'gaia_{col}'] = gaia_df.iloc[indices[mask]][col].values
    
    return matched

# ---------------------------
#  FUNCIÓN PRINCIPAL
# ---------------------------
def main():
    zpres = []
    
    # Consulta Gaia una vez al principio
    first_filter = SPLUS_FILTERS[0]
    first_image = IMAGE_TEMPLATE.format(filter=first_filter)
    
    if not os.path.exists(first_image):
        print(f"Imagen {first_image} no encontrada.")
        return
    
    # Extraer estrellas para determinar el centro del campo
    stars = extract_stars(first_image, max_stars=1000)
    if len(stars) == 0:
        print("No se pudieron extraer estrellas.")
        return
    
    ra_c = float(np.median(stars['ra']))
    dec_c = float(np.median(stars['dec']))
    
    print(f"Centro del campo: RA={ra_c:.6f}, Dec={dec_c:.6f}")
    
    # Consultar Gaia alrededor del centro del campo
    gaia_tbl = query_gaia_adql(ra_c, dec_c, radius_deg=FIELD_RADIUS_DEG)
    if gaia_tbl is None or len(gaia_tbl) == 0:
        print("No se obtuvo catálogo Gaia en la región.")
        return
    
    # Convertir a DataFrame y normalizar nombres de columnas
    gaia_df = gaia_tbl.to_pandas()
    gaia_df.columns = [col.lower() for col in gaia_df.columns]
    print(f"Fuentes Gaia recuperadas: {len(gaia_df)}")

    for filt in SPLUS_FILTERS:
        img = IMAGE_TEMPLATE.format(filter=filt)
        if not os.path.exists(img):
            print(f"Imagen {img} no encontrada — salto {filt}")
            continue

        print(f"\n--- Procesando {filt} ---")
        
        # Extraer estrellas
        stars = extract_stars(img, max_stars=5000)
        if stars is None or len(stars) == 0:
            print("No se detectaron estrellas válidas.")
            continue
        print(f"Estrellas detectadas: {len(stars)}")

        # Hacer crossmatch
        matched = crossmatch_kdtree(stars, gaia_df, match_radius_arcsec=MATCH_RAD_ARCSEC)
        print(f"Coincidencias dentro de {MATCH_RAD_ARCSEC}\": {len(matched)}")

        if len(matched) < MIN_CAL_STARS:
            print(f"Pocas coincidencias ({len(matched)}) < MIN_CAL_STARS ({MIN_CAL_STARS}). Saltando banda.")
            continue

        # Limitar el número de fuentes para GaiaXPy (máximo 1000)
        if len(matched) > 1000:
            matched = matched.sample(1000)
            
        # Pedir a GaiaXPy la fotometría sintética
        source_id_list = matched['gaia_source_id'].astype(str).tolist()
        print(f"Solicitando fotometría sintética para {len(source_id_list)} fuentes...")

        try:
            # Usar el sistema JPLUS de GaiaXPy
            synthetic_df = generate(source_id_list, 
                                  photometric_system='JPLUS', 
                                  save_file=False)
            
            # Renombrar columnas de JPLUS a S-PLUS
            synthetic_df = synthetic_df.rename(columns=JPLUS_TO_SPLUS)
            
            print("Fotometría sintética recibida. Columnas:", synthetic_df.columns.tolist())

        except Exception as e:
            print(f"Error en generate(): {e}")
            continue

        # Combinar datos
        synthetic_df['source_id'] = synthetic_df['source_id'].astype(str)
        matched['gaia_source_id'] = matched['gaia_source_id'].astype(str)
        merged = matched.merge(synthetic_df, left_on='gaia_source_id', right_on='source_id', how='inner')
        print(f"Matches con fotometría sintética: {len(merged)}")

        if len(merged) == 0:
            continue

        # Calcular ZP para el filtro actual
        mag_col = f'{filt}_mag'
        if mag_col not in merged.columns:
            print(f"No hay columna {mag_col} en synthetic_df")
            continue

        m_syn = merged[mag_col].astype(float)
        m_inst = merged['mag_inst'].astype(float)
        zp_star = m_inst - m_syn

        # Sigma-clip
        clip = sigma_clip(zp_star, sigma=3.0, maxiters=5)
        good = ~clip.mask
        zps_good = zp_star[good]
        if len(zps_good) < MIN_CAL_STARS:
            continue

        zp_med = np.median(zps_good)
        mad = np.median(np.abs(zps_good - zp_med))
        zp_std = 1.4826 * mad
        zp_err = zp_std / np.sqrt(len(zps_good))

        zpres.append({'filter': filt, 'ZP': float(zp_med), 'ZP_err': float(zp_err),
                      'N_cal': int(len(zps_good)), 'ra_field': ra_c, 'dec_field': dec_c})
        
        # Guardar datos para inspección
        merged.to_csv(f"matched_{filt}_with_synthetic.csv", index=False)

    # Guardar resultados
    if len(zpres) > 0:
        zp_df = pd.DataFrame(zpres)
        zp_df.to_csv(OUTPUT_ZP, index=False)
        print(f"Zeropoints guardados en: {OUTPUT_ZP}")
        
        # Generar reporte
        print("\n=== RESULTADOS DE CALIBRACIÓN ===")
        for zp in zpres:
            print(f"{zp['filter']}: ZP = {zp['ZP']:.3f} ± {zp['ZP_err']:.3f} (N={zp['N_cal']})")
    else:
        print("No se obtuvieron zeropoints.")

if __name__ == '__main__':
    main()
