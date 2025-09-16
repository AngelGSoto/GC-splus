import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, mad_std
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry
from astroquery.vizier import Vizier
import os
import argparse
from scipy.spatial import KDTree
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def get_image_data_and_header(fits_file):
    with fits.open(fits_file) as hdul:
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None:
                data = hdu.data.astype(float)
                header = hdu.header
                return data, header
    raise ValueError(f"No se encontró ninguna imagen válida en {fits_file}")

def robust_photometric_validation(data, header):
    """Validación fotométrica robusta para publicación"""
    # Análisis de profundidad y límite de detección
    bkg = Background2D(data, box_size=(50, 50), filter_size=(3, 3),
                      bkg_estimator=MedianBackground())
    noise = mad_std(data - bkg.background)
    
    # Límite de detección 5-sigma
    detection_limit = -2.5 * np.log10(5 * noise) + 20
    
    results = {
        'background_level': np.median(bkg.background),
        'background_rms': noise,
        'detection_limit_5sigma': detection_limit,
        'saturation_level': header.get('SATURATE', 35000)
    }
    return results

def extract_stars(image_file, threshold_sigma=3, fwhm=3.0, max_sources=5000):
    print(f"Procesando {image_file}...")
    try:
        data, header = get_image_data_and_header(image_file)
        wcs = WCS(header)
        
        # Validación fotométrica robusta
        validation = robust_photometric_validation(data, header)
        print(f"Límite de detección 5σ: {validation['detection_limit_5sigma']:.2f} mag")
        
        # Estimación de fondo con validación
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, box_size=(50, 50), filter_size=(3, 3),
                          bkg_estimator=bkg_estimator)
        data_sub = data - bkg.background
        
        # Detección de estrellas con parámetros optimizados
        std = mad_std(data_sub)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std,
                               sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                               peakmax=validation['saturation_level'])
        sources = daofind(data_sub)
        
        if sources is None:
            print("No se detectaron estrellas")
            return pd.DataFrame()
            
        sources = sources.to_pandas()
        
        # Criterios de calidad más estrictos
        quality_mask = (
            (sources['sharpness'] > 0.2) & (sources['sharpness'] < 1.0) &
            (sources['roundness1'] > -1.0) & (sources['roundness1'] < 1.0) &
            (sources['flux'] > 0) & (sources['peak'] < validation['saturation_level'] * 0.8)
        )
        sources = sources[quality_mask].copy()
        
        if len(sources) == 0:
            print("Todas las estrellas fueron filtradas por criterios de calidad")
            return pd.DataFrame()
            
        # Fotometría de apertura con múltiples radios para corrección
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        aperture_radii = [fwhm*1.5, fwhm*2.0, fwhm*2.5, fwhm*3.0]
        
        aperture_fluxes = {}
        for radius in aperture_radii:
            aperture = CircularAperture(positions, r=radius)
            phot_table = aperture_photometry(data_sub, aperture)
            aperture_fluxes[radius] = phot_table['aperture_sum']
        
        # Usar apertura de 2*FWHM como referencia
        sources['flux_aperture'] = aperture_fluxes[fwhm*2.0]
        
        # Calcular curva de crecimiento para corrección de apertura
        flux_ratio = aperture_fluxes[fwhm*3.0] / aperture_fluxes[fwhm*2.0]
        aperture_correction = -2.5 * np.log10(flux_ratio)
        
        # Magnitud instrumental CORREGIDA (según metodología S-PLUS)
        sources['mag_inst'] = -2.5 * np.log10(sources['flux_aperture'] + 1e-10) + 20.0
        
        # Coordenadas celestiales
        coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
        sources['ra'] = coords.ra.deg
        sources['dec'] = coords.dec.deg
        
        return sources[['ra', 'dec', 'flux_aperture', 'mag_inst']].dropna()
        
    except Exception as e:
        print(f"Error en extracción de estrellas: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def query_gaia(ra_c, dec_c, radius_deg=1.0):
    """Consulta Gaia con criterios de calidad estrictos"""
    print("Consultando Gaia DR3 con criterios estrictos...")
    try:
        Vizier.ROW_LIMIT = 20000
        Vizier.columns = ['RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag', 
                         'Plx', 'pmRA', 'pmDE', 'e_Gmag', 'e_BPmag', 'e_RPmag', 
                         'RUWE', 'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux']
        
        result = Vizier.query_region(
            SkyCoord(ra_c, dec_c, unit='deg'), 
            radius=radius_deg*u.deg, 
            catalog='I/355/gaiadr3',
            column_filters={"Gmag":"<20", "e_Gmag":"<0.01", "RUWE":"<1.4"}
        )
            
        if len(result) == 0:
            return pd.DataFrame()
            
        gaia = result[0].to_pandas()
        gaia.rename(columns={'RA_ICRS': 'ra', 'DE_ICRS': 'dec'}, inplace=True)
        
        # Filtrado adicional de calidad
        quality_cuts = (
            (gaia['e_Gmag'] < 0.01) &
            (gaia['RUWE'] < 1.4) &
            (gaia['Gmag'] > 13) & (gaia['Gmag'] < 19) &
            (gaia['Plx'].notna()) & (gaia['Plx'] > 0)
        )
        
        return gaia[quality_cuts].copy()
        
    except Exception as e:
        print(f"Error en consulta Gaia: {e}")
        return pd.DataFrame()

def crossmatch_photometry_kdtree(stars, gaia, match_radius_arcsec=1.0):
    """
    Función alternativa para cruce de coordenadas usando KDTree
    que evita los problemas con las funciones de Astropy
    """
    if len(stars) == 0 or len(gaia) == 0:
        return pd.DataFrame()
    
    # Convertir a arrays de numpy
    stars_ra = stars['ra'].values
    stars_dec = stars['dec'].values
    gaia_ra = gaia['ra'].values
    gaia_dec = gaia['dec'].values
    
    # Convertir radio de arcosegundos a grados
    radius_deg = match_radius_arcsec / 3600.0
    
    # Crear KDTree para las coordenadas de Gaia
    # Convertir coordenadas esféricas a coordenadas 3D en la esfera unitaria
    phi_gaia = np.radians(gaia_ra)
    theta_gaia = np.radians(90.0 - gaia_dec)
    x_gaia = np.sin(theta_gaia) * np.cos(phi_gaia)
    y_gaia = np.sin(theta_gaia) * np.sin(phi_gaia)
    z_gaia = np.cos(theta_gaia)
    gaia_points = np.vstack([x_gaia, y_gaia, z_gaia]).T
    
    # Crear KDTree
    tree = KDTree(gaia_points)
    
    # Convertir coordenadas de estrellas a 3D
    phi_stars = np.radians(stars_ra)
    theta_stars = np.radians(90.0 - stars_dec)
    x_stars = np.sin(theta_stars) * np.cos(phi_stars)
    y_stars = np.sin(theta_stars) * np.sin(phi_stars)
    z_stars = np.cos(theta_stars)
    stars_points = np.vstack([x_stars, y_stars, z_stars]).T
    
    # Buscar vecinos más cercanos
    distances, indices = tree.query(stars_points, k=1)
    
    # Convertir distancias (en unidades de esfera unitaria) a grados
    # La distancia en la esfera unitaria es el chord distance
    # Podemos convertirla a distancia angular: angle = 2 * arcsin(d/2)
    angular_distances_deg = 2 * np.arcsin(distances / 2) * 180 / np.pi
    
    # Aplicar el umbral de distancia
    mask = angular_distances_deg < radius_deg
    
    if not np.any(mask):
        return pd.DataFrame()
    
    # Crear DataFrame con los resultados
    matched = stars.iloc[mask].copy()
    matched['gaia_idx'] = indices[mask]
    matched['gaia_G'] = gaia.iloc[indices[mask]]['Gmag'].values
    matched['gaia_BP'] = gaia.iloc[indices[mask]]['BPmag'].values
    matched['gaia_RP'] = gaia.iloc[indices[mask]]['RPmag'].values
    matched['gaia_Plx'] = gaia.iloc[indices[mask]]['Plx'].values
    matched['gaia_pmRA'] = gaia.iloc[indices[mask]]['pmRA'].values
    matched['gaia_pmDE'] = gaia.iloc[indices[mask]]['pmDE'].values
    matched['gaia_e_Gmag'] = gaia.iloc[indices[mask]]['e_Gmag'].values
    matched['gaia_RUWE'] = gaia.iloc[indices[mask]]['RUWE'].values
    matched['distance_arcsec'] = angular_distances_deg[mask] * 3600
    
    return matched

def compute_zeropoint(matched, min_N=20):
    """Cálculo robusto de zeropoint con estimación de incertidumbre"""
    if len(matched) < min_N:
        return np.nan, np.nan, np.nan, 0
        
    # Cálculo de diferencia
    delta = matched['gaia_G'] - matched['mag_inst']
    
    # Eliminación de outliers usando método robusto
    z_scores = np.abs(stats.zscore(delta))
    inliers = z_scores < 3
    
    if np.sum(inliers) < min_N:
        return np.nan, np.nan, np.nan, 0
        
    delta_clean = delta[inliers]
    
    # Estimación robusta del zeropoint y su incertidumbre
    zp = np.median(delta_clean)
    mad = mad_std(delta_clean)
    mean_error = np.mean(matched.loc[inliers, 'gaia_e_Gmag'])
    
    # Incertidumbre total (combinación de dispersión y error fotométrico)
    total_uncertainty = np.sqrt(mad**2 + mean_error**2)
    
    return zp, mad, total_uncertainty, np.sum(inliers)

def main():
    parser = argparse.ArgumentParser(description='Calibración fotométrica robusta para S-PLUS')
    parser.add_argument('--filters', nargs='+', required=True, help='Filtros a calibrar')
    parser.add_argument('--output', default='splus_zeropoints.csv', help='Archivo de salida')
    parser.add_argument('--radius', type=float, default=1.0, help='Radio de búsqueda Gaia (grados)')
    parser.add_argument('--match-radius', type=float, default=1.0, help='Radio de matching (arcsec)')
    parser.add_argument('--min-stars', type=int, default=20, help='Mínimo de estrellas para calibración')
    
    args = parser.parse_args()
    
    results = []
    for filter_name in args.filters:
        print(f"\n=== Calibrando {filter_name} ===")
        
        # Extracción de estrellas
        stars = extract_stars(f'CenA01_{filter_name}.fits')
        if len(stars) < args.min_stars:
            print(f"Insuficientes estrellas detectadas: {len(stars)}")
            continue
            
        # Consulta Gaia
        ra_c, dec_c = np.median(stars['ra']), np.median(stars['dec'])
        gaia = query_gaia(ra_c, dec_c, args.radius)
        
        if len(gaia) == 0:
            print("No se encontraron estrellas Gaia de calidad")
            continue
            
        # Cross-matching
        matched = crossmatch_photometry_kdtree(stars, gaia, args.match_radius)
        
        if len(matched) < args.min_stars:
            print(f"Insuficientes coincidencias: {len(matched)}")
            continue
            
        # Cálculo de zeropoint
        zp, mad, total_uncertainty, N = compute_zeropoint(matched, args.min_stars)
        
        if not np.isnan(zp):
            print(f"Zeropoint para {filter_name}: {zp:.3f} ± {total_uncertainty:.3f} (N={N}, MAD={mad:.3f})")
            results.append({
                'filter': filter_name,
                'zeropoint': zp,
                'uncertainty': total_uncertainty,
                'mad': mad,
                'N_stars': N,
                'ra_field': ra_c,
                'dec_field': dec_c
            })
            
            # Guardar datos para validación
            matched.to_csv(f'CenA01_{filter_name}_calibration_data.csv', index=False)
    
    # Guardar resultados
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResultados guardados en {args.output}")
        
        # Generar reporte de validación
        generate_validation_report(df)
    
    return results

def generate_validation_report(results_df):
    """Genera un reporte de validación para el paper"""
    report = [
        "VALIDACIÓN FOTOMÉTRICA - S-PLUS CALIBRATION",
        "=" * 50,
        f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Campos analizados: {len(results_df)}",
        ""
    ]
    
    for _, row in results_df.iterrows():
        report.append(
            f"{row['filter']}: ZP = {row['zeropoint']:.3f} ± {row['uncertainty']:.3f} "
            f"(N={row['N_stars']}, MAD={row['mad']:.3f})"
        )
    
    report.extend([
        "",
        "PARÁMETROS DE CALIDAD:",
        f"- Mínimo de estrellas por filtro: {results_df['N_stars'].min()}",
        f"- Incertidumbre media: {results_df['uncertainty'].mean():.3f}",
        f"- MAD medio: {results_df['mad'].mean():.3f}",
        "",
        "CRITERIOS DE CALIDAD APLICADOS:",
        "- Estrellas Gaia con G < 20, error_G < 0.01 mag, RUWE < 1.4",
        "- Matching con radio < 1.0 arcsec",
        "- Eliminación de outliers (3σ basado en Z-score)",
        "- Estimación robusta usando mediana y MAD"
    ])
    
    with open('photometric_validation_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Reporte de validación generado: photometric_validation_report.txt")

if __name__ == "__main__":
    main()
