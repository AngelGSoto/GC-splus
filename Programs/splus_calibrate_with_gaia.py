import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground, Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry
from astroquery.vizier import Vizier
import os
import argparse
from scipy.spatial import KDTree
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

def check_image_quality(image_file):
    """Función para diagnosticar problemas en las imágenes"""
    try:
        data, header = get_image_data_and_header(image_file)
        
        # Estadísticas básicas
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        print(f"Estadísticas de {image_file}:")
        print(f"  Media: {mean:.2f}, Mediana: {median:.2f}, Std: {std:.2f}")
        print(f"  Mín: {np.min(data):.2f}, Máx: {np.max(data):.2f}")
        
        # Verificar si la imagen está mayormente vacía
        if std < 1.0:
            print("  ¡ADVERTENCIA: La imagen tiene muy poco ruido, puede estar vacía o calibrada!")
        
        # Verificar saturación
        saturated_pixels = np.sum(data > 30000)
        print(f"  Píxeles saturados: {saturated_pixels}")
        
        return True, mean, median, std
    except Exception as e:
        print(f"Error al verificar la imagen {image_file}: {e}")
        return False, 0, 0, 0

def extract_stars(image_file, threshold_sigma=3, fwhm=3.0, max_sources=5000, saturation_level=35000):
    print(f"Procesando {image_file}...")
    try:
        # Primero verificar la calidad de la imagen
        success, mean, median, std = check_image_quality(image_file)
        if not success:
            return pd.DataFrame()
            
        data, header = get_image_data_and_header(image_file)
        wcs = WCS(header)
        
        # Si la imagen tiene muy bajo ruido, podría estar calibrada
        # En este caso, no restamos fondo y usamos los datos directamente
        if std < 1.0:
            print("Imagen con bajo ruido - usando datos sin resta de fondo")
            data_sub = data
            # Usar un threshold absoluto en lugar de basado en sigma
            absolute_threshold = 5.0  # Ajustar según sea necesario
            daofind = DAOStarFinder(fwhm=fwhm, threshold=absolute_threshold,
                                   sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0)
        else:
            # Usar Background2D para una estimación más robusta del fondo
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, box_size=(50, 50), filter_size=(3, 3), 
                              bkg_estimator=bkg_estimator)
            data_sub = data - bkg.background
            
            # Calcular RMS usando sigma clipping
            _, _, std = sigma_clipped_stats(data_sub, sigma=3.0)
            
            # Detectar estrellas
            daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std,
                                   sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0)
        
        sources = daofind(data_sub)
        
        if sources is None or len(sources) == 0:
            print("No se detectaron estrellas con DAOStarFinder")
            # Intentar con un threshold más bajo si no se encuentran estrellas
            if std < 1.0:
                print("Intentando con threshold más bajo...")
                absolute_threshold = 1.0  # Threshold muy bajo para imágenes con poco ruido
                daofind = DAOStarFinder(fwhm=fwhm, threshold=absolute_threshold,
                                       sharplo=0.1, sharphi=1.5, roundlo=-2.0, roundhi=2.0)
                sources = daofind(data_sub)
                
                if sources is None or len(sources) == 0:
                    return pd.DataFrame()
            
        sources = sources.to_pandas().sort_values('flux', ascending=False)[:max_sources].copy()
        
        # Filtrar estrellas con parámetros anómalos (menos estricto para imágenes con bajo ruido)
        if std < 1.0:
            sources = sources[(sources['sharpness'] > 0.1) & (sources['sharpness'] < 1.5)]
            sources = sources[(sources['roundness1'] > -2.0) & (sources['roundness1'] < 2.0)]
        else:
            sources = sources[(sources['sharpness'] > 0.2) & (sources['sharpness'] < 1.0)]
            sources = sources[(sources['roundness1'] > -1.0) & (sources['roundness1'] < 1.0)]
        
        if len(sources) == 0:
            print("Todas las estrellas fueron filtradas por parámetros de calidad")
            return pd.DataFrame()
            
        xcoords = sources['xcentroid'].values
        ycoords = sources['ycentroid'].values
        
        # Medir el pico de flujo para cada estrella para detectar saturación
        peak_fluxes = []
        for x, y in zip(xcoords, ycoords):
            x_int, y_int = int(round(x)), int(round(y))
            # Extraer un pequeño recuadro alrededor de la estrella
            size = 5
            x_min = max(0, x_int - size)
            x_max = min(data.shape[1], x_int + size + 1)
            y_min = max(0, y_int - size)
            y_max = min(data.shape[0], y_int + size + 1)
            
            region = data[y_min:y_max, x_min:x_max]
            if region.size > 0:
                peak_flux = np.max(region)
                peak_fluxes.append(peak_flux)
            else:
                peak_fluxes.append(0)
        
        sources['peak_flux'] = peak_fluxes
        sources['saturated'] = sources['peak_flux'] > saturation_level
        
        # Convertir coordenadas de píxeles a coordenadas celestiales
        ra_dec_pairs = []
        for x, y in zip(xcoords, ycoords):
            try:
                coord = wcs.pixel_to_world(x, y)
                ra_dec_pairs.append((coord.ra.deg, coord.dec.deg))
            except:
                ra_dec_pairs.append((np.nan, np.nan))
        
        ra, dec = zip(*ra_dec_pairs)
        sources['ra'] = ra
        sources['dec'] = dec
        sources['id'] = np.arange(len(sources))
        
        # Fotometría de apertura con múltiples radios
        positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
        
        # Usar múltiples aperturas para elegir la mejor
        aperture_radii = [fwhm*1.5, fwhm*2.0, fwhm*2.5]
        best_fluxes = None
        
        for radius in aperture_radii:
            aperture = CircularAperture(positions, r=radius)
            phot_table = aperture_photometry(data_sub, aperture)
            fluxes = phot_table['aperture_sum']
            
            if best_fluxes is None:
                best_fluxes = fluxes
            else:
                # Preferir flujos más estables (no saturados)
                mask = (fluxes > 0) & (fluxes < saturation_level)
                if np.any(mask):
                    best_fluxes = np.where(mask, fluxes, best_fluxes)
        
        sources['flux_aperture'] = best_fluxes
        
        # Calcular magnitud instrumental
        sources['mag_inst'] = -2.5 * np.log10(sources['flux_aperture'] + 1e-10)
        
        # Filtrar estrellas saturadas y con flujo negativo o cero
        sources = sources[(~sources['saturated']) & (sources['flux_aperture'] > 0)]
        
        # Ajustar el rango de magnitudes según el caso
        if std < 1.0:
            # Para imágenes con bajo ruido, las magnitudes pueden ser más extremas
            sources = sources[(sources['mag_inst'] > -10) & (sources['mag_inst'] < 30)]
        else:
            sources = sources[(sources['mag_inst'] > 5) & (sources['mag_inst'] < 25)]
        
        print(f"Estrellas detectadas después de filtrar: {len(sources)}")
        return sources[['id', 'ra', 'dec', 'flux_aperture', 'peak_flux', 'saturated', 'mag_inst']].dropna()
        
    except Exception as e:
        print(f"Error extrayendo estrellas: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def query_gaia(ra_c, dec_c, radius_deg=1.5):
    print("Consultando Gaia DR3...")
    try:
        Vizier.ROW_LIMIT = 50000
        # Añadir más columnas de Gaia para mejor filtrado
        Vizier.columns = ['RA_ICRS','DE_ICRS','Gmag','BPmag','RPmag', 'Plx', 'pmRA', 'pmDE', 
                         'e_Gmag', 'e_BPmag', 'e_RPmag', 'RUWE']
        result = Vizier.query_region(
            SkyCoord(ra_c, dec_c, unit='deg'), 
            radius=radius_deg*u.deg, 
            catalog='I/355/gaiadr3')
            
        if len(result)==0:
            return pd.DataFrame()
            
        gaia = result[0].to_pandas()
        gaia.rename(columns={'RA_ICRS':'ra','DE_ICRS':'dec'}, inplace=True)
        
        # Filtrar estrellas con alta incertidumbre o mala calidad de datos
        gaia = gaia.dropna(subset=['ra', 'dec', 'Gmag'])
        
        # Filtrar por calidad de datos (RUWE < 1.4 es generalmente bueno)
        if 'RUWE' in gaia.columns:
            gaia = gaia[gaia['RUWE'] < 1.4]
        
        return gaia
        
    except Exception as e:
        print(f"Error consultando Gaia: {e}")
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

def compute_zeropoint(matched, mag_col_star='mag_inst', mag_col_gaia='gaia_G', min_N=10):
    if len(matched) < min_N:
        return np.nan, np.nan, 0
        
    # Filtrado más laxo para permitir más estrellas
    mask = (matched[mag_col_star].notna() & 
            matched[mag_col_gaia].notna() &
            (matched[mag_col_gaia] > 10) &  # Rango más amplio
            (matched[mag_col_gaia] < 20) &
            (matched['gaia_e_Gmag'] < 0.05) &  # Mayor tolerancia
            (matched['gaia_RUWE'] < 2.0) &     # Mayor tolerancia
            (matched['distance_arcsec'] < 2.0) # Mayor tolerancia
            )
            
    if np.sum(mask) < min_N:
        print(f"Solo {np.sum(mask)} estrellas pasaron el filtro, se necesitan al menos {min_N}")
        return np.nan, np.nan, 0
        
    delta = matched.loc[mask, mag_col_gaia] - matched.loc[mask, mag_col_star]
    
    # Eliminar outliers usando IQR
    Q1 = delta.quantile(0.25)
    Q3 = delta.quantile(0.75)
    IQR = Q3 - Q1
    mask_no_outliers = (delta >= (Q1 - 1.5 * IQR)) & (delta <= (Q3 + 1.5 * IQR))
    
    if np.sum(mask_no_outliers) < min_N:
        print(f"Solo {np.sum(mask_no_outliers)} estrellas después de eliminar outliers")
        return np.nan, np.nan, 0
        
    delta_clean = delta[mask_no_outliers]
    zp = np.median(delta_clean)
    rms = np.std(delta_clean)
    N = np.sum(mask_no_outliers)
    
    return zp, rms, N

def calibrate_filter(filter_name, radius_deg=1.5, match_radius_arcsec=1.0, threshold_sigma=3, saturation_level=35000):
    image_files = [f'CenA01_{filter_name}.fits', f'CenA01_{filter_name}.fits.fz']
    image_file = None
    
    for f in image_files:
        if os.path.exists(f):
            image_file = f
            break
            
    if image_file is None:
        print(f"Archivo no encontrado para filtro {filter_name}")
        return np.nan, np.nan, 0, pd.DataFrame(), pd.DataFrame()
    
    # Extraer estrellas
    stars = extract_stars(image_file, threshold_sigma=threshold_sigma, saturation_level=saturation_level)
    if len(stars) == 0:
        print(f"No se detectaron estrellas en {filter_name}")
        return np.nan, np.nan, 0, pd.DataFrame(), pd.DataFrame()
    
    print(f"Estrellas detectadas: {len(stars)} (después de filtrar saturadas)")
    
    # Guardar estrellas detectadas
    stars_output_file = f'CenA01_{filter_name}_detected_stars.csv'
    stars.to_csv(stars_output_file, index=False)
    print(f"Estrellas detectadas guardadas en {stars_output_file}")
    
    # Consultar Gaia
    ra_c, dec_c = stars['ra'].mean(), stars['dec'].mean()
    gaia = query_gaia(ra_c, dec_c, radius_deg=radius_deg)
    if len(gaia) == 0:
        print(f"No se encontraron estrellas Gaia para {filter_name}")
        return np.nan, np.nan, 0, stars, pd.DataFrame()
    
    print(f"Estrellas Gaia encontradas: {len(gaia)}")
    
    # Guardar datos de Gaia
    gaia_output_file = f'CenA01_{filter_name}_gaia_stars.csv'
    gaia.to_csv(gaia_output_file, index=False)
    print(f"Estrellas Gaia guardadas en {gaia_output_file}")
    
    # Cruzar catálogos usando KDTree
    matched = crossmatch_photometry_kdtree(stars, gaia, match_radius_arcsec)
    if len(matched) == 0:
        print(f"No hay matches para {filter_name}")
        return np.nan, np.nan, 0, stars, gaia
    
    print(f"Coincidencias encontradas: {len(matched)}")
    
    # Guardar coincidencias
    matched_output_file = f'CenA01_{filter_name}_matched_stars.csv'
    matched.to_csv(matched_output_file, index=False)
    print(f"Coincidencias estrellas-Gaia guardadas en {matched_output_file}")
    
    # Calcular zeropoint
    zp, rms, N = compute_zeropoint(matched)
    return zp, rms, N, stars, matched

def main():
    parser = argparse.ArgumentParser(description='Calibrar fotometría S-PLUS usando Gaia DR3')
    parser.add_argument('--filters', nargs='+', default=['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861'],
                        help='Lista de filtros a calibrar')
    parser.add_argument('--output', default='splus_zeropoints.csv',
                        help='Archivo de salida para los zeropoints')
    parser.add_argument('--radius', type=float, default=1.5,
                        help='Radio de búsqueda en grados para Gaia')
    parser.add_argument('--match-radius', type=float, default=1.0,
                        help='Radio de matching en arcosegundos')
    parser.add_argument('--threshold-sigma', type=float, default=3.0,
                        help='Sigma para el threshold de detección de estrellas')
    parser.add_argument('--saturation-level', type=float, default=35000,
                        help='Nivel de saturación en ADUs')
    parser.add_argument('--min-stars', type=int, default=10,
                        help='Número mínimo de estrellas para calcular zeropoint')
    parser.add_argument('--abs-threshold', type=float, default=5.0,
                        help='Threshold absoluto para imágenes con bajo ruido')
    args = parser.parse_args()
    
    results = []
    all_matched_stars = []
    
    for filter_name in args.filters:
        print(f"\n=== Calibrando filtro {filter_name} ===")
        zp, rms, N, stars, matched = calibrate_filter(
            filter_name, 
            args.radius, 
            args.match_radius,
            args.threshold_sigma,
            args.saturation_level
        )
        
        # Añadir información del filtro a las estrellas coincidentes
        if len(matched) > 0:
            matched['filter'] = filter_name
            all_matched_stars.append(matched)
        
        if not np.isnan(zp):
            print(f"Zeropoint para {filter_name}: {zp:.3f} ± {rms:.3f} (N={N})")
            results.append({
                'filter': filter_name,
                'zeropoint': zp,
                'rms': rms,
                'N_stars': N
            })
        else:
            print(f"No se pudo calibrar {filter_name}")
            results.append({
                'filter': filter_name,
                'zeropoint': np.nan,
                'rms': np.nan,
                'N_stars': 0
            })
    
    # Guardar resultados de zeropoints
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResultados de zeropoints guardados en {args.output}")
    
    # Guardar todas las estrellas coincidentes en un solo archivo
    if all_matched_stars:
        all_matched_df = pd.concat(all_matched_stars, ignore_index=True)
        all_matched_output = 'CenA01_all_matched_stars.csv'
        all_matched_df.to_csv(all_matched_output, index=False)
        print(f"Todas las coincidencias estrellas-Gaia guardadas en {all_matched_output}")
    
    # Mostrar resumen
    print("\nResumen de calibración:")
    for _, row in df.iterrows():
        if not np.isnan(row['zeropoint']):
            print(f"{row['filter']}: ZP = {row['zeropoint']:.3f} ± {row['rms']:.3f} (N={row['N_stars']})")
        else:
            print(f"{row['filter']}: No calibrado")
    
    # Información adicional sobre las coincidencias
    if all_matched_stars:
        total_matched = sum(len(m) for m in all_matched_stars)
        print(f"\nTotal de estrellas coincidentes encontradas: {total_matched}")
        
        # Estadísticas por filtro
        print("\nEstadísticas de coincidencias por filtro:")
        for filter_name in args.filters:
            filter_matches = [m for m in all_matched_stars if m['filter'].iloc[0] == filter_name]
            if filter_matches:
                match_count = len(filter_matches[0])
                print(f"{filter_name}: {match_count} coincidencias")

if __name__ == "__main__":
    main()
