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
from astropy.stats import sigma_clipped_stats
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path, zeropoints_file):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"El archivo de catálogo {catalog_path} no existe")
        
        if not os.path.exists(zeropoints_file):
            raise FileNotFoundError(f"El archivo de zeropoints {zeropoints_file} no existe")
        
        # Cargar zeropoints
        self.zeropoints_df = pd.read_csv(zeropoints_file)
        self.zeropoints = {}
        for _, row in self.zeropoints_df.iterrows():
            field = row['field']
            self.zeropoints[field] = {
                'F378': row['F378'],
                'F395': row['F395'],
                'F410': row['F410'],
                'F430': row['F430'],
                'F515': row['F515'],
                'F660': row['F660'],
                'F861': row['F861']
            }
        
        # Cargar catálogo original y guardar una copia
        self.original_catalog = pd.read_csv(catalog_path)
        self.catalog = self.original_catalog.copy()
        print(f"Catálogo cargado con {len(self.catalog)} fuentes")
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.pixel_scale = 0.55  # arcsec/pixel
        self.apertures = [3, 4, 5, 6]  # diámetros de apertura en arcsec
        
    def get_field_center_from_header(self, field_name, filter_name='F660'):
        """Obtiene el centro del campo desde el header de una imagen"""
        sci_file = os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz")
        if not os.path.exists(sci_file):
            sci_file = os.path.join(field_name, f"{field_name}_{filter_name}.fits")
            if not os.path.exists(sci_file):
                return None, None
        
        try:
            with fits.open(sci_file) as hdul:
                # Determinar la extensión correcta (0 o 1)
                if len(hdul) > 1 and hdul[1].data is not None:
                    header = hdul[1].header
                else:
                    header = hdul[0].header
                
                # Obtener centro del campo desde el header
                ra_center = header.get('CRVAL1', None)
                dec_center = header.get('CRVAL2', None)
                
                if ra_center is None or dec_center is None:
                    return None, None
                    
                return ra_center, dec_center
        except:
            return None, None
    
    def is_source_in_field(self, source_ra, source_dec, field_ra, field_dec, field_radius_deg=0.84):
        """Verifica si una fuente está dentro del campo usando coordenadas esféricas"""
        coord1 = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
        coord2 = SkyCoord(ra=field_ra*u.deg, dec=field_dec*u.deg)
        separation = coord1.separation(coord2).degree
        return separation <= field_radius_deg
    
    def process_field(self, field_name):
        """Procesa un campo completo"""
        print(f"\nProcesando campo: {field_name}")
        
        # Verificar si tenemos zeropoints para este campo
        if field_name not in self.zeropoints:
            print(f"No hay zeropoints para el campo {field_name}. Saltando...")
            return None
        
        # Obtener centro del campo desde el header
        field_ra, field_dec = self.get_field_center_from_header(field_name)
        if field_ra is None or field_dec is None:
            print(f"No se pudo obtener el centro del campo {field_name} desde el header. Saltando...")
            return None
        
        print(f"Centro del campo {field_name}: RA={field_ra:.6f}, Dec={field_dec:.6f}")
        
        # Filtrar objetos que están dentro del campo
        objects_in_field = []
        for idx, source in self.catalog.iterrows():
            if self.is_source_in_field(source['RAJ2000'], source['DEJ2000'], field_ra, field_dec):
                objects_in_field.append(source)
        
        if not objects_in_field:
            print(f"No hay objetos del catálogo en el campo {field_name}. Saltando...")
            return None
        
        objects_in_field = pd.DataFrame(objects_in_field)
        print(f"Hay {len(objects_in_field)} objetos en el campo {field_name}")
        
        field_zeropoints = self.zeropoints[field_name]
        print(f"Zeropoints para {field_name}: {field_zeropoints}")
        
        # Crear un DataFrame base con las fuentes que están en el campo
        results = objects_in_field[['T17ID', 'RAJ2000', 'DEJ2000']].copy()
        results.rename(columns={'T17ID': 'ID', 'RAJ2000': 'RA', 'DEJ2000': 'DEC'}, inplace=True)
        
        # Inicializar todas las columnas de filtro con NaN
        for filter_name in self.filters:
            for aperture in self.apertures:
                results[f'X_{filter_name}_{aperture}'] = np.nan
                results[f'Y_{filter_name}_{aperture}'] = np.nan
                results[f'FLUX_{filter_name}_{aperture}'] = np.nan
                results[f'FLUXERR_{filter_name}_{aperture}'] = np.nan
                results[f'MAG_{filter_name}_{aperture}'] = np.nan
                results[f'MAGERR_{filter_name}_{aperture}'] = np.nan
                results[f'SNR_{filter_name}_{aperture}'] = np.nan
        
        # Procesar cada filtro
        for filter_name in tqdm(self.filters, desc=f"Procesando {field_name}"):
            try:
                sci_file = os.path.join(field_name, f"{field_name}_{filter_name}.fits.fz")
                if not os.path.exists(sci_file):
                    sci_file = os.path.join(field_name, f"{field_name}_{filter_name}.fits")
                    if not os.path.exists(sci_file):
                        print(f"Imagen {field_name}_{filter_name} no encontrada. Saltando...")
                        continue
                
                # Cargar imagen científica
                with fits.open(sci_file) as hdul:
                    # Determinar la extensión correcta (0 o 1)
                    if len(hdul) > 1 and hdul[1].data is not None:
                        data = hdul[1].data.astype(float)
                        header = hdul[1].header
                    else:
                        data = hdul[0].data.astype(float)
                        header = hdul[0].header
                    
                    # Crear WCS - mejorado según el header de ejemplo
                    try:
                        wcs = WCS(header, relax=True)
                        # Verificar que el WCS sea válido
                        if not wcs.is_celestial:
                            raise ValueError("WCS no es celestial")
                    except Exception as wcs_error:
                        print(f"Error con WCS en {filter_name}: {wcs_error}. Usando WCS simple")
                        wcs = WCS(naxis=2)
                        wcs.wcs.crpix = [header.get('CRPIX1', 0), header.get('CRPIX2', 0)]
                        wcs.wcs.crval = [header.get('CRVAL1', 0), header.get('CRVAL2', 0)]
                        wcs.wcs.cdelt = [header.get('CD1_1', header.get('CDELT1', 1)), 
                                         header.get('CD2_2', header.get('CDELT2', 1))]
                        if 'CD1_1' in header and 'CD2_2' in header:
                            wcs.wcs.pc = [[header['CD1_1'], header.get('CD1_2', 0)],
                                         [header.get('CD2_1', 0), header['CD2_2']]]
                
                # Calcular mapa de errores (usando desviación estándar del fondo)
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                error_data = np.full_like(data, std)
                
                # Procesar cada fuente en este campo
                for idx, source in objects_in_field.iterrows():
                    result = self.measure_source(
                        source, data, error_data, wcs, filter_name, field_zeropoints)
                    if result:
                        # Actualizar los resultados con las mediciones de este filtro
                        source_id = source['T17ID']
                        mask = results['ID'] == source_id
                        if mask.any():
                            for key, value in result.items():
                                if key in results.columns:
                                    results.loc[mask, key] = value
                        
            except Exception as e:
                print(f"Error procesando {filter_name} en {field_name}: {e}")
                traceback.print_exc()
                continue
        
        return results
    
    def measure_source(self, source, data, error_data, wcs, filter_name, field_zeropoints):
        """Fotometría para una fuente individual con múltiples aperturas"""
        try:
            # Convertir coordenadas a píxeles
            coord = SkyCoord(ra=source['RAJ2000']*u.deg, dec=source['DEJ2000']*u.deg)
            x, y = wcs.world_to_pixel(coord)
            
            # Verificar que las coordenadas estén dentro de la imagen
            if (x < 0 or x >= data.shape[1] or y < 0 or y >= data.shape[0]):
                return None
            
            results = {}
            
            # Procesar cada apertura
            for aperture_size in self.apertures:
                # Convertir tamaños de apertura de arcseg a píxeles
                aper_radius_pixels = aperture_size / self.pixel_scale / 2.0  # Radio en píxeles
                annulus_inner_pixels = 6.0 / self.pixel_scale
                annulus_outer_pixels = 9.0 / self.pixel_scale
                
                # Verificar que la apertura esté completamente dentro de la imagen
                buffer = annulus_outer_pixels + 2
                if (x < buffer or x >= data.shape[1] - buffer or 
                    y < buffer or y >= data.shape[0] - buffer):
                    # Si la fuente está demasiado cerca del borde para esta apertura,
                    # marcamos con 99 en lugar de NaN para indicar que está en el campo
                    # pero no se pudo medir
                    results.update({
                        f'X_{filter_name}_{aperture_size}': x,
                        f'Y_{filter_name}_{aperture_size}': y,
                        f'FLUX_{filter_name}_{aperture_size}': 0.0,
                        f'FLUXERR_{filter_name}_{aperture_size}': 99.0,
                        f'MAG_{filter_name}_{aperture_size}': 99.0,
                        f'MAGERR_{filter_name}_{aperture_size}': 99.0,
                        f'SNR_{filter_name}_{aperture_size}': 0.0
                    })
                    continue
                
                # Crear aperturas
                aperture = CircularAperture((x, y), r=aper_radius_pixels)
                annulus_aperture = CircularAnnulus((x, y), 
                                                 r_in=annulus_inner_pixels, 
                                                 r_out=annulus_outer_pixels)
                
                # Realizar fotometría
                phot_table = aperture_photometry(data, aperture, error=error_data)
                bkg_phot_table = aperture_photometry(data, annulus_aperture, error=error_data)
                
                # Calcular fondo
                bkg_mean = bkg_phot_table['aperture_sum'] / annulus_aperture.area
                bkg_sum = bkg_mean * aperture.area
                
                # Calcular flujo neto y error
                final_sum = phot_table['aperture_sum'][0] - bkg_sum[0]
                
                # Calcular error (simplificado)
                final_error = np.sqrt(phot_table['aperture_sum_err'][0]**2 + (bkg_sum[0] * 0.1)**2)
                
                # Obtener el zeropoint para este filtro
                if filter_name not in field_zeropoints:
                    print(f"Advertencia: No hay zeropoint para {filter_name}")
                    zp = 0.0
                else:
                    zp = field_zeropoints[filter_name]
                
                # Para fuentes con flujo negativo o cero, usar 99 en lugar de NaN
                if final_sum <= 0:
                    results.update({
                        f'X_{filter_name}_{aperture_size}': x,
                        f'Y_{filter_name}_{aperture_size}': y,
                        f'FLUX_{filter_name}_{aperture_size}': 0.0,
                        f'FLUXERR_{filter_name}_{aperture_size}': 99.0,
                        f'MAG_{filter_name}_{aperture_size}': 99.0,
                        f'MAGERR_{filter_name}_{aperture_size}': 99.0,
                        f'SNR_{filter_name}_{aperture_size}': 0.0
                    })
                    continue
                
                # Calcular magnitud
                mag = zp - 2.5 * np.log10(final_sum)
                mag_err = 2.5 * final_error / (final_sum * np.log(10))
                
                # Calcular SNR
                snr = final_sum / final_error
                
                # Control de calidad - si SNR es bajo, usar 99 en lugar de NaN
                if snr < 3:
                    results.update({
                        f'X_{filter_name}_{aperture_size}': x,
                        f'Y_{filter_name}_{aperture_size}': y,
                        f'FLUX_{filter_name}_{aperture_size}': final_sum,
                        f'FLUXERR_{filter_name}_{aperture_size}': final_error,
                        f'MAG_{filter_name}_{aperture_size}': 99.0,
                        f'MAGERR_{filter_name}_{aperture_size}': 99.0,
                        f'SNR_{filter_name}_{aperture_size}': snr
                    })
                else:
                    results.update({
                        f'X_{filter_name}_{aperture_size}': x,
                        f'Y_{filter_name}_{aperture_size}': y,
                        f'FLUX_{filter_name}_{aperture_size}': final_sum,
                        f'FLUXERR_{filter_name}_{aperture_size}': final_error,
                        f'MAG_{filter_name}_{aperture_size}': mag,
                        f'MAGERR_{filter_name}_{aperture_size}': mag_err,
                        f'SNR_{filter_name}_{aperture_size}': snr
                    })
            
            return results
            
        except Exception as e:
            print(f"Error midiendo fuente {source.get('T17ID', 'unknown')} en {filter_name}: {e}")
            # En caso de error, devolver 99 para todas las aperturas
            results = {}
            for aperture_size in self.apertures:
                results.update({
                    f'X_{filter_name}_{aperture_size}': x if 'x' in locals() else 0,
                    f'Y_{filter_name}_{aperture_size}': y if 'y' in locals() else 0,
                    f'FLUX_{filter_name}_{aperture_size}': 0.0,
                    f'FLUXERR_{filter_name}_{aperture_size}': 99.0,
                    f'MAG_{filter_name}_{aperture_size}': 99.0,
                    f'MAGERR_{filter_name}_{aperture_size}': 99.0,
                    f'SNR_{filter_name}_{aperture_size}': 0.0
                })
            return results

# Lista de campos a procesar
fields = [
    'CenA01', 'CenA02', 'CenA03', 'CenA04', 'CenA05', 'CenA06', 
    'CenA07', 'CenA08', 'CenA09', 'CenA10', 'CenA11', 'CenA12',
    'CenA13', 'CenA14', 'CenA15', 'CenA16', 'CenA17', 'CenA18',
    'CenA19', 'CenA20', 'CenA21', 'CenA22', 'CenA23', 'CenA24'
]

# Ejecución principal
if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    zeropoints_file = 'all_fields_zero_points_splus_format.csv'
    
    if not os.path.exists(catalog_path):
        print(f"Error: No se encuentra el archivo de catálogo {catalog_path}")
        print("\nArchivos en el directorio actual:")
        for f in os.listdir('.'):
            print(f"  {f}")
    elif not os.path.exists(zeropoints_file):
        print(f"Error: No se encuentra el archivo de zeropoints {zeropoints_file}")
    else:
        try:
            all_results = []
            photometry = SPLUSPhotometry(catalog_path, zeropoints_file)
            
            for field in fields:
                results = photometry.process_field(field)
                if results is not None and len(results) > 0:
                    # Añadir información del campo
                    results['FIELD'] = field
                    all_results.append(results)
                    
                    # Guardar resultados individuales
                    output_file = f'{field}_gc_photometry.csv'
                    results.to_csv(output_file, index=False, float_format='%.6f')
                    print(f"Resultados para {field} guardados en {output_file}")
            
            # Combinar todos los resultados
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                
                # Reemplazar NaN por valores específicos en las columnas de fotometría
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        mag_col = f'MAG_{filter_name}_{aperture}'
                        magerr_col = f'MAGERR_{filter_name}_{aperture}'
                        flux_col = f'FLUX_{filter_name}_{aperture}'
                        fluxerr_col = f'FLUXERR_{filter_name}_{aperture}'
                        snr_col = f'SNR_{filter_name}_{aperture}'
                        x_col = f'X_{filter_name}_{aperture}'
                        y_col = f'Y_{filter_name}_{aperture}'
                        
                        # Para coordenadas, si son NaN, el objeto no está en el campo
                        # Para magnitudes/flujos, si son NaN, reemplazar con 99 (está en el campo pero no se midió)
                        final_results[flux_col].fillna(0.0, inplace=True)
                        final_results[fluxerr_col].fillna(99.0, inplace=True)
                        final_results[mag_col].fillna(99.0, inplace=True)
                        final_results[magerr_col].fillna(99.0, inplace=True)
                        final_results[snr_col].fillna(0.0, inplace=True)
                
                # Combinar con el catálogo original
                # Primero, renombrar columnas para hacer el merge
                final_results.rename(columns={'ID': 'T17ID', 'RA': 'RAJ2000', 'DEC': 'DEJ2000'}, inplace=True)
                
                # Hacer un merge left para mantener todas las fuentes del catálogo original
                merged_catalog = photometry.original_catalog.merge(
                    final_results, 
                    on='T17ID', 
                    how='left',
                    suffixes=('', '_phot')
                )
                
                # Eliminar columnas duplicadas de RA y DEC
                if 'RAJ2000_phot' in merged_catalog.columns:
                    merged_catalog.drop('RAJ2000_phot', axis=1, inplace=True)
                if 'DEJ2000_phot' in merged_catalog.columns:
                    merged_catalog.drop('DEJ2000_phot', axis=1, inplace=True)
                
                # Para las fuentes que no están en ningún campo, mantener NaN en las columnas de fotometría
                # Para las fuentes que están en algún campo pero no se midieron, ya tenemos 99
                
                # Reordenar columnas
                column_order = list(photometry.original_catalog.columns)
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        column_order.extend([
                            f'X_{filter_name}_{aperture}', f'Y_{filter_name}_{aperture}',
                            f'FLUX_{filter_name}_{aperture}', f'FLUXERR_{filter_name}_{aperture}',
                            f'MAG_{filter_name}_{aperture}', f'MAGERR_{filter_name}_{aperture}',
                            f'SNR_{filter_name}_{aperture}'
                        ])
                column_order.append('FIELD')
                
                # Asegurarse de que todas las columnas existan
                for col in column_order:
                    if col not in merged_catalog.columns:
                        merged_catalog[col] = np.nan
                
                merged_catalog = merged_catalog[column_order]
                
                # Guardar resultados finales
                output_file = 'all_fields_gc_photometry_merged.csv'
                merged_catalog.to_csv(output_file, index=False, float_format='%.6f')
                print(f"Fotometría completada. Resultados finales guardados en {output_file}")
                
                # Análisis de calidad
                print("\nResumen de calidad:")
                print(f"Total de fuentes en catálogo original: {len(photometry.original_catalog)}")
                print(f"Total de fuentes con mediciones: {len(final_results)}")
                
                for filter_name in photometry.filters:
                    for aperture in photometry.apertures:
                        mag_col = f'MAG_{filter_name}_{aperture}'
                        snr_col = f'SNR_{filter_name}_{aperture}'
                        
                        # Contar mediciones válidas (diferentes de 99)
                        valid_measurements = (merged_catalog[mag_col] != 99.0).sum()
                        if valid_measurements > 0:
                            valid_data = merged_catalog[merged_catalog[mag_col] != 99.0]
                            mean_snr = valid_data[snr_col].mean()
                            mean_mag = valid_data[mag_col].mean()
                            
                            print(f"{filter_name}_{aperture}: {valid_measurements} mediciones, "
                                  f"SNR medio: {mean_snr:.1f}, "
                                  f"Mag media: {mean_mag:.2f}")
                        else:
                            print(f"{filter_name}_{aperture}: 0 mediciones")
            else:
                print("No se obtuvieron resultados para ningún campo.")
                # Guardar el catálogo original si no hay resultados
                output_file = 'all_fields_gc_photometry_merged.csv'
                photometry.original_catalog.to_csv(output_file, index=False)
                print(f"No se procesaron campos. Se guarda el catálogo original en {output_file}")
                
        except Exception as e:
            print(f"Error durante la ejecución: {e}")
            traceback.print_exc()
