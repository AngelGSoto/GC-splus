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
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path):
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"El archivo de catálogo {catalog_path} no existe")
        
        self.catalog = pd.read_csv(catalog_path)
        print(f"Catálogo cargado con {len(self.catalog)} fuentes")
        
        print("\nPrimeras 5 coordenadas del catálogo:")
        for i, row in self.catalog.head().iterrows():
            print(f"Fuente {i}: RA={row['RAJ2000']}, DEC={row['DEJ2000']}")
        
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.zeropoints = {'F378':23.53, 'F395':23.60, 'F410':23.55, 
                          'F430':23.52, 'F515':24.35, 'F660':24.28, 'F861':24.67}
        self.pixel_scale = 0.55  # arcsec/pixel
        
        self.check_and_decompress()
        
    def check_and_decompress(self):
        """Verifica y descomprime archivos .fz si es necesario"""
        for filter_name in self.filters:
            # Para imágenes científicas
            fz_file = f'CenA01_{filter_name}.fits.fz'
            fits_file = f'CenA01_{filter_name}.fits'
            
            if os.path.exists(fz_file) and not os.path.exists(fits_file):
                print(f"Descomprimiendo {fz_file}...")
                with fits.open(fz_file) as hdul:
                    hdul.writeto(fits_file, overwrite=True)
            
            # Para mapas de peso
            fz_weight = f'CenA01_{filter_name}.weight.fits.fz'
            fits_weight = f'CenA01_{filter_name}.weight.fits'
            
            if os.path.exists(fz_weight) and not os.path.exists(fits_weight):
                print(f"Descomprimiendo {fz_weight}...")
                with fits.open(fz_weight) as hdul:
                    hdul.writeto(fits_weight, overwrite=True)
    
    def measure_photometry(self):
        """Pipeline principal de fotometría"""
        # Crear un DataFrame base con todas las fuentes y sus coordenadas
        results = self.catalog[['T17ID', 'RAJ2000', 'DEJ2000']].copy()
        results.rename(columns={'T17ID': 'ID', 'RAJ2000': 'RA', 'DEJ2000': 'DEC'}, inplace=True)
        
        # Inicializar todas las columnas de filtro con NaN
        for filter_name in self.filters:
            results[f'X_{filter_name}'] = np.nan
            results[f'Y_{filter_name}'] = np.nan
            results[f'FLUX_{filter_name}'] = np.nan
            results[f'FLUXERR_{filter_name}'] = np.nan
            results[f'MAG_{filter_name}'] = np.nan
            results[f'MAGERR_{filter_name}'] = np.nan
            results[f'SNR_{filter_name}'] = np.nan
        
        # Procesar cada filtro
        for filter_name in tqdm(self.filters, desc="Processing filters"):
            try:
                sci_file = f'CenA01_{filter_name}.fits'
                weight_file = f'CenA01_{filter_name}.weight.fits'
                
                # Verificar existencia de archivos
                if not os.path.exists(sci_file) or not os.path.exists(weight_file):
                    print(f"Advertencia: Archivos para {filter_name} no encontrados")
                    continue
                
                # Cargar imagen científica
                with fits.open(sci_file) as hdul:
                    # Los datos de S-PLUS están generalmente en la extensión 1
                    if len(hdul) > 1:
                        data = hdul[1].data.astype(float)
                        header = hdul[1].header
                    else:
                        data = hdul[0].data.astype(float)
                        header = hdul[0].header
                    
                    # Crear WCS
                    try:
                        wcs = WCS(header)
                    except:
                        print(f"Error con WCS en {filter_name}, usando WCS simple")
                        wcs = WCS(naxis=2)
                        wcs.wcs.crpix = [header.get('CRPIX1', 0), header.get('CRPIX2', 0)]
                        wcs.wcs.crval = [header.get('CRVAL1', 0), header.get('CRVAL2', 0)]
                        wcs.wcs.cdelt = [header.get('CDELT1', 1), header.get('CDELT2', 1)]
                
                # Cargar mapa de pesos
                with fits.open(weight_file) as hdul:
                    if len(hdul) > 1:
                        weight_data = hdul[1].data.astype(float)
                    else:
                        weight_data = hdul[0].data.astype(float)
                
                # Calcular mapa de errores
                error_data = 1.0 / np.sqrt(weight_data)
                error_data[~np.isfinite(error_data)] = np.nanmax(error_data[np.isfinite(error_data)])
                
                # Procesar cada fuente
                for idx, source in self.catalog.iterrows():
                    result = self.measure_source(
                        source, data, error_data, wcs, filter_name)
                    if result:
                        # Actualizar los resultados con las mediciones de este filtro
                        source_id = source['T17ID']
                        mask = results['ID'] == source_id
                        if mask.any():
                            for key, value in result.items():
                                if key in results.columns:
                                    results.loc[mask, key] = value
                        
            except Exception as e:
                print(f"Error procesando {filter_name}: {e}")
                traceback.print_exc()
                continue
        
        return results
    
    def measure_source(self, source, data, error_data, wcs, filter_name):
        """Fotometría para una fuente individual"""
        try:
            # Convertir coordenadas a píxeles
            coord = SkyCoord(ra=source['RAJ2000']*u.deg, dec=source['DEJ2000']*u.deg)
            x, y = wcs.world_to_pixel(coord)
            
            # Verificar que las coordenadas estén dentro de la imagen
            if (x < 0 or x >= data.shape[1] or y < 0 or y >= data.shape[0]):
                return None
            
            # Convertir tamaños de apertura de arcseg a píxeles
            aper_radius_pixels = 3.0 / self.pixel_scale
            annulus_inner_pixels = 6.0 / self.pixel_scale
            annulus_outer_pixels = 9.0 / self.pixel_scale
            
            # Verificar que la apertura esté completamente dentro de la imagen
            buffer = annulus_outer_pixels + 2
            if (x < buffer or x >= data.shape[1] - buffer or 
                y < buffer or y >= data.shape[0] - buffer):
                return None
            
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
            bkg_error = bkg_phot_table['aperture_sum_err'] / annulus_aperture.area * aperture.area
            
            # Calcular flujo neto y error
            final_sum = phot_table['aperture_sum'][0] - bkg_sum[0]
            final_error = np.sqrt(phot_table['aperture_sum_err'][0]**2 + bkg_error[0]**2)
            
            # Saltar fuentes con flujo negativo o cero
            if final_sum <= 0:
                return None
            
            # Calcular magnitud
            mag = self.zeropoints[filter_name] - 2.5 * np.log10(final_sum)
            mag_err = 2.5 * final_error / (final_sum * np.log(10))
            
            # Calcular SNR
            snr = final_sum / final_error
            
            # Control de calidad
            if snr < 3:
                return None
            
            # Devolver resultados para esta fuente y filtro
            return {
                f'X_{filter_name}': x,
                f'Y_{filter_name}': y,
                f'FLUX_{filter_name}': final_sum,
                f'FLUXERR_{filter_name}': final_error,
                f'MAG_{filter_name}': mag,
                f'MAGERR_{filter_name}': mag_err,
                f'SNR_{filter_name}': snr
            }
            
        except Exception as e:
            print(f"Error midiendo fuente {source.get('T17ID', 'unknown')} en {filter_name}: {e}")
            return None

# Ejecución principal
if __name__ == "__main__":
    catalog_path = '../TAP_1_J_MNRAS_3444_gc.csv'
    
    if not os.path.exists(catalog_path):
        print(f"Error: No se encuentra el archivo de catálogo {catalog_path}")
        print("\nArchivos en el directorio actual:")
        for f in os.listdir('.'):
            print(f"  {f}")
    else:
        try:
            photometry = SPLUSPhotometry(catalog_path)
            results = photometry.measure_photometry()
            
            if len(results) > 0:
                # Reordenar columnas para un formato más legible
                column_order = ['ID', 'RA', 'DEC']
                for filter_name in photometry.filters:
                    column_order.extend([
                        f'X_{filter_name}', f'Y_{filter_name}',
                        f'FLUX_{filter_name}', f'FLUXERR_{filter_name}',
                        f'MAG_{filter_name}', f'MAGERR_{filter_name}',
                        f'SNR_{filter_name}'
                    ])
                
                # Asegurarse de que todas las columnas existan
                for col in column_order:
                    if col not in results.columns:
                        results[col] = np.nan
                
                results = results[column_order]
                
                # Guardar resultados
                output_file = 'ngc5128_splus_photometry_3arcsec_wide.csv'
                results.to_csv(output_file, index=False, float_format='%.6f')
                print(f"Fotometría completada. Resultados guardados en {output_file}")
                
                # Análisis de calidad
                print("\nResumen de calidad:")
                print(f"Total de fuentes: {len(results)}")
                
                for filter_name in photometry.filters:
                    mag_col = f'MAG_{filter_name}'
                    snr_col = f'SNR_{filter_name}'
                    
                    valid_measurements = results[mag_col].notna().sum()
                    if valid_measurements > 0:
                        mean_snr = results[snr_col].mean()
                        mean_mag = results[mag_col].mean()
                        
                        print(f"{filter_name}: {valid_measurements} mediciones, "
                              f"SNR medio: {mean_snr:.1f}, "
                              f"Mag media: {mean_mag:.2f}")
                    else:
                        print(f"{filter_name}: 0 mediciones")
            else:
                print("No se obtuvieron resultados. Verifica:")
                print("1. Que los archivos FITS estén en el directorio correcto")
                print("2. Que las coordenadas del catálogo coincidan con las imágenes")
                print("3. Que los archivos no estén corruptos")
                
        except Exception as e:
            print(f"Error durante la ejecución: {e}")
            traceback.print_exc()
