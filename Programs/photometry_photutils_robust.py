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
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path):
        self.catalog = pd.read_csv(catalog_path)
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.zeropoints = {'F378':23.53, 'F395':23.60, 'F410':23.55, 
                          'F430':23.52, 'F515':24.35, 'F660':24.28, 'F861':24.67}
        self.pixel_scale = 0.55  # arcsec/pixel (de tu header PIXSCALE=0.55)
        
        # Verificar y descomprimir archivos si es necesario
        self.check_and_decompress()
        
    def check_and_decompress(self):
        """Verifica si los archivos están comprimidos y los descomprime si es necesario"""
        for filter_name in self.filters:
            # Para la imagen científica
            fz_file = f'CenA01_{filter_name}.fits.fz'
            fits_file = f'CenA01_{filter_name}.fits'
            
            if os.path.exists(fz_file) and not os.path.exists(fits_file):
                print(f"Descomprimiendo {fz_file}...")
                with fits.open(fz_file) as hdul:
                    hdul.writeto(fits_file, overwrite=True)
            
            # Para el mapa de pesos
            fz_weight = f'CenA01_{filter_name}.weight.fits.fz'
            fits_weight = f'CenA01_{filter_name}.weight.fits'
            
            if os.path.exists(fz_weight) and not os.path.exists(fits_weight):
                print(f"Descomprimiendo {fz_weight}...")
                with fits.open(fz_weight) as hdul:
                    hdul.writeto(fits_weight, overwrite=True)
    
    def measure_photometry(self):
        """Pipeline principal de fotometría"""
        results = []
        
        for filter_name in tqdm(self.filters, desc="Processing filters"):
            try:
                # Verificar que los archivos existan
                if not os.path.exists(f'CenA01_{filter_name}.fits'):
                    print(f"Advertencia: Archivo CenA01_{filter_name}.fits no encontrado")
                    continue
                
                # Cargar imagen y weight
                with fits.open(f'CenA01_{filter_name}.fits') as hdul:
                    data = hdul[1].data.astype(float)
                    header = hdul[1].header
                    wcs = WCS(header)
                
                if not os.path.exists(f'CenA01_{filter_name}.weight.fits'):
                    print(f"Advertencia: Archivo CenA01_{filter_name}.weight.fits no encontrado")
                    continue
                
                with fits.open(f'CenA01_{filter_name}.weight.fits') as hdul:
                    weight_data = hdul[1].data.astype(float)
                
                # Calcular error a partir del weight map
                error_data = 1.0 / np.sqrt(weight_data)
                error_data[~np.isfinite(error_data)] = np.max(error_data[np.isfinite(error_data)])
                
                # Fotometría para cada fuente
                for _, source in self.catalog.iterrows():
                    result = self.measure_source(
                        source, data, error_data, wcs, filter_name)
                    
                    if result:  # Solo añadir si la medición fue exitosa
                        results.append(result)
                        
            except Exception as e:
                print(f"Error procesando {filter_name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def measure_source(self, source, data, error_data, wcs, filter_name):
        """Fotometría para una fuente individual con apertura fija de 3 arcseg"""
        try:
            # Convertir coordenadas a píxeles
            coord = SkyCoord(ra=source['ra']*u.deg, dec=source['dec']*u.deg)
            x, y = wcs.world_to_pixel(coord)
            
            # Verificar que las coordenadas estén dentro de la imagen
            if (x < 10 or x >= data.shape[1]-10 or y < 10 or y >= data.shape[0]-10):
                return None
            
            # Convertir 3 arcseg a píxeles
            aper_radius_pixels = 3.0 / self.pixel_scale
            annulus_inner_pixels = 6.0 / self.pixel_scale  # 6 arcseg
            annulus_outer_pixels = 9.0 / self.pixel_scale  # 9 arcseg
            
            aperture = CircularAperture((x, y), r=aper_radius_pixels)
            annulus_aperture = CircularAnnulus((x, y), 
                                             r_in=annulus_inner_pixels, 
                                             r_out=annulus_outer_pixels)
            
            # Fotometría con estimación de error
            phot_table = aperture_photometry(data, aperture, error=error_data)
            bkg_phot_table = aperture_photometry(data, annulus_aperture, error=error_data)
            
            # Calcular fondo
            bkg_mean = bkg_phot_table['aperture_sum'] / annulus_aperture.area()
            bkg_sum = bkg_mean * aperture.area()
            bkg_error = bkg_phot_table['aperture_sum_err'] / annulus_aperture.area() * aperture.area()
            
            # Flujo final y error
            final_sum = phot_table['aperture_sum'] - bkg_sum
            final_error = np.sqrt(phot_table['aperture_sum_err']**2 + bkg_error**2)
            
            # Calcular magnitud y error
            mag = self.zeropoints[filter_name] - 2.5 * np.log10(final_sum)
            mag_err = 2.5 * final_error / (final_sum * np.log(10))
            
            # Calcular SNR
            snr = final_sum / final_error
            
            # Control de calidad: rechazar mediciones con SNR baja
            if snr.values[0] < 3:
                return None
            
            return {
                'source_id': source['T17ID'],
                'filter': filter_name,
                'ra': source['RAJ2000'],
                'dec': source['RAJ2000'],
                'x': x,
                'y': y,
                'flux': final_sum.values[0],
                'flux_err': final_error.values[0],
                'mag': mag.values[0],
                'mag_err': mag_err.values[0],
                'snr': snr.values[0],
                'aper_radius_arcsec': 3.0,
                'aper_radius_pixels': aper_radius_pixels
            }, print(source['RAJ2000'])
            
        except Exception as e:
            print(f"Error midiendo fuente {source['T17ID']} en {filter_name}: {e}")
            return None

# Uso del pipeline
if __name__ == "__main__":
    photometry = SPLUSPhotometry('../TAP_1_J_MNRAS_3444_gc.csv')
    results = photometry.measure_photometry()
    
    # Guardar resultados
    if len(results) > 0:
        results.to_csv('ngc5128_splus_photometry_3arcsec.csv', index=False)
        print(f"Fotometría completada para {len(results)} mediciones")
        
        # Análisis de calidad
        print("\nResumen de calidad:")
        for filter_name in photometry.filters:
            filter_data = results[results['filter'] == filter_name]
            if len(filter_data) > 0:
                print(f"{filter_name}: {len(filter_data)} mediciones, SNR medio: {filter_data['snr'].mean():.1f}")
    else:
        print("No se encontraron resultados. Verifica que los archivos estén descomprimidos y accesibles.")
