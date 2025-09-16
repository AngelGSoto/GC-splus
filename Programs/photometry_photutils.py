import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SPLUSPhotometry:
    def __init__(self, catalog_path):
        self.catalog = pd.read_csv(catalog_path)
        self.filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.zeropoints = {'F378':23.53, 'F395':23.60, 'F410':23.55, 
                          'F430':23.52, 'F515':24.35, 'F660':24.28, 'F861':24.67}
        
    def measure_photometry(self):
        """Pipeline principal de fotometría"""
        results = []
        
        for filter_name in tqdm(self.filters, desc="Processing filters"):
            try:
                # Cargar imagen y weight
                with fits.open(f'CenA01_{filter_name}.fits') as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    wcs = WCS(header)
                
                with fits.open(f'CenA01_{filter_name}.weight.fits') as hdul:
                    weight_data = hdul[0].data
                
                # Fotometría para cada fuente
                for _, source in self.catalog.iterrows():
                    mag, mag_err = self.measure_source(
                        source, data, weight_data, wcs, filter_name)
                    
                    results.append({
                        'source_id': source['id'],
                        'filter': filter_name,
                        'ra': source['ra'],
                        'dec': source['dec'],
                        'mag': mag,
                        'mag_err': mag_err
                    })
                    
            except FileNotFoundError:
                print(f"Warning: Files for {filter_name} not found")
                continue
        
        return pd.DataFrame(results)
    
    def measure_source(self, source, data, weight_data, wcs, filter_name):
        """Fotometría para una fuente individual"""
        # Convertir coordenadas a píxeles
        coord = SkyCoord(ra=source['ra']*u.deg, dec=source['dec']*u.deg)
        x, y = wcs.world_to_pixel(coord)
        
        # Parámetros de apertura (ajustar según FWHM)
        aper_radius = 3.0
        annulus_radius = 7.0
        annulus_width = 3.0
        
        # Aperturas
        aperture = CircularAperture((x, y), r=aper_radius)
        annulus_aperture = CircularAnnulus((x, y), 
                                         r_in=annulus_radius, 
                                         r_out=annulus_radius + annulus_width)
        
        # Fotometría
        phot_table = aperture_photometry(data, aperture, error=1/np.sqrt(weight_data))
        bkg_phot_table = aperture_photometry(data, annulus_aperture)
        
        # Calcular fondo
        bkg_mean = bkg_phot_table['aperture_sum'] / annulus_aperture.area()
        bkg_sum = bkg_mean * aperture.area()
        
        # Flujo final y magnitud
        final_sum = phot_table['aperture_sum'] - bkg_sum
        mag = self.zeropoints[filter_name] - 2.5 * np.log10(final_sum)
        
        # Error (aproximado)
        mag_err = 1.0857 / np.sqrt(final_sum)
        
        return mag.values[0], mag_err.values[0]

# Uso del pipeline
if __name__ == "__main__":
    photometry = SPLUSPhotometry('catalogo_ngc5128.csv')
    results = photometry.measure_photometry()
    
    # Guardar resultados
    results.to_csv('ngc5128_splus_photometry.csv', index=False)
    
    # Crear tabla pivot para análisis
    pivot_df = results.pivot_table(index='source_id', columns='filter', 
                                  values=['mag', 'mag_err'])
    pivot_df.to_csv('ngc5128_splus_photometry_pivot.csv')
