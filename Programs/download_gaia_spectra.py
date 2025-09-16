#!/usr/bin/env python3

"""
download_gaia_spectra.py

Descarga espectros de Gaia DR3 para estrellas en una región específica
definida por coordenadas (RA, Dec) y un radio.

Uso: python download_gaia_spectra.py <ra_center> <dec_center> <radius_deg> <output_dir>
"""

import os
import sys
import numpy as np
import pandas as pd
import requests
from astroquery.gaia import Gaia
import time
import warnings
warnings.filterwarnings('ignore')

def query_gaia_sources(ra_center, dec_center, radius_deg=1.0, max_sources=100):
    """Consulta fuentes Gaia DR3 con espectros XP en la región de interés."""
    try:
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, 
               phot_bp_mean_mag, phot_rp_mean_mag,
               ruwe, parallax, pmra, pmdec
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
        )
        AND has_xp_continuous = 'True'
        AND phot_g_mean_mag < 19
        AND ruwe < 1.4
        """
        
        print(f"Ejecutando consulta ADQL en Gaia...")
        job = Gaia.launch_job(query)
        results = job.get_results()
        
        # Verificar las columnas disponibles
        print("Columnas disponibles en los resultados:", results.colnames)
        
        if len(results) > max_sources:
            results = results[:max_sources]
            print(f"Se limitó a {max_sources} fuentes de {len(results)} encontradas")
        else:
            print(f"Se encontraron {len(results)} fuentes Gaia con espectros XP")
        
        return results
        
    except Exception as e:
        print(f"Error en consulta Gaia: {e}")
        return None

def download_gaia_spectrum(source_id, output_dir, max_retries=3):
    """Descarga el espectro de Gaia para un source_id dado."""
    for attempt in range(max_retries):
        try:
            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Nombre del archivo de salida
            filename = os.path.join(output_dir, f"{source_id}_spectrum.csv")
            
            # Si ya existe, no descargar de nuevo
            if os.path.exists(filename):
                print(f"Espectro para {source_id} ya existe")
                return filename
                
            # URL para descargar espectro en formato CSV
            url = f"https://gea.esac.esa.int/data-server/data?RETRIEVAL_TYPE=XP_SAMPLED&ID={source_id}&FORMAT=CSV"
            
            # Descargar el archivo
            print(f"Descargando espectro para {source_id} (intento {attempt+1})...")
            
            # Headers para la solicitud
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Realizar la solicitud
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Verificar que la respuesta no esté vacía
            if not response.text.strip():
                print(f"Respuesta vacía para {source_id}")
                return None
                
            # Guardar el espectro
            with open(filename, 'w') as f:
                f.write(response.text)
                
            print(f"Espectro guardado en {filename}")
            return filename
            
        except requests.exceptions.HTTPError as errh:
            print(f"Error HTTP para {source_id}: {errh}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
        except requests.exceptions.ConnectionError as errc:
            print(f"Error de conexión para {source_id}: {errc}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return None
        except requests.exceptions.Timeout as errt:
            print(f"Timeout para {source_id}: {errt}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return None
        except Exception as e:
            print(f"Error descargando espectro para {source_id} (intento {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None

def main():
    """Función principal."""
    # Verificar argumentos
    if len(sys.argv) != 5:
        print("Uso: python download_gaia_spectra.py <ra_center> <dec_center> <radius_deg> <output_dir>")
        print("Ejemplo: python download_gaia_spectra.py 202.848165 -44.847883 0.5 spectra")
        sys.exit(1)
    
    # Obtener parámetros
    ra_center = float(sys.argv[1])
    dec_center = float(sys.argv[2])
    radius_deg = float(sys.argv[3])
    output_dir = sys.argv[4]
    
    print(f"Buscando espectros en RA={ra_center}, Dec={dec_center}, Radio={radius_deg}°")
    
    # Consultar fuentes Gaia
    gaia_data = query_gaia_sources(ra_center, dec_center, radius_deg)
    
    if gaia_data is None or len(gaia_data) == 0:
        print("No se obtuvieron fuentes Gaia")
        sys.exit(1)
    
    # Convertir a DataFrame y verificar nombres de columnas
    gaia_df = gaia_data.to_pandas()
    
    # Verificar si la columna source_id existe (puede estar en minúsculas)
    if 'source_id' not in gaia_df.columns and 'SOURCE_ID' in gaia_df.columns:
        gaia_df['source_id'] = gaia_df['SOURCE_ID']
    elif 'source_id' not in gaia_df.columns:
        print("Error: No se encontró la columna 'source_id' en los resultados")
        print("Columnas disponibles:", gaia_df.columns.tolist())
        sys.exit(1)
    
    # Guardar catálogo de fuentes
    catalog_file = os.path.join(output_dir, "gaia_sources.csv")
    os.makedirs(output_dir, exist_ok=True)
    gaia_df.to_csv(catalog_file, index=False)
    print(f"Catálogo de fuentes guardado en {catalog_file}")
    
    # Descargar espectros
    print("Descargando espectros...")
    downloaded_files = []
    
    for i, row in gaia_df.iterrows():
        source_id = row['source_id']
        spectrum_file = download_gaia_spectrum(source_id, output_dir)
        
        if spectrum_file:
            downloaded_files.append(spectrum_file)
        
        # Pequeña pausa para no sobrecargar el servidor
        time.sleep(1)
        
        # Mostrar progreso
        if (i + 1) % 10 == 0:
            print(f"Procesadas {i + 1}/{len(gaia_df)} fuentes")
    
    print(f"Descarga completada. {len(downloaded_files)} espectros descargados en {output_dir}")

if __name__ == '__main__':
    main()
