#!/usr/bin/env python3

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path

def simple_crossmatch():
    """Cross-match simple entre archivos de referencia y Taylor"""
    taylor_file = "../TAP_1_J_MNRAS_3444_psc.csv"
    print("Cargando catálogo de Taylor...")
    taylor = pd.read_csv(taylor_file)
    print(f"Taylor: {len(taylor)} fuentes")
    
    for field_num in range(1, 25):
        field_name = f"CenA{field_num:02d}"
        ref_file = f"{field_name}_gaia_xp_matches_3arcsec.csv"
        output_file = f"{field_name}_reference_taylor_matches.csv"
        
        if not Path(ref_file).exists():
            print(f"Saltando {field_name} - archivo no encontrado")
            continue
        
        print(f"Procesando {field_name}...")
        ref_stars = pd.read_csv(ref_file)
        print(f"  Referencia: {len(ref_stars)} estrellas")
        
        # Limpiar coordenadas y resetear índices
        ref_stars = ref_stars.dropna(subset=['ra', 'dec'])
        ref_stars['ra'] = pd.to_numeric(ref_stars['ra'], errors='coerce')
        ref_stars['dec'] = pd.to_numeric(ref_stars['dec'], errors='coerce')
        ref_stars = ref_stars.dropna(subset=['ra', 'dec']).reset_index(drop=True)
        
        if len(ref_stars) == 0:
            print(f"  No hay coordenadas válidas en {ref_file}")
            continue
        
        print(f"  Coordenadas válidas: {len(ref_stars)}")
        
        try:
            # Crear coordenadas
            ref_coords = SkyCoord(ra=ref_stars['ra'].values * u.deg, 
                                dec=ref_stars['dec'].values * u.deg)
            taylor_coords = SkyCoord(ra=taylor['RAJ2000'].values * u.deg, 
                                   dec=taylor['DEJ2000'].values * u.deg)
            
            # ENFOQUE SIMPLIFICADO: usar match_to_catalog_sky
            idx, d2d, _ = ref_coords.match_to_catalog_sky(taylor_coords)
            
            # Filtrar por separación máxima
            max_sep = 1.0 * u.arcsec
            valid_matches = d2d < max_sep
            
            matches_count = np.sum(valid_matches)
            print(f"  Matches encontrados: {matches_count}")
            
            if matches_count == 0:
                continue
            
            # Crear DataFrame de resultados directamente
            matches_data = []
            
            for i in range(len(ref_stars)):
                if valid_matches[i]:
                    match_row = {}
                    
                    # Añadir datos de referencia
                    for col in ref_stars.columns:
                        match_row[f'ref_{col}'] = ref_stars.iloc[i][col]
                    
                    # Añadir datos de Taylor
                    taylor_idx = idx[i]
                    for col in taylor.columns:
                        match_row[f'taylor_{col}'] = taylor.iloc[taylor_idx][col]
                    
                    # Añadir separación
                    match_row['separation_arcsec'] = d2d[i].arcsec
                    
                    matches_data.append(match_row)
            
            # Crear DataFrame final
            matches_df = pd.DataFrame(matches_data)
            
            # Eliminar duplicados (mantener el match más cercano)
            if len(matches_df) > 0:
                matches_df = matches_df.sort_values('separation_arcsec')
                matches_df = matches_df.drop_duplicates(subset=['ref_ra', 'ref_dec'], keep='first')
            
            print(f"  Matches únicos: {len(matches_df)}")
            
            # Guardar resultados
            matches_df.to_csv(output_file, index=False)
            print(f"  Guardado: {output_file}")
            
        except Exception as e:
            print(f"  Error en cross-match: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    print("Iniciando cross-match simple...")
    simple_crossmatch()
    print("¡Cross-match completado!")
