#!/usr/bin/env python3
# 
# Corrección por extinción para campos S-PLUS personalizados (no main survey)

import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

# ============================================================================
# CONFIGURACIÓN - PARA CAMPOS PERSONALIZADOS S-PLUS
# ============================================================================

INPUT_FILE = "Results_Corrected/all_fields_photometry_COMPLETE.csv"
OUTPUT_FILE = INPUT_FILE.replace('.csv', '_extinction.csv')

# IMPORTANTE: Aunque mis campos no sean del main survey,
# el SISTEMA INSTRUMENTAL es el MISMO (T80-S en Cerro Tololo)
# Por lo tanto, los coeficientes oficiales S-PLUS SON APLICABLES

R_V = 3.1  # Valor estándar para la Vía Láctea

# ============================================================================
# COEFICIENTES OFICIALES S-PLUS - SON VÁLIDOS PARA TUS DATOS
# ============================================================================

# VALORES OFICIALES S-PLUS - Fitzpatrick (1999) - Tabla A.1
# Estos coeficientes son PROPIEDADES DEL SISTEMA INSTRUMENTAL:
# - Mismos filtros físicos (anchos y estrechos)
# - Mismo telescopio (SOAR)
# - Misma cámara (T80-S)
# - Misma ubicación (Cerro Tololo, misma atmósfera)
# Por lo tanto, SON APLICABLES a tus campos personalizados.

A_over_AV_SPLUS = {
    # Filtros estrechos S-PLUS - MISMO sistema instrumental, see Herpich et al. (2024)
    'F0378': 1.518, 'F0395': 1.459, 'F0410': 1.403,
    'F0430': 1.334, 'F0515': 1.098, 'F0660': 0.798, 'F0861': 0.539,
    # Filtros anchos S-PLUS - MISMO sistema instrumental
    'SPLUS_u': 1.610, 'SPLUS_g': 1.199, 'SPLUS_r': 0.864,
    'SPLUS_i': 0.648, 'SPLUS_z': 0.512
}

# VALORES PARA DECam/SDSS - Schlafly & Finkbeiner (2011)
A_over_AV_DECAM = {
    'DECAM_u': 1.569, 'DECAM_g': 1.214, 'DECAM_r': 0.858,
    'DECAM_i': 0.634, 'DECAM_z': 0.491
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def setup_dustmaps():
    """Configura dustmaps para usar mapa SFD (Schlegel et al. 1998)."""
    data_dir = os.path.expanduser('~/dustmaps_data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    config['data_dir'] = data_dir
    config_file = os.path.expanduser('~/.dustmapsrc')
    if not os.path.exists(config_file):
        config.reset()
    
    sfd_path = os.path.join(data_dir, 'sfd')
    if not os.path.exists(sfd_path):
        print("Descargando mapas de polvo SFD (~130 MB)...")
        from dustmaps.sfd import fetch
        fetch()
        print("✅ Mapas descargados.")

def add_extinction_correction(df, ra_col='RAJ2000', dec_col='DEJ2000'):
    """
    Corrección por extinción para datos S-PLUS de campos personalizados.
    
    NOTA CRÍTICA: Aunque tus datos no son del main survey de S-PLUS,
    el SISTEMA INSTRUMENTAL es IDÉNTICO (filtros, telescopio, cámara, ubicación).
    Por lo tanto, los coeficientes oficiales S-PLUS son aplicables.
    """
    print("="*70)
    print("CORRECCIÓN POR EXTINCIÓN - CAMPOS PERSONALIZADOS S-PLUS")
    print("="*70)
    print("TU CASO: Datos S-PLUS de campos fuera del main survey")
    print("RAZÓN: Mismo sistema instrumental → mismos coeficientes")
    print("="*70)
    
    # Calcular E(B-V) usando SFD map
    coords = SkyCoord(ra=df[ra_col].values * u.deg, 
                      dec=df[dec_col].values * u.deg, 
                      frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)
    
    # Calcular A_V = R_V × E(B-V)
    A_V = R_V * ebv
    
    # Agregar metadatos de extinción
    df['E_BV_SFD'] = ebv
    df['A_V'] = A_V
    df['R_V'] = R_V  # Para documentación
    
    print(f"\n📊 ESTADÍSTICAS DE EXTINCIÓN PARA TUS CAMPOS:")
    print(f"   • Objetos: {len(df)}")
    print(f"   • E(B-V) mínimo: {ebv.min():.4f}")
    print(f"   • E(B-V) máximo: {ebv.max():.4f}")
    print(f"   • E(B-V) mediano: {np.median(ebv):.4f}")
    print(f"   • A_V mediano: {np.median(A_V):.4f}")

    # 1. CORREGIR DATOS S-PLUS (tus datos personalizados)
    print("\n🔧 APLICANDO COEFICIENTES S-PLUS OFICIALES:")
    print("   (Válidos porque usas el MISMO sistema instrumental)")
    
    splus_filters = {
        'F0378': 'F378', 'F0395': 'F395', 'F0410': 'F410',
        'F0430': 'F430', 'F0515': 'F515', 'F0660': 'F660', 'F0861': 'F861'
    }
    
    for cigale_name, col_suffix in splus_filters.items():
        # Calcular A_λ usando coeficientes OFICIALES S-PLUS
        A_lambda = A_over_AV_SPLUS[cigale_name] * A_V
        correction_factor = 10**(0.4 * A_lambda)
        
        # Columnas para versión _3 (3 arcsec)
        flux_col = f'FLUX_{col_suffix}_3'
        fluxerr_col = f'FLUXERR_{col_suffix}_3'
        mag_col = f'MAG_{col_suffix}_3'
        magerr_col = f'MAGERR_{col_suffix}_3'
        
        if flux_col in df.columns:
            # Aplicar corrección
            df[f'{flux_col}_corr'] = df[flux_col] * correction_factor
            
            if fluxerr_col in df.columns:
                df[f'{fluxerr_col}_corr'] = df[fluxerr_col] * correction_factor
            
            if mag_col in df.columns:
                df[f'{mag_col}_corr'] = df[mag_col] - A_lambda
            
            if magerr_col in df.columns:
                df[f'{magerr_col}_corr'] = df[magerr_col]
            
            # Documentar coeficientes usados
            df[f'A_{cigale_name}'] = A_lambda
            df[f'A{cigale_name}_over_AV'] = A_over_AV_SPLUS[cigale_name]
            df[f'A{cigale_name}_source'] = 'SPLUS_official_TabA1'
            
            print(f"   ✓ {cigale_name}: A_λ/A_V={A_over_AV_SPLUS[cigale_name]:.3f}")
    
    # 2. CORREGIR DATOS DECam (de Taylor et al.)
    print("\n🔧 APLICANDO COEFICIENTES DECam (SDSS):")
    
    for band in ['u', 'g', 'r', 'i', 'z']:
        mag_col = f'{band}mag'
        err_col = f'e_{band}mag'
        coeff_key = f'DECAM_{band}'
        
        if mag_col in df.columns and coeff_key in A_over_AV_DECAM:
            A_lambda = A_over_AV_DECAM[coeff_key] * A_V
            
            df[f'{mag_col}_corr'] = df[mag_col] - A_lambda
            
            if err_col in df.columns:
                df[f'{err_col}_corr'] = df[err_col]
            
            df[f'A_{band}'] = A_lambda
            df[f'A{band}_over_AV'] = A_over_AV_DECAM[coeff_key]
            df[f'A{band}_source'] = 'DECAM_SF2011'
            
            print(f"   ✓ DECam {band}: A_λ/A_V={A_over_AV_DECAM[coeff_key]:.3f}")
    
    return df

def generate_methodology_text(df):
    """Genera texto listo para copiar en la sección de métodos del paper."""
    print("\n" + "="*70)
    print("📝 TEXTO PARA SECCIÓN DE MÉTODOS DE TU PAPER")
    print("="*70)
    
    methodology = f"""
METHODOLOGY - EXTINCTION CORRECTION FOR CUSTOM S-PLUS FIELDS

We correct all photometry for Galactic extinction using the SFD dust map 
(Schlegel, Finkbeiner & Davis 1998). For the S-PLUS narrow-band filters, 
we apply the A_λ/A_V coefficients from the official S-PLUS instrumental 
characterization (Herpich et al. 2024, Table A.1) using the 
Fitzpatrick (1999) extinction law with R_V=3.1. Although our S-PLUS 
observations target fields outside the main S-PLUS survey, the instrumental 
system (T80 telescope at CTIO, T80Cam camera, filters, and Cerro Tololo site) is 
identical, making these coefficients fully applicable. For the DECam broad-band 
filters from Taylor et al. (2021), we use the SDSS-specific coefficients from 
Schlafly & Finkbeiner (2011).

Our sample of {len(df)} globular cluster candidates in NGC 5128 shows 
E(B-V) values ranging from {df['E_BV_SFD'].min():.3f} to {df['E_BV_SFD'].max():.3f}, 
with a median of {df['E_BV_SFD'].median():.3f}, corresponding to a median 
A_V = {df['A_V'].median():.2f} mag.
"""
    
    print(methodology)
    
    # También mostrar diferencias típicas
    if 'F0378' in A_over_AV_SPLUS and 'DECAM_u' in A_over_AV_DECAM:
        print("\nNOTE ON CORRECTION MAGNITUDES:")
        print(f"For a typical E(B-V) = {df['E_BV_SFD'].median():.3f}:")
        for band, splus_key, decam_key in [('u', 'SPLUS_u', 'DECAM_u'),
                                           ('g', 'SPLUS_g', 'DECAM_g')]:
            if splus_key in A_over_AV_SPLUS and decam_key in A_over_AV_DECAM:
                A_splus = A_over_AV_SPLUS[splus_key] * df['A_V'].median()
                A_decam = A_over_AV_DECAM[decam_key] * df['A_V'].median()
                diff = A_splus - A_decam
                print(f"  • {band}-band: ΔA_λ(S-PLUS - DECam) = {diff:.3f} mag")

def main():
    """Función principal."""
    print("="*70)
    print("CORRECCIÓN EXTINCIÓN - CAMPOS S-PLUS PERSONALIZADOS")
    print("="*70)
    print("Aunque tus datos no sean del main survey de S-PLUS,")
    print("el SISTEMA INSTRUMENTAL es IDÉNTICO → coeficientes VÁLIDOS")
    print("="*70)
    
    # 1. Configurar dustmaps
    setup_dustmaps()
    
    # 2. Leer catálogo
    print(f"\n📖 Leyendo tus datos de campos personalizados: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   • Objetos en tus campos: {len(df)}")
    
    # 3. Aplicar corrección
    df_corr = add_extinction_correction(df)
    
    # 4. Generar texto para paper
    generate_methodology_text(df_corr)
    
    # 5. Guardar archivo
    print(f"\n💾 Guardando archivo corregido: {OUTPUT_FILE}")
    df_corr.to_csv(OUTPUT_FILE, index=False)
    
    # 6. Resumen final
    print("\n" + "="*70)
    print("✅ CORRECCIÓN COMPLETADA - COEFICIENTES VÁLIDOS")
    print("="*70)
    print("\n📊 RESUMEN:")
    print(f"   • Archivo de entrada: {INPUT_FILE}")
    print(f"   • Archivo de salida: {OUTPUT_FILE}")
    print(f"   • Columnas totales: {len(df_corr.columns)}")
    
    # Contar tipos de columnas
    original_cols = len([c for c in df_corr.columns if '_corr' not in c and 
                         not c.startswith('A_') and c not in ['E_BV_SFD', 'A_V', 'R_V']])
    corr_cols = len([c for c in df_corr.columns if '_corr' in c])
    ext_info_cols = len([c for c in df_corr.columns if c.startswith('A_') or 
                        c in ['E_BV_SFD', 'A_V', 'R_V']])
    
    print(f"   • Columnas originales: {original_cols}")
    print(f"   • Columnas '_corr' (valores corregidos): {corr_cols}")
    print(f"   • Columnas de info de extinción: {ext_info_cols}")
    
    print("\n🔍 PARA USAR EN TU ANÁLISIS:")
    print("   Usa las columnas con sufijo '_corr' para:")
    print("   • S-PLUS: 'FLUX_F378_3_corr', 'FLUXERR_F378_3_corr'")
    print("   • DECam: 'umag_corr', 'e_umag_corr' (luego convertir a flujos)")
    
    print("\n⚠️  RECORDATORIO IMPORTANTE:")
    print("   Los coeficientes S-PLUS SÍ son aplicables porque:")
    print("   1. Mismos filtros físicos")
    print("   2. Mismo telescopio (SOAR)")
    print("   3. Misma cámara (T80-S)")
    print("   4. Misma ubicación (Cerro Tololo)")
    print("   Solo la reducción/calibración es diferente, no el sistema instrumental.")

if __name__ == "__main__":
    main()
