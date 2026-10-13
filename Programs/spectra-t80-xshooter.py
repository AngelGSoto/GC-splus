#!/usr/bin/env python3
"""
Genera fotospectros de cúmulos globulares de NGC 5128 combinando
fotometría T80S+DECam con espectros X-Shooter de Hernández et al. (2018).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import glob

# ------------------------------------------------------------
# CONFIGURACIÓN (ajustá estas rutas si es necesario)
# ------------------------------------------------------------
CATALOG_PATH = "Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
SPECTRA_DIR = "../Spectre-Hernandez/"   # carpeta con los 14 archivos .fits
OUTPUT_DIR = "./photospectra_with_xshooter/"
APERTURE = 3           # apertura de la fotometría T80S
ZP_AB = -2.41          # punto cero AB para convertir magnitud a flujo f_lambda
FLUX_SCALE = 1e-16     # factor para pasar a unidades de 1e-16 erg/s/cm²/Å

# Longitudes de onda efectivas (Å)
FILTER_WAVES = {
    'u': 3485, 'g': 4803, 'r': 6250, 'i': 7660, 'z': 9110,
    'F378': 3770, 'F395': 3940, 'F410': 4094, 'F430': 4292,
    'F515': 5133, 'F660': 6614, 'F861': 8611
}

# Mapeo T17ID → nombre en Hernández et al. (2018)
T17_TO_HERNANDEZ = {
    'T17-1002': 'HH-080',
    'T17-1008': 'HGHH-40',
    'T17-1202': 'HGHH-29',
    'T17-1358': 'HGHH-04',
    'T17-1287': 'HGHH-11',
    'T17-1688': 'HGHH-34',
    'T17-1805': 'HGHH-21'
}

# ------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------
def build_spectra_index(spectra_dir):
    """Indexa los archivos FITS por nombre de objeto y brazo (UVB/VIS)."""
    index = {}
    fits_files = glob.glob(os.path.join(spectra_dir, "ADP.*.fits"))
    for f in fits_files:
        with fits.open(f) as hdul:
            # La información del objeto y del brazo está en el header primario
            hdr0 = hdul[0].header
            obj = hdr0.get('OBJECT', '').strip()
            # Intentar con HIERARCH ESO SEQ ARM
            arm = hdr0.get('HIERARCH ESO SEQ ARM', '').strip()
            # Si no, usar DISPELEM (también contiene 'VIS', 'UVB', 'NIR')
            if not arm:
                arm = hdr0.get('DISPELEM', '').strip()
        if obj not in index:
            index[obj] = {}
        if arm:   # Solo guardar si pudimos identificar el brazo
            index[obj][arm] = f
        else:
            print(f"⚠️  No se pudo identificar el brazo para {f}")
    return index


def read_spectrum(fits_file):
    """Lee longitud de onda (nm -> Å) y flujo (erg/cm²/s/Å) de un espectro 1D."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        wave_nm = data['WAVE']      # nm
        flux = data['FLUX']         # erg cm⁻² s⁻¹ Å⁻¹
    wave = wave_nm * 10.0           # convertir a Angstrom
    return wave, flux


def combine_uvb_vis(uvb_file, vis_file):
    """Combina los brazos UVB y VIS en un único espectro (wave, flux)."""
    w_uvb, f_uvb = read_spectrum(uvb_file)
    w_vis, f_vis = read_spectrum(vis_file)
    # Solape típico ~5340-5560 Å; tomamos UVB hasta 5500 Å, VIS desde 5500 Å
    mask_uvb = w_uvb < 5500
    mask_vis = w_vis >= 5500
    wave_comb = np.concatenate([w_uvb[mask_uvb], w_vis[mask_vis]])
    flux_comb = np.concatenate([f_uvb[mask_uvb], f_vis[mask_vis]])
    # Aseguramos orden creciente en longitud de onda
    sort_idx = np.argsort(wave_comb)
    return wave_comb[sort_idx], flux_comb[sort_idx]


def mag_to_flux(mag, wl_ang):
    """Convierte magnitud AB a flujo en unidades 1e-16 erg/s/cm²/Å."""
    if np.isnan(mag) or mag >= 90:
        return np.nan
    flux = (10**(-0.4 * (mag + ZP_AB))) / (wl_ang**2)
    return flux / FLUX_SCALE


def mag_err_to_flux_err(mag, mag_err, wl_ang):
    """Propaga el error de magnitud a error de flujo."""
    if np.isnan(mag) or np.isnan(mag_err) or mag >= 90 or mag_err <= 0:
        return np.nan
    c = (10**(-0.4 * ZP_AB)) / (wl_ang**2) / FLUX_SCALE
    b = -0.4 * np.log(10)
    flux = c * 10**(-0.4 * mag)
    flux_err = flux * abs(b) * mag_err
    return flux_err


# ------------------------------------------------------------
# PLOT PRINCIPAL
# ------------------------------------------------------------
def plot_gc_with_spectrum(t17id, catalog, spec_index, output_dir):
    """Genera la figura combinando fotometría y espectro."""
    
    # 1. Datos del catálogo
    source = catalog[catalog['T17ID'] == t17id]
    if source.empty:
        print(f"❌ {t17id} no encontrado en el catálogo.")
        return
    source = source.iloc[0]
    
    # 2. Identificar archivos del espectro
    hernandez_name = T17_TO_HERNANDEZ.get(t17id)
    if not hernandez_name or hernandez_name not in spec_index:
        print(f"❌ No se encontraron espectros para {t17id} ({hernandez_name})")
        return
    arms = spec_index[hernandez_name]
    if 'UVB' not in arms or 'VIS' not in arms:
        print(f"❌ Faltan brazos UVB/VIS para {hernandez_name}")
        return
    
    # 3. Combinar espectro UVB + VIS
    wave_spec, flux_spec = combine_uvb_vis(arms['UVB'], arms['VIS'])
    
    # 4. Extraer fotometría
    points_wave = []
    points_flux = []
    points_err = []
    
    # DECam
    for band in ['u', 'g', 'r', 'i', 'z']:
        mag_col = f'{band}mag'
        err_col = f'e_{band}mag'
        if mag_col in source and not pd.isna(source[mag_col]):
            mag = source[mag_col]
            wl = FILTER_WAVES[band]
            flux = mag_to_flux(mag, wl)
            if err_col in source and not pd.isna(source[err_col]):
                err = mag_err_to_flux_err(mag, source[err_col], wl)
            else:
                err = 0.1 * flux
            points_wave.append(wl)
            points_flux.append(flux)
            points_err.append(err)
    
    # T80S (apertura fija)
    for band in ['F378','F395','F410','F430','F515','F660','F861']:
        mag_col = f'MAG_{band}_{APERTURE}'
        err_col = f'MAGERR_{band}_{APERTURE}'
        if mag_col in source and not pd.isna(source[mag_col]):
            mag = source[mag_col]
            wl = FILTER_WAVES[band]
            flux = mag_to_flux(mag, wl)
            if err_col in source and not pd.isna(source[err_col]):
                err = mag_err_to_flux_err(mag, source[err_col], wl)
            else:
                err = 0.1 * flux
            points_wave.append(wl)
            points_flux.append(flux)
            points_err.append(err)
    
    points_wave = np.array(points_wave)
    points_flux = np.array(points_flux)
    points_err = np.array(points_err)
    
    # 5. Escalar el espectro a la fotometría (usando banda r)
    r_mask = (wave_spec > 6200) & (wave_spec < 6300)
    median_spec_r = np.median(flux_spec[r_mask]) if np.any(r_mask) else np.median(flux_spec)
    r_wave = FILTER_WAVES['r']
    idx_r = np.argmin(np.abs(points_wave - r_wave))
    flux_r_photo = points_flux[idx_r]
    scale_factor = flux_r_photo / median_spec_r
    flux_spec_scaled = flux_spec * scale_factor
    
    # 6. Graficar
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Espectro
    ax.plot(wave_spec, flux_spec_scaled, color='grey', alpha=0.7, lw=1.2,
            label='X-Shooter (Hernández+18)', zorder=2)
    
    # Colores
    colors_decam = {'u':"#CC00FF", 'g':"#006600", 'r':"#FF0000", 'i':"#990033", 'z':"#330034"}
    colors_t80s = {'F378':"#9900FF", 'F395':"#6600FF", 'F410':"#0000FF",
                   'F430':"#009999", 'F515':"#DD8000", 'F660':"#CC0066", 'F861':"#660033"}
    
    # DECam (cuadrados)
    for band in ['u','g','r','i','z']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(points_wave, wl))[0]
        if len(idx) > 0:
            ax.errorbar(wl, points_flux[idx], yerr=points_err[idx],
                        fmt='s', color=colors_decam[band], markeredgecolor='k',
                        markersize=10, capsize=3, zorder=4)
    
    # T80S (círculos)
    for band in ['F378','F395','F410','F430','F515','F660','F861']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(points_wave, wl))[0]
        if len(idx) > 0:
            ax.errorbar(wl, points_flux[idx], yerr=points_err[idx],
                        fmt='o', color=colors_t80s[band], markeredgecolor='k',
                        markersize=10, capsize=3, zorder=4)
    
    # ---------- Ajuste del eje Y ----------
    if len(points_flux) > 0:
        # Tomar el punto fotométrico más alto como referencia
        ymax_photo = 1.15 * np.nanmax(points_flux)
        # Calcular un límite robusto a partir del espectro (percentil 95)
        ymax_spec = np.nanpercentile(flux_spec_scaled, 95) * 305.2
        # Usar el menor de los dos para evitar que el espectro domine la escala
        ymax = min(ymax_photo, ymax_spec)
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(bottom=0)
    # ----------------------------------------
    
    # Formato
    ax.set_xlabel('Wavelength (Å)', fontsize=14)
    ax.set_ylabel('Flux (10⁻¹⁶ erg s⁻¹ cm⁻² Å⁻¹)', fontsize=14)
    ax.set_xlim(3000, 9300)
    ax.text(0.02, 0.98, f'{t17id}  =  {hernandez_name}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'photospectrum_{t17id}_with_spectrum.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {t17id} → {outfile}")


# ------------------------------------------------------------
# EJECUCIÓN
# ------------------------------------------------------------
if __name__ == "__main__":
    cat = pd.read_csv(CATALOG_PATH)
    spec_idx = build_spectra_index(SPECTRA_DIR)
    print("Espectros encontrados:")
    for obj, arms in spec_idx.items():
        print(f"  {obj}: {list(arms.keys())}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for t17id in T17_TO_HERNANDEZ.keys():
        plot_gc_with_spectrum(t17id, cat, spec_idx, OUTPUT_DIR)
