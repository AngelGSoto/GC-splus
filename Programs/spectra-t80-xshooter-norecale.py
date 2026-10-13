#!/usr/bin/env python3
"""
Fotospectros sin reescalado:
Comparación directa entre fotometría T80S+DECam (magnitudes AB corregidas a flujo absoluto)
y espectros X-Shooter (flujo absoluto), sin normalización artificial.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import uniform_filter1d
import os, glob

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
CATALOG_PATH = "Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
SPECTRA_DIR = "../Spectre-Hernandez/"
OUTPUT_DIR = "./photospectra_no_rescale/"
APERTURE = 3
FLUX_SCALE = 1e-16   # para mostrar los valores en 10⁻¹⁶ erg/s/cm²/Å

# Punto cero AB: f_λ = 10^(-0.4*(mag + 2.41)) / λ²
ZP_AB = 2.41          # ¡positivo, se usa como (mag + ZP_AB)!

FILTER_WAVES = {
    'u': 3485, 'g': 4803, 'r': 6250, 'i': 7660, 'z': 9110,
    'F378': 3770, 'F395': 3940, 'F410': 4094, 'F430': 4292,
    'F515': 5133, 'F660': 6614, 'F861': 8611
}

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
    index = {}
    fits_files = glob.glob(os.path.join(spectra_dir, "ADP.*.fits"))
    for f in fits_files:
        with fits.open(f) as hdul:
            hdr0 = hdul[0].header
            obj = hdr0.get('OBJECT', '').strip()
            arm = hdr0.get('HIERARCH ESO SEQ ARM', '').strip()
            if not arm:
                arm = hdr0.get('DISPELEM', '').strip()
        if obj not in index:
            index[obj] = {}
        if arm:
            index[obj][arm] = f
        else:
            print(f"⚠️  No se pudo identificar el brazo para {f}")
    return index

def read_spectrum(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        wave_nm = data['WAVE']      # nm
        flux = data['FLUX']         # erg cm⁻² s⁻¹ Å⁻¹
    wave = wave_nm * 10.0
    return wave, flux

def combine_uvb_vis(uvb_file, vis_file):
    w_uvb, f_uvb = read_spectrum(uvb_file)
    w_vis, f_vis = read_spectrum(vis_file)
    mask_uvb = w_uvb < 5500
    mask_vis = w_vis >= 5500
    wave_comb = np.concatenate([w_uvb[mask_uvb], w_vis[mask_vis]])
    flux_comb = np.concatenate([f_uvb[mask_uvb], f_vis[mask_vis]])
    sort_idx = np.argsort(wave_comb)
    return wave_comb[sort_idx], flux_comb[sort_idx]

def mag_to_flux(mag, wl_ang):
    """Convierte magnitud AB a flujo físico erg/s/cm²/Å (se divide entre 1e-16 para la gráfica)."""
    if np.isnan(mag) or mag >= 90:
        return np.nan
    flux = 10**(-0.4 * (mag + ZP_AB)) / (wl_ang**2)   # erg/s/cm²/Å
    return flux / FLUX_SCALE                           # ahora en 1e-16 erg/s/cm²/Å

def mag_err_to_flux_err(mag, mag_err, wl_ang):
    if np.isnan(mag) or np.isnan(mag_err) or mag >= 90 or mag_err <= 0:
        return np.nan
    flux = 10**(-0.4 * (mag + ZP_AB)) / (wl_ang**2)
    flux_err = flux * (0.4 * np.log(10) * mag_err)   # propagación correcta
    return flux_err / FLUX_SCALE

# ------------------------------------------------------------
# PLOT SIN REESCALAR
# ------------------------------------------------------------
def plot_gc_with_spectrum_noscale(t17id, catalog, spec_index, output_dir):
    source = catalog[catalog['T17ID'] == t17id]
    if source.empty:
        print(f"❌ {t17id} no encontrado.")
        return
    source = source.iloc[0]

    hernandez_name = T17_TO_HERNANDEZ.get(t17id)
    if not hernandez_name or hernandez_name not in spec_index:
        print(f"❌ No espectros para {t17id}")
        return
    arms = spec_index[hernandez_name]
    if 'UVB' not in arms or 'VIS' not in arms:
        print(f"❌ Faltan UVB/VIS para {hernandez_name}")
        return

    # Espectro combinado (sin tocar)
    wave_spec, flux_spec = combine_uvb_vis(arms['UVB'], arms['VIS'])
    # Suavizado para visualización
    flux_smoothed = uniform_filter1d(flux_spec, size=100)

    # Fotometría
    points_wave, points_flux, points_err = [], [], []
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

    # Figura
    fig, ax = plt.subplots(figsize=(12,6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Espectro suavizado (sin reescalar, con sus unidades propias)
    ax.plot(wave_spec, flux_smoothed, color='grey', alpha=0.45, lw=0.6,
            label='X-Shooter (Hernández+18)')

    colors_decam = {'u':"#CC00FF", 'g':"#006600", 'r':"#FF0000", 'i':"#990033", 'z':"#330034"}
    colors_t80s = {'F378':"#9900FF", 'F395':"#6600FF", 'F410':"#0000FF",
                   'F430':"#009999", 'F515':"#DD8000", 'F660':"#CC0066", 'F861':"#660033"}

    for band in ['u','g','r','i','z']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(points_wave, wl))[0]
        if len(idx)>0:
            ax.errorbar(wl, points_flux[idx], yerr=points_err[idx],
                        fmt='s', color=colors_decam[band], markeredgecolor='k',
                        markersize=10, capsize=3, zorder=5)

    for band in ['F378','F395','F410','F430','F515','F660','F861']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(points_wave, wl))[0]
        if len(idx)>0:
            ax.errorbar(wl, points_flux[idx], yerr=points_err[idx],
                        fmt='o', color=colors_t80s[band], markeredgecolor='k',
                        markersize=10, capsize=3, zorder=5)

    # Límite vertical basado en los puntos fotométricos (para que no se aplaste la gráfica)
    if len(points_flux) > 0:
        ymax = 1.15 * np.nanmax(points_flux)
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(bottom=0)

    ax.set_xlabel('Wavelength (Å)', fontsize=14)
    ax.set_ylabel('Flux (10⁻¹⁶ erg s⁻¹ cm⁻² Å⁻¹)', fontsize=14)
    ax.set_xlim(3000, 10000)
    ax.text(0.02, 0.98, f'{t17id}  =  {hernandez_name}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))
    ax.legend(loc='upper right')
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'photospectrum_{t17id}_noscale.pdf')
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
        plot_gc_with_spectrum_noscale(t17id, cat, spec_idx, OUTPUT_DIR)
