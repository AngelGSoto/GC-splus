#!/usr/bin/env python3
"""
Fotospectros GCs NGC 5128 con X-Shooter + fotometría T80S/DECam.
Conversión AB corregida, leyenda limpia, suavizado y ejes robustos (percentiles).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from scipy.ndimage import uniform_filter1d
import os, glob

# ============================================================
# PARÁMETROS AJUSTABLES
# ============================================================
SMOOTH_WINDOW = 20          # suavizado ligero (~4 Å)
YMIN_FACTOR = 0.0          # margen inferior (0 = empieza en 0)
YMAX_FACTOR = 1.55          # 25% de margen arriba del percentil 99
YMIN_USER = -1.4           # fijar manual (ej. 0) si no se usa automático
YMAX_USER = 4        # fijar manual (ej. 8000) si no se usa automático

CATALOG_PATH = "Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv"
SPECTRA_DIR = "../Spectre-Hernandez/"
OUTPUT_DIR = "./photospectra_with_xshooter/"
APERTURE = 3
ZP_AB = 2.41                # positivo (AB)
FLUX_SCALE = 1e-16

FILTER_WAVES = {
    'u': 3485, 'g': 4803, 'r': 6250, 'i': 7660, 'z': 9110,
    'F378': 3770, 'F395': 3940, 'F410': 4094, 'F430': 4292,
    'F515': 5133, 'F660': 6614, 'F861': 8611
}

T17_TO_HERNANDEZ = {
    'T17-1002': 'HH-080', 'T17-1008': 'HGHH-40',
    'T17-1202': 'HGHH-29', 'T17-1358': 'HGHH-04',
    'T17-1287': 'HGHH-11', 'T17-1688': 'HGHH-34',
    'T17-1805': 'HGHH-21'
}

# ============================================================
# FUNCIONES
# ============================================================
def build_spectra_index(spectra_dir):
    index = {}
    for f in glob.glob(os.path.join(spectra_dir, "ADP.*.fits")):
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
        wave_nm = data['WAVE']
        flux = data['FLUX']       # erg cm⁻² s⁻¹ Å⁻¹
    wave = wave_nm * 10.0
    return wave, flux

def combine_uvb_vis(uvb_file, vis_file):
    w_uvb, f_uvb = read_spectrum(uvb_file)
    w_vis, f_vis = read_spectrum(vis_file)
    mask_uvb = w_uvb < 5500
    mask_vis = w_vis >= 5500
    wave = np.concatenate([w_uvb[mask_uvb], w_vis[mask_vis]])
    flux = np.concatenate([f_uvb[mask_uvb], f_vis[mask_vis]])
    sort = np.argsort(wave)
    return wave[sort], flux[sort]

def mag_to_flux(mag, wl_ang):
    if np.isnan(mag) or mag >= 90:
        return np.nan
    flux = 10**(-0.4 * (mag + ZP_AB)) / (wl_ang**2)   # erg cm⁻² s⁻¹ Å⁻¹
    return flux / FLUX_SCALE                           # en 1e-16

def mag_err_to_flux_err(mag, mag_err, wl_ang):
    if np.isnan(mag) or np.isnan(mag_err) or mag >= 90 or mag_err <= 0:
        return np.nan
    flux = 10**(-0.4 * (mag + ZP_AB)) / (wl_ang**2)
    flux_err = flux * 0.4 * np.log(10) * mag_err
    return flux_err / FLUX_SCALE

# ============================================================
# PLOT PRINCIPAL (con límites robustos)
# ============================================================
def plot_gc_with_spectrum(t17id, catalog, spec_index, output_dir):
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

    wave_spec, flux_spec = combine_uvb_vis(arms['UVB'], arms['VIS'])
    flux_smoothed = uniform_filter1d(flux_spec, size=SMOOTH_WINDOW)

    # Fotometría
    pw, pf, pe = [], [], []
    for band in ['u','g','r','i','z']:
        mag_col = f'{band}mag'
        err_col = f'e_{band}mag'
        if mag_col in source and not pd.isna(source[mag_col]):
            mag = source[mag_col]
            wl = FILTER_WAVES[band]
            flux = mag_to_flux(mag, wl)
            err = mag_err_to_flux_err(mag, source[err_col], wl) if (err_col in source and not pd.isna(source[err_col])) else 0.1*flux
            pw.append(wl); pf.append(flux); pe.append(err)

    for band in ['F378','F395','F410','F430','F515','F660','F861']:
        mag_col = f'MAG_{band}_{APERTURE}'
        err_col = f'MAGERR_{band}_{APERTURE}'
        if mag_col in source and not pd.isna(source[mag_col]):
            mag = source[mag_col]
            wl = FILTER_WAVES[band]
            flux = mag_to_flux(mag, wl)
            err = mag_err_to_flux_err(mag, source[err_col], wl) if (err_col in source and not pd.isna(source[err_col])) else 0.1*flux
            pw.append(wl); pf.append(flux); pe.append(err)

    pw = np.array(pw); pf = np.array(pf); pe = np.array(pe)

    # Reescalado a la banda r
    r_mask = (wave_spec > 6200) & (wave_spec < 6300)
    med_spec_r = np.median(flux_smoothed[r_mask]) if np.any(r_mask) else np.median(flux_smoothed)
    r_wave = FILTER_WAVES['r']
    idx_r = np.argmin(np.abs(pw - r_wave))
    flux_r_photo = pf[idx_r]
    scale = flux_r_photo / med_spec_r
    flux_scaled = flux_smoothed * scale

    # Figura
    fig, ax = plt.subplots(figsize=(12,6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(wave_spec, flux_scaled, color='grey', alpha=0.4, lw=1.0,
            label='X-Shooter (Hernández+18)', zorder=2)

    colors_decam = {'u':"#CC00FF", 'g':"#006600", 'r':"#FF0000", 'i':"#990033", 'z':"#330034"}
    colors_t80s = {'F378':"#9900FF", 'F395':"#6600FF", 'F410':"#0000FF",
                   'F430':"#009999", 'F515':"#DD8000", 'F660':"#CC0066", 'F861':"#660033"}

    for band in ['u','g','r','i','z']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(pw, wl))[0]
        if len(idx)>0:
            ax.errorbar(wl, pf[idx], yerr=pe[idx], fmt='s', color=colors_decam[band],
                        markeredgecolor='k', markersize=15, capsize=5, zorder=5)

    for band in ['F378','F395','F410','F430','F515','F660','F861']:
        wl = FILTER_WAVES[band]
        idx = np.where(np.isclose(pw, wl))[0]
        if len(idx)>0:
            ax.errorbar(wl, pf[idx], yerr=pe[idx], fmt='o', color=colors_t80s[band],
                        markeredgecolor='k', markersize=15, capsize=13, zorder=5)

    # ---- Ajuste robusto del eje Y (ignora outliers) ----
    if YMIN_USER is not None and YMAX_USER is not None:
        ax.set_ylim(YMIN_USER, YMAX_USER)
    else:
        all_vals = np.concatenate([pf, flux_scaled])
        # Percentiles para evitar que un pico aislado estire la escala
        vmin = np.nanpercentile(all_vals, 1)     # ignora el 1% inferior
        vmax = np.nanpercentile(all_vals, 99)    # ignora el 1% superior
        dy = vmax - vmin if vmax > vmin else 1.0
        if YMIN_FACTOR > 0:
            ymin = vmin - YMIN_FACTOR * dy
            if ymin < 0 and vmin >= 0:
                ymin = 0
        else:
            ymin = 0
        ymax = vmax * YMAX_FACTOR
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Wavelength (Å)', fontsize=24)
    ax.set_ylabel(r'F$_\lambda$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=24)
    ax.set_xlim(3000, 10000)

    # Leyenda limpia
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='k', markeredgewidth=1.5, markersize=12,
               label='T80S narrow bands'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='w',
               markeredgecolor='k', markeredgewidth=1.5, markersize=12,
               label='DECam broad bands'),
        Line2D([0], [0], color='grey', lw=1.0, label='X-Shooter (Hernández+18)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=24,
              frameon=True, framealpha=0.9, edgecolor='black', facecolor='white')

    ax.text(0.02, 0.98, f'{t17id}  ({hernandez_name})', transform=ax.transAxes,
            fontsize=24, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))
    # Larger y-tick labels
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='x', labelsize=24)
    
    # Improve y-axis scaling
    #ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(output_dir, f'photospectrum_{t17id}_with_spectrum.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {t17id} → {out}")

# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    cat = pd.read_csv(CATALOG_PATH)
    spec_idx = build_spectra_index(SPECTRA_DIR)
    print("Espectros encontrados:")
    for obj, arms in spec_idx.items():
        print(f"  {obj}: {list(arms.keys())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for t17id in T17_TO_HERNANDEZ.keys():
        plot_gc_with_spectrum(t17id, cat, spec_idx, OUTPUT_DIR)
