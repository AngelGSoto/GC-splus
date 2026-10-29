#!/usr/bin/env python3
"""
Convert X-Shooter (Hernández+18) FITS spectra to ASCII .dat files
compatible with the magnitude-estimation script.

Output (ASCII):
    # wl(Angstrom) flux(erg cm-2 s-1 A-1)
    <wl> <flux>
"""

import numpy as np
import os
import glob
from astropy.io import fits

# ------------------------------------------------------------------
SPECTRA_DIR = "../Spectre-Hernandez/"   # carpeta con ADP.*.fits
OUTPUT_DIR  = "./spectra_ascii/"        # carpeta de salida .dat
COMBINE_ARMS = True                     # unir UVB+VIS
SPLIT_POINT  = 5500.0                   # Å (UVB < split, VIS >= split)
WAVE_FACTOR  = 10.0                     # nm -> Å (los FITS vienen en nm)
# ------------------------------------------------------------------


def build_spectra_index(spectra_dir):
    """Agrupa los FITS por OBJECT y brazo (UVB/VIS)."""
    index = {}
    for f in sorted(glob.glob(os.path.join(spectra_dir, "ADP.*.fits"))):
        with fits.open(f) as hdul:
            hdr0 = hdul[0].header
            obj = hdr0.get('OBJECT', '').strip()
            arm = hdr0.get('HIERARCH ESO SEQ ARM', '').strip()
            if not arm:
                arm = hdr0.get('DISPELEM', '').strip()
        if not obj:
            print(f"WARNING: OBJECT vacío en {f}")
            continue
        index.setdefault(obj, {})
        if arm:
            index[obj][arm] = f
        else:
            print(f"WARNING: sin brazo en {f}")
    return index


def read_spectrum(fits_file):
    """Devuelve (wave_A, flux) limpios, ordenados y sin NaN."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        wave = np.asarray(data['WAVE'], dtype=float) * WAVE_FACTOR
        flux = np.asarray(data['FLUX'], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[good], flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order]


def combine_uvb_vis(uvb_file, vis_file, split=SPLIT_POINT):
    """Une UVB (<split) y VIS (>=split) en un único espectro."""
    w_uvb, f_uvb = read_spectrum(uvb_file)
    w_vis, f_vis = read_spectrum(vis_file)
    wave = np.concatenate([w_uvb[w_uvb <  split], w_vis[w_vis >= split]])
    flux = np.concatenate([f_uvb[w_uvb <  split], f_vis[w_vis >= split]])
    order = np.argsort(wave)
    return wave[order], flux[order]


def save_ascii(name, wave, flux, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}.dat")
    with open(out, "w") as fh:
        fh.write("# wl(Angstrom) flux(erg cm-2 s-1 A-1)\n")
        for w, fl in zip(wave, flux):
            fh.write(f"{w:.4f} {fl:.8e}\n")
    print(f"OK {name}: {len(wave)} px  ->  {out}")


def main():
    spec_idx = build_spectra_index(SPECTRA_DIR)
    print(f"Encontrados {len(spec_idx)} objetos:")
    for obj, arms in spec_idx.items():
        print(f"  {obj}: {sorted(arms.keys())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for obj, arms in spec_idx.items():
        if COMBINE_ARMS and 'UVB' in arms and 'VIS' in arms:
            wave, flux = combine_uvb_vis(arms['UVB'], arms['VIS'])
            save_ascii(obj, wave, flux, OUTPUT_DIR)
        else:
            # Sin unión: guarda cada brazo por separado
            for arm, fpath in arms.items():
                wave, flux = read_spectrum(fpath)
                save_ascii(f"{obj}_{arm}", wave, flux, OUTPUT_DIR)


if __name__ == "__main__":
    main()
