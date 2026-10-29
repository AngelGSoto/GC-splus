#!/usr/bin/env python3
"""
photospectra_synthetic_v2.py -- Fig. 1 con fotometria sintetica del espectro X-Shooter
(respuesta al arbitro: convolucionar el espectro con las curvas de los filtros y comparar
con la fotometria observada).

Cambios respecto de la version anterior:
  * Bandas anchas: la sintetica se calcula convolucionando el espectro con las curvas de
    DECam (Filter_decam/), no con las S-PLUS (uJAVA, gSDSS, ...) del JSON.
  * Bandas angostas: se usan las magnitudes sinteticas del JSON (mismos filtros S-PLUS).
  * Normalizacion: el espectro y la sintetica se escalan para que la sintetica coincida con
    la fotometria observada en la banda r (integrada en la banda, no en un punto).
  * Panel inferior con los residuos Dm = m_obs - m_sint en cada banda.
  * Tabla con los residuos de los 7 GCs y resumen (mediana y dispersion por banda).

Ni la fotometria ni el espectro estan corregidos por extincion galactica (como en la Fig. 1).

Uso (desde anac_data, como el script anterior):
  python ../Programs/photospectra_synthetic_v2.py
  python ../Programs/photospectra_synthetic_v2.py --decam-prefix Filter_decam/DECam_
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

_trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy >= 2.0 / < 2.0

SMOOTH_WINDOW = 20
APERTURE = 3
FLUX_SCALE = 1e-16
C_AA = 2.99792458e18          # velocidad de la luz en A/s

NARROW = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
BROAD = ['u', 'g', 'r', 'i', 'z']
WAVE_EFF = {'u': 3543, 'g': 4770, 'r': 6231, 'i': 7625, 'z': 9134,     # Tabla A.1
            'F378': 3770, 'F395': 3940, 'F410': 4094, 'F430': 4292,
            'F515': 5133, 'F660': 6614, 'F861': 8611}
WAVE_ORIG = dict(WAVE_EFF, u=3485, g=4803, r=6250, i=7660, z=9110)   # script original
JSON_NARROW = {'F0378': 'F378', 'F0395': 'F395', 'F0410': 'F410', 'F0430': 'F430',
               'F0515': 'F515', 'F0660': 'F660', 'F0861': 'F861'}
T17_TO_HERNANDEZ = {'T17-1002': 'HH-080', 'T17-1008': 'HGHH-40', 'T17-1202': 'HGHH-29',
                    'T17-1358': 'HGHH-04', 'T17-1287': 'HGHH-11', 'T17-1688': 'HGHH-34',
                    'T17-1805': 'HGHH-21'}
COL_DECAM = {'u': "#CC00FF", 'g': "#006600", 'r': "#FF0000", 'i': "#990033", 'z': "#330034"}
COL_T80S = {'F378': "#9900FF", 'F395': "#6600FF", 'F410': "#0000FF", 'F430': "#009999",
            'F515': "#DD8000", 'F660': "#CC0066", 'F861': "#660033"}


# ---------------------------------------------------------------- utilidades
def mag_to_flam(mag, wl):
    """AB -> f_lambda (cgs) en la longitud de onda wl (A)."""
    return 10**(-0.4 * (mag + 48.6)) * C_AA / wl**2


def synth_ab_mag(wave, flam, fwave, ftrans):
    """Magnitud AB sintetica (detector de conteo de fotones) y fraccion de la banda cubierta."""
    T = np.interp(wave, fwave, ftrans, left=0.0, right=0.0)
    num = _trapz(flam * T * wave, wave)
    den = _trapz(T * C_AA / wave, wave)
    tot = _trapz(ftrans, fwave)
    cover = _trapz(T, wave) / tot if tot > 0 else 0.0
    if num <= 0 or den <= 0:
        return np.nan, cover
    return -2.5 * np.log10(num / den) - 48.6, cover


def load_filter(path):
    d = np.loadtxt(path, comments='#')
    w, t = d[:, 0], d[:, 1]
    if np.nanmax(w) < 2000:      # en nm -> A
        w = w * 10.0
    o = np.argsort(w)
    return w[o], np.clip(t[o], 0, None)


TELLURIC_WINDOWS = [(6860, 6960),   # O2 banda B
                    (7150, 7350),   # H2O
                    (7580, 7700),   # O2 banda A
                    (8100, 8400),   # H2O
                    (8900, 9850)]   # H2O


def mask_tellurics(wave, flux, windows=TELLURIC_WINDOWS, edge=15.0):
    """Reemplaza el flujo dentro de cada ventana telurica por una recta entre las medianas
    del continuo a ambos lados (estimacion maxima de la absorcion telurica)."""
    f = flux.copy()
    for lo, hi in windows:
        m = (wave >= lo) & (wave <= hi)
        L = (wave >= lo - edge) & (wave < lo)
        R = (wave > hi) & (wave <= hi + edge)
        if m.sum() == 0 or L.sum() == 0 or R.sum() == 0:
            continue
        yl, yr = np.median(flux[L]), np.median(flux[R])
        f[m] = yl + (yr - yl) * (wave[m] - lo) / (hi - lo)
    return f


def load_splus_narrow(path, syntphot_dir):
    """Curvas de las 7 bandas angostas S-PLUS con readfilterset (el mismo lector de syntphot)."""
    if not os.path.exists(path):
        return None
    import sys
    sys.path.insert(0, os.path.abspath(syntphot_dir))
    try:
        from readfilterset import readfilterset
    except ImportError:
        print(f"  (no se encontro readfilterset.py en {syntphot_dir}; se usa el JSON)")
        return None
    fs = readfilterset()
    fs.read(path)
    fs.uniform()
    out = {}
    for fid in np.unique(fs.filterset['ID_filter']):
        key = fid.decode() if isinstance(fid, bytes) else str(fid)
        if key in JSON_NARROW:
            sel = fs.filterset[fs.filterset['ID_filter'] == fid]
            o = np.argsort(sel['wl'])
            out[JSON_NARROW[key]] = (np.asarray(sel['wl'])[o], np.asarray(sel['transm'])[o])
    return out if len(out) == 7 else None


def build_spectra_index(spectra_dir):
    index = {}
    for f in glob.glob(os.path.join(spectra_dir, "ADP.*.fits")):
        with fits.open(f) as h:
            hdr = h[0].header
            obj = hdr.get('OBJECT', '').strip()
            arm = hdr.get('HIERARCH ESO SEQ ARM', '').strip() or hdr.get('DISPELEM', '').strip()
        if arm:
            index.setdefault(obj, {})[arm] = f
    return index


def read_spectrum(f):
    with fits.open(f) as h:
        d = h[1].data
        return np.ravel(d['WAVE']) * 10.0, np.ravel(d['FLUX'])


def combine_uvb_vis(uvb, vis, cut=5500.0):
    w1, f1 = read_spectrum(uvb)
    w2, f2 = read_spectrum(vis)
    w = np.concatenate([w1[w1 < cut], w2[w2 >= cut]])
    f = np.concatenate([f1[w1 < cut], f2[w2 >= cut]])
    o = np.argsort(w)
    ok = np.isfinite(f[o])
    return w[o][ok], f[o][ok]


# ---------------------------------------------------------------- por objeto
def process(t17id, cat, spec_idx, filters, synth_dir, outdir, narrow_curves=None, opt=None):
    src = cat[cat['T17ID'] == t17id]
    hname = T17_TO_HERNANDEZ[t17id]
    if src.empty or hname not in spec_idx or not {'UVB', 'VIS'} <= set(spec_idx[hname]):
        print(f"  {t17id}: faltan datos (catalogo o espectros UVB/VIS)")
        return None
    src = src.iloc[0]
    wave, flux = combine_uvb_vis(spec_idx[hname]['UVB'], spec_idx[hname]['VIS'])
    if opt is not None and opt.blue_cut:
        keep = wave >= opt.blue_cut
        wave, flux = wave[keep], flux[keep]
    if opt is not None and opt.mask_tellurics:
        flux = mask_tellurics(wave, flux)

    # fotometria observada (sin corregir por extincion)
    obs, eobs = {}, {}
    for b in BROAD:
        m, e = src.get(f'{b}mag', np.nan), src.get(f'e_{b}mag', np.nan)
        if np.isfinite(m) and m < 90:
            obs[b], eobs[b] = m, (e if np.isfinite(e) else 0.05)
    for b in NARROW:
        m, e = src.get(f'MAG_{b}_{APERTURE}', np.nan), src.get(f'MAGERR_{b}_{APERTURE}', np.nan)
        if np.isfinite(m) and m < 90:
            obs[b], eobs[b] = m, (e if np.isfinite(e) and e < 90 else 0.1)

    # sintetica: DECam por convolucion; angostas del JSON
    syn, cover = {}, {}
    for b in BROAD:
        syn[b], cover[b] = synth_ab_mag(wave, flux, *filters[b])
    jpath = os.path.join(synth_dir, f"{hname}-GCs-SPLUS21-magnitude.json")
    if narrow_curves:
        for b, (fw, ft) in narrow_curves.items():
            syn[b], _ = synth_ab_mag(wave, flux, fw, ft)
        if os.path.exists(jpath):                      # control: FITS frente a JSON
            js = json.load(open(jpath))
            d = [syn[b] - float(js[k]) for k, b in JSON_NARROW.items()
                 if k in js and np.isfinite(syn.get(b, np.nan))]
            if d:
                print(f"  {t17id}: angostas FITS - JSON: mediana {np.median(d):+.3f}, "
                      f"rango [{min(d):+.3f}, {max(d):+.3f}] mag")
    elif os.path.exists(jpath):
        js = json.load(open(jpath))
        for k, b in JSON_NARROW.items():
            try:
                v = float(js.get(k))
                if np.isfinite(v) and v < 90:
                    syn[b] = v
            except (TypeError, ValueError):
                pass
    else:
        print(f"  {t17id}: no existe {jpath} (sin sinteticas angostas)")

    # normalizacion en la banda r: la sintetica coincide con la observada en r
    if 'r' not in obs or not np.isfinite(syn.get('r', np.nan)):
        print(f"  {t17id}: sin r observada o sintetica; se omite")
        return None
    W = WAVE_ORIG if (opt is None or opt.waves == 'original') else WAVE_EFF
    # residuos: siempre normalizando la sintetica (integrada en la banda) a la r observada
    dmag_band = obs['r'] - syn['r']
    resid = {b: obs[b] - (syn[b] + dmag_band) for b in obs if np.isfinite(syn.get(b, np.nan))}
    # escala para la FIGURA
    if opt is None or opt.norm == 'median':
        fs_raw = uniform_filter1d(flux, SMOOTH_WINDOW)
        m = (wave > 6200) & (wave < 6300)
        f_r_obs = mag_to_flam(obs['r'], W['r'])
        k = f_r_obs / np.median(fs_raw[m])               # como el script original
        dmag = -2.5 * np.log10(k)
    else:
        dmag = dmag_band
        k = 10**(-0.4 * dmag)
    syn_sc = {b: v + dmag for b, v in syn.items() if np.isfinite(v)}
    print(f"  {t17id}: escala mediana/banda = {10**(-0.4*(dmag - dmag_band)):.3f} "
          f"({(dmag - dmag_band):+.3f} mag)")

    for b in BROAD:
        if cover.get(b, 1) < 0.99:
            print(f"  {t17id}: el espectro cubre el {100*cover[b]:.1f}% de la transmision de la banda {b}")

    # ---------------- figura
    with_res = opt is not None and opt.resid_panel
    if with_res:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                                      gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.05))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fs = uniform_filter1d(flux, SMOOTH_WINDOW) * k / FLUX_SCALE
    ax.plot(wave, fs, color='grey', alpha=0.45, lw=1.0, zorder=2)
    for b in obs:
        f = mag_to_flam(obs[b], W[b]) / FLUX_SCALE
        ef = f * 0.4 * np.log(10) * eobs[b]
        mk, col, cs = ('s', COL_DECAM[b], 5) if b in BROAD else ('o', COL_T80S[b], 10)
        ax.errorbar(W[b], f, yerr=ef, fmt=mk, color=col, mec='k', ms=14,
                    capsize=cs, zorder=5)
    for b, m in syn_sc.items():
        ax.plot(W[b], mag_to_flam(m, W[b]) / FLUX_SCALE, 'D', color='#FF1493',
                mec='k', mew=1.0, ms=8, zorder=6)
    ax.set_ylabel(r'F$_\lambda$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=21)
    ax.tick_params(labelsize=20)
    ax.text(0.02, 0.96, f'{t17id}  ({hname})', transform=ax.transAxes, fontsize=21,
            va='top', bbox=dict(facecolor='white', alpha=0.8))
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', mfc='w', mec='k', ms=12, label='T80S narrow bands'),
        Line2D([0], [0], marker='s', color='w', mfc='w', mec='k', ms=12, label='DECam broad bands'),
        Line2D([0], [0], marker='D', color='w', mfc='#FF1493', mec='k', ms=10,
               label='Synthetic (X-Shooter)'),
        Line2D([0], [0], color='grey', lw=1.2, label='X-Shooter (Hernández+18)')],
        loc='lower right', fontsize=17, framealpha=0.9)
    if opt is not None and opt.ylim:
        ax.set_ylim(*opt.ylim)
    else:
        ax.set_ylim(min(0, np.nanpercentile(fs, 0.5)), np.nanpercentile(fs, 99.5) * 1.45)
    ax.set_xlim(3000, 10000)
    if not with_res:
        ax.set_xlabel('Wavelength (Å)', fontsize=21)
        out = os.path.join(outdir, f'photospectrum_{t17id}_synthetic.pdf')
        plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  OK {t17id} -> {out}")
        return {'T17ID': t17id, 'Hernandez': hname, **{f'dm_{b}': resid.get(b, np.nan) for b in BROAD + NARROW}}

    for b, d in resid.items():
        mk, col = ('s', COL_DECAM[b]) if b in BROAD else ('o', COL_T80S[b])
        axr.errorbar(W[b], d, yerr=eobs[b], fmt=mk, color=col, mec='k', ms=11, capsize=4)
    axr.axhline(0, color='k', lw=1)
    axr.axvspan(7590, 7700, color='lightblue', alpha=0.4, lw=0)    # banda A del O2
    axr.axvspan(8100, 8400, color='lightblue', alpha=0.25, lw=0)   # H2O
    axr.axvspan(8900, 9800, color='lightblue', alpha=0.25, lw=0)   # H2O
    axr.set_ylabel(r'$m_{\rm obs}-m_{\rm synth}$', fontsize=20)
    axr.set_xlabel('Wavelength (Å)', fontsize=21)
    axr.tick_params(labelsize=20)
    lim = max(0.3, 1.2 * np.nanmax(np.abs(list(resid.values()))))
    axr.set_ylim(-lim, lim)            # positivo = observado mas debil que la sintetica
    axr.set_xlim(3000, 10000)
    out = os.path.join(outdir, f'photospectrum_{t17id}_synthetic.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK {t17id} -> {out}")
    return {'T17ID': t17id, 'Hernandez': hname, **{f'dm_{b}': resid.get(b, np.nan)
                                                  for b in BROAD + NARROW}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', default='Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    ap.add_argument('--spectra-dir', default='../Spectre-Hernandez/')
    ap.add_argument('--synth-dir', default='../Spectre-Hernandez/spectra_ascii/')
    ap.add_argument('--decam-prefix', default='Filter_decam/CTIO_DECam.',
                    help="prefijo de las curvas DECam (se agrega u.dat, g.dat, ...)")
    ap.add_argument('--outdir', default='./photospectra_synthetic_v2/')
    ap.add_argument('--splus-filters', default='../filters/SPLUS21.filter',
                    help="curvas S-PLUS (formato de readfilterset); si existe, las angostas se "
                         "calculan del MISMO espectro FITS en lugar de tomarlas del JSON")
    ap.add_argument('--norm', choices=['band', 'median'], default='median',
                    help="escala del espectro: 'median' = mediana 6200-6300 A frente a r observada "
                         "(como el script original); 'band' = fotometria sintetica r = r observada")
    ap.add_argument('--waves', choices=['original', 'tableA1'], default='original',
                    help="longitudes de onda para dibujar las bandas DECam")
    ap.add_argument('--resid-panel', action='store_true', help="agregar el panel de residuos")
    ap.add_argument('--ylim', type=float, nargs=2, default=[-1.4, 4.0],
                    help="limites del eje Y (por defecto, los del script original)")
    ap.add_argument('--blue-cut', type=float, default=None,
                    help="descartar el espectro por debajo de esta longitud de onda (A), donde el "
                         "borde de UVB es puro ruido; p.ej. 3200")
    ap.add_argument('--mask-tellurics', action='store_true',
                    help="interpolar el continuo a traves de las bandas teluricas antes de la "
                         "fotometria sintetica (mide cuanto de la diferencia es telurica)")
    ap.add_argument('--syntphot-dir', default=None,
                    help="carpeta con readfilterset.py (por defecto, --synth-dir)")
    a = ap.parse_args()
    if a.mask_tellurics:
        a.outdir = a.outdir.rstrip('/') + '_telluric_masked/'
    if a.blue_cut:
        a.outdir = a.outdir.rstrip('/') + f'_bluecut{a.blue_cut:g}/'
    os.makedirs(a.outdir, exist_ok=True)

    filters = {b: load_filter(f"{a.decam_prefix}{b}.dat") for b in BROAD}
    narrow_curves = load_splus_narrow(a.splus_filters, a.syntphot_dir or a.synth_dir)
    print("Bandas angostas: " + ("calculadas del espectro FITS con " + a.splus_filters
                                 if narrow_curves else "tomadas del JSON"))
    cat = pd.read_csv(a.catalog)
    cat['T17ID'] = cat['T17ID'].astype(str).str.strip()
    idx = build_spectra_index(a.spectra_dir)
    print(f"Espectros: {', '.join(f'{k} {sorted(v)}' for k, v in idx.items())}")

    rows = [r for t in T17_TO_HERNANDEZ if (r := process(t, cat, idx, filters, a.synth_dir, a.outdir, narrow_curves, a))]
    if not rows:
        return
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.outdir, 'residuals_obs_minus_synth.csv'), index=False, float_format='%.3f')
    print(f"\nResiduos m_obs - m_synth (normalizados en r), N = {len(tab)} GCs:")
    print(f"{'banda':>6} {'mediana':>8} {'disp(1.48 MAD)':>15} {'N':>3}")
    for b in BROAD + NARROW:
        v = tab[f'dm_{b}'].dropna().values
        if len(v):
            mad = 1.4826 * np.median(np.abs(v - np.median(v)))
            print(f"{b:>6} {np.median(v):+8.3f} {mad:15.3f} {len(v):3d}")
    print(f"\nOK tabla: {a.outdir}residuals_obs_minus_synth.csv")


if __name__ == '__main__':
    main()
