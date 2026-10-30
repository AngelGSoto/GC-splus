#!/usr/bin/env python3
'''
photo-spectra-syntetic-flux-v7.py -- Fig. A.1 (calibracion de la fotometria T80S),
al estilo de las figuras de calibracion de los data releases de S-PLUS.

Panel a) espectro Gaia XP (flujo), curvas de transmision de los 7 filtros angostos T80S
         (rellenas) y de las 5 bandas anchas SDSS u'g'r'i'z' (punteadas), y la fotometria
         SINTETICA de los filtros angostos.
Panel b) en magnitudes: el espectro en magnitud AB, la magnitud sintetica (relleno) y la
         instrumental + 20 (abierto) de cada filtro angosto, unidas por una linea punteada;
         el recuadro da el punto cero del campo (mediana de las 100 estrellas de referencia,
         con recorte a 2.5 sigma).

Las posiciones de las etiquetas de TODOS los filtros se ajustan a mano en los bloques
"POSICIONES DE LAS ETIQUETAS" (x_offset en A, y_offset en unidades de flujo del eje Y).

Uso (desde gaia_spectra_CenA01/):
  python ../../Programs/photo-spectra-syntetic-flux-v7.py \
      gaia_xp_spectrum_6087393007027689344-Ref-SPLUS21-magnitude --savefig --debug
'''
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import ascii

FILT = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
WL = [3785, 3950, 4100, 4300, 5150, 6600, 8610]
COL = ["#9900FF", "#6600FF", "#0000FF", "#009999", "#DD8000", "#CC0066", "#660033"]
LAB = [r'$J0378$', r'$J0395$', r'$J0410$', r'$J0430$', r'$J0515$', r'$J0660$', r'$J0861$']
ALPHA = [0.25, 0.25, 0.35, 0.25, 0.25, 0.25, 0.25]
BROAD = ['u', 'g', 'r', 'i', 'z']
BCOL = {'u': "#CC00FF", 'g': "#006600", 'r': "#FF0000", 'i': "#990033", 'z': "#330034"}
C_AA = 2.99792458e18
_trap = getattr(np, 'trapezoid', None) or np.trapz

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("source", help="prefijo del JSON de la fotometria sintetica (sin .json)")
ap.add_argument("--field", default="CenA01")
ap.add_argument("--inst-csv", default=None, help="por defecto ../<field>_gaia_xp_matches_3arcsec.csv")
ap.add_argument("--zp-file", default="../Results/all_fields_zero_points_splus_format_3arcsec.csv")
ap.add_argument("--broad-prefix", default="../../Filter_sdss/SDSS_")
ap.add_argument("--inst-offset", type=float, default=20.0,
                help="constante sumada a las magnitudes instrumentales en el panel b")
ap.add_argument("--box", choices=['zp', 'offset'], default='zp',
                help="valor en los recuadros: el punto cero (zp) o ZP - offset (offset)")
ap.add_argument("--ymin", type=float, default=0.15)
ap.add_argument("--ymax", type=float, default=0.52)
ap.add_argument("--savefig", action="store_true")
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()


def mag2flux(m, w):
    """AB -> f_lambda en 1e-14 erg/s/cm2/A."""
    return 10**(-(m + 2.41) / 2.5) / w**2 / 1e-14


def load_curve(path):
    d = np.loadtxt(path, comments='#')
    w, t = d[:, 0], d[:, 1]
    return (w * 10 if w.max() < 2000 else w), np.clip(t, 0, None)


# =====================================================================
# DATOS
# =====================================================================
syn = json.load(open(args.source + ".json"))
m_syn = np.array([float(syn[f'F0{f[1:]}']) for f in FILT])

specfile = args.source.split('-Ref')[0] + ".dat"
d = np.loadtxt(specfile, skiprows=1, dtype=np.dtype([('Wl', 'f8'), ('Flux', 'f8')]))
x, flam = d['Wl'], d['Flux']
y = flam / 1e-14
gaia_id = specfile.replace("gaia_xp_spectrum_", "").replace(".dat", "")

inst = pd.read_csv(args.inst_csv or f"../{args.field}_gaia_xp_matches_3arcsec.csv",
                   dtype={'source_id': str})
inst = inst[inst['source_id'].str.strip() == gaia_id].iloc[0]
zps = pd.read_csv(args.zp_file)
zps = zps[zps['field'] == args.field].iloc[0]
m_inst = np.array([inst[f'mag_inst_3.0_{f}'] for f in FILT], dtype=float)
m_inst[m_inst > 90] = np.nan
zp = np.array([zps[f] for f in FILT], dtype=float)
m_cal = m_inst + zp

broad = {}
for b in BROAD:
    try:
        fw, ft = load_curve(f"{args.broad_prefix}{b}.dat")
    except OSError:
        broad = {}
        print(f"(no se encontraron las curvas {args.broad_prefix}*.dat; sin bandas anchas)")
        break
    broad[b] = dict(fw=fw, ft=ft)

if args.debug or not args.savefig:
    print(f"Estrella {gaia_id} ({args.field})")
    print(f"{'filtro':>7} {'m_syn':>8} {'m_inst':>8} {'ZP_campo':>9} {'m_inst+ZP':>10} {'dif':>7}")
    for l, a, mi, z, mc in zip(LAB, m_syn, m_inst, zp, m_cal):
        print(f"{l.strip('$'):>7} {a:8.3f} {mi:8.3f} {z:9.3f} {mc:10.3f} {mc - a:+7.3f}")

if not args.savefig:
    raise SystemExit

# =====================================================================
# FIGURA
# =====================================================================
fig, (ax, axb) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True,
                              gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.05))
for a_ in (ax, axb):
    a_.spines["top"].set_visible(False)
    a_.spines["right"].set_visible(False)
    a_.tick_params(axis='both', labelsize=21)

# ---------------------------------------------------------------- panel a
ax.set_xlim(3000, 9500)
ax.set_ylim(args.ymin, args.ymax)
ax.set_ylabel(r'Flux $(\mathrm{10^{-14}\,erg\,s^{-1}\,cm^{-2}\,\AA^{-1}})$', fontsize=20)
h = (args.ymax - args.ymin) * 0.8          # altura de las curvas: 80% del rango del eje Y

# ----- filtros angostos (curvas rellenas)
for f, col, lab, alp in zip(FILT, COL, LAB, ALPHA):
    try:
        t = ascii.read(f'../../filters/F0{f[1:]}.dat')
        xt, yt = np.array(t['col1']), np.array(t['col2'])
    except Exception as e:
        if args.debug:
            print(f"no se pudo leer la curva de {f}: {e}")
        continue
    ys = yt / yt.max() * h + args.ymin
    ax.fill(xt, ys, color=col, alpha=alp, zorder=2)

    max_y = ys.max()
    max_x = xt[np.argmax(yt)]

    # ===== POSICIONES DE LAS ETIQUETAS: FILTROS ANGOSTOS =====
    if lab == r'$J0861$':
        x_offset = 180
        y_offset = 0.0
    elif lab == r'$J0378$':
        x_offset = -280
        y_offset = -0.23
    elif lab == r'$J0395$':
        x_offset = -0.0
        y_offset = -0.13
    elif lab == r'$J0410$':
        x_offset = -40
        y_offset = -0.08
    elif lab == r'$J0430$':
        x_offset = -40
        y_offset = 0.0
    elif lab == r'$J0515$':
        x_offset = 0
        y_offset = -0.0
    elif lab == r'$J0660$':
        x_offset = 0
        y_offset = 0.0
    else:
        x_offset = 0
        y_offset = 0.015

    ax.text(max_x + x_offset, max_y + y_offset, lab, color=col, fontsize=18,
            ha='center', va='bottom', zorder=3, fontweight='bold')

# ----- bandas anchas (curvas punteadas)
for b, v in broad.items():
    ys = v['ft'] / v['ft'].max() * h + args.ymin
    ax.plot(v['fw'], ys, ls=':', lw=2.0, color=BCOL[b], alpha=0.8, zorder=2)

    max_y = ys.max()
    max_x = v['fw'][np.argmax(v['ft'])]

    # ===== POSICIONES DE LAS ETIQUETAS: BANDAS ANCHAS =====
    if b == 'u':
        x_offset = -60
        y_offset = 0.006
    elif b == 'g':
        x_offset = -400
        y_offset = -0.03
    elif b == 'r':
        x_offset = -330
        y_offset = -0.01
    elif b == 'i':
        x_offset = 0
        y_offset = 0.006
    elif b == 'z':
        x_offset = 360
        y_offset = -0.1
    else:
        x_offset = 0
        y_offset = 0.006

    ax.text(max_x + x_offset, max_y + y_offset, b, color=BCOL[b], fontsize=17,
            ha='center', va='bottom', style='italic', fontweight='bold', zorder=3)

# ----- espectro y fotometria sintetica de los filtros angostos
ax.plot(x, y, lw=3.0, color="black", alpha=0.5, zorder=10)
for w, ms, col in zip(WL, m_syn, COL):
    ax.scatter(w, mag2flux(ms, w), c=col, marker='o', s=230, edgecolors='k', lw=1.5, zorder=20)

ax.text(0.02, 0.96, 'a)', transform=ax.transAxes, fontsize=20, va='top',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white"))
ax.text(0.98, 0.96, f"Gaia DR3 {gaia_id}", transform=ax.transAxes, fontsize=16, va='top',
        ha='right', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

# ---------------------------------------------------------------- panel b
ok = (flam > 0) & (x > 3300)
m_spec = -2.5 * np.log10(flam[ok] * x[ok]**2 / C_AA) - 48.6
axb.plot(x[ok], m_spec, color='0.45', lw=2.5, zorder=2)
off = args.inst_offset
for w, ms, mi, z, col in zip(WL, m_syn, m_inst, zp, COL):
    axb.scatter(w, ms, marker='o', s=170, c=col, edgecolors='k', lw=1.2, zorder=6)
    if np.isfinite(mi):
        axb.scatter(w, mi + off, marker='o', s=150, facecolors='white', edgecolors=col,
                    lw=2.5, zorder=6)
        axb.plot([w, w], [mi + off, ms], ls=':', color=col, lw=1.8, zorder=4)
        val = z if args.box == 'zp' else z - off
        axb.text(w, max(ms, mi + off) + 0.35, f"{val:.2f}", rotation=90, fontsize=13,
                 ha='center', va='top', color=col,
                 bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor=col, lw=1.5))
lo = np.nanmin(np.r_[m_syn, m_inst + off]) - 1.0
hi = np.nanmax(np.r_[m_syn, m_inst + off]) + 2.0
axb.set_ylim(hi, lo)                                   # magnitudes: brillante arriba
axb.set_ylabel('Magnitude', fontsize=20)
axb.set_xlabel(r'Wavelength $[\mathrm{\AA}]$', fontsize=21)
axb.grid(alpha=0.2, ls='--')
axb.text(0.02, 0.94, 'b)', transform=axb.transAxes, fontsize=20, va='top',
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white"))
axb.legend(handles=[
    Line2D([0], [0], color='0.45', lw=2.5, label='Gaia XP spectrum'),
    Line2D([0], [0], marker='o', color='w', mfc='0.5', mec='k', ms=12,
           label='Synthetic magnitude (Gaia XP)'),
    Line2D([0], [0], marker='o', color='w', mfc='white', mec='k', mew=2, ms=12,
           label='Instrumental magnitude')],
    loc='lower right', fontsize=14, framealpha=0.9)

out = args.source + "-flux-zp_narrow.pdf"
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"Figura guardada: {out}")
