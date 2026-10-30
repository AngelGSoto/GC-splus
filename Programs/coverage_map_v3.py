#!/usr/bin/env python3
"""
coverage_map.py -- respuesta al arbitro (cobertura y completitud espacial de las observaciones nuevas).

Figura similar a la Fig. 1 de Taylor et al. (2017): huella de los 24 apuntados T80S (leida del WCS
de las imagenes de cada campo) sobre las posiciones de los candidatos a GC de T17, con la
completitud de la muestra final en cada campo. Tambien guarda una tabla por campo.

Muestras:
  * T17: candidatos del catalogo de Taylor et al. (2017) con P(GC) >= --pmin (muestra inicial)
  * seleccion fotometrica: catalogo de 1394 objetos (--selected)
  * muestra final: chi2_nu <= 2 en out/results.fits (884 GCs)

Uso (desde anac_data/):
  python ../Programs/coverage_map_v3.py
  python ../Programs/coverage_map_v3.py --band F660 --pmin 0.7
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

RA0, DEC0, D_KPC = 201.365063, -43.019112, 3800.0

# ---- estilo de la figura de datos (Sect. 2.1): editar aqui ----
C_POINTS = '#1B4F72'      # candidatos de T17 (azul profundo)
C_FIELD_FILL = '#9B8FD1'  # relleno de los campos T80S (lavanda)
C_FIELD_EDGE = '#5E4FA2'  # borde de los campos
C_FIELD_TEXT = '#4B3C8C'  # numero de los campos
FS_AXES, FS_TICKS, FS_LEGEND, FS_FIELD = 26, 21, 19, 17
PAD_BOTTOM = 0.9          # espacio extra abajo para la leyenda (grados)


def footprint(field, band):
    for ext in (f'{field}/{field}_{band}.fits.fz', f'{field}/{field}_{band}.fits'):
        if os.path.exists(ext):
            with fits.open(ext) as hdul:
                for h in hdul:
                    if h.data is not None and h.header.get('NAXIS', 0) >= 2:
                        try:
                            w = WCS(h.header)
                            if w.is_celestial:
                                return w.calc_footprint(axes=(h.header['NAXIS1'], h.header['NAXIS2']))
                        except Exception:
                            continue
    return None


def inside(poly, ra, dec):
    from matplotlib.path import Path
    # aproximacion plana local (campos de ~1.4 deg): RA escalada por cos(dec)
    c = np.cos(np.radians(DEC0))
    p = Path(np.c_[(poly[:, 0] - RA0) * c, poly[:, 1] - DEC0])
    return p.contains_points(np.c_[(ra - RA0) * c, dec - DEC0])


def fov_square(ra_c, dec_c, side):
    """Cuadrado de lado `side` (deg) en el plano tangente, centrado en (ra_c, dec_c)."""
    h = np.radians(side / 2)
    xi = np.array([-h, h, h, -h]); eta = np.array([-h, -h, h, h])
    a0, d0 = np.radians(ra_c), np.radians(dec_c)
    den = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, den))
    return np.c_[np.degrees(ra), np.degrees(dec)]


def find_col(df, options):
    for o in options:
        if o in df.columns:
            return o
    low = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in low:
            return low[o.lower()]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--t17', default='../TAP_1_J_MNRAS_3444_gc.csv')
    ap.add_argument('--selected', default='Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    ap.add_argument('--results', default='CIGALE-DEcam-extinction/out/results.fits')
    ap.add_argument('--band', default='F660', help='imagen usada para la huella de cada campo')
    ap.add_argument('--pmin', type=float, default=0.7)
    ap.add_argument('--pcol', default=None, help='columna de P(GC) en T17 (se detecta sola)')
    ap.add_argument('--data-figure', action='store_true',
                    help="figura para la Sect. 2.1: todos los candidatos de T17 (sin corte en P(GC)) y "
                         "los apuntados, sin la muestra final ni la completitud")
    ap.add_argument('--fov', type=float, default=None,
                    help="dibujar cuadrados del campo nominal de este lado (deg), p.ej. 1.4 para T80Cam, "
                         "centrados en cada apuntado (los conteos siguen usando la huella real)")
    ap.add_argument('--exclude', nargs='*', default=[],
                    help="campos que no se dibujan (p.ej. CenA24, alejado y sin candidatos)")
    ap.add_argument('--outdir', default='figures/coverage')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # ---- catalogos
    t17 = pd.read_csv(a.t17)
    rac, decc = find_col(t17, ['RAJ2000', 'RA', 'ra']), find_col(t17, ['DEJ2000', 'DEC', 'dec'])
    pcol = a.pcol or find_col(t17, ['pGC', 'PGC', 'P_GC', 'Pgc', 'pgc', 'Prob', 'prob', 'P'])
    if pcol is None:
        raise SystemExit(f"No encuentro la columna de P(GC) en {a.t17}; usa --pcol. Columnas: {list(t17.columns)}")
    idc = find_col(t17, ['T17ID', 'ID', 'Name'])
    if a.data_figure:
        a.pmin = -np.inf                                # todos los candidatos
    t17 = t17[t17[pcol] >= a.pmin].copy()
    sel = pd.read_csv(a.selected)
    sel = sel.drop_duplicates(subset=['RAJ2000', 'DEJ2000'], keep='first')
    res = Table.read(a.results, hdu=1).to_pandas()
    res['id'] = res['id'].astype(str).str.strip()
    good_ids = set(res.loc[res['best.reduced_chi_square'] <= 2.0, 'id'])
    sel['id'] = sel['T17ID'].astype(str).str.strip()
    final = sel[sel['id'].isin(good_ids)]
    print(f"T17 con P(GC) >= {a.pmin} ({pcol}): {len(t17)} | seleccion: {len(sel)} | final: {len(final)}")

    # ---- huellas
    fields = sorted({os.path.basename(d) for d in glob.glob('CenA[0-9][0-9]') if os.path.isdir(d)})
    polys = {f: footprint(f, a.band) for f in fields}
    polys = {f: p for f, p in polys.items() if p is not None}
    print(f"Campos con huella: {len(polys)} de {len(fields)}")

    rows = []
    in_any = np.zeros(len(t17), bool)
    for f, p in polys.items():
        mt = inside(p, t17[rac].values, t17[decc].values)
        ms = inside(p, sel['RAJ2000'].values, sel['DEJ2000'].values)
        mf = inside(p, final['RAJ2000'].values, final['DEJ2000'].values)
        in_any |= mt
        rows.append(dict(field=f, ra_c=p[:, 0].mean(), dec_c=p[:, 1].mean(), N_T17=mt.sum(),
                         N_selected=ms.sum(), N_final=mf.sum(),
                         compl_final=mf.sum() / mt.sum() if mt.sum() else np.nan))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.outdir, 'coverage_per_field.csv'), index=False, float_format='%.3f')
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f"\nCandidatos T17 dentro de algun campo T80S: {in_any.sum()} de {len(t17)} "
          f"({100 * in_any.mean():.1f}%)")
    print("(los campos se superponen, asi que la suma por campo puede contar objetos dos veces)")

    # ---- figura
    fig, ax = plt.subplots(figsize=(10, 10))
    if a.data_figure:
        ax.scatter(t17[rac], t17[decc], s=12, color=C_POINTS, lw=0, alpha=0.75, zorder=3,
                   label='GC candidates (Taylor et al. 2017)')
    else:
        ax.scatter(t17[rac], t17[decc], s=6, color='0.6', lw=0, alpha=0.7,
                   label=f'T17 candidates, $P$(GC)$\\geq${a.pmin} (N = {len(t17)})')
        ax.scatter(final['RAJ2000'], final['DEJ2000'], s=12, color='royalblue', lw=0, alpha=0.85,
                   label=f'Final sample (N = {len(final)})')
    import matplotlib.patheffects as pe
    shown = tab[~tab['field'].isin(a.exclude)]
    draw = {}
    for _, r in shown.iterrows():
        draw[r['field']] = (fov_square(r['ra_c'], r['dec_c'], a.fov) if a.fov else polys[r['field']])
    for _, r in shown.iterrows():
        p = draw[r['field']]
        if a.data_figure:
            ax.add_patch(Polygon(p, closed=True, fc=C_FIELD_FILL, alpha=0.13, ec='none', zorder=0))
            ax.add_patch(Polygon(p, closed=True, fill=False, ec=C_FIELD_EDGE, lw=1.4, alpha=0.9, zorder=1))
        else:
            ax.add_patch(Polygon(p, closed=True, fc='0.5', alpha=0.07, ec='none', zorder=0))
            ax.add_patch(Polygon(p, closed=True, fill=False, ec='k', lw=1.0, alpha=0.7, zorder=1))
        txt = r['field'].replace('CenA', '')
        if r['N_T17'] > 0 and not a.data_figure:
            txt += f"\n{int(r['N_final'])}/{int(r['N_T17'])}"
        ax.text(r['ra_c'], r['dec_c'], txt, ha='center', va='center', fontsize=FS_FIELD,
                color=(C_FIELD_TEXT if a.data_figure else 'firebrick'),
                fontweight='bold', zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    # circulos de radio proyectado
    t = np.linspace(0, 2 * np.pi, 400)
    for rk in (50, 100, 150):
        rd = np.degrees(rk / D_KPC)
        ax.plot(RA0 + rd * np.cos(t) / np.cos(np.radians(DEC0)), DEC0 + rd * np.sin(t),
                ls='--', color='0.4', lw=1)
        ax.text(RA0 + 0.7 * rd / np.cos(np.radians(DEC0)), DEC0 - 0.7 * rd, f'{rk} kpc', ha='left',
                va='top', fontsize=FS_FIELD - 1, color='0.25', zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax.plot(RA0, DEC0, marker='*', ms=22, color='gold', mec='k', zorder=7)
    allp = np.vstack([draw[f] for f in shown['field']])
    pad = 0.15
    ax.set_xlim(allp[:, 0].max() + pad, allp[:, 0].min() - pad)   # Este a la izquierda
    ax.set_ylim(allp[:, 1].min() - PAD_BOTTOM, allp[:, 1].max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(DEC0)))
    ax.set_xlabel('RA (deg)', fontsize=FS_AXES)
    ax.set_ylabel('Dec (deg)', fontsize=FS_AXES)
    ax.tick_params(labelsize=FS_TICKS)
    h, l = ax.get_legend_handles_labels()
    if a.data_figure:
        h.append(Polygon([[0, 0]], closed=True, fc=C_FIELD_FILL, alpha=0.35, ec=C_FIELD_EDGE, lw=1.4))
        l.append('T80S pointings (this work)')
    ax.legend(h, l, loc='lower left', fontsize=FS_LEGEND, framealpha=0.92, markerscale=2.5)
    plt.tight_layout()
    out = os.path.join(a.outdir, 'coverage_map_data' if a.data_figure else 'coverage_map')
    plt.savefig(out + '.pdf', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f"\nOK figura: {out}.pdf | tabla: {a.outdir}/coverage_per_field.csv")


if __name__ == '__main__':
    main()
