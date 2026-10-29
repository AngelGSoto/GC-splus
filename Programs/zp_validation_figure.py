#!/usr/bin/env python3
"""
zp_validation_figure.py -- comparacion entre las magnitudes CALIBRADAS (instrumental + ZP)
y las SINTETICAS (Gaia XP) de todas las estrellas de referencia (respuesta al arbitro).

Reproduce exactamente el calculo de ZeroPoints_calculations_final.py:
  * magnitudes instrumentales 3": CenAXX_gaia_xp_matches_3arcsec.csv (mag_inst_3.0_FXXX)
  * sinteticas: gaia_spectra_CenAXX/gaia_xp_spectrum_<source_id>-Ref-SPLUS21-magnitude.json
  * ZP_i = m_sint - m_inst, validos si finitos y 15 < ZP_i < 30
  * ZP del campo = mediana con recorte sigma iterativo a 2.5 sigma
Luego, para cada estrella: m_cal = m_inst + ZP_campo, residuo = m_cal - m_sint.

Ademas:
  * compara el ZP recalculado con el guardado en Results/all_fields_zero_points_splus_format_3arcsec.csv
  * imprime las magnitudes calibradas de una estrella (por defecto la de la Fig. A.1)

Uso (desde anac_data):
  python ../Programs/zp_validation_figure.py
  python ../Programs/zp_validation_figure.py --star 6087393007027689344
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
from astropy.stats import sigma_clipped_stats

FILTERS = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
JKEY = {f: 'F0' + f[1:] for f in FILTERS}
LABEL = {f: 'J0' + f[1:] for f in FILTERS}
COLORS = ["#9900FF", "#6600FF", "#0000FF", "#009999", "#DD8000", "#CC0066", "#660033"]


def load_field(csv):
    field = os.path.basename(csv).replace('_gaia_xp_matches_3arcsec.csv', '')
    df = pd.read_csv(csv, dtype={'source_id': str})
    df['source_id'] = df['source_id'].astype(str).str.strip()
    rows = []
    for _, r in df.iterrows():
        sid = r['source_id']
        js = glob.glob(f"gaia_spectra_{field}/*{sid}*-Ref-SPLUS21-magnitude.json")
        if not js:
            continue
        syn = json.load(open(js[0]))
        for f in FILTERS:
            mi = r.get(f'mag_inst_3.0_{f}', np.nan)
            ms = syn.get(JKEY[f], None)
            try:
                ms = float(ms)
            except (TypeError, ValueError):
                continue
            zp = ms - mi
            if np.isfinite(zp) and 15 < zp < 30:
                rows.append(dict(field=field, source_id=sid, filter=f, m_inst=mi, m_syn=ms,
                                 err=r.get(f'mag_err_3.0_{f}', np.nan), zp_i=zp))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zp-file', default='Results/all_fields_zero_points_splus_format_3arcsec.csv')
    ap.add_argument('--star', default='6087393007027689344', help="source_id de la estrella de la Fig. A.1")
    ap.add_argument('--outdir', default='zp_validation')
    ap.add_argument('--ylim', type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    files = sorted(glob.glob('CenA*_gaia_xp_matches_3arcsec.csv'))
    data = pd.concat([load_field(f) for f in files], ignore_index=True)
    if data.empty:
        print("No se encontraron estrellas con fotometria instrumental y sintetica.")
        return

    # ZP por campo y filtro (mismo calculo que ZeroPoints_calculations_final.py)
    zp_rows = []
    for (fld, f), g in data.groupby(['field', 'filter']):
        if len(g) >= 5:
            _, med, std = sigma_clipped_stats(g['zp_i'].values, sigma=2.5)
        else:
            med, std = np.median(g['zp_i']), np.std(g['zp_i'])
        zp_rows.append(dict(field=fld, filter=f, zp=med, zp_std=std, n=len(g)))
    zp = pd.DataFrame(zp_rows)
    data = data.merge(zp, on=['field', 'filter'])
    data['m_cal'] = data['m_inst'] + data['zp']
    data['resid'] = data['m_cal'] - data['m_syn']
    data['clipped'] = np.abs(data['zp_i'] - data['zp']) > 2.5 * data['zp_std']

    # control: ZP recalculado frente al guardado
    if os.path.exists(a.zp_file):
        old = pd.read_csv(a.zp_file).melt(id_vars='field', var_name='filter', value_name='zp_file')
        cmp_ = zp.merge(old, on=['field', 'filter'])
        d = (cmp_['zp'] - cmp_['zp_file']).abs()
        print(f"Control ZP recalculado - ZP del archivo: max |diferencia| = {d.max():.2e} mag "
              f"({len(cmp_)} campo-filtro)")
    else:
        print(f"(no se encontro {a.zp_file}; no se hace el control)")

    # resumen
    print(f"\nCampos: {data['field'].nunique()} | estrellas: {data['source_id'].nunique()}")
    print(f"{'filtro':>7} {'N':>5} {'N/campo':>8} {'mediana':>8} {'disp(1.48MAD)':>13} "
          f"{'recortadas':>10} {'sigma_ZP(med. campos)':>22}")
    summ = []
    for f in FILTERS:
        g = data[data['filter'] == f]
        if g.empty:
            continue
        k = g[~g['clipped']]
        mad = 1.4826 * np.median(np.abs(k['resid'] - np.median(k['resid'])))
        zf = zp[zp['filter'] == f]
        err_zp = np.median(1.253 * zf['zp_std'] / np.sqrt(zf['n']))
        summ.append(dict(filter=LABEL[f], N=len(g), N_per_field=np.median(zf['n']),
                         median=np.median(k['resid']), scatter=mad,
                         frac_clipped=g['clipped'].mean(), zp_err=err_zp))
        print(f"{LABEL[f]:>7} {len(g):5d} {np.median(zf['n']):8.0f} {np.median(k['resid']):+8.3f} "
              f"{mad:13.3f} {100*g['clipped'].mean():9.1f}% {err_zp:22.4f}")
    pd.DataFrame(summ).to_csv(os.path.join(a.outdir, 'zp_validation_summary.csv'), index=False,
                              float_format='%.4f')
    zp.to_csv(os.path.join(a.outdir, 'zero_points_recomputed.csv'), index=False, float_format='%.4f')

    # estrella de la Fig. A.1
    s = data[data['source_id'] == str(a.star)]
    if not s.empty:
        print(f"\nEstrella {a.star} ({s['field'].iloc[0]}):")
        for _, r in s.sort_values('filter').iterrows():
            print(f"  {LABEL[r['filter']]}: sintetica {r['m_syn']:.3f}  calibrada {r['m_cal']:.3f} "
                  f"± {r['err']:.3f}  (dif {r['resid']:+.3f})")
        s[['filter', 'm_syn', 'm_cal', 'err', 'resid']].to_csv(
            os.path.join(a.outdir, f'star_{a.star}.csv'), index=False, float_format='%.4f')

    # figura: un panel por filtro
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes = axes.ravel()
    for i, f in enumerate(FILTERS):
        ax = axes[i]
        g = data[data['filter'] == f]
        k = g[~g['clipped']]
        ax.scatter(k['m_syn'], k['resid'], s=6, alpha=0.35, color=COLORS[i], edgecolors='none')
        ax.scatter(g.loc[g['clipped'], 'm_syn'], g.loc[g['clipped'], 'resid'], s=8, marker='x',
                   color='0.5', alpha=0.5, lw=0.8)
        ax.axhline(0, color='k', lw=1)
        mad = 1.4826 * np.median(np.abs(k['resid'] - np.median(k['resid'])))
        ax.text(0.04, 0.95, f"{LABEL[f]}\nN = {len(g)}\nσ = {mad:.3f} mag", transform=ax.transAxes,
                va='top', fontsize=17, bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))
        ax.set_ylim(-a.ylim, a.ylim)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.2, ls='--')
        if i % 4 == 0:
            ax.set_ylabel(r'$m_{\rm cal} - m_{\rm syn}$ (mag)', fontsize=19)
        if i >= 3:
            ax.set_xlabel(r'$m_{\rm syn}$ (Gaia XP, AB mag)', fontsize=18)
    axes[-1].axis('off')
    plt.tight_layout()
    base = os.path.join(a.outdir, 'zp_validation_all_stars')
    plt.savefig(base + '.pdf', bbox_inches='tight')
    plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nOK figura: {base}.pdf | tablas en {a.outdir}/")


if __name__ == '__main__':
    main()
