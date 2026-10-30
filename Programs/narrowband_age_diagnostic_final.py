#!/usr/bin/env python3
"""
narrowband_age_diagnostic_final.py -- respuesta al arbitro (Fig. 6/G.1): ?hay algun color con bandas
angostas que distinga GCs jovenes y viejos con el mismo color (g - z)?

Para cada color candidato C:
  1. ajusta C = polinomio de grado 2 en (g - z) usando los GCs VIEJOS (> 8 Gyr);
  2. calcula el residuo dC de jovenes y viejos respecto de esa relacion;
  3. mide la separacion: S = (mediana_jov - mediana_viej) / sigma_viej (1.48 MAD),
     AUC (Mann-Whitney; 0.5 = no separa, 1 = separa perfectamente) y p de Mann-Whitney.
Se hace con los flujos del MODELO de mejor ajuste (results.fits, sin ruido: poder discriminante
de los modelos) y con los flujos OBSERVADOS (observations.fits, solo S/N > 3 en las bandas del
color: si la senal sobrevive al ruido).
CUIDADO: las edades vienen de los mismos ajustes, asi que la separacion observada es en parte
circular; los colores del modelo muestran que bandas aportan la informacion de edad.

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/narrowband_age_diagnostic_final.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import mannwhitneyu

BANDS = {'J0378': 'F0378', 'J0395': 'F0395', 'J0410': 'F0410', 'J0430': 'F0430',
         'J0515': 'F0515', 'J0660': 'F0660', 'J0861': 'F0861',
         'u': 'sdss.up', 'g': 'sdss.gp', 'r': 'sdss.rp', 'i': 'sdss.ip', 'z': 'sdss.zp'}
COLOURS = [('J0378', 'J0410'), ('J0395', 'J0410'), ('J0378', 'g'), ('J0395', 'g'),
           ('J0410', 'g'), ('J0430', 'g'), ('J0410', 'J0430'), ('J0515', 'g'),
           ('J0515', 'r'), ('J0660', 'r'), ('J0861', 'i'), ('J0861', 'z'),
           ('u', 'g'), ('u', 'J0378'), ('r', 'i'), ('i', 'z')]
NARROW = {k for k in BANDS if k.startswith('J')}


def mag(f_mjy):
    f = np.asarray(f_mjy, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(f > 0, -2.5 * np.log10(f * 1e-3 / 3631.0), np.nan)


def separation(gz, c, young, deg=2, gz_range=None):
    ok = np.isfinite(gz) & np.isfinite(c)
    if gz_range is not None:
        ok &= (gz >= gz_range[0]) & (gz <= gz_range[1])
    gz, c, young = gz[ok], c[ok], young[ok]
    if (~young).sum() < 20 or young.sum() < 8:
        return None
    p = np.polyfit(gz[~young], c[~young], deg)
    d = c - np.polyval(p, gz)
    do, dy = d[~young], d[young]
    sig = 1.4826 * np.median(np.abs(do - np.median(do)))
    u, pval = mannwhitneyu(dy, do, alternative='two-sided')
    auc = u / (len(dy) * len(do))
    return dict(N_old=len(do), N_young=len(dy), shift=np.median(dy) - np.median(do), sigma_old=sig,
                S=(np.median(dy) - np.median(do)) / sig if sig > 0 else np.nan,
                AUC=max(auc, 1 - auc), p=pval, poly=p, gz=gz, c=c, young=young, d=d)


def plot_colour_single(store, name, a):
    """Un panel: colores observados de jovenes y viejos vs (g-z), con una recta para cada grupo."""
    rm, ro = store[name]
    gz, c, y = ro['gz'], ro['c'], ro['young']
    model_shift = np.median(rm['d'][rm['young']]) - np.median(rm['d'][~rm['young']])
    obs_shift = np.median(ro['d'][y]) - np.median(ro['d'][~y])
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.scatter(gz[~y], c[~y], s=100, color='steelblue', alpha=0.6, edgecolors='k', lw=0.3,
               label=f'Old (N = {(~y).sum()})')
    ax.scatter(gz[y], c[y], s=100, color='darkorange', alpha=0.85, edgecolors='k', lw=0.3,
               label=f'Young (N = {y.sum()})')
    xx = np.linspace(gz.min(), gz.max(), 100)
    ax.plot(xx, np.polyval(ro['poly'], xx), color='steelblue', ls='--', lw=2.4, label='Fit to old GCs')
    py = np.polyfit(gz[y], c[y], len(ro['poly']) - 1)
    ax.plot(xx, np.polyval(py, xx), color='darkorange', ls='--', lw=2.4, label='Fit to young GCs')
    ax.text(0.03, 0.85, f"AUC = {ro['AUC']:.2f}\nOffset: {obs_shift:+.2f} mag\n(models: {model_shift:+.2f} mag)",
            transform=ax.transAxes, va='top', fontsize=13,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))
    ax.set_xlabel(r'$(g-z)_0$', fontsize=19)
    ax.set_ylabel(f'$({name})_0$', fontsize=19)
    ax.tick_params(labelsize=15)
    lo, hi = np.nanmin(c), np.nanmax(c)
    rg = hi - lo
    ax.set_ylim(lo - 0.08 * rg, hi + 0.55 * rg)          # franja superior para leyenda y recuadro
    dx = gz.max() - gz.min()
    ax.set_xlim(gz.min() - 0.05 * dx, gz.max() + 0.05 * dx)
    ax.legend(loc='upper left', fontsize=13, framealpha=0.9, ncol=2)
    plt.tight_layout()
    out = os.path.join(a.outdir, 'narrowband_age_colour_single')
    plt.savefig(out + '.pdf', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f"OK figura de un panel: {out}.pdf (desplazamiento observado {obs_shift:+.3f}, modelos {model_shift:+.3f} mag)")


def plot_offset_single(store, name, a):
    """Un panel: desplazamientos observados de jovenes y viejos + prediccion de los modelos."""
    rm, ro = store[name]
    do, dy = ro['d'][~ro['young']], ro['d'][ro['young']]
    model_shift = np.median(rm['d'][rm['young']]) - np.median(rm['d'][~rm['young']])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    lo, hi = np.percentile(ro['d'], [1, 99])
    bins = np.linspace(lo, hi, 15)
    for d, col, lab in [(do, 'steelblue', f'Old (N = {len(do)})'),
                        (dy, 'darkorange', f'Young (N = {len(dy)})')]:
        ax.hist(d, bins=bins, density=True, histtype='stepfilled', color=col, alpha=0.35)
        ax.hist(d, bins=bins, density=True, histtype='step', color=col, lw=2.2, label=lab)
        ax.axvline(np.median(d), color=col, ls='--', lw=2.2)
    ax.axvline(np.median(do) + model_shift, color='k', ls=':', lw=2.2,
               label=f'Models: young offset ({model_shift:+.2f} mag)')
    ax.text(0.03, 0.96, f"AUC = {ro['AUC']:.2f}", transform=ax.transAxes, va='top', fontsize=19,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))
    ax.set_xlabel(rf'Offset in $({name})_0$ (mag)', fontsize=18)
    ax.set_ylabel('Normalised number', fontsize=18)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)          # espacio para la leyenda
    ax.tick_params(labelsize=15)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(a.outdir, 'narrowband_age_offset_single')
    plt.savefig(out + '.pdf', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f"OK figura de un panel: {out}.pdf (desplazamiento de los modelos: {model_shift:+.3f} mag)")


def plot_offset_hist(store, names, a):
    """Distribucion de los desplazamientos respecto de la relacion de los viejos."""
    fig, axes = plt.subplots(2, len(names), figsize=(6.4 * len(names), 9.2), squeeze=False)
    for j, name in enumerate(names):
        for i, (r, lab) in enumerate(zip(store[name], ['Best-fit models', f'Observed (S/N > {a.snr:g})'])):
            ax = axes[i, j]
            if r is None:
                ax.set_visible(False)
                continue
            do, dy = r['d'][~r['young']], r['d'][r['young']]
            lo, hi = np.percentile(r['d'], [1, 99])
            bins = np.linspace(lo, hi, 18 if i == 1 else 26)
            ax.hist(do, bins=bins, density=True, histtype='stepfilled', color='steelblue', alpha=0.35)
            ax.hist(do, bins=bins, density=True, histtype='step', color='steelblue', lw=2,
                    label=f'Old (N = {len(do)})')
            ax.hist(dy, bins=bins, density=True, histtype='stepfilled', color='darkorange', alpha=0.35)
            ax.hist(dy, bins=bins, density=True, histtype='step', color='darkorange', lw=2,
                    label=f'Young (N = {len(dy)})')
            ax.axvline(np.median(do), color='steelblue', ls='--', lw=2)
            ax.axvline(np.median(dy), color='darkorange', ls='--', lw=2)
            ax.axvline(0, color='k', lw=0.8, alpha=0.5)
            ax.text(0.03, 0.96, f"{lab}\nAUC = {r['AUC']:.2f}", transform=ax.transAxes, va='top',
                    fontsize=17, bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))
            ax.set_xlabel(rf'$\Delta({name})$ at fixed $(g-z)_0$ (mag)', fontsize=19)
            ax.set_ylabel('Normalised number', fontsize=19)
            ax.tick_params(labelsize=16)
            ax.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    out = os.path.join(a.outdir, 'narrowband_age_offsets')
    plt.savefig(out + '.pdf', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=130, bbox_inches='tight')
    print(f"OK figura de desplazamientos: {out}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='out/results.fits')
    ap.add_argument('--obs', default='out/observations.fits')
    ap.add_argument('--age-cut', type=float, default=8.0)
    ap.add_argument('--snr', type=float, default=3.0)
    ap.add_argument('--nplot', type=int, default=4, help='colores a mostrar en la figura')
    ap.add_argument('--plot-colours', nargs='+', default=None,
                    help="colores a dibujar, p.ej. J0378-J0410 u-g (por defecto, los mejores angostos)")
    ap.add_argument('--single', action='store_true',
                    help="un solo panel con los colores observados (con --hist: histogramas de los desplazamientos)")
    ap.add_argument('--hist', action='store_true',
                    help="figura de las distribuciones de los desplazamientos (en vez de color-color)")
    ap.add_argument('--deg', type=int, default=2, help='grado del polinomio de la relacion de los viejos')
    ap.add_argument('--gz-range', type=float, nargs=2, default=None,
                    help='usar solo GCs en este rango de (g-z), p.ej. 0.65 1.10')
    ap.add_argument('--outdir', default='narrowband_age_diagnostic')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    res = Table.read(a.results, hdu=1).to_pandas()
    obs = Table.read(a.obs, hdu=1).to_pandas()
    for t in (res, obs):
        t['id'] = t['id'].astype(str).str.strip()
    res = res[res['best.reduced_chi_square'] <= 2.0]
    df = res.merge(obs, on='id', how='inner')
    young = (df['bayes.stellar.age_m_star'].values / 1000.0) <= a.age_cut
    print(f"GCs con chi2 <= 2: {len(df)} (jovenes {young.sum()}, viejos {(~young).sum()})")

    M, O, SN = {}, {}, {}
    for k, b in BANDS.items():
        if f'best.{b}' not in df or b not in df:
            raise SystemExit(f"Falta la banda {b} (columnas best.{b} / {b}). "
                             f"Columnas: {[c for c in df.columns if 'F0' in c or 'sdss' in c][:20]}")
        M[k] = mag(df[f'best.{b}'])
        O[k] = mag(df[b])
        err = df.get(f'{b}_err', pd.Series(np.nan, index=df.index)).values
        SN[k] = df[b].values / err

    rows, store = [], {}
    gz_m, gz_o = M['g'] - M['z'], O['g'] - O['z']
    for b1, b2 in COLOURS:
        name = f'{b1}-{b2}'
        rm = separation(gz_m, M[b1] - M[b2], young, a.deg, a.gz_range)
        sel = np.ones(len(df), bool)
        for b in (b1, b2):
            if b in NARROW:
                sel &= SN[b] > a.snr
        ro = separation(gz_o[sel], (O[b1] - O[b2])[sel], young[sel], a.deg, a.gz_range)
        store[name] = (rm, ro)
        rows.append(dict(colour=name,
                         S_model=rm['S'] if rm else np.nan, AUC_model=rm['AUC'] if rm else np.nan,
                         shift_model=rm['shift'] if rm else np.nan,
                         N_obs_young=ro['N_young'] if ro else 0, N_obs_old=ro['N_old'] if ro else 0,
                         S_obs=ro['S'] if ro else np.nan, AUC_obs=ro['AUC'] if ro else np.nan,
                         p_obs=ro['p'] if ro else np.nan))
    tab = pd.DataFrame(rows).sort_values('AUC_model', ascending=False)
    pd.set_option('display.width', 160)
    print("\nSeparacion jovenes-viejos a (g - z) fijo (S en unidades de la dispersion de los viejos):")
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    tab.to_csv(os.path.join(a.outdir, 'narrowband_age_separation.csv'), index=False, float_format='%.4f')

    # figura: los mejores colores (segun el modelo), modelo arriba y observado abajo
    best = [c for c in tab['colour'] if c.split('-')[0] in NARROW or c.split('-')[1] in NARROW][:a.nplot]
    if a.plot_colours:
        best = [c for c in a.plot_colours if c in store]
    fig, axes = plt.subplots(2, len(best), figsize=(6.2 * len(best), 10.5), squeeze=False)
    for j, name in enumerate(best):
        for i, (r, lab) in enumerate(zip(store[name], ['Best-fit models', f'Observed (S/N>{a.snr:g})'])):
            ax = axes[i, j]
            if r is None:
                ax.set_visible(False)
                continue
            ax.scatter(r['gz'][~r['young']], r['c'][~r['young']], s=22, color='steelblue', alpha=0.45,
                       lw=0, label=f"Old (N={(~r['young']).sum()})")
            ax.scatter(r['gz'][r['young']], r['c'][r['young']], s=34, color='darkorange', alpha=0.8,
                       edgecolors='k', lw=0.3, label=f"Young (N={r['young'].sum()})")
            xx = np.linspace(np.nanpercentile(r['gz'], 1), np.nanpercentile(r['gz'], 99), 100)
            ax.plot(xx, np.polyval(r['poly'], xx), 'k--', lw=1.5)
            ax.text(0.04, 0.96, f"{lab}\nAUC = {r['AUC']:.2f}",
                    transform=ax.transAxes, va='top', fontsize=22,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))
            ax.set_xlabel(r'$(g-z)_0$', fontsize=21)
            ax.set_ylabel(f'$({name})_0$', fontsize=21)
            ax.tick_params(labelsize=17)
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=100, markerscale=1.5)
    plt.tight_layout()
    out = os.path.join(a.outdir, 'narrowband_age_diagnostic')
    plt.savefig(out + '.pdf', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nOK figura: {out}.pdf | tabla: {a.outdir}/narrowband_age_separation.csv")
    if a.single and not a.hist:
        plot_colour_single(store, best[0], a)
    elif a.hist and a.single:
        plot_offset_single(store, best[0], a)
    elif a.hist:
        plot_offset_hist(store, best, a)
    print("Recordatorio: la separacion OBSERVADA es en parte circular (las edades salen de los mismos colores).")


if __name__ == '__main__':
    main()
