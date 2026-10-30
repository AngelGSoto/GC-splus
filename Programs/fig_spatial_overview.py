#!/usr/bin/env python3
"""
fig_spatial_overview.py -- nueva Fig. 7 (respuesta al arbitro: la figura original estaba
demasiado cargada). Tres paneles con puntos pequenos y la misma escala de color ([Fe/H]):
  (a) GCs viejos (> 8 Gyr), (b) GCs jovenes (<= 8 Gyr), (c) ampliacion de la region central
  (todos los GCs con buen ajuste). Los GCs con mal ajuste (chi2 > 2) se muestran como puntos
  grises muy pequenos en (a) y (b), como referencia de la cobertura.
X crece hacia el Este (a la derecha, como en la figura original) e Y hacia el Norte.

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/fig_spatial_overview.py
  python ../../Programs/fig_spatial_overview.py --zoom 30 --age-cut 8
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_analysis_final_paper_v2 import RA_CENTER, DEC_CENTER, DISTANCE_MPC  # noqa: E402


def tangent_xy_kpc(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(RA_CENTER), np.radians(DEC_CENTER)
    cosc = np.sin(d0)*np.sin(d) + np.cos(d0)*np.cos(d)*np.cos(a - a0)
    xi = np.cos(d)*np.sin(a - a0) / cosc
    eta = (np.cos(d0)*np.sin(d) - np.sin(d0)*np.cos(d)*np.cos(a - a0)) / cosc
    return xi * DISTANCE_MPC * 1000, eta * DISTANCE_MPC * 1000


def rings(ax, radii, label=True, fs=16):
    t = np.linspace(0, 2*np.pi, 400)
    for r in radii:
        ax.plot(r*np.cos(t), r*np.sin(t), ls='--', lw=1.1, color='0.55', zorder=1)
        if label:
            import matplotlib.patheffects as pe
            ax.text(r*np.cos(np.radians(58)), r*np.sin(np.radians(58)), f"{r:g} kpc",
                    fontsize=fs, color='0.3', ha='left', va='bottom', zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="out/results.fits")
    ap.add_argument("--catalog", default="../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
    ap.add_argument("--age-cut", type=float, default=8.0)
    ap.add_argument("--zoom", type=float, default=30.0, help="semi-ancho del panel central (kpc)")
    ap.add_argument("--vmin", type=float, default=-1.6)
    ap.add_argument("--vmax", type=float, default=0.1)
    ap.add_argument("--size", type=float, default=34, help="tamano de los puntos en (a) y (b)")
    ap.add_argument("--size-zoom", type=float, default=55, help="tamano de los puntos en (c)")
    ap.add_argument("--outdir", default="figures_paper")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t = Table.read(a.results, hdu=1).to_pandas()
    t['id_str'] = t['id'].astype(str).str.strip()
    t['FeH'] = np.log10(t['bayes.stellar.metallicity'] / 0.02)
    t['Age'] = t['bayes.stellar.age_m_star'] / 1000.0
    t['good'] = t['best.reduced_chi_square'] <= 2.0
    cat = pd.read_csv(a.catalog)
    cat['id_str'] = cat['T17ID'].astype(str).str.strip()
    t = t.merge(cat[['id_str', 'RAJ2000', 'DEJ2000']], on='id_str', how='inner')
    t = t[t['RAJ2000'].notna()]
    t['X'], t['Y'] = tangent_xy_kpc(t['RAJ2000'].values, t['DEJ2000'].values)

    good, poor = t[t['good']], t[~t['good']]
    old = good[good['Age'] > a.age_cut]
    young = good[good['Age'] <= a.age_cut]
    print(f"Buen ajuste: {len(good)} (viejos {len(old)}, jovenes {len(young)}) | mal ajuste: {len(poor)}")

    lim = np.ceil(np.nanmax(np.abs(np.r_[t['X'], t['Y']])) / 25) * 25
    fig, axes = plt.subplots(1, 3, figsize=(23, 8.2), gridspec_kw=dict(wspace=0.17))
    kw = dict(cmap='RdYlBu_r', vmin=a.vmin, vmax=a.vmax, edgecolors='k', linewidths=0.4, zorder=3)

    panels = [(axes[0], old, f"(a) Old GCs ($>{a.age_cut:g}$ Gyr, N = {len(old)})", lim, True),
              (axes[1], young, f"(b) Young GCs ($\\leq{a.age_cut:g}$ Gyr, N = {len(young)})", lim, True),
              (axes[2], good, f"(c) Central region (all, N = "
                              f"{int(((good['X'].abs() < a.zoom) & (good['Y'].abs() < a.zoom)).sum())})",
               a.zoom, False)]
    for ax, df, title, L, show_poor in panels:
        if show_poor:
            ax.scatter(poor['X'], poor['Y'], s=3, color='0.75', zorder=2, lw=0)
            rings(ax, [30, 60, 100, 150])
        else:
            rings(ax, [5, 10, 20], fs=15)
        sc = ax.scatter(df['X'], df['Y'], c=df['FeH'], s=(a.size if show_poor else a.size_zoom), **kw)
        ax.scatter([0], [0], marker='*', s=260, color='gold', edgecolors='k', lw=1, zorder=5)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
        ax.set_title(title, fontsize=22)
        ax.set_xlabel(r'$\Delta X$ (kpc)', fontsize=24)
        ax.tick_params(labelsize=20)
    axes[0].set_ylabel(r'$\Delta Y$ (kpc)', fontsize=24)
    axes[1].tick_params(labelleft=False)          # (a) y (b) comparten la escala en Y
    for ax in axes[:2]:                           # evitar que se encimen 200 y -200
        ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    # recuadro de la region ampliada en el panel (a)
    z = a.zoom
    axes[0].plot([-z, z, z, -z, -z], [-z, -z, z, z, -z], color='k', lw=1.2, zorder=4)
    # orientacion
    axes[0].annotate('', xy=(0.88, 0.07), xytext=(0.78, 0.07), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', lw=1.5))
    axes[0].annotate('', xy=(0.78, 0.17), xytext=(0.78, 0.07), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', lw=1.5))
    axes[0].text(0.895, 0.07, 'E', transform=axes[0].transAxes, fontsize=18, va='center')
    axes[0].text(0.78, 0.18, 'N', transform=axes[0].transAxes, fontsize=18, ha='center')

    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.012)
    cb.set_label('[Fe/H] (dex)', fontsize=24)
    cb.ax.tick_params(labelsize=20)

    base = os.path.join(a.outdir, 'Spatial_overview_age_FeH')
    plt.savefig(base + '.pdf', bbox_inches='tight')
    plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
    print(f"OK figura: {base}.pdf")


if __name__ == '__main__':
    main()
