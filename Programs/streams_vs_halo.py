#!/usr/bin/env python3
"""
streams_vs_halo.py -- comparacion de los GCs asociados a subestructuras del halo con los
GCs del halo "relajado" (pedido del arbitro, comentario 11).

* Usa EXACTAMENTE la clasificacion del paper (get_labels_forced_n3 de
  gmm_analysis_final_paper_v2.py: 240 / 435 / 209).
* Asociacion: mismas regiones y radio que crossmatch_streams_with_gmm.py (10 kpc,
  centro mas cercano). Reproduce la Tabla K.1 como control.
* Halo relajado: GCs a mas de RADIUS_KPC de cualquier region, en el MISMO rango de
  radios proyectados que los GCs de las subestructuras (y, como alternativa, con
  ponderacion para igualar la distribucion radial).
* Compara: [Fe/H] (KS, AD), edades (KS), fracciones de MP/Int/MR (con IC binomial).

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/streams_vs_halo.py
"""
import os
import sys

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.stats import ks_2samp, anderson_ksamp, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_analysis_final_paper_v2 import (get_labels_forced_n3, calculate_projected_radii,  # noqa: E402
                                         DISTANCE_MPC)

RADIUS_KPC = 10.0
KPC_TO_DEG = np.degrees(1.0 / (DISTANCE_MPC * 1000.0))
REGIONS = {
    'Dw3 core': ('13:30:21.5', '-42:11:33'), 'Dw3 North': ('13:28:27.6', '-41:51:00'),
    'Dw3 South': ('13:31:02.6', '-42:21:54'), 'Dw3 SE': ('13:32:18.0', '-42:36:54'),
    'Dw3 NW': ('13:23:57.6', '-41:42:00'), 'Second stream': ('13:31:54.0', '-41:51:00'),
    'Shell 1': ('13:26:45.6', '-42:22:30'), 'Shell 2': ('13:23:21.6', '-43:14:06'),
}
POPS = ['MP', 'Int', 'MR']


def group(name):
    return 'Dw3 stream' if name.startswith('Dw3') else name


def frac_ci(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    ci = binomtest(k, n).proportion_ci(confidence_level=0.68)
    return k / n, ci.low, ci.high


def main():
    df = Table.read('out/results.fits', hdu=1).to_pandas()
    df['id_str'] = df['id'].astype(str).str.strip()
    df = df[df['best.reduced_chi_square'] <= 2.0].copy()
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / 0.02)
    df['Age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0
    q1, q3 = df['FeH'].quantile(0.01), df['FeH'].quantile(0.99)
    df = df[(df['FeH'] >= q1 - 1.5*(q3-q1)) & (df['FeH'] <= q3 + 1.5*(q3-q1))]
    cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    cat['id_str'] = cat['T17ID'].astype(str).str.strip()
    df = df.merge(cat[['id_str', 'RAJ2000', 'DEJ2000']], on='id_str', how='left').reset_index(drop=True)
    df = df.rename(columns={'RAJ2000': 'RA', 'DEJ2000': 'DEC'})

    print("Clasificacion del paper:")
    lab = get_labels_forced_n3(df['FeH'].values)[0]
    df['pop'] = [POPS[i] for i in lab]
    R, valid = calculate_projected_radii(df)
    df = df[valid].reset_index(drop=True)
    df['R_kpc'] = R

    gc = SkyCoord(df['RA'].values*u.deg, df['DEC'].values*u.deg)
    names = list(REGIONS)
    seps = np.column_stack([gc.separation(SkyCoord(ra, de, unit=('hourangle', 'deg'))).deg / KPC_TO_DEG
                            for ra, de in REGIONS.values()])
    best = np.argmin(seps, axis=1)
    inside = seps[np.arange(len(df)), best] <= RADIUS_KPC
    df['region'] = np.where(inside, np.array(names)[best], 'None')
    df['group'] = [group(r) if r != 'None' else 'None' for r in df['region']]

    # ---- control: Tabla K.1 ----
    print(f"\nControl (Tabla K.1, radio {RADIUS_KPC:g} kpc):")
    print(f"{'region':<16} {'N':>3} {'[Fe/H]':>14} {'edad':>14} {'f_Int':>6}")
    for key, col in [(n, 'region') for n in names] + [(g, 'group') for g in ['Dw3 stream', 'Second stream', 'Shell 1', 'Shell 2']]:
        s = df[df[col] == key]
        if len(s) == 0:
            continue
        print(f"{key + (' (all)' if col == 'group' else ''):<16} {len(s):>3} "
              f"{s.FeH.mean():6.2f}±{s.FeH.std():5.2f} {s.Age_Gyr.mean():6.2f}±{s.Age_Gyr.std():5.2f} "
              f"{(s['pop'] == 'Int').mean():6.2f}")

    # ---- comparacion con el halo relajado ----
    st_all = df[df['group'] != 'None']
    halo = df[df['group'] == 'None']
    rmin, rmax = st_all.R_kpc.min(), st_all.R_kpc.max()
    halo_r = halo[(halo.R_kpc >= rmin) & (halo.R_kpc <= rmax)]
    print(f"\nGCs en subestructuras: {len(st_all)} (R = {rmin:.0f}-{rmax:.0f} kpc) | "
          f"halo relajado en ese rango de radios: {len(halo_r)}")

    def compare(st, label):
        # halo relajado en el MISMO rango de radios que este grupo
        halo_r = halo[(halo.R_kpc >= st.R_kpc.min()) & (halo.R_kpc <= st.R_kpc.max())]
        print(f"\n== {label} (N={len(st)}) frente al halo relajado (N={len(halo_r)}) ==")
        print(f"   [Fe/H] medio: {st.FeH.mean():+.2f} ± {st.FeH.std()/np.sqrt(len(st)):.2f}  "
              f"vs {halo_r.FeH.mean():+.2f} ± {halo_r.FeH.std()/np.sqrt(len(halo_r)):.2f} (error de la media)")
        print(f"   [Fe/H]: KS p = {ks_2samp(st.FeH, halo_r.FeH).pvalue:.3g}   "
              f"AD p ~ {anderson_ksamp([st.FeH.values, halo_r.FeH.values]).significance_level:.3g}")
        print(f"   edad media: {st.Age_Gyr.mean():.2f} vs {halo_r.Age_Gyr.mean():.2f} Gyr;  "
              f"KS p = {ks_2samp(st.Age_Gyr, halo_r.Age_Gyr).pvalue:.3g}")
        print(f"   R mediano: {st.R_kpc.median():.0f} vs {halo_r.R_kpc.median():.0f} kpc")
        for p in POPS:
            f1, l1, h1 = frac_ci(int((st['pop'] == p).sum()), len(st))
            f2, l2, h2 = frac_ci(int((halo_r['pop'] == p).sum()), len(halo_r))
            print(f"   f_{p:<3}: {f1:.2f} [{l1:.2f},{h1:.2f}]  vs  {f2:.2f} [{l2:.2f},{h2:.2f}]")

    compare(st_all, "Todas las subestructuras")
    compare(df[df['group'] == 'Dw3 stream'], "Corriente Dw3")
    os.makedirs('Streams', exist_ok=True)
    df[['id_str', 'RA', 'DEC', 'R_kpc', 'FeH', 'Age_Gyr', 'pop', 'region', 'group']].to_csv(
        'Streams/streams_vs_halo_assignment.csv', index=False)
    print("\nOK: Streams/streams_vs_halo_assignment.csv")


if __name__ == "__main__":
    main()
