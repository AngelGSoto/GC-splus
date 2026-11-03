#!/usr/bin/env python3
"""
gradient_age_robustness.py -- respuesta al arbitro (robustez de las pendientes, Sect. 4.2).

Gradiente de metalicidad ([Fe/H] vs log R) de la muestra completa (chi2_nu <= 2) y de los GCs
viejos (> AGE_CUT Gyr) y jovenes (<= AGE_CUT), con minimos cuadrados sin pesos, y sus
incertidumbres por jackknife y por bootstrap con reemplazo.

Resultado (29 oct 2026, corrida del paper):
  Todos    N=884  -0.110 +/- 0.035 | jackknife 0.035 | bootstrap 0.035 | <0: 100%
  Viejos   N=720  -0.129 +/- 0.039 | jackknife 0.039 | bootstrap 0.038 | <0: 100%
  Jovenes  N=164  -0.013 +/- 0.070 | jackknife 0.070 | bootstrap 0.067 | <0:  57%

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/gradient_age_robustness.py
"""
import argparse

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import linregress

RA0, DEC0, D_KPC = 201.365063, -43.019112, 3800.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='out/results.fits')
    ap.add_argument('--catalog', default='../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    ap.add_argument('--age-cut', type=float, default=8.0)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    res = Table.read(a.results, hdu=1).to_pandas()
    res['id'] = res['id'].astype(str).str.strip()
    cat = pd.read_csv(a.catalog)
    cat['id'] = cat['T17ID'].astype(str).str.strip()
    df = res[res['best.reduced_chi_square'] <= 2].merge(cat[['id', 'RAJ2000', 'DEJ2000']], on='id')

    dx = (df['RAJ2000'] - RA0) * np.cos(np.radians(DEC0))
    dy = df['DEJ2000'] - DEC0
    df['logR'] = np.log10(np.radians(np.hypot(dx, dy)) * D_KPC)          # radio proyectado (kpc)
    df['FeH'] = np.log10(df['bayes.stellar.metallicity'] / 0.02)
    age = df['bayes.stellar.age_m_star'] / 1000.0

    rng = np.random.default_rng(a.seed)
    for name, m in [('Todos', np.ones(len(df), bool)),
                    (f'Viejos >{a.age_cut:g}', age > a.age_cut),
                    (f'Jovenes <={a.age_cut:g}', age <= a.age_cut)]:
        x, y = df.loc[m, 'logR'].values, df.loc[m, 'FeH'].values
        r = linregress(x, y)
        n = len(x)
        jk = np.array([linregress(np.delete(x, i), np.delete(y, i)).slope for i in range(n)])
        s_jk = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        bs = np.array([linregress(x[i], y[i]).slope for i in (rng.integers(0, n, n) for _ in range(a.nboot))])
        print(f"{name:12s} N={n:4d} pendiente={r.slope:+.3f} +/- {r.stderr:.3f} (formal, p={r.pvalue:.3g}) | "
              f"jackknife +/- {s_jk:.3f} | bootstrap +/- {bs.std():.3f} | fraccion < 0: {np.mean(bs < 0):.3f}")


if __name__ == '__main__':
    main()
