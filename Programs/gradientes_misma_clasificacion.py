#!/usr/bin/env python3
"""
gradientes_misma_clasificacion.py

Comparacion JUSTA de los gradientes entre modelos (comentario 3 iii del arbitro):
mismos objetos y misma clasificacion en MP / Int / MR (la del paper), cambiando
solo las metalicidades por las de cada configuracion de CIGALE.

Asi se separan dos efectos:
  * el cambio de las metalicidades de cada GC  (lo que mide este script), y
  * el cambio de la descomposicion GMM en si    (lo que mide gmm_analysis_final_paper_v2.py
    corrido en cada carpeta).

Uso (desde anac_data):
  python ../Programs/gradientes_misma_clasificacion.py \\
      --ref CIGALE-DEcam-extinction  CIGALE-bc03 CIGALE-bursty
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import linregress, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_analysis_final_paper_v2 import (get_labels_forced_n3, calculate_projected_radii,  # noqa: E402
                                         POP_SHORT_3)

Z_SOLAR = 0.02


def load_feh(run, chi2_cut=True):
    t = Table.read(os.path.join(run, "out", "results.fits"), hdu=1).to_pandas()
    t["id_str"] = t["id"].astype(str).str.strip()
    if chi2_cut:
        t = t[t["best.reduced_chi_square"] <= 2.0]
    t["FeH"] = np.log10(t["bayes.stellar.metallicity"] / Z_SOLAR)
    t["Age_Gyr"] = t["bayes.stellar.age_m_star"] / 1000.0
    return t


def slope_stats(x, y, nboot, rng):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lr = linregress(x, y)
    bs = np.empty(nboot)
    for k in range(nboot):
        j = rng.integers(0, len(x), len(x))
        bs[k] = linregress(x[j], y[j]).slope
    return dict(slope=lr.slope, err=lr.stderr, boot_std=bs.std(), p_neg=np.mean(bs < 0), N=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="carpetas de las otras corridas")
    ap.add_argument("--ref", default="CIGALE-DEcam-extinction")
    ap.add_argument("--catalog", default="Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # --- referencia: exactamente como gmm_analysis_final_paper(_v2).py ---
    df = load_feh(a.ref)
    Q1, Q3 = df["FeH"].quantile(0.01), df["FeH"].quantile(0.99)
    df = df[(df["FeH"] >= Q1 - 1.5*(Q3-Q1)) & (df["FeH"] <= Q3 + 1.5*(Q3-Q1))]
    cat = pd.read_csv(a.catalog)
    cat["T17ID_str"] = cat["T17ID"].astype(str).str.strip()
    df = pd.merge(df, cat[["RAJ2000", "DEJ2000", "T17ID_str"]], left_on="id_str",
                  right_on="T17ID_str", how="left").rename(columns={"RAJ2000": "RA", "DEJ2000": "DEC"})
    print(f"Referencia {a.ref}: {len(df)} GCs")
    print("Clasificacion del paper (muestra completa):")
    lab_full = get_labels_forced_n3(df["FeH"].values)[0]
    old = (df["Age_Gyr"] > 8.0).values
    print("Clasificacion del paper (GCs viejos):")
    lab_old = np.full(len(df), -1)
    lab_old[old] = get_labels_forced_n3(df.loc[old, "FeH"].values)[0]
    radii, valid = calculate_projected_radii(df)
    logr = np.full(len(df), np.nan)
    logr[valid.values] = np.log10(radii)

    models = [(os.path.basename(a.ref.rstrip("/")), df.set_index("id_str")["FeH"])]
    for run in a.runs:
        t = load_feh(run, chi2_cut=False).set_index("id_str")
        models.append((os.path.basename(run.rstrip("/")), t["FeH"]))
    ids = df["id_str"].values

    for title, labels, sel in [("Muestra completa (clasificacion del paper)", lab_full, np.ones(len(df), bool)),
                               ("GCs viejos (>8 Gyr segun el paper; clasificacion del paper)", lab_old, old)]:
        print(f"\n{title}")
        print(f"{'modelo':26s} " + "  ".join(f"{p:>27s}" for p in POP_SHORT_3) + f"  {'global':>27s}")
        for name, feh_map in models:
            feh = feh_map.reindex(ids).values
            cells = []
            for i in range(3):
                m = sel & (labels == i)
                st = slope_stats(logr[m], feh[m], a.nboot, rng)
                cells.append(f"{st['slope']:+.3f}±{st['boot_std']:.3f} ({st['N']:3d}) [{st['p_neg']:.2f}]")
            stg = slope_stats(logr[sel], feh[sel], a.nboot, rng)
            cells.append(f"{stg['slope']:+.3f}±{stg['boot_std']:.3f} ({stg['N']:3d}) [{stg['p_neg']:.2f}]")
            print(f"{name:26s} " + "  ".join(c.rjust(27) for c in cells))
    print("\nFormato: pendiente OLS ± error bootstrap (N) [fraccion de pendientes bootstrap < 0]")
    print("Mismos objetos y misma asignacion a MP/Int/MR en todos los modelos; "
          "solo cambian las metalicidades.")


if __name__ == "__main__":
    main()
