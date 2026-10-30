#!/usr/bin/env python3
"""
spacial_distribution.py -- pruebas preliminares para el comentario 11 del arbitro:
distribucion espacial de las poblaciones MP, Int y MR (misma clasificacion del paper).

Calcula, para cada poblacion y para la muestra completa (referencia):
  1. Radio proyectado mediano (con IC 68% por bootstrap), cuartiles, y pruebas KS y
     Anderson-Darling entre las distribuciones radiales de cada par de poblaciones.
  2. Elipses de correlacion (momentos de segundo orden de X, Y): cociente de ejes b/a
     y angulo de posicion (PA, de N hacia E) del eje mayor, con errores por bootstrap.
     Se calculan para todos los radios y para R < --rlim kpc (zona mas uniforme).
  3. Distribucion azimutal (angulo de posicion de cada GC, plegado a 0-180 grados):
     pruebas KS entre poblaciones. Si se da --pa-gal, tambien respecto del eje mayor.
  4. Figura: mapas de densidad suavizados (KDE) de MP, Int y MR con sus elipses, y
     paneles con las distribuciones radiales acumuladas y azimutales.

Como la completitud y la cobertura no son uniformes y afectan por igual a todas las
poblaciones, lo relevante son las DIFERENCIAS entre poblaciones, no los valores absolutos.

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/spacial_distribution.py
  python ../../Programs/spacial_distribution.py --pa-gal 35 --rlim 60
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.table import Table
from scipy.stats import ks_2samp, anderson_ksamp, gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_analysis_final_paper_v2 import (get_labels_forced_n3, COLORS_3, POP_SHORT_3,  # noqa: E402
                                         RA_CENTER, DEC_CENTER, DISTANCE_MPC)

POPS = ["MP", "Int", "MR"]


def load(results, catalog):
    df = Table.read(results, hdu=1).to_pandas()
    df["id_str"] = df["id"].astype(str).str.strip()
    df = df[df["best.reduced_chi_square"] <= 2.0].copy()
    df["FeH"] = np.log10(df["bayes.stellar.metallicity"] / 0.02)
    q1, q3 = df["FeH"].quantile(0.01), df["FeH"].quantile(0.99)
    df = df[(df["FeH"] >= q1 - 1.5*(q3-q1)) & (df["FeH"] <= q3 + 1.5*(q3-q1))]
    cat = pd.read_csv(catalog)
    cat["T17ID_str"] = cat["T17ID"].astype(str).str.strip()
    df = df.merge(cat[["RAJ2000", "DEJ2000", "T17ID_str"]], left_on="id_str",
                  right_on="T17ID_str", how="left")
    return df.reset_index(drop=True)


def tangent_xy_kpc(ra, dec):
    """Coordenadas en el plano tangente (kpc): X hacia el Este, Y hacia el Norte."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(RA_CENTER), np.radians(DEC_CENTER)
    cosc = np.sin(d0)*np.sin(d) + np.cos(d0)*np.cos(d)*np.cos(a - a0)
    xi = np.cos(d)*np.sin(a - a0) / cosc
    eta = (np.cos(d0)*np.sin(d) - np.sin(d0)*np.cos(d)*np.cos(a - a0)) / cosc
    k = DISTANCE_MPC * 1000.0          # kpc por radian
    return xi * k, eta * k


def ellipse_params(x, y):
    """b/a y PA (grados, de N hacia E) del eje mayor, a partir de los momentos."""
    cov = np.cov(np.vstack([x - x.mean(), y - y.mean()]))
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]                    # autovector del autovalor mayor
    pa = np.degrees(np.arctan2(major[0], major[1])) % 180.0   # atan2(E, N)
    return float(np.sqrt(w[0] / w[1])), float(pa), cov


def circ_diff(a, b):
    d = (a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def boot_ellipse(x, y, rng, nboot):
    q0, pa0, _ = ellipse_params(x, y)
    qs, pas = [], []
    for _ in range(nboot):
        j = rng.integers(0, len(x), len(x))
        q, pa, _ = ellipse_params(x[j], y[j])
        qs.append(q)
        pas.append(pa0 + ((pa - pa0 + 90) % 180 - 90))   # desenrollar alrededor de pa0
    return q0, np.std(qs), pa0, np.std(pas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="out/results.fits")
    ap.add_argument("--catalog", default="../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
    ap.add_argument("--pa-gal", type=float, default=None,
                    help="PA del eje mayor de la galaxia (grados, de N hacia E), opcional")
    ap.add_argument("--rlim", type=float, default=60.0, help="radio (kpc) para las elipses internas")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default="distribucion_espacial")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.results, a.catalog)
    print("Clasificacion del paper (muestra completa):")
    labels = get_labels_forced_n3(df["FeH"].values)[0]
    ok = df["RAJ2000"].notna().values
    X, Y = tangent_xy_kpc(df.loc[ok, "RAJ2000"].values, df.loc[ok, "DEJ2000"].values)
    lab = labels[ok]
    R = np.hypot(X, Y)
    PA = np.degrees(np.arctan2(X, Y)) % 180.0          # azimut plegado (simetria)
    groups = {"All": np.ones(len(R), bool)}
    groups.update({p: lab == i for i, p in enumerate(POPS)})
    out = {"N_total": int(len(R)), "rlim_kpc": a.rlim, "pa_gal": a.pa_gal, "groups": {}}

    # ---- 1. distribuciones radiales ----
    print(f"\n1) Distribucion radial (N total = {len(R)})")
    print(f"{'grupo':5s} {'N':>4s} {'R_mediano':>10s} {'IC68':>16s} {'R25':>7s} {'R75':>7s}")
    for g, m in groups.items():
        r = R[m]
        meds = [np.median(r[rng.integers(0, len(r), len(r))]) for _ in range(a.nboot)]
        p16, p84 = np.percentile(meds, [16, 84])
        out["groups"][g] = dict(N=int(m.sum()), R_median=float(np.median(r)),
                                R_median_p16=float(p16), R_median_p84=float(p84),
                                R25=float(np.percentile(r, 25)), R75=float(np.percentile(r, 75)))
        print(f"{g:5s} {m.sum():4d} {np.median(r):10.1f} [{p16:6.1f},{p84:6.1f}] "
              f"{np.percentile(r, 25):7.1f} {np.percentile(r, 75):7.1f}")
    print("\n   Pruebas entre poblaciones (distribucion radial):")
    out["radial_tests"] = {}
    for i in range(3):
        for j in range(i + 1, 3):
            r1, r2 = R[groups[POPS[i]]], R[groups[POPS[j]]]
            ks = ks_2samp(r1, r2)
            ad = anderson_ksamp([r1, r2])
            key = f"{POPS[i]}-{POPS[j]}"
            out["radial_tests"][key] = dict(KS_p=float(ks.pvalue), AD_p=float(ad.significance_level))
            print(f"   {key:7s}: KS p = {ks.pvalue:.3g}   AD p ~ {ad.significance_level:.3g}")

    # ---- 2. elipses de correlacion ----
    print(f"\n2) Elipses de correlacion (b/a y PA del eje mayor, de N hacia E)")
    print(f"{'grupo':5s} {'zona':>10s} {'N':>4s} {'b/a':>12s} {'PA (grados)':>16s}")
    out["ellipses"] = {}
    for zone, zmask in [("todos", np.ones(len(R), bool)), (f"R<{a.rlim:g}", R < a.rlim)]:
        for g, m in groups.items():
            mm = m & zmask
            if mm.sum() < 20:
                continue
            q, eq, pa, epa = boot_ellipse(X[mm], Y[mm], rng, a.nboot)
            out["ellipses"][f"{g}|{zone}"] = dict(N=int(mm.sum()), q=q, q_err=eq, PA=pa, PA_err=epa)
            print(f"{g:5s} {zone:>10s} {mm.sum():4d} {q:6.2f}±{eq:4.2f} {pa:9.1f}±{epa:5.1f}")

    # ---- 3. distribucion azimutal ----
    print("\n3) Distribucion azimutal (PA plegado a 0-180 grados): KS entre poblaciones")
    out["azimuthal_tests"] = {}
    for i in range(3):
        for j in range(i + 1, 3):
            ks = ks_2samp(PA[groups[POPS[i]]], PA[groups[POPS[j]]])
            key = f"{POPS[i]}-{POPS[j]}"
            out["azimuthal_tests"][key] = float(ks.pvalue)
            print(f"   {key:7s}: KS p = {ks.pvalue:.3g}")
    if a.pa_gal is not None:
        dpa = circ_diff(PA, a.pa_gal)          # 0 = eje mayor, 90 = eje menor
        print(f"\n   Fraccion de GCs a menos de 45 grados del eje mayor (PA_gal = {a.pa_gal:g}):")
        out["near_major_axis_frac"] = {}
        for g, m in groups.items():
            f = np.mean(dpa[m] < 45)
            ef = np.std([np.mean(dpa[m][rng.integers(0, m.sum(), m.sum())] < 45)
                         for _ in range(a.nboot)])
            out["near_major_axis_frac"][g] = dict(f=float(f), err=float(ef))
            print(f"   {g:5s}: {f:.2f} ± {ef:.2f}  (0.50 si es isotropa)")

    # ---- 4. figura ----
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.28)
    lim = np.ceil(np.nanmax(np.abs(np.r_[X, Y])) / 25) * 25
    gx, gy = np.mgrid[-lim:lim:200j, -lim:lim:200j]
    kde_all = gaussian_kde(np.vstack([X, Y]))
    dens_all = kde_all(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    for i, p in enumerate(POPS):
        ax = fig.add_subplot(gs[0, i])
        m = groups[p]
        kde = gaussian_kde(np.vstack([X[m], Y[m]]), bw_method=kde_all.factor)
        dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.imshow(np.log10(dens.T / dens.max() + 1e-4), origin="lower", cmap="Greys",
                  extent=[-lim, lim, -lim, lim], vmin=-3, vmax=0, aspect="equal")
        ax.contour(gx, gy, dens_all / dens_all.max(), levels=[0.05, 0.2, 0.5],
                   colors="gray", linewidths=1, linestyles="--")
        ax.scatter(X[m], Y[m], s=10, color=COLORS_3[i], alpha=0.6, edgecolors="none")
        for zone, ls in [("todos", "-"), (f"R<{a.rlim:g}", ":")]:
            e = out["ellipses"].get(f"{p}|{zone}")
            if e:
                xm, ym = X[m].mean(), Y[m].mean()
                sel = m & ((R < a.rlim) if zone != "todos" else True)
                _, _, cov = ellipse_params(X[sel], Y[sel])
                w = np.linalg.eigvalsh(cov)
                ax.add_patch(Ellipse((X[sel].mean(), Y[sel].mean()), 4*np.sqrt(w[1]), 4*np.sqrt(w[0]),
                                     angle=90 - e["PA"], fill=False, color=COLORS_3[i], lw=2.5, ls=ls))
        ax.set_title(f"{p} (N={m.sum()})", fontsize=26)
        ax.set_xlabel("ΔX (kpc, E)", fontsize=26)
        if i == 0:
            ax.set_ylabel("ΔY (kpc, N)", fontsize=26)
        ax.tick_params(labelsize=25)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    ax = fig.add_subplot(gs[1, 0])
    for i, p in enumerate(POPS):
        r = np.sort(R[groups[p]])
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=COLORS_3[i], lw=3, label=p)
    r = np.sort(R)
    ax.plot(r, np.arange(1, len(r) + 1) / len(r), color="k", lw=2, ls="--", label="All")
    ax.set_xscale("log"); ax.set_xlabel("Projected radius (kpc)", fontsize=26)
    ax.set_ylabel("Cumulative fraction", fontsize=25); ax.tick_params(labelsize=25)
    ax.legend(fontsize=19, loc="upper left")#; ax.grid(alpha=0.2, ls="--")

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 180, 13)
    for i, p in enumerate(POPS):
        h, _ = np.histogram(PA[groups[p]], bins=bins)
        ax.step(0.5*(bins[1:]+bins[:-1]), h / h.sum(), where="mid", color=COLORS_3[i], lw=3, label=p)
    ax.axhline(1/12, color="gray", ls="--", lw=1.5)
    if a.pa_gal is not None:
        ax.axvline(a.pa_gal % 180, color="k", ls=":", lw=2, label="Galaxy PA")
    ax.set_xlabel("Position angle (deg, N→E, folded)", fontsize=25)
    ax.set_ylabel("Fraction", fontsize=25); ax.tick_params(labelsize=25)
    ax.legend(fontsize=19)###; ax.grid(alpha=0.2, ls="--")

    ax = fig.add_subplot(gs[1, 2])
    for i, p in enumerate(POPS):
        gd = out["groups"][p]
        ax.errorbar(i, gd["R_median"], yerr=[[gd["R_median"] - gd["R_median_p16"]],
                                             [gd["R_median_p84"] - gd["R_median"]]],
                    fmt="o", ms=14, color=COLORS_3[i], capsize=6, lw=2.5)
    ax.axhline(out["groups"]["All"]["R_median"], color="k", ls="--", lw=1.5, label="All GCs")
    ax.set_xticks(range(3)); ax.set_xticklabels(POPS, fontsize=25)
    ax.set_ylabel("Median projected radius (kpc)", fontsize=24); ax.tick_params(labelsize=25)
    ax.legend(fontsize=19)#; ax.grid(alpha=0.2, ls="--")

    base = os.path.join(a.outdir, "Spatial_Distribution_Populations")
    plt.savefig(f"{base}.pdf", bbox_inches="tight"); plt.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
    plt.close()
    with open(os.path.join(a.outdir, "spatial_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOK figura: {base}.pdf   |   JSON: {a.outdir}/spatial_results.json")


if __name__ == "__main__":
    main()
