#!/usr/bin/env python3
"""
comparar_espectroscopia_modelos.py

Compara varias corridas de CIGALE con la espectroscopia (Beasley+08, Hernandez+18)
usando SIEMPRE LOS MISMOS OBJETOS: los de la corrida de referencia (paper) con
chi2_nu <= 2 que tienen contraparte espectroscopica (94 de Beasley, 7 de Hernandez).

Usa exactamente la misma logica que compare-luis-beasley-svea.py
(Z_sun = 0.02, radios de cruce 3" y 4", MAE = media |dif|, bias = media dif,
std = np.std), asi que para la referencia debe dar 94 / 0.279 / +0.177.

Para cada corrida imprime:
  A) metricas sobre los MISMOS objetos que el paper (comparacion justa entre modelos);
  B) metricas con el corte chi2 <= 2 de la propia corrida (como el script original);
  C) diferencias objeto por objeto respecto del paper sobre la muestra final del paper.
No modifica ni crea figuras.

Uso (desde anac_data):
  python comparar_espectroscopia_modelos.py CIGALE-decamcurves CIGALE-nodust CIGALE-bc03 CIGALE-bursty
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

BASE = Path("/media/luis/Data/GC-Porto-Alegre")
Z_SOLAR = 0.02
CHI2_MAX = 2.0


def norm(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_run(run_dir):
    df = Table.read(Path(run_dir) / "out" / "results.fits", hdu=1).to_pandas()
    if df["id"].dtype == object:
        df["id"] = df["id"].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    Z, eZ = df["bayes.stellar.metallicity"], df["bayes.stellar.metallicity_err"]
    ok = (Z > 0) & (eZ > 0)
    out = pd.DataFrame({
        "ID": norm(df["id"]),
        "FeH": np.where(ok, np.log10(Z.where(ok) / Z_SOLAR), np.nan),
        "FeH_err": np.where(ok, (eZ / Z).where(ok) / np.log(10), np.nan),
        "age": df["bayes.stellar.age_m_star"] / 1000.0,
        "chi2r": df["best.reduced_chi_square"],
    })
    return out.set_index("ID")


def crossmatch(my_cat, lit, radius):
    c1 = SkyCoord(my_cat["RAJ2000"].values * u.deg, my_cat["DEJ2000"].values * u.deg)
    c2 = SkyCoord(lit["RA"].values * u.deg, lit["DEC"].values * u.deg)
    i1, i2, sep, _ = c2.search_around_sky(c1, radius * u.arcsec)
    m = pd.DataFrame({"T17ID": norm(my_cat["T17ID"].iloc[i1]).values,
                      "FeH_lit": lit["FeH"].values[i2], "sep": sep.arcsec})
    return m.loc[m.groupby("T17ID")["sep"].idxmin()].set_index("T17ID")


def metrics(d):
    d = d.dropna()
    if len(d) == 0:
        return dict(N=0, MAE=np.nan, bias=np.nan, std=np.nan, f03=np.nan)
    return dict(N=len(d), MAE=np.mean(np.abs(d)), bias=np.mean(d), std=np.std(d),
                f03=100 * np.mean(np.abs(d) <= 0.3))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="carpetas de corridas a comparar")
    p.add_argument("--ref", default="CIGALE-DEcam-extinction", help="corrida de referencia (paper)")
    p.add_argument("--out", default="comparacion_modelos_espectroscopia.csv")
    a = p.parse_args()

    my_cat = pd.read_csv(BASE / "anac_data/Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
    bea = pd.read_csv(BASE / "globular_clusters_ngc5128_Beasley.csv").rename(
        columns={"RAJ2000": "RA", "DEJ2000": "DEC", "[M/H]": "FeH"})
    her = pd.read_csv(BASE / "globular_clusters_ngc5128.csv").rename(
        columns={"RA (J2000)": "RA", "Dec (J2000)": "DEC", "[Fe/H] (dex)": "FeH"})
    for t in (bea, her):
        t["FeH"] = pd.to_numeric(t["FeH"], errors="coerce")
    lit = {"Beasley": crossmatch(my_cat, bea, 3.0), "Hernandez": crossmatch(my_cat, her, 4.0)}

    ref = load_run(a.ref)
    ref_good = ref[(ref["chi2r"] <= CHI2_MAX) & ref["FeH"].notna()]
    fixed = {k: v.index.intersection(ref_good.index) for k, v in lit.items()}
    print(f"Referencia: {a.ref} | chi2<=2: {len(ref_good)} | objetos fijos: "
          f"Beasley {len(fixed['Beasley'])}, Hernandez {len(fixed['Hernandez'])}")

    rows = []
    for run in [a.ref] + a.runs:
        r = load_run(run)
        good = r[(r["chi2r"] <= CHI2_MAX) & r["FeH"].notna()]
        row = {"run": run, "N_chi2_ok": len(good)}
        for src, m in lit.items():
            ids = fixed[src]                                     # A) mismos objetos
            mA = metrics(r.reindex(ids)["FeH"] - m.loc[ids, "FeH_lit"])
            ok_here = (r.reindex(ids)["chi2r"] <= CHI2_MAX).sum()
            idsB = m.index.intersection(good.index)              # B) corte propio
            mB = metrics(good.loc[idsB, "FeH"] - m.loc[idsB, "FeH_lit"])
            for k, v in mA.items():
                row[f"{src}_same_{k}"] = v
            row[f"{src}_same_chi2ok"] = ok_here
            for k, v in mB.items():
                row[f"{src}_own_{k}"] = v
        common = ref_good.index.intersection(r.index)            # C) objeto por objeto
        d = (r.loc[common, "FeH"] - ref_good.loc[common, "FeH"]).dropna()
        row["dFeH_median"] = d.median()
        row["dFeH_sigmaMAD"] = 1.4826 * np.median(np.abs(d - d.median()))
        row["frac_within_ref_err"] = 100 * np.mean(np.abs(d) <= ref_good.loc[d.index, "FeH_err"])
        row["FeH_err_median"] = good["FeH_err"].median()
        # ancho de la posterior sobre los MISMOS objetos (muestra final del paper)
        row["FeH_err_median_same"] = r.reindex(ref_good.index)["FeH_err"].median()
        row["FeH_err_ratio_same"] = np.nanmedian(
            r.reindex(ref_good.index)["FeH_err"].values / ref_good["FeH_err"].values)
        rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)

    print("\nA) MISMOS objetos que el paper (comparacion justa entre modelos)")
    print(f"{'corrida':28s} {'N':>4s} {'MAE':>6s} {'bias':>7s} {'std':>6s} {'<0.3':>6s} {'chi2ok':>7s}   (Beasley)")
    for _, x in t.iterrows():
        print(f"{x['run']:28s} {x['Beasley_same_N']:4.0f} {x['Beasley_same_MAE']:6.3f} "
              f"{x['Beasley_same_bias']:+7.3f} {x['Beasley_same_std']:6.3f} "
              f"{x['Beasley_same_f03']:5.1f}% {x['Beasley_same_chi2ok']:7.0f}")
    print(f"{'corrida':28s} {'N':>4s} {'MAE':>6s} {'bias':>7s}   (Hernandez)")
    for _, x in t.iterrows():
        print(f"{x['run']:28s} {x['Hernandez_same_N']:4.0f} {x['Hernandez_same_MAE']:6.3f} "
              f"{x['Hernandez_same_bias']:+7.3f}")

    print("\nB) Corte chi2 <= 2 de cada corrida (como el script original)")
    print(f"{'corrida':28s} {'Nchi2':>6s} {'N_B':>4s} {'MAE_B':>6s} {'bias_B':>7s} {'err[Fe/H]':>10s}")
    for _, x in t.iterrows():
        print(f"{x['run']:28s} {x['N_chi2_ok']:6.0f} {x['Beasley_own_N']:4.0f} "
              f"{x['Beasley_own_MAE']:6.3f} {x['Beasley_own_bias']:+7.3f} {x['FeH_err_median']:10.3f}")

    print("\nC) Objeto por objeto vs el paper (muestra final del paper)")
    print(f"{'corrida':28s} {'mediana dFeH':>13s} {'sigma(MAD)':>11s} {'|d|<err_paper':>14s}")
    for _, x in t.iterrows():
        print(f"{x['run']:28s} {x['dFeH_median']:+13.3f} {x['dFeH_sigmaMAD']:11.3f} "
              f"{x['frac_within_ref_err']:13.1f}%")
    print("\nD) Ancho de la posterior en [Fe/H] sobre los MISMOS objetos (muestra final del paper)")
    print(f"{'corrida':28s} {'sigma mediana':>14s} {'mediana(sigma/sigma_paper)':>27s}")
    for _, x in t.iterrows():
        print(f"{x['run']:28s} {x['FeH_err_median_same']:14.3f} {x['FeH_err_ratio_same']:27.2f}")
    print(f"\nTabla completa: {a.out}")


if __name__ == "__main__":
    main()
