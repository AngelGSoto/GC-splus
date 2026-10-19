import numpy as np, pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.stats import wilcoxon

def load(run):
    t = Table.read(f"{run}/out/results.fits").to_pandas()
    t["id"] = t["id"].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x)).str.strip()
    Z, eZ = t["bayes.stellar.metallicity"], t["bayes.stellar.metallicity_err"]
    t["feh"] = np.log10(Z.where(Z > 0) / 0.02)
    t["sig"] = (eZ / Z).where(Z > 0) / np.log(10)
    return t.set_index("id")

n, b = load("CIGALE-DEcam-extinction"), load("CIGALE-broadonly")
gn, gb = n["best.reduced_chi_square"] <= 2, b["best.reduced_chi_square"] <= 2
common = gn[gn].index.intersection(gb[gb].index)
print(f"Buen ajuste: 12 bandas {gn.sum()} | 5 bandas {gb.sum()} | en ambas {len(common)}")

sn, sb = n.loc[common, "sig"], b.loc[common, "sig"]
stat, p = wilcoxon(sn, sb)
print(f"sigma[Fe/H] mediana (mismos {len(common)}): 12 bandas {sn.median():.3f} | 5 bandas {sb.median():.3f} | Wilcoxon p = {p:.2g}")
print(f"fracción de objetos con sigma menor usando 12 bandas: {np.mean(sn < sb)*100:.1f}%")

cat = pd.read_csv("Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
def match(path, ra, de, col, rad):
    L = pd.read_csv(path); L[col] = pd.to_numeric(L[col], errors="coerce")
    c1 = SkyCoord(cat["RAJ2000"].values*u.deg, cat["DEJ2000"].values*u.deg)
    c2 = SkyCoord(L[ra].values*u.deg, L[de].values*u.deg)
    i1, i2, s, _ = c2.search_around_sky(c1, rad*u.arcsec)
    m = pd.DataFrame({"id": cat["T17ID"].astype(str).str.strip().iloc[i1].values,
                      "lit": L[col].values[i2], "s": s.arcsec})
    return m.loc[m.groupby("id")["s"].idxmin()].set_index("id")["lit"].dropna()

for name, lit in [("Beasley", match("../globular_clusters_ngc5128_Beasley.csv", "RAJ2000", "DEJ2000", "[M/H]", 3)),
                  ("Hernandez", match("../globular_clusters_ngc5128.csv", "RA (J2000)", "Dec (J2000)", "[Fe/H] (dex)", 4))]:
    ids = lit.index.intersection(common)
    for lab, r in [("12 bandas", n), ("5 bandas ", b)]:
        d = r.loc[ids, "feh"] - lit.loc[ids]
        print(f"{name:9s} {lab} N={len(ids):3d}  MAE {np.mean(np.abs(d)):.3f}  bias {np.mean(d):+.3f}")
