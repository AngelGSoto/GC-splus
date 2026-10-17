import sys
from astropy.table import Table
import numpy as np
orig_path = "/media/luis/Data/GC-Porto-Alegre/anac_data/CIGALE-DEcam-extinction/out/results.fits"
new_path = (sys.argv[1] if len(sys.argv) > 1 else "t_check") + "/out/results.fits"
fix = lambda s: s.apply(lambda x: x.decode() if isinstance(x, bytes) else str(x)).str.strip()
orig = Table.read(orig_path).to_pandas(); orig["id"] = fix(orig["id"])
new = Table.read(new_path).to_pandas();   new["id"] = fix(new["id"])
m = new.merge(orig, on="id", suffixes=("_n", "_o"))
print(new_path, "| objetos:", len(m),
      "| chi2 nuevo/original (mediana):", round(float(np.median(m["best.chi_square_n"] / m["best.chi_square_o"])), 3))
for c in ["best.chi_square", "bayes.stellar.metallicity", "bayes.stellar.age_m_star"]:
    d = np.abs(m[c + "_n"] - m[c + "_o"])
    print(f"   {c}: dif. máx {d.max():.3g}, mediana {d.median():.3g}")
