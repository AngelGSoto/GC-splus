from astropy.table import Table
import pandas as pd, numpy as np
d = "/media/luis/Data/GC-Porto-Alegre/anac_data/CIGALE-DEcam-extinction/"
fix = lambda s: s.apply(lambda x: x.decode() if isinstance(x, bytes) else str(x)).str.strip()
obs = Table.read(d + "out/observations.fits").to_pandas(); obs["id"] = fix(obs["id"])
inp = pd.read_csv(d + "gc_splus_cigale_fixed.txt", sep=r"\s+"); inp["id"] = fix(inp["id"])
m = obs.merge(inp, on="id", suffixes=("_obs", "_inp"))
for b in ["F0660", "F0861"]:
    ok = m[b + "_inp"] > 0
    r = m.loc[ok, b + "_err_obs"] / m.loc[ok, b + "_err_inp"]
    pred = np.sqrt(1 + (0.05 * m.loc[ok, b + "_inp"] / m.loc[ok, b + "_err_inp"])**2)
    print(b, "cociente de errores obs/inp: mediana", round(r.median(), 4),
          "| si incluyera 5%:", round(pred.median(), 4))
