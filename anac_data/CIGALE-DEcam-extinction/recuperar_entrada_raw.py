import pandas as pd, numpy as np
inp = pd.read_csv("gc_input_enero2026.txt", sep=r"\s+")
bands = [c for c in inp.columns if c not in ("id", "redshift") and not c.endswith("_err")]
for b in bands:
    f, e = inp[b].to_numpy(float), inp[b + "_err"].to_numpy(float)
    raw2 = e**2 - (0.05 * f)**2
    bad = (f > 0) & (raw2 <= 0)
    if bad.any():
        print(f"AVISO {b}: {bad.sum()} objetos con error <= 5% del flujo; se dejan sin cambiar")
    inp[b + "_err"] = np.where((f > 0) & (raw2 > 0), np.sqrt(np.clip(raw2, 0, None)), e)
inp.to_csv("gc_input_enero2026_raw.txt", sep=" ", index=False, float_format="%.6e")
print("Listo: gc_input_enero2026_raw.txt", len(inp), "objetos")
