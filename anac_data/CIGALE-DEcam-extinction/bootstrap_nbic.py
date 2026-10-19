import numpy as np
from astropy.table import Table
from sklearn.mixture import GaussianMixture
t = Table.read("out/results.fits").to_pandas()
t = t[t["best.reduced_chi_square"] <= 2.0]
Z, eZ = t["bayes.stellar.metallicity"], t["bayes.stellar.metallicity_err"]
feh = np.log10(Z / 0.02).values
err = ((eZ / Z) / np.log(10)).values
age = (t["bayes.stellar.age_m_star"] / 1000).values
rng = np.random.default_rng(2026)
NREAL = 300

def best_n(X, seed):
    bic = [GaussianMixture(n, n_init=3, random_state=seed).fit(X).bic(X) for n in range(2, 6)]
    return 2 + int(np.argmin(bic))

for name, sel in [("Todos", np.ones(len(feh), bool)), ("Viejos >8 Gyr", age > 8)]:
    f, e = feh[sel], err[sel]
    for metodo in ["remuestreo", "perturbacion"]:
        best = []
        for k in range(NREAL):
            if metodo == "remuestreo":
                X = f[rng.integers(0, len(f), len(f))].reshape(-1, 1)
            else:
                X = (f + rng.normal(0, e)).reshape(-1, 1)
            best.append(best_n(X, k))
        best = np.array(best)
        print(f"{name:14s} {metodo:13s}: n=2 {np.mean(best==2)*100:5.1f}%  "
              f"n=3 {np.mean(best==3)*100:5.1f}%  n>=4 {np.mean(best>=4)*100:5.1f}%")
