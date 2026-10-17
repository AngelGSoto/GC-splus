#!/usr/bin/env python3
"""
comparar_gradientes_modelos.py -- compara los gradientes de varias corridas
(comentario 3 iii del arbitro), leyendo los JSON de gmm_analysis_final_paper_v2.py.

Uso (desde anac_data):
  python ../Programs/comparar_gradientes_modelos.py \\
      CIGALE-DEcam-extinction/GMM_Final_Paper_paper_*/gmm_final_paper_results.json \\
      CIGALE-bc03/GMM_Final_Paper_bc03_*/gmm_final_paper_results.json \\
      CIGALE-bursty/GMM_Final_Paper_bursty_*/gmm_final_paper_results.json
"""
import json, sys

POPS = ["MP", "Int", "MR"]
runs = [json.load(open(f)) for f in sys.argv[1:]]
if not runs:
    sys.exit(__doc__)

for key, rkey, title in [("gradients_all_gcs", "robustness_all_gcs", "Muestra completa"),
                         ("gradients_old_gcs", "robustness_old_gcs", "GCs viejos (>8 Gyr)")]:
    print(f"\n{title}: pendiente OLS ± error bootstrap (N) [P(pendiente<0), bootstrap con re-ajuste del GMM]")
    print(f"{'corrida':12s} " + "  ".join(f"{p:>30s}" for p in POPS))
    for r in runs:
        cells = []
        for p in POPS:
            g = (r.get(key) or {}).get(p); rb = (r.get(rkey) or {}).get(p)
            if g is None:
                cells.append(f"{'—':>30s}"); continue
            err = rb["bootstrap"]["std"] if rb else g["slope_err"]
            pneg = rb["gmm_refit_bootstrap"]["p_neg"] if rb else float("nan")
            cells.append(f"{g['slope']:+.3f}±{err:.3f} ({g['n_points']:3d}) [{pneg:4.2f}]".rjust(30))
        print(f"{r['label']:12s} " + "  ".join(cells))

print("\nModelo adoptado n=3 (muestra completa): medias de las componentes")
for r in runs:
    m = r["adopted_n3_full"]["means"]; n = r["adopted_n3_full"]["N"]
    print(f"{r['label']:12s} " + "  ".join(f"{p}: {m[i]:+.2f} (N={n[i]})" for i, p in enumerate(POPS))
          + f"   | min BIC: n={r['best_n_bic_full']} (todos), n={r['best_n_bic_old']} (viejos)")
