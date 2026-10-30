#!/usr/bin/env python3
"""
 -- figura del modelo GMM adoptado para el texto principal.

  Panel izquierdo: distribucion de [Fe/H] de los GCs con chi2_nu <= 2 y el modelo
                   adoptado de 3 componentes (mismo estilo que las figuras del paper).
  Panel derecho:   porcentaje de replicas del bootstrap en que se selecciona cada
                   numero de componentes, para la muestra completa y los GCs viejos
                   (lee el JSON de gmm_bootstrap_validation.py).

Uso (desde la carpeta de la corrida del paper):
  python ../../Programs/fig_gmm_adoptado.py \\
      --bootstrap GMM_Final_Paper_bootstrap_seed2026_*/gmm_final_paper_results.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmm_analysis_final_paper_v2 import validate_gmm_model, COLORS_3, POP_SHORT_3  # noqa: E402


YTOP_LEFT = 1.15   # limite superior del eje Y del panel izquierdo (x maximo de los datos)
YTOP_RIGHT = 80   # limite superior del eje Y del panel derecho (%)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="out/results.fits")
    ap.add_argument("--bootstrap", required=True,
                    help="JSON de gmm_bootstrap_validation.py (acepta comodines)")
    ap.add_argument("--out", default="figures_paper/FullSample_Adopted_n3_Bootstrap")
    a = ap.parse_args()

    files = sorted(glob.glob(a.bootstrap))
    if not files:
        sys.exit(f"No se encontro el JSON del bootstrap: {a.bootstrap}")
    boot = json.load(open(files[-1]))
    bkey = "bootstrap_resample_perturbed" if "bootstrap_resample_perturbed" in boot \
        else "bootstrap_perturbed"
    seed, nboot = boot.get("seed", "?"), boot.get("n_bootstrap", "?")

    # --- datos y modelo adoptado (igual que el script del paper) ---
    df = Table.read(a.results, hdu=1).to_pandas()
    df = df[df["best.reduced_chi_square"] <= 2.0]
    feh = np.log10(df["bayes.stellar.metallicity"] / 0.02).values
    q1, q3 = np.quantile(feh, [0.01, 0.99])
    feh = feh[(feh >= q1 - 1.5*(q3-q1)) & (feh <= q3 + 1.5*(q3-q1))]
    g = validate_gmm_model(feh, 3)
    o = np.argsort(g.means_.flatten())
    mu = g.means_.flatten()[o]
    sd = np.sqrt(g.covariances_.flatten())[o]
    w = g.weights_.flatten()[o]
    seps = [abs(mu[i+1]-mu[i]) / np.sqrt(sd[i]**2 + sd[i+1]**2) for i in range(2)]
    X = feh.reshape(-1, 1)
    print(f"N={len(feh)}  BIC={g.bic(X):.1f}  " +
          "  ".join(f"{POP_SHORT_3[i]}: mu={mu[i]:.3f} sigma={sd[i]:.3f} f={w[i]:.3f}" for i in range(3)))

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5))

    # --- panel izquierdo: mismo estilo que las figuras originales ---
    ax1 = axes[0]
    ax1.hist(feh, bins=30, density=True, alpha=0.3, color="lightgray",
             edgecolor="black", linewidth=0.8, label=f"N={len(feh)}")
    x = np.linspace(feh.min()-0.3, feh.max()+0.3, 1000)
    tot = np.zeros_like(x)
    for i in range(3):
        pc = w[i]*norm.pdf(x, mu[i], sd[i])
        tot += pc
        ax1.plot(x, pc, "--", linewidth=2.5, alpha=0.8, color=COLORS_3[i],
                 label=f"{POP_SHORT_3[i]}: μ={mu[i]:.2f}, σ={sd[i]:.2f}")
        xf = np.linspace(mu[i]-sd[i], mu[i]+sd[i], 100)
        ax1.fill_between(xf, 0, w[i]*norm.pdf(xf, mu[i], sd[i]), color=COLORS_3[i], alpha=0.2)
    ax1.plot(x, tot, "k-", linewidth=3.5, label="GMM fit (n=3)", alpha=0.9)
    ax1.set_xlabel("[Fe/H] (dex)", fontsize=29)
    ax1.set_ylabel("Probability Density", fontsize=29)
    ax1.set_title(f"(a) Adopted model (n=3), N={len(feh)}", fontsize=26)
    ax1.tick_params(axis="both", which="major", labelsize=25)
    ax1.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.0, 0.89), framealpha=0.9)
    ax1.text(0.02, 0.98, f"Sep. = {seps[0]:.2f}σ, {seps[1]:.2f}σ", transform=ax1.transAxes,
             fontsize=20, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ymax_h = max(np.histogram(feh, bins=30, density=True)[0].max(), tot.max())
    ax1.set_ylim(0, YTOP_LEFT*ymax_h)
    #ax1.grid(True, alpha=0.2, linestyle="--")

    # --- panel derecho: seleccion en el bootstrap ---
    ax2 = axes[1]
    nvals = [2, 3, 4, 5]
    xx = np.arange(len(nvals))
    width = 0.38
    for off, key, color, lab in [(-width/2, "full_sample", "steelblue", "All GCs"),
                                 (+width/2, "old_gcs", "indianred", "Old GCs (>8 Gyr)")]:
        b = boot[bkey][key]
        tot_b = b["total"]
        pct = [b["counts"][str(n)]/tot_b*100 if tot_b else 0 for n in nvals]
        bars = ax2.bar(xx + off, pct, width, color=color, alpha=0.8, edgecolor="black",
                       linewidth=1.5, label=lab)
        for bar, p in zip(bars, pct):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                     f"{p:.1f}%", ha="center", va="bottom", fontsize=17)
    ax2.axvspan(xx[1] - 0.5, xx[1] + 0.5, color="#2ca02c", alpha=0.10, zorder=0)
    ax2.text(xx[1], 0.93*YTOP_RIGHT, "Adopted", ha="center", fontsize=22, color="#1f7a1f", fontweight="bold")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(nvals)
    ax2.set_xlim(-0.6, len(nvals) - 0.4)
    ax2.set_ylim(0, YTOP_RIGHT)
    ax2.set_xlabel("Number of Components", fontsize=29)
    ax2.set_ylabel("Selected in bootstrap (%)", fontsize=29)
    ax2.set_title(f"(b) Bootstrap model selection ({nboot} realisations)", fontsize=25)
    ax2.tick_params(axis="both", which="major", labelsize=26)
    #ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.legend(fontsize=20, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(f"{a.out}.pdf", bbox_inches="tight")
    plt.savefig(f"{a.out}.png", dpi=300, bbox_inches="tight")
    print(f"OK figura: {a.out}.pdf  (bootstrap: {files[-1]}, semilla {seed})")


if __name__ == "__main__":
    main()
