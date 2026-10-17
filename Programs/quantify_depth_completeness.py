#!/usr/bin/env python3
"""
gc_sample_pipeline.py -- un solo script, un solo archivo de fotometria, todos los numeros.

Todo se deriva del MISMO catalogo crudo, de forma determinista:

  1. Deduplicacion de mediciones repetidas (campos superpuestos) -> fotometria canonica
  2. Seleccion por calidad fotometrica                             -> muestra seleccionada
  3. Diagnosticos: errores recortados (S/N = 2), filtros "reales",
     comparacion con una seleccion previa (p.ej. OPTIMIZED_1394)
  4. Profundidad narrow-band (S/N mediana vs magnitud)  -> Sect. 2  (padre: T17 con r valida)
  5. Completitud vs r, vs radio proyectado y vs color    -> Sect. 3  (padre: P(GC) >= 0.7)

Reproducir la muestra del paper (debe dar 1394):
  python gc_sample_pipeline.py \\
      --raw       all_fields_photometry_COMPLETE_extinction.csv \\
      --reference TAP_1_J_MNRAS_3444_gc.csv \\
      --cigale    CIGALE-DEcam-extinction/out/results.fits \\
      --dedup first \\
      --compare-selected all_fields_photometry_OPTIMIZED_1394objects.csv \\
      --outdir run_paper

Version mejorada para la revision (luego hay que volver a correr CIGALE):
  python gc_sample_pipeline.py ... --dedup best --exclude-capped \\
      --depth-xaxis r --keep-nonpositive-snr --outdir run_v2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ("F378", "F395", "F410", "F430", "F515", "F660", "F861")
LABELS = {f: "J0" + f[1:] for f in BANDS}
ERR_THR = {"F378": 0.8, "F395": 0.7, "F410": 0.5, "F430": 0.5,
           "F515": 0.4, "F660": 0.3, "F861": 0.4}
SNR_THR = {"F378": 1.5, "F395": 1.5, "F410": 2.0, "F430": 2.0,
           "F515": 3.0, "F660": 4.0, "F861": 3.0}
COLORS = {"F378": "#2b3a8f", "F395": "#3d5fbf", "F410": "#4f8ec9",
          "F430": "#59b7a4", "F515": "#f0a63b", "F660": "#e05a3c",
          "F861": "#8e2b2b"}
CAP_ERR = 1.0857362047581296 / 2.0          # magerr equivalente a S/N = 2
NGC5128 = ("13h25m27.6s", "-43d01m09s")


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def norm_id(s: pd.Series) -> pd.Series:
    """'1202', ' 1202 ', 1202.0 -> '1202'."""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def cols(f: str, ap: str) -> dict:
    return {"err": f"MAGERR_{f}_{ap}_corr", "snr": f"SNR_{f}_{ap}",
            "flux": f"FLUX_{f}_{ap}_corr", "mag": f"MAG_{f}_{ap}"}


def num(df: pd.DataFrame, c: str):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else None


def wilson(k: int, n: int, z: float = 1.0):
    """Intervalo de Wilson (z=1 -> 68 %), en porcentaje."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def crossing(g: pd.DataFrame, thr: float, bw: float):
    """Magnitud donde la S/N mediana cruza thr (interpolacion lineal entre bins contiguos)."""
    g = g.sort_values("bin")
    pts = list(zip(g["bin"], g["median_snr"]))
    for (m0, s0), (m1, s1) in zip(pts, pts[1:]):
        if s0 >= thr > s1 and 0 < m1 - m0 <= bw + 1e-6:
            return m0 + (thr - s0) / (s1 - s0) * (m1 - m0)
    return None


# ----------------------------------------------------------------------------
# 1. deduplicacion
# ----------------------------------------------------------------------------
def deduplicate(raw: pd.DataFrame, mode: str, key: list, ap: str) -> pd.DataFrame:
    if mode == "first":
        out = raw.drop_duplicates(subset=key, keep="first")
    else:  # "best": mayor S/N en J0660 + J0861; empates -> orden del archivo
        score = pd.Series(0.0, index=raw.index)
        for f in ("F660", "F861"):
            s = num(raw, cols(f, ap)["snr"])
            if s is not None:
                score += s.fillna(-1e6)
        out = (raw.assign(_score=score)
                  .sort_values("_score", ascending=False, kind="mergesort")
                  .drop_duplicates(subset=key, keep="first")
                  .drop(columns="_score")
                  .sort_index())
    print(f"[dedup:{mode}] {len(raw)} filas -> {len(out)} objetos unicos "
          f"({len(raw) - len(out)} duplicados eliminados)")
    return out.copy()


# ----------------------------------------------------------------------------
# 2. seleccion
# ----------------------------------------------------------------------------
def good_masks(df, ap, err_thr, snr_thr, cap, tol):
    good, capped = {}, {}
    for f in BANDS:
        c = cols(f, ap)
        e = num(df, c["err"])
        if e is None:
            continue
        m = e.notna() & (e > 0) & (e < err_thr[f])
        s = num(df, c["snr"])
        if s is not None:
            m &= s.notna() & (s >= snr_thr[f])
        fl = num(df, c["flux"])
        if fl is not None:
            m &= fl.notna() & (fl > 0)
        good[f] = m
        capped[f] = e.notna() & ((e - cap).abs() <= tol)
    return good, capped


# ----------------------------------------------------------------------------
# 4. profundidad
# ----------------------------------------------------------------------------
def depth_curves(df, ap, xaxis, rcol, bw, keep_nonpos, cap, tol, min_n):
    curves, depths = {}, []
    for f in BANDS:
        c = cols(f, ap)
        s, e = num(df, c["snr"]), num(df, c["err"])
        x = df[rcol] if xaxis == "r" else num(df, c["mag"])
        if s is None or x is None:
            continue
        ok = x.notna() & (x > 0) & (x < 90) & s.notna()
        if not keep_nonpos:
            ok &= s > 0
        d = pd.DataFrame({"x": x[ok], "snr": s[ok]})
        d["capped"] = ((e[ok] - cap).abs() <= tol) if e is not None else False
        d["bin"] = np.floor(d["x"] / bw) * bw + bw / 2
        g = (d.groupby("bin")
               .agg(n=("snr", "size"), median_snr=("snr", "median"),
                    frac_capped=("capped", "mean"))
               .reset_index())
        g.insert(0, "filter", f)
        curves[f] = g
        use = g[g["n"] >= min_n]
        depths.append({"filter": f, "n_objects": int(ok.sum()),
                       "depth_snr3": crossing(use, 3.0, bw),
                       "depth_snr5": crossing(use, 5.0, bw)})
    return curves, pd.DataFrame(depths)


# ----------------------------------------------------------------------------
# 5. completitud
# ----------------------------------------------------------------------------
def completeness(parent: pd.DataFrame, group: pd.Series, xname: str, xval: pd.Series):
    rows = []
    for key, grp in parent.groupby(group, observed=True):
        n = len(grp)
        kp, kf, kb = (int(grp[c].sum()) for c in ("has_good_phot", "has_good_fit", "combined"))
        pl, ph = wilson(kp, n)
        bl, bh = wilson(kb, n)
        sl, sh = wilson(kb, kp)
        xv = float(key.mid) if (xname == "r" and hasattr(key, "mid")) else float(xval.loc[grp.index].median())
        rows.append({xname: xv, "bin": str(key), "n_parent": n,
                     "n_phot": kp, "frac_phot": 100 * kp / n, "phot_lo": pl, "phot_hi": ph,
                     "n_fit": kf, "n_both": kb, "frac_both": 100 * kb / n,
                     "both_lo": bl, "both_hi": bh,
                     "chi2_survival": 100 * kb / kp if kp else np.nan,
                     "surv_lo": sl, "surv_hi": sh})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------
def _style(legend_fs):
    """Estilo original de quantify_depth_completeness.py."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": legend_fs,
        "axes.labelsize": 9.5, "figure.dpi": 150,
    })


def fig_depth(curves, out, xaxis, min_n, xlim, ylim=(1.0, 200.0)):
    _style(7.0)
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    for f in BANDS:
        g = curves.get(f)
        if g is None:
            continue
        g = g[g["n"] >= min_n].sort_values("bin")
        if g.empty:
            continue
        ax.plot(g["bin"], g["median_snr"], marker="o", markersize=3.5, linewidth=0.9,
                color=COLORS[f], label=LABELS[f])
    ax.axhline(5.0, color="0.4", linestyle="--", linewidth=0.7)
    ax.axhline(3.0, color="0.4", linestyle=":", linewidth=0.7)
    ylo, yhi = ylim
    f5 = (np.log10(5.0) - np.log10(ylo)) / (np.log10(yhi) - np.log10(ylo))
    f3 = (np.log10(3.0) - np.log10(ylo)) / (np.log10(yhi) - np.log10(ylo))
    ax.text(0.18, f5, r"S/N $=5$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10., color="0.35")
    ax.text(0.18, f3, r"S/N $=3$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10., color="0.35")
    ax.set_xlabel(r"$r$-mag" if xaxis == "r" else "Narrow-band magnitude (mag)", fontsize=15)
    ax.set_ylabel("Median S/N", fontsize=15)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8,
              handlelength=1.2, borderaxespad=0.4, fontsize=10)
    ax.set_axisbelow(True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_completeness(tab, xname, xlabel, out, chi2_max, logx=False):
    _style(7.5)
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    x = tab[xname].to_numpy(float)
    y = tab["frac_phot"].to_numpy(float)
    ax.errorbar(x, y,
                yerr=np.vstack([np.clip(y - tab["phot_lo"], 0, None),
                                np.clip(tab["phot_hi"] - y, 0, None)]),
                marker="s", markersize=3.5, linewidth=1.0, capsize=2,
                color="#3d5fbf", label="Photometric quality")
    y = tab["frac_both"].to_numpy(float)
    ax.errorbar(x, y,
                yerr=np.vstack([np.clip(y - tab["both_lo"], 0, None),
                                np.clip(tab["both_hi"] - y, 0, None)]),
                marker="o", markersize=3.5, linewidth=1.2, capsize=2,
                color="#2b3a8f",
                label=rf"Photometric $+$ CIGALE $\chi^2_\nu \leq {chi2_max}$")
    ax.axhline(50.0, color="0.4", linestyle=":", linewidth=0.7)
    ax.axhline(90.0, color="0.4", linestyle="--", linewidth=0.7)
    ax.text(0.02, 0.43, "50%", transform=ax.transAxes, fontsize=10, color="0.4")
    ax.text(0.02, 0.81, "90%", transform=ax.transAxes, fontsize=10, color="0.4")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Fraction of GC initial sample (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_axisbelow(True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, type=Path,
                   help="Catalogo crudo con todas las mediciones (p.ej. COMPLETE_extinction.csv)")
    p.add_argument("--reference", required=True, type=Path, help="Catalogo T17 (ID, r, ...)")
    p.add_argument("--cigale", type=Path, default=None, help="results.fits de CIGALE (opcional)")
    p.add_argument("--compare-selected", type=Path, default=None,
                   help="Seleccion previa para verificar (p.ej. OPTIMIZED_1394objects.csv)")
    p.add_argument("--outdir", type=Path, default=Path("gc_pipeline_out"))
    p.add_argument("--id-col", default="T17ID")
    p.add_argument("--rmag-col", default="rmag")
    p.add_argument("--color-cols", nargs=2, default=None, metavar=("BLUE", "RED"),
                   help="Columnas del catalogo T17 para el color (p.ej. gmag imag)")
    p.add_argument("--ra-col", default="RAJ2000")
    p.add_argument("--dec-col", default="DEJ2000")
    p.add_argument("--distance-mpc", type=float, default=3.8)
    p.add_argument("--aperture", default="3")
    p.add_argument("--dedup", choices=("first", "best"), default="first",
                   help="first = reproduce el paper; best = mayor S/N en J0660+J0861")
    p.add_argument("--dedup-key", nargs="+", default=["RAJ2000", "DEJ2000"])
    p.add_argument("--min-filters", type=int, default=3)
    p.add_argument("--require", nargs="*", default=[], choices=BANDS,
                   help="Filtros que deben ser buenos y NO recortados (p.ej. F660 F861)")
    p.add_argument("--exclude-capped", action="store_true",
                   help="No contar como 'buenos' los filtros con error recortado (S/N=2)")
    p.add_argument("--cap-tol", type=float, default=2e-3)
    p.add_argument("--prob-min", type=float, default=0.7, help="Padre de la Sect. 3")
    p.add_argument("--depth-prob-min", type=float, default=0.0, help="Padre de la Sect. 2")
    p.add_argument("--chi2-max", type=float, default=2.0)
    p.add_argument("--depth-xaxis", choices=("own", "r"), default="own")
    p.add_argument("--keep-nonpositive-snr", action="store_true")
    p.add_argument("--bin-width", type=float, default=0.5)
    p.add_argument("--min-n-per-bin", type=int, default=3)
    p.add_argument("--min-bin-size", type=int, default=10)
    p.add_argument("--n-radial-bins", type=int, default=6)
    p.add_argument("--depth-xlim", nargs=2, type=float, default=(15.5, 22.5))
    return p.parse_args()


def main() -> int:
    a = parse()
    a.outdir.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    say(f"Configuracion: dedup={a.dedup}, min_filters={a.min_filters}, "
        f"exclude_capped={a.exclude_capped}, require={a.require}, "
        f"prob_min={a.prob_min}, depth_prob_min={a.depth_prob_min}, "
        f"depth_xaxis={a.depth_xaxis}, keep_nonpositive_snr={a.keep_nonpositive_snr}")

    # --- 1. fotometria canonica ------------------------------------------------
    raw = pd.read_csv(a.raw)
    raw[a.id_col] = norm_id(raw[a.id_col])
    phot = deduplicate(raw, a.dedup, a.dedup_key, a.aperture)
    say(f"Objetos unicos: {len(phot)}")
    if phot[a.id_col].duplicated().any():
        say(f"AVISO: {phot[a.id_col].duplicated().sum()} IDs repetidos tras deduplicar por "
            f"{a.dedup_key}; revisa la clave de deduplicacion.")

    usecols = [a.id_col, a.rmag_col] + (list(a.color_cols) if a.color_cols else [])
    ref = pd.read_csv(a.reference)
    missing = [c for c in usecols if c not in ref.columns]
    if missing:
        raise SystemExit(f"Columnas ausentes en --reference: {missing}")
    ref = ref[usecols].copy()
    ref[a.id_col] = norm_id(ref[a.id_col])
    ref = ref.drop_duplicates(a.id_col).rename(columns={a.rmag_col: "r_T17"})
    phot = phot.drop(columns=[c for c in ref.columns if c != a.id_col and c in phot.columns])
    phot = phot.merge(ref, on=a.id_col, how="left")
    r = pd.to_numeric(phot["r_T17"], errors="coerce")
    phot["valid_r"] = r.notna() & (r > 0) & (r < 30)

    prob = num(phot, "Prob")
    if prob is None:
        say("AVISO: sin columna 'Prob'; se asume P(GC)=1 para todos.")
        prob = pd.Series(1.0, index=phot.index)
    phot["in_prob"] = prob >= a.prob_min

    # --- 2. seleccion -----------------------------------------------------------
    good, capped = good_masks(phot, a.aperture, ERR_THR, SNR_THR, CAP_ERR, a.cap_tol)
    phot["n_good"] = sum(m.astype(int) for m in good.values())
    phot["n_real_good"] = sum((good[f] & ~capped[f]).astype(int) for f in good)
    for f in good:
        phot[f"capped_{f}"] = capped[f]
    count = phot["n_real_good"] if a.exclude_capped else phot["n_good"]
    sel = count >= a.min_filters
    for f in a.require:
        sel &= good[f] & ~capped[f]
    phot["has_good_phot"] = sel

    # --- CIGALE -----------------------------------------------------------------
    phot["redchi2"] = np.nan
    if a.cigale is not None:
        from astropy.table import Table
        cig = Table.read(str(a.cigale), hdu=1).to_pandas()
        cig = pd.DataFrame({a.id_col: norm_id(cig["id"].astype(str)),
                            "redchi2": cig["best.reduced_chi_square"]}).drop_duplicates(a.id_col)
        phot = phot.drop(columns="redchi2").merge(cig, on=a.id_col, how="left")
        say(f"CIGALE: {len(cig)} objetos ajustados, {int((cig.redchi2 <= a.chi2_max).sum())} "
            f"con chi2_nu <= {a.chi2_max:g}")
    phot["has_good_fit"] = phot["redchi2"].notna() & (phot["redchi2"] <= a.chi2_max)
    phot["combined"] = phot["has_good_phot"] & phot["has_good_fit"]

    selected = phot[phot["in_prob"] & phot["has_good_phot"]]
    final = selected[selected["has_good_fit"]]
    say(f"\nP(GC) >= {a.prob_min}: {int(phot['in_prob'].sum())}")
    say(f"Seleccion fotometrica: {len(selected)}")
    say(f"Muestra final (+ chi2_nu <= {a.chi2_max:g}): {len(final)}")
    if a.cigale is not None:
        n_fit_in = int((phot["has_good_fit"] & phot["in_prob"]).sum())
        if n_fit_in != len(final):
            say(f"AVISO: {n_fit_in - len(final)} buenos ajustes de CIGALE quedan FUERA de esta "
                "seleccion -> la fotometria/seleccion no coincide con la que vio CIGALE.")

    # --- 3. diagnosticos --------------------------------------------------------
    say("\nDiagnostico por filtro (muestra seleccionada | muestra final):")
    say("  filtro  recortados   %recort.  S/N>3 (final)  S/N>5 (final)")
    diag = []
    for f in good:
        s = num(final, cols(f, a.aperture)["snr"])
        n3 = int((s > 3).sum()) if s is not None else 0
        n5 = int((s > 5).sum()) if s is not None else 0
        nc = int(selected[f"capped_{f}"].sum())
        diag.append({"filter": f, "n_capped_selected": nc,
                     "pct_capped_selected": 100 * nc / max(len(selected), 1),
                     "pct_snr3_final": 100 * n3 / max(len(final), 1),
                     "pct_snr5_final": 100 * n5 / max(len(final), 1)})
        say(f"  {LABELS[f]}  {nc:9d}   {diag[-1]['pct_capped_selected']:7.1f}"
            f"  {diag[-1]['pct_snr3_final']:12.1f}  {diag[-1]['pct_snr5_final']:12.1f}")
    pd.DataFrame(diag).to_csv(a.outdir / "diagnostics_per_filter.csv", index=False)
    nr = selected["n_real_good"]
    say(f"Seleccionados con >= {a.min_filters} filtros buenos NO recortados: "
        f"{int((nr >= a.min_filters).sum())} de {len(selected)}")
    say("Distribucion de n_real_good en la seleccion: "
        + ", ".join(f"{k}:{v}" for k, v in nr.value_counts().sort_index().items()))

    if a.compare_selected is not None:
        old = pd.read_csv(a.compare_selected)
        old[a.id_col] = norm_id(old[a.id_col])
        A, B = set(selected[a.id_col]), set(old[a.id_col])
        say(f"\nComparacion con {a.compare_selected.name}: en ambas {len(A & B)}, "
            f"solo nueva {len(A - B)}, solo previa {len(B - A)}")
        m = phot.merge(old, on=a.id_col, suffixes=("", "_old"))
        for f in BANDS:
            c = cols(f, a.aperture)["err"]
            if c in m.columns and c + "_old" in m.columns:
                d = (pd.to_numeric(m[c], errors="coerce")
                     - pd.to_numeric(m[c + "_old"], errors="coerce")).abs()
                say(f"  {LABELS[f]}: {int((d > 1e-4).sum())} objetos con MAGERR distinto "
                    f"(max {d.max():.3f})")

    # --- 4. profundidad (Sect. 2) -----------------------------------------------
    if a.depth_prob_min > 0:
        pdepth = phot[phot["valid_r"] & (prob >= a.depth_prob_min)]
    else:
        pdepth = phot[phot["valid_r"]]
    curves, depths = depth_curves(pdepth, a.aperture, a.depth_xaxis, "r_T17", a.bin_width,
                                  a.keep_nonpositive_snr, CAP_ERR, a.cap_tol, a.min_n_per_bin)
    say(f"\nProfundidad (padre = {len(pdepth)}, eje x = {a.depth_xaxis}):")
    for _, rw in depths.iterrows():
        d3 = f"{rw.depth_snr3:.2f}" if pd.notna(rw.depth_snr3) else "n/a"
        d5 = f"{rw.depth_snr5:.2f}" if pd.notna(rw.depth_snr5) else "n/a"
        say(f"  {LABELS[rw['filter']]}: S/N=3 -> {d3}   S/N=5 -> {d5}")
    depths.to_csv(a.outdir / "depth_per_filter.csv", index=False)
    pd.concat(curves.values()).to_csv(a.outdir / "depth_curves.csv", index=False)
    fig_depth(curves, a.outdir / "snr_depth.pdf", a.depth_xaxis, a.min_n_per_bin,
              tuple(a.depth_xlim))

    # --- 5. completitud (Sect. 3) -----------------------------------------------
    parent = phot[phot["in_prob"] & phot["valid_r"]].copy()
    say(f"\nCompletitud: padre = {len(parent)}")
    rr = pd.to_numeric(parent["r_T17"], errors="coerce")
    edges = np.arange(15.0, 23.0 + 1e-9, a.bin_width)
    rbin = pd.cut(rr, edges)
    tab_r = completeness(parent, rbin, "r", rr)
    tab_r = tab_r[tab_r["n_parent"] >= a.min_bin_size]
    tab_r.to_csv(a.outdir / "completeness_vs_r.csv", index=False)
    fig_completeness(tab_r, "r", r"$r$-mag", a.outdir / "completeness_r.pdf", a.chi2_max)
    say(tab_r[["r", "n_parent", "frac_phot", "frac_both", "chi2_survival"]]
        .to_string(index=False, float_format="%.1f"))

    if a.ra_col in parent.columns and a.dec_col in parent.columns:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c0 = SkyCoord(*NGC5128)
        cc = SkyCoord(pd.to_numeric(parent[a.ra_col]).values * u.deg,
                      pd.to_numeric(parent[a.dec_col]).values * u.deg)
        parent["R_kpc"] = cc.separation(c0).rad * a.distance_mpc * 1e3
        rad_edges = np.unique(np.quantile(parent["R_kpc"],
                                          np.linspace(0, 1, a.n_radial_bins + 1)))
        tab_R = completeness(parent, pd.cut(parent["R_kpc"], rad_edges, include_lowest=True),
                             "R_kpc", parent["R_kpc"])
        tab_R.to_csv(a.outdir / "completeness_vs_radius.csv", index=False)
        fig_completeness(tab_R, "R_kpc", "Projected radius (kpc)",
                         a.outdir / "completeness_radius.pdf", a.chi2_max, logx=True)
        say("\nCompletitud vs radio (todas las magnitudes):")
        say(tab_R[["R_kpc", "n_parent", "frac_phot", "frac_both", "chi2_survival"]]
            .to_string(index=False, float_format="%.1f"))
        # control: radio a magnitud fija (r < 20.3, donde la completitud en r es plana)
        bright = parent[rr < 20.3]
        if len(bright) >= 3 * a.min_bin_size:
            tab_Rb = completeness(bright, pd.cut(bright["R_kpc"], rad_edges, include_lowest=True),
                                  "R_kpc", bright["R_kpc"])
            tab_Rb.to_csv(a.outdir / "completeness_vs_radius_r_lt_20.3.csv", index=False)
            say("\nCompletitud vs radio (solo r < 20.3):")
            say(tab_Rb[["R_kpc", "n_parent", "frac_phot", "frac_both", "chi2_survival"]]
                .to_string(index=False, float_format="%.1f"))
    else:
        say(f"Sin columnas {a.ra_col}/{a.dec_col}: se omite la completitud vs radio.")

    if a.color_cols:
        b, rc = (pd.to_numeric(parent[c], errors="coerce") for c in a.color_cols)
        parent["color"] = b - rc
        okc = parent["color"].notna()
        tab_c = completeness(parent[okc], pd.qcut(parent.loc[okc, "color"], 4),
                             "color", parent.loc[okc, "color"])
        tab_c.to_csv(a.outdir / "completeness_vs_color.csv", index=False)
        say(f"\nCompletitud vs color ({a.color_cols[0]} - {a.color_cols[1]}, cuartiles):")
        say(tab_c[["color", "n_parent", "frac_phot", "frac_both", "chi2_survival"]]
            .to_string(index=False, float_format="%.2f"))

    med_p, med_f = float(rr.median()), float(rr[parent["combined"]].median())
    say(f"\nMediana r: padre {med_p:.3f}, muestra final {med_f:.3f}, offset {med_f - med_p:+.3f} mag")

    # --- salidas -----------------------------------------------------------------
    phot.to_csv(a.outdir / "canonical_photometry.csv", index=False)
    selected.drop(columns=[c for c in selected.columns if c.startswith("capped_")]) \
            .to_csv(a.outdir / "selected_sample.csv", index=False)
    (a.outdir / "summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\nSalidas en {a.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
