#!/usr/bin/env python3
"""Completeness of the narrow-band imaging and of the scientific sample.

Produces TWO separate figures, one per paper section:

  snr_depth.pdf         -> Section 2 (photometry only, no CIGALE mentioned)
                           7 filters, median S/N vs magnitude, S/N = 3 and 5 lines.

  completeness.pdf      -> Section 3 (after CIGALE has been introduced)
                           3 curves: photometric quality, CIGALE chi2_nu <= 2,
                           and the combined fraction, vs r magnitude.

The photometric quality cuts can be specified either as a single uniform
threshold (--err-thresholds 0.2) or as a per-filter list matching the
iterative selection adopted in the paper
(--err-thresholds "F378:0.8,F395:0.7,F410:0.5,F430:0.5,F515:0.4,F660:0.3,F861:0.4").

Usage:
  # Paper selection (iterative per-filter thresholds)
  python quantify_depth_completeness.py \\
      --parent    all_fields_photometry_FULL_3209objects.csv \\
      --cigale    CIGALE-DEcam-extinction/out/results.fits \\
      --reference TAP_1_J_MNRAS_3444_gc.csv \\
      --err-thresholds "F378:0.8,F395:0.7,F410:0.5,F430:0.5,F515:0.4,F660:0.3,F861:0.4" \\
      --output-dir figures/completeness_paper

  # Strict robustness test (uniform 0.2 mag)
  python quantify_depth_completeness.py ... --err-thresholds 0.2 \\
      --output-dir figures/completeness_strict
"""

from __future__ import annotations
import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NARROW_BANDS = ("F378", "F395", "F410", "F430", "F515", "F660", "F861")
DEFAULT_THRESHOLDS = {
    "F378": 0.8, "F395": 0.7, "F410": 0.5, "F430": 0.5,
    "F515": 0.4, "F660": 0.3, "F861": 0.4,
}
FILTER_COLORS = {
    "F378": "#2b3a8f", "F395": "#3d5fbf", "F410": "#4f8ec9",
    "F430": "#59b7a4", "F515": "#f0a63b", "F660": "#e05a3c",
    "F861": "#8e2b2b",
}
MAG_ERROR_FACTOR = 1.0857362047581296
Z_SOLAR = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_thresholds(spec: str) -> Dict[str, float]:
    """Accept 'F378:0.8,F395:0.7,...' or a single uniform float."""
    spec = spec.strip()
    try:
        uniform = float(spec)
        return {f: uniform for f in NARROW_BANDS}
    except ValueError:
        pass
    thresholds = {}
    for item in spec.split(","):
        key, val = item.split(":")
        thresholds[key.strip().upper()] = float(val)
    for f in NARROW_BANDS:
        thresholds.setdefault(f, DEFAULT_THRESHOLDS[f])
    return thresholds


def find_error_columns(df: pd.DataFrame) -> Dict[str, str]:
    found = {}
    for f in NARROW_BANDS:
        for suffix in ("_3_corr", "_3", ""):
            col = f"MAGERR_{f}{suffix}"
            if col in df.columns:
                found[f] = col
                break
    return found


def find_snr_columns(df: pd.DataFrame) -> Dict[str, str]:
    found = {}
    for f in NARROW_BANDS:
        col = f"SNR_{f}_3"
        if col in df.columns:
            found[f] = col
        else:
            for suffix in ("_3_corr", ""):
                col2 = f"SNR_{f}{suffix}"
                if col2 in df.columns:
                    found[f] = col2
                    break
    return found


def count_good_filters(
    row: pd.Series,
    err_cols: Dict[str, str],
    thresholds: Dict[str, float],
) -> int:
    n = 0
    for filt, col in err_cols.items():
        val = row.get(col)
        if pd.notna(val) and 0 < val < thresholds[filt]:
            n += 1
    return n


def wilson(k: int, n: int, z: float = 1.0) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def magnitude_bin(v: float, w: float) -> float:
    return math.floor(v / w) * w


def estimate_snr_depth(
    curve: Sequence[dict], threshold: float, bin_width: float
) -> Optional[float]:
    pts = [(p["bin_center"], p["median_snr"])
           for p in curve if p["median_snr"] is not None]
    pts.sort()
    for (mb, sb), (mf, sf) in zip(pts, pts[1:]):
        if (sb >= threshold and sf < threshold and mf > mb
                and mf - mb <= bin_width + 1e-6):
            f = (threshold - sb) / (sf - sb)
            return mb + f * (mf - mb)
    return None


def write_csv(path: Path, rows: Sequence[dict], cols: Sequence[str]) -> None:
    import csv as _csv
    with path.open("w", encoding="utf-8", newline="") as h:
        w = _csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Figure 1 (Section 2): S/N vs magnitude
# ---------------------------------------------------------------------------
def make_snr_figure(
    snr_curves: Dict[str, List[dict]],
    out_path: Path,
    bin_width: float,
    min_n_per_bin: int = 5,
    xlim: Tuple[float, float] = (17.0, 22.5),
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": 7.0,
        "axes.labelsize": 9.5, "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)

    for filt in NARROW_BANDS:
        curve = snr_curves.get(filt, [])
        pts = [
            (p["bin_center"], p["median_snr"])
            for p in curve
            if p["median_snr"] is not None and p["n_with_snr"] >= min_n_per_bin
        ]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(
            xs, ys, marker="o", markersize=2.5, linewidth=0.9,
            color=FILTER_COLORS[filt], label=filt,
        )

    ax.axhline(5.0, color="0.4", linestyle="--", linewidth=0.7)
    ax.axhline(3.0, color="0.4", linestyle=":", linewidth=0.7)
    ax.text(0.98, 0.96, r"S/N $=5$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="0.35")
    ax.text(0.98, 0.86, r"S/N $=3$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="0.35")

    ax.set_xlabel("Magnitude (bin centre, mag)")
    ax.set_ylabel("Median S/N")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.5, 200)
    ax.legend(loc="lower left", ncol=2, columnspacing=0.8,
              handlelength=1.2, borderaxespad=0.4)
    ax.grid(True, which="major", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)

    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 (Section 3): completeness vs r
# ---------------------------------------------------------------------------
def make_completeness_figure(
    combined_df: pd.DataFrame,
    out_path: Path,
    chi2_max: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "axes.labelsize": 9.5, "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)

    x = combined_df["r_center"].values
    ax.errorbar(
        x, combined_df["frac_phot"],
        yerr=[combined_df["frac_phot"] - combined_df["phot_lo"],
              combined_df["phot_hi"] - combined_df["frac_phot"]],
        marker="s", markersize=3.5, linewidth=1.0, capsize=2,
        color="#3d5fbf", label="Photometric quality",
    )
    ax.errorbar(
        x, combined_df["frac_fit"],
        yerr=[combined_df["frac_fit"] - combined_df["fit_lo"],
              combined_df["fit_hi"] - combined_df["frac_fit"]],
        marker="^", markersize=3.5, linewidth=1.0, capsize=2,
        color="#e05a3c",
        label=rf"CIGALE $\chi^2_\nu \leq {chi2_max}$",
    )
    ax.errorbar(
        x, combined_df["frac_both"],
        yerr=[combined_df["frac_both"] - combined_df["both_lo"],
              combined_df["both_hi"] - combined_df["frac_both"]],
        marker="o", markersize=3.5, linewidth=1.2, capsize=2,
        color="#2b3a8f", label="Both",
    )

    ax.axhline(50.0, color="0.4", linestyle=":", linewidth=0.7)
    ax.axhline(90.0, color="0.4", linestyle="--", linewidth=0.7)
    ax.text(0.02, 0.51, "50%", transform=ax.transAxes,
            fontsize=7, color="0.4")
    ax.text(0.02, 0.91, "90%", transform=ax.transAxes,
            fontsize=7, color="0.4")

    ax.set_xlabel(r"$r$ magnitude")
    ax.set_ylabel("Fraction of T17 parent sample (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)

    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parent", required=True, type=Path,
                   help="Full-sample photometry CSV (with T17ID and filter columns).")
    p.add_argument("--cigale", required=True, type=Path,
                   help="CIGALE FITS output (hdu=1).")
    p.add_argument("--reference", required=True, type=Path,
                   help="TAP catalog with T17ID and rmag.")
    p.add_argument("--reference-mag", default="rmag")
    p.add_argument("--id-col", default="T17ID")
    p.add_argument("--cigale-id-col", default="id")
    p.add_argument("--aperture", default="3",
                   help="Aperture suffix for MAG/SNR/MAGERR columns (default 3).")
    p.add_argument(
        "--err-thresholds",
        default="F378:0.8,F395:0.7,F410:0.5,F430:0.5,F515:0.4,F660:0.3,F861:0.4",
        help="Either a single float (uniform cut) or 'FILTER:VALUE,...' per-filter.",
    )
    p.add_argument("--min-filters", type=int, default=3)
    p.add_argument("--chi2-max", type=float, default=2.0)
    p.add_argument("--bin-width", type=float, default=0.5)
    p.add_argument("--min-bin-size", type=int, default=10,
                   help="Min objects per bin for the completeness curve (default 10).")
    p.add_argument("--min-n-per-snr-bin", type=int, default=5,
                   help="Min objects per magnitude bin to plot an S/N point "
                        "(default 5).")
    p.add_argument("--snr-xlim-low", type=float, default=17.0)
    p.add_argument("--snr-xlim-high", type=float, default=22.5)
    p.add_argument("--plot-format", choices=("pdf", "png"), default="pdf")
    p.add_argument("--output-dir", type=Path, default=Path("completeness"))
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.err_thresholds)
    print("Per-filter error thresholds used:")
    for f in NARROW_BANDS:
        print(f"  {f}: {thresholds[f]:.2f} mag")
    print(f"Min filters passing: {args.min_filters}")
    print(f"chi2_max: {args.chi2_max}\n")

    # --- Load inputs -----------------------------------------------------
    parent = pd.read_csv(args.parent)
    parent[args.id_col] = parent[args.id_col].astype(str).str.strip()

    ref = pd.read_csv(args.reference)[[args.id_col, args.reference_mag]]
    ref[args.id_col] = ref[args.id_col].astype(str).str.strip()
    ref = ref.drop_duplicates(subset=args.id_col, keep="first")
    rmag_map = dict(zip(ref[args.id_col], ref[args.reference_mag]))

    from astropy.table import Table
    cig = Table.read(str(args.cigale), hdu=1).to_pandas()
    cig_id = args.cigale_id_col if args.cigale_id_col in cig.columns else "id"
    cig[cig_id] = cig[cig_id].astype(str).str.strip()
    cig = cig[[cig_id, "best.reduced_chi_square"]].rename(
        columns={cig_id: args.id_col, "best.reduced_chi_square": "redchi2"}
    )

    err_cols = find_error_columns(parent)
    snr_cols = find_snr_columns(parent)
    print(f"Error columns found: {list(err_cols.values())}")
    print(f"SNR columns found  : {list(snr_cols.values())}\n")

    # --- Per-object classification ---------------------------------------
    parent = parent.copy()
    parent[args.reference_mag] = parent[args.id_col].map(rmag_map).astype(float)
    parent["n_good_filters"] = parent.apply(
        lambda r: count_good_filters(r, err_cols, thresholds), axis=1
    )
    parent = parent.merge(cig, on=args.id_col, how="left")
    parent["has_good_phot"] = parent["n_good_filters"] >= args.min_filters
    parent["has_good_fit"] = (
        parent["redchi2"].notna() & (parent["redchi2"] <= args.chi2_max)
    )
    parent["combined"] = parent["has_good_phot"] & parent["has_good_fit"]

    parent = parent[
        parent[args.reference_mag].notna()
        & (parent[args.reference_mag] > 0)
        & (parent[args.reference_mag] < 30)
    ]

    n_parent = len(parent)
    n_phot = int(parent["has_good_phot"].sum())
    n_fit = int(parent["has_good_fit"].sum())
    n_both = int(parent["combined"].sum())

    print(f"Parent objects with valid rmag: {n_parent}")
    print(f"  with good photometry      : {n_phot}")
    print(f"  with good SED fit         : {n_fit}")
    print(f"  with both (combined)      : {n_both}\n")

    # --- S/N curves per filter -------------------------------------------
    snr_curves: Dict[str, List[dict]] = {f: [] for f in NARROW_BANDS}
    snr_depth_rows: List[dict] = []
    for filt in NARROW_BANDS:
        snr_by_bin: Dict[float, List[float]] = defaultdict(list)
        err_by_bin: Dict[float, List[float]] = defaultdict(list)
        mag_col = f"MAG_{filt}_{args.aperture}"
        for _, row in parent.iterrows():
            mag = row.get(mag_col)
            if pd.isna(mag) or mag <= 0 or mag >= 90:
                continue
            b = magnitude_bin(float(mag), args.bin_width)
            snr_col = snr_cols.get(filt)
            if snr_col is not None:
                s = row.get(snr_col)
                if pd.notna(s) and s > 0:
                    snr_by_bin[b].append(float(s))
            err_col = err_cols.get(filt)
            if err_col is not None:
                e = row.get(err_col)
                if pd.notna(e) and e > 0:
                    err_by_bin[b].append(float(e))
        for b in sorted(set(snr_by_bin) | set(err_by_bin)):
            snrs = snr_by_bin.get(b, [])
            errs = err_by_bin.get(b, [])
            snr_curves[filt].append({
                "bin_center": b + args.bin_width / 2,
                "n_with_snr": len(snrs),
                "median_snr": statistics.median(snrs) if snrs else None,
                "n_with_err": len(errs),
                "median_err": statistics.median(errs) if errs else None,
            })
        curve = snr_curves[filt]
        snr_depth_rows.append({
            "filter": filt,
            "snr_depth_3": estimate_snr_depth(curve, 3.0, args.bin_width),
            "snr_depth_5": estimate_snr_depth(curve, 5.0, args.bin_width),
        })

    # --- Combined completeness vs rmag -----------------------------------
    bins = np.arange(15.0, 23.0 + 1e-9, args.bin_width)
    parent["bin"] = pd.cut(parent[args.reference_mag], bins)
    grouped = parent.groupby("bin", observed=True)

    rows = []
    for interval, grp in grouped:
        n = len(grp)
        if n < args.min_bin_size:
            continue
        k_phot = int(grp["has_good_phot"].sum())
        k_fit = int(grp["has_good_fit"].sum())
        k_both = int(grp["combined"].sum())
        phot_lo, phot_hi = wilson(k_phot, n)
        fit_lo, fit_hi = wilson(k_fit, n)
        both_lo, both_hi = wilson(k_both, n)
        rows.append({
            "r_center": interval.mid, "n_parent": n,
            "n_phot": k_phot, "frac_phot": 100 * k_phot / n,
            "phot_lo": phot_lo, "phot_hi": phot_hi,
            "n_fit": k_fit, "frac_fit": 100 * k_fit / n,
            "fit_lo": fit_lo, "fit_hi": fit_hi,
            "n_both": k_both, "frac_both": 100 * k_both / n,
            "both_lo": both_lo, "both_hi": both_hi,
        })
    combined_df = pd.DataFrame(rows)

    subsample_med = float(np.median(
        parent[parent["combined"]][args.reference_mag]
    )) if n_both else float("nan")
    parent_med = float(np.median(parent[args.reference_mag]))

    # --- Write CSVs ------------------------------------------------------
    write_csv(
        args.output_dir / "snr_depth_per_filter.csv",
        snr_depth_rows,
        ["filter", "snr_depth_3", "snr_depth_5"],
    )
    combined_df.to_csv(
        args.output_dir / "combined_completeness.csv", index=False,
    )
    write_csv(
        args.output_dir / "snr_depth_curves.csv",
        [entry | {"filter": f}
         for f, curve in snr_curves.items() for entry in curve],
        ["filter", "bin_center", "n_with_snr", "median_snr",
         "n_with_err", "median_err"],
    )

    # --- Summary text ----------------------------------------------------
    summary = args.output_dir / "completeness_summary.txt"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write(f"Parent objects with valid rmag : {n_parent}\n")
        fh.write(f"With good photometry           : {n_phot}\n")
        fh.write(f"With good CIGALE fit           : {n_fit}\n")
        fh.write(f"With both (final scientific)   : {n_both}\n\n")
        fh.write(f"Median r (parent)              : {parent_med:.3f}\n")
        fh.write(f"Median r (combined subsample)  : {subsample_med:.3f}\n")
        fh.write(f"Offset                         : "
                 f"{subsample_med - parent_med:+.3f} mag\n\n")
        fh.write("Per-filter S/N depths (mag):\n")
        for row in snr_depth_rows:
            d3 = f"{row['snr_depth_3']:.2f}" if row["snr_depth_3"] else "n/a"
            d5 = f"{row['snr_depth_5']:.2f}" if row["snr_depth_5"] else "n/a"
            fh.write(f"  {row['filter']}: S/N=3 -> {d3}   S/N=5 -> {d5}\n")
        fh.write("\nCombined completeness vs rmag:\n")
        fh.write("  r_center  n  phot%  fit%  both%\n")
        for _, r in combined_df.iterrows():
            fh.write(
                f"  {r['r_center']:6.2f}  {int(r['n_parent']):4d}  "
                f"{r['frac_phot']:5.1f}  {r['frac_fit']:5.1f}  "
                f"{r['frac_both']:5.1f}\n"
            )

    # --- Figures ---------------------------------------------------------
    snr_fig = args.output_dir / f"snr_depth.{args.plot_format}"
    comp_fig = args.output_dir / f"completeness.{args.plot_format}"

    make_snr_figure(
        snr_curves,
        snr_fig,
        bin_width=args.bin_width,
        min_n_per_bin=args.min_n_per_snr_bin,
        xlim=(args.snr_xlim_low, args.snr_xlim_high),
    )
    make_completeness_figure(combined_df, comp_fig, chi2_max=args.chi2_max)

    # --- Console summary -------------------------------------------------
    print("Per-filter S/N depths (mag):")
    for row in snr_depth_rows:
        d3 = f"{row['snr_depth_3']:.2f}" if row["snr_depth_3"] else "n/a"
        d5 = f"{row['snr_depth_5']:.2f}" if row["snr_depth_5"] else "n/a"
        print(f"  {row['filter']}:  S/N=3 -> {d3}   S/N=5 -> {d5}")

    print("\nCombined completeness vs rmag:")
    print(combined_df.to_string(index=False, float_format="%.1f"))

    print(f"\nMedian r (parent)             : {parent_med:.3f}")
    print(f"Median r (combined subsample) : {subsample_med:.3f}")
    print(f"Offset                        : "
          f"{subsample_med - parent_med:+.3f} mag")

    print(f"\nSaved {args.output_dir / 'completeness_summary.txt'}")
    print(f"Saved {args.output_dir / 'combined_completeness.csv'}")
    print(f"Saved {args.output_dir / 'snr_depth_per_filter.csv'}")
    print(f"Saved {snr_fig}   (Section 2 figure)")
    print(f"Saved {comp_fig}  (Section 3 figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

