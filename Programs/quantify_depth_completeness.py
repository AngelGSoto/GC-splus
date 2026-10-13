#!/usr/bin/env python3
"""Estimate per-filter photometric depth and catalog-based completeness.

The depth estimate is the measured magnitude where the binned median S/N
crosses a requested threshold. Completeness is measured against a parent
catalog supplied with ``--reference``; it is not an artificial-source
recovery test and therefore does not measure image-detection completeness.

Example (wide catalog):
    python Programs/quantify_depth_completeness.py \\
        --photometry anac_data/Results/all_fields_photometry.csv \\
        --reference TAP_1_J_MNRAS_3444_gc.csv --reference-mag rmag \\
        --output-dir anac_data/Results/depth_completeness

Example (long catalog with source_id, filter, mag, mag_err, and snr columns):
    python Programs/quantify_depth_completeness.py \\
        --photometry teste/ngc5128_splus_photometry_3arcsec.csv \\
        --id-col source_id --format long
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FILTERS = ("F378", "F395", "F410", "F430", "F515", "F660", "F861")
MAG_ERROR_FACTOR = 1.0857362047581296
WILSON_Z_68 = 1.0


def read_csv(path: Path) -> Tuple[List[str], List[dict]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is empty or has no CSV header")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def parse_number(value: Optional[str], *, positive: bool = False) -> Optional[float]:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def valid_magnitude(value: Optional[str]) -> Optional[float]:
    magnitude = parse_number(value)
    if magnitude is None or magnitude <= 0 or magnitude >= 90:
        return None
    return magnitude


def choose_id_column(headers: Sequence[str], requested: str) -> str:
    if requested != "auto":
        if requested not in headers:
            raise ValueError(f"ID column {requested!r} is absent; columns: {', '.join(headers)}")
        return requested
    for candidate in ("T17ID", "source_id", "ID", "id", "recno"):
        if candidate in headers:
            return candidate
    raise ValueError("Could not infer an ID column; pass --id-col explicitly")


def rows_by_id(rows: Iterable[dict], id_col: str, label: str) -> Dict[str, dict]:
    result = {}
    for line_number, row in enumerate(rows, start=2):
        key = (row.get(id_col) or "").strip()
        if not key:
            raise ValueError(f"{label}: blank identifier on CSV line {line_number}")
        if key in result:
            raise ValueError(f"{label}: duplicate identifier {key!r}")
        result[key] = row
    return result


def get_snr(row: Optional[dict], filter_name: str, aperture: str, *, long_format: bool) -> Optional[float]:
    if row is None:
        return None
    snr_col = "snr" if long_format else f"SNR_{filter_name}_{aperture}"
    snr = parse_number(row.get(snr_col), positive=True)
    if snr is not None:
        return snr
    error_col = "mag_err" if long_format else f"MAGERR_{filter_name}_{aperture}"
    error = parse_number(row.get(error_col), positive=True)
    return MAG_ERROR_FACTOR / error if error is not None else None


def get_measurement(
    row: Optional[dict], filter_name: str, aperture: str, *, long_format: bool
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if row is None:
        return None, None, None
    mag_col = "mag" if long_format else f"MAG_{filter_name}_{aperture}"
    error_col = "mag_err" if long_format else f"MAGERR_{filter_name}_{aperture}"
    magnitude = valid_magnitude(row.get(mag_col))
    error = parse_number(row.get(error_col), positive=True)
    snr = get_snr(row, filter_name, aperture, long_format=long_format)
    return magnitude, error, snr


def magnitude_bin(value: float, width: float) -> float:
    return math.floor(value / width) * width


def wilson_interval(successes: int, total: int) -> Tuple[Optional[float], Optional[float]]:
    if total == 0:
        return None, None
    fraction = successes / total
    denominator = 1 + WILSON_Z_68**2 / total
    center = (fraction + WILSON_Z_68**2 / (2 * total)) / denominator
    half_width = (
        WILSON_Z_68
        * math.sqrt(
            fraction * (1 - fraction) / total
            + WILSON_Z_68**2 / (4 * total**2)
        )
        / denominator
    )
    return 100 * max(0.0, center - half_width), 100 * min(1.0, center + half_width)


def estimate_snr_depth(curve: Sequence[dict], threshold: float, bin_width: float) -> Optional[float]:
    points = [
        (point["mag_bin_center"], point["median_snr"])
        for point in curve
        if point["median_snr"] is not None
    ]
    points.sort()
    for (bright_mag, bright_snr), (faint_mag, faint_snr) in zip(points, points[1:]):
        if (
            bright_snr >= threshold
            and faint_snr < threshold
            and faint_mag > bright_mag
            and faint_mag - bright_mag <= 1.5 * bin_width
        ):
            fraction = (threshold - bright_snr) / (faint_snr - bright_snr)
            return bright_mag + fraction * (faint_mag - bright_mag)
    return None


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate S/N depth and completeness relative to a parent GC catalog."
    )
    parser.add_argument("--photometry", required=True, type=Path, help="S-PLUS photometry CSV")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Parent catalog containing the full target sample (required for completeness)",
    )
    parser.add_argument(
        "--reference-mag",
        help="Magnitude column used to bin the parent sample, e.g. rmag; requires --reference",
    )
    parser.add_argument(
        "--id-col",
        help="Set both ID columns to this name; use the individual options if they differ",
    )
    parser.add_argument(
        "--photometry-id-col",
        default="auto",
        help="Photometry ID column, or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--reference-id-col",
        default="auto",
        help="Parent-catalog ID column, or 'auto' (default: auto)",
    )
    parser.add_argument("--field-col", default="FIELD", help="Field column (default: FIELD)")
    parser.add_argument(
        "--format",
        choices=("auto", "wide", "long"),
        default="auto",
        help="Photometry layout; auto-detects long tables with filter/mag columns",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=list(DEFAULT_FILTERS),
        help="Filters to analyze (default: the seven S-PLUS narrow bands)",
    )
    parser.add_argument("--aperture", default="3", help="Aperture suffix in wide columns (default: 3)")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Magnitude-bin width (default: 0.5)")
    parser.add_argument(
        "--min-snr",
        type=float,
        default=0.0,
        help="Minimum S/N for the 'usable' fraction; default 0 applies no S/N cut",
    )
    parser.add_argument(
        "--max-mag-error",
        type=float,
        help="Optional maximum magnitude error for the 'usable' fraction",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("depth_completeness"))
    return parser


def run(args: argparse.Namespace) -> None:
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if args.min_snr < 0:
        raise ValueError("--min-snr cannot be negative")
    if args.max_mag_error is not None and args.max_mag_error <= 0:
        raise ValueError("--max-mag-error must be positive")
    if args.reference_mag and not args.reference:
        raise ValueError("--reference-mag requires --reference")

    phot_headers, phot_rows = read_csv(args.photometry)
    phot_id_col = choose_id_column(
        phot_headers, args.id_col or args.photometry_id_col
    )
    if args.format == "auto":
        long_format = {"filter", "mag"}.issubset(phot_headers)
    else:
        long_format = args.format == "long"

    phot_by_id = {}
    if long_format:
        required = {"filter", "mag"}
        missing = required.difference(phot_headers)
        if missing:
            raise ValueError(f"Long-format catalog missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(phot_rows, start=2):
            key = (row.get(phot_id_col) or "").strip()
            filter_name = (row.get("filter") or "").strip().upper()
            if not key or not filter_name:
                raise ValueError(f"Photometry: blank ID or filter on CSV line {line_number}")
            source = phot_by_id.setdefault(key, {"row": row, "measurements": {}})
            if filter_name in source["measurements"]:
                raise ValueError(f"Photometry: duplicate {filter_name} measurement for {key!r}")
            source["measurements"][filter_name] = row
    else:
        if not any(f"MAG_{name}_{args.aperture}" in phot_headers for name in args.filters):
            raise ValueError(
                "No expected wide-format magnitude columns found; check --format, --filters, "
                "and --aperture"
            )
        phot_by_id = {
            key: {"row": row, "measurements": {}}
            for key, row in rows_by_id(phot_rows, phot_id_col, "Photometry").items()
        }

    reference_rows = None
    reference_headers = []
    if args.reference:
        reference_headers, rows = read_csv(args.reference)
        reference_id_col = choose_id_column(
            reference_headers, args.id_col or args.reference_id_col
        )
        if args.reference_mag and args.reference_mag not in reference_headers:
            raise ValueError(
                f"Reference magnitude column {args.reference_mag!r} is absent from {args.reference}"
            )
        reference_rows = rows_by_id(rows, reference_id_col, "Reference catalog")
    elif args.reference_mag:
        raise ValueError("--reference-mag requires --reference")

    if reference_rows is not None:
        sample = reference_rows
    else:
        sample = {key: source["row"] for key, source in phot_by_id.items()}

    field_col = args.field_col
    def sample_field(key: str, row: dict) -> str:
        if reference_rows is not None:
            if field_col not in reference_headers:
                return "ALL"
            return (row.get(field_col) or "").strip() or "UNKNOWN"
        return (
            (row.get(field_col) or "").strip()
            or (phot_by_id.get(key, {}).get("row", {}).get(field_col) or "").strip()
            or "ALL"
        )

    fields = sorted({sample_field(key, row) for key, row in sample.items()})
    summary_rows = []
    completeness_rows = []
    depth_rows = []

    for field in fields:
        sample_items = []
        for key, row in sample.items():
            if sample_field(key, row) == field:
                sample_items.append((key, row, phot_by_id.get(key)))

        for filter_name in args.filters:
            filter_name = filter_name.upper()
            if not long_format:
                mag_col = f"MAG_{filter_name}_{args.aperture}"
                if mag_col not in phot_headers:
                    raise ValueError(f"Photometry column {mag_col!r} is missing")
            snr_by_mag_bin = defaultdict(list)
            error_by_mag_bin = defaultdict(list)
            completeness_by_bin = defaultdict(lambda: {"n": 0, "detected": 0, "usable": 0})
            n_matched = n_detected = n_usable = n_reference_mag = 0

            for key, reference_row, source in sample_items:
                measurement_row = None
                if source is not None:
                    if long_format:
                        measurement_row = source["measurements"].get(filter_name)
                    else:
                        measurement_row = source["row"]
                magnitude, error, snr = get_measurement(
                    measurement_row, filter_name, args.aperture, long_format=long_format
                )
                matched = source is not None and (
                    not long_format or filter_name in source["measurements"]
                )
                detected = magnitude is not None
                usable = detected and (snr is None or snr >= args.min_snr)
                if args.min_snr > 0:
                    usable = detected and snr is not None and snr >= args.min_snr
                if args.max_mag_error is not None:
                    usable = usable and error is not None and error <= args.max_mag_error
                n_matched += int(matched)
                n_detected += int(detected)
                n_usable += int(usable)

                if magnitude is not None:
                    mag_bin = magnitude_bin(magnitude, args.bin_width)
                    if snr is not None:
                        snr_by_mag_bin[mag_bin].append(snr)
                    if error is not None:
                        error_by_mag_bin[mag_bin].append(error)

                if args.reference is not None and args.reference_mag:
                    ref_magnitude = valid_magnitude(reference_row.get(args.reference_mag))
                    if ref_magnitude is not None:
                        ref_bin = magnitude_bin(ref_magnitude, args.bin_width)
                        bin_data = completeness_by_bin[ref_bin]
                        bin_data["n"] += 1
                        bin_data["detected"] += int(detected)
                        bin_data["usable"] += int(usable)
                        n_reference_mag += 1

            depth_curve = []
            for mag_bin in sorted(set(snr_by_mag_bin) | set(error_by_mag_bin)):
                snrs = snr_by_mag_bin.get(mag_bin, [])
                errors = error_by_mag_bin.get(mag_bin, [])
                depth_curve.append(
                    {
                        "field": field,
                        "filter": filter_name,
                        "mag_bin_low": mag_bin,
                        "mag_bin_high": mag_bin + args.bin_width,
                        "mag_bin_center": mag_bin + args.bin_width / 2,
                        "n_with_snr": len(snrs),
                        "median_snr": statistics.median(snrs) if snrs else None,
                        "n_with_mag_error": len(errors),
                        "median_mag_error": statistics.median(errors) if errors else None,
                    }
                )
            depth_rows.extend(depth_curve)
            depth_snr3 = estimate_snr_depth(depth_curve, 3.0, args.bin_width)
            depth_snr5 = estimate_snr_depth(depth_curve, 5.0, args.bin_width)

            sample_size = len(sample_items)
            summary_rows.append(
                {
                    "field": field,
                    "filter": filter_name,
                    "n_sample": sample_size,
                    "n_catalog_match": n_matched,
                    "n_detected": n_detected,
                    "n_usable": n_usable,
                    "n_reference_mag": n_reference_mag,
                    "detection_fraction_pct": (
                        100 * n_detected / sample_size if sample_size else None
                    ),
                    "usable_fraction_pct": (
                        100 * n_usable / sample_size if sample_size else None
                    ),
                    "completeness_binned_on": args.reference_mag or "",
                    "median_snr_depth_3": depth_snr3,
                    "median_snr_depth_5": depth_snr5,
                }
            )

            for mag_bin, values in sorted(completeness_by_bin.items()):
                n_bin = values["n"]
                det_low, det_high = wilson_interval(values["detected"], n_bin)
                use_low, use_high = wilson_interval(values["usable"], n_bin)
                completeness_rows.append(
                    {
                        "field": field,
                        "filter": filter_name,
                        "reference_mag": args.reference_mag,
                        "mag_bin_low": mag_bin,
                        "mag_bin_high": mag_bin + args.bin_width,
                        "mag_bin_center": mag_bin + args.bin_width / 2,
                        "n_reference": n_bin,
                        "n_detected": values["detected"],
                        "detection_completeness_pct": 100 * values["detected"] / n_bin,
                        "detection_68pct_low": det_low,
                        "detection_68pct_high": det_high,
                        "n_usable": values["usable"],
                        "usable_completeness_pct": 100 * values["usable"] / n_bin,
                        "usable_68pct_low": use_low,
                        "usable_68pct_high": use_high,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "depth_summary.csv"
    completeness_path = args.output_dir / "completeness_by_magnitude.csv"
    depth_path = args.output_dir / "snr_depth_curves.csv"
    write_csv(
        summary_path,
        summary_rows,
        (
            "field", "filter", "n_sample", "n_catalog_match", "n_detected", "n_usable",
            "n_reference_mag", "detection_fraction_pct", "usable_fraction_pct",
            "completeness_binned_on", "median_snr_depth_3", "median_snr_depth_5",
        ),
    )
    write_csv(
        completeness_path,
        completeness_rows,
        (
            "field", "filter", "reference_mag", "mag_bin_low", "mag_bin_high",
            "mag_bin_center", "n_reference", "n_detected", "detection_completeness_pct",
            "detection_68pct_low", "detection_68pct_high", "n_usable",
            "usable_completeness_pct", "usable_68pct_low", "usable_68pct_high",
        ),
    )
    write_csv(
        depth_path,
        depth_rows,
        (
            "field", "filter", "mag_bin_low", "mag_bin_high", "mag_bin_center",
            "n_with_snr", "median_snr", "n_with_mag_error", "median_mag_error",
        ),
    )

    print(f"Saved {summary_path}")
    print(f"Saved {depth_path}")
    if args.reference:
        print(f"Saved {completeness_path}")
    else:
        print("No --reference supplied: completeness is not estimated.")
    print("\nMedian-S/N depth estimates (mag):")
    print(f"{'Field':<16} {'Filter':<8} {'S/N=3':>10} {'S/N=5':>10}")
    for row in summary_rows:
        d3 = f"{row['median_snr_depth_3']:.2f}" if row["median_snr_depth_3"] is not None else "n/a"
        d5 = f"{row['median_snr_depth_5']:.2f}" if row["median_snr_depth_5"] is not None else "n/a"
        print(f"{row['field']:<16} {row['filter']:<8} {d3:>10} {d5:>10}")
    if args.reference and not args.reference_mag:
        print(
            "\nNote: parent-catalog matching was used for overall detection fractions; "
            "pass --reference-mag to also calculate magnitude-binned completeness."
        )
    if not args.reference:
        print(
            "Warning: fractions without a separate parent catalog describe only the supplied "
            "photometry table and are not sample completeness."
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, csv.Error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
