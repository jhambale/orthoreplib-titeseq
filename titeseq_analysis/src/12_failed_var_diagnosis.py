#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2025
@author: jhambale

Diagnoses variants that failed Kd fitting in script 10.

For every experiment directory supplied with -i, the script:
  1. Loads tite_seq_Kds_all_variants.csv  (fitted Kd results + pass/fail labels)
  2. Loads tite_seq_raw_repX.csv files    (raw bin counts per concentration)
  3. Joins them on aa_sequence so each failed variant carries both its
     fit metadata and its raw count data.
  4. Computes per-variant diagnostics:
        - total_counts_repX   : sum of all raw bin counts across all bins and
                                concentrations for replicate X
        - max_counts_repX     : maximum single-bin count across all concs / bins
        - low_count_flag_repX : True if total_counts < --count-thresh
        - dynamic_range_repX  : max(score) - min(score) across shared concs
                                (re-derived directly from the Kd CSV's fitted
                                 As parameter, a proxy for the log10 fluorescence
                                 amplitude used during fitting)
        - fit_residual_repX   : mean absolute residual between the raw scores
                                (derived from raw counts) and the best-fit Hill
                                curve, re-evaluated at each concentration
  5. Writes a per-experiment CSV of failed-variant diagnostics.
  6. Prints a summary table grouped by fail_reason.

Example run
-----------
python 12_failed_var_diagnosis.py \\
    -i ../../titeseq_data/outputs/jh_001 \\
       ../../titeseq_data/outputs/jh_002 \\
       ../../titeseq_data/outputs/jh_003 \\
    -o ../../titeseq_data/outputs/failed_diagnosis \\
    --count-thresh 50
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hill function (mirrors script 10)
# ---------------------------------------------------------------------------


def func_E50(concs: np.ndarray, Kd: float, As: float, B: float) -> np.ndarray:
    """Standard Hill / saturation binding function returning log10(fluorescence)."""
    fluorescence = (concs / (concs + Kd)) * As + B
    return np.log10(np.maximum(fluorescence, 1e-9))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_concentration_label(label: str) -> float:
    """Convert a concentration label such as '80nM' or '10^-7M' to nM float."""
    cleaned = label.replace(" ", "")
    nm_match = re.match(r"^([\d.]+)nM$", cleaned, re.IGNORECASE)
    if nm_match:
        return float(nm_match.group(1))
    sci_match = re.search(r"10\^-(?P<val>.*)M", cleaned)
    if sci_match:
        return 10 ** (-float(sci_match.group("val"))) * 1e9
    raise ValueError(f"Cannot parse concentration label: '{label}'")


def load_raw_counts(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all tite_seq_raw_repX.csv files from raw_dir.

    Returns a dict mapping rep_id (e.g. 'rep1') -> DataFrame.
    """
    raw_dfs: dict[str, pd.DataFrame] = {}
    for f in sorted(raw_dir.glob("tite_seq_raw_rep*.csv")):
        match = re.search(r"rep(\d+)", f.name, re.IGNORECASE)
        rep_id = f"rep{match.group(1)}" if match else f.stem
        raw_dfs[rep_id] = pd.read_csv(f)
    if not raw_dfs:
        print(f"  [WARN] No tite_seq_raw_repX.csv files found in {raw_dir}")
    return raw_dfs


def total_counts_for_variant(row: pd.Series, count_cols: list[str]) -> int:
    """Sum all raw bin counts for a variant across the given columns."""
    return int(row[count_cols].fillna(0).sum())


def max_count_for_variant(row: pd.Series, count_cols: list[str]) -> int:
    """Maximum raw bin count for a variant across the given columns."""
    return int(row[count_cols].fillna(0).max())


def get_conc_labels_from_raw(df: pd.DataFrame) -> list[str]:
    """Extract ordered unique concentration labels from raw count column names.

    Raw count columns follow the pattern  {conc}_{bin}_raw_counts, e.g.
    '0.64nM_bin0_raw_counts'. We extract the {conc} part in order.
    """
    seen: dict[str, None] = {}  # ordered set via insertion-order dict
    for col in df.columns:
        m = re.match(r"^(.+?)_bin\d+_raw_counts$", col)
        if m:
            seen[m.group(1)] = None
    return list(seen.keys())


def compute_score_from_counts(row: pd.Series, conc_label: str) -> float:
    """Compute log10(weighted mean fluorescence) from raw bin counts.

    Bins 0-3 are treated as uniform fluorescence intervals [0, 1, 2, 3]
    (relative units), consistent with how normalised scores are computed
    in scripts 08/09.  Returns NaN if total counts == 0.
    """
    bin_weights = np.array([0, 1, 2, 3], dtype=float)
    counts = np.array(
        [row.get(f"{conc_label}_bin{b}_raw_counts", 0) for b in range(4)],
        dtype=float,
    )
    total = counts.sum()
    if total == 0:
        return np.nan
    weighted_mean = (counts * bin_weights).sum() / total
    return np.log10(max(weighted_mean, 1e-9))


def fit_residual(
    scores: np.ndarray,
    concs: np.ndarray,
    Kd: float,
    As: float,
    B: float,
) -> float:
    """Mean absolute residual between observed log10 scores and fitted Hill curve.

    Only finite (non-NaN) score points are included.
    Returns NaN if fewer than 2 finite points exist or if fit parameters are NaN.
    """
    if any(np.isnan(v) for v in [Kd, As, B]):
        return np.nan
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return np.nan
    predicted = func_E50(concs[mask], Kd, As, B)
    return float(np.mean(np.abs(scores[mask] - predicted)))


# ---------------------------------------------------------------------------
# Per-experiment processing
# ---------------------------------------------------------------------------


def process_experiment(
    exp_dir: Path,
    count_thresh: int,
) -> pd.DataFrame | None:
    """Process one experiment directory and return a diagnostics DataFrame
    for all failed variants (passed == False).

    Returns None if required input files are missing.
    """
    kd_csv = exp_dir / "tite_seq_Kds_all_variants.csv"
    raw_dir = exp_dir.parent / "raw_tite_seq_counts"

    if not kd_csv.exists():
        print(f"  [ERROR] Kd CSV not found: {kd_csv}")
        return None
    if not raw_dir.is_dir():
        print(f"  [ERROR] raw_tite_seq_counts directory not found: {raw_dir}")
        return None

    kd_df = pd.read_csv(kd_csv)
    raw_dfs = load_raw_counts(raw_dir)

    if not raw_dfs:
        return None

    # Work only with failed variants
    failed_df = kd_df[kd_df["passed"] == False].copy()
    if failed_df.empty:
        print(f"  No failed variants in {exp_dir.name}")
        return None

    print(
        f"  {exp_dir.name}: {len(failed_df)} failed variants "
        f"(out of {len(kd_df)} total)"
    )

    # Identify rep IDs present in both the Kd CSV and raw count files
    # Rep IDs in Kd CSV are inferred from columns like rep1_Kd, rep2_Kd …
    kd_rep_ids = sorted(
        {
            m.group(1)
            for col in kd_df.columns
            for m in [re.match(r"^(rep\d+)_Kd$", col)]
            if m
        }
    )
    available_reps = [r for r in kd_rep_ids if r in raw_dfs]
    if not available_reps:
        print(
            f"  [WARN] No overlapping rep IDs between Kd CSV and raw counts "
            f"for {exp_dir.name}.  Kd reps: {kd_rep_ids}, raw reps: {list(raw_dfs)}"
        )
        return None

    # For each replicate, join raw counts onto failed_df and compute diagnostics
    diag_df = failed_df[["variant_id", "aa_sequence", "fail_reason"]].copy()

    for rep in available_reps:
        raw = raw_dfs[rep]

        # --- Identify raw count columns for this replicate -----------------
        count_cols = [c for c in raw.columns if c.endswith("_raw_counts")]
        conc_labels = get_conc_labels_from_raw(raw)

        # --- Merge on aa_sequence ------------------------------------------
        raw_sub = raw[["aa_sequence"] + count_cols].copy()
        merged = diag_df.merge(raw_sub, on="aa_sequence", how="left")

        # --- Per-variant raw count statistics ------------------------------
        diag_df[f"{rep}_total_counts"] = (
            merged[count_cols].fillna(0).sum(axis=1).astype(int)
        )
        diag_df[f"{rep}_max_counts"] = (
            merged[count_cols].fillna(0).max(axis=1).astype(int)
        )
        diag_df[f"{rep}_low_count_flag"] = diag_df[f"{rep}_total_counts"] < count_thresh

        # --- Derive scores from raw counts and compute fit quality ---------
        # Concentration values (nM) in the same order as columns
        try:
            conc_nM = np.array([parse_concentration_label(lbl) for lbl in conc_labels])
        except ValueError as e:
            print(f"  [WARN] Could not parse concentration labels for {rep}: {e}")
            diag_df[f"{rep}_dynamic_range"] = np.nan
            diag_df[f"{rep}_fit_residual"] = np.nan
            continue

        # Build score series for every failed variant
        scores_list: list[np.ndarray] = []
        for _, failed_row in failed_df.iterrows():
            raw_row = raw[raw["aa_sequence"] == failed_row["aa_sequence"]]
            if raw_row.empty:
                scores_list.append(np.full(len(conc_labels), np.nan))
            else:
                r = raw_row.iloc[0]
                scores = np.array(
                    [compute_score_from_counts(r, lbl) for lbl in conc_labels]
                )
                scores_list.append(scores)

        dyn_ranges = []
        residuals = []
        for i, scores in enumerate(scores_list):
            finite = scores[np.isfinite(scores)]
            dyn_ranges.append(
                float(finite.max() - finite.min()) if len(finite) >= 2 else np.nan
            )

            # Retrieve fit params from Kd CSV (may be NaN for no_fit cases)
            kd_val = float(failed_df.iloc[i].get(f"{rep}_Kd", np.nan))
            as_val = float(failed_df.iloc[i].get(f"{rep}_As", np.nan))
            b_val = float(failed_df.iloc[i].get(f"{rep}_B", np.nan))
            residuals.append(fit_residual(scores, conc_nM, kd_val, as_val, b_val))

        diag_df[f"{rep}_dynamic_range"] = dyn_ranges
        diag_df[f"{rep}_fit_residual"] = residuals

    # Add experiment label
    diag_df.insert(0, "experiment", exp_dir.name)

    return diag_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose failed Kd variants: collect raw read counts and "
            "Hill-fit quality metrics for every variant where passed == False."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="EXP_DIR",
        help=(
            "One or more experiment output directories, each expected to contain "
            "tite_seq_Kds2/tite_seq_Kds_all_variants.csv and raw_tite_seq_counts/."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Directory where per-experiment diagnosis CSVs will be written.",
    )
    parser.add_argument(
        "--count-thresh",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Total raw count threshold below which a variant is flagged as "
            "low-coverage (low_count_flag). Default: 50."
        ),
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    exp_dirs = [Path(p) for p in args.input]
    missing = [p for p in exp_dirs if not p.is_dir()]
    if missing:
        for p in missing:
            print(f"[ERROR] Directory not found: {p}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_diag: list[pd.DataFrame] = []

    for exp_dir in exp_dirs:
        print(f"\nProcessing: {exp_dir}")
        diag = process_experiment(exp_dir, args.count_thresh)
        if diag is None:
            continue

        # Per-experiment CSV
        out_csv = out_dir / f"{exp_dir.name}_failed_diagnosis.csv"
        diag.to_csv(out_csv, index=False)
        print(f"  Saved -> {out_csv}")

        # Summary grouped by fail_reason
        rep_cols = [c for c in diag.columns if c.endswith("_total_counts")]
        summary_cols = (
            ["fail_reason"]
            + rep_cols
            + [c for c in diag.columns if c.endswith("_low_count_flag")]
        )
        summary = (
            diag[summary_cols]
            .groupby("fail_reason")
            .agg(
                count=("fail_reason", "size"),
                **{col: (col, "mean") for col in rep_cols},
                **{
                    col: (col, "sum")
                    for col in diag.columns
                    if col.endswith("_low_count_flag")
                },
            )
            .reset_index()
        )
        print(f"\n  Summary for {exp_dir.name}:")
        print(summary.to_string(index=False))

        all_diag.append(diag)

    # Combined CSV across all experiments
    if len(all_diag) > 1:
        combined = pd.concat(all_diag, ignore_index=True)
        combined_csv = out_dir / "all_experiments_failed_diagnosis.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined CSV saved -> {combined_csv}")

        # Overall summary
        rep_cols = [c for c in combined.columns if c.endswith("_total_counts")]
        print("\nOverall summary across all experiments:")
        overall = (
            combined.groupby(["experiment", "fail_reason"])
            .agg(
                count=("fail_reason", "size"),
                **{col: (col, "mean") for col in rep_cols},
            )
            .reset_index()
        )
        print(overall.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
