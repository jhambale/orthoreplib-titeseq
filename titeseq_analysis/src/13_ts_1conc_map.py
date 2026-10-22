#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2026 May 11
@author: jhambale

13_ts_1conc_map.py
==================
Maps single-concentration library scores (from library_dfs/) against
full Tite-Seq Kd estimates (from tite_seq_Kds2/passed_variants.csv).

Steps
-----
1. Load all CSVs from --library-dir.
2. For each CSV compute a normality score for replicate_1 and replicate_2
   using the Shapiro-Wilk W statistic.  The CSV whose *average* W across both
   replicates is closest to 1 (most normally distributed) is selected.
3. Save overlay histograms (rep1 & rep2) for every library CSV, annotated
   with quintile boundaries derived from the mean of (rep1, rep2).
4. Load --passed-variants CSV from tite_seq_Kds2.
5. Inner-join EACH library CSV to passed_variants on aa_sequence and run
   the mapping/comparison for every concentration.  Output files are named
   with the library stem as a suffix.  The most-normally distributed
   concentration additionally gets a '_mostnormal' tag in its file names.
6. For every matched variant compute:
      - Kd_avg          : mean(rep1_Kd, rep2_Kd)
      - norm_score_avg  : mean(replicate_1, replicate_2) from library CSV
      - Kd_quintile     : quintile rank based on Kd_avg   (1=lowest Kd / best)
      - score_quintile  : quintile rank based on norm_score_avg (1=lowest score)
7. Report quintile agreement: fraction of variants in the same quintile for
   both metrics (after inverting Kd so that high score ~ low Kd).
8. Save a scatter plot of Kd_avg vs norm_score_avg coloured by quintile agreement.
9. Write a results CSV with all mapped variants.

Usage
-----
python 13_ts_1conc_map.py \\
    --library-dir  ../../titeseq_data/outputs/jh_002/library_dfs \\
    --passed-variants ../../titeseq_data/outputs/jh_002/tite_seq_Kds/tite_seq_Kds_passed_variants.csv \\
    --output-dir   ../../titeseq_data/outputs/jh_002/ts_1conc_map
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_conc_nm(filename: str) -> float:
    """Extract the numeric nM concentration from a library CSV filename.

    Accepts formats like: '80nmS200T50.csv', '0.64nmS200T50.csv', '2000nmS200T100.csv'
    Returns the concentration as a float (nM), or np.nan if unparseable.
    """
    m = re.search(r"([\d.]+)\s*nm", filename, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def normality_score(series: pd.Series) -> float:
    """Return Shapiro-Wilk W statistic for a 1-D numeric series (higher = more normal).
    Returns np.nan if too few finite values.
    """
    vals = series.dropna().values.astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.nan
    w, _ = stats.shapiro(vals[:5000])  # shapiro capped at 5 000 samples
    return float(w)


def quintile_labels(series: pd.Series) -> pd.Series:
    """Assign quintile labels 1-5 to a series using its own quantiles.
    Ties are handled with the 'average' method inside pd.qcut.
    Label 1 = lowest values, 5 = highest.
    """
    return pd.qcut(series, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)


def quintile_boundaries(series: pd.Series) -> list[float]:
    """Return the 0, 20, 40, 60, 80, 100 percentile cut-points of a series."""
    vals = series.dropna()
    return [float(np.percentile(vals, p)) for p in [0, 20, 40, 60, 80, 100]]


# ---------------------------------------------------------------------------
# Step 1-2: Load library CSVs & pick most-normal concentration
# ---------------------------------------------------------------------------


def load_library_csvs(lib_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSVs in lib_dir, keyed by file stem."""
    dfs: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(lib_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "replicate_1" not in df.columns or "replicate_2" not in df.columns:
                print(f"  [SKIP] {csv_path.name}: missing replicate columns")
                continue
            dfs[csv_path.stem] = df
        except Exception as exc:
            print(f"  [WARN] Could not read {csv_path.name}: {exc}")
    return dfs


def select_best_library(lib_dfs: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Return (name, df) of the library CSV whose replicates are most normally
    distributed (highest mean Shapiro-Wilk W across rep1 and rep2)."""
    scores: dict[str, float] = {}
    for name, df in lib_dfs.items():
        w1 = normality_score(df["replicate_1"])
        w2 = normality_score(df["replicate_2"])
        ws = [w for w in [w1, w2] if not np.isnan(w)]
        scores[name] = float(np.mean(ws)) if ws else np.nan

    print("\nShapiro-Wilk W scores (mean of rep1 & rep2):")
    for name, w in sorted(
        scores.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999
    ):
        conc = parse_conc_nm(name)
        print(f"  {name:<28s}  conc={conc:>8.2f} nM   W={w:.6f}")

    best_name = max(
        scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else -999
    )
    print(
        f"\n>>> Best (most normal) concentration: {best_name}  (W={scores[best_name]:.6f})\n"
    )
    return best_name, lib_dfs[best_name]


# ---------------------------------------------------------------------------
# Step 3: Histogram plots for all library CSVs
# ---------------------------------------------------------------------------


def plot_library_histograms(
    lib_dfs: dict[str, pd.DataFrame],
    best_name: str,
    out_dir: Path,
) -> None:
    """For each library CSV save an overlaid rep1/rep2 histogram with quintile lines."""
    hist_dir = out_dir / "library_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for name, df in lib_dfs.items():
        r1 = df["replicate_1"].dropna().values
        r2 = df["replicate_2"].dropna().values
        avg = (df["replicate_1"] + df["replicate_2"]) / 2.0
        boundaries = quintile_boundaries(avg)

        conc = parse_conc_nm(name)
        w1 = normality_score(df["replicate_1"])
        w2 = normality_score(df["replicate_2"])

        fig, ax = plt.subplots(figsize=(8, 5))

        # Choose bins from combined range
        all_vals = np.concatenate([r1, r2])
        all_vals = all_vals[np.isfinite(all_vals)]
        bins = (
            np.linspace(all_vals.min(), all_vals.max(), 60) if len(all_vals) > 1 else 30
        )

        ax.hist(
            r1,
            bins=bins,
            alpha=0.55,
            color="steelblue",
            label=f"rep1  (W={w1:.4f})",
            density=True,
        )
        ax.hist(
            r2,
            bins=bins,
            alpha=0.55,
            color="tomato",
            label=f"rep2  (W={w2:.4f})",
            density=True,
        )

        # Quintile boundary lines (inner boundaries only: 20/40/60/80 %ile)
        quintile_colors = ["#6a994e", "#a7c957", "#ffd60a", "#f4a261", "#e63946"]
        for i, bnd in enumerate(boundaries):
            lw = 1.0 if i in (0, 5) else 1.5
            ls = "--" if i in (0, 5) else "-"
            alpha = 0.4 if i in (0, 5) else 0.85
            label_txt = None
            if 0 < i < 5:
                label_txt = f"Q{i}/Q{i + 1} ({bnd:.3f})"
            ax.axvline(
                bnd,
                color=quintile_colors[min(i, 4)],
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                label=label_txt,
            )

        star = "  ★ BEST" if name == best_name else ""
        ax.set_title(f"{name}  [{conc:.2f} nM]{star}", fontsize=12)
        ax.set_xlabel("Normalised score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        out_path = hist_dir / f"{name}_histogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved histogram -> {out_path.name}")


# ---------------------------------------------------------------------------
# Step 4-6: Load passed_variants and map
# ---------------------------------------------------------------------------


def load_passed_variants(csv_path: Path) -> pd.DataFrame:
    """Load tite_seq_Kds_passed_variants.csv."""
    df = pd.read_csv(csv_path)
    if "aa_sequence" not in df.columns:
        raise ValueError(
            f"passed_variants CSV missing 'aa_sequence' column: {csv_path}"
        )
    return df


def map_and_compare(
    lib_df: pd.DataFrame,
    lib_name: str,
    pv_df: pd.DataFrame,
    out_dir: Path,
    file_tag: str = "",
) -> pd.DataFrame:
    """
    Inner-join library df to passed_variants on aa_sequence.
    Compute Kd_avg, norm_score_avg, quintile labels, and agreement.
    Save scatter plot and results CSV, both suffixed with *file_tag*
    (e.g. the library stem or '_mostnormal').
    Returns the merged DataFrame.
    """
    # ------------------------------------------------------------------
    # Validate columns
    # ------------------------------------------------------------------
    for col in ["rep1_Kd", "rep2_Kd"]:
        if col not in pv_df.columns:
            raise ValueError(f"passed_variants CSV missing '{col}' column.")
    for col in ["replicate_1", "replicate_2"]:
        if col not in lib_df.columns:
            raise ValueError(f"library CSV missing '{col}' column.")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    lib_sub = lib_df[["aa_sequence", "replicate_1", "replicate_2"]].copy()
    pv_sub = pv_df[["aa_sequence", "rep1_Kd", "rep2_Kd"]].copy()

    merged = pd.merge(pv_sub, lib_sub, on="aa_sequence", how="inner")
    n_total_pv = len(pv_df)
    n_total_lib = len(lib_df)
    n_merged = len(merged)
    print(
        f"Mapping: {n_total_pv} passed variants  x  {n_total_lib} library variants  "
        f"->  {n_merged} matches on aa_sequence"
    )

    if n_merged == 0:
        print(
            "  [WARN] No matching sequences found.  Check that both files share sequences."
        )
        return merged

    # ------------------------------------------------------------------
    # Average scores
    # ------------------------------------------------------------------
    merged["Kd_avg"] = (merged["rep1_Kd"] + merged["rep2_Kd"]) / 2.0
    merged["norm_score_avg"] = (merged["replicate_1"] + merged["replicate_2"]) / 2.0

    # ------------------------------------------------------------------
    # Quintile ranks
    # ------------------------------------------------------------------
    # NOTE: For Kd, lower = better (tighter binding).
    #       For norm_score, higher = better (more active).
    # We invert Kd so that quintile rank 5 = lowest Kd (best) in both.
    merged["Kd_quintile"] = quintile_labels(merged["Kd_avg"])  # 1=lowest Kd
    merged["score_quintile"] = quintile_labels(
        merged["norm_score_avg"]
    )  # 1=lowest score

    # Invert Kd quintile so Q5=best for both metrics
    merged["Kd_quintile_inv"] = (
        6.0 - merged["Kd_quintile"]
    )  # 1=highest Kd (worst), 5=lowest Kd (best)

    # Agreement: same quintile when comparing inverted Kd to score quintile
    merged["quintile_agreement"] = merged["Kd_quintile_inv"] == merged["score_quintile"]

    pct_agree = merged["quintile_agreement"].mean() * 100
    print(
        f"\nQuintile agreement (same quintile in Kd_inv vs norm_score): "
        f"{pct_agree:.1f}% of {n_merged} matched variants"
    )

    # Per-quintile breakdown
    print("\nPer-quintile breakdown (norm_score quintile):")
    print(f"  {'Score Q':<10}{'# variants':<15}{'# agree':<12}{'% agree'}")
    for q in [1, 2, 3, 4, 5]:
        sub = merged[merged["score_quintile"] == q]
        n = len(sub)
        a = sub["quintile_agreement"].sum()
        print(f"  Q{q:<9}{n:<15}{a:<12}{(a / n * 100) if n > 0 else 0:.1f}%")

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_agree = {True: "#2a9d8f", False: "#e76f51"}
    for agree_val, grp in merged.groupby("quintile_agreement"):
        ax.scatter(
            grp["norm_score_avg"],
            np.log10(grp["Kd_avg"]),
            c=colors_agree[agree_val],
            alpha=0.55,
            s=18,
            linewidths=0,
            label=f"{'Same quintile' if agree_val else 'Different quintile'} (n={len(grp)})",
        )

    # Quintile boundary lines for norm_score_avg (x-axis)
    x_bounds = quintile_boundaries(merged["norm_score_avg"])
    y_bounds = quintile_boundaries(np.log10(merged["Kd_avg"]))
    q_colors = ["#6a994e", "#a7c957", "#ffd60a", "#f4a261", "#e63946"]
    for i, bnd in enumerate(x_bounds):
        if 0 < i < 5:
            ax.axvline(
                bnd,
                color=q_colors[i],
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                label=f"Score Q{i}/Q{i + 1}",
            )
    for i, bnd in enumerate(y_bounds):
        if 0 < i < 5:
            ax.axhline(bnd, color=q_colors[i], linewidth=1.2, linestyle=":", alpha=0.7)

    ax.set_xlabel("Norm score avg  (lib replicate mean)", fontsize=11)
    ax.set_ylabel("log10(Kd_avg)  [nM]", fontsize=11)
    ax.set_title(
        f"Tite-Seq Kd  vs  Library Score  ({lib_name})\n"
        f"Quintile agreement: {pct_agree:.1f}%  |  n={n_merged}",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    tag = f"_{file_tag}" if file_tag else ""
    scatter_path = out_dir / f"Kd_vs_normscore_scatter{tag}.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"\nScatter plot saved -> {scatter_path}")

    # ------------------------------------------------------------------
    # Results CSV
    # ------------------------------------------------------------------
    results_csv = out_dir / f"mapped_variants{tag}.csv"
    merged.to_csv(results_csv, index=False)
    print(f"Results CSV saved -> {results_csv}")

    return merged


# ---------------------------------------------------------------------------
# Kd histogram
# ---------------------------------------------------------------------------


def plot_kd_histogram(merged: pd.DataFrame, out_dir: Path) -> None:
    """Save an overlaid rep1_Kd / rep2_Kd histogram (log10 scale) with
    quintile boundary lines derived from Kd_avg, mirroring the library
    histogram style exactly.  Called once for the whole run (Kd data is
    independent of library concentration)."""
    r1 = np.log10(merged["rep1_Kd"].dropna().values)
    r2 = np.log10(merged["rep2_Kd"].dropna().values)
    avg_log = np.log10(merged["Kd_avg"].dropna())
    boundaries = quintile_boundaries(avg_log)

    w1 = normality_score(pd.Series(r1))
    w2 = normality_score(pd.Series(r2))

    fig, ax = plt.subplots(figsize=(8, 5))

    all_vals = np.concatenate([r1, r2])
    all_vals = all_vals[np.isfinite(all_vals)]
    bins = np.linspace(all_vals.min(), all_vals.max(), 60) if len(all_vals) > 1 else 30

    ax.hist(
        r1,
        bins=bins,
        alpha=0.55,
        color="steelblue",
        label=f"rep1_Kd  (W={w1:.4f})",
        density=True,
    )
    ax.hist(
        r2,
        bins=bins,
        alpha=0.55,
        color="tomato",
        label=f"rep2_Kd  (W={w2:.4f})",
        density=True,
    )

    quintile_colors = ["#6a994e", "#a7c957", "#ffd60a", "#f4a261", "#e63946"]
    for i, bnd in enumerate(boundaries):
        lw = 1.0 if i in (0, 5) else 1.5
        ls = "--" if i in (0, 5) else "-"
        alpha = 0.4 if i in (0, 5) else 0.85
        label_txt = None
        if 0 < i < 5:
            label_txt = f"Q{i}/Q{i + 1} ({10**bnd:.2f} nM)"
        ax.axvline(
            bnd,
            color=quintile_colors[min(i, 4)],
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
            label=label_txt,
        )

    ax.set_title("Tite-Seq Kd  —  rep1 & rep2 overlay", fontsize=12)
    ax.set_xlabel("log10(Kd)  [nM]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out_path = out_dir / "Kd_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Kd histogram saved -> {out_path}")


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------


def print_summary_table(merged: pd.DataFrame) -> None:
    """Print a cross-tabulation of Kd quintile (inverted) vs score quintile."""
    if merged.empty:
        return
    ct = pd.crosstab(
        merged["Kd_quintile_inv"].astype(int),
        merged["score_quintile"].astype(int),
        rownames=["Kd quintile (5=best)"],
        colnames=["Score quintile"],
    )
    print("\nCross-tabulation  Kd-quintile (rows)  vs  Score-quintile (cols):")
    print(ct.to_string())
    diag_frac = np.trace(ct.values) / ct.values.sum() * 100
    print(f"\n  Diagonal fraction (exact agreement): {diag_frac:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Select the most normally distributed single-concentration library "
            "and map its scores to Tite-Seq Kd estimates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--library-dir",
        "-l",
        required=True,
        metavar="DIR",
        help="Directory containing library_dfs CSVs (one per concentration).",
    )
    p.add_argument(
        "--passed-variants",
        "-p",
        required=True,
        metavar="CSV",
        help="Path to tite_seq_Kds2/tite_seq_Kds_passed_variants.csv.",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        required=True,
        metavar="DIR",
        help="Output directory (created if absent).",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    lib_dir = Path(args.library_dir)
    pv_csv = Path(args.passed_variants)
    out_dir = Path(args.output_dir)

    # Validate inputs
    if not lib_dir.is_dir():
        print(f"[ERROR] library-dir not found: {lib_dir}", file=sys.stderr)
        sys.exit(1)
    if not pv_csv.is_file():
        print(f"[ERROR] passed-variants CSV not found: {pv_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ------------------------------------------------------------------
    # 1-2. Load & rank library CSVs
    # ------------------------------------------------------------------
    print(f"\nLoading library CSVs from: {lib_dir}")
    lib_dfs = load_library_csvs(lib_dir)
    if not lib_dfs:
        print("[ERROR] No valid library CSVs found.", file=sys.stderr)
        sys.exit(1)

    best_name, best_df = select_best_library(lib_dfs)

    # ------------------------------------------------------------------
    # 3. Histograms for all concentrations
    # ------------------------------------------------------------------
    print("\nSaving library histograms...")
    plot_library_histograms(lib_dfs, best_name, out_dir)

    # ------------------------------------------------------------------
    # 4. Load passed_variants
    # ------------------------------------------------------------------
    print(f"\nLoading passed variants from: {pv_csv}")
    pv_df = load_passed_variants(pv_csv)
    print(f"  {len(pv_df)} passed variants loaded.")

    # ------------------------------------------------------------------
    # 5-7. Map & compare  (all concentrations)
    # ------------------------------------------------------------------
    print(f"\nRunning Kd mapping for all {len(lib_dfs)} library concentrations...")
    best_merged: pd.DataFrame = pd.DataFrame()
    for lib_name, lib_df in lib_dfs.items():
        print(f"\n--- {lib_name} ---")
        is_best = lib_name == best_name
        # Most-normal concentration gets the _mostnormal tag; all others use the lib stem
        file_tag = f"{lib_name}_mostnormal" if is_best else lib_name
        merged = map_and_compare(lib_df, lib_name, pv_df, out_dir, file_tag=file_tag)
        if is_best and not merged.empty:
            best_merged = merged

    # ------------------------------------------------------------------
    # 8. Kd histogram  (single, concentration-independent)
    # ------------------------------------------------------------------
    if not best_merged.empty:
        plot_kd_histogram(best_merged, out_dir)

    # ------------------------------------------------------------------
    # 9. Summary table (most-normal concentration)
    # ------------------------------------------------------------------
    print(f"\n=== Summary table for most-normal concentration ({best_name}) ===")
    print_summary_table(best_merged)

    print("\nDone.")


if __name__ == "__main__":
    main()
