#!/usr/bin/env python3

"""
Created on January 22 11:30:29 2025
@author: alcantar; modified by yy and JH
example run: python 10_compute_Kds.py -i ../../data/outputs/jh_001/normalized_tite_seq_scores/ -p 32
Fixed 06_compute_Kds.py:
- Resolves worker argument mismatches.
- Applies consistent plotting defaults and correlation outputs.
"""

import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
from tqdm import tqdm
from utils_ts import make_dir

# --- Core Math & Fitting ---


def func_E50(concs: np.ndarray, Kd: float, As: float, B: float) -> np.ndarray:
    """Standard Hill function returning log10(fluorescence)."""
    fluorescence = ((concs) / (concs + Kd)) * As + B
    return np.log10(np.maximum(fluorescence, 1e-9))


def try_fit(concs, scores, bounds):
    """Encapsulates curve_fit with error handling and initial guessing.

    Returns (popt, pcov) on success, or (None, None) on failure.
    pcov is the parameter covariance matrix from curve_fit; its diagonal
    gives the variance of each fitted parameter, so sqrt(pcov[0,0]) is
    the 1-sigma uncertainty on Kd.
    """
    mask = np.isfinite(scores)
    if not mask.any() or len(scores[mask]) < 3:
        return None, None

    lin_scores = 10 ** scores[mask]
    p0 = [
        np.median(concs),
        max(lin_scores.max() - lin_scores.min(), 1e-3),
        max(lin_scores.min(), 1e-3),
    ]
    p0 = np.clip(p0, bounds[0], bounds[1])

    try:
        popt, pcov = curve_fit(
            func_E50, concs[mask], scores[mask], p0=p0, bounds=bounds, maxfev=10000
        )
        return popt, pcov
    except Exception:
        return None, None


def fitted_midpoint_concentration(
    measured_scores: np.ndarray, fit_concs: np.ndarray, popt: np.ndarray
) -> tuple[float, float]:
    """Find where the fitted curve reaches the measured log-fluorescence midpoint."""
    finite_scores = measured_scores[np.isfinite(measured_scores)]
    if finite_scores.size == 0:
        return np.nan, np.nan

    midpoint_score = float((finite_scores.min() + finite_scores.max()) / 2)
    fitted_scores = func_E50(fit_concs, *popt)
    if not np.nanmin(fitted_scores) <= midpoint_score <= np.nanmax(fitted_scores):
        return np.nan, midpoint_score

    kd, amplitude, background = popt
    midpoint_fluorescence = 10**midpoint_score
    response_above_background = midpoint_fluorescence - background
    denominator = amplitude - response_above_background

    if (
        not np.isfinite(midpoint_fluorescence)
        or amplitude <= 0
        or response_above_background <= 0
        or denominator <= 0
    ):
        return np.nan, midpoint_score

    midpoint_conc = kd * response_above_background / denominator
    if not np.min(fit_concs) <= midpoint_conc <= np.max(fit_concs):
        return np.nan, midpoint_score
    return float(midpoint_conc), midpoint_score


# --- Worker Function ---

# --- Worker Function ---


def fit_variant_worker(task_args) -> Dict:
    """Worker function to process a single variant across replicates."""
    (
        variant_data,
        concs_array,
        bounds,
        conc_fits,
        plot_dir,
        y_limits,
        conc_labels,
        rep_ids,
        r2_cutoff,
        boundary_anchors,
    ) = task_args

    v_id = variant_data["variant_id"]
    safe_id = variant_data["safe_id"]
    aa_seq = variant_data["aa_sequence"]

    res = {
        "variant_id": v_id,
        "aa_sequence": aa_seq,
        "passed": False,
        "fail_reason": None,
    }

    all_scores = []
    all_fits = []
    ec50_points = []
    rep_fail_reasons = []

    # Thresholds
    variation_threshold = (
        0.5  # minimum log10 score range (~3-fold); below this is low_amplitude
    )
    for rep in rep_ids:
        scores = variant_data[f"{rep}_scores"]
        all_scores.append(scores)

        scores_for_fit = scores.copy()
        concs_for_fit = concs_array.copy()

        # Augment with imaginary boundary anchors where the variant qualifies
        if boundary_anchors is not None:
            anchors = boundary_anchors["rep_anchors"][rep]
            if scores_for_fit[0] <= anchors["low_thresh"]:
                concs_for_fit = np.concatenate(
                    [[boundary_anchors["low_conc"]], concs_for_fit]
                )
                scores_for_fit = np.concatenate(
                    [[anchors["low_score"]], scores_for_fit]
                )
                res[f"{rep}_low_anchor_applied"] = True
            else:
                res[f"{rep}_low_anchor_applied"] = False
        finite_mask = np.isfinite(scores_for_fit)
        popt, pcov = (
            try_fit(concs_for_fit, scores_for_fit, bounds)
            if finite_mask.any()
            else (None, None)
        )

        rep_fail = None
        if popt is None:
            rep_fail = "no_fit"
            res.update(
                {
                    f"{rep}_Kd": np.nan,
                    f"{rep}_Kd_std": np.nan,
                    f"{rep}_As": np.nan,
                    f"{rep}_B": np.nan,
                    f"{rep}_r2": np.nan,
                    f"{rep}_adj_r2": np.nan,
                }
            )
            all_fits.append(np.full_like(conc_fits, np.nan))
            ec50_points.append((np.nan, np.nan))
            res[f"{rep}_EC50_midpoint_nM"] = np.nan
            res[f"{rep}_midpoint_log10_fluorescence"] = np.nan
        else:
            kd, as_val, b_val = popt
            kd_std = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan
            ec50_midpoint, midpoint_score = fitted_midpoint_concentration(
                scores, conc_fits, popt
            )
            res.update(
                {
                    f"{rep}_Kd": kd,
                    f"{rep}_Kd_std": kd_std,
                    f"{rep}_As": as_val,
                    f"{rep}_B": b_val,
                    f"{rep}_EC50_midpoint_nM": ec50_midpoint,
                    f"{rep}_midpoint_log10_fluorescence": midpoint_score,
                }
            )
            all_fits.append(func_E50(conc_fits, *popt))
            ec50_points.append((ec50_midpoint, midpoint_score))

            # R² and adjusted R² computed on the same data points used for fitting
            _concs_used = concs_for_fit[finite_mask]
            _scores_used = scores_for_fit[finite_mask]
            _y_pred = func_E50(_concs_used, *popt)
            _rss = np.sum((_scores_used - _y_pred) ** 2)
            _tss = np.sum((_scores_used - np.mean(_scores_used)) ** 2)
            _r2 = 1 - (_rss / _tss) if _tss > 0 else np.nan
            _n = len(_concs_used)
            _p = len(popt)  # 3 parameters: Kd, As, B
            _adj_r2 = (
                1 - ((1 - _r2) * (_n - 1) / (_n - _p - 1))
                if (_n > _p + 1 and np.isfinite(_r2))
                else np.nan
            )
            res[f"{rep}_r2"] = _r2
            res[f"{rep}_adj_r2"] = _adj_r2

            dynamic_range = np.nanmax(scores) - np.nanmin(scores)

            # Determine whether the curve is oriented correctly: peak
            # fluorescence should appear at a higher concentration than the
            # trough (rising sigmoidal).  If the concentration at the max is
            # lower than the concentration at the min the curve is inverted or
            # sufficiently noisy to be untrustworthy.
            finite_score_mask = np.isfinite(scores)
            if finite_score_mask.any():
                concs_finite = concs_array[finite_score_mask]
                scores_finite = scores[finite_score_mask]
                conc_at_max = concs_finite[np.argmax(scores_finite)]
                conc_at_min = concs_finite[np.argmin(scores_finite)]
                curve_inverted = conc_at_max < conc_at_min
            else:
                curve_inverted = False

            # Failure hierarchy (evaluated in order; first match wins):
            #
            # 1. low_amplitude: fires if either —
            #    a) the observed fluorescence rise is below variation_threshold
            #       (curve too flat to trust the fitted response midpoint), or
            #    b) the curve is inverted: the concentration at peak fluorescence
            #       is lower than the concentration at minimum fluorescence,
            #       indicating a non-sigmoidal or noisy titration.
            #
            if dynamic_range < variation_threshold or curve_inverted:
                rep_fail = "low_amplitude"
            elif not np.isfinite(_r2) or _r2 < r2_cutoff:
                rep_fail = "low_r2"

        rep_fail_reasons.append(rep_fail)

    # Global Filter: Variant passes only if ALL replicates pass all criteria
    if all(r is None for r in rep_fail_reasons) and len(rep_fail_reasons) > 0:
        res["passed"] = True
    else:
        # Report the highest-priority failure reason across replicates
        priority = [
            "low_amplitude",
            "low_r2",
            "no_fit",
        ]
        for reason in priority:
            if reason in rep_fail_reasons:
                res["fail_reason"] = reason
                break

    if plot_dir:
        if res["passed"]:
            subfolder = "passed"
        elif res["fail_reason"] == "low_amplitude":
            subfolder = "failed/low_amplitude"
        elif res["fail_reason"] == "low_r2":
            subfolder = "failed/low_r2"
        else:
            subfolder = "failed/other"
        target_dir = plot_dir / subfolder
        # Note: Directories are pre-created in main()

        plot_variant(
            concs_numeric=concs_array,
            raw_scores=all_scores,
            fit_concs=conc_fits,
            fit_scores=all_fits,
            ec50_points=ec50_points,
            title=v_id,
            path=target_dir / f"{safe_id}.png",
            y_min=y_limits[0],
            y_max=y_limits[1],
            labels=conc_labels,
            rep_ids=rep_ids,
        )

    return res


# --- Helpers ---


def plot_variant(
    concs_numeric,
    raw_scores,
    fit_concs,
    fit_scores,
    ec50_points,
    title,
    path,
    y_min,
    y_max,
    labels,
    rep_ids,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"rep2": "#DE9191", "rep3": "#AF3E3E"}
    default_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(rep_ids)))

    for i, rep in enumerate(rep_ids):
        color = palette.get(rep, default_colors[i])
        label = (
            f"Replicate {rep[-1]}"
            if rep.startswith("rep") and rep[3:].isdigit()
            else rep
        )
        ax.plot(
            concs_numeric,
            raw_scores[i],
            linestyle="none",
            marker="o",
            markersize=8,
            markeredgecolor="black",
            color=color,
            label=label,
            zorder=3,
        )
        if not np.all(np.isnan(fit_scores[i])):
            ax.plot(
                fit_concs,
                fit_scores[i],
                linestyle="-",
                linewidth=3,
                color=color,
                zorder=2,
            )
        ec50_conc, ec50_score = ec50_points[i]
        if np.isfinite(ec50_conc) and np.isfinite(ec50_score):
            ax.plot(
                ec50_conc,
                ec50_score,
                linestyle="none",
                marker="D",
                markersize=8,
                markeredgecolor="black",
                color=color,
                label=f"{label} midpoint EC50",
                zorder=4,
            )

    ax.set_xscale("log")
    plotted_ec50s = [
        ec50_conc
        for ec50_conc, ec50_score in ec50_points
        if np.isfinite(ec50_conc) and np.isfinite(ec50_score)
    ]
    x_limits = np.concatenate([concs_numeric, plotted_ec50s])
    ax.set_xlim(x_limits.min() * 0.8, x_limits.max() * 1.2)
    ax.xaxis.set_major_locator(mticker.FixedLocator(concs_numeric))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim([y_min, y_max])
    ax.set_xticks(concs_numeric)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Variant: {title}")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("log10(Fluorescence)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="upper left", frameon=True)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_global_correlations(
    results_df: pd.DataFrame,
    rep_ids: List[str],
    output_path: Path,
    label: str = "Variants",
):
    """Generates a density-colored scatter plot of log10(EC50) between replicates.

    Parameters
    ----------
    results_df : DataFrame
        Rows to plot — caller is responsible for subsetting before passing in.
    rep_ids : list of str
        Replicate identifiers; only the first two are used.
    output_path : Path
        Destination file (.png appended if no suffix).
    label : str
        Human-readable description shown inside the plot (e.g. "All Variants").
    """
    if len(rep_ids) < 2:
        return

    r1, r2 = rep_ids[0], rep_ids[1]
    subset = results_df.copy()

    ec50_col_1 = f"{r1}_EC50_midpoint_nM"
    ec50_col_2 = f"{r2}_EC50_midpoint_nM"
    finite_mask = np.isfinite(subset[ec50_col_1]) & np.isfinite(
        subset[ec50_col_2]
    )
    subset = subset[finite_mask]
    if subset.empty:
        print(
            f"No variants with valid midpoint EC50 values to plot for subset: "
            f"'{label}'."
        )
        return

    x = np.log10(subset[ec50_col_1].values)
    y = np.log10(subset[ec50_col_2].values)

    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Symmetric axis range derived from both replicates together
    all_vals = np.concatenate([x, y])
    ax_min = np.floor(all_vals.min()) - 0.2
    ax_max = np.ceil(all_vals.max()) + 0.2

    fig, ax = plt.subplots(figsize=(8, 7))

    # y=x identity line behind the scatter
    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        linestyle="--",
        linewidth=0.8,
        color="black",
        zorder=1,
    )

    scatter = ax.scatter(
        x, y, c=z, s=15, cmap="magma", edgecolor=None, alpha=0.8, zorder=2
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Point Density", rotation=270, labelpad=15)

    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"{r1} $\\log_{{10}}(EC_{{50}})$ (nM)", size=14)
    ax.set_ylabel(f"{r2} $\\log_{{10}}(EC_{{50}})$ (nM)", size=14)

    p_corr, _ = stats.pearsonr(x, y)
    s_corr, _ = stats.spearmanr(x, y)

    textstr = (
        f"{label} ($N={len(subset)}$)\n"
        f"Pearson $R = {p_corr:.3f}$\n"
        f"Spearman $R = {s_corr:.3f}$"
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    outfile = output_path if output_path.suffix else output_path.with_suffix(".png")
    plt.savefig(outfile, dpi=300)
    plt.close()


def compute_boundary_anchors(
    merged_df: pd.DataFrame,
    shared_labels: List[str],
    concs_array: np.ndarray,
    rep_ids: List[str],
    output_dir: Path,
    proximity_fraction: float = 0.90,
) -> Dict:
    """Compute an imaginary low-concentration anchor for curve-fit extension.

    The median scores at the lowest and highest tested concentrations define
    the low-end fluorescence floor and dynamic range. A variant qualifies for
    an anchor if its score at the lowest concentration remains near that floor.
    The typical fold interval is inferred from the measured concentrations, and
    the imaginary concentration is placed two intervals below the lowest one.

    Also saves a CSV and low-boundary histogram PNG into output_dir.
    """
    low_label = shared_labels[0]
    high_label = shared_labels[-1]
    fold_interval = detect_concentration_interval(concs_array)
    low_conc_imaginary = concs_array.min() / (fold_interval**2)

    rep_anchors = {}
    rows = []

    fig_low, ax_low = plt.subplots(figsize=(7, 4))
    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(rep_ids)))

    for i, rep in enumerate(rep_ids):
        low_col = f"{rep}_score_{low_label}"
        high_col = f"{rep}_score_{high_label}"
        low_scores = merged_df[low_col].dropna().values
        high_scores = merged_df[high_col].dropna().values

        low_median = float(np.median(low_scores))
        high_median = float(np.median(high_scores))
        total_range = high_median - low_median

        low_thresh = low_median + (1.0 - proximity_fraction) * total_range

        n_low = int((low_scores <= low_thresh).sum())

        rep_anchors[rep] = {
            "low_score": low_median,
            "low_thresh": low_thresh,
        }
        rows.append(
            {
                "replicate": rep,
                "low_boundary_conc_nM": concs_array.min(),
                "high_boundary_conc_nM": concs_array.max(),
                "low_boundary_score": low_median,
                "high_boundary_score": high_median,
                "low_proximity_thresh": low_thresh,
                "n_variants_near_low": n_low,
                "detected_fold_interval": fold_interval,
                "imaginary_low_conc_nM": low_conc_imaginary,
            }
        )

        color = palette[i]
        ax_low.hist(
            low_scores, bins=40, alpha=0.6, color=color, label=rep, edgecolor="none"
        )
        ax_low.axvline(
            low_median, color=color, linestyle="-", linewidth=1.5, label=f"{rep} median"
        )
        ax_low.axvline(
            low_thresh,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"{rep} 90% thresh",
        )

    ax_low.set_xlabel("log\u2081\u2080 fluorescence score", fontsize=12)
    ax_low.set_ylabel("Variant count", fontsize=12)
    ax_low.set_title(
        f"Boundary score distribution — {shared_labels[0]} (low boundary)",
        fontsize=13,
    )
    ax_low.spines["top"].set_visible(False)
    ax_low.spines["right"].set_visible(False)
    handles, labels_leg = ax_low.get_legend_handles_labels()
    ax_low.legend(handles, labels_leg, fontsize=9, frameon=True)
    fig_low.tight_layout()
    fig_low.savefig(output_dir / "boundary_scores_low.png", dpi=200)
    plt.close(fig_low)

    # Save summary CSV
    pd.DataFrame(rows).to_csv(output_dir / "boundary_anchor_stats.csv", index=False)

    print("-" * 30)
    print("Boundary anchor summary:")
    for r in rows:
        print(
            f"  {r['replicate']}: "
            f"low_median={r['low_boundary_score']:.4f} "
            f"(thresh<={r['low_proximity_thresh']:.4f}, n={r['n_variants_near_low']}), "
            f"high_median={r['high_boundary_score']:.4f}"
        )
    print(f"  Detected concentration interval: {fold_interval:.4g}-fold")
    print(f"  Imaginary low concentration: {low_conc_imaginary:.4g} nM")
    print("-" * 30)

    return {
        "low_conc": low_conc_imaginary,
        "rep_anchors": rep_anchors,
    }


def sanitize_filename(name: str) -> str:
    """Convert an aa_mutations string like "['A23T', 'V99I']" into a
    filesystem-safe name like 'A23T_V99I'.
    Strips brackets and quotes, replaces commas+spaces with underscores,
    and collapses any remaining whitespace.
    """
    name = name.replace("[", "").replace("]", "").replace("'", "")
    name = re.sub(r",\s*", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name or "parent"


def concentration_sort_key(label: str) -> float:
    """Sort concentration labels in ascending numeric order.

    Handles two formats:
      - XnM / X.YnM  (e.g. '0.64nM', '3.2nM', '16nM')  -> sort by nM value
      - 10^-NM       (e.g. '10^-7M')                     -> convert to nM
    """
    cleaned = label.replace(" ", "")
    # XnM format
    nm_match = re.match(r"^([\d.]+)nM$", cleaned, re.IGNORECASE)
    if nm_match:
        return float(nm_match.group(1))
    # 10^-N M format
    sci_match = re.search(r"10\^-(?P<val>[-\d\.]+)", cleaned)
    if sci_match:
        return 10 ** (-float(sci_match.group("val"))) * 1e9
    raise ValueError(f"Cannot parse concentration label: '{label}'")


def detect_concentration_interval(concs_array: np.ndarray) -> float:
    """Infer the typical fold spacing from positive measured concentrations."""
    values = np.asarray(concs_array, dtype=float)
    concentrations = np.unique(values[np.isfinite(values) & (values > 0)])
    if concentrations.size < 2:
        raise ValueError(
            "Boundary augmentation requires at least two distinct positive "
            "concentrations to infer the concentration interval."
        )

    log_intervals = np.diff(np.log(concentrations))
    return float(np.exp(np.median(log_intervals)))


# --- Main Logic ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Path to normalized scores directory"
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=mp.cpu_count(), help="CPU cores to use"
    )
    parser.add_argument(
        "--r2-cutoff",
        type=float,
        default=0.90,
        help="Minimum per-replicate R-squared required for a fit to pass "
        "(default: 0.90)",
    )
    parser.add_argument(
        "-b",
        "--boundary-augmentation",
        action="store_true",
        default=False,
        help="If set, augment fitting data with an imaginary low-concentration "
        "anchor for variants near the fluorescence floor. "
        "If omitted, variants are fit to the measured concentrations only.",
    )
    args = parser.parse_args()

    if not 0 <= args.r2_cutoff <= 1:
        parser.error("--r2-cutoff must be between 0 and 1.")
    r2_cutoff = args.r2_cutoff
    print(f"Fit filters: low amplitude; per-replicate R² >= {r2_cutoff:.2f}")
    lower_bound = args.boundary_augmentation

    input_dir = Path(args.input)
    # 1. Output directory one level up
    if lower_bound:
        output_dir = input_dir.parent / "tite_seq_Kds_leftaugment"
    else:
        output_dir = input_dir.parent / "tite_seq_Kds"
    plot_dir = output_dir / "score_vs_concs_plots"

    # Pre-create directories to avoid race conditions in multiprocessing
    make_dir(str(plot_dir))
    make_dir(str(plot_dir / "passed"))
    make_dir(str(plot_dir / "failed" / "low_amplitude"))
    make_dir(str(plot_dir / "failed" / "low_r2"))
    make_dir(str(plot_dir / "failed" / "other"))
    if args.boundary_augmentation:
        make_dir(str(output_dir / "boundary_anchors"))

    # 2. Discover all CSV files and identify replicates
    csv_files = sorted(input_dir.glob("*.csv"))
    replicate_dfs = {}
    for f in csv_files:
        # Extract repID (e.g., 'rep2' or 'rep3') from filename
        match = re.search(r"rep(\d+)", f.name.lower())
        rep_id = f"rep{match.group(1)}" if match else f.stem
        df = pd.read_csv(f)
        # if rep_id == "rep2":
        #     # Drop problematic 10^-7 concentration for replicate 2 only
        #     dropped = [c for c in df.columns if c.strip() == "score_10^-7M"]
        #     if dropped:
        #         print(f"Removing 10^-7 concentration from {rep_id}: {dropped}")
        #         df = df.drop(columns=dropped)
        replicate_dfs[rep_id] = df

    rep_ids = sorted(replicate_dfs.keys())
    print(f"Found replicates: {', '.join(rep_ids)}")
    if len(rep_ids) < 2:
        raise ValueError(
            "At least two replicate input files are required to fit shared sequences."
        )

    # Fit only variants present in every replicate input.
    sequence_sets = [
        set(df["aa_sequence"].dropna().astype(str)) for df in replicate_dfs.values()
    ]
    shared_sequences = set.intersection(*sequence_sets)
    if not shared_sequences:
        raise ValueError("No amino-acid sequences are shared across replicate files.")
    print(
        f"Sequences present in every replicate: {len(shared_sequences)} "
        f"(of {len(set.union(*sequence_sets))} unique sequences)"
    )
    for rep in rep_ids:
        df = replicate_dfs[rep]
        before = len(df)
        replicate_dfs[rep] = df[df["aa_sequence"].astype(str).isin(shared_sequences)].copy()
        print(
            f"  {rep}: excluded {before - len(replicate_dfs[rep])} rows not shared "
            "across replicates"
        )

    # 3. Merge the shared variants across replicates.
    metadata_cols = ["aa_sequence", "aa_mutations"]
    merged_df = None
    for rep in rep_ids:
        df = replicate_dfs[rep]
        score_cols = [c for c in df.columns if c.startswith("score_")]
        df = df.rename(columns={c: f"{rep}_{c}" for c in score_cols})

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on=metadata_cols, how="inner")

    # Identify shared concentration labels across replicates (exclude only the true 0M baseline)
    label_sets = []
    for rep in rep_ids:
        rep_prefix = f"{rep}_score_"
        labels = []
        for c in merged_df.columns:
            if not c.startswith(rep_prefix):
                continue
            lbl = c.replace(rep_prefix, "")
            if lbl.strip() == "0M":
                continue
            labels.append(lbl)
        label_sets.append(set(labels))

    shared_labels = sorted(set.intersection(*label_sets), key=concentration_sort_key)
    print(f"Shared concentration labels (excluding 0M): {shared_labels}")

    # Fixed y-axis scaling for variant plots
    y_limits = (1.5, 4.5)
    print("-" * 30)
    print(f"Using fixed y-axis range: {y_limits[0]} to {y_limits[1]}")
    print("-" * 30)
    # 4. Prepare fitting design
    concs_array = np.array([concentration_sort_key(lbl) for lbl in shared_labels])

    # 4b. Optionally compute boundary anchors (saves CSV + histograms)
    if args.boundary_augmentation:
        print("Boundary augmentation: ENABLED")
        boundary_anchors = compute_boundary_anchors(
            merged_df=merged_df,
            shared_labels=shared_labels,
            concs_array=concs_array,
            rep_ids=rep_ids,
            output_dir=output_dir / "boundary_anchors",
        )
        # Extend the plotted fit curve below the lowest measured concentration.
        conc_fits = np.logspace(
            np.log10(boundary_anchors["low_conc"]),
            np.log10(concs_array.max()),
            300,
        )
    else:
        print(
            "Boundary augmentation: DISABLED (fitting to measured concentrations only)"
        )
        boundary_anchors = None
        conc_fits = np.logspace(
            np.log10(concs_array.min()),
            np.log10(concs_array.max()),
            300,
        )
    # Bounds: [Kd (nM), Amplitude, Background]
    bounds = ([1e-5, 0, 0], [1e5, 1e7, 1e6])

    tasks = []
    for _, row in merged_df.iterrows():
        aa_muts = row["aa_mutations"]
        v_id = (
            "parent"
            if (not aa_muts or aa_muts == "[]" or pd.isna(aa_muts))
            else str(aa_muts)
        )
        safe_id = sanitize_filename(v_id)
        task_data = {
            "variant_id": v_id,
            "safe_id": safe_id,
            "aa_sequence": row["aa_sequence"],
        }

        for rep in rep_ids:
            scores = row[[f"{rep}_score_{l}" for l in shared_labels]].values.astype(
                float
            )
            task_data[f"{rep}_scores"] = scores

        tasks.append(
            (
                task_data,
                concs_array,
                bounds,
                conc_fits,
                plot_dir,
                y_limits,
                shared_labels,
                rep_ids,
                r2_cutoff,
                boundary_anchors,
            )
        )

    # 5. Multiprocessing with Progress Bar
    print(f"Fitting {len(tasks)} variants using {args.processes} processes...")
    results = []
    with mp.Pool(args.processes) as pool:
        for res in tqdm(
            pool.imap_unordered(fit_variant_worker, tasks),
            total=len(tasks),
            desc="Calculating Kds",
        ):
            results.append(res)

    # 6. Export Results and Correlation Plot
    final_df = pd.DataFrame(results).sort_values("variant_id")
    passed_df = final_df[final_df["passed"] == True].copy()
    low_amp_df = final_df[final_df["fail_reason"] == "low_amplitude"].copy()

    failure_order = ["low_amplitude", "low_r2", "no_fit"]
    failed_df = final_df[final_df["passed"] == False]
    fail_counts = failed_df["fail_reason"].value_counts()
    print("-" * 30)
    print(f"Total variants: {len(final_df)}")
    print(f"  Passed: {len(passed_df)}")
    for reason in failure_order:
        count = fail_counts.get(reason, 0)
        print(f"  Failed ({reason}): {count}")
    print("-" * 30)

    final_df.to_csv(output_dir / "tite_seq_Kds_all_variants.csv", index=False)
    passed_df.to_csv(output_dir / "tite_seq_Kds_passed_variants.csv", index=False)
    low_amp_df.to_csv(
        output_dir / "tite_seq_Kds_low_amplitude_variants.csv", index=False
    )

    # ------------------------------------------------------------------ #
    # Correlation plots — progressively filtered subsets                 #
    # ------------------------------------------------------------------ #
    # Subset 2: remove low_amplitude failures
    no_low_amp_df = final_df[final_df["fail_reason"] != "low_amplitude"].copy()

    correlation_subsets = [
        (final_df, "All Variants", "EC50_correlation_1_all.png"),
        (
            no_low_amp_df,
            "Without Low-Amplitude Failures",
            "EC50_correlation_2_no_low_amplitude.png",
        ),
        (
            passed_df,
            "Passed Variants",
            "EC50_correlation_3_passed.png",
        ),
    ]

    for subset_df, subset_label, filename in correlation_subsets:
        print(f"Generating correlation plot: {subset_label} (N={len(subset_df)})...")
        plot_global_correlations(
            subset_df, rep_ids, output_dir / filename, label=subset_label
        )

    # ------------------------------------------------------------------ #
    # R² / adjusted-R² fit-quality summary                                #
    # ------------------------------------------------------------------ #
    thresholds = sorted({0.60, 0.75, args.r2_cutoff, 0.95})
    r2_cols = [f"{rep}_r2" for rep in rep_ids]
    adj_r2_cols = [f"{rep}_adj_r2" for rep in rep_ids]

    for summary_label, df in [
        ("All variants", final_df),
        ("Passed variants", passed_df),
    ]:
        n_total = len(df)
        print("=" * 60)
        print(f"Fit quality — {summary_label} (N={n_total})")
        print("=" * 60)

        for rep in rep_ids:
            r2_vals = df[f"{rep}_r2"].dropna()
            adj_r2_vals = df[f"{rep}_adj_r2"].dropna()
            n_fit = len(r2_vals)
            if n_fit == 0:
                print(f"  {rep}: no valid fits")
                continue
            thresh_r2 = "  ".join(
                f">={t:.2f}: {int((r2_vals >= t).sum())}/{n_fit}" for t in thresholds
            )
            thresh_adj = "  ".join(
                f">={t:.2f}: {int((adj_r2_vals >= t).sum())}/{n_fit}"
                for t in thresholds
            )
            print(
                f"  {rep}  R²     mean={r2_vals.mean():.3f}  median={r2_vals.median():.3f}  {thresh_r2}"
            )
            print(
                f"  {rep}  adj-R² mean={adj_r2_vals.mean():.3f}  median={adj_r2_vals.median():.3f}  {thresh_adj}"
            )

        # Mean R² across all reps per variant (only when >1 replicate)
        if len(rep_ids) > 1:
            mean_r2 = df[r2_cols].mean(axis=1).dropna()
            mean_adj_r2 = df[adj_r2_cols].mean(axis=1).dropna()
            n_fit = len(mean_r2)
            thresh_r2 = "  ".join(
                f">={t:.2f}: {int((mean_r2 >= t).sum())}/{n_fit}" for t in thresholds
            )
            thresh_adj = "  ".join(
                f">={t:.2f}: {int((mean_adj_r2 >= t).sum())}/{n_fit}"
                for t in thresholds
            )
            print(
                f"  mean(reps)  R²     mean={mean_r2.mean():.3f}  median={mean_r2.median():.3f}  {thresh_r2}"
            )
            print(
                f"  mean(reps)  adj-R² mean={mean_adj_r2.mean():.3f}  median={mean_adj_r2.median():.3f}  {thresh_adj}"
            )

    print("=" * 60)

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
