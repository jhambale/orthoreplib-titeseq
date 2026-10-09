#!/usr/bin/env python3

"""Calculate and plot thresholds for one or more pooled rank-abundance files.

Example:
    python 04b_pooled_seqthresh.py \
        -i ../../titeseq_data/outputs/jh_010/pooled_reads/rank_abundance*_sums.txt \
        --method knee

note for the min_count threshold option the following math is used to determine the threshold: min_count = max(10, total_reads * 0.00001). \
The greater value of either 10 reads or 0.001% of the total reads is used as the threshold for min_count. \
"""

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CUMULATIVE_CUTOFFS = (0.90, 0.95, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Determine and plot thresholds for pooled rank-abundance files"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        help="One or more rank-abundance *_sums.txt files (shell wildcards are supported)",
    )
    parser.add_argument(
        "--method",
        choices=["knee_cumulative", "knee_rankabundance", "min_count", "cumulative"],
        default="knee_cumulative",
        help="Threshold method to mark as selected in each report and plot (default: knee_cumulative)",
    )
    parser.add_argument(
        "--cumulative-cutoff",
        type=float,
        choices=CUMULATIVE_CUTOFFS,
        default=0.95,
        help="Cumulative coverage threshold used when --method=cumulative (default: 0.95)",
    )
    return parser.parse_args()


def resolve_input_files(inputs):
    files = []
    for input_pattern in inputs:
        matches = glob.glob(input_pattern)
        files.extend(Path(match).resolve() for match in matches)

    unique_files = sorted(set(files))
    if not unique_files:
        raise ValueError("No input files matched the supplied paths or patterns.")
    if any(not path.is_file() for path in unique_files):
        missing = [str(path) for path in unique_files if not path.is_file()]
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))

    input_dirs = {path.parent for path in unique_files}
    if len(input_dirs) != 1 or next(iter(input_dirs)).name != "pooled_reads":
        raise ValueError("All input files must be in the same pooled_reads directory.")
    return unique_files


def load_rank_abundance(input_file):
    df = pd.read_csv(input_file, sep="\t")
    required_columns = {"rank", "count", "sequence"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{input_file} is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    if df.empty:
        raise ValueError(f"{input_file} contains no rank-abundance records.")

    for column in ("rank", "count"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    if (df["rank"] <= 0).any() or (df["count"] <= 0).any():
        raise ValueError(f"{input_file} must have positive ranks and read counts.")

    df = df.sort_values("rank").reset_index(drop=True)
    total_reads = df["count"].sum()
    df["cumulative_frac"] = df["count"].cumsum() / total_reads
    return df


def find_knee_cumulative(df):
    """Find knee on cumulative coverage curve (10^3 to 10^4 rank range).

    Uses polynomial smoothing to find steepest concave-down section, indicating
    transition from abundant to rare variants.
    """
    log_rank = np.log10(df["rank"].to_numpy(dtype=float))
    cumfrac = df["cumulative_frac"].to_numpy(dtype=float)

    # Focus on the 10^3 to 10^4 rank range where the knee typically occurs
    mask = (log_rank >= 3.0) & (log_rank <= 4.0)
    if mask.sum() < 10:
        return 0

    x_region = log_rank[mask]
    y_region = cumfrac[mask]

    try:
        p = np.polyfit(x_region, y_region, 5)
        pp = np.polyder(p, 2)  # 2nd derivative polynomial
    except np.linalg.LinAlgError:
        return 0

    # Evaluate 2nd derivative on fine grid
    x_fine = np.linspace(x_region.min(), x_region.max(), 200)
    d2y = np.polyval(pp, x_fine)

    # Find minimum (most negative = steepest concave down)
    idx_min = np.argmin(d2y)
    knee_rank_log = x_fine[idx_min]

    # Find closest actual rank in the dataframe
    actual_idx = np.argmin(np.abs(log_rank - knee_rank_log))
    return actual_idx


def find_knee_rankabundance(df):
    """Find knee on log-log rank-abundance curve (10^3 to 10^4 rank range).

    Uses polynomial smoothing of log(count) vs log(rank) to find steepest
    concave-down section.
    """
    log_rank = np.log10(df["rank"].to_numpy(dtype=float))
    log_count = np.log10(df["count"].to_numpy(dtype=float))

    # Focus on the 10^3 to 10^4 rank range
    mask = (log_rank >= 3.0) & (log_rank <= 4.0)
    if mask.sum() < 10:
        return 0

    x_region = log_rank[mask]
    y_region = log_count[mask]

    try:
        p = np.polyfit(x_region, y_region, 5)
        pp = np.polyder(p, 2)  # 2nd derivative polynomial
    except np.linalg.LinAlgError:
        return 0

    # Evaluate 2nd derivative on fine grid
    x_fine = np.linspace(x_region.min(), x_region.max(), 200)
    d2y = np.polyval(pp, x_fine)

    # Find minimum (most negative = steepest concave down)
    idx_min = np.argmin(d2y)
    knee_rank_log = x_fine[idx_min]

    # Find closest actual rank in the dataframe
    actual_idx = np.argmin(np.abs(log_rank - knee_rank_log))
    return actual_idx


def threshold_calcs(df, method="knee", cumulative_cutoff=0.95):
    total_reads = int(df["count"].sum())
    min_count = max(10, int(total_reads * 0.00001))
    min_count_mask = df["count"] >= min_count
    min_count_indices = np.flatnonzero(min_count_mask.to_numpy())
    min_count_last = int(min_count_indices[-1]) if len(min_count_indices) else None

    knee_cum_index = find_knee_cumulative(df)
    knee_cum_rank = int(df.iloc[knee_cum_index]["rank"])
    knee_cum_count = int(df.iloc[knee_cum_index]["count"])
    knee_cum_variants = knee_cum_index + 1

    knee_ra_index = find_knee_rankabundance(df)
    knee_ra_rank = int(df.iloc[knee_ra_index]["rank"])
    knee_ra_count = int(df.iloc[knee_ra_index]["count"])
    knee_ra_variants = knee_ra_index + 1

    cumulative = {}
    for cutoff in CUMULATIVE_CUTOFFS:
        index = min(
            int(np.searchsorted(df["cumulative_frac"].to_numpy(), cutoff, side="left")),
            len(df) - 1,
        )
        cumulative[cutoff] = {
            "index": index,
            "rank": int(df.iloc[index]["rank"]),
            "count": int(df.iloc[index]["count"]),
            "fraction": float(df.iloc[index]["cumulative_frac"]),
            "variants": index + 1,
        }

    threshold_results = {
        "min_count": {
            "value": min_count,
            "rank": (
                int(df.iloc[min_count_last]["rank"])
                if min_count_last is not None
                else None
            ),
            "count": min_count,
            "variants": int(min_count_mask.sum()),
        },
        "knee_cumulative": {
            "value": knee_cum_count,
            "rank": knee_cum_rank,
            "count": knee_cum_count,
            "variants": knee_cum_variants,
        },
        "knee_rankabundance": {
            "value": knee_ra_count,
            "rank": knee_ra_rank,
            "count": knee_ra_count,
            "variants": knee_ra_variants,
        },
        "cumulative": cumulative,
        "selected_method": method,
        "selected_cumulative_cutoff": cumulative_cutoff,
    }
    return threshold_results


def write_thresholds(thresholds, input_file, output_file):
    selected_method = thresholds["selected_method"]
    selected_cutoff = thresholds["selected_cumulative_cutoff"]
    lines = [
        f"Input file: {input_file}",
        f"Selected method: {selected_method}",
        f"Selected cumulative cutoff: {selected_cutoff:.2f}",
        "",
        "method\tthreshold\tcutoff\tthreshold_rank\tthreshold_count\tvariants_retained\tselected",
    ]

    for method in ("min_count", "knee_cumulative", "knee_rankabundance"):
        result = thresholds[method]
        is_selected = method == selected_method
        lines.append(
            f"{method}\t{result['value']}\tNA\t"
            f"{result['rank'] if result['rank'] is not None else 'NA'}\t"
            f"{result['count']}\t{result['variants']}\t"
            f"{'SELECTED' if is_selected else ''}"
        )

    for cutoff in CUMULATIVE_CUTOFFS:
        result = thresholds["cumulative"][cutoff]
        is_selected = selected_method == "cumulative" and cutoff == selected_cutoff
        lines.append(
            f"cumulative\t{result['fraction']:.8g}\t{cutoff:.2f}\t"
            f"{result['rank']}\t{result['count']}\t{result['variants']}\t"
            f"{'SELECTED' if is_selected else ''}"
        )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_read_count_threshold(thresholds):
    method = thresholds["selected_method"]
    if method in ("knee_cumulative", "knee_rankabundance", "min_count"):
        return int(thresholds[method]["count"])
    if method == "cumulative":
        cutoff = thresholds["selected_cumulative_cutoff"]
        return int(thresholds["cumulative"][cutoff]["count"])
    raise ValueError(f"Unsupported whitelist threshold method: {method}")


def write_whitelist(rank_abundance_files, dataframes, thresholds):
    """Write sequences passing each file's selected count threshold in every file."""
    passing_ranks = []
    for input_file, df, threshold_result in zip(
        rank_abundance_files, dataframes, thresholds
    ):
        count_threshold = selected_read_count_threshold(threshold_result)
        passing = df.loc[df["count"] >= count_threshold, ["sequence", "rank"]]
        if passing["sequence"].duplicated().any():
            raise ValueError(f"{input_file} contains duplicate sequences.")
        passing_ranks.append(
            passing.set_index("sequence")["rank"].rename(input_file.stem)
        )

    common_sequences = set(passing_ranks[0].index)
    for ranks in passing_ranks[1:]:
        common_sequences.intersection_update(ranks.index)

    rank_table = pd.concat(
        [ranks.loc[sorted(common_sequences)] for ranks in passing_ranks],
        axis=1,
    )
    rank_table.insert(0, "sequence", rank_table.index)
    rank_table.index.name = None
    rank_columns = [input_file.stem for input_file in rank_abundance_files]
    rank_table["average_rank"] = rank_table[rank_columns].mean(axis=1)
    rank_table.sort_values(
        ["average_rank", "sequence"], inplace=True, ignore_index=True
    )
    return rank_table, len(rank_table), count_threshold


def plot_rank_abundance(df, thresholds, input_file, output_file):
    selected_method = thresholds["selected_method"]
    selected_cutoff = thresholds["selected_cumulative_cutoff"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    rank = df["rank"].to_numpy(dtype=float)
    count = df["count"].to_numpy(dtype=float)

    axes[0].plot(rank, count, color="black", lw=1.2, label="Rank abundance")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Rank (log scale)")
    axes[0].set_ylabel("Read count (log scale)")
    axes[0].set_title("Rank abundance")

    min_count = thresholds["min_count"]
    if min_count["rank"] is not None:
        axes[0].axhline(
            min_count["value"],
            color="tab:blue",
            linestyle="--",
            linewidth=2.2 if selected_method == "min_count" else 1.2,
            label=f"Min count: {min_count['value']} reads, rank {min_count['rank']}"
            + (" [SELECTED]" if selected_method == "min_count" else ""),
        )

    knee_cum = thresholds["knee_cumulative"]
    axes[0].axvline(
        knee_cum["rank"],
        color="tab:orange",
        linestyle="-.",
        linewidth=2.2 if selected_method == "knee_cumulative" else 1.2,
        label=f"Knee (cumul.): rank {knee_cum['rank']}, {knee_cum['count']} reads"
        + (" [SELECTED]" if selected_method == "knee_cumulative" else ""),
    )
    axes[0].scatter([knee_cum["rank"]], [knee_cum["count"]], color="tab:orange", zorder=3)

    knee_ra = thresholds["knee_rankabundance"]
    axes[0].axvline(
        knee_ra["rank"],
        color="tab:red",
        linestyle=":",
        linewidth=2.2 if selected_method == "knee_rankabundance" else 1.2,
        label=f"Knee (rank-abund.): rank {knee_ra['rank']}, {knee_ra['count']} reads"
        + (" [SELECTED]" if selected_method == "knee_rankabundance" else ""),
    )
    axes[0].scatter([knee_ra["rank"]], [knee_ra["count"]], color="tab:red", zorder=3)

    axes[1].plot(rank, df["cumulative_frac"], color="black", lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rank (log scale)")
    axes[1].set_ylabel("Cumulative fraction of reads")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Cumulative read coverage")

    # Add knee markers to cumulative plot
    knee_cum = thresholds["knee_cumulative"]
    axes[1].axvline(
        knee_cum["rank"],
        color="tab:orange",
        linestyle="-.",
        linewidth=2.2 if selected_method == "knee_cumulative" else 1.2,
        label=f"Knee (cumul.): rank {knee_cum['rank']}"
        + (" [SELECTED]" if selected_method == "knee_cumulative" else ""),
    )

    knee_ra = thresholds["knee_rankabundance"]
    axes[1].axvline(
        knee_ra["rank"],
        color="tab:red",
        linestyle=":",
        linewidth=2.2 if selected_method == "knee_rankabundance" else 1.2,
        label=f"Knee (rank-abund.): rank {knee_ra['rank']}"
        + (" [SELECTED]" if selected_method == "knee_rankabundance" else ""),
    )

    for cutoff in CUMULATIVE_CUTOFFS:
        result = thresholds["cumulative"][cutoff]
        selected = selected_method == "cumulative" and cutoff == selected_cutoff
        color = {0.90: "tab:green", 0.95: "tab:orange", 0.99: "tab:purple"}[cutoff]
        axes[1].axhline(
            cutoff,
            color=color,
            linestyle="--",
            linewidth=2.2 if selected else 1.1,
            label=f"{cutoff:.0%}: rank {result['rank']}, {result['count']} reads"
            + (" [SELECTED]" if selected else ""),
        )
        axes[1].scatter(
            [result["rank"]],
            [result["fraction"]],
            color=color,
            zorder=3,
        )

    for axis in axes:
        axis.legend(fontsize="small")
        axis.grid(True, which="both", alpha=0.2)
    fig.suptitle(
        f"{input_file.stem}\nSelected threshold method: {selected_method}"
        + (
            f" ({selected_cutoff:.0%})"
            if selected_method == "cumulative"
            else ""
        )
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_files = resolve_input_files(args.input)
    output_dir = input_files[0].parent.parent / "seqthresh"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes = []
    threshold_results = []
    for input_file in input_files:
        df = load_rank_abundance(input_file)
        thresholds = threshold_calcs(
            df,
            method=args.method,
            cumulative_cutoff=args.cumulative_cutoff,
        )
        output_stem = input_file.stem
        threshold_file = output_dir / f"{output_stem}_thresholds.txt"
        plot_file = output_dir / f"{output_stem}_thresholds.png"
        write_thresholds(thresholds, input_file, threshold_file)
        plot_rank_abundance(df, thresholds, input_file, plot_file)
        dataframes.append(df)
        threshold_results.append(thresholds)
        print(f"Thresholds written to: {threshold_file}")
        print(f"Threshold plot written to: {plot_file}")

    whitelist_rank_table, whitelist_size, count_threshold = write_whitelist(
        input_files, dataframes, threshold_results
    )

    whitelist_file = output_dir / f"final_{args.method}_whitelist.csv"
    whitelist_rank_table.to_csv(whitelist_file, index=False)


    print(f"Final whitelist ({whitelist_size} shared sequences) from {args.method} filtering written to: {whitelist_file}")


if __name__ == "__main__":
    main()
