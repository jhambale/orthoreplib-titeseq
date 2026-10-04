#!/usr/bin/env python3

"""
Created on April 07 17:50 2026
@author: @jhambale

Reads a batch of raw cell-sort CSV files, groups them by cell number (yXX)
and concentration (_YYYYnM), sums Total Events and Sorted Counts across the
sort + sortcomp complement files in each group, normalizes every group's total
events to the same target level, scales the Sorted Counts proportionally, and
writes one output CSV per cell-number+concentration group.

example run:
    python 01_cellcount_norm.py \
        -i ../../titeseq_data/raw/jh_001/cell_ct_raw/*.csv \
        -o ../../titeseq_data/raw/jh_001/cell_ct_out/
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="normalize cell-sort counts across multiple CSV files."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more input CSV files (supports shell glob expansion).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for normalized CSV files.",
    )
    parser.add_argument(
        "--target-events",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Target total-event level to normalize to. "
            "Defaults to the maximum summed total events found across all groups."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

# Matches examples: y55_0p64nMsort.csv, y55_0p64nMsortcomp.csv,
# y102_400nMsort2.csv, y102_400nMsort3comp.csv, y96rpt_3p2nMsort.csv,
# yTI7475_1nMrep1_sortcomp2.csv
_FILE_RE = re.compile(
    r"(?P<cell>y[A-Za-z0-9]+)"  # cell id e.g. yTI7475, y55, or y96rpt with optional alphanumeric suffix
    r"_"
    r"(?P<conc>[^_]+?)"  # raw concentration string e.g. 0p64, 1, or 2000
    r"nM"
    r"(?:rep(?P<rep>\d+))?"  # optional replicate like 'rep1', 'rep2', 'rep3'
    r"_?"  # optional underscore separator before sort label
    r"sort"  # literal 'sort'
    r"(?:\d*)?"  # optional numeric suffix like '1', '2', '3'
    r"(?:(?P<comp>comp)(?:\d*)?)?"  # optional 'comp' suffix with trailing numeric suffixes like 'comp2'
    r"\.csv$",
    re.IGNORECASE,
)


def parse_filename(path: str):
    """
    Return (cell_id, conc_float, replicate, is_comp) for a valid sort-count CSV filename,
    or None if the filename does not match the expected pattern.

    The concentration string uses 'p' as a decimal point placeholder
    (e.g. '0p64' -> 0.64, '3p2' -> 3.2, '16' -> 16.0).
    """
    basename = os.path.basename(path)
    m = _FILE_RE.match(basename)
    if m is None:
        return None

    cell_id = m.group("cell").lower()  # 'y55' or 'y96rpt'
    conc_raw = m.group("conc")  # '0p64', '3p2', '2000', …
    replicate = m.group("rep")  # '1', '2', etc.; None when no replicate is present
    is_comp = m.group("comp") is not None  # True when 'comp' present

    conc_str = conc_raw.replace("p", ".")  # '0p64' -> '0.64'
    try:
        conc_float = float(conc_str)
    except ValueError:
        print(
            f"[WARNING] Could not convert concentration '{conc_raw}' "
            f"to a number in file '{basename}' — skipping.",
            file=sys.stderr,
        )
        return None

    return cell_id, conc_float, replicate, is_comp


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_and_group(file_paths: list[str]):
    """
    Parse every file path, load its CSV, and return a nested dict:

        groups[(cell_id, conc_float, replicate)] -> list of DataFrames

    Each DataFrame has already been tagged with its source filename so
    messages can refer back to it if needed.
    """
    groups: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
    skipped = 0

    for path in file_paths:
        parsed = parse_filename(path)
        if parsed is None:
            print(
                f"[WARNING] Skipping unrecognized filename: {os.path.basename(path)}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        cell_id, conc_float, replicate, _ = parsed  # is_comp not needed after grouping

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[ERROR] Could not read '{path}': {exc}", file=sys.stderr)
            skipped += 1
            continue

        # Validate required columns
        required = {"Sort Gate", "Total Event", "Sorted Count"}
        missing = required - set(df.columns)
        if missing:
            print(
                f"[ERROR] File '{os.path.basename(path)}' is missing columns "
                f"{missing} — skipping.",
                file=sys.stderr,
            )
            skipped += 1
            continue

        df = df.copy()
        df["_source_file"] = os.path.basename(path)
        groups[(cell_id, conc_float, replicate)].append(df)

    if skipped:
        print(f"[INFO] {skipped} file(s) were skipped.", file=sys.stderr)

    return groups


def sum_group(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all DataFrames in a group (sort + sortcomp), then sum
    'Total Event' and 'Sorted Count' for each unique Sort Gate.

    Returns a tidy DataFrame with columns:
        Sort Gate | Total Event (sum) | Sorted Count (sum)
    """
    combined = pd.concat(frames, ignore_index=True)

    summed = (
        combined.groupby("Sort Gate", sort=False)[["Total Event", "Sorted Count"]]
        .sum()
        .reset_index()
    )
    return summed


def normalize_groups(
    summed_groups: dict[tuple, pd.DataFrame],
    target_events: float | None,
) -> dict[tuple, pd.DataFrame]:
    """
    For every (cell_id, conc, replicate) group:
        group_max     = max(Total Event across all gates in that group)
        scale_factor  = group_max / Total Event   (per gate row; = 1.0 for the max gate)
        normalized Sorted Count = round(Sorted Count * scale_factor)
        normalized Total Event  = group_max  (same for every row in the group)

    target_events is unused in the per-gate calculation but is kept as an optional
    parameter so callers can override the printed reference value if desired.
    """
    normalized: dict[tuple, pd.DataFrame] = {}
    for key, df in summed_groups.items():
        df = df.copy()
        # The gate with the most total events sets the reference ceiling for
        # this group. Every gate is scaled up to that ceiling independently.
        group_max_events = float(df["Total Event"].max())

        # Per-row scale factor: >= 1.0 for all gates; exactly 1.0 for the
        # gate that already has the maximum total event count.
        scale = group_max_events / df["Total Event"]

        df["normalized Total Event"] = group_max_events
        df["normalized Sorted Count"] = np.round(df["Sorted Count"] * scale).astype(int)
        df["Scale Factor"] = scale

        normalized[key] = df

    # Expose the global max for the caller's info message (if target_events
    # was not supplied, derive it from what we just computed).
    if target_events is None:
        all_maxes = [
            float(df["normalized Total Event"].iloc[0]) for df in normalized.values()
        ]
        target_events = max(all_maxes) if all_maxes else 0.0

    return normalized, target_events  # type: ignore


def build_output_filename(
    cell_id: str, conc_float: float, replicate: str | None
) -> str:
    """
    Convert back to a clean, human-readable filename.
    e.g.  ('y55', 0.64, None) -> 'y55_0.64nM_norm.csv'
          ('yTI7475', 1.0, '1') -> 'yti7475_1.0nMrep1_norm.csv'
    """
    rep_suffix = f"rep{replicate}" if replicate is not None else ""
    return f"{cell_id}_{conc_float}nM{rep_suffix}_norm.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Create output directory if needed
    # ------------------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load and group files
    # ------------------------------------------------------------------
    print(f"[INFO] Processing {len(args.input)} input file(s) …", file=sys.stderr)
    groups = load_and_group(args.input)

    if not groups:
        print("[ERROR] No valid files could be grouped. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Sum counts within each (cell, conc, replicate) group
    # ------------------------------------------------------------------
    summed_groups: dict[tuple, pd.DataFrame] = {}
    for key, frames in groups.items():
        summed_groups[key] = sum_group(frames)

    # ------------------------------------------------------------------
    # 4. Report grouping summary (sorted by cell then concentration)
    # ------------------------------------------------------------------
    print("\n[INFO] Groups found:", file=sys.stderr)
    for (cell_id, conc, replicate), df in sorted(
        summed_groups.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        n_gates = len(df)
        n_files = len(groups[(cell_id, conc, replicate)])
        tot_ev = df["Total Event"].sum()
        tot_sort = df["Sorted Count"].sum()
        print(
            f"  {cell_id:>4s}  {conc:>10.4g} nM"
            f"{f' rep{replicate}' if replicate is not None else ''}  |  "
            f"{n_files} file(s), {n_gates} gate(s)  |  "
            f"total events = {tot_ev:>12,.0f}  |  "
            f"sorted = {tot_sort:>10,.0f}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 5. normalize
    # ------------------------------------------------------------------
    normalized_groups, target_used = normalize_groups(summed_groups, args.target_events)
    print(
        f"\n[INFO] Normalising all gates to {target_used:,.0f} total events.",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # 6. Write output CSVs
    # ------------------------------------------------------------------
    written = 0
    for (cell_id, conc, replicate), df in sorted(
        normalized_groups.items(),  # type: ignore
        key=lambda x: (x[0][0], x[0][1], x[0][2] or ""),
    ):
        out_name = build_output_filename(cell_id, conc, replicate)
        out_path = os.path.join(args.output, out_name)

        # Column order for the output
        out_cols = [
            "Sort Gate",
            "Total Event",
            "Sorted Count",
            "normalized Total Event",
            "normalized Sorted Count",
            "Scale Factor",
        ]

        # Append a totals row summing Total Event and Sorted Count only
        totals_row = pd.DataFrame(
            [
                {
                    "Sort Gate": "Total",
                    "Total Event": df["Total Event"].sum(),
                    "Sorted Count": df["Sorted Count"].sum(),
                    "normalized Total Event": "",
                    "normalized Sorted Count": "",
                    "Scale Factor": "",
                }
            ]
        )
        out_df = pd.concat([df[out_cols], totals_row[out_cols]], ignore_index=True)
        out_df.to_csv(out_path, index=False)
        print(f"[INFO]   Wrote: {out_path}", file=sys.stderr)
        written += 1

    print(f"\n[INFO] Done — {written} normalized CSV(s) written to '{args.output}'.")


if __name__ == "__main__":
    main()
