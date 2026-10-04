#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 01 March 2026
@author: @jhambale

Reads a library CSV of Kds produced by 10_compute_Kds.py and tallies every unique point mutation found
in the 'variant_id' column.

A point mutation is any token matching the pattern  AxB  where:
  A  = single-letter wild-type amino acid
  x  = residue position (integer)
  B  = single-letter mutant amino acid

For each unique mutation the script prints:
  - the mutation label  (e.g. S7G)
  - the raw count       (number of sequences that carry it)
  - the frequency       (count / total sequences in the file)

Results are sorted by descending frequency.

For each input file the script writes into the output directory:
  - {stem}_all.csv              all unique point mutations, counts, frequencies
  - {stem}_position_freq_all.png  per-position summed frequency bar plot
  - {stem}_thresh{t}.csv        mutations whose frequency >= -t threshold
  - {stem}_posfilter.csv        proximity-filtered set: for every pair of
                                mutations whose positions are <= 5 aa apart,
                                only the more-frequent mutation is kept

Example run
-----------
python 11_unique_muts.py \\
    -i ../../titeseq_data/cleaned_dyn2_library_scores.csv \\
    -o ../../titeseq_data/unique_muts \\
    -t 0.05 \\
    --regions ../../../../nb_region_id/analysis/baker_nb_regions.txt

# Multiple files at once:
python 11_unique_muts.py \\
    -i ../../titeseq_data/cleaned_*.csv \\
    -o ../../titeseq_data/unique_muts \\
    -t 0.05 \\
    --regions ../../../../nb_region_id/analysis/baker_nb_regions.txt
"""

import argparse
import ast
import re
import sys
from collections import Counter, defaultdict

# from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
_POINT_MUT_RE = re.compile(r"^[A-Z]\d+[A-Z]$")
_POSITION_RE = re.compile(r"[A-Z](\d+)[A-Z]")

PROXIMITY_THRESHOLD = 2  # positions; pairs <= this distance are "close"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_variant_id(raw_value) -> list[str]:
    """Return a list of point-mutation strings from one cell in variant_id.

    The column stores values serialised by pandas/Python, e.g.:
        "['S7G', 'V12I', 'A50V']"
        "['S7N']"
        NaN / empty string  (treated as no mutations)

    Only tokens that match the AxB pattern are kept; indels or other
    annotations are silently ignored.
    """
    if pd.isna(raw_value) or str(raw_value).strip() in ("", "[]"):
        return []

    try:
        parsed = ast.literal_eval(str(raw_value))
    except (ValueError, SyntaxError):
        parsed = re.findall(r"[A-Z]\d+[A-Z]", str(raw_value))

    if isinstance(parsed, str):
        parsed = [parsed]

    return [tok for tok in parsed if _POINT_MUT_RE.match(str(tok))]


def mut_position(mut: str) -> int:
    """Extract the integer residue position from a mutation string like A50V."""
    m = _POSITION_RE.match(mut)
    if not m:
        raise ValueError(f"Cannot extract position from mutation: {mut!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def load_region_map(regions_txt: Path) -> dict[int, str]:
    """Parse a two-column tab-separated region file produced by 01_region_id.py.

    Expected format (with header):
        position\tregion
        1\tframework1
        2\tframework1
        ...

    Returns a dict mapping integer residue position -> region label string.
    """
    region_map: dict[int, str] = {}
    with regions_txt.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("position"):
                continue  # skip blank lines and header
            parts = line.split("\t")
            if len(parts) != 2:
                print(
                    f"  [WARN] regions file line {lineno} has unexpected format, skipping: {line!r}"
                )
                continue
            try:
                region_map[int(parts[0])] = parts[1]
            except ValueError:
                print(
                    f"  [WARN] regions file line {lineno}: non-integer position, skipping: {line!r}"
                )
    print(
        f"  Loaded {len(region_map)} position→region mappings from {regions_txt.name}"
    )
    return region_map


def annotate_position_region(
    df: pd.DataFrame, region_map: dict[int, str] | None
) -> pd.DataFrame:
    """Add 'aa_position' and 'region' columns to a mutation DataFrame.

    'aa_position' is the integer residue position extracted from the mutation
    string (e.g. 'S52G' -> 52).

    'region' is looked up from region_map when provided; positions absent from
    the map are labelled 'unknown'.
    """
    df = df.copy()
    df["aa_position"] = df["mutation"].apply(
        lambda m: int(_POSITION_RE.match(m).group(1))  # type: ignore[union-attr]
    )
    if region_map is not None:
        df["region"] = df["aa_position"].map(lambda p: region_map.get(p, "unknown"))
    else:
        df["region"] = "unknown"
    return df


def build_mut_df(counter: Counter, total_sequences: int) -> pd.DataFrame:
    """Return a DataFrame with columns mutation / count / frequency,
    sorted by descending frequency."""
    rows = [
        {"mutation": mut, "count": count, "frequency": count / total_sequences}
        for mut, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def export_csv(df: pd.DataFrame, out_path: Path, label: str) -> None:
    df.sort_values(by="aa_position").to_csv(out_path, index=False)
    print(f"  {label} CSV saved  -> {out_path}")


def export_position_barplot(
    counter: Counter,
    total_sequences: int,
    out_path: Path,
    file_stem: str,
) -> None:
    """Bar chart of per-position summed mutation frequency."""
    pos_counts: dict[int, int] = defaultdict(int)
    for mut, count in counter.items():
        m = _POSITION_RE.match(mut)
        if m:
            pos_counts[int(m.group(1))] += count

    if not pos_counts:
        print("  [WARN] No position data to plot — skipping bar chart.")
        return

    positions = sorted(pos_counts.keys())
    freq_sums = [pos_counts[p] / total_sequences for p in positions]

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.35), 5))

    ax.bar(
        positions,
        freq_sums,
        width=0.7,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("Residue position", fontsize=12)
    ax.set_ylabel("Summed mutation frequency", fontsize=12)
    ax.set_title(
        f"Per-position mutation frequency\n{file_stem}",
        fontsize=11,
        pad=10,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(p) for p in positions],
        rotation=90,
        fontsize=max(5, min(9, 200 // len(positions))),
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved       -> {out_path}")


# ---------------------------------------------------------------------------
# Proximity filter
# ---------------------------------------------------------------------------


def proximity_filter(all_df: pd.DataFrame) -> pd.DataFrame:
    """From the full mutation set, remove lower-frequency mutations that sit
    within PROXIMITY_THRESHOLD positions of a more-frequent mutation.

    Algorithm (greedy, frequency-descending):
      1. Sort mutations by descending frequency.
      2. Walk through the sorted list; accept a mutation only if no
         already-accepted mutation is within PROXIMITY_THRESHOLD positions.
      3. Repeat until the accepted set is stable (no further removals
         triggered by newly admitted mutations — one pass suffices because
         we process in frequency order).
    """
    # Work with rows sorted highest → lowest frequency
    df = all_df.sort_values("frequency", ascending=False).reset_index(drop=True)

    accepted_positions: list[int] = []
    accepted_mask = [False] * len(df)

    for idx, row in df.iterrows():
        pos = mut_position(row["mutation"])
        too_close = any(
            abs(pos - ap) <= PROXIMITY_THRESHOLD for ap in accepted_positions
        )
        if not too_close:
            accepted_mask[idx] = True  # type:ignore
            accepted_positions.append(pos)

    filtered = df[accepted_mask].reset_index(drop=True)
    n_removed = len(df) - len(filtered)
    print(
        f"  Proximity filter : kept {len(filtered)} / {len(df)} mutations "
        f"({n_removed} removed, ≤{PROXIMITY_THRESHOLD} aa from a more-frequent mutation)"
    )
    return filtered  # type:ignore


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------


def analyze_file(
    csv_path: Path,
    out_dir: Path,
    thresh: float | None,
    region_map: dict[int, str] | None,
) -> None:
    """Load one CSV, count unique point mutations, print a report,
    and export all output files."""

    print(f"\n{'=' * 70}")
    print(f"  File : {csv_path.name}")
    print(f"{'=' * 70}")

    df = pd.read_csv(csv_path)

    if "variant_id" not in df.columns:
        print(
            f"  [ERROR] Column 'variant_id' not found in {csv_path.name}. "
            f"Available columns: {list(df.columns)}"
        )
        return

    total_sequences = len(df)
    print(f"  Total sequences (rows) : {total_sequences:,}")

    # ---- Count every point mutation ------------------------------------
    counter: Counter = Counter()
    for raw in df["variant_id"]:
        for mut in parse_variant_id(raw):
            counter[mut] += 1

    if not counter:
        print("  No point mutations found.")
        return

    n_unique = len(counter)
    print(f"  Unique point mutations : {n_unique:,}\n")

    # ---- Console table -------------------------------------------------
    col_w = max(len(m) for m in counter) + 2
    print(f"  {'Mutation':<{col_w}}  {'Count':>8}  {'Frequency':>10}")
    print(f"  {'-' * col_w}  {'-' * 8}  {'-' * 10}")
    for mut, count in counter.most_common():
        freq = count / total_sequences
        print(f"  {mut:<{col_w}}  {count:>8,}  {freq:>10.4f}")

    # ---- Build base DataFrame ------------------------------------------
    stem = csv_path.stem
    all_df = build_mut_df(counter, total_sequences)
    all_df = annotate_position_region(all_df, region_map)

    # ---- 1. _all CSV + plot --------------------------------------------
    export_csv(all_df, out_dir / f"{stem}_all.csv", label="_all")
    export_position_barplot(
        counter,
        total_sequences,
        out_dir / f"{stem}_position_freq_all.png",
        file_stem=stem,
    )

    # ---- 2. threshold-filtered CSV -------------------------------------
    if thresh is not None:
        thresh_df = all_df[all_df["count"] >= thresh].reset_index(
            drop=True
        )  # aa_position/region already present
        tag = int(thresh)
        thresh_path = out_dir / f"{stem}_thresh{tag}.csv"
        export_csv(thresh_df, thresh_path, label=f"_thresh{tag}")
        print(
            f"  Threshold count >= {tag} : {len(thresh_df)} / {n_unique} mutations retained"
        )

    # ---- 3. proximity-filtered CSV -------------------------------------
    # proximity_filter sorts/filters rows but preserves all columns,
    # so aa_position and region carry through automatically.
    print()
    posfilter_df = proximity_filter(all_df)
    posfilter_path = out_dir / f"{stem}_posfilter.csv"
    export_csv(posfilter_df, posfilter_path, label="_posfilter")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count and report unique point mutation frequencies from the "
            "'variant_id' column of a cleaned library CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more cleaned library CSV file paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Directory where output CSVs and plots will be written "
        "(created if it does not exist).",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        type=int,
        default=None,
        metavar="COUNT",
        help="Minimum absolute count threshold. Mutations observed at least "
        "this many times are written to a separate _thresh{t}.csv. Omit to skip.",
    )
    parser.add_argument(
        "--regions",
        default=None,
        metavar="TXT",
        help="Path to the two-column region assignment file produced by "
        "nb_region_id/analysis/01_region_id.py (tab-separated: position, region). "
        "When provided, every output CSV gains 'aa_position' and 'region' columns.",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    paths = [Path(p) for p in args.input]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
        sys.exit(1)

    if args.thresh is not None and args.thresh < 1:
        print(
            "[ERROR] -t / --thresh must be a positive integer count.", file=sys.stderr
        )
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load region map once; shared across all input files
    region_map: dict[int, str] | None = None
    if args.regions is not None:
        regions_path = Path(args.regions)
        if not regions_path.exists():
            print(f"[ERROR] regions file not found: {regions_path}", file=sys.stderr)
            sys.exit(1)
        region_map = load_region_map(regions_path)

    for csv_path in paths:
        analyze_file(csv_path, out_dir, args.thresh, region_map)

    print()  # trailing newline


if __name__ == "__main__":
    main()
