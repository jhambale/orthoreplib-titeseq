#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on February 26 15:38 2026
@author: @jhambale

Intakes a directory of demultiplexed fastq subfolders, a tab-delimited trace
text file, and an output directory.

For each row in the trace file the script:
  1. Finds the subfolder inside the input directory whose name contains the
     matching sourcefile_hash.
  2. Inside that subfolder, locates the forward and reverse fastq.gz files
     that carry both barcode_F and barcode_R in their filenames.
  3. Copies those two files to the output directory, renaming them:
         {sample_id}_R1_001.fastq.gz   (forward)
         {sample_id}_R2_001.fastq.gz   (reverse)

Trace file expected columns (tab-separated, with header):
    sample_number  experiment_name  sample_id  sample_name
    barcode_F  barcode_R  sourcefile_hash

Example run:
    python 24_demux_bc_rename.py \\
        -d /path/to/demux_root \\
        -t /path/to/88_singlecell1_trace.txt \\
        -o /path/to/output_dir

    # Or with short flags:
    python 24_demux_bc_rename.py \\
        -d ../../titeseq_data/ngs_raw/jh_027/demultiplex \\
        -t ../../titeseq_data/ngs_raw/jh_027/88_singlecell1_trace.txt \\
        -o ./renamed_fastqs
"""

import argparse
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Locate demultiplexed fastq.gz files by barcode pair and source-file "
            "hash, then copy/rename them according to a trace text file."
        )
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        metavar="INPUT_DIR",
        help=(
            "Root directory that contains timestamped subfolders produced by "
            "separate_by_barcodes.py  (e.g. 20251218_104428_output__7ba68133/)."
        ),
    )
    parser.add_argument(
        "-t",
        "--trace",
        required=True,
        metavar="TRACE_TXT",
        help=(
            "Tab-delimited trace file with columns: sample_number, "
            "experiment_name, sample_id, sample_name, barcode_F, barcode_R, "
            "sourcefile_hash."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="OUTPUT_DIR",
        help="Directory where renamed fastq.gz files will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be copied without actually copying any files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trace-file helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "sample_id",
    "barcode_f",
    "barcode_r",
    "sourcefile_hash",
}


def load_trace(trace_path: str) -> list[dict]:
    """
    Parse the trace TSV and return a list of row dicts.
    Column names are lower-cased so the lookup is case-insensitive.
    Raises ValueError if required columns are missing.
    """
    rows = []
    with open(trace_path, "r", newline="") as fh:
        raw_header = fh.readline().rstrip("\n")
        if not raw_header:
            raise ValueError(f"Trace file is empty: {trace_path}")

        headers = [h.strip().lower() for h in raw_header.split("\t")]
        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise ValueError(
                f"Trace file is missing required column(s): {missing}\n"
                f"  Found: {headers}"
            )

        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue  # skip blank lines
            fields = line.split("\t")
            if len(fields) != len(headers):
                print(
                    f"  [WARNING] Line {lineno} has {len(fields)} fields "
                    f"(expected {len(headers)}); skipping.",
                    file=sys.stderr,
                )
                continue
            rows.append(dict(zip(headers, [f.strip() for f in fields])))

    if not rows:
        raise ValueError(f"Trace file contains no data rows: {trace_path}")

    return rows


# ---------------------------------------------------------------------------
# Subfolder / file lookup helpers
# ---------------------------------------------------------------------------


def find_subfolder_for_hash(root_dir: str, source_hash: str) -> str | None:
    """
    Return the path of the first subfolder inside *root_dir* whose name
    contains *source_hash* (case-insensitive substring match).
    Returns None if no match is found.
    """
    source_hash_lower = source_hash.lower()
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input directory not found: {root_dir}")

    for entry in sorted(entries):
        full = os.path.join(root_dir, entry)
        if os.path.isdir(full) and source_hash_lower in entry.lower():
            return full
    return None


def find_fastq_pair(
    subfolder: str,
    barcode_f: str,
    barcode_r: str,
) -> tuple[str | None, str | None]:
    """
    Inside *subfolder*, locate the forward and reverse fastq.gz files whose
    names contain both *barcode_f* and *barcode_r*.

    Expected filename pattern (produced by separate_by_barcodes.py):
        {n}_forward_{bcF}_{bcR}.fastq.gz
        {n}_reverse_{bcF}_{bcR}.fastq.gz

    Returns a (forward_path, reverse_path) tuple.  Either element is None if
    the file was not found.
    """
    bcF_upper = barcode_f.upper()
    bcR_upper = barcode_r.upper()

    forward_path = None
    reverse_path = None

    try:
        filenames = os.listdir(subfolder)
    except FileNotFoundError:
        raise FileNotFoundError(f"Subfolder not found: {subfolder}")

    for fname in filenames:
        if not fname.endswith(".fastq.gz"):
            continue
        fname_upper = fname.upper()
        # Both barcodes must appear in the filename
        if bcF_upper not in fname_upper or bcR_upper not in fname_upper:
            continue
        # Distinguish forward vs reverse by the direction token in the name
        if "_FORWARD_" in fname_upper:
            forward_path = os.path.join(subfolder, fname)
        elif "_REVERSE_" in fname_upper:
            reverse_path = os.path.join(subfolder, fname)

    return forward_path, reverse_path


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main():
    args = parse_arguments()

    input_dir = os.path.abspath(args.directory)
    trace_path = os.path.abspath(args.trace)
    output_dir = os.path.abspath(args.output)
    dry_run = args.dry_run

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(trace_path):
        print(f"[ERROR] Trace file does not exist: {trace_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load trace ───────────────────────────────────────────────────────────
    print(f"Loading trace file: {trace_path}")
    try:
        rows = load_trace(trace_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(rows)} sample(s) found in trace file.")

    # ── Create output directory ───────────────────────────────────────────────
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print(f"[DRY RUN] Output directory would be: {output_dir}")

    # ── Per-row processing ────────────────────────────────────────────────────
    success_count = 0
    warning_count = 0

    # Cache subfolder lookups so we don't re-scan for the same hash repeatedly
    subfolder_cache: dict[str, str | None] = {}

    for row in rows:
        sample_id = row["sample_id"]
        barcode_f = row["barcode_f"]
        barcode_r = row["barcode_r"]
        src_hash = row["sourcefile_hash"]

        print(
            f"\n-- Sample: {sample_id}  (bcF={barcode_f}, bcR={barcode_r}, hash={src_hash})"
        )

        # 1. Locate the matching subfolder
        if src_hash not in subfolder_cache:
            subfolder_cache[src_hash] = find_subfolder_for_hash(input_dir, src_hash)

        subfolder = subfolder_cache[src_hash]
        if subfolder is None:
            print(
                f"  [WARNING] No subfolder found containing hash '{src_hash}' "
                f"in {input_dir} — skipping sample.",
                file=sys.stderr,
            )
            warning_count += 1
            continue

        print(f"  Found subfolder: {os.path.basename(subfolder)}")

        # 2. Locate the fastq pair inside that subfolder
        fwd_src, rev_src = find_fastq_pair(subfolder, barcode_f, barcode_r)

        if fwd_src is None:
            print(
                f"  [WARNING] Forward fastq not found for "
                f"bcF={barcode_f}, bcR={barcode_r} in {subfolder}",
                file=sys.stderr,
            )
            warning_count += 1
        else:
            print(f"  Forward source : {os.path.basename(fwd_src)}")

        if rev_src is None:
            print(
                f"  [WARNING] Reverse fastq not found for "
                f"bcF={barcode_f}, bcR={barcode_r} in {subfolder}",
                file=sys.stderr,
            )
            warning_count += 1
        else:
            print(f"  Reverse source : {os.path.basename(rev_src)}")

        # 3. Copy and rename
        for src, read_tag in [(fwd_src, "R1"), (rev_src, "R2")]:
            if src is None:
                continue
            dest_name = f"{sample_id}_{read_tag}_001.fastq.gz"
            dest_path = os.path.join(output_dir, dest_name)

            if dry_run:
                print(
                    f"  [DRY RUN] Would copy  {os.path.basename(src)}  ->  {dest_name}"
                )
            else:
                shutil.copy2(src, dest_path)
                print(f"  Copied  {os.path.basename(src)}  ->  {dest_name}")

        # Count as success only if both files were found
        if fwd_src is not None and rev_src is not None:
            success_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}"
        f"Done.  {success_count}/{len(rows)} samples copied successfully; "
        f"{warning_count} warning(s)."
    )


if __name__ == "__main__":
    main()
