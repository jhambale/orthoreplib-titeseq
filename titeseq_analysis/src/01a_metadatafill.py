#!/usr/bin/env python3

"""
Fill a metadata skeleton with optional cell counts, barcode indices, and
fluorescence statistics.

The metadata skeleton is copied to ``*_metadata_filled.txt`` in its original
directory.  Sample names and binder names are taken from the skeleton; the
optional input files only add measurements and barcode values.

Created on August 31 00:29:00 2026
@author: jhambale
"""

import argparse
import csv
import re
import sys
from pathlib import Path


BIN_NAMES = {0: "NoB", 1: "LowB", 2: "MedB", 3: "HighB"}
BIN_FROM_NAME = {name: number for number, name in BIN_NAMES.items()}
FASTQ_SUFFIX = re.compile(r"_R1_001(?:\.fastq)?(?:\.gz)?$", re.IGNORECASE)
# Cell-count outputs include the concentration before the suffix, for example
# ``yTI7475_r1_1.0nM_norm.csv``. Keep the replicate in the group while
# removing only the concentration portion.
GROUP_FILE = re.compile(
    r"^(?P<group>.+)_[0-9]+(?:\.[0-9]+)?nM_norm$",
    re.IGNORECASE,
)
FLUOR_FILE = re.compile(r"^(?P<group>.+)_stats$")
REQUIRED_METADATA_COLUMNS = {
    "sample_id",
    "binder_name",
    "bin",
    "cell_count",
    "barcode_f",
    "barcode_r",
}


class InputFormatError(ValueError):
    """Raised when an input file is structurally valid but unusable."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill a metadata skeleton with optional FACS-derived data."
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="Metadata skeleton text file.",
    )
    parser.add_argument(
        "cell_counts",
        type=Path,
        nargs="?",
        default=None,
        help="Optional directory containing *_norm.csv cell-count files.",
    )
    parser.add_argument(
        "barcodes",
        type=Path,
        nargs="?",
        default=None,
        help="Optional trace text file containing barcode_F and barcode_R.",
    )
    parser.add_argument(
        "fluorescence",
        type=Path,
        nargs="?",
        default=None,
        help="Optional directory containing fluorescence statistics CSV files.",
    )
    return parser.parse_args()


def sample_group(sample_id):
    """Return the sample prefix shared by metadata, count, and stats files."""
    name = Path(str(sample_id)).name
    name = FASTQ_SUFFIX.sub("", name)
    return re.sub(r"_b[0-3]$", "", name)


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise InputFormatError(f"{path} has no tab-delimited header")
        return list(reader)


def require_columns(path, fieldnames, required):
    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise InputFormatError(
            f"{path} is missing required column(s): {', '.join(missing)}"
        )


def require_complete_bins(values, description):
    expected = set(BIN_NAMES)
    actual = set(values)
    if actual != expected:
        missing = ", ".join(str(number) for number in sorted(expected - actual))
        extra = ", ".join(str(number) for number in sorted(actual - expected))
        details = []
        if missing:
            details.append(f"missing bin(s) {missing}")
        if extra:
            details.append(f"unexpected bin(s) {extra}")
        raise InputFormatError(f"{description} must contain exactly four bins ({'; '.join(details)})")


def load_cell_counts(directory):
    counts = {}
    for path in sorted(directory.glob("*_norm.csv")):
        match = GROUP_FILE.match(path.stem)
        if not match:
            raise InputFormatError(
                f"cell-count filename {path.name} must end in '<sample_group>_norm.csv'"
            )
        group = match.group("group")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            require_columns(
                path,
                rows.fieldnames,
                {"Sort Gate", "normalized Sorted Count"},
            )
            file_counts = {}
            for row in rows:
                gate = row.get("Sort Gate", "").strip().split(" - ", 1)[0]
                if gate not in BIN_FROM_NAME:
                    continue
                value = row.get("normalized Sorted Count", "").strip()
                bin_number = BIN_FROM_NAME[gate]
                if bin_number in file_counts:
                    raise InputFormatError(
                        f"{path} contains duplicate {gate} rows"
                    )
                if not value:
                    raise InputFormatError(
                        f"{path} has an empty normalized Sorted Count for {gate}"
                    )
                try:
                    if int(value) < 0:
                        raise ValueError
                except ValueError as exc:
                    raise InputFormatError(
                        f"{path} has an invalid normalized Sorted Count for {gate}: {value!r}"
                    ) from exc
                file_counts[bin_number] = value
        require_complete_bins(file_counts, str(path))
        for bin_number, value in file_counts.items():
            key = (group, bin_number)
            if key in counts:
                raise InputFormatError(
                    f"multiple cell-count files provide {group}, bin {bin_number}"
                )
            counts[key] = value
    if not counts:
        raise InputFormatError(f"no *_norm.csv files found in {directory}")
    return counts


def load_barcodes(path):
    barcodes = {}
    rows = read_tsv(path)
    columns = rows[0].keys() if rows else []
    require_columns(path, columns, {"barcode_F", "barcode_R"})
    if "sample_name" not in columns and "sample_id" not in columns:
        raise InputFormatError(f"{path} must contain sample_name or sample_id")
    for row in rows:
        sample = row.get("sample_name") or row.get("sample_id")
        forward = row.get("barcode_F") or row.get("barcode_f")
        reverse = row.get("barcode_R") or row.get("barcode_r")
        sample = str(sample or "").strip()
        forward = str(forward or "").strip().upper()
        reverse = str(reverse or "").strip().upper()
        if not sample or not forward or not reverse:
            raise InputFormatError(f"{path} contains a barcode row with an empty value")
        if sample in barcodes:
            raise InputFormatError(f"{path} contains duplicate sample {sample!r}")
        if not re.fullmatch(r"[ACGT]+", forward) or not re.fullmatch(r"[ACGT]+", reverse):
            raise InputFormatError(
                f"{path} contains a non-DNA barcode for sample {sample!r}"
            )
        barcodes[sample] = (forward, reverse)
    if not barcodes:
        raise InputFormatError(f"{path} contains no barcode rows")
    return barcodes


def load_fluorescence(directory):
    fluorescence = {}
    for path in sorted(directory.glob("*.csv")):
        match = FLUOR_FILE.match(path.stem)
        if not match:
            raise InputFormatError(
                f"fluorescence filename {path.name} must end in '<sample_group>_stats.csv'"
            )
        group = match.group("group")
        values = {}
        current_bin = None
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not any(cell.strip() for cell in row):
                    continue
                if len(row) < 5:
                    raise InputFormatError(
                        f"{path} contains a row with fewer than five columns"
                    )
                label = row[0].strip()
                if label.startswith("PE-A-Compensated") and current_bin is not None:
                    if current_bin in values:
                        raise InputFormatError(
                            f"{path} contains duplicate PE-A-Compensated values for "
                            f"{BIN_NAMES[current_bin]}"
                        )
                    value = row[4].strip()
                    try:
                        float(value)
                    except ValueError as exc:
                        raise InputFormatError(
                            f"{path} has an invalid PE-A-Compensated mean for "
                            f"{BIN_NAMES[current_bin]}: {value!r}"
                        ) from exc
                    values[current_bin] = value
                elif label.endswith(" - 1"):
                    current_bin = BIN_FROM_NAME.get(label.split(" - ", 1)[0])
        require_complete_bins(values, str(path))
        if group in fluorescence:
            raise InputFormatError(f"multiple fluorescence files provide sample group {group}")
        fluorescence[group] = values
    if not fluorescence:
        raise InputFormatError(f"no fluorescence CSV files found in {directory}")
    return fluorescence


def output_path(metadata):
    if metadata.name.endswith("_metadata.txt"):
        stem = metadata.name[: -len("_metadata.txt")]
        return metadata.with_name(stem + "_metadata_filled.txt")
    return metadata.with_name(metadata.stem + "_filled" + metadata.suffix)


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {args.metadata}")
    for path, label in (
        (args.cell_counts, "cell-count directory"),
        (args.fluorescence, "fluorescence directory"),
    ):
        if path is not None and not path.is_dir():
            raise NotADirectoryError(f"{label} does not exist: {path}")
    if args.barcodes is not None and not args.barcodes.is_file():
        raise FileNotFoundError(f"barcode file does not exist: {args.barcodes}")

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("metadata file has no tab-delimited header")
        rows = list(reader)
    require_columns(args.metadata, fieldnames, REQUIRED_METADATA_COLUMNS)
    if not rows:
        raise InputFormatError(f"{args.metadata} contains no metadata rows")

    cell_counts = load_cell_counts(args.cell_counts) if args.cell_counts else {}
    barcodes = load_barcodes(args.barcodes) if args.barcodes else {}
    fluorescence = (
        load_fluorescence(args.fluorescence) if args.fluorescence else {}
    )

    if fluorescence and "mean_fluorescence" not in fieldnames:
        insert_at = (
            fieldnames.index("cell_count") + 1
            if "cell_count" in fieldnames
            else len(fieldnames)
        )
        fieldnames = (
            fieldnames[:insert_at]
            + ["mean_fluorescence"]
            + fieldnames[insert_at:]
        )

    metadata_keys = set()
    groups = {}
    for row in rows:
        group = sample_group(row.get("sample_id", ""))
        if not group:
            raise InputFormatError("metadata contains a row with an empty sample_id")
        try:
            bin_number = int(row["bin"])
        except (TypeError, ValueError) as exc:
            raise InputFormatError(
                f"metadata has a non-integer bin for sample {row.get('sample_id')!r}"
            ) from exc
        if bin_number not in BIN_NAMES:
            raise InputFormatError(
                f"metadata has invalid bin {bin_number}; expected 0, 1, 2, or 3"
            )
        key = (group, bin_number)
        if key in metadata_keys:
            raise InputFormatError(
                f"metadata contains duplicate sample group/bin: {group}, bin {bin_number}"
            )
        metadata_keys.add(key)
        groups.setdefault(group, set()).add(bin_number)
        if args.cell_counts:
            value = cell_counts.get((group, bin_number))
            if value is None:
                raise InputFormatError(
                    f"cell-count inputs have no value for {group}, bin {bin_number}"
                )
            else:
                row["cell_count"] = value
        if args.barcodes:
            barcode = barcodes.get(row.get("binder_name", "")) or barcodes.get(
                row.get("sample_id", "")
            )
            if barcode is None:
                raise InputFormatError(
                    f"barcode input has no row for {row.get('binder_name')!r}"
                )
            else:
                row["barcode_f"], row["barcode_r"] = barcode
        if args.fluorescence:
            value = fluorescence.get(group, {}).get(bin_number)
            if value is None:
                raise InputFormatError(
                    f"fluorescence inputs have no value for {group}, bin {bin_number}"
                )
            else:
                row["mean_fluorescence"] = value

    expected_keys = set(metadata_keys)
    if args.cell_counts:
        unexpected = set(cell_counts) - expected_keys
        if unexpected:
            raise InputFormatError(
                "cell-count inputs contain sample group/bin combinations absent from "
                f"metadata: {sorted(unexpected)}"
            )
    for group, bins in groups.items():
        require_complete_bins(bins, f"metadata sample group {group}")
    if args.fluorescence:
        unexpected = {
            (group, bin_number)
            for group, values in fluorescence.items()
            for bin_number in values
        } - expected_keys
        if unexpected:
            raise InputFormatError(
                "fluorescence inputs contain sample group/bin combinations absent from "
                f"metadata: {sorted(unexpected)}"
            )

    destination = output_path(args.metadata)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {destination}")
if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, InputFormatError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
