#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on December 28, 2025
@author: alcantar, modified by vh
"""

# Import this parsing file prior to run

import glob as glob
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from utils_ts import *


def parse_filename(sample_name: str, rep_id_map: dict = None) -> Tuple[str, str]:
    """
    Parse a mutation-analysis CSV filename and return (rep, column_name).

    Two filename formats are supported:

    1. TiteSeq format (produced by scripts 04/05 for jh-style experiments):
           {rep_id}b{bin}_{conc}nMthresh{N}_S{N}_mutation_analysis.csv
       e.g. 55b0_0.64nMthresh6_S120.0_mutation_analysis.csv
            96rptb3_3.2nMthresh4_mutation_analysis.csv
       -> rep_id '55' or '96rpt' looked up in rep_id_map to get 'rep1' or 'rep2',
          column_name '0.64nM_bin0_raw_counts'

       rep_id_map must be supplied for this format: a dict mapping the
       replicate-ID string found in the filename (e.g. '55' or '96rpt') to
       the canonical rep label (e.g. 'rep1').  Build it from the metadata:
           rep_id_map = dict(zip(
               meta_df['binder_name'].str.extract(r'^(.+?)_b')[0],
               'rep' + meta_df['replicate'].astype(str)
           ))

    2. LIB format (legacy):
           LIB..._V2_5_5B1_S12_mutation_analysis*.csv
       V2/V3 => rep2/rep3.  5_5B1 => 10^-5.5 bin1.
       0negB1 => 0M bin1.
    """
    stem = Path(sample_name).stem

    # --- Format 1: TiteSeq  {rep_id}b{bin}_{conc}nM… ---
    # Matches e.g. '55b0_0.64nMthresh6_S120.0' or '96rptb3_3.2nMdiagnostic'
    ts_match = re.match(
        r"^(?P<rep_id>.+?)b(?P<bin>\d+)_(?P<conc>[\d.]+)nM",
        stem,
    )
    if ts_match:
        rep_id = ts_match.group("rep_id")
        bin_no = ts_match.group("bin")
        conc_raw = ts_match.group("conc")

        if rep_id_map is None:
            raise ValueError(
                f"TiteSeq filename detected ('{Path(sample_name).name}') but no "
                f"rep_id_map was supplied. Pass the metadata-derived rep_id_map "
                f"to parse_filename so replicate IDs (e.g. '55', '96rpt') can be resolved."
            )
        if rep_id not in rep_id_map:
            raise ValueError(
                f"rep_id '{rep_id}' from '{Path(sample_name).name}' not found in "
                f"rep_id_map (keys: {list(rep_id_map.keys())})."
            )

        rep = rep_id_map[rep_id]

        # Format concentration: drop trailing '.0' for whole numbers
        conc_float = float(conc_raw)
        conc_str = str(int(conc_float)) if conc_float == int(conc_float) else conc_raw
        column_name = f"{conc_str}nM_bin{bin_no}_raw_counts"
        return rep, column_name

    # --- Format 2: LIB  …_V2_5_5B1_S12_… ---
    stem_parts = stem.split("_")

    # Replicate token: V2 / V3, case-insensitive
    rep_token = next(
        (p for p in stem_parts if p.upper().startswith("V") and p[1:].isdigit()), None
    )
    if not rep_token:
        raise ValueError(
            f"Cannot parse filename — neither TiteSeq nor LIB format matched: {sample_name}"
        )
    rep_number = rep_token[1:]
    if rep_number not in {"2", "3"}:
        raise ValueError(
            f"Unexpected replicate token '{rep_token}' in filename: {sample_name}"
        )
    rep = f"rep{rep_number}"

    # Concentration/bin token: e.g., 5B1, 5_5B1, or 0negB1
    bin_idx = None
    conc_token = None
    bin_no = None
    for i, part in enumerate(stem_parts):
        m = re.match(r"(?P<conc>[0-9]+|0neg)B(?P<bin>[0-9]+)", part)
        if m:
            bin_idx = i
            conc_token = m.group("conc")
            bin_no = m.group("bin")
            break
    if bin_idx is None or conc_token is None or bin_no is None:
        raise ValueError(
            f"Cannot find bin/concentration token in filename: {sample_name}"
        )

    if conc_token.lower() == "0neg":
        concentration = "0M"
    else:
        exponent = conc_token
        if bin_idx > 0 and stem_parts[bin_idx - 1].isdigit():
            exponent = f"{stem_parts[bin_idx - 1]}.{conc_token}"
        concentration = f"10^-{exponent}M"

    column_name = f"{concentration}_bin{bin_no}_raw_counts"
    return rep, column_name
