#!/bin/bash

######################################################

# This script performs the following:
# combines specified fastqs
# outputs are written to directory where fastqs are stored


# Usage: bash merge_binfastq.sh \
# ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex1_R1_001.fastq.gz
# ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex2_R1_001.fastq.gz
# ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex_merged_R1_001.fastq.gz

######################################################



# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex1_R1_001.fastq.gz ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex2_R1_001.fastq.gz ../../minibinders_orthorep_data/ngs_raw/ex_001/demultiplex/ex_merged_R1_001.fastq.gz"
    exit 1
fi

FASTQ1="$1"
FASTQ2="$2"
MERGED="$3"

# Unzip, merge, and re-gzip
zcat "$FASTQ1" "$FASTQ2" | gzip > "$MERGED"
