#!/bin/bash
# ============================================================
# 04a_pool_rank_abundance.sh
# POST-LOOP ONLY: pools per-bin NGmerge outputs and generates
# a rank-abundance table per replicate/concentration group
# for downstream threshold analysis in pandas.
#
# Run after 04_filter_ngmerge_reads.sh has completed successfully.
#
# Arguments:
#   $1  metadata .txt (tab-separated, same file used in main script)
#   $2  experiment_name (must match metadata column 2)
#   $3+ R1 fastq.gz files passed as wildcard (same as main script)
#
# Example:
#   bash 04a_pool_rank_abundance.sh \
#     ../../titeseq_data/raw/jh_010/demultiplex/jh_010_metadata.txt \
#     jh_010 \
#     ../../titeseq_data/raw/jh_010/demultiplex/*_R1_001.fastq.gz
# ============================================================

METADATA=$1
experiment_name=$2
shift 2

# Derive DATA_ROOT from metadata file path
# Metadata is expected at: <data_root>/raw/<experiment_name>/demultiplex/<file>.txt
METADATA_DIR=$(dirname "$METADATA")
DATA_ROOT=$(cd "$METADATA_DIR/../../.." && pwd)

# Output directory and log file — appends to existing main script log
output_dir_prefix="$DATA_ROOT/outputs/$experiment_name"
log_file="$output_dir_prefix/04_filter_ngmerge_reads_console_output.txt"

echo "" | tee -a "$log_file"
echo "========== POOLING AND RANK ABUNDANCE ANALYSIS ==========" | tee -a "$log_file"
echo "Started: $(date)" | tee -a "$log_file"
echo "Experiment: $experiment_name" | tee -a "$log_file"
echo "DATA_ROOT: $DATA_ROOT" | tee -a "$log_file"

# Create pooled reads output directory
pooled_dir="$DATA_ROOT/outputs/$experiment_name/pooled_reads"
mkdir -p "$pooled_dir"

# ============================================================
# Step 1: Build set of selected sample prefixes from input FASTQs
# ============================================================
declare -A selected_sample_prefixes=()
for input_fastq in "$@"; do
    input_basename=$(basename -- "$input_fastq")
    input_prefix=${input_basename%_R[12]_001.fastq*}
    selected_sample_prefixes["$input_prefix"]=1
done

echo "Input sample prefixes detected: ${#selected_sample_prefixes[@]}" | tee -a "$log_file"

# ============================================================
# Step 2: Read metadata and group samples into pool keys
# Pool key = replicate|concentration
# All bins sharing a replicate+concentration are pooled together
# ============================================================
declare -a pool_keys=()
declare -A pool_labels=()
declare -A pool_concentrations=()
declare -A pool_flanks_f=()
declare -A pool_flanks_r=()
declare -A pool_fastqs=()

while IFS=$'\t' read -r sample_number metadata_experiment sample_id binder_name \
    concentration bin replicate cell_count mean_fluorescence \
    barcode_f barcode_r metadata_flank_f metadata_flank_r \
    reference reference_size; do

    # Skip header row
    [[ "$sample_number" == "sample_number" ]] && continue
    # Skip rows not belonging to this experiment
    [[ "$metadata_experiment" == "$experiment_name" ]] || continue

    sample_prefix=${sample_id%_R[12]_001.fastq*}
    # Skip rows not matching supplied input FASTQs
    [[ -n "${selected_sample_prefixes[$sample_prefix]+selected}" ]] || continue

    group_label=${binder_name%%_*}
    pool_key="${replicate}|${concentration}"

    assembled_fastq="$DATA_ROOT/outputs/$experiment_name/merged_reads/${sample_prefix}_out_ngmerged.assembled.fastq.gz"

    if [[ ! -s "$assembled_fastq" ]]; then
        echo "ERROR: Missing or empty NGmerge FASTQ for sample $sample_id:" | tee -a "$log_file"
        echo "       Expected: $assembled_fastq" | tee -a "$log_file"
        exit 1
    fi

    if [[ -z "${pool_fastqs[$pool_key]+present}" ]]; then
        # First time seeing this pool key — register it
        pool_keys+=("$pool_key")
        pool_labels["$pool_key"]=$group_label
        pool_concentrations["$pool_key"]=$concentration
        pool_flanks_f["$pool_key"]=$metadata_flank_f
        pool_flanks_r["$pool_key"]=$metadata_flank_r
    elif [[ "${pool_labels[$pool_key]}" != "$group_label" ||
            "${pool_flanks_f[$pool_key]}" != "$metadata_flank_f" ||
            "${pool_flanks_r[$pool_key]}" != "$metadata_flank_r" ]]; then
        echo "ERROR: Inconsistent group label or flanks for pool key $pool_key" | tee -a "$log_file"
        echo "       Existing label: ${pool_labels[$pool_key]}, new: $group_label" | tee -a "$log_file"
        exit 1
    fi

    # Append this bin's assembled FASTQ path to the pool
    pool_fastqs["$pool_key"]+="$assembled_fastq"$'\n'

done < "$METADATA"

if [[ ${#pool_keys[@]} -eq 0 ]]; then
    echo "ERROR: No metadata rows matched the supplied FASTQs for experiment $experiment_name" | tee -a "$log_file"
    exit 1
fi

echo "Pool groups identified: ${#pool_keys[@]}" | tee -a "$log_file"
for pk in "${pool_keys[@]}"; do
    echo "  Pool key: $pk  label: ${pool_labels[$pk]}  conc: ${pool_concentrations[$pk]}nM  flank_f: ${pool_flanks_f[$pk]}  flank_r: ${pool_flanks_r[$pk]}" | tee -a "$log_file"
done

# ============================================================
# Step 3: For each pool group, concatenate FASTQs, run seqkit
# amplicon, build rank-abundance table
# ============================================================
for pool_key in "${pool_keys[@]}"; do

    group_label=${pool_labels[$pool_key]}
    concentration=${pool_concentrations[$pool_key]}
    pool_name="${group_label}_${concentration}nM"

    pooled_fastq="$pooled_dir/${pool_name}_pooled.fastq"
    pooled_bed="$pooled_dir/${pool_name}_pooled_variants.bed"
    rank_abundance_raw="$pooled_dir/rank_abundance_${pool_name}.txt"
    rank_abundance_final="$pooled_dir/rank_abundance_${pool_name}_sums.txt"

    echo "" | tee -a "$log_file"
    echo "--- Processing pool: $pool_name ---" | tee -a "$log_file"

    # Initialize empty pooled FASTQ
    : > "$pooled_fastq"

    # Decompress and concatenate all per-bin NGmerge assembled FASTQs
    while IFS= read -r assembled_fastq; do
        [[ -n "$assembled_fastq" ]] || continue
        echo "  Adding: $(basename $assembled_fastq)" | tee -a "$log_file"
        if ! gzip -cd -- "$assembled_fastq" >> "$pooled_fastq"; then
            echo "ERROR: Failed to decompress: $assembled_fastq" | tee -a "$log_file"
            exit 1
        fi
    done <<< "${pool_fastqs[$pool_key]}"

    echo "Pooled FASTQ lines: $(wc -l < "$pooled_fastq")" | tee -a "$log_file"

    # Run seqkit amplicon on pooled FASTQ using flanks from metadata
    echo "Running seqkit amplicon for $pool_name..." | tee -a "$log_file"
    echo "  flank_f: ${pool_flanks_f[$pool_key]}" | tee -a "$log_file"
    echo "  flank_r: ${pool_flanks_r[$pool_key]}" | tee -a "$log_file"

    if ! seqkit amplicon -m 0 -P -s \
        -F "${pool_flanks_f[$pool_key]}" \
        -R "${pool_flanks_r[$pool_key]}" \
        -r 21:-21 --bed \
        "$pooled_fastq" > "$pooled_bed" 2>&1; then
        echo "ERROR: seqkit amplicon failed for $pool_name" | tee -a "$log_file"
        echo "       Check $pooled_bed for seqkit error output" | tee -a "$log_file"
        exit 1
    fi

    echo "Pooled BED records: $(wc -l < "$pooled_bed")" | tee -a "$log_file"

    # Extract col 7 sequences, count, sort by descending abundance
    echo "Building rank-abundance table for $pool_name..." | tee -a "$log_file"
    awk '{print $7}' "$pooled_bed" \
        | sort \
        | uniq -c \
        | sort -rn \
        | awk 'BEGIN{OFS="\t"; print "rank\tcount\tsequence"} {print NR, $1, $2}' \
        > "$rank_abundance_raw"

    total_reads=$(awk 'NR>1 {sum+=$2} END {print sum}' "$rank_abundance_raw")
    total_unique=$(awk 'NR>1' "$rank_abundance_raw" | wc -l)

    echo "$pool_name variant reads (post-flank filter): $total_reads" | tee -a "$log_file"
    echo "$pool_name unique sequences: $total_unique" | tee -a "$log_file"

    # Append fractional abundance and cumulative fraction columns
    awk -v total="$total_reads" \
        'BEGIN{OFS="\t"; cumsum=0}
         NR==1 {print $0, "frac_abundance", "cumulative_frac"; next}
         {
           frac = $2 / total;
           cumsum += frac;
           print $0, frac, cumsum
         }' "$rank_abundance_raw" > "$rank_abundance_final"

    echo "Rank-abundance table written to: $rank_abundance_final" | tee -a "$log_file"

    # Gzip pooled intermediates to recover disk space
    if ! gzip -f -- "$pooled_fastq" "$pooled_bed"; then
        echo "ERROR: Failed to gzip pooled files for $pool_name" | tee -a "$log_file"
        exit 1
    fi

    echo "Pooled intermediates compressed." | tee -a "$log_file"

done

echo "" | tee -a "$log_file"
echo "========== POOLING COMPLETE ==========" | tee -a "$log_file"

{
    echo ""
    echo "=========================================="
    echo "04a_pool_rank_abundance.sh completed: $(date)"
    echo "=========================================="
    echo "Output directory: $pooled_dir"
    echo "Log file: $log_file"
    echo "=========================================="
} | tee -a "$log_file"
