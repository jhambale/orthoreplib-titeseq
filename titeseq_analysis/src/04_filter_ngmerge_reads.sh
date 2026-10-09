#!/bin/bash

######################################################

# This script performs the following:
# i) FastQC - extracts quality metrics from raw fastq files
# ii) Fastp - performs quality filtering and trimming on raw reads
# iii) NGmerge - merges quality filtered / trimmed forward and reverse reads
# iv) Seqkit - extracts general summary statistics for sequencing files
# v) minimap2 - performs alignment to a reference sequence
# outputs are written to ../../manzanaa_data/manzanaa_outputs directory

### Arguments
## $1: metadata in .txt format (tab separated)
# headers should include:
# sample_number (e.g. 1); experiment_name (e.g. jh-001);
# sample_name (e.g. lenient-171-246-376_R1_001.fastq.gz) - should match .fastq.gz file names;
# barcode_f (e.g., atgc); barcode_r (e.g., atgc)

## $2: experiment_name given explicity as a separate argument. should match the experiment name given in metadata

## $3+ list of fastq.gz files to be run through (loaded as wildcard)
# Only specify one of each read pair; the other
# filename is assumed (R2 if R1, or R1 if R2.)
# Example 1: bash 04_filter_ngmerge_reads.sh \
# ../../minibinders_orthorep_data/raw/jh_024/demultiplex/jh_024_metadata.txt \
# jh_024 \
# ../../minibinders_orthorep_data/raw/jh_024/demultiplex/*_R1_001.fastq.gz
#
### OUTPUTS
## md5sum
# ${output_dir_prefix}/demultiplex/md5sum_check.txt
#
##FastQC
## $fastqc_out_path.html : fastqc report -- web link
## $fastqc_out_path.zip : fastqc report -- detailed files
#
## Fastp
## ${fastp_out_path}_out_R1.fastq.gz : quality filtered forward read
## ${fastp_out_path}_out_R2.fastq.gz  : quality filtered reverse read
#
## NGmerge
## ${pear_out_path}_out_merged.assembled.fastq : merged fastq file that passed
# qc/qa filters
#
## Seqkit
## ${seqkit_stats_out_path}_filtering_stats.txt: statistics on number of reads
# containing barcode

## minimap2
## ${alignment_out_path}.bam: alignment file

## samtools
## ${alignment_out_path}_unmapped.fastq: set of reads that did not map to reference fasta

## NOTES
# this script assumes the file name ends in R[12]_001.fastq.gz
# any string is allowed before R[12], as long as the forward and reverse read
# prefix is the same

######################################################


METADATA=$1
experiment_name=$2
shift 2
# REFLOC=$1
# shift

# experiment_name=$3
echo $experiment_name

# Derive the data root from the metadata file path.
# The metadata is expected to live at: <data_root>/raw/<experiment_name>/demultiplex/<file>.txt
# So the data root is three levels above the directory containing the metadata file.
METADATA_DIR=$(dirname "$METADATA")
DATA_ROOT=$(cd "$METADATA_DIR/../../.." && pwd)

# check for appropriate download of fastq files through md5check
md5sum $@ > "$DATA_ROOT/raw/$experiment_name/demultiplex/md5sum_check.txt"
if ! md5sum -c "$DATA_ROOT/raw/$experiment_name/demultiplex/md5sum_check.txt" > "$DATA_ROOT/raw/$experiment_name/demultiplex/md5_check.log" 2>&1; then
    echo "Error: corrupted file download."
    cat "$DATA_ROOT/raw/$experiment_name/demultiplex/md5_check.log"
    exit 1
fi



# check to make sure files have the expected formats
for fastq in $@
do
	if [ -z $(basename $fastq | grep -i .fastq) ]
	then
		echo $(basename $fastq) "does not have .fastq suffix - aborting"
		exit 1
	fi

done
echo "all files are the correct format. continuing..."

# loop through all fastq files and extract file names and metadata required for
# writing outputs to correct folder
for fastq in "$@"
do
	fname=$(basename $fastq)
	dname=$(dirname $fastq)
	fpath=$dname/${fname%_R[12]_001.fastq*}
	experiment_name=$(cat $METADATA | grep $fname | cut -f2)
	reference_name=$(cat $METADATA | grep $fname | cut -f14)
    reference_size=$(cat $METADATA | grep $fname | cut -f15)
	flank_f=$(cat $METADATA | grep $fname | cut -f12)
	flank_r=$(cat $METADATA | grep $fname | cut -f13)

  echo Experiment name $experiment_name
    fastqc_out_path="$DATA_ROOT/outputs/$experiment_name/fastqc_output/"
    fastp_out_path="$DATA_ROOT/outputs/$experiment_name/fastp_output/${fname%_R[12]_001.fastq*}"
    pear_out_path="$DATA_ROOT/outputs/$experiment_name/merged_reads/${fname%_R[12]_001.fastq*}"
	seqkit_stats_out_path="$DATA_ROOT/outputs/$experiment_name/seqkit_stats/${fname%_R[12]_001.fastq*}"
	seqkit_filtered_read_variants_out_path="$DATA_ROOT/outputs/$experiment_name/merged_variants_filtered_reads/${fname%_R[12]_001.fastq*}"
	alignment_out_path="$DATA_ROOT/outputs/$experiment_name/alignments/${fname%_R[12]_001.fastq*}"
    unmapped_out_path="$DATA_ROOT/outputs/$experiment_name/alignments/unmapped/${fname%_R[12]_001.fastq*}"
	reference_path="$DATA_ROOT/raw/$experiment_name/references/${reference_name%.fasta}"


	# define prefix for all output directories
	output_dir_prefix="$DATA_ROOT/outputs/$experiment_name"
	echo $output_dir_prefix

	# define log file path for capturing all console output
	log_file="$output_dir_prefix/04_filter_ngmerge_reads_console_output.txt"
	if [ ! -e ${pear_out_path}_out_merged.assembled.fastq ]
	then

	  # create output directories if they don't exist already
		if [[ "$fastq" == "$1" ]]
			then

				# create general output folder for current experiment
				if [ ! -d "$output_dir_prefix" ]
			  then
					echo Making directory $output_dir_prefix
			    mkdir $output_dir_prefix
				fi

				# output directory for fastqc results
				if [ ! -d "$output_dir_prefix/fastqc_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastqc_output/
			    mkdir $output_dir_prefix/fastqc_output/
				fi

				# output directory for fastp results
	      if [ ! -d "$output_dir_prefix/fastp_output/" ]
			  then
					echo Making directory $output_dir_prefix/fastp_output/
			    mkdir $output_dir_prefix/fastp_output/
				fi
				# output directory for fastp results
	      if [ ! -d "$output_dir_prefix/alignments/" ]
			  then
					echo Making directory $output_dir_prefix/alignments/
			    mkdir $output_dir_prefix/alignments/
                mkdir $output_dir_prefix/alignments/unmapped/
				fi

				# output directory for merged reads
	      if [ ! -d "$output_dir_prefix/merged_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_reads/
			    mkdir $output_dir_prefix/merged_reads/
				fi

				# output directory for merged reads
	      if [ ! -d "$output_dir_prefix/merged_variants_filtered_reads/" ]
			  then
					echo Making directory $output_dir_prefix/merged_variants_filtered_reads/
			    mkdir $output_dir_prefix/merged_variants_filtered_reads/
				fi

				# output directory for seqkit stats
	      if [ ! -d "$output_dir_prefix/seqkit_stats/" ]
			  then
					echo Making directory $output_dir_prefix/seqkit_stats/
			    mkdir $output_dir_prefix/seqkit_stats/
				fi

				# Initialize log file with timestamp and module versions
				{
					echo "=========================================="
					echo "Script: 04_filter_ngmerge_reads.sh"
					echo "Started: $(date)"
					echo "=========================================="
					echo ""
					echo "--- Module Versions ---"
					echo "FastQC version:"
					fastqc --version 2>&1 || echo "FastQC not found"
					echo ""
					echo "Fastp version:"
					fastp --version 2>&1 || echo "Fastp not found"
					echo ""
					echo "NGmerge version:"
					NGmerge --version 2>&1 || echo "NGmerge not found"
					echo ""
					echo "Seqkit version:"
					seqkit version 2>&1 || echo "Seqkit not found"
					echo ""
					echo "Minimap2 version:"
					minimap2 --version 2>&1 || echo "Minimap2 not found"
					echo ""
					echo "Samtools version:"
					samtools --version 2>&1 || echo "Samtools not found"
					echo ""
					echo "=========================================="
					echo "Experiment: $experiment_name"
					echo "Output Directory: $output_dir_prefix"
					echo "=========================================="
					echo ""
				} > "$log_file"
			fi

		 if [[ "$reference_name" == "pjh3_nbonly" ]]
		 then
			 echo "Nanobody is: pJH3."
             echo reference size is $reference_size
			 min_merge_len=250
			 max_merge_len=520

   #  	 elif [[ "$reference_name" == "mb317" ]]
   #  		 then
   #  			 echo "Nanobody is: mb317."
   #  			 #expected amplicon is 193bp -- allowing for +/- 3bp error
   #  			 min_merge_len=229
   #  			 max_merge_len=249

   #  	 elif [[ "$reference_name" == "mb376" ]]
   #  	 then
   #  		 echo "Nanobody is: mb376."
   #  		 #expected amplicon is 193bp -- allowing for +/- 3bp error
   #  		 min_merge_len=268
   #  		 max_merge_len=288

   #  	 else
   #  		 echo $reference_name "Minibinder must be: mb171, mb317, or mb376."
   #  		 exit 1
    	 fi

      # run fastqc on forward and reverse reads
	  echo "Processing FastQC for $fname" | tee -a "$log_file"
	  fastqc ${fpath}_R1_001.fastq.gz -o $fastqc_out_path 2>&1 | tee -a "$log_file"
	  fastqc ${fpath}_R2_001.fastq.gz -o $fastqc_out_path 2>&1 | tee -a "$log_file"

	  # run fastp (with default settings) on forward and reverse reads
		# save quality report
	  echo "Processing Fastp for $fname" | tee -a "$log_file"
	  fastp \
	      -i ${fpath}_R1_001.fastq.gz \
	      -I ${fpath}_R2_001.fastq.gz \
	      -o ${fastp_out_path}_out_R1.fastq.gz \
	      -O ${fastp_out_path}_out_R2.fastq.gz \
	      -j ${fastp_out_path}.json \
	      -h ${fastp_out_path}.html \
          -q 30 \
          -u 15 \
          --length_required 50 2>&1 | tee -a "$log_file"
	#
	  # run pear (i.e., merge) on quality filtered forward and reverse reads

      # let min_merge_len=$reference_size-10
      # let max_merge_len=$reference_size+10

	 #  pear \
	 #      -f ${fastp_out_path}_out_R1.fastq.gz \
	 #      -r ${fastp_out_path}_out_R2.fastq.gz \
	 #      -n $min_merge_len \
  #         -m $max_merge_len \
	 #      -o ${pear_out_path}_out_merged \
		#   -j 7 \

		# # remove additional pear output files that will not be used
	 #  rm ${pear_out_path}_out_merged.unassembled.forward.fastq
	 #  rm ${pear_out_path}_out_merged.unassembled.reverse.fastq
	 #  rm ${pear_out_path}_out_merged.discarded.fastq

      NGmerge \
          -1 "${fastp_out_path}_out_R1.fastq.gz" \
          -2 "${fastp_out_path}_out_R2.fastq.gz" \
          -o "${pear_out_path}_out_ngmerged.assembled.fastq" \
          -n 30 \
          -f ${pear_out_path}_out_ngmerged.unassembled.fastq \
          -v > ${pear_out_path}_out_ngmerge_stats.txt 2>&1

      echo "NGmerge completed. Log excerpt:" | tee -a "$log_file"
      cat ${pear_out_path}_out_ngmerge_stats.txt | tee -a "$log_file"

      gunzip "${pear_out_path}_out_ngmerged.assembled.fastq"
  fi

	### FILTER FASTQ FILES USING SEQKIT
	echo Beginning filtering for $fname | tee -a "$log_file"

	# extract reads that contain the desired barcode configuration
	# (e.g., attcNNNNNNNNNcAattcNNNNNNNNNcAATT for doubles)
	echo "Processing Seqkit for $fname" | tee -a "$log_file"
	cat ${pear_out_path}_out_ngmerged.assembled.fastq | seqkit stats > ${seqkit_stats_out_path}_filtering_stats.txt 2>&1
	cat ${seqkit_stats_out_path}_filtering_stats.txt | tee -a "$log_file"
	cat ${pear_out_path}_out_ngmerged.assembled.fastq | \
	    seqkit amplicon -m 0 -P -s -F $flank_f -R $flank_r -r 21:-21 --bed \
		> ${seqkit_filtered_read_variants_out_path}_variants.txt 2>&1
	echo "Seqkit filtering completed" | tee -a "$log_file"
	if ! gzip -f -- "${seqkit_filtered_read_variants_out_path}_variants.txt"; then
		echo "ERROR: Failed to gzip ${seqkit_filtered_read_variants_out_path}_variants.txt" | tee -a "$log_file"
		exit 1
	fi
	# rm ${pear_out_path}_out_merged.assembled.fastq
	# bowtie2 \
	#       --sensitive-local \
	#       -x $reference_path \
	#       -U ${pear_out_path}_out_ngmerged.assembled.fastq \
	#       --no-unal \
	#       -p 7 | samtools view -bS - > ${alignment_out_path}.bam

    echo "Processing Minimap2 and Samtools for $fname" | tee -a "$log_file"

    minimap2 \
    -ax sr \
    -t 7 \
    -B 2 \
    -O 2,24 \
    -k 11 \
    $reference_path.fasta \
    ${pear_out_path}_out_ngmerged.assembled.fastq 2>> "$log_file" | samtools view -bS - 2>> "$log_file" > ${alignment_out_path}.bam

    gzip ${pear_out_path}_out_ngmerged.assembled.fastq

    if [ ! -f "${alignment_out_path}.bam" ] || [ ! -s "${alignment_out_path}.bam" ]; then
        echo "ERROR: BAM file creation failed or is empty" | tee -a "$log_file"
        exit 1
    fi
    echo "Minimap2 alignment completed successfully" | tee -a "$log_file"

    samtools sort -o ${unmapped_out_path}.sorted.bam ${alignment_out_path}.bam 2>> "$log_file"
    if [ ! -f "${unmapped_out_path}.sorted.bam" ]; then
        echo "ERROR: Sorted BAM file creation failed" | tee -a "$log_file"
        exit 1
    fi

    samtools index ${unmapped_out_path}.sorted.bam 2>> "$log_file"
    echo "Samtools sort and index completed successfully" | tee -a "$log_file"

    samtools stats ${unmapped_out_path}.sorted.bam > ${unmapped_out_path}_sortedstats.txt 2>> "$log_file"
    echo "Samtools stats saved" | tee -a "$log_file"

    samtools view -b -f 4 ${unmapped_out_path}.sorted.bam > ${unmapped_out_path}_unmapped.bam 2>> "$log_file"
    samtools fastq ${unmapped_out_path}_unmapped.bam > ${unmapped_out_path}_unmapped.fastq 2>> "$log_file"

    if [ ! -f "${unmapped_out_path}_unmapped.fastq" ] || [ ! -s "${unmapped_out_path}_unmapped.fastq" ]; then
        echo "WARNING: Unmapped FASTQ file is empty or missing" | tee -a "$log_file"
    else
        echo "Samtools unmapped reads extracted successfully" | tee -a "$log_file"
        echo "Processing FastQC on unmapped reads for $fname" | tee -a "$log_file"
        fastqc ${unmapped_out_path}_unmapped.fastq -o "$DATA_ROOT/outputs/$experiment_name/alignments/unmapped/" 2>> "$log_file"
    fi


			echo _____________new sample____________________________ | tee -a "$log_file"

done

# ============================================================
# POST-LOOP: Pool NGmerge outputs and generate rank-abundance
# ============================================================
echo "" | tee -a "$log_file"
echo "========== POOLING AND RANK ABUNDANCE ANALYSIS ==========" | tee -a "$log_file"

pooled_dir="$DATA_ROOT/outputs/$experiment_name/pooled_reads"
mkdir -p "$pooled_dir"

declare -A selected_sample_prefixes=()
for input_fastq in "$@"; do
    input_basename=$(basename -- "$input_fastq")
    input_prefix=${input_basename%_R[12]_001.fastq*}
    selected_sample_prefixes["$input_prefix"]=1
done

declare -a pool_keys=()
declare -A pool_labels=()
declare -A pool_concentrations=()
declare -A pool_flanks_f=()
declare -A pool_flanks_r=()
declare -A pool_fastqs=()

while IFS=$'\t' read -r sample_number metadata_experiment sample_id binder_name concentration bin replicate cell_count mean_fluorescence barcode_f barcode_r metadata_flank_f metadata_flank_r reference reference_size; do
    [[ "$sample_number" == "sample_number" ]] && continue
    [[ "$metadata_experiment" == "$experiment_name" ]] || continue

    sample_prefix=${sample_id%_R[12]_001.fastq*}
    [[ -n "${selected_sample_prefixes[$sample_prefix]+selected}" ]] || continue

    group_label=${binder_name%%_*}
    pool_key="${replicate}|${concentration}"
    assembled_fastq="$DATA_ROOT/outputs/$experiment_name/merged_reads/${sample_prefix}_out_ngmerged.assembled.fastq.gz"

    if [[ ! -s "$assembled_fastq" ]]; then
        echo "ERROR: Missing NGmerge FASTQ for metadata sample $sample_id: $assembled_fastq" | tee -a "$log_file"
        exit 1
    fi

    if [[ -z "${pool_fastqs[$pool_key]+present}" ]]; then
        pool_keys+=("$pool_key")
        pool_labels["$pool_key"]=$group_label
        pool_concentrations["$pool_key"]=$concentration
        pool_flanks_f["$pool_key"]=$metadata_flank_f
        pool_flanks_r["$pool_key"]=$metadata_flank_r
    elif [[ "${pool_labels[$pool_key]}" != "$group_label" ||
            "${pool_flanks_f[$pool_key]}" != "$metadata_flank_f" ||
            "${pool_flanks_r[$pool_key]}" != "$metadata_flank_r" ]]; then
        echo "ERROR: Inconsistent group label or flanks for replicate $replicate at ${concentration}nM" | tee -a "$log_file"
        exit 1
    fi

    pool_fastqs["$pool_key"]+="$assembled_fastq"$'\n'
done < "$METADATA"

if [[ ${#pool_keys[@]} -eq 0 ]]; then
    echo "ERROR: No metadata samples matched the supplied FASTQ files and experiment $experiment_name" | tee -a "$log_file"
    exit 1
fi

for pool_key in "${pool_keys[@]}"; do
    group_label=${pool_labels[$pool_key]}
    concentration=${pool_concentrations[$pool_key]}
    pool_name="${group_label}_${concentration}nM"
    pooled_fastq="$pooled_dir/${pool_name}_pooled.fastq"
    pooled_bed="$pooled_dir/${pool_name}_pooled_variants.bed"
    rank_abundance_raw="$pooled_dir/rank_abundance_${pool_name}.txt"
    rank_abundance_final="$pooled_dir/rank_abundance_${pool_name}_sums.txt"

    echo "Pooling replicate $group_label at ${concentration}nM..." | tee -a "$log_file"
    : > "$pooled_fastq"
    while IFS= read -r assembled_fastq; do
        [[ -n "$assembled_fastq" ]] || continue
        if ! gzip -cd -- "$assembled_fastq" >> "$pooled_fastq"; then
            echo "ERROR: Failed to decompress NGmerge FASTQ: $assembled_fastq" | tee -a "$log_file"
            exit 1
        fi
    done <<< "${pool_fastqs[$pool_key]}"
    echo "Pooled FASTQ lines: $(wc -l < "$pooled_fastq")" | tee -a "$log_file"

    echo "Running seqkit amplicon for $pool_name..." | tee -a "$log_file"
    if ! seqkit amplicon -m 0 -P -s \
        -F "${pool_flanks_f[$pool_key]}" -R "${pool_flanks_r[$pool_key]}" -r 21:-21 --bed \
        "$pooled_fastq" > "$pooled_bed" 2>&1; then
        echo "ERROR: Seqkit amplicon failed for $pool_name; see $pooled_bed" | tee -a "$log_file"
        exit 1
    fi
    echo "Pooled BED records: $(wc -l < "$pooled_bed")" | tee -a "$log_file"

    echo "Building rank-abundance table for $pool_name..." | tee -a "$log_file"
    awk '{print $7}' "$pooled_bed" \
        | sort \
        | uniq -c \
        | sort -rn \
        | awk 'BEGIN{OFS="\t"; print "rank\tcount\tsequence"} \
               {print NR, $1, $2}' \
        > "$rank_abundance_raw"

    total_reads=$(awk 'NR>1 {sum+=$2} END {print sum}' "$rank_abundance_raw")
    total_unique=$(awk 'NR>1' "$rank_abundance_raw" | wc -l)
    echo "$pool_name variant reads (post-flank filter): $total_reads" | tee -a "$log_file"
    echo "$pool_name unique sequences: $total_unique" | tee -a "$log_file"

    awk -v total="$total_reads" \
        'BEGIN{OFS="\t"; cumsum=0}
         NR==1 {print $0, "frac_abundance", "cumulative_frac"; next}
         {
           frac = $2 / total;
           cumsum += frac;
           print $0, frac, cumsum
         }' "$rank_abundance_raw" > "$rank_abundance_final"

    echo "Rank-abundance table written to: $rank_abundance_final" | tee -a "$log_file"
    if ! gzip -f -- "$pooled_fastq" "$pooled_bed"; then
        echo "ERROR: Failed to gzip pooled files for $pool_name" | tee -a "$log_file"
        exit 1
    fi
done

echo "========== POOLING COMPLETE ==========" | tee -a "$log_file"

# Final log entry
{
	echo ""
	echo "=========================================="
	echo "Script Completed: $(date)"
	echo "=========================================="
	echo "Full console output saved to:"
	echo "$log_file"
	echo "=========================================="
} >> "$log_file"

echo "Script completed. Full output saved to: $log_file"
