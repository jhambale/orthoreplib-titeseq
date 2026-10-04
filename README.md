TITESEQ STACK FOR DRIFTED LIBRARIES

Author: Jacob Hambalek
README Version 1.0. Date 3 OCT 2026

Acknowledgements: Scripts were structured with guidance from Miguel Alcantar, PhD and Yutong Yu, PhD.

This project, including scripts and documentation, were constructed with help from Zed Integrated Agents and Github Copilot.

Once you have installed the TITESEQ stack, you can use the following commands to run the analysis on your drifted libraries. Please ensure that you have all dependencies installed and that your environment is properly configured.

## Data directory layout

The scripts use paths beneath `titeseq_data/`. Keep this layout when preparing an experiment; the `jh_001` raw-input and output directories are empty placeholders in this repository so that files are placed in the expected locations. Raw inputs belong under `titeseq_data/raw/<experiment>/` (for example, FASTQ files and metadata under `demultiplex/`, reference FASTA files under `references/`, and cell-count inputs under `cell_ct_raw/`). Git tracks the empty directory scaffold only; input files are ignored.

The repository includes only the `jh_001` example under `raw/` and `outputs/`, with the expected subfolder names below it. These directories are empty in a fresh clone. Raw input files and generated output files are ignored. Place input data in the appropriate `raw/jh_001/` subfolder; `outputs/jh_001/` is for analysis products.

```text
titeseq_data/
├── raw/
│   └── jh_001/
│       ├── cell_ct_out/
│       ├── cell_ct_raw/
│       ├── demultiplex/
│       │   └── jh_001_metadata.txt
│       ├── demux_in/
│       │   └── jh_001_bc_sample_trace.txt
│       ├── fluor_raw/
│       ├── raw_fastq/
│       └── references/
└── outputs/
    └── jh_001/
        ├── alignments/
        ├── fastp_output/
        ├── fastqc_output/
        ├── library_dfs/
        ├── library_plots/
        ├── merged_reads/
        ├── merged_variants_filtered_reads/
        ├── mutation_dfs/
        ├── normalized_tite_seq_counts/
        ├── normalized_tite_seq_scores/
        ├── raw_tite_seq_counts/
        ├── seqkit_stats/
        ├── tite_seq_Kds/
        ├── ts_library_0.64nM_sens120/
        ├── ts_library_3.2nM_sens120/
        ├── ts_library_16nM_sens120/
        ├── ts_library_80nM_sens120/
        ├── ts_library_400nM_sens120/
        └── ts_library_2000nM_sens120/
```

Git tracks only these two example input text files, the ignore rules, and empty `.gitkeep` placeholders needed to preserve the remaining directories. FASTQ data and all generated output data are ignored.

Note that the subfolders in the 'outputs/' directory are created by the scripts and will not exist until the scripts are run. The subfolders in the 'raw/' directory must be created manually and populated with the appropriate input data before running the scripts.

Starting with your input data:

The critical data will be a set of FASTQ files containing the sequencing reads from your drifted libraries. Ensure that the files are named appropriately and are located in the specified input directory.

It is possible that the FASTQ files received contain reads from multiple bins within the same library. If this is the case, you will need to demultiplex the reads before proceeding with the analysis. The github repository "demux-amplicon-editing" (https://github.com/daspacio9/demux-amplicon-editing), authored by daspacio9, provides a useful tool for demultiplexing amplicon sequencing data. Please refer to the repository's documentation for instructions on how to use the demultiplexing tool. It will output separate FASTQ files for each input sample (forward and reverse reads required). Then, use script 03_demux_bc_rename.py to rename the demultiplexed files according to the TITESEQ naming convention (as guided by the example jh_001_bc_sample_trace.txt). This will ensure that the downstream analysis scripts can correctly identify and process the reads from each library.

There is a whole host of associated data that is essential for running the analysis. This includes reference sequences, cell counts, fluorescence values, and a metadata text file. Ensure that these files are placed in the appropriate directories as specified in the data directory layout. An example metadata file is included within raw/jh_001/demultiplex/jh_001_metadata.txt. This file contains the necessary information for the analysis scripts to correctly interpret the data. Please refer to this example when preparing your own metadata file.

The following are instructions for how to organize the associated data files:
"fluor_raw/" should contain the raw fluorescence values (each bin's geometric mean is key value) for each sorting condition (4 bins per concentration). These files are directly exported from the FACS instrument (see instrument for how to access and export).
"cell_count_raw/" should contain the raw cell counts for each sorting condition (4 bins per concentration). These files are directly exported from the FACS instrument (see instrument for how to access and export).

Some cell sorting experiments can lead to an uneven number of events counted for each bin. This often arises on Sony SH800 sorters that sort the bins sequentially rather than simultaneously. Script 01_cellcount_norm.py will normalize the cell counts to ensure that the total number of events is equal across all bins (adjusting proportionally to the highest event count). Note that this does not replace physically sorting a comparable number of events per bin (i.e. try to keep the correction ratio to less than 10%). The resultant normalized cell counts will be used in the downstream analysis to calculate the TITE-Seq scores and Kd values. The normalized cell counts will be saved in the "cell_ct_out/" directory.

Once the desired cell counts and fluorescence values are in place in the appropriate directories, use script 01a_metadatafill.py to populate the metadata file with the cell count and fluorescence data for each bin. The example metadata file included in the repository (raw/jh_001/demultiplex/jh_001_metadata.txt) shows the expected format and structure of the metadata file. Ensure that your metadata file follows this format to avoid errors during analysis.

Before proceeding, ensure that the appropriate reference FASTA files are present in the "references/" directory. These reference sequences will be used for aligning the merged reads and identifying mutations.

Once the metadata file is generated using 01a_metadatafill.py, the desired FASTA files are in the reference directory, and the appropriately named FASTQ files are in the demultiplex directory, you may run script 04_filter_ngmerge_reads.sh. This bash script will analyze, filter the reads by quality score, and merge the paired-end reads. The merged reads will then be compared to the reference fasta sequences to identify mutations. This output (.bam) will be stored in the "alignments/" directory.

Once the .bam files have been generated, the next script to run is 05_mutation_analysis.py. This script will analyze the .bam data to identify variants and how frequently they appear, storing the variants in each bin as a mutation dataframe (1 mutation csv per bin). The mutation dataframe will be saved in the "mutation_dfs/" directory. The filtering for the .bam can be adjusted (i.e. read threshold, percentile of total reads) using the modified script 05b_mutation_analysis_diag.py. This modified script outputs both the desired filtered .csv and a report (.png of graphs) describing where different filtering cutoffs would exist based on read count distribution. See the description at the top of the script for more information.

This next section describes how to build a library from a drift experiment (i.e. the composition of library sequences is unknown). The approach adopted here defines the library as the set of sequences seen at every concentration for each replicate. There are two scripts (used sequentially) that search for these sequences.

First, the .csv files for each concentration output from script '05_' will be used as inputs for script 06_nd_analysis_oneconc.py. This script must be run from the command line for each concentration tested. This will calculate the single concentration rank ordered normalized scoring (known as "sort-seq"). A separate subfolder will be generated within output/jh_XXX for each concentration, providing in depth analysis of the normalized sequence scores, groupings, mutational distances, etc. Additionally, and more importantly, the script will place a copy of the .csv file for each concentration in the "library_dfs/" directory. These .csv files will be used as inputs for the next script.

Script 07_create_raw_tite_seq_counts.py will take the .csv files from the "library_dfs/" directory and combine them into a single dataframe per replicate that contains all sequences seen across all concentrations. This dataframe will be saved in the "raw_tite_seq_counts/" directory. The output of this script is one .csv file per replicate that contains the raw sequencing counts of each variant across all concentrations.

Once the raw counts for the library replicates have been generated, normalizing the counts and establishing fluorescence scores is the next step. The replicate csv files output from script 07 are used as the inputts for script 08_create_normalized_tite_seq_counts.py. This script will normalize the counts for each variant by cell fraction and total read count, known as the "normalized relative abundance" of each variant at each concentration (see the following reference for more information on the scoring approach: Angenent-Mari, N.M., Garruss, A.S., Soenksen, L.R. et al. A deep learning approach to programmable RNA switches. Nat Commun 11, 5057 (2020). https://doi.org/10.1038/s41467-020-18677-1). The cell fraction is drawn from the number of cells sorted input in the metadata text file.

The normalized relative abundance values are then multiplied by the geometric mean of the fluorescence values for each bin to generate a normalized TITE-Seq score for each variant at each concentration. This is enacting by running script 09_create_normalized_tite_seq_scores.py. The output of this script is one .csv file per replicate that contains the normalized TITE-Seq scores of each variant across all concentrations. These files will be saved in the "normalized_tite_seq_scores/" directory.

It is from these fluorescent score csv files that the Kd values are calculated. The script 10_compute_Kds.py will take the normalized TITE-Seq scores and fit a curve to the data to calculate the Kd value for each variant. There is a series of checks undertaken (amplitude change, confidence in the fit) to determine the quality of the function fit. Variants are either assigned a PASS if all criteria are met or FAIL if any one of the criteria are not met.
The outputs of this script (all existing within the "tite_seq_Kds/" sub folder) are as follows:
- A .csv file containing the Kd values for all variants across each replicate.
- A .csv file containing the Kd values for only the passed EC50 variants across each replicate.
- A .csv file containing the Kd values for only variants which failed due to low amplitude gain.
- EC50 correlation plots (as .png files) comparing the EC50 for each replicate after each successive step of filtering (i.e. all variants, remove low amplitude, remove poor fit, remove EC50 values outside the tested concentration range).
- A subfolder containing the curve fits for each variant (as .png files) for each variant separated into 'passed' and 'failed' groups.

This product is the key output of the pipeline, and any subsequent scripts are just additionall analysis on the Kd values. A brief description of each script is as follows:
- 11_unique_muts.py: identifying point mutations and their frequencies above a user set cutoff
- 12_failed_var_diagnosis.py: assesses failed variants across multiple experiments and prepares some key characterisitics for each variant (i.e. total sequencing reads, replicate agreement, etc.)
- 13_ts_1conc_map.py: compares the distribution of EC50 scores to the normalized scores at a single concentration within the experiment in the form of a quintile-quintile plot. Further examines which concentration is most informative for the library based on how well the single concentration scores correlate with the EC50 values (i.e. maximizing quintile match).
- 14_cluster_eval_variants.py: performs Gaussian Means clustering on the variants based on their EC50 values, then generates further plots for each clustered group (hamming distance from wild type, pairwise distance between the groups, etc.).
- 15_network_nxrep.py: generates a network graph of the variants based on their mutational distances (nodes and lines) and EC50 values (color fill of the nodes). This script uses the NetworkX library to create a visual representation of the relationships between variants, allowing for easy identification of clusters and outliers. Only the most connected subset of all variants are shown.
- 17_drift_dispersity.py: generates dimensional reduction plots for the all variants and each subcluster (reduction uses PACMAP).
- 18_ec50scatter_interact.py: Generates a local html page of an interactive clustered scatter plot of the EC50 values for passed variants. The user can click any variant and see highlighted all variants 1 amino acid away.
- 19_force_directed_layout.py: another 2D representation of the library variants that adjusts the 2D position of each variant based on mutational distance to neighbors as well as EC50 value (closer indicates greater similarity).

There are also handful of housekeeping scripts containing functions that are called on in other scripts (parse_ts, utils_nd, utils_ts). These must remain within the same directory as the other scripts to ensure proper function.
