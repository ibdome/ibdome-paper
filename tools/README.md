# GenePy Score Calculator

## Description

`genepy_icbi.py` is a Python script designed to process Variant Effect Predictor (VEP) output files to calculate a **GenePy score**. This score represents the cumulative pathogenic burden of genetic variants within each gene for a given individual. The calculation integrates variant pathogenicity (using CADD scores) and allele frequency (using gnomAD frequencies) to produce a single, interpretable score per gene.

The script is optimized for performance, using parallel processing to handle large volumes of VEP data efficiently.

---

## Features

-   **Gene-Level Scoring:** Aggregates variant effects to a gene-level score for each individual.
-   **Pathogenicity & Rarity Weighting:** The score is weighted by both the predicted deleteriousness of a variant (CADD score) and its rarity in the population (gnomAD allele frequency).
-   **Zygosity Consideration:** The score for homozygous variants is doubled to reflect their increased biological impact.
-   **Parallel Processing:** Utilizes `joblib` to significantly speed up the processing of multiple VEP files.
-   **Flexible Input:** Can recursively find and process VEP files (`_vep.txt.gz`) in specified directories.
-   **Customizable Filtering:** Allows users to choose whether to include all variants or only those in coding regions.

---

## Dependencies

The script requires the following Python libraries:

-   `pandas`
-   `numpy`
-   `joblib`

You can install these dependencies using pip:

```bash
pip install pandas numpy joblib
```

---

## How the GenePy Score is Calculated

The GenePy score for a specific gene in an individual is the sum of scores from all relevant variants found in that gene. The score for each individual variant is calculated as follows:

1.  **CADD Score Scaling:** The raw CADD score (`CADD_RAW`) of each variant is normalized to a value between 0 and 1 (`CADD_scaled`) based on the minimum and maximum `CADD_RAW` values observed across all input files.
    
    $CADD_{scaled} = \frac{CADD_{RAW} - CADD_{RAW,min}}{CADD_{RAW,max} - CADD_{RAW,min}}$
    
2.  **Allele Frequency Weighting:** A frequency weight is calculated from the gnomAD allele frequency (`gnomAD_AF`). This term gives higher weight to rarer variants.
    
    $FrequencyWeight = -\log_{10}((1 - gnomAD_{AF}) \times gnomAD_{AF})$
    
3.  **Variant Score Calculation:** The base score for a variant is the product of its scaled CADD score and its frequency weight.
    
    $VariantScore_{base} = CADD_{scaled} \times FrequencyWeight$
    
4.  **Zygosity Adjustment:** If a variant is homozygous (`HOM`), its base score is doubled.
    
    $VariantScore_{final} = \begin{cases} 2 \times VariantScore_{base} & \text{if Zygosity is HOM} \\ VariantScore_{base} & \text{if Zygosity is HET} \end{cases}$
    

---

## Usage

The script is executed from the command line.

### Arguments

-   `--vep_directories` (Required): A comma-separated list of directory paths to search for VEP files. The search is recursive.
-   `--vep_file_pattern` (Optional): The file name pattern to identify VEP files. Default is `_vep.txt.gz`.
-   `--coding_only` (Optional): Set to `True` to filter for coding variants only, or `False` to include all variants. Default is `True`.
-   `--n_jobs` (Optional): The number of CPU cores to use for parallel processing. Use `-1` to use all available cores. Default is `1`.
-   `--output_file` (Optional): The name for the output file. Default is `genepy_scores.tsv`.

### Example Command

```bash
python genepy_icbi.py \
    --vep_directories /path/to/data/dir1,/path/to/data/dir2 \
    --n_jobs -1 \
    --output_file my_project_genepy_scores.tsv
```

---

## Input File Format

The script expects VEP-annotated text files, which can be gzipped (`.gz`). The files must be tab-separated and contain a header line that starts with `#`. The following columns are required:

-   `#Uploaded_variation`
-   `Location`
-   `Gene`
-   `Consequence`
-   `ZYG` (Zygosity)
-   `SYMBOL`
-   `gnomAD_AF`
-   `CADD_RAW`
-   `IND` (Individual ID)

---

## Output File Format

The script generates a single tab-separated values (`.tsv`) file with the following columns:

-   `Gene`: The Ensembl gene ID. If not available, the genomic location is used.
-   `IND`: The individual's identifier.
-   `SYMBOL`: The gene symbol.
-   `score`: The calculated GenePy score for that gene in that individual.

---

## Reference

This script is based on the methodology described in the following paper:

-   Mossotto, E. et al. GenePy - a score for estimating gene pathogenicity in individuals using next-generation sequencing data. *BMC Bioinformatics* **20**, 254 (2019).

---

## License

This software is distributed under the **MIT License**. See the license text in the script file for more details.
