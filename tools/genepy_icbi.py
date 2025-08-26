#!/usr/bin/env python
# Copyright (c) 2024 Dietmar Rieder

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
import os
import glob
import gzip
import argparse
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

# Global
coding_variants: List[str] = [
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "synonymous_variant",
]

gnomad_individuals: int = 76156
gnomad_alleles: int = gnomad_individuals * 2


def find_vep_files(directories: List[str], pattern: str = "_vep.txt.gz") -> List[str]:
    """
    Finds all files matching a pattern in a list of directories,
    searching recursively in subdirectories.

    Args:
      directories: A list of directory paths to search in.
      pattern: The file pattern to match (e.g., "_vep.txt").

    Returns:
      A list of file paths that match the pattern.
    """
    vep_files: List[str] = []
    for directory in directories:
        for filename in glob.iglob(
            os.path.join(directory, "**", f"*{pattern}"), recursive=True
        ):
            vep_files.append(filename)
    return vep_files


def read_vep_tab(
    file_path: str, cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Reads a VEP tab-separated file, which may be gzipped,
    with comments starting with '##' and a header line starting with '#'.

    Args:
      file_path: Path to the VEP output file.

    Returns:
      A pandas DataFrame containing the VEP data, or None if the header
      line is not found or an error occurs.
    """
    try:
        # Open the file (gzipped or not)
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt") as f:
                lines: List[str] = f.readlines()
        else:
            with open(file_path, "r") as f:
                lines: List[str] = f.readlines()

        # Find the header line (starts with '#')
        header_index: Optional[int] = next(
            (i for i, line in enumerate(lines) if not line.startswith("## ")), None
        )

        # If a header line is found, read the data
        if header_index is not None:
            vep_data: pd.DataFrame = pd.read_csv(
                file_path,
                sep="\t",
                header=0,
                skiprows=range(header_index),
                usecols=cols,
                compression="infer",  # Let pandas handle compression
                low_memory=False,
            )
            return vep_data
        else:
            print("Header line not found in the file.")
            return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def find_min_max_cadd_raw(
    vep_files: List[str], n_jobs: int = 1
) -> Tuple[float, float, np.ndarray]:
    """
    Finds the minimum and maximum "CADD_RAW" values across multiple VEP files.

    Args:
      vep_files: A list of VEP files.

    Returns:
      A tuple containing the minimum and maximum "CADD_RAW" values.
    """

    def process_file(file_path: str) -> Optional[Tuple[float, float, np.ndarray]]:
        """Processes a single file to find min, max, and all CADD_RAW values."""
        try:
            df: Optional[pd.DataFrame] = read_vep_tab(file_path, cols=["CADD_RAW"])
            if df is not None:
                df["CADD_RAW"] = df["CADD_RAW"].replace("-", np.nan)
                df["CADD_RAW"] = pd.to_numeric(df["CADD_RAW"])
                if "CADD_RAW" in df:
                    return (
                        df["CADD_RAW"].min(),
                        df["CADD_RAW"].max(),
                        df["CADD_RAW"].to_numpy(),
                    )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        return None

    # Parallelize file processing
    results: List[Optional[Tuple[float, float, np.ndarray]]] = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file_path) for file_path in vep_files
    )

    # Initialize min, max, and array for all values
    min_cadd_raw: float = float("inf")
    max_cadd_raw: float = float("-inf")
    cadd_raw_all: np.ndarray = np.array([])

    # Aggregate results from all files
    for result in results:
        if result is not None:
            file_min, file_max, file_values = result
            min_cadd_raw = min(min_cadd_raw, file_min)
            max_cadd_raw = max(max_cadd_raw, file_max)
            cadd_raw_all = np.append(cadd_raw_all, file_values)

    return min_cadd_raw, max_cadd_raw, cadd_raw_all


def add_freqs_column(vep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column to the VEP DataFrame with a tuple containing
    modified gnomAD_AF values, using the global gnomad_alleles value.
    The first tuple value is 1 - (the second value).

    Args:
      vep_df: The pandas DataFrame containing the VEP data.

    Returns:
      The modified DataFrame with the new tuple column.
    """

    if "gnomAD_AF" not in vep_df:
        print("Error: 'gnomAD_AF' column not found in the DataFrame.")
        return vep_df

    def calculate_value(row: pd.Series) -> Tuple[float, float]:
        gnomad_af: float = row["gnomAD_AF"]

        if gnomad_af == 0 or pd.isna(gnomad_af):
            second_value: float = 1 / gnomad_alleles
        elif gnomad_af == 1:
            second_value: float = 1 - 1 / gnomad_alleles
        else:
            second_value: float = gnomad_af

        first_value: float = 1 - second_value
        return (first_value, second_value)

    vep_df["freqs"] = vep_df.apply(calculate_value, axis=1)
    return vep_df


def add_score_column(vep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a "score" column to the VEP DataFrame calculated using CADD_scaled,
    the "freqs" column (containing tuples), and the ZYZ column.

    Args:
      vep_df: The pandas DataFrame containing the VEP data.

    Returns:
      The modified DataFrame with the "score" column.
    """

    if (
        "CADD_scaled" not in vep_df.columns
        or "freqs" not in vep_df.columns
        or "ZYG" not in vep_df.columns
    ):
        print(
            "Error: 'CADD_scaled', 'freqs', or 'ZYG' column not found in the DataFrame."
        )
        print(vep_df.columns)
        return vep_df

    def calculate_score(row: pd.Series) -> float:
        base_score: float = row["CADD_scaled"] * (
            -np.log10(row["freqs"][0] * row["freqs"][1])
        )
        if row["ZYG"] == "HOM":
            return 2 * base_score
        else:
            return base_score

    vep_df["score"] = vep_df.apply(calculate_score, axis=1)
    return vep_df


def filter_vep_data(vep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the VEP DataFrame to keep only variants where at least one of the
    comma-separated "Consequence" values is in the coding_variants list.

    Args:
      vep_df: The pandas DataFrame containing the VEP data.

    Returns:
      The filtered DataFrame.
    """

    if "Consequence" not in vep_df:
        print("Error: 'Consequence' column not found in the DataFrame.")
        return vep_df

    def check_consequence(consequence_str: str) -> bool:
        consequences: List[str] = consequence_str.split(",")
        return any(c in coding_variants for c in consequences)

    filtered_df: pd.DataFrame = vep_df[vep_df["Consequence"].apply(check_consequence)]
    return filtered_df


def sum_scores_by_gene_and_individual(vep_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Sums the "score" values by gene (Gene column) and individual (IND column),
    using the Location column if Gene is "-".
    Includes the SYMBOL column in the output.

    Args:
      vep_df: The pandas DataFrame containing the VEP data with a "score" column.

    Returns:
      A new DataFrame with summed scores, indexed by gene and individual.
    """

    if (
        "Gene" not in vep_df
        or "IND" not in vep_df
        or "score" not in vep_df
        or "Location" not in vep_df
        or "SYMBOL" not in vep_df
    ):
        print(
            "Error: 'Gene', 'IND', 'score', 'Location', or 'SYMBOL' column not found in the DataFrame."
        )
        return None

    # Set Gene to Location if Gene is "-"
    vep_df["Gene"] = vep_df.apply(
        lambda row: row["Location"] if row["Gene"] == "-" else row["Gene"], axis=1
    )

    # Group by Gene, IND, and SYMBOL, then sum the scores
    summed_scores: pd.DataFrame = (
        vep_df.groupby(["Gene", "IND", "SYMBOL"])["score"].sum().reset_index()
    )

    return summed_scores


def process_vep_file(
    file_path: str,
    cadd_minmax: Tuple[float, float, np.ndarray],
    coding_only: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Processes a single VEP file, calculates scores, and sums them by gene and individual.

    Args:
      file_path: Path to the VEP file.
      cadd_minmax: Tuple containing the minimum and maximum CADD_RAW values.
      coding_only: Whether to filter for coding variants only.

    Returns:
      A DataFrame with summed scores for the processed file.
    """
    try:
        vep_df: Optional[pd.DataFrame] = read_vep_tab(
            file_path,
            cols=[
                "#Uploaded_variation",
                "Location",
                "Gene",
                "Consequence",
                "ZYG",
                "SYMBOL",
                "gnomAD_AF",
                "CADD_RAW",
                "IND",
            ],
        )
        if vep_df is not None:
            vep_df = vep_df.rename(
                columns={"#Uploaded_variation": "Uploaded_variation"}
            )

            for col in ["CADD_RAW", "gnomAD_AF"]:
                vep_df[col] = vep_df[col].replace("-", np.nan)
                vep_df[col] = pd.to_numeric(vep_df[col])

            vep_df["CADD_scaled"] = (vep_df["CADD_RAW"] - cadd_minmax[0]) / (
                cadd_minmax[1] - cadd_minmax[0]
            )
            vep_df = add_freqs_column(vep_df)
            vep_df = add_score_column(vep_df)
            if coding_only:
                vep_df = filter_vep_data(vep_df)
            scores_summed_df: Optional[
                pd.DataFrame
            ] = sum_scores_by_gene_and_individual(vep_df.copy())
            return scores_summed_df
        else:
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def process_vep_files_parallel(
    directories: List[str],
    pattern: str = "_vep.txt.gz",
    cadd_minmax: Optional[Tuple[float, float, np.ndarray]] = None,
    coding_only: bool = True,
    n_jobs: int = -1,
) -> Optional[pd.DataFrame]:
    """
    Processes multiple VEP files in parallel, calculates scores, and sums them.

    Args:
      directories: A list of directory paths to search for VEP files.
      pattern: The file pattern to match (e.g., "_vep.txt.gz").
      cadd_minmax: Tuple containing the minimum and maximum CADD_RAW values.
      coding_only: Whether to filter for coding variants only.
      n_jobs: Number of CPU cores to use for parallel processing (-1 uses all cores).

    Returns:
      A DataFrame with summed scores for all processed files.
    """

    vep_files: List[str] = find_vep_files(directories, pattern)

    # Use joblib to parallelize the processing of files
    results: List[Optional[pd.DataFrame]] = Parallel(n_jobs=n_jobs)(
        delayed(process_vep_file)(file_path, cadd_minmax, coding_only)
        for file_path in vep_files
    )

    # Concatenate the results from all files
    genepy_scores_df: Optional[pd.DataFrame] = pd.concat(results, ignore_index=True)
    return genepy_scores_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process VEP files and calculate gene scores."
    )
    parser.add_argument(
        "--vep_directories",
        type=str,
        required=True,
        help="Comma-separated list of directories to search for VEP files.",
    )
    parser.add_argument(
        "--vep_file_pattern",
        type=str,
        default="_vep.txt.gz",
        help="Pattern to match VEP file names (default: _vep.txt.gz).",
    )
    parser.add_argument(
        "--coding_only",
        type=bool,
        default=True,
        help="Whether to filter for coding variants only (default: True).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of CPU cores to use for parallel processing (-1 uses all cores, default: 1).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="genepy_scores.tsv",
        help="Name of the output file (default: genepy_scores.tsv).",
    )
    args = parser.parse_args()

    vep_directories_to_search: List[str] = args.vep_directories.split(",")
    vep_file_pattern: str = args.vep_file_pattern
    coding_only: bool = args.coding_only
    n_jobs: int = args.n_jobs
    output_file: str = args.output_file

    vep_files: List[str] = find_vep_files(
        directories=vep_directories_to_search, pattern=vep_file_pattern
    )
    cadd_minmax: Tuple[float, float, np.ndarray] = find_min_max_cadd_raw(
        vep_files=vep_files, n_jobs=n_jobs
    )

    genepy_scores_df: Optional[pd.DataFrame] = process_vep_files_parallel(
        directories=vep_directories_to_search,
        pattern=vep_file_pattern,
        cadd_minmax=cadd_minmax,
        coding_only=coding_only,
        n_jobs=n_jobs,
    )

    if genepy_scores_df is not None:
        # Save the DataFrame to a tab-separated file
        genepy_scores_df.to_csv(output_file, sep="\t", index=False)
        print(f"DataFrame saved to {output_file}")
