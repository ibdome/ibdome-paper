#!/usr/bin/env nextflow

process de_analysis {

  errorStrategy "finish"

  publishDir "${params.resultDir}", mode: "link"

  cpus = params.cpus

  input:
    path sample_sheet
    path raw_counts
    each contrast

  output:
    path out_dir

  script:
    def test = contrast.TEST
    def ref = contrast.REF
    def condcol = contrast.CONDCOL != null ? contrast.CONDCOL : "group"
    def sample_col = params.sample_col != null ? params.sample_col : "sample_id"
    def covariate_formula = params.covariate_formula == "" ? "" : '--covariate_formula="' + params.covariate_formula + '"'
    out_dir = test + "_vs_" + ref
    def remove_batch_effect = params.remove_batch_effect ? "--remove_batch_effect" : ''
    def batch_col = params.batch_col != "" ? "--batch_col=${params.batch_col}" : ''
    
    """
    mkdir -p ${out_dir}
    runDESeq2.R ${sample_sheet} ${raw_counts} \\
      --result_dir=${out_dir} \\
      --sample_col=${sample_col}\\
      --c1=${test} \\
      --c2=${ref} \\
      --condition_col=${condcol} \\
      ${remove_batch_effect} \\
      ${batch_col} \\
      ${covariate_formula} \\
      --plot_title="${test} vs ${ref}" \\
      --n_cpus=${task.cpus} \\
      --fdr_cutoff=0.05 \\
      --save_workspace
    """
}

workflow {
    sample_sheet = Channel.fromPath(params.sample_sheet, checkIfExists:true)
    raw_counts = Channel.fromPath(params.raw_counts, checkIfExists:true)
    contrasts = Channel.fromList(params.contrasts)

    de_analysis(sample_sheet, raw_counts, contrasts)
}