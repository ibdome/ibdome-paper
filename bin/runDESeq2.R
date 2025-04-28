#!/usr/bin/env Rscript
'runDESeq2_ICBI.R

Usage:
  runDESeq2_ICBI.R <sample_sheet> <count_table> --result_dir=<res_dir> --c1=<c1> --c2=<c2> [options]
  runDESeq2_ICBI.R --help

Arguments:
  <sample_sheet>                CSV file with the sample annotations.
  <count_table>                 TSV file with the read counts

Mandatory options:
  --result_dir=<res_dir>        Output directory
  --c1=<c1>                     Contrast level 1 (perturbation). Needs to be contained in condition_col.
  --c2=<c2>                     Contrast level 2 (baseline). Needs to be contained in condition_col.

Optional options:
  --nfcore                      Indicate that the input samplesheet is from the nf-core RNA-seq ppipeline.
                                Will merge entries from the same sample and infer the sample_id from `group` and `replicate`.
                                If set, this option overrides `sample_col`.
  --condition_col=<cond_col>    Column in sample annotation that contains the condition [default: group]
  --sample_col=<sample_col>     Column in sample annotation that contains the sample names
                                (needs to match the colnames of the count table). [default: sample]
  --paired_grp=<paired_grp>     Column that conatins the name of the paired samples, when dealing with
                                paired data.
  --remove_batch_effect         Indicate that batch effect correction should be applied [default: FALSE]
                                If batch effect correction should be performed, a batch column is needed in the
                                samplesheet (see also --batch_col below)
  --batch_col=<batch_col>       Optional: column in sample annotation that contains the batch
  --covariate_formula=<formula> Formula to model additional covariates (need to be columns in the samplesheet)
                                that will be appended to the formula built from `condition_col`.
                                E.g. `+ age + sex`. Per default, no covariates are modelled.
  --plot_title=<title>          Title shown above plots. Is built from contrast per default.
  --prefix=<prefix>             Results file prefix. Is built from contrasts per default.
  --fdr_cutoff=<fdr>            False discovery rate for GO analysis and volcano plots [default: 0.1]
  --fc_cutoff=<log2 fc cutoff>  Fold change (log2) cutoff for volcano plots [default: 1]
  --gtf_file=<gtf>              Path to the GTF file used for featurecounts. If specified, a Biotype QC
                                will be performed.
  --gene_id_type=<id_type>      Type of the identifier in the `gene_id` column compatible with AnnotationDbi [default: ENSEMBL]
  --n_cpus=<n_cpus>             Number of cores to use for DESeq2 [default: 1]
  --organism=<human|mouse>      Ensebml annotation db [default: human]
  --save_workspace              Save R workspace for this analysis [default: FALSE]
  --save_init_workspace         Save R workspace before analysis for manual step by step debugging [default: FALSE]
  --save_sessioninfo            Save R sessionInfo() to keep info about library version [default: TRUE]
  --config_file			            Config File used to run the Pipeline, needed for the report [default: "/home/floriani/myScratch/gitlab/nextflow.config"]
' -> doc

library("conflicted")
library("docopt")
arguments <- docopt(doc, version = "0.1")

print(arguments)

library("BiocParallel")
library("DESeq2")
library("IHW")
library("ggplot2")
library("pcaExplorer")
#library("topGO")
#library("clusterProfiler")
#library("ReactomePA")
library("writexl")
library("readr")
library("dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("count", "dplyr")
library("EnhancedVolcano")
library("ggpubr")
library("tibble")
library("stringr")
library("ggrepel")
conflict_prefer("paste", "base")
conflict_prefer("rename", "dplyr")
remove_ensg_version = function(x) gsub("\\.[0-9]*$", "", x)

#### Get parameters from docopt

# Input and output
sampleAnnotationCSV <- arguments$sample_sheet
readCountFile <- arguments$count_table
results_dir = arguments$result_dir
dir.create(results_dir, recursive=TRUE, showWarnings=FALSE)
paired_grp <- arguments$paired_grp

# prefix and plot title
prefix <- arguments$prefix
plot_title <- arguments$plot_title

# Sample information and contrasts
nfcore = arguments$nfcore
cond_col = arguments$condition_col
sample_col = arguments$sample_col
contrast = c(cond_col, arguments$c1, arguments$c2)
gene_id_type = arguments$gene_id_type
covariate_formula = arguments$covariate_formula
remove_batch_effect = arguments$remove_batch_effect
batch_col = arguments$batch_col

# Cutoff
fdr_cutoff = as.numeric(arguments$fdr_cutoff)
fc_cutoff = as.numeric(arguments$fc_cutoff)

# Other
n_cpus = as.numeric(arguments$n_cpus)

# set organism (human or mouse)
organism = arguments$organism

# save R workspace
save_ws = arguments$save_workspace

if (organism == "human") {
    anno_db = "org.Hs.eg.db"
    org_kegg = "hsa"
    org_reactome = "human"
    org_wp = "Homo sapiens"
} else if (organism == "mouse") {
    anno_db = "org.Mm.eg.db"
    org_kegg = "mmu"
    org_reactome = "mouse"
    org_wp = "Mus musculus"
} else {
    msg <- paste0("Organism not implemented: ", organism)
    stop(msg)
}
library(anno_db, character.only = TRUE)


############### Sanitize parameters and read input data
register(MulticoreParam(workers = n_cpus))

if (is.null(plot_title)) {
  plot_title = paste0(contrast[[2]], " vs. ", contrast[[3]])
}
if (is.null(prefix)) {
  prefix = paste0(contrast[[2]], "_", contrast[[3]])
}


allSampleAnno <- read_csv(sampleAnnotationCSV)

sampleAnno <- allSampleAnno %>%
  filter(get(cond_col) %in% contrast[2:3])


# Let's see if we use all samples from samplesheet in DESeq 
sampleSubset = FALSE
if (length(base::setdiff(allSampleAnno[[cond_col]], contrast[2:3])) > 0) {
  sampleSubset = TRUE
}


# Add sample col based on condition and replicate if sample col is not explicitly specified
# and make samplesheet distinct (in case the 'merge replicates' functionality was used).
if(nfcore) {
  sample_col = "sample"
  sampleAnno = sampleAnno %>%
    select(-fastq_1, -fastq_2) %>%
    distinct()
  allSampleAnno = allSampleAnno %>%
    select(-fastq_1, -fastq_2) %>%
    distinct()
}

if (is.null(covariate_formula)) {
  covariate_formula = ""
}
if (remove_batch_effect) {
  if (batch_col %in% names(sampleAnno)) {
    # Convert batches to factors if a batch_col is present
    allSampleAnno[[batch_col]] <- as.factor(allSampleAnno[[batch_col]])
    
    print("Correcting possible batch effects")

    if (! grepl(paste0("+", batch_col), covariate_formula)) {
      covariate_formula = paste0("+", batch_col, covariate_formula)
    }
  } else {
    stop("No batch_col found in sampleSheet, please check")
  }
}
if(is.null(paired_grp)) {
  design_formula <- as.formula(paste0("~", cond_col, covariate_formula))
} else {
  design_formula <- as.formula(paste0("~", paired_grp , " +", cond_col, covariate_formula))
}



count_mat <- read_tsv(readCountFile)
if (gene_id_type == "ENSEMBL") {
  count_mat = count_mat %>% mutate(gene_id= remove_ensg_version(gene_id))
}

ensg_to_genesymbol = count_mat %>% select(gene_id, gene_name)
ensg_to_desc = AnnotationDbi::select(get(anno_db), count_mat$gene_id %>% unique(), keytype = gene_id_type, columns = c("GENENAME")) %>%
  distinct(across(!!gene_id_type), .keep_all = TRUE)

# if we do DESeq on sampleSubset we save also the full count mat for generating a full PCA plot
if (sampleSubset) {
  count_mat_full = count_mat %>%
    select(c(gene_id, allSampleAnno[[sample_col]])) %>%
    column_to_rownames("gene_id") %>%
    round() # salmon does not necessarily contain integers
}

count_mat = count_mat %>%
  select(c(gene_id, sampleAnno[[sample_col]])) %>%
  column_to_rownames("gene_id") %>%
  round() # salmon does not necessarily contain integers



save_plot <- function(filename, p, width=NULL, height=NULL) {
  if (!is.null(width) && !is.null(height)) {
    ggsave(file.path(paste0(filename, ".png")), plot = p, width = width, height = height)
    ggsave(file.path(paste0(filename, ".svg")), plot = p, width = width, height = height)
  } else {
    ggsave(file.path(paste0(filename, ".png")), plot = p)
    ggsave(file.path(paste0(filename, ".svg")), plot = p)
  }
}

################# Start processing
dds <- DESeqDataSetFromMatrix(countData = count_mat,
                              colData = sampleAnno,
                              design = design_formula)

# if we use only a subset of samples for DEseq, make also full dds for a generating a full PCA plot
if (sampleSubset) {
  dds_full <- DESeqDataSetFromMatrix(countData = count_mat_full,
                                colData = allSampleAnno,
                                design = as.formula(paste0("~", cond_col)))

  dds_full <- DESeq(dds_full, parallel = (n_cpus > 1))
}

# count number of detected genes
gene_count <- sapply(
  sampleAnno[[sample_col]], function(s) {
    c <- length(count_mat[[s]][(count_mat[[s]] >10)])
    } 
  ) |>
  enframe() |>
  mutate(group=sampleAnno[[cond_col]][sampleAnno[[sample_col]] == name]) |>
  dplyr::rename(sample=name, genes=value)

p <- ggplot(gene_count, aes(sample, genes, fill=group)) + 
  geom_bar(stat = "identity", color="black") +
  scale_color_brewer(type="qual", palette="Set1") +
  ggtitle("Detected genes") +
  theme_bw()+
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), 
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle=90, vjust=0.5)
        )

save_plot(file.path(results_dir, paste0(prefix, "_number_of_detected_genes")), p, width=10, height=7)
write_tsv(gene_count, file.path(results_dir, paste0(prefix, "_number_of_detected_genes.tsv")))

## keep only genes where we have >= 10 reads in total
# keep <- rowSums(counts(dds)) >= 10

## keep only genes where we have >= 10 reads per samplecondition in total
keep <- rowSums(counts(collapseReplicates(dds, dds[[cond_col]]))) >= 10
dds <- dds[keep,]

# save filtered count file
write_tsv(counts(dds) %>% 
            as_tibble(rownames = "gene_id") %>%
            left_join(ensg_to_genesymbol) %>%
            left_join(ensg_to_desc, by = c("gene_id" = gene_id_type) ) %>%
            rename(genes_description = GENENAME),
          file.path(results_dir, paste0(prefix, "_detectedGenesRawCounts_min_10_reads_in_one_condition.tsv")))

# save normalized filtered count file
dds <- estimateSizeFactors(dds)
write_tsv(counts(dds, normalized=TRUE) %>%
            as_tibble(rownames = "gene_id") %>%
            left_join(ensg_to_genesymbol) %>%
            left_join(ensg_to_desc, by = c("gene_id" = gene_id_type) ) %>%
            rename(genes_description = GENENAME),
          file.path(results_dir, paste0(prefix, "_detectedGenesNormalizedCounts_min_10_reads_in_one_condition.tsv")))

# Set the reference to the contrast level 2 (baseline) given by the --c2 option
dds[[cond_col]] = relevel( dds[[cond_col]], contrast[[3]])

# run DESeq
dds <- DESeq(dds, parallel = (n_cpus > 1))

# get normalized counts
nc <- counts(dds, normalized=T)

### IHW

# use of IHW for p value adjustment of DESeq2 results
resIHW <- results(dds, filterFun=ihw, contrast=contrast)

resIHW <- as.data.frame(resIHW ) |>
  rownames_to_column(var = "gene_id") |>
  as_tibble() |>
  arrange(padj)

resSHRINK  <- lfcShrink(dds, contrast= contrast, type ="normal") #specifying "normal" because "apeglm" need coef instead of contrast. 
resSHRINK <- as.data.frame(resSHRINK) |>
  rownames_to_column(var = "gene_id") |>
  as_tibble() |>
  arrange(padj) |>
  rename(lfcSE_shrink = lfcSE) |>
  rename(log2FoldChange_shrink = log2FoldChange)
 
resIHW <- left_join(resIHW, select(resSHRINK, c(gene_id,log2FoldChange_shrink,lfcSE_shrink)), by="gene_id")
resIHW  <- resIHW |>
  left_join(ensg_to_genesymbol) |>
  left_join(ensg_to_desc, by = c("gene_id" = gene_id_type) ) |>
  rename(genes_description = GENENAME) |>
  arrange(pvalue)
  
summary(resIHW)
sum(resIHW$padj < fdr_cutoff, na.rm=TRUE)
##################### PCA plots

ddsl <- list(contrast = dds)

# run PCA also for full sample set
if (sampleSubset) {
  ddsl <- append(ddsl, list(full = dds_full))
}

bplapply(names(ddsl), function(dds_name) {

  pca_prefix = ifelse(dds_name == "contrast", prefix, "all_samples")

  vsd <- vst(ddsl[[dds_name]], blind=FALSE)
  
  if (remove_batch_effect) {
      intgroup <- c(cond_col, batch_col)
      shape <- batch_col
      if (cond_col == "group"){
        my_aes <- aes(PC1, PC2, color=get(paste0(cond_col,".1")), shape=get(shape))
      } else {
        my_aes <- aes(PC1, PC2, color=get(cond_col), shape=get(shape))
      }
    
    } else {
      intgroup <- c(cond_col)
      my_aes <- aes(PC1, PC2, color=get(cond_col))
    }

  pcaData <- plotPCA(vsd, intgroup=intgroup, returnData=TRUE)
  percentVar <- round(100 * attr(pcaData, "percentVar"))

  p <- ggplot(pcaData, my_aes) +
    geom_point(size=3) +
    xlab(paste0("PC1: ",percentVar[1],"% variance")) +
    ylab(paste0("PC2: ",percentVar[2],"% variance")) +
    geom_text_repel(aes(label=name),vjust=2) +
    scale_color_brewer(type="qual", palette="Set1") +
    labs(colour= cond_col) +
    labs(shape= batch_col) +
    theme_bw()

  save_plot(file.path(results_dir, paste0(pca_prefix, "_PCA")), p, width=10, height=7)

  # PCA plot after removing batch effects
  if (remove_batch_effect) {
    assay(vsd) <- limma::removeBatchEffect(assay(vsd), vsd[[batch_col]])

    pcaData <- plotPCA(vsd, intgroup=intgroup, returnData=TRUE)
    percentVar <- round(100 * attr(pcaData, "percentVar"))

    p <- ggplot(pcaData, my_aes) +
      geom_point(size=3) +
      xlab(paste0("PC1: ",percentVar[1],"% variance")) +
      ylab(paste0("PC2: ",percentVar[2],"% variance")) +
      geom_text_repel(vjust = 0,hjust = 0.2, nudge_x = -1, nudge_y = 0.5, aes(label = name))+
      ggtitle(paste0("PCA: ", plot_title, " after batch effect correction")) +
      scale_color_brewer(type="qual", palette="Set1") +
      labs(colour= cond_col) +
      labs(shape= batch_col) +
      theme_bw()

    save_plot(file.path(results_dir, paste0(pca_prefix, "_PCA_after_batch_effect_correction")), p, width=10, height=7)

  }
})

# Filter for adjusted p-value < fdr_cutoff
resIHWsig <- resIHW %>% filter(padj < fdr_cutoff)

# significant genes as DE gene FDR < fdr_cutoff & abs(logfoldchange) > fc_cutoff , all genes as background
resIHWsig_fc <- resIHWsig %>% filter(abs(log2FoldChange) > fc_cutoff)

# Stop here if we do not have any DE genes
if(nrow(resIHWsig_fc) < 1) {
  stop("NO significant DE genes found: check fc_cutoff and fdr_cutoff!")
}


#### result list
de_res_list <- list(IHWallGenes = resIHW, IHWsigGenes = resIHWsig, IHWsigFCgenes = resIHWsig_fc)

#### write results to TSV and XLSX files
lapply(names(de_res_list), function(res) {
  fc_suffix <- ifelse(res == "IHWsigFCgenes", paste0("_", 2^fc_cutoff, "_fold"), "")
  write_tsv(de_res_list[[res]], file.path(results_dir, paste0(prefix, "_", res, fc_suffix, ".tsv")))
  write_xlsx(de_res_list[[res]], file.path(results_dir, paste0(prefix, "_" , res, fc_suffix, ".xlsx")))
})

########## Volcano plot
p <- EnhancedVolcano(resIHW,
                lab = resIHW$gene_name,
                x = "log2FoldChange",
                y = "pvalue",
                pCutoff = 1e-6,
                FCcutoff = fc_cutoff,
                subtitle = "",
                legendPosition = "right",
                caption = paste0("fold change cutoff: ", round(2**fc_cutoff, 1), ", p-value cutoff: ", 1e-6),
                title = plot_title)

save_plot(file.path(results_dir, paste0(prefix, "_volcano")), p, width = 9, height = 7)


p <- EnhancedVolcano(resIHW,
                lab = resIHW$gene_name,
                x = "log2FoldChange",
                y = "padj",
                pCutoff = fdr_cutoff,
                FCcutoff = fc_cutoff,
                subtitle = "",
                legendPosition = "right",
                caption = paste0("fold change cutoff: ", round(2**fc_cutoff, 1), ", adj.p-value cutoff: ", fdr_cutoff),
                title = plot_title)

save_plot(file.path(results_dir, paste0(prefix, "_volcano_padj")), p, width = 9, height = 7)

# Save R ws
if (save_ws) {
  save.image(file = file.path(results_dir, paste0(prefix, "_ws.RData")))
}
