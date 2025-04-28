NA_COLOR <- "#999999"

#' @export
COLORS <- list(
    disease = c(
        "Crohn's disease" = "#1b9e77",
        "Ulcerative colitis" = "#d95f02",
        "Indeterminate colitis" = "#7570b3",
        "non-IBD" = "#e7298a",
        "NA" = NA_COLOR
    ),
    tissue = c(
        "colon" = "#E69F00",
        "ileum" = "#56B4E9",
        "small intestine" = "#009E73",
        "caecum" = "#F0E442",
        "ileocecal valve" = "#0072B2",
        "rectum" = "#D55E00",
        "whole blood" = "#CC79A7",
        "anastomosis" = "#B3DE68",
        "NA" = NA_COLOR
    ),
    tissue_coarse = c(
      "colon" = "#E69F00",
      "ileum" = "#56B4E9",
      "NA" = NA_COLOR
    ),
    tissue_annotated = c(
        "colon" = "#E69F00",
        "ileum" = "#56B4E9",
        "small intestine" = "#009E73",
        "caecum" = "#F0E442",
        "ileocecal valve" = "#0072B2",
        "rectum" = "#D55E00",
        "whole blood" = "#CC79A7",
        "anastomosis" = "#B3DE68",
        "NA" = NA_COLOR
    ),
    treatment = c(
        "5ASA" = "#1B9E77",
        "6-MP" = "#D95F02",
        "AZA" = "#7570B3",
        "Adalimumab" = "#E7298A",
        "Budesonide" = "#66A61E",
        "Ciclosporin" = "#E6AB02",
        "Cyclophosphamide" = "#A6761D",
        "E.coli Nissle" = "#666666",
        "Filgotinib" = "#1B9E77",
        "Golimumab" = "#D95F02",
        "Hydrocortisone acetate" = "#7570B3",
        "Infliximab" = "#E7298A",
        "MTX" = "#66A61E",
        "Prednisolone" = "#E6AB02",
        "Sulfasalazine" = "#A6761D",
        "Tacrolimus" = "#666666",
        "Tofacitinib" = "#1B9E77",
        "Ursodeoxycholic acid" = "#D95F02",
        "Ustekinumab" = "#7570B3",
        "Vedolizumab" = "#E7298A"
    ),
    sex = c(
      "male" = "#8da0cb",
      "female" = "#fc8d62"
    ),
    localization = c(
      "L1: terminal ileum" = "#984ea3",
      "L2: colon" = "#e41a1c",
      "L3: ileocolon"= "#ffff33",
      "E1: proctitis" = "#1f78b4",
      "E2: left sided colitis" = "#33a02c",
      "E3: extensive colitis" = "#ff7f00"
    ),
    sampling_procedure = c(
      "biopsy" = "coral4",
      "resection"= "cornsilk"
    ),
    group = c(
      "IBD_inflamed" = "#BF5B17", 
      "IBD_non_inflamed" = "#386CB0",
      "non-IBD" = "#e7298a",
      "NA" = NA_COLOR
    ),
    inflammation_status = c(
      "inflamed" = "#BF5B17", 
      "non_inflamed" = "#386CB0",
      "NA" = NA_COLOR
    ),
    group_fine = c(
      "CD_inflamed" = "#1b9e77",
      "CD_non_inflamed" = "#60d6ab",
      "UC_inflamed" = "#d95f02",
      "UC_non_inflamed" = "#eaa221",
      "nonIBD" =  "#e7298a",
      "Other" = NA_COLOR
    ),
    group_tissue = c(
      "CD_colon" = "#E69F00", #F2CE7F", 
      "CD_ileum" = "#56B4E9", #AAD9F4", 
      "UC_colon" = "#d95f02", #ECAE80",
      "NA" = NA_COLOR,
      "nonIBD_ileum" = "#e7298a",
      "nonIBD_colon" = "#e7298a"
    ),
    datasets = c(
      "ibdome_berlin" = "#ffcfa3",
      "ibdome_erlangen" = "#ffa3a3"
    ),
    `study center` = c(
      "Berlin" = "#ffcfa3",
      "Erlangen" = "#ffa3a3"
    )
)
