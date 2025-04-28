#!/usr/bin/env python

# ### Import libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro, pearsonr, spearmanr
from pathlib import Path
import re
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patheffects import withStroke

### Plot function
def plot_maker(fold_files, score, title, savename):
    combined_df = pd.concat((pd.read_csv(f) for f in fold_files), ignore_index=True)
    
    if score == "cortina":
        score = 'normalized_naini_cortina_score'
    elif score == "riley":
        score = 'normalized_riley_score'
    else:
        print("Error, no valid score")
        return
    
    combined_df[score] = pd.to_numeric(combined_df[score], errors='coerce')  # Handle conversion errors
    combined_df['pred'] = pd.to_numeric(combined_df['pred'], errors='coerce')
    
    pearson_corr, p_value = pearsonr(combined_df[score], combined_df['pred'])
    
    plt.figure(figsize=(18, 18))  # Adjust figure size
    ax = sns.regplot(
        x=score,
        y='pred',
        data=combined_df,
        scatter_kws={'s': 100, 'color': 'teal', 'alpha': 0.7},  
        line_kws={'color': 'darkorange', 'lw': 4}, 
        ci=95  # Display 95% confidence interval
    )
    
    plt.text(0.04, 0.95, f'Pearson Correlation = {pearson_corr:.3f}\nP-value = {p_value:.2e}',
             fontsize=35, verticalalignment='top', transform=ax.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', edgecolor='navy', facecolor='lightgray', alpha=0.9))
    
    if score == "normalized_naini_cortina_score":
        plt.xlabel('Normalized Naini-Cortina Score', fontsize=37)
    elif score == "normalized_riley_score":
        plt.xlabel('Normalized Riley Score', fontsize=37)
    plt.ylabel('Predicted Score', fontsize=37)
    plt.title(title, fontsize=51, fontweight='bold')

    ax.tick_params(axis='both', labelsize=27)  # Set size for both x and y tick labels
    plt.grid(True, linestyle='-', linewidth=0.7)
    plt.tight_layout()
    
    outdir = Path("./results/correlation_plots/")
    outdir.mkdir(parents=True, exist_ok=True)

    out_file = outdir / f"{savename}.png"
    
    print(f'Writing output to file: {out_file}')
    plt.savefig(out_file, dpi=300)


### Main result Cortina/Riley

# File paths for the 5 folds
fold_files = [
    './results/IBD_cortina_UNI2_Berlin_mil/fold-0/patient-preds.csv',
    './results/IBD_cortina_UNI2_Berlin_mil/fold-1/patient-preds.csv',
    './results/IBD_cortina_UNI2_Berlin_mil/fold-2/patient-preds.csv',
    './results/IBD_cortina_UNI2_Berlin_mil/fold-3/patient-preds.csv',
    './results/IBD_cortina_UNI2_Berlin_mil/fold-4/patient-preds.csv',
]
plot_maker(fold_files, "cortina", "Correlation of True and Predicted Scores", "Cortina_UNI2_attmil_CV")

# File paths for the 5 folds
fold_files = [
    './results/IBD_riley_Virchow2_Berlin_mil/fold-0/patient-preds.csv',
    './results/IBD_riley_Virchow2_Berlin_mil/fold-1/patient-preds.csv',
    './results/IBD_riley_Virchow2_Berlin_mil/fold-2/patient-preds.csv',
    './results/IBD_riley_Virchow2_Berlin_mil/fold-3/patient-preds.csv',
    './results/IBD_riley_Virchow2_Berlin_mil/fold-4/patient-preds.csv',
]
plot_maker(fold_files, "riley", "Correlation of True and Predicted Scores", "Riley_Virchow2_attmil_CV")


### Average fold model deploy

# Define the paths to the 5 fold prediction files
base_path = "./results/deploy_erlangen_cortina_UNI2_mil_fold-[0-4]/patient-preds.csv"
fold_paths = [base_path.replace("[0-4]", str(i)) for i in range(5)]

# Load all predictions into a list of DataFrames
fold_dfs = [pd.read_csv(path) for path in fold_paths]

# Combine predictions into a single DataFrame
# Add a column to indicate the fold
for i, df in enumerate(fold_dfs):
    df["fold"] = i

# Concatenate all DataFrames
combined_df = pd.concat(fold_dfs)

# Group by PATIENT and calculate the average prediction and other metrics
ensemble_df = combined_df.groupby("PATIENT").agg({
    "normalized_naini_cortina_score": "first",  
    "pred": "mean"
}).reset_index()

output_columns = ["PATIENT", "normalized_naini_cortina_score", "pred"]
output_df = ensemble_df[output_columns]

# Define the output path for the ensemble predictions
output_dir = "./results/deploy_erlangen_cortina_UNI2_mil_ensemble_5fold"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "patient-preds.csv")

# Save the ensemble predictions to CSV
output_df.to_csv(output_path, index=False)
print(f"Ensemble predictions saved to: {output_path}")
plot_maker([output_path], "cortina", "Correlation of True and Predicted Scores", "Cortina_UNI2_attmil_deploy_ensemble")

# Define the paths to the 5 fold prediction files
base_path = "./results/deploy_erlangen_riley_Virchow_mil_fold-[0-4]/patient-preds.csv"
fold_paths = [base_path.replace("[0-4]", str(i)) for i in range(5)]

# Load all predictions into a list of DataFrames
fold_dfs = [pd.read_csv(path) for path in fold_paths]

# Combine predictions into a single DataFrame
# Add a column to indicate the fold
for i, df in enumerate(fold_dfs):
    df["fold"] = i

# Concatenate all DataFrames
combined_df = pd.concat(fold_dfs)

# Group by PATIENT and calculate the average prediction and other metrics
ensemble_df = combined_df.groupby("PATIENT").agg({
    "normalized_riley_score": "first",  # True score is the same across folds
    "pred": "mean"
}).reset_index()

output_columns = ["PATIENT", "normalized_riley_score", "pred"]
output_df = ensemble_df[output_columns]

# Define the output path for the ensemble predictions
output_dir = "./results/deploy_erlangen_riley_Virchow2_mil_ensemble_5fold"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "patient-preds.csv")

# Save the ensemble predictions to CSV
output_df.to_csv(output_path, index=False)
print(f"Ensemble predictions saved to: {output_path}")
plot_maker([output_path], "riley", "Correlation of True and Predicted Scores", "Riley_Virchow2_attmil_deploy_ensemble")


## bMIS IPSS and ENDO score correlations

### Creating the Naini Cortina predictions score table

base_dir = "./results/IBD_cortina_UNI2_Berlin_mil"
erlangen_pred = "./results/deploy_erlangen_cortina_UNI2_mil_ensemble_5fold/patient-preds.csv"

# List to hold dataframes
dfs = []

df = pd.read_csv(erlangen_pred)
df_erlangen = df[['PATIENT', 'pred']]
dfs.append(df_erlangen)

for i in range(5):
    file_path = os.path.join(base_dir, f"fold-{i}/patient-preds.csv")
    df = pd.read_csv(file_path)
    df_filtered = df[['PATIENT', 'pred']]
    dfs.append(df_filtered)

pred_df = pd.concat(dfs, ignore_index=True)
pred_df


meta_cortina = pd.read_csv("./results/metadata_HE/clini_table_cortina.csv")
meta_cortina_rna = pd.read_csv("./results/metadata_HE/clini_table_cortina_rna.csv")

meta_cortina_pred = pd.merge(meta_cortina, pred_df, on='PATIENT', how='left')
meta_cortina_rna_pred = pd.merge(meta_cortina_rna, pred_df, on='PATIENT', how='left')


### Reading bMIS/IPSS/ENDO scores

bMIS_df = pd.read_csv("./results/ExtendedDataTable4_bMISscores.tsv", sep="\t")
bMIS_df = bMIS_df[['sample_id', 'bMIS_IBD', 'bMIS_CD', 'bMIS_UC']]
IPSS_df = pd.read_csv("./results/ExtendedDataTable3_IPSSscores.tsv", sep = "\t")
IPSS_df = IPSS_df[['subject_id', 'date', 'CD.IPSS', 'UC.IPSS']]
ENDO_df = pd.read_csv("./results/metadata_HE/Endo_scores.csv")
ENDO_df = ENDO_df[['subject_id', 'sample_id', 'date', 'SES_CD', 'UCEIS']]

#Merging with predictions tables
bMIS_df = pd.merge(meta_cortina_rna_pred, bMIS_df, on='sample_id', how='left')
IPSS_df = pd.merge(meta_cortina_pred, IPSS_df, on=['subject_id', 'date'], how='left')
ENDO_df = pd.merge(meta_cortina_pred, ENDO_df, on=['subject_id', 'date'], how='left')

# Dropping NaNs
bMIS_df = bMIS_df.dropna(subset=['pred', 'bMIS_CD'])
IPSS_df = IPSS_df.dropna(subset=['pred', 'CD.IPSS'])
ENDO_df = ENDO_df.dropna(subset=['pred', 'SES_CD'])

# Calculating IPSS and Endo scores
corr_original_CD_IPSS, p_value_original_CD_IPSS = pearsonr(IPSS_df['normalized_naini_cortina_score'], IPSS_df['CD.IPSS'])
corr_pred_CD_IPSS, p_value_pred_CD_IPSS = pearsonr(IPSS_df['pred'], IPSS_df['CD.IPSS'])
print(f"CD.IPSS Pearson R original score: {corr_original_CD_IPSS:.3f} → CD.IPSS Pearson R predicted score: {corr_pred_CD_IPSS:.3f}")

corr_original_SES_CD, p_value_original_SES_CD = pearsonr(ENDO_df['normalized_naini_cortina_score'], ENDO_df['SES_CD'])
corr_pred_SES_CD, p_value_pred_SES_CD = pearsonr(ENDO_df['pred'], ENDO_df['SES_CD'])
print(f"SES_CD Pearson R original score: {corr_original_SES_CD:.3f} → SES_CD Pearson R predicted score: {corr_pred_SES_CD:.3f}")

# Calculate Pearson correlation coefficients for bMIS_CD
corr_original_bMIS_CD, p_value_original_bMIS_CD = pearsonr(bMIS_df['normalized_naini_cortina_score'], bMIS_df['bMIS_CD'])
corr_pred_bMIS_CD, p_value_pred_bMIS_CD = pearsonr(bMIS_df['pred'], bMIS_df['bMIS_CD'])

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(36, 18))  # Adjusted to match style

# Common scatter and line styles
scatter_kws = {'s': 100, 'color': 'teal', 'alpha': 0.7}
line_kws = {'color': 'darkorange', 'lw': 4}

# Left plot: Original Score vs bMIS_CD
ax1 = sns.regplot(x='normalized_naini_cortina_score', y='bMIS_CD', data=bMIS_df,
                  ax=axes[0], scatter_kws=scatter_kws, line_kws=line_kws, ci=95)
axes[0].set_title('Original Score vs bMIS_CD', fontsize=51, fontweight='bold')
axes[0].set_xlabel('normalized modified Naini-Cortina Score', fontsize=37)
axes[0].set_ylabel('bMIS_CD', fontsize=37)
axes[0].tick_params(axis='both', labelsize=27)
axes[0].grid(True, linestyle='-', linewidth=0.7)

# Add Pearson R annotation
axes[0].text(0.04, 0.95, f'Pearson Correlation = {corr_original_bMIS_CD:.3f}\nP-value = {p_value_original_bMIS_CD:.2e}',
             fontsize=35, verticalalignment='top', transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', edgecolor='navy', facecolor='lightgray', alpha=0.9))

# Right plot: Pred vs bMIS_CD
ax2 = sns.regplot(x='pred', y='bMIS_CD', data=bMIS_df,
                  ax=axes[1], scatter_kws=scatter_kws, line_kws=line_kws, ci=95)
axes[1].set_title('Predicted Score vs bMIS_CD', fontsize=51, fontweight='bold')
axes[1].set_xlabel('Predicted Score', fontsize=37)
axes[1].set_ylabel('bMIS_CD', fontsize=37)
axes[1].tick_params(axis='both', labelsize=27)
axes[1].grid(True, linestyle='-', linewidth=0.7)

# Add Pearson R annotation
axes[1].text(0.04, 0.95, f'Pearson Correlation = {corr_pred_bMIS_CD:.3f}\nP-value = {p_value_pred_bMIS_CD:.2e}',
             fontsize=35, verticalalignment='top', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', edgecolor='navy', facecolor='lightgray', alpha=0.9))

plt.tight_layout()
outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)

out_file = outdir / "bMIS_CD_corr_plot.png"

print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300)


### Creating the Riley predictions score table

base_dir = "./results/IBD_riley_Virchow2_Berlin_mil/"
erlangen_pred = "./results/deploy_erlangen_riley_Virchow2_mil_ensemble_5fold/patient-preds.csv"

# List to hold dataframes
dfs = []

df = pd.read_csv(erlangen_pred)
df_erlangen = df[['PATIENT', 'pred']]
dfs.append(df_erlangen)

for i in range(5):
    file_path = os.path.join(base_dir, f"fold-{i}/patient-preds.csv")
    df = pd.read_csv(file_path)
    df_filtered = df[['PATIENT', 'pred']]
    dfs.append(df_filtered)

pred_df = pd.concat(dfs, ignore_index=True)
pred_df

meta_riley = pd.read_csv("./results/metadata_HE/clini_table_riley.csv")
meta_riley_rna = pd.read_csv("./results/metadata_HE/clini_table_riley_rna.csv")

meta_riley_pred = pd.merge(meta_riley, pred_df, on='PATIENT', how='left')
meta_riley_rna_pred = pd.merge(meta_riley_rna, pred_df, on='PATIENT', how='left')

### Reading bMIS/IPSS/ENDO scores

bMIS_df = pd.read_csv("./results/ExtendedDataTable4_bMISscores.tsv", sep="\t")
bMIS_df = bMIS_df[['sample_id', 'bMIS_IBD', 'bMIS_CD', 'bMIS_UC']]
IPSS_df = pd.read_csv("./results/ExtendedDataTable3_IPSSscores.tsv", sep = "\t")
IPSS_df = IPSS_df[['subject_id', 'date', 'CD.IPSS', 'UC.IPSS']]
ENDO_df = pd.read_csv("./results/metadata_HE/Endo_scores.csv")
ENDO_df = ENDO_df[['subject_id', 'sample_id', 'date', 'SES_CD', 'UCEIS']]

#Merging with predictions tables
bMIS_df = pd.merge(meta_riley_rna_pred, bMIS_df, on='sample_id', how='left')
IPSS_df = pd.merge(meta_riley_pred, IPSS_df, on=['subject_id', 'date'], how='left')
ENDO_df = pd.merge(meta_riley_pred, ENDO_df, on=['subject_id', 'date'], how='left')

# Dropping NaNs
bMIS_df = bMIS_df.dropna(subset=['pred', 'bMIS_UC'])
IPSS_df = IPSS_df.dropna(subset=['pred', 'UC.IPSS'])
ENDO_df = ENDO_df.dropna(subset=['pred', 'UCEIS'])

# Calculating IPSS and Endo score
corr_original_UC_IPSS, p_value_original_UC_IPSS = pearsonr(IPSS_df['normalized_riley_score'], IPSS_df['UC.IPSS'])
corr_pred_UC_IPSS, p_value_pred_UC_IPSS = pearsonr(IPSS_df['pred'], IPSS_df['UC.IPSS'])
print(f"UC.IPSS Pearson R original score: {corr_original_UC_IPSS:.3f} → UC.IPSS Pearson R predicted score: {corr_pred_UC_IPSS:.3f}")

corr_original_UCEIS, p_value_original_UCEIS = pearsonr(ENDO_df['normalized_riley_score'], ENDO_df['UCEIS'])
corr_pred_UCEIS, p_value_pred_UCEIS = pearsonr(ENDO_df['pred'], ENDO_df['UCEIS'])
print(f"UCEIS Pearson R original score: {corr_original_UCEIS:.3f} → SES_CD Pearson R predicted score: {corr_pred_UCEIS:.3f}")

# Calculate Pearson correlation coefficients for bMIS_UC
corr_original_bMIS_UC, p_value_original_bMIS_UC = pearsonr(bMIS_df['normalized_riley_score'], bMIS_df['bMIS_UC'])
corr_pred_bMIS_UC, p_value_pred_bMIS_UC = pearsonr(bMIS_df['pred'], bMIS_df['bMIS_UC'])

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(36, 18))  # Adjusted to match style

# Common scatter and line styles
scatter_kws = {'s': 100, 'color': 'teal', 'alpha': 0.7}
line_kws = {'color': 'darkorange', 'lw': 4}

# Left plot: Original Score vs bMIS_UC
ax1 = sns.regplot(x='normalized_riley_score', y='bMIS_UC', data=bMIS_df,
                  ax=axes[0], scatter_kws=scatter_kws, line_kws=line_kws, ci=95)
axes[0].set_title('Original Score vs bMIS_UC', fontsize=51, fontweight='bold')
axes[0].set_xlabel('normalized modified Riley Score', fontsize=37)
axes[0].set_ylabel('bMIS_UC', fontsize=37)
axes[0].tick_params(axis='both', labelsize=27)
axes[0].grid(True, linestyle='-', linewidth=0.7)

# Add Pearson R annotation
axes[0].text(0.04, 0.95, f'Pearson Correlation = {corr_original_bMIS_UC:.3f}\nP-value = {p_value_original_bMIS_UC:.2e}',
             fontsize=35, verticalalignment='top', transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', edgecolor='navy', facecolor='lightgray', alpha=0.9))

# Right plot: Pred vs bMIS_UC
ax2 = sns.regplot(x='pred', y='bMIS_UC', data=bMIS_df,
                  ax=axes[1], scatter_kws=scatter_kws, line_kws=line_kws, ci=95)
axes[1].set_title('Predicted Score vs bMIS_UC', fontsize=51, fontweight='bold')
axes[1].set_xlabel('Predicted Score', fontsize=37)
axes[1].set_ylabel('bMIS_UC', fontsize=37)
axes[1].tick_params(axis='both', labelsize=27)
axes[1].grid(True, linestyle='-', linewidth=0.7)

# Add Pearson R annotation
axes[1].text(0.04, 0.95, f'Pearson Correlation = {corr_pred_bMIS_UC:.3f}\nP-value = {p_value_pred_bMIS_UC:.2e}',
             fontsize=35, verticalalignment='top', transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.6', edgecolor='navy', facecolor='lightgray', alpha=0.9))

plt.tight_layout()
outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)

out_file = outdir / "bMIS_UC_corr_plot.png"

print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300)

## Other correlation plots performance

r_values = np.array([
    [round(corr_original_CD_IPSS, 3), round(corr_pred_CD_IPSS, 3)],      # CD.IPSS
    [round(corr_original_UC_IPSS, 3), round(corr_pred_UC_IPSS, 3)],      # UC.IPSS
    [round(corr_original_UCEIS, 3), round(corr_pred_UCEIS, 3)],          # UCEIS
    [round(corr_original_SES_CD, 3), round(corr_pred_SES_CD, 3)]         # SES_CD
])

row_labels = ['CD.IPSS', 'UC.IPSS', 'UCEIS', 'SES_CD']
column_labels = ['Original Score', 'Predicted Score']

df = pd.DataFrame(r_values, index=row_labels, columns=column_labels)

fig, ax = plt.subplots(figsize=(4, 8))

divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="3%", pad=0.4)

sns.heatmap(df, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1,
            cbar_ax=cax, cbar_kws={'orientation': 'horizontal'},
            annot_kws={"size": 13}, xticklabels=True, yticklabels=True,
            linewidths=1, linecolor='white', ax=ax)

cax.xaxis.set_label_position('top')
cax.set_xlabel("Pearson's R", labelpad=10, fontsize=13) 

ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=13)

plt.tight_layout()
outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)
out_file = outdir / "other_corr_plots_performance.png"
print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300, bbox_inches='tight')

## Foundation Models Performance

r_values = np.array([
    [0.745, 0.801, 0.792, 0.784],  # Original score row
    [0.903, 0.929, 0.933, 0.927]   # Predicted score row
])

row_labels = ['normalized modified\n Naini-Cortina Score', 'normalized modified\n Riley Score']
column_labels = ['CHIEF', 'UNI2', 'Virchow2', 'H-optimus-0']

df = pd.DataFrame(r_values, index=row_labels, columns=column_labels)

fig, ax = plt.subplots(figsize=(8, 2.8))

sns.heatmap(df, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1,
            cbar_kws={'label': 'Pearson\'s R'}, annot_kws={"size": 13},
            yticklabels=False, linewidths=1, linecolor='white')

for i, label in enumerate(row_labels):
    ax.text(-0.9, i + 0.5, label, ha='center', va='center', rotation=0, fontsize=13)
ax.tick_params(axis='x', labelsize=13)

plt.title('Pearson correlation coefficient', fontsize=16)
plt.tight_layout()

outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)
out_file = outdir / "foundation_models_performance.png"
print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300, bbox_inches='tight')

## Donut plot main figure + histo/RNA/Protein/WES

# Define the color scheme
berlin_color = (255 / 255, 178 / 255, 102 / 255, 0.6)  # Berlin cohort
erlangen_color = (255 / 255, 102 / 255, 102 / 255, 0.6)  # Erlangen cohort
cd_color = (27/ 255, 158/ 255, 119/ 255,1)  # Crohn's disease
uc_color = (217/ 255, 95/ 255, 2/ 255,1)  # Ulcerative colitis
non_ibd_color = (231/ 255, 41/ 255, 138/ 255,1)  # non-IBD
indeterminate_colitis_color = (117 / 255, 112 / 255, 179 / 255, 1)  # Indeterminate colitis

# Lighter versions for RNA-seq, protein, and WES subsets
rna_berlin_color = (255 / 255, 204 / 255, 153 / 255, 0.6)  # Light version for Berlin RNA-seq
rna_erlangen_color = (255 / 255, 153 / 255, 153 / 255, 0.6)  # Light version for Erlangen RNA-seq
rna_protein_berlin_color = (255 / 255, 229 / 255, 204 / 255, 0.6)
rna_protein_erlangen_color = (255 / 255, 204 / 255, 204 / 255, 0.6)
rna_protein_exome_berlin_color = (255 / 255, 243 / 255, 224 / 255, 0.6)
rna_protein_exome_erlangen_color = (255 / 255, 224 / 255, 224 / 255, 0.6)
no_data_color = "#d3d3d3"  # Light grey for samples without further data

# Data for the disease type distribution (outer circle) and cohort breakdown (inner circle)
disease_types = [539, 321, 116, 26]  # CD, UC, non-IBD, Indeterminate colitis
disease_colors = [cd_color, uc_color, non_ibd_color, indeterminate_colitis_color]

# Breakdown for Berlin and Erlangen for the inner circle of disease distribution
cd_inner_disease = [378, 161]
uc_inner_disease = [248, 73]
non_ibd_inner_disease = [112, 4]
indeterminate_colitis_inner_disease = [14, 12]


explode = (0.0, 0.0)  
# Function to make autopct with glowing effect
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{v:d}'.format(v=val)
    return my_autopct

# Function to apply glow to text
def add_glow_effect(text):
    text.set_path_effects([
        withStroke(linewidth=1, foreground="white"),  # Outer white glow
        withStroke(linewidth=0, foreground="black")  # Inner black stroke for contrast
    ])

# Function to create nested donut plots
def create_nested_donut(ax, outer_data, inner_data, inner2_data, inner3_data, outer_colors, inner_colors, inner2_colors, inner3_colors, title, inner_texts):
    # Outer donut (Berlin vs Erlangen)
    texts = ax.pie(outer_data, radius=1, colors=outer_colors, startangle=90, explode=explode, 
       autopct=make_autopct(outer_data), pctdistance=0.91,
       wedgeprops=dict(width=0.19, edgecolor='w',linewidth=5))

    # Set font size for autopct values in the outer donut
    for text in texts[2]:  # The autopct texts are the third element in the returned tuple
        plt.setp(text, fontsize=16, fontweight = 'bold')  # Set desired font size
    
    # Inner circles pies
    inner_texts = ax.pie(inner_data, radius=0.8, colors=inner_colors, startangle=90, 
       autopct=make_autopct_inner(inner_data), 
       pctdistance=0.88, wedgeprops=dict(width=0.19, edgecolor='w',linewidth=5))
    
    # Set font size for autopct values in the inner circles
    for text in inner_texts[2]:
        plt.setp(text, fontsize=16, fontweight = 'bold')

    inner_texts1 = ax.pie(inner2_data, radius=0.6, colors=inner2_colors, startangle=90, 
       autopct=make_autopct_inner(inner2_data), 
       pctdistance=0.83, wedgeprops=dict(width=0.19, edgecolor='w',linewidth=5))
    for text in inner_texts1[2]:
        plt.setp(text, fontsize=16, fontweight = 'bold')
        
    inner_texts2 = ax.pie(inner3_data, radius=0.4, colors=inner3_colors, startangle=90, 
       autopct=make_autopct_inner(inner3_data), 
       pctdistance=0.73, wedgeprops=dict(width=0.19, edgecolor='w',linewidth=5))
    for text in inner_texts2[2]:
        plt.setp(text, fontsize=16, fontweight = 'bold')
        
    # Set title
    ax.set_title(title, size=30)

# Create the figure and axes (only 1 axis for the first donut plot)
fig, axs = plt.subplots(1, 1, figsize=(15, 13))  # Adjusted for one plot instead of multiple

# First donut plot: Disease type distribution with inner circle for Berlin and Erlangen breakdown
outer_circle_colors = [cd_color, uc_color, non_ibd_color, indeterminate_colitis_color]
inner_circle_colors = [berlin_color, erlangen_color]

# Outer circle
text_main = axs.pie(
    disease_types, startangle=90, colors=disease_colors, explode=None,
    autopct=make_autopct(disease_types), pctdistance=0.8,
    wedgeprops=dict(width=0.4, edgecolor='w')
)

# Make autopct values bigger and apply glow
for text in text_main[2]:  # The autopct texts are the third element in the returned tuple
    plt.setp(text, fontsize=22, color='black')  # Set font size and color (no bold)
    add_glow_effect(text)  # Apply the glow effect

# Inner circle
inner_circle_data = (
    cd_inner_disease + uc_inner_disease +
    non_ibd_inner_disease + indeterminate_colitis_inner_disease
)
axs.pie(
    inner_circle_data, radius=0.6,
    colors=[
        berlin_color, erlangen_color, berlin_color, erlangen_color,
        berlin_color, erlangen_color, berlin_color, erlangen_color
    ],
    startangle=90, wedgeprops=dict(width=0.2, edgecolor='w')
)

# Add a white circle at the center for the donut effect
centre_circle = plt.Circle((0, 0), 0.40, color='white', fc='white', linewidth=1.25)
axs.add_artist(centre_circle)

# Title and legend
axs.set_title("Disease Type Distribution with Cohorts", size=30)
axs.legend(
    [
        'Crohn\'s Disease', 'Ulcerative Colitis', 'non-IBD',
        'Indeterminate colitis', "Berlin", "Erlangen"
    ],
    loc="upper right", bbox_to_anchor=(1.4, 1),
    title="Disease Type", frameon=False
)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)
out_file = outdir / "donut_chart_H&E_extended.png"
print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300, bbox_inches='tight')

## Donut plot supp figure NCS/RS+RNA

ncs_outer_data = [514, 185]  # Berlin, Erlangen
ncs_inner_data = [166, 514-166, 75, 185-75]  # Berlin RNA-seq, No RNA-seq, Erlangen RNA-seq, No RNA-seq

rs_outer_data = [472, 84]  # Berlin, Erlangen
rs_inner_data = [135, 472-135, 23, 84-23]

# Function to create nested donut plots
def create_nested_donut(ax, outer_data, inner_data, outer_colors, inner_colors, title):
    # Outer donut (Berlin vs Erlangen)
    texts = ax.pie(outer_data, radius=1, colors=outer_colors, startangle=90, explode=(0.0, 0.0), 
       autopct=make_autopct(outer_data), pctdistance=0.85, wedgeprops=dict(width=0.29, edgecolor='w',linewidth=5))

    # Set font size for autopct values in the outer donut
    for text in texts[2]:  # The autopct texts are the third element in the returned tuple
        plt.setp(text, fontsize=18, fontweight = 'bold')  # Set desired font size
    
    # Inner circles pies
    inner_texts = ax.pie(inner_data, radius=0.7, colors=inner_colors, startangle=90, 
           autopct=make_autopct_inner(inner_data), pctdistance=0.85, wedgeprops=dict(width=0.19, edgecolor='w',linewidth=5))
    
    # Set font size for autopct values in the inner circles
    for text in inner_texts[2]:
        plt.setp(text, fontsize=16, fontweight = 'bold')
    # Set title
    ax.set_title(title, size=24)

def make_autopct_inner(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # Check if the value corresponds to the first or third index (non-grey values)
        if values.index(val) % 2 == 0:
            return '{v:d}'.format(v=val)
        else:
            return ''
    return my_autopct

# Create the figure with an extra subplot for the legend
fig, axs = plt.subplots(1, 3, figsize=(24, 12))

# Create the main plots

# Create nested donuts for each disease type
create_nested_donut(axs[0], ncs_outer_data, ncs_inner_data,
                    [berlin_color, erlangen_color], 
                    [rna_berlin_color, no_data_color, rna_erlangen_color, no_data_color],
                    "Modified Naini Cortina score")

create_nested_donut(axs[1], rs_outer_data, rs_inner_data, 
                    [berlin_color, erlangen_color], 
                    [rna_berlin_color, no_data_color, rna_erlangen_color, no_data_color],
                    "Modified Riley score")

# Hide the axes for the empty slots
axs[2].axis('off')

# Place the legend in the bottom-right subplot
legend_labels = [
    'Berlin histopathology', 'Erlangen histopathology', 'Berlin RNA-seq histopathology', 'Erlangen RNA-seq histopathology'
]
legend_colors = [
    berlin_color, erlangen_color, rna_berlin_color, rna_erlangen_color, 
    rna_protein_berlin_color, rna_protein_erlangen_color,
    rna_protein_exome_berlin_color, rna_protein_exome_erlangen_color
]

handles = [plt.Line2D([0], [0], color=color, lw=20) for color in legend_colors]
axs[2].legend(handles=handles, labels=legend_labels, loc="center", frameon=False, fontsize=30)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)
out_file = outdir / "donut_chart_H&E_extended_with_legend.png"
print(f'Writing output to file: {out_file}')
plt.savefig(out_file, dpi=300, bbox_inches='tight')