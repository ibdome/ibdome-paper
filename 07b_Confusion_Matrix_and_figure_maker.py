#!/usr/bin/env python

### Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

### Functions
def cm_maker(cm, title, outfile):
    plt.figure(figsize=(8, 8)) 
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["CD", "UC"],
        yticklabels=["CD", "UC"],
        annot_kws={"size": 30},  # number size
        cbar=False
    )
    plt.xlabel("Predicted label", fontsize=24)
    plt.ylabel("True label", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.title(title, fontsize=26)
    print(f'Saving confusion matrix: {outfile}')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

def make_autopct(values, hide_zeros=False):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        if hide_zeros and val == 0:
            return ''
        return '{v:d}'.format(v=val)
    return my_autopct

# Function to create nested donut plots
def create_nested_donut(ax, outer_data, inner_data, outer_colors, inner_colors, title, hide_inner_zeros=True):
    # Show counts on outer ring only if >1 category
    show_outer_label = len(outer_data) > 1
    outer_autopct = make_autopct(outer_data) if show_outer_label else None

    # Outer donut
    texts_outer = ax.pie(
        outer_data,
        radius=1,
        colors=outer_colors,
        startangle=90,
        explode=(0.0,) * len(outer_data),
        autopct=outer_autopct,
        pctdistance=0.85,
        wedgeprops=dict(width=0.29, edgecolor='w', linewidth=5)
    )

    if show_outer_label:
        for text in texts_outer[2]:
            plt.setp(text, fontsize=18, fontweight='bold')
    else:
        # If only one segment, put total in center
        ax.text(0, 0, str(sum(outer_data)), ha='center', va='center', fontsize=24, fontweight='bold')

    # Inner donut
    texts_inner = ax.pie(
        inner_data,
        radius=0.7,
        colors=inner_colors,
        startangle=90,
        autopct=make_autopct(inner_data, hide_zeros=hide_inner_zeros),
        pctdistance=0.85,
        wedgeprops=dict(width=0.19, edgecolor='w', linewidth=5)
    )
    for text in texts_inner[2]:
        plt.setp(text, fontsize=16, fontweight='bold')

    ax.set_title(title, size=24)

#### CONFUSION MATRIX ####

# Loading the predictions
pred_all = pd.read_csv("./results/IBD_classifier_Virchow2/all_tissue/patient-preds.csv")
pred_all_inflamed = pd.read_csv("./results/IBD_classifier_Virchow2/all_tissue_inflamed/patient-preds.csv")
# clinical tables
df_berlin = pd.read_csv("./results/metadata_HE/clini_table_disease_all_berlin.csv")
df_erlangen = pd.read_csv("./results/metadata_HE/clini_table_disease_all_erlangen.csv")

labels = ["Crohn's disease", "Ulcerative colitis"]

cm_all = confusion_matrix(pred_all["disease"], pred_all["pred"], labels=labels)
cm_all_inflamed = confusion_matrix(pred_all_inflamed["disease"], pred_all_inflamed["pred"], labels=labels)

outdir = Path("./results/correlation_plots/")
outdir.mkdir(parents=True, exist_ok=True)

# Generate plots
cm_maker(cm_all, "All Tissue", outdir / "confusion_matrix_all_tissue.png")
cm_maker(cm_all_inflamed, "All Tissue Inflamed", outdir / "confusion_matrix_all_tissue_inflamed.png")

#### DONUT PLOTS ####

# Outer data: total patients per cohort
all_tissue_outer_data = [len(df_berlin), len(df_erlangen)]  # Berlin, Erlangen

# Inner data: counts per disease
# Berlin UC
berlin_uc = (df_berlin['disease'] == "Ulcerative colitis").sum()
# Berlin CD = total - UC 
berlin_cd = len(df_berlin) - berlin_uc

# Erlangen UC 
erlangen_uc = (df_erlangen['disease'] == "Ulcerative colitis").sum()
# Erlangen CD = total - UC 
erlangen_cd = len(df_erlangen) - erlangen_uc

all_tissue_inner_data = [berlin_uc, berlin_cd, erlangen_uc, erlangen_cd]  

# Define the color scheme
berlin_color = (255 / 255, 178 / 255, 102 / 255, 0.6)  # Berlin cohort
erlangen_color = (255 / 255, 102 / 255, 102 / 255, 0.6)  # Erlangen cohort
cd_color = (27/ 255, 158/ 255, 119/ 255,1)  # Crohn's disease
uc_color = (217/ 255, 95/ 255, 2/ 255,1)  # Ulcerative colitis

# Create the figure with an extra subplot for the legend
fig, axs = plt.subplots(1, 2, figsize=(18, 10))

# Create nested donuts for each disease type
create_nested_donut(axs[0], all_tissue_outer_data, all_tissue_inner_data,
                    [berlin_color, erlangen_color], 
                    [uc_color, cd_color, uc_color, cd_color],
                    "Disease types per cohort")

# Hide the axes for the empty slots
axs[1].axis('off')

# Legend 
legend_labels = [
    'Berlin all tissue', 'Erlangen all tissue', 'Crohn\'s Disease', 'Ulcerative Colitis'
]
legend_colors = [
    berlin_color, erlangen_color, cd_color, uc_color
]

handles = [plt.Line2D([0], [0], color=color, lw=20) for color in legend_colors]
axs[1].legend(handles=handles, labels=legend_labels, loc="center", frameon=False, fontsize=30)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
outfile = outdir / "disease_type_per_cohort.png"
print(f'Saving donut plot: {outfile}')
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

### Second donut plot: Erlangen Inflamed ###

# Getting inflammatio status from metadata 
outer_data_all_tissue = [len(df_erlangen)]
inflamed_erl = ((df_erlangen[["normalized_naini_cortina_score", "normalized_riley_score"]] > 0).any(axis=1)).sum()
inner_data_all_tissue = [inflamed_erl, len(df_erlangen) - inflamed_erl]
all_tissue_color = ['#ffa3a3']  
inner_colors = ['#bf5a17', '#386cb0']  # Inflamed orange, Non-inflamed blue

# Create subplot with donut + legend
fig, axs = plt.subplots(1, 2, figsize=(16, 10))

create_nested_donut(
    axs[0],
    outer_data_all_tissue,
    inner_data_all_tissue,
    all_tissue_color,
    inner_colors,
    ""
)

# legend only
axs[1].axis('off')
legend_labels = [
    'Erlangen all tissue', 'Inflamed', 'Non-inflamed'
]
legend_colors = [
    all_tissue_color[0], inner_colors[0], inner_colors[1]
]
handles = [plt.Line2D([0], [0], color=color, lw=20) for color in legend_colors]
axs[1].legend(handles=handles, labels=legend_labels, loc="center", frameon=False, fontsize=30)

fig.suptitle("Erlangen cohort: Inflamed vs Non-inflamed", fontsize=36, x=0.35, y=0.85)
plt.tight_layout()
outfile = outdir / "inflamed_state_Erlangen.png"
print(f'Saving donut plot: {outfile}')
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()