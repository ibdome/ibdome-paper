<img src="IBDome_Logo.png" width="120"> 

# External tool documentation


Additional documentation for the STAMP and Marugoto pipelines for manual use.


### WSI Embeddings extraction with STAMP

>**NOTE:**
>To use the UNI2 and Virchow2 models, you must have a [Hugging Face](https://huggingface.co/) account with access to the respective model repositories. 
>Please refer to the [UNI2 repository](https://huggingface.co/MahmoodLab/UNI2-h) and the [Virchow2 repository](https://huggingface.co/paige-ai/Virchow2) for licensing, fair use, and access details.

In order to extract embeddings with a foundation model you would need to setup the `config.yaml` file.

Two configuration files (`config_uni2.yaml` and `config_virchow2.yaml`) inside `STAMP` are already provided to extract embeddings as an example.
Start by setting up the models and preprocessing for the actual extraction:
```bash
stamp --config config_uni2.yaml setup
stamp --config config_uni2.yaml preprocess
```
#### Switching Between Models (UNI2 ↔ Virchow2)
There is a version conflict between **STAMP** and **UNI2**:
 - `stamp 1.1.1` depends on `timm>=1.0.15`
 - `uni 0.1.0` depends on `timm==0.9.8`
To switch and use **Virchow2**, upgrade `timm`:
```bash
pip install timm --upgrade
```
Then run:
```bash
stamp --config config_virchow2.yaml setup
stamp --config config_virchow2.yaml preprocess
```

### Disease activity prediction with marugoto

#### 5 Fold Cross Validation and Deployment

To simplify the process, two bash scripts are provided inside the `marugoto`folder:

 - `run_crossval_berlin.sh`: for 5-fold cross-validation on the Berlin cohort
 - `run_deploy_erlangen.sh`: for deploying the trained models onto the Erlangen cohort

You can run them with either `riley` or `cortina` as argument:

```bash
bash run_crossval_berlin.sh riley
bash run_crossval_berlin.sh cortina

bash run_deploy_erlangen.sh riley
bash run_deploy_erlangen.sh cortina
```
>**NOTE:**
>If you encounter a "permission denied" error, you may need to make the scripts executable first:
> ```bash
> chmod +x run_crossval_berlin.sh run_deploy_erlangen.sh
> ```
### Manual Commands

Alternatively, if you prefer to run everything manually, you can use the following commands:

**Cross Validation**:
```bash
python -m marugoto.mil crossval \
    --clini_excel <path/to/clinic/table.csv> \
    --slide_csv <path/to/slide/table.csv> \
    --feature_dir <path/to/extracted/features/folder> \
    --target_label <target_label> \
    --output_path <path/to/output/folder> \
    --n_splits <number_of_folds>
```
**Deployment**:
```bash
python -m marugoto.mil deploy \
    --clini_table <path/to/clinic/table.csv> \
    --slide_csv <path/to/slide/table.csv> \
    --feature-dir <path/to/extracted/features/folder> \
    --target_label <target_label> \
    --model-path <path/to/trained/model/export.pkl> \
    --output-path <path/to/output/folder>
```

### Heatmap Generation for IBD WSIs with marugoto
This script generates **Attention heatmaps** for a given WSI (Whole Slide Image)
**Usage**
```bash
python generate_heatmap.py --wsi_name wsi_name --score_type score_type [--superimpose] [--threshold_map THRESHOLD]
```
**Positional Arguments**
 - `wsi_name`: Filename of the WSI (e.g., `TRR241-B-Re-2019-162609_col_transv_-_2023-11-08_13.55.28.ndpi`)
 - `score_type`: Either `riley` or `cortina`depending on which scoring system and model to use.
 
 **Optional Arguments**
 - `--superimpose`: If set, the attention heatmap will be overlaid on the WSI. If not set, the heatmap and WSI are displayed side-by-side.
 - `--threshold_map`: Attention threshold for visualization (default: 0.0). e.g. THRESHOLD=0.4 only regions with attention > 0.4 will be shown.

Example usage:
```bash
python generate_heatmap.py --wsi_name TRR241-B-Re-2019-162609_col_transv_-_2023-11-08_13.55.28.ndpi --score_type riley --superimpose --threshold_map 0.4
python generate_heatmap.py --wsi_name TRR241-B-Bx-2019-129947_ileum_-_2023-11-06_10.26.13.ndpi --score_type cortina
```
#### Custom Heatmap Generation (Advanced)
If you want to generate a heatmap manually (e.g., for a custom model, checkpoint, or setting), you can run the heatmap generation script directly.

**Important**:
You will need to manually specify:
 - The path to the corresponding model checkpoint folder responsible for the chosen WSI prediction (check `patient-preds.csv')
 - The path to your .h5 features folder
 - The full path to the specific WSI you want to generate.
 - (Optional) Scaling factors (`--heatmap_scale_x` and `--heatmap_scale_y`) if using superimposing and you notice misalignments.

Example command:
```bash
python marugoto/visualizations/mil_heatmaps.py \
    --train_dir path/to/your/export.pkl/folder \
    --h5_feature_dir path/to/your/features.h5/folder \
    --outdir path/to/output/folder \
    --ws_path path/to/your/WSI.ndpi \
    --alpha 0.6 \
    --superimpose \
    --threshold_map 0.4 \
    --heatmap_scale_x 0.917 
    --heatmap_scale_y 0.993
```
If you don't include `--superimpose`, no scaling is needed, and the heatmap and WSI will be displayed side-by-side.

### Disease Subtype Prediction with STAMP

To simplify the process, a bash script is provided to train a UC vs. CD classifier:

 - `07a_UC_CD_Classifier.sh`: train on the Berlin cohort and deploy on Erlangen cohort (**all Tissue** and **all tissue inflamed** subcohorts)

Statistics are also automatically generated and saved in the `model_statistics` subfolder.

```bash
bash 07a_UC_CD_Classifier.sh
```

#### Manual usage

For users who prefer to run the steps individually or customize the configuration, the same workflow can be executed manually using the `stamp` command-line tool:
```bash
stamp --config config.yaml train
stamp --config config.yaml deploy
stamp --config config.yaml statistics
```
**Notes**:
 - The commands perform the same steps as the bash script: training, deployment, and statistics calculation.
 - All input data paths and metadata are defined in the configuration file (`config.yaml` in this example). 
 - Users can create alternative configuration files with different paths or settings if needed.


### Imaging Feature Matrix Extraction
Run the following script after completing `05a` and `05b`:
Run:

```bash
bash 05c_Imaging_feature_matrix_extraction.sh
```
This script performs a 5-fold cross-validation using the Virchow2 model to generate a set of models covering all slides, and then extracts the imaging feature matrix using these models.
#### Notes for advanced use:
 - If you want to extract the feature matrix with a different model or setup, you can train your own models and update the input paths inside `generate_img_features_classifier.py`
 - The default configuration uses `config_virchow2_class_train_all.yaml` for training the 5-fold models.
