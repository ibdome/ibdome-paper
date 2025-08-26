<img src="IBDome_Logo.png" width="120"> 

# IBDome paper


Source code for the analyses accompanying the IBDome paper. 

## Environment setup and database download

### Clone the repo

```
git clone https://github.com/ibdome/ibdome-paper.git
```

### Change directory and create a data folder

```
export IBDOME_BASEDIR=$(realpath ibdome-paper)

cd $IBDOME_BASEDIR
mkdir data
cd data
```

### Download the IBDome database

Download the ibdome_v1.0.1.sqlite database from: https://ibdome.org/#!/data_download and store it in the data directory.

```
wget https://ibdome.org/static/ibdome_v1.0.1.zip
```

### Download the imaging data

You might need to install `ncftp` in order to perform recursive FTP downloads.

e.g.

Fedora, RedHat, Rocky Linux and other rpm based Linux distributions

```
sudo dnf install ncftp
```

Ubuntu, Debian and other apt based Linux distributions

```
sudo apt update
sudo apt install ncftp
```

Get the data

```
ncftpget -T -R ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/753/S-BIAD1753/Files/imaging
cd ..
```

### CytoSig setup

Paper ([Jiang et al., Nat Methods 2021](https://doi.org/10.1038/s41592-021-01274-5)) and [app](https://cytosig.ccr.cancer.gov/)
Script: https://github.com/data2intelligence/CytoSig


```bash
conda create -n cytosig.v0.1 -y python=3.8 'numpy>=1.19' 'pandas>=1.1.4' 'gcc>=4.2' 'openpyxl>=3.0.9' gsl=2.6 setuptools scipy xlsxwriter
conda activate cytosig.v0.1

cd external_tools
git clone https://github.com/data2intelligence/ridge_significance.git
git clone https://github.com/data2intelligence/data_significance.git
git clone https://github.com/data2intelligence/CytoSig.git

cd ridge_significance
python setup.py install
python -m unittest tests.regression

cd ../data_significance
python3 setup.py install
python3 -m unittest tests.regression

cd ../CytoSig
python setup.py install
python -m unittest tests.prediction
conda deactivate
cd ..
```

### MOFA2 setup

Create a conda environment for the Python version of MOFA2 (mofapy2):

```bash
conda create -n MOFA_env python=3.13.3 pip -y && conda run -n MOFA_env pip install mofapy2==0.7.2
```

### WSI Embeddings extraction setup

First clone the forked and updated version of **STAMP (v1 branch)**, which includes the latest Foundation Models available for use:
```bash
git clone -b v1 https://github.com/sandrocarollo/STAMP.git
cd STAMP
```
Create a new conda environment:
```bash
conda create -n stamp python=3.10
conda activate stamp
conda install -c conda-forge libstdcxx-ng=12
```
Install the STAMP package:
```bash
pip install .
```
>**NOTE:**
>To use the UNI2 and Virchow2 models, you must have a [Hugging Face](https://huggingface.co/) account with access to the respective model repositories. 
>Please refer to the [UNI2 repository](https://huggingface.co/MahmoodLab/UNI2-h) and the [Virchow2 repository](https://huggingface.co/paige-ai/Virchow2) for licensing, fair use, and access details.


### Disease activity prediction setup

To predict disease activity from WSI embeddings, we use the **marugoto** pipeline.

First clone the forked and updated version of **marugoto**, which includes stratified training and attention heatmap generation:
```bash
cd ..
git clone -b attmil-regression https://github.com/sandrocarollo/marugoto.git
cd marugoto
```
Create and activate the dedicated environment:
```bash
conda deactivate
conda env create -f env_marugoto.yml
conda activate marugoto
pip install .
```

**NOTE:** For additional information regarding the STAMP and marugoto pipeline usage please refer to the additional [documentation](https://github.com/ibdome/ibdome-paper/blob/main/external_tools/README.md).

## Reproducing the Results
The scripts in this repository are numbered in the order they should be executed to fully reproduce the results of our paper.

The repository contains a mix of **R Markdown** (`.Rmd`), **Bash** (`.sh`), and **Python** (`.py`) scripts.

They must be executed in ascending numerical order, as outputs of one step are often inputs for the next.

Each script type is run differently:

- **R Markdown (`.Rmd`)**
    2 run possibilities: 
  - Interactive: open in RStudio and click **Knit** or by
  - Command line:
    ```bash
    Rscript -e "rmarkdown::render('0X_markdown_file.Rmd')"
    ```

- **Bash (`.sh`)**
  - Run directly in the shell:
    ```bash
    bash 0X_bash_script.sh
    ```

- **Python (`.py`)**
  - Run with Python:
    ```bash
    python 0X_python_script.py
    ```

### Execution Order
Run the scripts starting from `01_IBDome_overview.Rmd` and activate the correct Conda environment if needed.
```bash
Rscript -e "rmarkdown::render('01_IBDome_overview.Rmd')"
Rscript -e "rmarkdown::render('02_IPSS.Rmd')"
Rscript -e "rmarkdown::render('03a_gene_signatures.Rmd')"

bash 03b_deseq2.sh

Rscript -e "rmarkdown::render('03c_DE_downstream.Rmd')"

conda activate cytosig.v0.1
bash 03d_runCytoSig.sh
conda deactivate

Rscript -e "rmarkdown::render('03e_cytosig_downstream.Rmd')"
Rscript -e "rmarkdown::render('04_protein_panel.Rmd')"
Rscript -e "rmarkdown::render('05a_Extract_histoscores.Rmd')"

conda activate stamp
bash 05b_Embedding_extraction.sh
bash 05c_Imaging_feature_matrix_extraction.sh
conda deactivate 

Rscript -e "rmarkdown::render('05d_MOFA.Rmd')"

conda activate marugoto
bash 06a_Disease_activity_prediction.sh
bash 06b_Attention_Heatmap_generation.sh

python 06c_correlation_plots_and_figure_maker.py
python 06d_sankey_dash.py
conda deactivate 

conda activate stamp
bash 07a_UC_CD_Classifier.sh
python 07b_Confusion_Matrix_and_figure_maker.py
conda deactivate
```
**Notes**:
 - Use the right Conda environment for each step (`cytosig.v0.1`, `marugoto`, `stamp`).
 - Each script may take a significant amount of time depending on hardware resources.
 - Script numbers correspond to the figure numbers in the paper for easier reference.


# Contact

Please use the [issue tracker][issue-tracker].

# Citation

> Plattner, C., Sturm, G., Kühl, A.A., Atreya, R., Carollo, S., ... & Becker, C., Siegmund, B., Trajanoski, Z. (2025). IBDome: An integrated molecular, histopathological, and clinical atlas of inflammatory bowel diseases. bioRxiv. [doi:10.1101/2025.03.26.645544](https://doi.org/10.1101/2025.03.26.645544 ) 

[issue-tracker]: https://github.com/ibdome/ibdome-paper/issues