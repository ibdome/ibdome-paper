<img src="IBDome_Logo.png" width="120"> 

# IBDome paper


Source code for the analyses accompanying the IBDome paper. 

## Environment setup and database download

### Clone the repo

```
git clone https://github.com/orgs/ibdome/repositories/ibdome-paper.git
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

You might need to install `ncftp` in order to perform recusive FTP downloads.

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

**NOTE:** For additional information regarding the STAMP and marugoto pipeline usage please refer to the additional [documentation](https://gitlab.i-med.ac.at/icbi-lab/ibdome/ibdome-paper/-/blob/main/external_tools/README.md).

# Contact

Please use the [issue tracker][issue-tracker].

# Citation

> Plattner, C., Sturm, G., Kühl, A.A., Atreya, R., Carollo, S., ... & Becker, C., Siegmund, B., Trajanoski, Z. (2025). IBDome: An integrated molecular, histopathological, and clinical atlas of inflammatory bowel diseases. bioRxiv. [doi:10.1101/2025.03.26.645544](https://doi.org/10.1101/2025.03.26.645544 ) 

[issue-tracker]: https://github.com/ibdome/ibdome-paper/issues