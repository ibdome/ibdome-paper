#!/bin/bash
source ~/.bash_profile
conda activate stamp
### UNI2 MODEL ###
#setting up the UNI2 model
echo "[-] Setting up UNI2 model for embeddings extraction"
stamp --config ./external_tools/STAMP/config_uni2.yaml setup
#UNI2 embeddings extraction
stamp --config ./external_tools/STAMP/config_uni2.yaml preprocess
echo "[-] Embeddings extraction complete"

### Virchow2 model ###
#upgrading timm version
pip install timm --upgrade
#setting up the Virchow2 model
echo "[-] Setting up Virchow2 model for embeddings extraction"
stamp --config ./external_tools/STAMP/config_virchow2.yaml setup
#Virchow2 embeddings extraction
stamp --config ./external_tools/STAMP/config_virchow2.yaml preprocess
echo "[-] Embeddings extraction complete"
conda deactivate