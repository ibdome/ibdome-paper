#!/bin/bash
source ~/.bash_profile
conda activate stamp

### Virchow2 model ###

echo "[-] Training model on the whole Berlin cohort"
stamp --config ./external_tools/STAMP/config_virchow2_class_train.yaml train
# Deploy part
echo "[-] Deploying model on the Erlangen cohort (All tissue)"
stamp --config ./external_tools/STAMP/config_virchow2_class_deploy_all.yaml deploy
echo "[-] Generating statistics..."
stamp --config ./external_tools/STAMP/config_virchow2_class_deploy_all.yaml statistics

echo "[-] Deploying model on the Erlangen cohort (All tissue inflamed)"
stamp --config ./external_tools/STAMP/config_virchow2_class_deploy_all_inflamed.yaml deploy
echo "[-] Generating statistics..."
stamp --config ./external_tools/STAMP/config_virchow2_class_deploy_all_inflamed.yaml statistics

echo "[-] Done."
conda deactivate