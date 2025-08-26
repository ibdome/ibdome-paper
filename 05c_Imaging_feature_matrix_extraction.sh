#!/bin/bash

### 5fcv Berlin+Erlangen ###

echo "[-] Training model with 5fcv on the whole Berlin+Erlangen cohort"
stamp --config ./external_tools/STAMP/config_virchow2_class_train_all.yaml crossval
echo "[-] 5FCV Done."

### Imaging feature matrix extraction
echo "[-] Imaging feature matrix generation"
python external_tools/marugoto/generate_img_features_classifier.py
echo "[-] Done."