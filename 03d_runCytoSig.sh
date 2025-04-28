#!/bin/bash
source ~/.bash_profile
conda activate cytosig.v0.1
external_tools/CytoSig/CytoSig/CytoSig_run.py -i results/CytoSig/cyto_mat.csv -o "results/CytoSig/ibdome" -e 1
conda deactivate