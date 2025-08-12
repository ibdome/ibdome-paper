#!/bin/bash
source ~/.bash_profile
conda activate marugoto
python ./external_tools/marugoto/generate_heatmap.py --wsi_name TRR241-B-Re-2019-162609_col_transv_-_2023-11-08_13.55.28.ndpi --score_type riley --superimpose --threshold_map 0.4
python ./external_tools/marugoto/generate_heatmap.py --wsi_name TRR241-B-Bx-2019-129947_ileum_-_2023-11-06_10.26.13.ndpi --score_type cortina
conda deactivate