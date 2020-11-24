#!/bin/bash
#SBATCH --mem-per-cpu=10G -t 6:00:00 -p short
#SBATCH --job-name=Modularity --output=slurm_out/100_edge_scores_wm/%j-True-999-1.01-edge_scores-wm-False.out
python run_lin_regr_w_feature_sel.py 999 1.01 edge_scores wm True False