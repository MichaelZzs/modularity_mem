import os, itertools, time
import numpy as np

'''
run - number of iterations, used primarily for generating null dist., 
e.g. np.arange(1000) will give you 1000 iterations, and [0] for 1 iter
p_thresh - feature selection threshold, from 0 to 1
features - features used in the regression, all or pc_wmd or edge_scores
task - task performance, ltm or wm
shuffled - shuffle task performance, True or False (False for true model)
cv - whether to use feature indices taken from both wmd and pc indices ["False" is default | "True" means cv]
'''
run = np.arange(1000)
p_thresh = [.01, .02, .03, .04, .05, 1.01]
features = ["pc_wmd", "all", "edge_scores"]
task = ["ltm", "wm"]
shuffled = ["True"]
cv = ["False"]

for curr_combination in list(itertools.product(run, p_thresh, features, task, shuffled, cv)):
    run = curr_combination[0]
    p_thresh = curr_combination[1]
    features = curr_combination[2]
    task = curr_combination[3]
    shuffled = curr_combination[4]
    cv = curr_combination[5]
    # print(run, p_thresh, features, task, shuffled)

    outdir = 'slurm_out/{num}_{feat}_{t}'.format(num=str(p_thresh)[2:] if p_thresh < 1 else 100, feat=features, t=task)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    shell_script_file = open('run_lin_regr_w_feature_sel.sh','w')

    shell_script_file.write('#!/bin/bash\n'
                            '#SBATCH --mem-per-cpu=10G -t 6:00:00 -p short\n' # --mail-type=FAIL
                            '#SBATCH --job-name=Modularity --output={}/%j-{}-{}-{}-{}-{}-{}.out\n'.format(outdir, shuffled, run, p_thresh, features, task, cv))

    shell_script_file.write('python run_lin_regr_w_feature_sel.py {} {} {} {} {} {}'.format(run, p_thresh, features, task, shuffled, cv))

    shell_script_file.close()
    os.system('sbatch run_lin_regr_w_feature_sel.sh')
