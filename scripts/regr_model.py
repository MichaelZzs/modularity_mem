import brain_graphs
import scipy.io as sio
import numpy as np
import pandas as pd
from datetime import date
import itertools
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from matplotlib.pyplot import text
import shutil
from sklearn.linear_model import LinearRegression
import sys


def nan_pearsonr(x,y):
    x = np.array(x)
    y = np.array(y)
    isnan = np.sum([x,y],axis=0)
    isnan = np.isnan(isnan) == False
    return pearsonr(x[isnan],y[isnan])


def generate_correlation_map(x, y):
    """
    Correlate each n with each m.
    ----------
    Parameters
    x : np.array, shape N X T.
    y : np.array, shape M X T.
    Returns: np.array, N X M array in which each element is a correlation coefficient.
    ----------
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def super_edge_predict_new(v, p_thresh, features, cv="False"):
    subject_pcs = v[0]
    subject_wmds = v[1]
    subject_mods = v[2]

    task_perf = v[3]
    t = v[4]

    task_matrices = v[5]
    return_features = v[6]
    use_matrix = v[7]

    fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
    fit_mask[t] = False

    """
    Feature selection on pc/wmd
    """
    thresh_pcs = []
    thresh_wmds = []

    for i in range(subject_pcs.shape[1]):
        pc = pearsonr(v[0][fit_mask, i], task_perf[fit_mask])
        wmd = pearsonr(v[1][fit_mask, i], task_perf[fit_mask])
        thresh_pcs.append([pc[0], pc[1]])
        thresh_wmds.append([wmd[0], wmd[1]])

    thresh_pcs = np.array(thresh_pcs)
    thresh_wmds = np.array(thresh_wmds)

    # redefining subject pcs and wmds
    if cv == "True":
        idx = (thresh_pcs[:, 1] < p_thresh) & (thresh_wmds[:, 1] < p_thresh)
        pc_idx = idx
        wmd_idx = idx
    else:
        pc_idx = thresh_pcs[:, 1] < p_thresh
        wmd_idx = thresh_wmds[:, 1] < p_thresh

    subject_pcs = subject_pcs[:, pc_idx]
    subject_wmds = subject_wmds[:, wmd_idx]

    # return nans if there are too few features
    if len(subject_pcs[1]) < 3 or len(subject_wmds[1]) < 3:
        return 0, np.nan, np.nan, np.nan, np.nan

    if use_matrix == True:
        flat_matrices = np.zeros((subject_pcs.shape[0], len(np.tril_indices(368, -1)[0])))
        for s in range(subject_pcs.shape[0]):
            m = task_matrices[s]
            flat_matrices[s] = m[np.tril_indices(368, -1)]
        perf_edge_corr = \
        generate_correlation_map(task_perf[fit_mask].reshape(1, -1), flat_matrices[fit_mask].transpose())[0]

        '''
        Feature selection for edge scores
        '''
        thresh_es = []
        for i in range(flat_matrices.shape[1]):
            es = pearsonr(flat_matrices[fit_mask, i], task_perf[fit_mask])
            thresh_es.append(es)

        thresh_es = np.array(thresh_es)
        es_idx = thresh_es[:, 1] < p_thresh
        perf_edge_corr = perf_edge_corr[es_idx]

        perf_edge_scores = np.zeros((subject_pcs.shape[0]))
        for s in range(subject_pcs.shape[0]):
            perf_edge_scores[s] = pearsonr(flat_matrices[s][es_idx], perf_edge_corr)[0]

    perf_pc_corr = np.zeros(subject_pcs.shape[1])
    for i in range(subject_pcs.shape[1]):
        perf_pc_corr[i] = nan_pearsonr(task_perf[fit_mask], subject_pcs[fit_mask, i])[0]
    perf_wmd_corr = np.zeros(subject_wmds.shape[1])
    for i in range(subject_wmds.shape[1]):
        perf_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask], subject_wmds[fit_mask, i])[0]

    task_pc = np.zeros(subject_pcs.shape[0])
    task_wmd = np.zeros(subject_pcs.shape[0])
    for s in range(subject_pcs.shape[0]):
        task_pc[s] = nan_pearsonr(subject_pcs[s], perf_pc_corr)[0]
        task_wmd[s] = nan_pearsonr(subject_wmds[s], perf_wmd_corr)[0]

    if use_matrix == True:
        if features == "all":
            pvals = np.array([task_pc, task_wmd, perf_edge_scores, subject_mods]).transpose()
        elif features == "pc_wmd":
            pvals = np.array([task_pc, task_wmd]).transpose()
        elif features == "edge_scores":
            pvals = np.array([perf_edge_scores]).transpose()
    # elif use_matrix == False:
    #     pvals = np.array([task_pc, task_wmd, subject_mods]).transpose()

    train = np.ones(len(pvals)).astype(bool)
    train[t] = False
    model = LinearRegression(normalize=True)

    reg = model.fit(pvals[train], task_perf[train])
    coef = reg.coef_

    result = model.predict(pvals[t].reshape(1, -1))[0]
    if return_features == True:
        return pvals[t], result
    return result, pc_idx, wmd_idx, es_idx, coef


# 1: graph measures
mat = np.load('/gpfs/milgram/scratch60/chun/mz456/modularitymem_newest/05_to_15_thresh_368_100920.npz')
mods = np.mean(mat['mod'], axis=1)
pcs = mat['pc']
wmds = mat['wmd']
subs = mat['sub']

# 2: task performances and task matrices (correlation mat.)
corr_mat = sio.loadmat('/gpfs/milgram/project/chun/ql225/HCP_RecMem/derivatives/WM_avg_mats_368_20200506.mat')
behav_df = pd.read_csv('/gpfs/milgram/project/chun/mz456/modularity_newest/708_sub_df.csv')

# task performances
ltm = np.array(behav_df['dPrime_new'])
wm = np.array(behav_df['WM_Task_2bk_Acc'])

# task matrices, overlap bt behavioral df and corr mats
matching_inds = pd.Series(np.squeeze(corr_mat['WM_sub_list'])).isin(behav_df['sub_id'])
sub_mats = corr_mat['sub_mats_avg_WM'][:,:,matching_inds]  # sub mats are our task matrices

# 3: return values
return_features = False
use_matrix = True

# model prediction
run = int(sys.argv[1])
run = f'{run:04}' if run < 1000 else run

p_thresh = float(sys.argv[2])
features = sys.argv[3]

task = sys.argv[4]
if task == "ltm":
    behav = ltm
elif task == "wm":
    behav = wm

shuffled = sys.argv[5]
if shuffled == "True":
    np.random.shuffle(behav)

cv = sys.argv[6]

# save file
outpath = '/gpfs/milgram/scratch60/chun/mz456/modularitymem_newest2/lin_regr_{num}'.format(num=str(p_thresh)[2:] if p_thresh < 1 else 100)
if not os.path.isdir(outpath):
    os.makedirs(outpath, exist_ok=True)
if shuffled == "True":
    outpath = os.path.join(outpath, 'shuffled_{}_{}{cv}'.format(features, task, cv="_cv" if cv == "True" else ""))
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

today = date.today()
timestamp = today.strftime("%m%d%y")
filename = '{}_{}_{}_{}_{}{cv}'.format(run, task, shuffled, features, timestamp, cv="_cv" if cv == "True" else "")
outfile = os.path.join(outpath, filename)

res = []
for t in range(len(subs)):
    v = [pcs, wmds, mods, behav, t, sub_mats.T, False, use_matrix]
    res.append(super_edge_predict_new(v, p_thresh, features, cv))

# extract model measures
pred = []
pc_idx = []
wmd_idx = []
es_idx = []
coefs =[]
for i in range(len(res)):
    print(i)
    pred.append(res[i][0])
    pc_idx.append(res[i][1])
    wmd_idx.append(res[i][2])
    es_idx.append(res[i][3])
    coefs.append(res[i][4])

np.savez(outfile, actual=behav, predicted=pred, pc_nodes=pc_idx, wmd_nodes=wmd_idx, edge_scores=es_idx, coefs=coefs)
