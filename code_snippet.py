import brain_graphs
import scipy.io as sio
import numpy as np
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


def individual_graph_analyes_wc(variables):
    subject = variables[0]
    print(subject)
    s_matrix = variables[1]
    pc = []
    mod = []
    wmd = []
    memlen = []
    for cost in np.array(range(5,16))*0.01:
        temp_matrix = s_matrix.copy()
        graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=True,mst=True)
        assert(np.diff([cost,graph.density()])[0] < .01)
        del(temp_matrix)
        graph = graph.community_infomap(edge_weights='weight')
        graph = brain_graphs.brain_graph(graph)
        pc.append(np.array(graph.pc))
        wmd.append(np.array(graph.wmd))
        mod.append(graph.community.modularity)
        memlen.append(len(graph.community.sizes()))
        del(graph)
    return (mod,np.nanmean(pc,axis=0),np.nanmean(wmd,axis=0),np.nanmean(memlen),subject)


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

    thresh_pcs = []
    thresh_wmds = []

    for i in range(subject_pcs.shape[1]):
        pc = pearsonr(v[0][fit_mask, i], task_perf[fit_mask])
        wmd = pearsonr(v[1][fit_mask, i], task_perf[fit_mask])
        thresh_pcs.append([pc[0], pc[1]])
        thresh_wmds.append([wmd[0], wmd[1]])

    thresh_pcs = np.array(thresh_pcs)
    thresh_wmds = np.array(thresh_wmds)

    if cv == "True":
        idx = (thresh_pcs[:, 1] < p_thresh) & (thresh_wmds[:, 1] < p_thresh)
        pc_idx = idx
        wmd_idx = idx
    else:
        pc_idx = thresh_pcs[:, 1] < p_thresh
        wmd_idx = thresh_wmds[:, 1] < p_thresh

    subject_pcs = subject_pcs[:, pc_idx]
    subject_wmds = subject_wmds[:, wmd_idx]

    if len(subject_pcs[1]) < 3 or len(subject_wmds[1]) < 3:
        return 0, np.nan, np.nan, np.nan, np.nan

    if use_matrix == True:
        flat_matrices = np.zeros((subject_pcs.shape[0], len(np.tril_indices(368, -1)[0])))
        for s in range(subject_pcs.shape[0]):
            m = task_matrices[s]
            flat_matrices[s] = m[np.tril_indices(368, -1)]
        perf_edge_corr = generate_correlation_map(task_perf[fit_mask].reshape(1, -1), flat_matrices[fit_mask].transpose())[0]

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
    # model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
    
    reg = model.fit(pvals[train], task_perf[train])
    coef = reg.coef_

    result = model.predict(pvals[t].reshape(1, -1))[0]
    if return_features == True:
        return pvals[t], result
    return result, pc_idx, wmd_idx, es_idx, coef

