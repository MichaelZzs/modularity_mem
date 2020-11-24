import brain_graphs
import scipy.io as sio
import numpy as np
import pandas as pd
from datetime import date
import os

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


# correlation mat, behavioral df generated in analysis
corr_mat = sio.loadmat('/gpfs/milgram/project/chun/ql225/HCP_RecMem/derivatives/WM_avg_mats_368_20200506.mat')
behav_df = pd.read_csv('/gpfs/milgram/project/chun/mz456/modularity_newest/708_sub_df.csv')

# overlap bt behavioral df and corr mats
matching_inds = pd.Series(np.squeeze(corr_mat['WM_sub_list'])).isin(behav_df['sub_id'])
sub_mats = corr_mat['sub_mats_avg_WM'][:,:,matching_inds]

# format input for func
mods = []  # modularity
pcs = []  # participation coefficient
wmds = []  # within module degree
memlens = []  # community sizes
subs = []  # subject ids
for i in range(np.shape(sub_mats)[2]):
    sub_mat = np.triu(sub_mats[:,:,i], 1) + np.triu(sub_mats[:,:,i], 1).transpose()
    variables = [behav_df['sub_id'][i], sub_mat]
    results = individual_graph_analyes_wc(variables)
    mods.append(results[0])
    pcs.append(results[1])
    wmds.append(results[2])
    memlens.append(results[3])
    subs.append(results[4])

# save file
outpath = '/gpfs/milgram/scratch60/chun/mz456/modularitymem_newest/'
today = date.today()
timestamp = today.strftime("%m%d%y")
filename = '05_to_15_thresh_368_{}'.format(timestamp)
outfile = os.path.join(outpath, filename)
np.savez(outfile, mod=mods, pc=pcs, wmd=wmds, memlen=memlens, sub=subs)