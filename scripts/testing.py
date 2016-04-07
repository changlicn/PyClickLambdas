import numpy as np
import cPickle as pickle

with open('./data/model_query_softmax_lambdas_v2_collection.pkl') as ifile:
    MQD = pickle.load(ifile)

# For reproducibility -- re-seed the click models' RNGs.
for click_model_type in MQD:
    for query in MQD[click_model_type]:
        MQD[click_model_type][query]['model'].seed = 42
    relevances = MQD[click_model_type][query]['relevances']


np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
q = '2548'
lambdas = MQD['UBM'][q]['stats'][1000000]['lambdas'][0]
N_tot = MQD['UBM'][q]['stats'][1000000]['total_counts'][0]
N_view = MQD['UBM'][q]['stats'][1000000]['viewed_counts'][0]
print lambdas; print N_tot; print N_view
relevances = MQD['UBM'][q]['relevances']
deltas = relevances[:, None] - relevances[None, :]
for i in range(10):
    for j in range(10):
      if deltas[i,j] > 0.01: 
        print i,j, (lambdas[i,j]/N_tot[i,j])/ deltas[i,j], (lambdas[i,j]/N_tot[i,j]-lambdas[j,i]/N_tot[j,i]) / deltas[i,j], relevances[i], relevances[j]
gammas = MQD['UBM']['2548']['model'].p_examination
print gammas.T[[9]+range(9),:]
print MQD['UBM'][q]['stats'][1000000]['cutoff']