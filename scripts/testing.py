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


Docs = range(10)
for ind,u in enumerate(Docs):
    for v in Docs[i+1:]:
        for x in set(Docs) - set([u,v]):
            Lambda[u,v] += \
                (1-gamma[0,1]*r[v]) * gamma[0,2] * r[u] * n[v,u,x]/N_tot[v,u] - \
                (1-gamma[0,1]*r[u]) * gamma[0,2] * r[v] * n[u,v,x]/N_tot[u,v] + \
                (1-gamma[0,1]*r[v]) * (1-gamma[0,2]*r[x]) * gamma[0,3] * r[u] * n[v,x,u]/N_tot[v,u] - \
                (1-gamma[0,1]*r[u]) * (1-gamma[0,2]*r[x]) * gamma[0,3] * r[v] * n[u,x,v]/N_tot[u,v] + \
                (1-gamma[0,1]*r[x]) * (1-gamma[0,2]*r[v]) * gamma[0,3] * r[u] * n[x,v,u]/N_tot[v,u] - \
                (1-gamma[0,1]*r[x]) * (1-gamma[0,2]*r[u]) * gamma[0,3] * r[v] * n[x,u,v]/N_tot[u,v] + \
                (1-gamma[0,1]*r[v]) * gamma[0,2] * r[x] * gamma[2,3] * r[u] * n[v,x,u]/N_tot[v,u] - \
                (1-gamma[0,1]*r[u]) * gamma[0,2] * r[x] * gamma[2,3] * r[v] * n[u,x,v]/N_tot[u,v] + \
                gamma[0,1] * r[x] * (1-gamma[1,2]*r[v]) * gamma[1,3] * r[u] * n[x,v,u]/N_tot[v,u] - \
                gamma[0,1] * r[x] * (1-gamma[1,2]*r[u]) * gamma[1,3] * r[v] * n[x,u,v]/N_tot[u,v]

