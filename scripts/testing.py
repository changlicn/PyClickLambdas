import numpy as np
np.set_printoptions(edgeitems=3,infstr='inf',linewidth=9999, nanstr='nan', precision=3, suppress=False, threshold=1000, formatter=None)
import cPickle as pkl
from collections import Counter
import gzip as gz

# with open('./data/model_query_softmax_lambdas_v2_collection.pkl') as ifile:
#     MQD = pickle.load(ifile)

# # For reproducibility -- re-seed the click models' RNGs.
# for click_model_type in MQD:
#     for query in MQD[click_model_type]:
#         MQD[click_model_type][query]['model'].seed = 42
#     relevances = MQD[click_model_type][query]['relevances']


# np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
# q = '2548'
# lambdas = MQD['UBM'][q]['stats'][1000000]['lambdas'][0]
# N_tot = MQD['UBM'][q]['stats'][1000000]['total_counts'][0]
# N_view = MQD['UBM'][q]['stats'][1000000]['viewed_counts'][0]
# print lambdas; print N_tot; print N_view
# relevances = MQD['UBM'][q]['relevances']
# deltas = relevances[:, None] - relevances[None, :]
# for i in range(10):
#     for j in range(10):
#       if deltas[i,j] > 0.01: 
#         print i,j, (lambdas[i,j]/N_tot[i,j])/ deltas[i,j], (lambdas[i,j]/N_tot[i,j]-lambdas[j,i]/N_tot[j,i]) / deltas[i,j], relevances[i], relevances[j]
# gammas = MQD['UBM']['2548']['model'].p_examination
# print gammas.T[[9]+range(9),:]
# print MQD['UBM'][q]['stats'][1000000]['cutoff']


def count_impressions_of_length_3(Imps):
    # This method takes in a list of tuples, where each tuple consists of three
    # numbers between 0 and 9.
    C = Counter(Imps)
    N_uvx = np.zeros([10,10,10])
    for k in C:
        N_uvx[k[0],k[1],k[2]] = C[k]
    N_uv = np.zeros([10,10])
    for k in C:
        for i,j in [(0,1),(0,2),(1,2)]:
            N_uv[k[i],k[j]] += C[k]
    return N_uvx, N_uv


fh = gz.open('data/model_query_impressions_multinomial_lambdas_v2_collection_c3.pkl.gz'); Data = pkl.load(fh); fh.close()
Imps = Data['UBM']['2548']['stats'][1000000]['impressions'][0]
tImps = [tuple(imp) for imp in Imps]
n_uvx, n_uv = count_impressions_of_length_3(tImps)
r = Data['UBM']['2548']['relevances']
gammas = Data['UBM']['2548']['model'].p_examination
gamma = gammas.T[[9]+range(9),:]

max_ratio = 0
min_ratio = 100
Docs = range(10)
for ind,u in enumerate(Docs):
    for v in Docs:
      if r[u]-r[v] > 0:
        print "u , v =",u,",",v
        t1 = (gamma[0,2]+2*gamma[0,3]) * (r[u] - r[v])
        print t1, "=", gamma[0,2]+2*gamma[0,3], "x", r[u] - r[v]
        t2 = (gamma[2,3]-gamma[0,3]) * gamma[0,2] * ( (r * (r[u] * n_uvx[v,:,u]/n_uvx[v,:,u].sum() - r[v] * n_uvx[u,:,v]/n_uvx[u,:,v].sum())).sum()
                                                      -gamma[0,1] * r[u] * r[v] * (r * (n_uvx[v,:,u]/n_uvx[v,:,u].sum() - n_uvx[u,:,v]/n_uvx[u,:,v].sum())).sum() )
        print t2, "=", (gamma[2,3]-gamma[0,3]) * gamma[0,2], "x (", (r * (r[u] * n_uvx[v,:,u]/n_uvx[v,:,u].sum() - r[v] * n_uvx[u,:,v]/n_uvx[u,:,v].sum())).sum(), 
        print "+",-gamma[0,1] * r[u] * r[v] * (r * (n_uvx[v,:,u]/n_uvx[v,:,u].sum() - n_uvx[u,:,v]/n_uvx[u,:,v].sum())).sum(), ")"
        t3 = (gamma[1,3]-gamma[0,3]) * gamma[0,1] * ( (r * (r[u] * n_uvx[:,v,u]/n_uvx[:,v,u].sum() - r[v] * n_uvx[:,u,v]/n_uvx[:,u,v].sum())).sum()
                                                      -gamma[1,2] * r[u] * r[v] * (r * (n_uvx[:,v,u]/n_uvx[:,v,u].sum() - n_uvx[:,u,v]/n_uvx[:,u,v].sum())).sum() )
        print t3, "=",(gamma[1,3]-gamma[0,3]) * gamma[0,1], "x (", (r * (r[u] * n_uvx[:,v,u]/n_uvx[:,v,u].sum() - r[v] * n_uvx[:,u,v]/n_uvx[:,u,v].sum())).sum(), 
        print "+",-gamma[1,2] * r[u] * r[v] * (r * (n_uvx[:,v,u]/n_uvx[:,v,u].sum() - n_uvx[:,u,v]/n_uvx[:,u,v].sum())).sum(), ")"
        print 80*'-'
        print t1+t2+t3, "=", (t1+t2+t3)/(r[u]-r[v]), "x", r[u]-r[v]
        print 80*'-'


            # print "n_uvx[%d,%d,%d]/n_uv[%d,%d] =" % (u,x,v,u,v), n_uvx[u,x,v]/n_uv[u,v], 
            # print "    n_uvx[%d,%d,%d]/n_uv[%d,%d] =" % (v,x,u,v,u), n_uvx[v,x,u]/n_uv[v,u],
            # ratio = (n_uvx[u,x,v]/n_uv[u,v])/(n_uvx[v,x,u]/n_uv[v,u])
            # max_ratio = max(max_ratio,ratio)
            # min_ratio = min(min_ratio,ratio)
            # print "    ratio =", ratio
            # Lambda[u,v] += \
            #     (1-gamma[0,1]*r[v]) * gamma[0,2] * r[u] * n_uvx[v,u,x]/n_uv[v,u] - \
            #     (1-gamma[0,1]*r[u]) * gamma[0,2] * r[v] * n_uvx[u,v,x]/n_uv[u,v] + \
            #     (1-gamma[0,1]*r[v]) * (1-gamma[0,2]*r[x]) * gamma[0,3] * r[u] * n_uvx[v,x,u]/n_uv[v,u] - \
            #     (1-gamma[0,1]*r[u]) * (1-gamma[0,2]*r[x]) * gamma[0,3] * r[v] * n_uvx[u,x,v]/n_uv[u,v] + \
            #     (1-gamma[0,1]*r[x]) * (1-gamma[0,2]*r[v]) * gamma[0,3] * r[u] * n_uvx[x,v,u]/n_uv[v,u] - \
            #     (1-gamma[0,1]*r[x]) * (1-gamma[0,2]*r[u]) * gamma[0,3] * r[v] * n_uvx[x,u,v]/n_uv[u,v] + \
            #     (1-gamma[0,1]*r[v]) * gamma[0,2] * r[x] * gamma[2,3] * r[u] * n_uvx[v,x,u]/n_uv[v,u] - \
            #     (1-gamma[0,1]*r[u]) * gamma[0,2] * r[x] * gamma[2,3] * r[v] * n_uvx[u,x,v]/n_uv[u,v] + \
            #     gamma[0,1] * r[x] * (1-gamma[1,2]*r[v]) * gamma[1,3] * r[u] * n_uvx[x,v,u]/n_uv[v,u] - \
            #     gamma[0,1] * r[x] * (1-gamma[1,2]*r[u]) * gamma[1,3] * r[v] * n_uvx[x,u,v]/n_uv[u,v]

