# -*- coding: utf-8 -*-

import numpy as np

import cPickle as pickle

from joblib import Parallel, delayed, cpu_count


def compute_lambdas_parallel(click_model_type, query, click_model, n_documents,
                             n_impressions, n_repeats, seed=42):

    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')
    
    random_state = np.random.RandomState(seed)

    lambdas = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='float64')
    lcounts = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='int32')
    
    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            lcounts_ = lcounts[stage][r]

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                lcounts_ += lcounts[stage - 1][r]

            for n in range(n_imps):
                # SHUFFLING THE RANKING
                random_state.shuffle(ranking)
                clicks = click_model.get_clicks(ranking, identity)

                if clicks.any():
                    last_click_rank = np.where(clicks)[0][-1]
                    for i in range(last_click_rank):
                        d_i = ranking[i]
                        for j in range(i + 1, last_click_rank + 1):
                                if clicks[i] < clicks[j]:
                                    d_j = ranking[j]
                                    lambdas_[d_i, d_j] -= 1.0
                                    lambdas_[d_j, d_i] += 1.0
                                    lcounts_[d_j, d_i] += 1

            with np.errstate(invalid='ignore'):
                np.copyto(lambdas_, np.nan_to_num(lambdas_ / (lcounts_ + lcounts_.T)))
        
    return click_model_type, query, n_impressions, lambdas, lcounts


if __name__ == '__main__':
    # Load the click models for the queries of interest.
    with open('./model_query_collection.pkl') as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_type in MQD:
        for query in MQD[click_model_type]:
            MQD[click_model_type][query]['model'].seed = 42

    # Number of estimates of lambas.
    n_repeats = 10

    # Total number of impressions for each query.
    n_impressions = [1000, 2100, 4500, 10000, 21000,
                     45000, 100000, 210000, 450000, 1000000]

    lambdas_counts = Parallel(n_jobs=cpu_count())(
                        delayed(compute_lambdas_parallel)(
                            click_model_type, query,
                            MQD[click_model_type][query]['model'],
                            len(MQD[click_model_type][query]['relevances']),
                            n_impressions, n_repeats)
                        for click_model_type in ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']
                        for query in MQD[click_model_type].keys())

    # Copy the lambdas and counts into the dictionary.
    # Parallel preserves the order of the results.
    for stats in lambdas_counts:
        click_model_type, query, n_imps, lambdas, lcounts = stats
        MQD[click_model_type][query]['stats'] = {}
        for i, n in enumerate(n_imps):
            MQD[click_model_type][query]['stats'][n] = {'lambdas': lambdas[i],
                                                        'lcounts': lcounts[i]}

    with open('./model_query_lambdas_10reps_collection.pkl', 'wb') as ofile:
        pickle.dump(MQD, ofile, protocol=-1)
