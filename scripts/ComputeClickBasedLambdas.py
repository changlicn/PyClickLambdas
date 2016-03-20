# -*- coding: utf-8 -*-

import numpy as np

from RankingSampler import RankingSampler

import cPickle as pickle

from joblib import Parallel, delayed, cpu_count


def compute_uniform_lambdas_parallel(click_model_type, query, click_model, n_documents,
                                     n_impressions, n_repeats, seed=42):

    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')
    
    random_state = np.random.RandomState(seed)

    lambdas = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='float64')
    total_counts = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='int32')
    
    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            total_counts_ = total_counts[stage][r]

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                total_counts_ += total_counts[stage - 1][r]

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
                                    total_counts_[d_j, d_i] += 1

            with np.errstate(invalid='ignore'):
                np.copyto(lambdas_, np.nan_to_num(lambdas_ / (total_counts_ + total_counts_.T)))
        
    return click_model_type, query, n_impressions, lambdas, total_counts, None


def compute_nonuniform_lambdas_parallel(click_model_type, query, click_model,
                                        scores, n_impressions, n_repeats,
                                        cutoff=5, seed=42):

    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')

    random_state = np.random.RandomState(seed)

    n_documents = len(scores)

    # lambdas[i, j] == # of times document i was clicked and j was above it
    # and NOT clicked!
    lambdas = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='float64')

    # total_counts[i, j] == # of times document i was presented below document j
    # and both were above the last clicked rank.
    total_counts = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='int32')

    # viewed_counts[i, j] == # of times document i was presented below document j
    # and both were above the cutoff rank.
    viewed_counts = np.zeros((len(n_impressions), n_repeats, n_documents, n_documents), dtype='int32')

    sampler = RankingSampler(scores)

    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            total_counts_ = total_counts[stage][r]
            viewed_counts_ = viewed_counts[stage][r]

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                total_counts_ += total_counts[stage - 1][r]
                viewed_counts_ += viewed_counts[stage - 1][r]

            for n in range(n_imps):
                # Sample a ranking using 'softmax' Plackett-Luce model.
                sampler.softmax_ranking(out=ranking)
                clicks = click_model.get_clicks(ranking[:cutoff], identity)

                if clicks.any():
                    last_considered_rank = np.where(clicks)[0][-1]
                else:
                    last_considered_rank = 0

                for i in range(cutoff - 1):
                    d_i = ranking[i]
                    for j in range(i + 1, cutoff):
                        d_j = ranking[j]

                        if j < last_considered_rank:
                            total_counts_[d_j, d_i] += 1

                        if clicks[i] < clicks[j]:
                            lambdas_[d_j, d_i] += 1.0

                        viewed_counts_[d_j, d_i] += 1.0
        
    return click_model_type, query, n_impressions, lambdas, total_counts, viewed_counts


if __name__ == '__main__':
    # Load the click models for the queries of interest.
    with open('./data/model_query_collection.pkl') as ifile:
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

    lambdas_type = 'non-uniform'

    if lambdas_type == 'uniform':
        lambdas_counts = Parallel(n_jobs=cpu_count())(
                            delayed(compute_uniform_lambdas_parallel)(
                                click_model_type, query,
                                MQD[click_model_type][query]['model'],
                                len(MQD[click_model_type][query]['relevances']),
                                n_impressions, n_repeats)
                            for click_model_type in ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']
                            for query in MQD[click_model_type].keys())

    elif lambdas_type == 'non-uniform':
        lambdas_counts = Parallel(n_jobs=cpu_count())(
                            delayed(compute_nonuniform_lambdas_parallel)(
                                click_model_type, query,
                                MQD[click_model_type][query]['model'],
                                MQD[click_model_type][query]['relevances'],
                                n_impressions, n_repeats)
                            for click_model_type in ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']
                            for query in MQD[click_model_type].keys())

    # Copy the lambdas and counts into the dictionary.
    # Parallel preserves the order of the results.
    for stats in lambdas_counts:
        click_model_type, query, n_imps, lambdas, total_counts, viewed_counts = stats
        MQD[click_model_type][query]['stats'] = {}
        for i, n in enumerate(n_imps):
            MQD[click_model_type][query]['stats'][n] = {'lambdas': lambdas[i],
                                                        'total_counts': total_counts[i],
                                                        'viewed_counts': None if viewed_counts is None else viewed_counts[i]}

    if lambdas_type == 'uniform':
        with open('./data/model_query_uniform_lambdas_10reps_collection.pkl', 'wb') as ofile:
            pickle.dump(MQD, ofile, protocol=-1)
    elif lambdas_type == 'non-uniform':
        with open('./data/model_query_nonuniform_lambdas_10reps_collection.pkl', 'wb') as ofile:
            pickle.dump(MQD, ofile, protocol=-1)