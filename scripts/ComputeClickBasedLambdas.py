# -*- coding: utf-8 -*-

import numpy as np

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler
from samplers import MultinomialRankingSampler

import cPickle as pickle

from joblib import Parallel, delayed, cpu_count

from timeit import default_timer as timer


def compute_lambdas_parallel_v1(click_model_name, query, click_model, scores,
                                n_impressions, n_repeats, sampler_type,
                                store_impressions=False, cutoff=10, seed=31):
    '''
    Compute click-based lambdas in the following fashion.

    For each impression in which documents d_i was presented below document d_j
    and d_i was clicked and d_j not 1 is added to lambda_ij and subtract from
    lambda_ji. The number of times the pair of documents was presented this
    way is kept as total counts. In addition viewed counts is the portion of
    impressions in which the two documents (in the same order) were above
    the last clicked document.

    Based on these lambdas and counts, 2 version of final lambdas are defined:
        1) viewed_lambda_ij = lambda_ij / (viewed_count_ij + viewed_count_ji)
        2) total_lambda_ij = lambda_ij / (total_count_ij + total_count_ji)

    In both cases the lambda values end up being in the range [-1, 1].
    '''
    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')

    sampler = sampler_type(scores, random_state=np.random.RandomState(seed))

    n_documents = len(scores)

    # lambdas[i, j] - number of times document i was clicked and j was above it
    # and NOT clicked (was skipped)!
    lambdas = np.zeros((len(n_impressions), n_repeats,
                        n_documents, n_documents), dtype='float64')

    total_lambdas = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='float64')

    # total_counts[i, j] == number of times document i was presented below
    # document j and both were above the cutoff rank.
    total_counts = np.zeros((len(n_impressions), n_repeats,
                            n_documents, n_documents), dtype='int32')

    viewed_lambdas = np.zeros((len(n_impressions), n_repeats,
                              n_documents, n_documents), dtype='float64')

    # viewed_counts[i, j] == number of times document i was presented below
    # document j and both were above the last clicked rank.
    viewed_counts = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='int32')

    if store_impressions:
        impressions = -np.ones((n_repeats, n_impressions[-1], cutoff),
                               dtype='int32')
    else:
        impressions = None
    
    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            total_counts_ = total_counts[stage][r]
            viewed_counts_ = viewed_counts[stage][r]

            impressions_offset = 0

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                total_counts_ += total_counts[stage - 1][r]
                viewed_counts_ += viewed_counts[stage - 1][r]
                impressions_offset = n_impressions[stage - 1]

            for n in range(n_imps):
                sampler.sample(out=ranking)
                clicks = click_model.get_clicks(ranking[:cutoff], identity)

                if store_impressions:
                    impressions[r, impressions_offset + n] = ranking[:cutoff]

                if clicks.any():
                    last_click_rank = np.where(clicks)[0][-1]
                else:
                    last_click_rank = 0

                for i in range(cutoff - 1):
                    d_i = ranking[i]
                    for j in range(i + 1, cutoff):
                        d_j = ranking[j]

                        if j <= last_click_rank:
                            viewed_counts_[d_j, d_i] += 1.0

                        if clicks[i] < clicks[j]:
                            lambdas_[d_i, d_j] -= 1.0
                            lambdas_[d_j, d_i] += 1.0

                        total_counts_[d_j, d_i] += 1.0

            with np.errstate(invalid='ignore'):
                np.copyto(viewed_lambdas[stage][r], np.nan_to_num(lambdas_ / (viewed_counts_ + viewed_counts_.T)))
                np.copyto(total_lambdas[stage][r], np.nan_to_num(lambdas_ / (total_counts_ + total_counts_.T)))
        
    return (click_model_name, query, cutoff, n_impressions, sampler_type.__name__,
            lambdas, total_counts, viewed_counts, total_lambdas, viewed_lambdas,
            impressions)


def compute_lambdas_parallel_v2(click_model_name, query, click_model, scores,
                                n_impressions, n_repeats, sampler_type,
                                store_impressions=False, cutoff=10, seed=31):
    '''
    Compute click-based lambdas in the following fashion.

    For each impression in which documents d_i was presented below document d_j
    and d_i was clicked and d_j not 1 is added to lambda_ij. The number of times
    the pair of documents was presented this way is kept as total counts. 
    In addition viewed counts is the portion of impressions in which the two
    documents (in the same order) were above the last clicked document.

    Based on these lambdas and counts, 2 version of final lambdas are defined:
        1) viewed_lambda_ij = (lambda_ij / viewed_count_ij) - (lambda_ji / viewed_count_ji)
        2) total_lambda_ij = (lambda_ij / total_count_ij) - (lambda_ji / total_count_ji)

    In both cases the lambda values end up being in the range [-1, 1].
    '''
    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')

    sampler = sampler_type(scores, random_state=np.random.RandomState(seed))

    n_documents = len(scores)

    # lambdas[i, j] - number of times document i was clicked and j was above it
    # and NOT clicked (was skipped)!
    lambdas = np.zeros((len(n_impressions), n_repeats,
                        n_documents, n_documents), dtype='float64')

    total_lambdas = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='float64')

    # total_counts[i, j] == number of times document i was presented below
    # document j and both were above the cutoff rank.
    total_counts = np.zeros((len(n_impressions), n_repeats,
                            n_documents, n_documents), dtype='int32')

    viewed_lambdas = np.zeros((len(n_impressions), n_repeats,
                              n_documents, n_documents), dtype='float64')

    # viewed_counts[i, j] == number of times document i was presented below
    # document j and both were above the last clicked rank.
    viewed_counts = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='int32')

    if store_impressions:
        impressions = -np.ones((n_repeats, n_impressions[-1], cutoff),
                               dtype='int32')
    else:
        impressions = None

    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            total_counts_ = total_counts[stage][r]
            viewed_counts_ = viewed_counts[stage][r]

            impressions_offset = 0

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                total_counts_ += total_counts[stage - 1][r]
                viewed_counts_ += viewed_counts[stage - 1][r]
                impressions_offset = n_impressions[stage - 1]

            for n in range(n_imps):
                sampler.sample(out=ranking)
                clicks = click_model.get_clicks(ranking[:cutoff], identity)

                if store_impressions:
                    impressions[r, impressions_offset + n] = ranking[:cutoff]

                if clicks.any():
                    last_click_rank = np.where(clicks)[0][-1]
                else:
                    last_click_rank = 0

                for i in range(cutoff - 1):
                    d_i = ranking[i]
                    for j in range(i + 1, cutoff):
                        d_j = ranking[j]

                        if j <= last_click_rank:
                            viewed_counts_[d_j, d_i] += 1.0

                        if clicks[i] < clicks[j]:
                            lambdas_[d_j, d_i] += 1.0

                        total_counts_[d_j, d_i] += 1.0

            with np.errstate(invalid='ignore'):
                np.copyto(viewed_lambdas[stage][r], np.nan_to_num(lambdas_ / viewed_counts_) - np.nan_to_num(lambdas_.T / viewed_counts_.T))
                np.copyto(total_lambdas[stage][r], np.nan_to_num(lambdas_ / total_counts_) - np.nan_to_num(lambdas_.T / total_counts_.T))
        
    return (click_model_name, query, cutoff, n_impressions, sampler_type.__name__,
            lambdas, total_counts, viewed_counts, total_lambdas, viewed_lambdas,
            impressions)


def compute_lambdas_parallel_v3(click_model_name, query, click_model, scores,
                                n_impressions, n_repeats, sampler_type,
                                store_impressions=False, cutoff=10, seed=31):
    '''
    Compute click-based lambdas in the following fashion.

    For each impression in which documents d_i was presented below document d_j
    and d_i was clicked and d_j not or d_i was presented above d_j and d_i was
    clicked and d_j not while both these documents were above another clicked
    document 1 is added to lambda_ij and subtract from lambda_ji. The number of
    times the pair of documents was presented this way is kept as total counts.
    In addition viewed counts is the portion of impressions in which the two
    documents (in the same order) were above the last clicked document.

    Based on these lambdas and counts, 2 version of final lambdas are defined:
        1) viewed_lambda_ij = lambda_ij / (viewed_count_ij + viewed_count_ji)
        2) total_lambda_ij = lambda_ij / (total_count_ij + total_count_ji)

    In both cases the lambda values end up being in the range [-1, 1].
    '''
    if not isinstance(n_impressions, (tuple, list)):
        n_impressions = (n_impressions,)

    n_stage_impressions = np.diff(np.r_[0, n_impressions])

    if (n_stage_impressions < 0).any():
        raise ValueError('`n_impressions` must be in increasing order')

    sampler = sampler_type(scores, random_state=np.random.RandomState(seed))

    n_documents = len(scores)

    # lambdas[i, j] - number of times document i was clicked and j was above it
    # and NOT clicked (was skipped)!
    lambdas = np.zeros((len(n_impressions), n_repeats,
                        n_documents, n_documents), dtype='float64')

    total_lambdas = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='float64')

    # total_counts[i, j] == number of times document i was presented below
    # document j and both were above the cutoff rank.
    total_counts = np.zeros((len(n_impressions), n_repeats,
                            n_documents, n_documents), dtype='int32')

    viewed_lambdas = np.zeros((len(n_impressions), n_repeats,
                              n_documents, n_documents), dtype='float64')

    # viewed_counts[i, j] == number of times document i was presented below
    # document j and both were above the last clicked rank.
    viewed_counts = np.zeros((len(n_impressions), n_repeats,
                             n_documents, n_documents), dtype='int32')

    if store_impressions:
        impressions = -np.ones((n_repeats, n_impressions[-1], cutoff),
                               dtype='int32')
    else:
        impressions = None

    ranking = np.arange(n_documents, dtype='int32')
    identity = np.arange(n_documents, dtype='int32')

    for stage, n_imps in enumerate(n_stage_impressions):
        for r in range(n_repeats):
            lambdas_ = lambdas[stage][r]
            total_counts_ = total_counts[stage][r]
            viewed_counts_ = viewed_counts[stage][r]

            impressions_offset = 0

            # Avoid unnecessarry recomputation of the statistics
            # by reusing it from previous stage.
            if stage > 0:
                lambdas_ += lambdas[stage - 1][r]
                total_counts_ += total_counts[stage - 1][r]
                viewed_counts_ += viewed_counts[stage - 1][r]
                impressions_offset = n_impressions[stage - 1]

            for n in range(n_imps):
                sampler.sample(out=ranking)
                clicks = click_model.get_clicks(ranking[:cutoff], identity)

                if store_impressions:
                    impressions[r, impressions_offset + n] = ranking[:cutoff]

                if clicks.any():
                    last_click_rank = np.where(clicks)[0][-1]
                else:
                    last_click_rank = 0

                for i in range(cutoff - 1):
                    d_i = ranking[i]
                    for j in range(i + 1, cutoff):
                        d_j = ranking[j]

                        if j <= last_click_rank:
                            viewed_counts_[d_j, d_i] += 1.0

                            if clicks[i] > clicks[j]:
                                lambdas_[d_i, d_j] += 1.0
                                lambdas_[d_j, d_i] -= 1.0

                        if clicks[i] < clicks[j]:
                            lambdas_[d_i, d_j] -= 1.0
                            lambdas_[d_j, d_i] += 1.0

                        total_counts_[d_j, d_i] += 1.0

            with np.errstate(invalid='ignore'):
                np.copyto(viewed_lambdas[stage][r], np.nan_to_num(lambdas_ / (viewed_counts_ + viewed_counts_.T)))
                np.copyto(total_lambdas[stage][r], np.nan_to_num(lambdas_ / (total_counts_ + total_counts_.T)))

    return (click_model_name, query, cutoff, n_impressions, sampler_type.__name__,
            lambdas, total_counts, viewed_counts, total_lambdas, viewed_lambdas,
            impressions)


def compute_lambdas(MQD, click_models, n_repeats, n_impressions, compute_lambdas_method,
                    ranking_sampler, cutoff, store_impressions, output_filepath):
    # Run the computation of lambdas in paralell with the specified
    # `compute_lambdas_method` method and `ranking_sampler`.
    lambdas_counts = Parallel(n_jobs=cpu_count())(
                        delayed(compute_lambdas_method)(
                            click_model_name, query,
                            MQD[click_model_name][query]['model'],
                            MQD[click_model_name][query]['relevances'],
                            n_impressions,
                            n_repeats,
                            ranking_sampler,
                            store_impressions,
                            cutoff)
                        for click_model_name in click_models
                        for query in MQD[click_model_name].keys())

    # Copy the lambdas and counts into a dictionary. Note that Parallel
    # preserves the order of the results, but we still keep track of the
    # associated click model and queries names not to mix something up.
    for stats in lambdas_counts:
        click_model_name, query, cutoff, n_imps, ranking_sampler_name,\
        lambdas, total_counts, viewed_counts, total_lambdas, viewed_lambdas,\
        impressions = stats

        MQD[click_model_name][query]['stats'] = {}

        for i, n in enumerate(n_imps):
            MQD[click_model_name][query]['stats'][n] = {'lambdas': lambdas[i],
                                                        'total_lambdas': total_lambdas[i],
                                                        'viewed_lambdas': viewed_lambdas[i],
                                                        'total_counts': total_counts[i],
                                                        'viewed_counts': viewed_counts[i],
                                                        'cutoff': cutoff,
                                                        'impressions': impressions,
                                                        'ranking_sampler': ranking_sampler_name}

    with open(output_filepath, 'wb') as ofile:
        pickle.dump(MQD, ofile, protocol=-1)


if __name__ == '__main__':
    # Load the click models for the queries of interest.
    with open('./data/model_query_collection.pkl') as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_name in MQD:
        for query in MQD[click_model_name]:
            MQD[click_model_name][query]['model'].seed = 42

    # The click models for which the lambdas will be computed.
    click_models = ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']

    # Number of estimates of lambas.
    n_repeats = 10

    # Total number of impressions for each query.
    n_impressions = [1000, 2100, 4500, 10000, 21000,
                     45000, 100000, 210000, 450000, 1000000]

    # The method responsible for computation of lambdas.
    compute_lambdas_method = compute_lambdas_parallel_v1
    # compute_lambdas_method = compute_lambdas_parallel_v2
    # compute_lambdas_method = compute_lambdas_parallel_v3

    # The specific ranking sampler.
    # ranking_sampler = UniformRankingSampler
    # ranking_sampler = SoftmaxRankingSampler
    ranking_sampler = MultinomialRankingSampler

    # The cutoff rank - the maximum number of 'visible' documents.
    cutoff = 5

    # If True, the output will contain the impressions from which
    # the lambdas were computed.
    store_impressions = False

    start = timer()

    compute_lambdas(MQD, click_models, n_repeats,
                    n_impressions, compute_lambdas_method,
                    ranking_sampler, cutoff, store_impressions,
                    './data/model_query_multinomial_lambdas_v1_collection_c5.pkl')

    end = timer()

    print 'Elapsed time: %.2fs' % (end - start)
