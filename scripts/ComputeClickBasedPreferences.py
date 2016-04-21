# -*- coding: utf-8 -*-

import numpy as np

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler
from samplers import MultinomialRankingSampler

import cPickle as pickle

from joblib import Parallel, delayed, cpu_count

from itertools import permutations

from timeit import default_timer as timer


def compute_preferences_parallel(click_model_name, query, click_model,
                                 relevances, cutoff):
    n_documents = len(relevances)
    identity = np.arange(n_documents, dtype='int32')
    preferences = np.zeros((n_documents, n_documents), dtype='float64')

    for r_ij in permutations(range(len(relevances)), cutoff):
        c_j = click_model.get_clickthrough_rate(r_ij[:-1], identity)
        c_i = click_model.get_clickthrough_rate(r_ij, identity)
        preferences[r_ij[-1], r_ij[-2]] = (c_i - c_j)

    return click_model_name, query, preferences


def compute_preferences(MQD, click_models, cutoff, output_filepath):
    preferences = Parallel(n_jobs=cpu_count())(
                        delayed(compute_preferences_parallel)(
                            click_model_name, query,
                            MQD[click_model_name][query]['model'],
                            MQD[click_model_name][query]['relevances'],
                            cutoff)
                        for click_model_name in click_models
                        for query in MQD[click_model_name].keys())

    for stats in preferences:
        click_model_name, query, prefs = stats
        MQD[click_model_name][query]['preferences'] = prefs

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

    # The click models for which the preferences will be computed.
    click_models = ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']

    # The cutoff rank - the maximum number of 'visible' documents.
    cutoff = 5

    start = timer()

    compute_preferences(MQD, click_models, cutoff, './data/model_query_preferences_c5.pkl')

    end = timer()

    print 'Elapsed time: %.2fs' % (end - start)

    print 'Checking validity of preferences:'

    for click_model_name in click_models:
        for query in MQD[click_model_name]:
            preferences = MQD[click_model_name][query]['preferences']
            deltas = MQD[click_model_name][query]['relevances']
            deltas = deltas[:, None] - deltas[None, :]

            print '%s check: %s' % (click_model_name,
                                    not (preferences * deltas >= 0).all())