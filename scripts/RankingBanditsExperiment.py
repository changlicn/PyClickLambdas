#!/usr/bin/python2
# -*- coding: utf-8 -*-

'''
Script for running experiments with different ranking algorithms, queries,
and feedback models.
'''

import numpy as np

import os
import argparse
import cPickle as pickle

import RankingBanditAlgorithm

from joblib import Parallel, delayed


class RankingBanditExperiment(object):
    def __init__(self, click_model_name, query, ranking_model, click_model,
                 n_documents, n_impressions, cutoff, outputdir):
        self.click_model_name = click_model_name
        self.query = query
        self.ranking_model = ranking_model
        self.click_model = click_model
        self.n_documents = n_documents
        self.n_impressions = n_impressions
        self.cutoff = cutoff
        self.outputdir = outputdir

    def get_output_filename(self):
        return '_'.join(map(str, [self.ranking_model.__class__.__name__,
                                  self.click_model_name, self.query,
                                  self.cutoff, self.n_impressions]))

    def execute(self):
        # Used internally by the click model.
        identity = np.arange(self.n_documents, dtype='int32')

        # History of all rankings produced by the model.
        rankings = np.empty((self.n_impressions, self.n_documents),
                           dtype='int32')

        # Run for the specified number of iterations.
        for t in xrange(self.n_impressions):
            # Current ranking vector.
            ranking = rankings[t]

            # Get a ranking based on the current state of the model...
            self.ranking_model.get_ranking(ranking=ranking)

            # get user clicks on that ranking...
            clicks = self.click_model.get_clicks(ranking[:self.cutoff],
                                                 identity)

            # ... and allow the model to learn from them.
            self.ranking_model.set_feedback(ranking, clicks)

        history = {}
        history['query'] = self.query
        history['click_model'] = self.click_model_name
        history['cutoff'] = self.cutoff
        history['ranking_model'] = self.ranking_model.__class__.__name__
        history['rankings'] = rankings

        with open(os.path.join(self.outputdir, self.get_output_filename()), 'wb') as ofile:
            pickle.dump(history, ofile, protocol=-1)


def prepare_click_models(source='./data/model_query_collection.pkl'):
    with open(source) as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_name in MQD:
        for query in MQD[click_model_name]:
            MQD[click_model_name][query]['model'].seed = 42

    return MQD


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    subparsers = parser.add_subparsers(help='choose ranking algorithm', dest='ranking_model')

    for ranker_algorithm_name in RankingBanditAlgorithm.get_available_algorithms():
        ranker_parser = subparsers.add_parser(ranker_algorithm_name)
        getattr(RankingBanditAlgorithm, ranker_algorithm_name).update_parser(ranker_parser)

    parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity level')
    parser.add_argument('-q', '--query', choices=['all'] + MQD['UBM'].keys(), default='all', help='query for which the experiment is executed')
    parser.add_argument('-m', '--click-model', choices=['all'] + MQD.keys(), default='all', help='user model used for generating clicks')
    parser.add_argument('-n', '--n-impressions', type=int, default=1, help='number of impressions')
    parser.add_argument('-c', '--cutoff', type=int, default=10, help='impressions will consist of only this number of documents')
    parser.add_argument('-w', '--n-workers', type=int, default=1, help='number of worker threads')
    parser.add_argument("output", help="output directory")

    return vars(parser.parse_args())


def prepare_experiments(MQD, ranking_model_name, ranking_model_args,
                        click_model_names, queries, n_impressions,
                        cutoff, outputdir):
    '''
    Method that prepares experiments.
    '''
    experiments = []

    ranking_model_args['cutoff'] = cutoff

    for click_model_name in click_model_names:
        for query in queries:
            relevances = MQD[click_model_name][query]['relevances']
            n_documents = len(relevances)

            ranking_model_args['relevances'] = relevances
            ranking_model_args['n_documents'] = n_documents
            ranking_model_args['random_state'] = np.random.RandomState(42)

            ranking_model = getattr(RankingBanditAlgorithm, ranking_model_name)(**ranking_model_args)

            click_model = MQD[click_model_name][query]['model']

            experiments.append(RankingBanditExperiment(click_model_name, query,
                                                       ranking_model, click_model,
                                                       n_documents, n_impressions,
                                                       cutoff, outputdir))
    return experiments


def parallel_helper(obj, methodname, *args, **kwargs):
    '''
    Helper function to avoid pickling problems when using Parallel loops.
    '''
    return getattr(obj, methodname)(*args, **kwargs)


if __name__ == '__main__':
    # Load click models trained for selected queries.
    MQD = prepare_click_models()

    kwargs = parse_command_line_arguments()

    # ===============================================================
    # Get the global (not algorithm specific) command line arguments,
    # which are:

    # ranking model name ...
    ranking_model_name = kwargs.pop('ranking_model')

    # click model name(s) ...
    click_model_names = kwargs.pop('click_model')
    if click_model_names == 'all':
        click_model_names = MQD.keys()
    else:
        click_model_names = [click_model_names]

    # query ID(s) ...
    queries = kwargs.pop('query')
    if queries == 'all':
        queries = MQD['UBM'].keys()
    else:
        queries = [queries]

    # the number of impressions (time steps) ...
    n_impressions = kwargs.pop('n_impressions')

    # the cutoff rank ...
    cutoff = kwargs.pop('cutoff')

    # the number of worker threads ...
    n_jobs = kwargs.pop('n_workers')

    # the verbosity level ...
    verbose = kwargs.pop('verbose')

    # and, finally, the output directory.
    outputdir = kwargs.pop('output')
    # ===============================================================

    # Make sure output path exists.
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Prepare experiments based on the parsed parameters...
    experiments = prepare_experiments(MQD, ranking_model_name, kwargs,
                                      click_model_names, queries, n_impressions,
                                      cutoff, outputdir)

    # and run them, conveniently, in parallel loops.
    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(parallel_helper)(experiment, 'execute')
        for experiment in experiments)