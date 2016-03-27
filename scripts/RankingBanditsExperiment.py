#!/usr/bin/python2
# -*- coding: utf-8 -*-

'''
Script for running experiments with different ranking algorithms, queries,
and feedback models.
'''

import numpy as np

import argparse
import cPickle as pickle

import ClickLambdasAlgorithm
import RankingBanditAlgorithm

from timeit import default_timer as timer


class RankingBanditExperiment(object):
    def __init__(self, ranking_model, click_model, n_documents,
                 n_impressions, cutoff):
        self.ranking_model = ranking_model
        self.click_model = click_model
        self.n_documents = n_documents
        self.n_impressions = n_impressions
        self.cutoff = cutoff

    def execute(self):
        # Used internally by the click model.
        identity = np.arange(self.n_documents, dtype='int32')

        ranking = np.arange(self.n_documents, dtype='int32')

        for t in xrange(self.n_impressions):
            self.ranking_model.advance(ranking)

            clicks = self.click_model.get_clicks(ranking[:self.cutoff],
                                                 identity)

            self.ranking_model.feedback(ranking, clicks)


def prepare_click_models(source='./data/model_query_collection.pkl'):
    with open(source) as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_name in MQD:
        for query in MQD[click_model_name]:
            MQD[click_model_name][query]['model'].seed = 42

    return MQD


def prepare_experiments(MQD, ranking_model_name, ranking_model_args,
                        click_model_names, queries, n_impressions,
                        cutoff):
    '''
    Method that prepares experiments.
    '''
    experiments = []
    ranking_model_args_orig = ranking_model_args

    for click_model_name in click_model_names:
        for query in queries:
            relevances = MQD[click_model_name][query]['relevances']
            n_documents = len(relevances)

            ranking_model_args = ranking_model_args_orig.copy()
            ranking_model_args['random_state'] = np.random.RandomState(42)
            ranking_model_args['n_documents'] = n_documents

            ranking_model = getattr(RankingBanditAlgorithm, ranking_model_name)(**ranking_model_args)
            click_model = MQD[click_model_name][query]['model']

            experiments.append(RankingBanditExperiment(ranking_model, click_model,
                                                       n_documents, n_impressions,
                                                       cutoff))

    return experiments


if __name__ == '__main__':
    # Load click models trained for selected queries.
    MQD = prepare_click_models()

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    subparsers = parser.add_subparsers(help='choose ranking algorithm', dest='ranking_model')

    uniform_parser = subparsers.add_parser('UniformRankingAlgorithm')

    softmax_parser = subparsers.add_parser('SoftmaxRakingAlgorithm')

    cascade_ucb1_parser = subparsers.add_parser('CascadeUCB1Algorithm')
    cascade_ucb1_parser.add_argument('-a', '--alpha', type=float, default=1.5, required=True, help='alpha parameter')

    cascade_klucb_parser = subparsers.add_parser('CascadeKLUCBAlgorithm')

    cascade_ts_parser = subparsers.add_parser('CascadeThompsonSamplerAlgorithm')
    cascade_ts_parser.add_argument('-a', '--alpha', type=float, default=1.0, required=True, help='alpha parameter')
    cascade_ts_parser.add_argument('-b', '--beta', type=float, default=1.0, required=True, help='beta parameter')

    cascade_exp3_parser  = subparsers.add_parser('CascadeExp3Algorithm')
    cascade_exp3_parser.add_argument('-g', '--gamma', type=float, default=0.01, required=True, help='gamma parameter')

    copeland_parser = subparsers.add_parser('CopelandRakingAlgorithm')
    copeland_parser.add_argument('-a', '--alpha', type=float, default=0.51, required=True, help='alpha parameter')
    copeland_parser.add_argument('-f', '--feedback', choices=['SkipClickLambdasAlgorithm'], required=True, help='feedback model')

    parser.add_argument('-q', '--query', nargs='+', choices=['all'] + MQD['UBM'].keys(), default='all', help='query for which the experiment is executed')
    parser.add_argument('-m', '--click-model', nargs='+', choices=['all'] + MQD.keys(), default='all', help='user model used for generating clicks')
    parser.add_argument('-n', '--n-impressions', type=int, default=1, help='number of impressions')
    parser.add_argument('-c', '--cutoff', type=int, help='impressions will consist of only this number of documents')
    parser.add_argument("output", help="output directory")

    ranking_model_args = vars(parser.parse_args())

    ranking_model_name = ranking_model_args.pop('ranking_model')

    click_model_names = ranking_model_args.pop('click_model')
    if click_model_names == 'all':
        click_model_names = MQD.keys()

    queries = ranking_model_args.pop('query')
    if queries == 'all':
        queries = MQD['UBM'].keys()

    n_impressions = ranking_model_args.pop('n_impressions')
    cutoff = ranking_model_args.get('cutoff')

    outputdir = ranking_model_args.pop('output')

    if 'feedback' in ranking_model_args:
        ranking_model_args['feedback_model_type'] = getattr(ClickLambdasAlgorithm, ranking_model_args.pop('feedback'))

    experiments = prepare_experiments(MQD, ranking_model_name, ranking_model_args,
                                      click_model_names, queries, n_impressions,
                                      cutoff)

    start = timer()
    for experiment in experiments:
        experiment.execute()
    end = timer()

    print 'Elapsed time: %.2fs' % (end - start)




