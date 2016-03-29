#!/usr/bin/python2
# -*- coding: utf-8 -*-

'''
Script for evaluation of different ranking algorithms using various notions
of regret.
'''

import numpy as np

import os
import argparse
import cPickle as pickle

import RankingBanditAlgorithm

from glob import glob
from joblib import Parallel, delayed


class BaseRegretEvaluator(object):
    '''
    Base class for the implementation of ranking algorithms' regret
    evaluation method.
    '''
    def evaluate(self, history):
        '''
        Given the history of rankings produced by a ranking algorithm,
        this method should calculate its regret.

        Parameters
        ----------
        history: dict
            The ranking algorithm history.
        '''
        pass


class ClickthroughRateRegretEvaluator(BaseRegretEvaluator):
    def __init__(self, click_model_data):
        self.click_model_data = click_model_data

    def evaluate(self, history, cutoff=None):
        rankings = history['rankings']
        regret = np.empty(rankings.shape[0], dtype='float64')

        if cutoff is None:
            cutoff = rankings.shape[1]
        else:
            cutoff = min(cutoff, rankings.shape[1])

        if cutoff <= 0:
            raise ValueError('cutoff must be positive integer')

        click_model = self.click_model_data['model']
        ideal_ranking = self.click_model_data['ideal_ranking']

        # Used internally by the click model.
        identity = np.arange(rankings.shape[1], dtype='int32')

        ideal_ctr = click_model.get_clickthrough_rate(ideal_ranking, identity,
                                                      cutoff=cutoff)

        for t, ranking in enumerate(rankings):
            curr_ctr = click_model.get_clickthrough_rate(ranking, identity,
                                                         cutoff=cutoff)
            regret[t] = ideal_ctr - curr_ctr

        return regret


def load_click_models(source='./data/model_query_collection.pkl'):
    with open(source) as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_name in MQD:
        for query in MQD[click_model_name]:
            MQD[click_model_name][query]['model'].seed = 42

    return MQD


def load_model_rankings(inputfile):
    with open(inputfile) as ifile:
        return pickle.load(ifile)


def create_output_filename(filename):
    return filename.rstrip('_rankings.pkl') + '_regret.pkl'


def evaluate_ranking_algorithm(inputfile, outputfile, MQD, click_model_names=None, cutoff=None):
    ranker_history = load_model_rankings(inputfile)

    # Use the model from training for the evaluation if other
    # click model is not specified.
    if click_model_names is None:
        click_model_names = [ranker_history['click_model']]

    output_data = {}
    output_data['training_click_model'] = ranker_history['click_model']
    output_data['training_cutoff'] = ranker_history['cutoff']
    output_data['test_click_model'] = {}

    for click_model_name in click_model_names:
        click_model_regret = {}
        evaluator = ClickthroughRateRegretEvaluator(MQD[click_model_name][ranker_history['query']])
        click_model_regret['regret'] = evaluator.evaluate(ranker_history, cutoff=cutoff)
        click_model_regret['cutoff'] = cutoff

        output_data['test_click_model'][click_model_name] = click_model_regret        

    with open(outputfile, 'wb') as ofile:
        pickle.dump(output_data, ofile, protocol=-1)


if __name__ == '__main__':
    # Load click models for evaluation.
    MQD = load_click_models()

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity level')
    parser.add_argument('-m', '--click-model', choices=['all'] + MQD.keys(), default='all', help='user model used for evaluation')
    parser.add_argument('-c', '--cutoff', type=int, default=10, help='cutoff rank')
    parser.add_argument('-w', '--n-workers', type=int, default=1, help='number of worker threads')
    parser.add_argument("input", help="input directory or file")
    parser.add_argument("output", help="output directory or file")

    arguments = parser.parse_args()

    # click model name(s) ...
    click_model_names = arguments.click_model
    if click_model_names == 'all':
        click_model_names = MQD.keys()
    else:
        click_model_names = [click_model_names]

    if os.path.isdir(arguments.input):
        ifilepaths = glob(os.path.join(arguments.input, '*_rankings.pkl'))

        if not os.path.isdir(arguments.output):
            raise ValueError('if input is a directory so must be the output')

        ofilepaths = [os.path.join(arguments.output, fn) for fn in \
                      map(create_output_filename, map(os.path.basename, ifilepaths))]

    elif os.path.exists(arguments.input):
        ifilepaths = [arguments.input]

        if os.path.isdir(arguments.output):
            raise ValueError('if input is a regular file so must be the output')

        ofilepaths = [arguments.output]

    else:
        raise ValueError('%s file does not  exist' % arguments.input)

    n_jobs = arguments.n_workers
    verbose = arguments.verbose
    cutoff = arguments.cutoff

    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_ranking_algorithm)(ifile, ofile, MQD,
                                            click_model_names=click_model_names,
                                            cutoff=cutoff)
        for ifile, ofile in zip(ifilepaths, ofilepaths))