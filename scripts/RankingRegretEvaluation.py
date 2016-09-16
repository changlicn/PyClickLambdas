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
    def evaluate(self, info, rankings):
        '''
        Given the experiment info and rankings produced by a ranking
        algorithm this method should calculate its regret.

        Parameters
        ----------
        info: dict
            Parameters of the experiment for a particular ranking
            algorithm.

        rankings: array, shape=[n_impressions, n_documents]
            The history of rankings.
        '''
        pass


class ClickthroughRateRegretEvaluator(BaseRegretEvaluator):
    def __init__(self, click_model):
        self.click_model = click_model

    def evaluate(self, info, rankings):
        regret = np.empty(rankings.shape[0], dtype='float64')

        cutoff = info['cutoff']

        # Used internally by the click model.
        identity = np.arange(info['n_documents'], dtype='int32')

        # Get the ideal top-`cutoff` ranking for the click model ...
        ideal_ranking = self.click_model.get_ideal_ranking(cutoff=cutoff)

        # ... and compute its clickthrough rate.
        ideal_ctr = self.click_model.get_clickthrough_rate(ideal_ranking,
                                                           identity,
                                                           cutoff=cutoff)

        for t, ranking in enumerate(rankings):
            curr_ctr = self.click_model.get_clickthrough_rate(ranking, identity,
                                                              cutoff=cutoff)
            regret[t] = ideal_ctr - curr_ctr

        return regret


def load_model_rankings(ifilepath):
    with open(ifilepath) as ifile:
        info = pickle.load(ifile)
    return info, np.load(get_rankings_filepath(ifilepath))


def create_output_filename(filename):
    return filename.rstrip('_experiment.nfo') + '_regret'


def get_rankings_filepath(filepath):
    return filepath.rstrip('_experiment.nfo') + '_rankings.npy'


def evaluate_ranking_algorithm(ifilepath, outputfile):
    info, rankings = load_model_rankings(ifilepath)

    click_model = info['click_model']
    
    # Make sure we re-seed the click model in order to
    # replicate the clicks in the same way they were
    # produced in the particular experiment.
    click_model.seed = info['seed']

    evaluator = ClickthroughRateRegretEvaluator(click_model)

    regret = evaluator.evaluate(info, rankings)

    np.save(outputfile, regret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity level')
    parser.add_argument('-w', '--n-workers', type=int, default=1, help='number of worker threads')
    parser.add_argument('-o', '--output', help='output directory or file')
    parser.add_argument('input', help='input directory or file')

    arguments = parser.parse_args()

    if os.path.isdir(arguments.input):
        ifilepaths = glob(os.path.join(arguments.input, '*.nfo'))

        if arguments.output is None:
            arguments.output = arguments.input

        if not os.path.isdir(arguments.output):
            raise ValueError('if input is a directory so must be the output')

        ofilepaths = [os.path.join(arguments.output, fn) for fn in \
                      map(create_output_filename, map(os.path.basename, ifilepaths))]

    elif os.path.exists(arguments.input):
        ifilepaths = [arguments.input]

        if arguments.output is None:
            arguments.output = os.path.dirname(arguments.input)

        if os.path.isdir(arguments.output):
            ofilepaths = [os.path.join(arguments.output,
                                       create_output_filename(
                                           os.path.basename(arguments.input)))]
        else:
            ofilepaths = [arguments.output]
    else:
        raise ValueError('%s file does not  exist' % arguments.input)

    Parallel(n_jobs=arguments.n_workers, verbose=arguments.verbose)(
        delayed(evaluate_ranking_algorithm)(ifile, ofile)
        for ifile, ofile in zip(ifilepaths, ofilepaths))