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

from RankingRegretEvaluation import ClickthroughRateRegretEvaluator
from RankingRegretEvaluation import ExpectedClickCountRegretEvaluator

from joblib import Parallel, delayed


class RankingBanditExperiment(object):
    def __init__(self, query, click_model, ranking_model, n_documents,
                 n_impressions, cutoff, compute_regret, store_rankings,
                 seed, outputdir):
        self.query = query
        self.click_model = click_model
        self.ranking_model = ranking_model
        self.n_documents = n_documents
        self.n_impressions = n_impressions
        self.cutoff = cutoff
        self.compute_regret = compute_regret
        self.store_rankings = store_rankings
        self.seed = seed
        self.outputdir = outputdir

        # Sanity check that ranking model is really focused
        # only to find top 'cutoff'-ranking.
        if ranking_model.cutoff != cutoff:
            raise ValueError('ranking model is set with cutoff %d but'
                             ' experiment is run with cutoff %d'
                             % (ranking_model.cutoff, cutoff))

    def get_output_filepath(self, suffix=None):
        filename = '_'.join(map(str, [self.ranking_model.getName(),
                                      self.click_model.getName(), self.query,
                                      self.cutoff, self.n_impressions]))

        # Append the suffix if it is specified and non-empty.
        if suffix is not None and len(suffix) > 0:
            filename += '_' + suffix

        return os.path.join(self.outputdir, filename)

    def execute(self):
        # Used internally by the click model.
        identity = np.arange(self.n_documents, dtype='int32')

        # History of all rankings produced by the model.
        rankings = -np.ones((self.n_impressions, self.n_documents),
                            dtype='int32')
        
        self.click_model.seed = self.seed
        
        # print 'ideal ranking:', self.click_model.get_ideal_ranking(cutoff=self.cutoff, satisfied=False)

        # Run for the specified number of iterations.
        for t in xrange(self.n_impressions):
            # Current ranking vector.
            ranking = rankings[t]

            # Get a ranking based on the current state of the model...
            self.ranking_model.get_ranking(ranking=ranking)

            # Just to make sure the algorithm always works
            # with documents for which we have got feedback.
            _ranking = ranking[:self.cutoff]

            # if t % 1000 == 0:
            #     print t, _ranking

            # get user clicks on that ranking...
            clicks = self.click_model.get_clicks(_ranking, identity)

            # ... and allow the model to learn from them.
            self.ranking_model.set_feedback(_ranking, clicks)

        # print 'final ranking:', _ranking
        # print 'ideal ranking:', self.click_model.get_ideal_ranking(cutoff=self.cutoff, satisfied=False)

        self.ranking_model.cleanup()
        
        # To get consistent results with regret calculated
        # later from the saved rankings. Still, the safest
        # thing to do is to call the following line before
        # simulating the clicks.
        self.click_model.seed = self.seed

        info = {}
        info['query'] = self.query
        info['click_model'] = self.click_model
        info['ranking_model'] = self.ranking_model
        info['cutoff'] = self.ranking_model.cutoff
        info['n_documents'] = self.n_documents
        info['n_impressions'] = self.n_impressions
        info['seed'] = self.seed
        info['regret_type'] = self.compute_regret 

        # Save the specifications of the experiment...
        with open(self.get_output_filepath(suffix='experiment') + '.nfo', 'wb') as ofile:
            pickle.dump(info, ofile, protocol=-1)

        # ... the rankings (based on specified options) ...
        if self.store_rankings or self.compute_regret is None:
            np.save(self.get_output_filepath(suffix='rankings'), rankings[:, :self.cutoff])

        # ... and (optionally) the cumulative regret.
        if self.compute_regret is not None:
            # Create an instance of simple CTR regret evaluator.
            if self.compute_regret == 'ctr':
                evaluator = ClickthroughRateRegretEvaluator(self.click_model)
            elif self.compute_regret == 'ecc':
                evaluator = ExpectedClickCountRegretEvaluator(self.click_model)
            else:
                raise ValueError('unknown regret type: %s' % self.compute_regret)

            regret = evaluator.evaluate(info, rankings[:, :self.cutoff])

            # print 'regret:', regret.cumsum()

            # Save the regret beside rankings.
            np.save(self.get_output_filepath(suffix='regret'), regret)


def load_click_models(source):
    with open(source) as ifile:
        return pickle.load(ifile)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    subparsers = parser.add_subparsers(help='choose ranking algorithm', dest='ranking_model')

    for ranker_algorithm_name in RankingBanditAlgorithm.get_available_algorithms():
        ranker_parser = subparsers.add_parser(ranker_algorithm_name)
        getattr(RankingBanditAlgorithm, ranker_algorithm_name).update_parser(ranker_parser)

    parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity level')
    parser.add_argument('-i', '--input', required=True, help='path to (query, click models)-pair pickle file')
    parser.add_argument('-r', '--regret', choices=('ctr', 'ecc'), help='the type of regret that is calculated: ctr (clickthrough rate), ecc (expected click count), regret is not calculated, if not specified')
    parser.add_argument('-d', '--store_rankings', action='store_true', help='indicates that the sequence of rankings made by the algorithm should be stored (they are if `regret` option is not specified)')
    parser.add_argument('-q', '--query', default='all', nargs='+', help='query/ies for which the experiment is executed')
    parser.add_argument('-m', '--click-model', default='all', nargs='+', help='user model/s used for generating clicks')
    parser.add_argument('-n', '--n-impressions', type=int, default=1, help='number of impressions')
    parser.add_argument('-c', '--cutoff', type=int, default=10, help='impressions will consist of only this number of documents')
    parser.add_argument('-w', '--n-workers', type=int, default=1, help='number of worker threads')
    parser.add_argument('-s', '--seed', type=int, default=42, help='seed for initialization of random number generators')
    parser.add_argument("output", help="output directory")

    return vars(parser.parse_args())


def prepare_experiments(MQD, ranking_model_name, ranking_model_args,
                        click_model_names, queries, n_impressions,
                        cutoff, compute_regret, store_ranking, seed,
                        outputdir):
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
            ranking_model_args['random_state'] = np.random.RandomState(31 * seed)
            ranking_model_args['n_impressions'] = n_impressions

            ranking_model = getattr(RankingBanditAlgorithm, ranking_model_name)(**ranking_model_args)

            click_model = MQD[click_model_name][query]['model']

            experiments.append(RankingBanditExperiment(query, click_model, ranking_model,
                                                       n_documents, n_impressions, cutoff,
                                                       compute_regret, store_ranking, seed,
                                                       outputdir))
    return experiments


def parallel_helper(obj, methodname, *args, **kwargs):
    '''
    Helper function to avoid pickling problems when using Parallel loops.
    '''
    return getattr(obj, methodname)(*args, **kwargs)


if __name__ == '__main__':
    kwargs = parse_command_line_arguments()

    # Load click models trained for selected queries.    
    MQD = load_click_models(kwargs.pop('input'))

    # ===============================================================
    # Get the global (not algorithm specific) command line arguments,
    # which are:

    # ranking model name ...
    ranking_model_name = kwargs.pop('ranking_model')

    # click model name(s) ...
    click_model_names = kwargs.pop('click_model')
    if 'all' in click_model_names:
        click_model_names = MQD.keys()
    else:
        diff = np.setdiff1d(click_model_names, MQD.keys())
        if len(diff) > 0:
            raise ValueError('some click models are not available: %s' % diff)

    # query ID(s) ...
    queries = kwargs.pop('query')
    if 'all' in queries:
        queries = MQD[MQD.keys()[0]].keys()
    else:
        diff = np.setdiff1d(queries, MQD[MQD.keys()[0]].keys())
        if len(diff) > 0:
            raise ValueError('some queries are not available: %s' % diff)

    # the regret computation indicator
    compute_regret = kwargs.pop('regret')
    
    # the save ranking indicator
    store_rankings = kwargs.pop('store_rankings')

    # the number of impressions (time steps) ...
    n_impressions = kwargs.pop('n_impressions')

    # the cutoff rank ...
    cutoff = kwargs.pop('cutoff')

    # the number of worker threads ...
    n_jobs = kwargs.pop('n_workers')

    # the verbosity level ...
    verbose = kwargs.pop('verbose')

    # seed for random number generator ...
    seed = kwargs.pop('seed')

    # and, finally, the output directory.
    outputdir = kwargs.pop('output')
    # ===============================================================

    # Make sure output path exists.
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Prepare experiments based on the parsed parameters...
    experiments = prepare_experiments(MQD, ranking_model_name, kwargs,
                                      click_model_names, queries, n_impressions,
                                      cutoff, compute_regret, store_rankings,
                                      seed, outputdir)

    # and run them, conveniently, in parallel loops.
    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(parallel_helper)(experiment, 'execute')
        for experiment in experiments)
