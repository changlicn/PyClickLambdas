import numpy as np

import cPickle as pickle

from RankingBanditAlgorithm import SoftmaxRakingAlgorithm
from RankingBanditAlgorithm import UniformRankingAlgorithm

from ClickLambdasAlgorithm import SkipClickLambdasAlgorithm

from timeit import default_timer as timer


class RankingBanditExperiment(object):
    def __init__(self, click_model, ranking_model, lambdas_model,
                 n_documents, n_impressions, cutoff):
        self.click_model = click_model
        self.ranking_model = ranking_model
        self.lambdas_model = lambdas_model
        self.n_documents = n_documents
        self.n_impressions = n_impressions
        self.cutoff = cutoff

    def execute(self):
        # Used internally by the click model.
        identity = np.arange(self.n_documents, dtype='int32')

        for t in xrange(self.n_impressions):
            lambdas, counts = self.lambdas_model.get_parameters()

            ranking = self.ranking_model.get_ranking(lambdas, counts,
                                                    cutoff=self.cutoff)

            clicks = self.click_model.get_clicks(ranking[:self.cutoff],
                                                 identity)

            self.lambdas_model.update(ranking[:self.cutoff], clicks)


def prepare_experiment_example(MQD,ranking_algorithm=''):
    '''
    Method that prepares an experiment.

    Summary: Experiment with uniform rankings of query 2548, clicks
             simulated from PBM model, 10000 impressions, cutoff 5.
    '''
    random_state = np.random.RandomState(42)

    click_model_name = 'PBM'
    query_id = '2548'

    relevances = MQD[click_model_name][query_id]['relevances']
    n_documents = len(relevances)

    # CLICK MODEL FOR SPECIFIC QUERY 
    click_model = MQD[click_model_name][query_id]['model']

    # RANKING MODEL
    ranking_model = UniformRankingAlgorithm(n_documents, random_state=random_state)

    # LAMBDAS MODEL
    lambdas_model = SkipClickLambdasAlgorithm(n_documents)

    # The total number of impressions for the query.
    n_impressions = 10000

    # The cutoff rank (the maximum number of 'visible' documents).
    cutoff = 5

    experiment = RankingBanditExperiment(click_model, ranking_model,
                                         lambdas_model, n_documents,
                                         n_impressions, cutoff)

    return experiment


if __name__ == '__main__':
    # Load the click models for the queries of interest.
    with open('./data/model_query_collection.pkl') as ifile:
        MQD = pickle.load(ifile)

    # For reproducibility -- re-seed the click models' RNGs.
    for click_model_name in MQD:
        for query in MQD[click_model_name]:
            MQD[click_model_name][query]['model'].seed = 42

    experiment = prepare_experiment_example(MQD)

    start = timer()
    experiment.execute() 
    end = timer()

    print 'Elapsed time: %.2fs' % (end - start)




