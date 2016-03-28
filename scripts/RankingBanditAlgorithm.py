# -*- coding: utf-8 -*-

import sys
import inspect
import numpy as np

import ClickLambdasAlgorithm

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler

from rankbs import CascadeUCB1
from rankbs import CascadeKL_UCB
from rankbs import CascadeLambdaMachine
from rankbs import CascadeThompsonSampler
from rankbs import CascadeExp3


def get_available_algorithms():
    algorithms = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if (not name.startswith('Base') and name.endswith('Algorithm')):
            algorithms.append(name)
    return algorithms


class BaseRankingBanditAlgorithm(object):
    '''
    Base class for the implementation of a ranking algorithm that should adapt
    and learn from the feedback in a form of clicks.
    '''
    def __init__(self, *args, **kwargs):
        try:
            self.n_documents = kwargs['n_documents']
            self.cutoff = kwargs['cutoff']
            self.random_state = kwargs['random_state']
            if self.random_state is None:
                random_state = np.random.RandomState()
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        '''
        Add algorithm specific parameters to the command line parser (from
        argparse module) using `parser.add_argument` method. The parameter
        values will be passed to the algorithm's __init__ under the name of
        the option added to the parser.
        '''
        pass

    def get_ranking(self, ranking=None):
        '''
        Produces a ranking based on the current state of the model.

        *** IMPORTANT ***
        The returned array must be a numpy int32 array.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            An optional output array. If None, the method instantiates a new
            array and returns it.

        Returns
        -------
        ranking : array of ints, shape = [n_documents]
            A ranking of documents.
        '''
        pass

    def set_feedback(self, ranking, clicks):
        '''
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints, shape = [cutoff]
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints, shape = [cutoff]
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        pass


class BaseLambdasRankingBanditAlgorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(BaseLambdasRankingBanditAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.feedback_model = kwargs['feedback'](self.n_documents)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(BaseLambdasRankingBanditAlgorithm, cls).update_parser(parser)
        parser.add_argument('-f', '--feedback',
                            choices=ClickLambdasAlgorithm.get_available_algorithms(),
                            required=True, help='feedback model')

    def set_feedback(self, ranking, clicks):
        self.feedback_model.update(ranking, clicks)


class UniformRankingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Creates rankings by uniformly shuffling documents.
    '''
    def __init__(self, *args, **kwargs):
        super(UniformRankingAlgorithm, self).__init__(*args, **kwargs)
        self.ranker = UniformRankingSampler(np.empty(self.n_documents,
                                                     dtype='float64'),
                                            random_state=self.random_state)

    def get_ranking(self, ranking=None):
        return self.ranker.sample(ranking)


class SoftmaxRakingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Creates rankings by sampling without replacement from a distribution
    over the documents. The distribution is just a softmax function applied
    on the relevance scores.
    '''
    def __init__(self, *args, **kwargs):
        super(SoftmaxRakingAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = SoftmaxRankingSampler(kwargs['relevances'],
                                                random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    def get_ranking(self, ranking=None):
        return self.ranker.sample(ranking)


class CascadeUCB1Algorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeUCB1Algorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = CascadeUCB1(self.n_documents, alpha=kwargs['alpha'],
                                      random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeUCB1Algorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=1.5,
                            required=True, help='alpha parameter')

    def get_ranking(self, ranking=None):
        return self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeKLUCBAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeKLUCBAlgorithm, self).__init__(*args, **kwargs)
        self.ranker = CascadeKL_UCB(self.n_documents,
                                    random_state=self.random_state)

    def get_ranking(self, ranking=None):
        return self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeLambdaMachineAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeLambdaMachineAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = CascadeLambdaMachine(self.n_documents,
                                               burnin=kwargs['burnin'],
                                               sigma=kwargs['sigma'],
                                               random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeLambdaMachineAlgorithm, cls).update_parser(parser)
        parser.add_argument('-b', '--burnin', type=int, default=1000,
                            required=True, help='burn-in time')
        parser.add_argument('-s', '--sigma', type=float, default=1.0,
                            required=True, help='sigma parameter')

    def get_ranking(self, ranking=None):
        return self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeThompsonSamplerAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeThompsonSamplerAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = CascadeThompsonSampler(self.n_documents,
                                                 alpha=kwargs['alpha'],
                                                 beta=kwargs['beta'],
                                                 random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeThompsonSamplerAlgorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=1.0,
                            required=True, help='alpha parameter')
        parser.add_argument('-b', '--beta', type=float, default=1.0,
                            required=True, help='alpha parameter')

    def get_ranking(self, ranking=None):
        return self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeExp3Algorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeExp3Algorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = CascadeExp3(self.n_documents, gamma=kwargs['gamma'],
                                      random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeExp3Algorithm, cls).update_parser(parser)
        parser.add_argument('-g', '--gamma', type=float, default=0.01,
                            required=True, help='gamma parameter')

    def get_ranking(self, ranking=None):
        return self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CopelandRakingAlgorithm(BaseLambdasRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CopelandRakingAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.t = 0
            self.alpha = kwargs['alpha']
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CopelandRakingAlgorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')

    def get_ranking(self, ranking=None):
        # Get the required statistics from the feedback model.
        lambdas, counts, n_viewed = self.feedback_model.statistics()

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods get_ranking and set_feedback is going to be called
        # in this order each time step.
        self.t += 1

        # Number of query documents and cutoff are available field variables.
        K = self.n_documents

        if ranking is None:
            ranking = np.empty(K, dtype='int32')

        L = np.array(lambdas)
        L[range(K),range(K)] = 0.
        N = np.array(counts)
        N[range(K),range(K)] = 1.
        P = L/np.maximum(N,1) - L.T/np.maximum(N.T,1)

        V = np.array(n_viewed)
        V[range(K),range(K)] = 1.
        non_diag = 0. * V + 1.
        V[range(K),range(K)] = 0.

        UCB = P + 2*np.sqrt(self.alpha * np.log(self.t) * non_diag / V)
        LCB = P - 2*np.sqrt(self.alpha * np.log(self.t) * non_diag / V)

        return ranking