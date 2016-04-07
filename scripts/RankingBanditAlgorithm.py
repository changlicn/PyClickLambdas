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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm. Defaults to the class name if not
        overriden in the extended classes.
        '''
        return cls.__name__

    def get_ranking(self, ranking):
        '''
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            The output array for the ranking.
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
            self.feedback_model = getattr(ClickLambdasAlgorithm, kwargs['feedback'])(self.n_documents)
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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'UniformRanker'

    def get_ranking(self, ranking):
        self.ranker.sample(ranking)


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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'SoftmaxRanker'

    def get_ranking(self, ranking):
        self.ranker.sample(ranking)


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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeUCB1'

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeKLUCBAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeKLUCBAlgorithm, self).__init__(*args, **kwargs)
        self.ranker = CascadeKL_UCB(self.n_documents,
                                    random_state=self.random_state)

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeKL-UCB'

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'LambdaRanker'

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeThompsonSampler'

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeExp3'

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

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

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CopelandRanker'

    def get_ranking(self, ranking):
        # Get the required statistics from the feedback model.
        lambdas, counts, n_viewed = self.feedback_model.statistics()

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods get_ranking and set_feedback is going to be called
        # in this order each time step.
        self.t += 1

        # Number of query documents and cutoff are available field variables.
        K = self.n_documents

        # Put the ranking produced by the algorithm into ranking,
        # which is an array of ints with shape = [self.n_documents]
        ranking[:] = np.arange(K, dtype='int32')