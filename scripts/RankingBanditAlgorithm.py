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
            self.feedback_model = getattr(ClickLambdasAlgorithm, kwargs['feedback'])(self.n_documents, self.cutoff)
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


class RelativeRankingAlgorithm(BaseLambdasRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(RelativeRankingAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.t = 1
            self.alpha = kwargs['alpha']
            self.N_exp = 100
            self.T_exp = self.N_exp * (self.n_documents*self.cutoff) ** 2
            self.C = []
            self.shuffler = UniformRankingSampler(np.empty(self.n_documents,
                                                           dtype='float64'),
                                                  random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

        # Validate the type of the feedback model.
        if not isinstance(self.feedback_model,
                          ClickLambdasAlgorithm.RefinedSkipClickLambdasAlgorithm):
            raise ValueError('expected RefinedSkipClickLambdasAlgorithm for '
                             'feedback_model but received %s'
                             % type(self.feedback_model).__name__)

    @classmethod
    def update_parser(cls, parser):
        super(RelativeRankingAlgorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'RelativeRankingAlgorithm'

    def get_ranking(self, ranking):
        # Get the required statistics from the feedback model.
        Lambdas, N = self.feedback_model.statistics()

        # Number of query documents and cutoff are available field variables.
        L = self.n_documents
        K = self.cutoff

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods get_ranking and set_feedback is going to be called
        # in this order each time step.
        self.t += 1

        # Sanity check that the arrays are in order klij.
        if Lambdas.shape != (L, L, K, K):
            raise ValueError('misordered dimension in lambdas and counts')

        # Lambda_ij is the same as Lambdas
        Lambda_ij = Lambdas

        # Lambda_ji is the transpose of Lambda_ij. This operation
        # is very cheap in NumPy >= 1.10 because only a view needs
        # to be created.
        Lambda_ji = np.swapaxes(Lambda_ij, 0, 1)

        # N_ij is the same as N.
        N_ij = N

        # N_ji is the transpose of N_ij. Similarly to construction
        # of Lambda_ji this can turn out to be very cheap.
        N_ji = np.swapaxes(N_ij, 0, 1)

        # P is the frequentist mean.
        P = Lambda_ij / N_ij - Lambda_ji / N_ji

        # C is the size of the confidence interval
        C = (np.sqrt(self.alpha * np.log(self.t + 1) / N_ij) + 
             np.sqrt(self.alpha * np.log(self.t + 1) / N_ji))

        # Get LCB.
        LCB = P - C

        # The partial order.
        P_t = (LCB > 0).any(axis=(2, 3))

        if self.t < self.T_exp:
            self.shuffler.sample(ranking)
        elif self.t == self.T_exp:
            self.C = P.sum(axis=(1,2,3)).argsort()
            ranking[:K] = self.C[:K]
            self.feedback_model.reset()
        else:
            # topKI = [P_t[C[i + 1], C[i]] for i in range(K - 1)].
            topKI = P_t[self.C[1:K], self.C[:(K - 1)]]
            # bottomKI = [P_t[C_[K + i], C[K - 1]] for i in range (L - K)].
            bottomKI = P_t[self.C[K:], self.C[K - 1]]

            I = np.r_[np.where(topKI)[0], (K + np.where(bottomKI)[0])]
            if topKI.any() or bottomKI.any():
                k = I.min()

                if k < K - 1:
                    tempC = np.array(self.C, dtype='int32')
                    tempC[k], tempC[k+1] = self.C[k+1], self.C[k]
                    self.C = np.array(tempC, dtype='int32')
                    self.feedback_model.lambdas[:,:,k:,k:] = 1.
                    self.feedback_model.counts[:,:,k:,k:] = 2.
                    Lambdas, N = self.feedback_model.statistics()
                    if (Lambdas[:,:,k:,k:] != 1.).any():
                        print "Lambdas[:,:,k:,k:] ="
                        print Lambdas[:,:,k:,k:]
                        raise ValueError, 'It seems like the reset did not take effect: the above matrix should be all 1s!'
                
                elif k > K - 1:
                    tempC = np.array(self.C, dtype='int32')
                    tempC[K-1], tempC[k] = self.C[k], self.C[K-1]
                    self.C = np.array(tempC, dtype='int32')


            # topKN = [P_t[C[i], C[i + 1]] for i in range(K - 1)].
            topKN = P_t[self.C[:(K - 1)], self.C[1:K]]
            # bottomKN = [P_t[C[K - 1], C_[K + i]] for i in range (L - K)].
            bottomKN = P_t[self.C[K - 1], self.C[K:]]

            ranking[:K] = self.C[:K]

            if not (topKN.all() and bottomKN.all()):
                N = np.r_[np.where(~topKN)[0], (K + np.where(~bottomKN)[0])]
                k = self.random_state.choice(N)

                if k < K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[k], ranking[k + 1] = self.C[k + 1], self.C[k]

                elif k > K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[K - 2] = self.C[k]
                    else:
                        ranking[K - 2] = self.C[K - 1]
                        ranking[K - 1] = self.C[k]



class RelativeRankingAlgorithmV1_TooSlow(BaseLambdasRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(RelativeRankingAlgorithmV1_TooSlow, self).__init__(*args, **kwargs)
        try:
            self.t = 1
            self.alpha = kwargs['alpha']
            self.C = []
            self.shuffler = UniformRankingSampler(np.empty(self.n_documents,
                                                           dtype='float64'),
                                                  random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

        # Validate the type of the feedback model.
        if not isinstance(self.feedback_model,
                          ClickLambdasAlgorithm.RefinedSkipClickLambdasAlgorithm):
            raise ValueError('expected RefinedSkipClickLambdasAlgorithm for '
                             'feedback_model but received %s'
                             % type(self.feedback_model).__name__)

    @classmethod
    def update_parser(cls, parser):
        super(RelativeRankingAlgorithmV1_TooSlow, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'RelativeRankingAlgorithmV1_TooSlow'

    def get_chain_in(self, P_t):
        # The number of (other) documents beating each document.
        n_beating_d = P_t.sum(axis=0)

        # Queue keeps documents in the order in which they
        # should be ranked.
        queue = np.where(n_beating_d == 0)[0].tolist()

        # There cannot be more than 1 unbeaten document
        # if we are looking for a single chain.
        if len(queue) > 1:
            return []

        # Topological order (preference ordering) of
        # vertices (documents) in the graph induced
        # by P_t (preferences).
        chain = []

        indicator = np.zeros(self.n_documents,dtype="bool")

        for d in queue:
            indicator[d] = True

        while len(queue) > 0:
            u = queue.pop(0)
            for v in xrange(self.n_documents):
                n_beating_d[v] -= P_t[u, v]
                if not indicator[v] and n_beating_d[v] == 0:
                    queue.append(v)
                    indicator[v] = True
            chain.append(u)
            if len(chain) == self.cutoff:
                break

        # A preference cycle was detected?
        if len(chain) != self.cutoff:
            return []

        for d in (set(range(self.n_documents)) - set(chain)):
            chain.append(d)

        # Check there is total ordering in top K documents,
        # (if not return empty array) ...
        for i in xrange(self.cutoff - 1):
            if P_t[chain[i], chain[i + 1]] != 1:
                return []

        # ... and the K-th document beats all
        # the lower ranked documents...
        for i in xrange(self.cutoff, self.n_documents):
            if P_t[chain[self.cutoff - 1], chain[i]] != 1:
                return []

        # ... return the chain if all conditions
        # above are satisfied.
        return np.array(chain, dtype='int32')

    def get_ranking(self, ranking):
        # Get the required statistics from the feedback model.
        Lambdas, N = self.feedback_model.statistics()

        # Number of query documents and cutoff are available field variables.
        L = self.n_documents
        K = self.cutoff

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods get_ranking and set_feedback is going to be called
        # in this order each time step.
        self.t += 1

        # Sanity check that the arrays are in order klij.
        if Lambdas.shape != (L, L, K, K):
            raise ValueError('misordered dimension in lambdas and counts')

        # Lambda_ij is the same as Lambdas
        Lambda_ij = Lambdas

        # Lambda_ji is the transpose of Lambda_ij. This operation
        # is very cheap in NumPy >= 1.10 because only a view needs
        # to be created.
        Lambda_ji = np.swapaxes(Lambda_ij, 0, 1)

        # N_ij is the same as N.
        N_ij = N

        # N_ji is the transpose of N_ij. Similarly to construction
        # of Lambda_ji this can turn out to be very cheap.
        N_ji = np.swapaxes(N_ij, 0, 1)

        # p is the frequentist mean.
        p = Lambda_ij / N_ij - Lambda_ji / N_ji

        # c is the size of the confidence interval
        c = (np.sqrt(self.alpha * np.log(self.t + 1) / N_ij) + 
             np.sqrt(self.alpha * np.log(self.t + 1) / N_ji))

        # Get LCB.
        LCB = p - c

        # The partial order.
        P_t = (LCB > 0).any(axis=(2, 3))

        if self.C != []:
            # aboveK = [P_t[C[i + 1], C[i]] for i in range(K - 1)].
            aboveK = P_t[self.C[1:K], self.C[:(K - 1)]]
            # belowK = [P_t[C[K + i], C_[K - 1]] for i in range (L - K)].
            belowK = P_t[self.C[K:], self.C[K - 1]]

            if aboveK.any() or belowK.any():
                self.C = []
                self.feedback_model.reset()

        if self.C == []:
            chain = self.get_chain_in(P_t)

            if chain != []:
                self.C = chain
                ranking[:K] = self.C[:K]

            else:
                self.shuffler.sample(ranking)

        else:
            # topK = [P_t[C[i], C[i + 1]] for i in range(K - 1)].
            topK = P_t[self.C[:(K - 1)], self.C[1:K]]
            # bottomK = [P_t[C[K - 1], C_[K + i]] for i in range (L - K)].
            bottomK = P_t[self.C[K - 1], self.C[K:]]

            ranking[:K] = self.C[:K]

            if not (topK.all() and bottomK.all()):
                N = np.r_[np.where(~topK)[0], (K + np.where(~bottomK)[0])]
                k = self.random_state.choice(N)

                if k < K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[k], ranking[k + 1] = self.C[k + 1], self.C[k]

                elif k > K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[K - 2] = self.C[k]
                    else:
                        ranking[K - 2] = self.C[K - 1]
                        ranking[K - 1] = self.C[k]




class CoarseRelativeRankingAlgorithm(BaseLambdasRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CoarseRelativeRankingAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.t = 1
            self.alpha = kwargs['alpha']
            self.C = []
            self.shuffler = UniformRankingSampler(np.empty(self.n_documents,
                                                           dtype='float64'),
                                                  random_state=self.random_state)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

        # Validate the type of the feedback model.
        if not isinstance(self.feedback_model,
                          ClickLambdasAlgorithm.SkipClickLambdasAlgorithm):
            raise ValueError('expected SkipClickLambdasAlgorithm for '
                             'feedback_model but received %s'
                             % type(self.feedback_model).__name__)

    @classmethod
    def update_parser(cls, parser):
        super(CoarseRelativeRankingAlgorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')

    @classmethod
    def getName(cls):
        '''
        Returns the name of the algorithm.
        '''
        return 'CoarseRelativeRankingAlgorithm'

    def get_chain_in(self, P_t):
        # The number of (other) documents beating each document.
        n_beating_d = P_t.sum(axis=0)

        # Queue keeps documents in the order in which they
        # should be ranked.
        queue = np.where(n_beating_d == 0)[0].tolist()

        # There cannot be more than 1 unbeaten document
        # if we are looking for a single chain.
        if len(queue) > 1:
            return []

        # Topological order (preference ordering) of
        # vertices (documents) in the graph induced
        # by P_t (preferences).
        chain = []

        indicator = np.zeros(self.n_documents,dtype="bool")

        for d in queue:
            indicator[d] = True

        while len(queue) > 0:
            u = queue.pop(0)
            for v in xrange(self.n_documents):
                n_beating_d[v] -= P_t[u, v]
                if not indicator[v] and n_beating_d[v] == 0:
                    queue.append(v)
                    indicator[v] = True
            chain.append(u)
            if len(chain) == self.cutoff:
                break

        # A preference cycle was detected?
        if len(chain) != self.cutoff:
            return []

        for d in (set(range(self.n_documents)) - set(chain)):
            chain.append(d)

        # Check there is total ordering in top K documents,
        # (if not return empty array) ...
        for i in xrange(self.cutoff - 1):
            if P_t[chain[i], chain[i + 1]] != 1:
                return []

        # ... and the K-th document beats all
        # the lower ranked documents...
        for i in xrange(self.cutoff, self.n_documents):
            if P_t[chain[self.cutoff - 1], chain[i]] != 1:
                return []

        # ... return the chain if all conditions
        # above are satisfied.
        return np.array(chain, dtype='int32')

    def get_ranking(self, ranking):
        # Get the required statistics from the feedback model.
        Lambdas, N = self.feedback_model.statistics()

        # Number of query documents and cutoff are available field variables.
        L = self.n_documents
        K = self.cutoff

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods get_ranking and set_feedback is going to be called
        # in this order each time step.
        self.t += 1

        # Lambda_ij is the same as Lambdas
        Lambda_ij = Lambdas

        # Lambda_ji is the transpose of Lambda_ij.
        Lambda_ji = Lambda_ij.T

        # N_ij is the same as N.
        N_ij = N

        # N_ji is the transpose of N_ij.
        N_ji = N_ij.T

        # p is the frequentist mean.
        p = Lambda_ij / N_ij - Lambda_ji / N_ji

        # c is the size of the confidence interval
        c = (np.sqrt(self.alpha * np.log(self.t + 1) / N_ij) + 
             np.sqrt(self.alpha * np.log(self.t + 1) / N_ji))

        # Get LCB.
        LCB = p - c

        # The partial order.
        P_t = (LCB > 0)

        if self.C != []:
            # aboveK = [P_t[C[i + 1], C[i]] for i in range(K - 1)].
            aboveK = P_t[self.C[1:K], self.C[:(K - 1)]]
            # belowK = [P_t[C[K + i], C_[K - 1]] for i in range (L - K)].
            belowK = P_t[self.C[K:], self.C[K - 1]]

            if aboveK.any() or belowK.any():
                self.C = []
                self.feedback_model.reset()

        if self.C == []:
            chain = self.get_chain_in(P_t)

            if chain != []:
                self.C = chain
                ranking[:K] = self.C[:K]

            else:
                self.shuffler.sample(ranking)

        else:
            # topK = [P_t[C[i], C[i + 1]] for i in range(K - 1)].
            topK = P_t[self.C[:(K - 1)], self.C[1:K]]
            # bottomK = [P_t[C[K - 1], C_[K + i]] for i in range (L - K)].
            bottomK = P_t[self.C[K - 1], self.C[K:]]

            ranking[:K] = self.C[:K]

            if not (topK.all() and bottomK.all()):
                N = np.r_[np.where(~topK)[0], (K + np.where(~bottomK)[0])]
                k = self.random_state.choice(N)

                if k < K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[k], ranking[k + 1] = self.C[k + 1], self.C[k]

                elif k > K - 1:
                    if self.random_state.rand() < 0.5:
                        ranking[K - 2] = self.C[k]
                    else:
                        ranking[K - 2] = self.C[K - 1]
                        ranking[K - 1] = self.C[k]