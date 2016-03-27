# -*- coding: utf-8 -*-

import numpy as np

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler

from rankbs import CascadeUCB1
from rankbs import CascadeKL_UCB
from rankbs import CascadeLambdaMachine
from rankbs import CascadeThompsonSampler
from rankbs import CascadeExp3


class BaseRankingBanditAlgorithm(object):
    '''
    Base class for the implementation of a ranking algorithm that should adapt
    and learn from the feedback in a form of clicks.
    '''
    def __init__(self, n_documents=None, cutoff=None, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        
        self.n_documents = n_documents
        self.cutoff = cutoff
        self.random_state = random_state

    def advance(self, ranking=None):
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

    def feedback(self, ranking, clicks):
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


class UniformRankingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Creates rankings by uniformly shuffling documents.
    '''
    def __init__(self, n_documents=None, cutoff=None, random_state=None):
        super(UniformRankingAlgorithm, self).__init__(n_documents=n_documents,
                                                      cutoff=cutoff,
                                                      random_state=random_state)

        self.ranker = UniformRankingSampler(np.empty(n_documents, dtype='float64'),
                                             random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.sample(out=ranking)


class SoftmaxRakingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Creates rankings by sampling without replacement from a distribution
    over the documents. The distribution is just a softmax function applied
    on the relevance scores.
    '''
    def __init__(self, relevances=None, n_documents=None,
                 cutoff=None, random_state=None):
        super(SoftmaxRakingAlgorithm, self).__init__(n_documents=n_documents,
                                                     cutoff=cutoff,
                                                     random_state=random_state)

        self.ranker = SoftmaxRankingSampler(relevances,
                                             random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.sample(ranking)


class CascadeUCB1Algorithm(BaseRankingBanditAlgorithm):

    def __init__(self, alpha=1.5, n_documents=None,
                 cutoff=None, random_state=None):
        super(CascadeUCB1Algorithm, self).__init__(n_documents=n_documents,
                                                   cutoff=cutoff,
                                                   random_state=random_state)

        self.ranker = CascadeUCB1(n_documents, alpha=alpha,
                                  random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.advance(ranking)

    def feedback(self, ranking, clicks):
        self.ranker.feedback(ranking, clicks)


class CascadeKLUCBAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, n_documents=None, cutoff=None, random_state=None):
        super(CascadeKLUCBAlgorithm, self).__init__(n_documents=n_documents,
                                                   cutoff=cutoff,
                                                   random_state=random_state)

        self.ranker = CascadeKL_UCB(n_documents, random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.advance(ranking)

    def feedback(self, ranking, clicks):
        self.ranker.feedback(ranking, clicks)


class CascadeLambdaMachineAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, burnin=0, sigma=1.0, n_documents=None,
                 cutoff=None, random_state=None):
        super(CascadeLambdaMachineAlgorithm, self).__init__(n_documents=n_documents,
                                                            cutoff=cutoff,
                                                            random_state=random_state)

        self.ranker = CascadeLambdaMachine(n_documents, T=burnin, sigma=sigma,
                                           random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.advance(ranking)

    def feedback(self, ranking, clicks):
        self.ranker.feedback(ranking, clicks)


class CascadeThompsonSamplerAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, alpha=1.0, beta=1.0, n_documents=None,
                 cutoff=None, random_state=None):
        super(CascadeThompsonSamplerAlgorithm, self).__init__(n_documents=n_documents,
                                                              cutoff=cutoff,
                                                              random_state=random_state)

        self.ranker = CascadeThompsonSampler(n_documents, alpha=alpha, beta=beta,
                                             random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.advance(ranking)

    def feedback(self, ranking, clicks):
        self.ranker.feedback(ranking, clicks)


class CascadeExp3Algorithm(BaseRankingBanditAlgorithm):

    def __init__(self, gamma=0.01, n_documents=None,
                 cutoff=None, random_state=None):
        super(CascadeExp3Algorithm, self).__init__(n_documents=n_documents,
                                                   cutoff=cutoff,
                                                   random_state=random_state)

        self.ranker = CascadeExp3(n_documents, gamma=gamma,
                                  random_state=self.random_state)

    def advance(self, ranking=None):
        return self.ranker.advance(ranking)

    def feedback(self, ranking, clicks):
        self.ranker.feedback(ranking, clicks)


class CopelandRakingAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, alpha=0.51, feedback_model_type=None, n_documents=None,
                 cutoff=None, random_state=None):
        super(CopelandRakingAlgorithm, self).__init__(n_documents=n_documents,
                                                      cutoff=cutoff,
                                                      random_state=random_state)
        if feedback_model_type is None:
            raise ValueError('missing feedback model type')

        self.t = 0
        self.alpha = alpha
        self.feedback_model = feedback_model_type(n_documents)

    def advance(self, ranking=None):
        # Get the required statistics from the feedback model.
        lambdas, counts, n_viewed = self.feedback_model.statistics()

        # Keep track of time inside the model. It is guaranteed that the pair
        # of methods advance and feedback is going to be called in this order
        # in each time step.
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

    def feedback(self, ranking, clicks):
        self.feedback_model.update(ranking, clicks)