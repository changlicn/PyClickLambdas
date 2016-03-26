import numpy as np

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler

class BaseRankingBanditAlgorithm(object):
    '''
    Base class for the implementation of a ranking algorithm that should adapt
    and learn from the feedback in a form of 'click-based lambdas'.
    '''
    def __init__(self, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        
        self.random_state = random_state

    def get_ranking(self, lambdas, counts, cutoff=10):
        '''
        Receives lambdas and associated counts and produces a ranking
        (potentially) based on them.

        *** IMPORTANT ***
        The returned array must be a numpy int32 array.

        Parameters
        ----------
        lambdas : array of floats, shape = [n_documents, n_documents]
            Click-based lambdas computed by a specific method implemented
            by `ClickLambdaAlgorithm`.

        counts : array of floats, shape = [n_documents, n_documents]
            Counts associated with `lambdas`.

        cutoff : int
            The cutoff rank.

        Returns
        -------
        ranking : array of ints, shape = [n_documents]
            A ranking of documents.
        '''
        pass


class UniformRankingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Produces uniform rankings.
    '''
    def __init__(self, n_documents, random_state=None):
        super(UniformRankingAlgorithm, self).__init__(random_state=random_state)
        self.sampler = UniformRankingSampler(np.empty(n_documents),
                                             random_state=self.random_state)
        self.ranking = np.empty(n_documents, dtype='int32')

    def get_ranking(self, lambdas, counts, cutoff=10):
        self.sampler.sample(out=self.ranking)
        return self.ranking


class SoftmaxRakingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Produces rankings based on the relevance scores of the query passed
    through a soft-max (multinomial logistic) function.
    '''
    def __init__(self, relevances, random_state=None):
        super(UniformRankingAlgorithm, self).__init__(random_state=random_state)
        self.sampler = SoftmaxRankingSampler(relevances,
                                             random_state=self.random_state)
        self.ranking = np.empty(len(relevances), dtype='int32')

    def get_ranking(self, lambdas, counts, cutoff=10):
        self.sampler.sample(out=self.ranking)
        return self.ranking


class CopelandRakingAlgorithm(BaseRankingBanditAlgorithm):
    '''
    Produces rankings based on the relevance scores of the query passed
    through a soft-max (multinomial logistic) function.
    '''
    def __init__(self, alpha=0.51, random_state=None):
        super(UniformRankingAlgorithm, self).__init__(random_state=random_state)
        self.alpha = alpha

    def get_ranking(self, lambdas, counts, n_viewed, t, cutoff=10):
        K = lambdas.shape[0]
        out_ranking = np.empty(K, dtype='int32')

        L = np.array(lambdas)
        L[range(K),range(K)] = 0.
        N = np.array(counts)
        N[range(K),range(K)] = 1.
        P = L/np.maximum(N,1) - L.T/np.maximum(N.T,1)

        V = np.array(n_viewed)
        V[range(K),range(K)] = 1.
        non_diag = 0. * V + 1.
        V[range(K),range(K)] = 0.

        UCB = P + 2*np.sqrt(alpha * np.log(t) * non_diag / V)
        LCB = P - 2*np.sqrt(alpha * np.log(t) * non_diag / V)

        

        return out_ranking
