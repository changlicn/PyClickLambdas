# -*- coding: utf-8 -*-

import sys
import inspect
import numpy as np

import ClickLambdasAlgorithm

from samplers import UniformRankingSampler
from samplers import SoftmaxRankingSampler

from rankbs import UCB1
from rankbs import Exp3
from rankbs import RelativeUCB1
from rankbs import CascadeUCB1
from rankbs import CascadeKL_UCB
from rankbs import CascadeLambdaMachine
from rankbs import CascadeThompsonSampler
from rankbs import CascadeExp3

from rankbs import get_kl_ucb
from rankbs import get_kl_lcb


# To avoid for-loops we vectorize these:
get_kl_ucb = np.vectorize(get_kl_ucb)
get_kl_lcb = np.vectorize(get_kl_lcb)


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
            self.n_impressions = kwargs['n_impressions']
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

    def getName(self):
        '''
        Returns the name of the algorithm. This method must be implemented
        in the derived classes.
        '''
        raise NotImplementedError()

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

    def cleanup(self):
        '''
        This method is called right after the experiment before the ranking
        model is saved with the results.
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

    def getName(self):
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

    def getName(self):
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
                                      feedback=kwargs['feedback'],
                                      random_state=self.random_state)
            self.feedback = kwargs['feedback']
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeUCB1Algorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=1.5,
                            required=True, help='alpha parameter')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeUCB1' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeKLUCBAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeKLUCBAlgorithm, self).__init__(*args, **kwargs)
        self.ranker = CascadeKL_UCB(self.n_documents, feedback=kwargs['feedback'],
                                    random_state=self.random_state)
        self.feedback = kwargs['feedback']
    
    @classmethod
    def update_parser(cls, parser):
        super(CascadeKLUCBAlgorithm, cls).update_parser(parser)
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeKL-UCB' + ('[' + self.feedback.upper() + ']')

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

    def getName(self):
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
                                                 feedback=kwargs['feedback'],
                                                 random_state=self.random_state)
            self.feedback = kwargs['feedback']
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeThompsonSamplerAlgorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=1.0,
                            required=True, help='alpha parameter')
        parser.add_argument('-b', '--beta', type=float, default=1.0,
                            required=True, help='alpha parameter')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeThompsonSampler' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)


class CascadeExp3Algorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(CascadeExp3Algorithm, self).__init__(*args, **kwargs)
        try:
            self.ranker = CascadeExp3(self.n_documents, gamma=kwargs['gamma'],
                                      feedback=kwargs['feedback'],
                                      random_state=self.random_state)
            self.feedback = kwargs['feedback']
        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(CascadeExp3Algorithm, cls).update_parser(parser)
        parser.add_argument('-g', '--gamma', type=float, default=0.01,
                            required=True, help='gamma parameter')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'CascadeExp3' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        self.ranker.get_ranking(ranking)

    def set_feedback(self, ranking, clicks):
        self.ranker.set_feedback(ranking, clicks)

        
class MergeRankAlgorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(MergeRankAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.D = np.arange(self.n_documents, dtype='int32')
            self.C = np.zeros(self.n_documents, dtype='float64')
            self.N = np.zeros(self.n_documents, dtype='float64')
            self.S = []
            self.t = 0
            self.T = kwargs['n_impressions']
            self.use_kl = (kwargs['method'] == 'kl')
            self.feedback = kwargs['feedback']
            np.set_printoptions(linewidth=np.inf)
        except KeyError as e:
            raise ValueError('missing %s argument' % e)
    
    @classmethod
    def update_parser(cls, parser):
        super(MergeRankAlgorithm, cls).update_parser(parser)
        parser.add_argument('-m', '--method', choices=['ch', 'kl'], required=False,
                            default='ch', help='specify type of confidence bounds used '
                           'by the ranking algorithm')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return ('MergeRank' + ('KL' if getattr(self, 'use_kl', False) else '') 
                            + ('[' + self.feedback.upper() + ']'))

    def get_ranking(self, ranking):
        if self.t < self.T:
            for ds in np.array_split(self.D, self.S):
                self.random_state.shuffle(ds)
        ranking[:self.cutoff] = self.D[:self.cutoff]

    def set_feedback(self, ranking, clicks):
        cutoff = self.cutoff
        
        if self.feedback == 'fc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[0] + 1

        elif self.feedback == 'lc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[-1] + 1
            
        for d, c in zip(ranking[:cutoff], clicks[:cutoff]):
            self.C[d] += c
            self.N[d] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            mus = np.nan_to_num(self.C / self.N)
            cnf = np.log(self.n_documents * self.T) / self.N

        if self.use_kl: # use Kullback-Leibler bounds
            ucb = get_kl_ucb(mus, cnf)
            lcb = get_kl_lcb(mus, cnf) 
        else: # use Chernoff-Hoeffding bounds
            cbs = np.sqrt(cnf)
            ucb = mus + cbs
            lcb = mus - cbs

        # The positions of old + new splits.
        nextS = []

        for ds, s in zip(np.array_split(self.D, self.S), np.r_[0, self.S]):
            # We keep the old split positions.
            nextS.append(s)
            
            # If the group consists of only a single document
            # there is nothing to do.
            if len(ds) == 1:
                continue
            
            # Sort the documents within a group by their CTR estimates.
            ds[:] = ds[np.argsort(mus[ds])[::-1]]
            
            # Sort the upper and lower confidence bounds as well.
            group_ucb = ucb[ds]
            group_lcb = lcb[ds]
            
            group_max_ucb = -np.ones_like(group_ucb)
            group_min_lcb = +np.ones_like(group_lcb)

            # Find the minimal LCB for consecutive documents
            # from top to bottom.
            group_min_lcb[0] = group_lcb[0]
            for i in range(1, len(ds)):
                group_min_lcb[i] = min(group_lcb[i],
                                       group_min_lcb[i - 1])
            
            # Find the maximal UCB for consecutive documents
            # from bottom to top.
            group_max_ucb[-1] = group_ucb[-1]
            for i in range(2, len(ds) + 1):
                group_max_ucb[-i] = max(group_ucb[-i],
                                        group_max_ucb[-(i - 1)])
            
            # See whether documents in the group cannot be
            # split into smaller groups.
            for i in range(1, len(ds)):
                if group_min_lcb[i - 1] > group_max_ucb[i]:
                    nextS.append(s + i)
            
        # Update the splits(omitting the 1st, which is
        # just an auxiliary index).
        self.S = np.array(nextS[1:], dtype='int32')
            
#         if self.t % 10000 == 0:
#             print self.S
#             print np.vstack([ucb[self.D], mus[self.D], lcb[self.D]])

        self.t += 1
        

class QuickRankAlgorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(QuickRankAlgorithm, self).__init__(*args, **kwargs)

        try:
            self.T = kwargs['n_impressions']
            self.feedback = kwargs['feedback']
            
            self.D = range(self.n_documents)
            self.K = self.cutoff
            self.D_f = []
            self.R = []
            
            self.v = np.zeros(self.n_documents, dtype='float64')
            self.n = np.zeros(self.n_documents, dtype='float64')
            self.m = 0
            self.N = 0
            self.delta = 1.0

            self.stack = []
            self.finished = False

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(QuickRankAlgorithm, cls).update_parser(parser)
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'QuickRank' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        # If only a single document remained in the active
        # set...
        while len(self.D) == 1:
            # ... extend the ranking prefix...
            self.R.append(self.D[0])
            
             # ... "go back" one stack frame,
            # if there is any, ...
            if len(self.stack) > 0:
                self.D, self.K, self.D_f = self.stack.pop()

#                 print 'self.R =', self.R
#                 print 'self.D =', self.D
#                 print 'self.D_f =', self.D_f
#                 print 'self.K =', self.K
#                 print 'self.stack =', self.stack
#                 print
            
                self.v.fill(0)
                self.n.fill(0)
                self.m = 0
                self.N = 0
                self.delta = 1.0
            # ... otherwise, finish exploring.
            else:
                self.finished = True
                break

        # When the time horizon was reached or the documents
        # were completely separated, we present what ever
        # ranking we ended up with.
        if self.finished:
            ranking[:self.cutoff] = self.R[:self.cutoff]
            return                

        # We proceed in rounds. In each round we show
        # a randomized ranking...
        if self.m <= np.log2(self.T):
            # ... and once a certain number of impressions was made, we
            # try to separate the documents into better and worse in terms
            # of their click-through rate estimates.                
            if self.N > 2 * np.log(self.T * len(self.D)) / self.delta**2:
                # We proceed to the next round.
                self.m += 1
                self.delta /= 2.0
                
                # Once a certain number of impressions was reached (depending
                # on the round), documents' click-through rate estimates are
                # computed...
                with np.errstate(invalid='ignore'):
                    mus = np.nan_to_num(self.v / self.n)[self.D]
                
                indices = np.argsort(mus)[::-1]
                
                # ... and sorted in descending order...
                mus = mus[indices]
                # ... together with the corresponding documents.
                self.D = [self.D[i] for i in indices]                
                
                # Finally, we try to find a split, which separates the documents
                # into 2 groups - ones which are superior to the others as far
                # as their click-through rate estimates are concerned.
                s = -1
                for i in range(1, len(self.D)):
                    if mus[i - 1] - self.delta > mus[i] + self.delta:
                        s = i
                
                # If we found a split position...
                if s != -1:
                    # ... we create a new stack frame into which we postpone
                    # the processing of inferior documents from the active set
                    # (only if needed). This simulates the recursive nature
                    # of the algorithm.
                    if s < self.K:
                        self.stack.append((self.D[s:], self.K - s, self.D_f[:]))
                    
                    self.D_f = self.D[s:] + self.D_f
                    self.D = self.D[:s]
                    self.K = min(self.K, s)
                    
#                     print 'self.R =', self.R
#                     print 'self.D =', self.D
#                     print 'self.D_f =', self.D_f
#                     print 'self.K =', self.K
#                     print 'self.stack =', self.stack
#                     print
                    
                    self.v.fill(0)
                    self.n.fill(0)

                    self.m = 0
                    self.N = 0
                    self.delta = 1.0

            self.N += 1
                    
            # Concatenating the (already identified and hopefully optimal)
            # ranking prefix with a random K-permutation of documents
            # from the active set and (fixed) ranking of documents
            # from the frozen set. 
            ranking[:self.cutoff] = np.r_[self.R,
                                          self.random_state.choice(self.D,
                                                                   size=self.K,
                                                                   replace=False),
                                          self.D_f][:self.cutoff]
        else:
            self.finished = True
            
            with np.errstate(invalid='ignore'):
                indices = np.argsort(np.nan_to_num(self.v / self.n)[self.D])[::-1]                
                        
            self.R.extend([self.D[i] for i in indices])
            self.R.extend(self.D_f)

            ranking[:self.cutoff] = self.R[:self.cutoff]


    def set_feedback(self, ranking, clicks):
        cutoff = self.cutoff
        
        if self.feedback == 'fc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[0] + 1

        elif self.feedback == 'lc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[-1] + 1

        for d, c in zip(ranking[len(self.R):cutoff], clicks[len(self.R):cutoff]):
            self.v[d] += c
            self.n[d] += 1


class ShuffleAndSplitAlgorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(ShuffleAndSplitAlgorithm, self).__init__(*args, **kwargs)

        try:
            self.R = np.arange(self.n_documents, dtype='int32')
            self.v = np.zeros(self.n_documents, dtype='float64')
            self.n = np.zeros(self.n_documents, dtype='float64')
            self.T = kwargs['n_impressions']
            self.feedback = kwargs['feedback']
            self.m = 0
            self.N = 0
            self.delta = 1.0
            self.S = np.array([], dtype='int32')
            self.finished = False

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(ShuffleAndSplitAlgorithm, cls).update_parser(parser)
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='lc', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'ShuffleAndSplit' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        if self.finished:
            ranking[:self.cutoff] = self.R[:self.cutoff]

        # We proceed in rounds. In each round we show
        # a randomized ranking...
        elif self.m <= np.log2(self.T):
            # ... and once a certain number of impressions was made, we
            # try to separate the documents into "relevance" groups 
            # according to their click-through rate estimates.
            if self.N > 2 * np.log(self.T * self.cutoff) / self.delta**2:
                # We proceed to the next round.
                self.m += 1
                self.delta /= 2.0
                
                # This will hold the split positions for
                # the next round.
                nextS = []
                
                # Compute documents' click-through rate estimates...
                with np.errstate(invalid='ignore'):
                    mus = np.nan_to_num(self.v / self.n)[self.R]
                                
                # ... and based on the current confidence intervals around them,
                # find out whether some of the document "relevance" group should
                # not be split.
                for mus, offset in zip(np.array_split(mus, self.S), np.r_[0, self.S]):
                    # We keep the old split positions.
                    nextS.append(offset)
                    
                    # If a group consists of a single document
                    # there is nothing to do.
                    if len(mus) == 1:
                        continue
                    
                    indices = np.argsort(mus)[::-1]

                    # Sorted CTR estimates of documents within the group...
                    mus = mus[indices]  
                    # ... and the corresponding document indices.
                    ds = self.R[indices + offset]
                    
                    self.R[offset:(offset + len(ds))] = ds
                            
                    # Go through the documents within the "relevance" group...
                    for s in range(1, len(ds)):
                        # ... and if we find consecutive pair of not overlapping
                        # confidence intervals , we split the group between them.
                        if mus[s - 1] - self.delta > mus[s] + self.delta:
                            nextS.append(offset + s)
                
                # Update the splits for the next round (omitting the 1st,
                # which is just an auxiliary index).
                self.S = np.array(nextS[1:], dtype='int32')
            
            self.N += 1
            
            for ds in np.array_split(self.R, self.S):
                self.random_state.shuffle(ds)
                
            ranking[:self.cutoff] = self.R[:self.cutoff]

        else:
            self.finished = True

            final_ranking = []

            with np.errstate(invalid='ignore'):
                mus = np.nan_to_num(self.v / self.n)[self.R]

            # No more exploratory impressions allowed, so we sort
            # the documents within each group according to their
            # estimated click-through rates and hope for the best.
            
            for mus, offset in np.zip(np.array_split(mus, self.S), np.r_[0, self.S]):
                final_ranking.extend(self.R[np.argsort(mus)[::-1] + offset])

            self.R[:] = final_ranking

    def set_feedback(self, ranking, clicks):
        cutoff = self.cutoff
        
        if self.feedback == 'fc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[0] + 1

        elif self.feedback == 'lc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[-1] + 1
                
        for d, c in zip(ranking[:cutoff], clicks[:cutoff]):
            self.v[d] += c
            self.n[d] += 1


class RankedBanditsUCB1Algorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(RankedBanditsUCB1Algorithm, self).__init__(*args, **kwargs)
        try:
            # Create one MAB for each rank.
            self.rankers = [UCB1(self.n_documents, alpha=kwargs['alpha'],
                                 random_state=self.random_state)
                            for _ in range(self.cutoff)]
            self.t = 0
            self.T = kwargs['n_impressions']
            self.feedback = kwargs['feedback']
            self.__tmp_ranking = np.empty(self.cutoff, dtype='int32')

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(RankedBanditsUCB1Algorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='ff', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'RankedBanditsUCB1' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        D = set(range(self.n_documents))

        if self.t < self.T: 
            for r, ranker in enumerate(self.rankers):
                self.__tmp_ranking[r] = ranker.get_arm()
                
                if self.__tmp_ranking[r] in ranking[:r]:
                    ranking[r] = self.random_state.choice(list(D))
                else:
                    ranking[r] = self.__tmp_ranking[r]
                
                D.remove(ranking[r])
        else:
            for r, ranker in enumerate(self.rankers):
                ranking[r] = ranker.get_arm()

    def set_feedback(self, ranking, clicks):
        if self.t < self.T:
            cutoff = self.cutoff
        
            if self.feedback == 'fc':
                crs = np.flatnonzero(clicks)
                if crs.size > 0:
                    cutoff = crs[0] + 1

            elif self.feedback == 'lc':
                crs = np.flatnonzero(clicks)
                if crs.size > 0:
                    cutoff = crs[-1] + 1
            
            for d, dhat, c, ranker in zip(ranking[:cutoff], self.__tmp_ranking[:cutoff],
                                          clicks, self.rankers[:cutoff]):
                c = 1 if c and d == dhat else 0                    
                ranker.update_arm(dhat, c)


class RankedBanditsExp3Algorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(RankedBanditsExp3Algorithm, self).__init__(*args, **kwargs)
        try:
            self.t = 0
            self.T = kwargs['n_impressions']
            self.feedback = kwargs['feedback']
            # Create one MAB for each rank.
            if kwargs['adaptive']:
                self.rankers = [Exp3(self.n_documents,
                                     random_state=self.random_state)
                                for _ in range(self.cutoff)]
            else:
                self.rankers = [Exp3(self.n_documents, T=self.T,
                                     random_state=self.random_state)
                                for _ in range(self.cutoff)]
                
            self.__tmp_ranking = np.empty(self.cutoff, dtype='int32')

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(RankedBanditsExp3Algorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--adaptive', action='store_true',
                            help='if present the underlaying Exp3 models will not '
                            'exploit the information about the number of impressions')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='ff', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'RankedBanditsExp3' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        D = set(range(self.n_documents))

        if self.t < self.T: 
            for r, ranker in enumerate(self.rankers):
                self.__tmp_ranking[r] = ranker.get_arm()
                
                if self.__tmp_ranking[r] in ranking[:r]:
                    ranking[r] = self.random_state.choice(list(D))
                else:
                    ranking[r] = self.__tmp_ranking[r]
                
                D.remove(ranking[r])
        else:
            for r, ranker in enumerate(self.rankers):
                ranking[r] = ranker.get_arm()

    def set_feedback(self, ranking, clicks):
        if self.t < self.T:
            cutoff = self.cutoff
        
            if self.feedback == 'fc':
                crs = np.flatnonzero(clicks)
                if crs.size > 0:
                    cutoff = crs[0] + 1

            elif self.feedback == 'lc':
                crs = np.flatnonzero(clicks)
                if crs.size > 0:
                    cutoff = crs[-1] + 1
            
            for d, dhat, c, ranker in zip(ranking[:cutoff], self.__tmp_ranking[:cutoff],
                                          clicks, self.rankers[:cutoff]):
                c = 1 if c and d == dhat else 0                    
                ranker.update_arm(dhat, c)


class RelativeCascadeUCB1Algorithm(BaseRankingBanditAlgorithm):
    def __init__(self, *args, **kwargs):
        super(RelativeCascadeUCB1Algorithm, self).__init__(*args, **kwargs)
        try:
            # The ranker at the top rank uses all documents...
            self.top_ranker = RelativeUCB1(self.n_documents, alpha=kwargs['alpha'],
                                           random_state=self.random_state)

            # ... while the rankers at the lower ranks uses 1 less,
            # since they are relative.
            self.rankers = [RelativeUCB1(self.n_documents - 1, alpha=kwargs['alpha'],
                                         random_state=self.random_state)
                            for _ in range(self.n_documents)]
            self.feedback = kwargs['feedback']
            self.__tmp_rankings = np.empty(self.n_documents - 1, dtype='int32')

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(RelativeCascadeUCB1Algorithm, cls).update_parser(parser)
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=True, help='alpha parameter')
        parser.add_argument('-f', '--feedback', type=str, choices=['fc', 'lc', 'ff'],
                            default='ff', required=False, help='specify the way the click feedback '
                            'is processed - fc: down to the first click, lc: down to '
                            ' the last click, ff: full feedback')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'RelativeCascadeUCB1Algorithm' + ('[' + self.feedback.upper() + ']')

    def get_ranking(self, ranking):
        # Sample a document at the highest rank...
        self.top_ranker.get_arms(ranking) 
        # ... and for each subsequent rank ...
        for r in range(1, self.cutoff):
            # ... pick the ranker associated with the document above ...
            ranker = self.rankers[ranking[r - 1]]
            # ... get the relative ranking of the other documents ...
            ranker.get_arms(self.__tmp_rankings)
            # ... adjust the indices because `d` never appears in `__tmp_rankings` ...
            self.__tmp_rankings += (self.__tmp_rankings >= ranking[r - 1])
            # ... and finally pick a document that has not appeared yet.
            for d in self.__tmp_rankings:        
                if d not in ranking[:r]:
                    ranking[r] = d
                    break

    def set_feedback(self, ranking, clicks):
        self.top_ranker.set_feedback(ranking[0], clicks[0])
        prev_d = ranking[0]

        cutoff = self.cutoff
        
        if self.feedback == 'fc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[0] + 1

        elif self.feedback == 'lc':
            crs = np.flatnonzero(clicks)
            if crs.size > 0:
                cutoff = crs[-1] + 1

        for curr_d, c in zip(ranking[1:cutoff], clicks[1:cutoff]):
            # Pick the ranker associated with the document ranked above
            # and update it according to the click feedback.
            self.rankers[prev_d].set_feedback(curr_d - (curr_d > prev_d), c)
            prev_d = curr_d


class RelativeRankingAlgorithm(BaseLambdasRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(RelativeRankingAlgorithm, self).__init__(*args, **kwargs)
        try:
            self.C = []
            self.t = 0
            self.t_explore = kwargs['explore']
            self.alpha = kwargs['alpha']
            self.uniform = (kwargs['method'] == 'uniform')
            
            if kwargs['method'] == 'uniform':
                self.shuffler = UniformRankingSampler(np.empty(self.n_documents,
                                                               dtype='float64'),
                                                      random_state=self.random_state)

            elif kwargs['method'] == 'ucb':
                self.shuffler = CascadeUCB1(self.n_documents, alpha=self.alpha,
                                            first_click=kwargs['first_click'],
                                            random_state=self.random_state)

            elif kwargs['method'] == 'kl-ucb':
                self.shuffler = CascadeKL_UCB(self.n_documents,
                                              first_click=kwargs['first_click'],
                                              random_state=self.random_state)
            else:
                raise ValueError('unrecognized value for \'method\' option: %s'
                                 % kwargs['method'])

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
        parser.add_argument('-m', '--method', choices=['uniform', 'ucb', 'kl-ucb'],
                            required=True, help='specify bandit algorithm used '
                           'for ranking preselection')
        parser.add_argument('-a', '--alpha', type=float, default=0.51,
                            required=False, help='alpha parameter')
        parser.add_argument('-e', '--explore', type=int, default=10000, help='the number '
                            'of time steps allocated for initial exploration')
        parser.add_argument('-c', '--first-click', action='store_true',
                            required=False, help='consider feedback only up to '
                            'the first click instead of the whole list (relevant '
                            ' only when `ucb` or `kl-ucb` is used for ranking '
                            'preselection)')

    def getName(self):
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

        # Sanity check that the arrays are in order ijkl.
#         if Lambdas.shape != (L, L, K, K):
#             raise ValueError('misordered dimension in lambdas and counts')

        if self.t < self.t_explore:
            if self.uniform:
                self.shuffler.sample(ranking)
            else:
                self.shuffler.get_ranking(ranking)
        elif self.t == self.t_explore:
            if self.uniform:
                Lambda_ij = Lambdas
                Lambda_ji = np.swapaxes(Lambda_ij, 0, 1)

                N_ij = N
                N_ji = np.swapaxes(N_ij, 0, 1)
    
                P = Lambda_ij / N_ij - Lambda_ji / N_ji
    
                self.C = np.argsort(P.sum(axis=(1, 2, 3)))[::-1]
            else:
                self.C = self.shuffler.get_ranking()

            ranking[:K] = self.C[:K]
            self.feedback_model.reset()
            
#             print 'exploration ranking:', self.C[:K]
        else:
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
            C = (np.sqrt(self.alpha * np.log(self.t) / N_ij) + 
                 np.sqrt(self.alpha * np.log(self.t) / N_ji))

            # Get LCB.
            LCB = P - C

            # The partial order.
            P_t = (LCB > 0).any(axis=(2, 3))

            # Indices of incorrectly ordered pairs of documents in C.
            I = np.flatnonzero(np.r_[P_t[self.C[1:K], self.C[:(K - 1)]],
                                     P_t[self.C[K:], self.C[K - 1]]])

            if len(I) > 0:
################
                print 'Incorrectly ordered:', I, 'ranking:', self.C[:K], 'time:', self.t
                k = I.min()
                
                if k < K - 1:
                    self.C[k], self.C[k + 1] = self.C[k + 1], self.C[k]

                    self.feedback_model.lambdas[:, :, k:, k:] = 1
                    self.feedback_model.counts[:, :, k:, k:] = 2
                    # Sanity check that the feedback model statistics have changed.
                    if (self.feedback_model.statistics()[0][:, :, k:, k:] != 1).any():
                        raise ValueError, 'Feedback model has not changed as was expected!!!'
                
                else:
                    self.C[K - 1], self.C[k + 1] = self.C[k + 1], self.C[K - 1]

            ranking[:K] = self.C[:K]
            
############
            if self.t % 20000 == 0:
                print 'Non-conforming:', np.flatnonzero(~np.r_[P_t[self.C[:(K - 1)], self.C[1:K]],
                                                               P_t[self.C[K - 1], self.C[K:]]]),\
                       'ranking:', self.C[:K],'time:', self.t
              
            # Indicator of non-conforming pairwise orderings in C, above cutoff.
            N = np.flatnonzero(~np.r_[P_t[self.C[:(K - 1)], self.C[1:K]]])

            if len(N) > 0:                    
                prev_swapped_k = -1
                for k in N:
                    if k != prev_swapped_k:
                        if (self.feedback_model.counts[self.C[k], self.C[k + 1], k + 1, k] <
                            self.feedback_model.counts[self.C[k + 1], self.C[k], k + 1, k]):
                            ranking[k] = self.C[k + 1]
                            ranking[k + 1] = self.C[k]
                            prev_swapped_k = k + 1
                            
            # Indicator of non-conforming pairwise orderings in C, below cutoff.
            M = K + np.flatnonzero(~np.r_[P_t[self.C[K - 1], self.C[K:]]])

            if len(M) > 0:
                if self.random_state.rand() < 0.5:
                    k = self.random_state.choice(M)
                    if self.random_state.rand() < 0.5:
                        ranking[K - 2] = self.C[k]
                        ranking[K - 1] = self.C[K - 1]
                    else:
                        ranking[K - 2] = self.C[K - 1]
                        ranking[K - 1] = self.C[k]
            
            # Choose a pair at random and propagate it to the top.
            k = self.random_state.choice(K - 1)
            
            if k > 0:
                d, e = ranking[k], ranking[k + 1]
                ranking[2:k + 2] = ranking[0:k]
                ranking[0], ranking[1] = d, e
            
#             # Indicator of non-conforming pairwise orderings in C, above cutoff.
#             N = np.flatnonzero(~np.r_[P_t[self.C[:(K - 1)], self.C[1:K]]])
                        
#             if len(N) > 0:
#                 prev_swapped_k = -1
#                 for k in N:
#                     if k != prev_swapped_k:
#                         if (self.feedback_model.counts[self.C[k], self.C[k + 1], k + 1, k] <
#                             self.feedback_model.counts[self.C[k + 1], self.C[k], k + 1, k]):
#                             ranking[k] = self.C[k + 1]
#                             ranking[k + 1] = self.C[k]
#                             prev_swapped_k = k + 1
                            
#             # Indicator of non-conforming pairwise orderings in C, below cutoff.
#             M = K + np.flatnonzero(~np.r_[P_t[self.C[K - 1], self.C[K:]]])

#             if len(M) > 0:
#                 if self.random_state.rand() < 0.5:
#                     k = self.random_state.choice(M)

#                     if self.random_state.rand() < 0.5:
#                         ranking[K - 2] = self.C[k]
#                         ranking[K - 1] = self.C[K - 1]
#                     else:
#                         ranking[K - 2] = self.C[K - 1]
#                         ranking[K - 1] = self.C[k]

    def set_feedback(self, ranking, clicks):
        if self.t < self.t_explore:
            if not self.uniform:
                self.shuffler.set_feedback(ranking, clicks)
        super(RelativeRankingAlgorithm, self).set_feedback(ranking, clicks)
        

# class RelativeRankingAlgorithm(BaseLambdasRankingBanditAlgorithm):

#     def __init__(self, *args, **kwargs):
#         super(RelativeRankingAlgorithm, self).__init__(*args, **kwargs)
#         try:
#             self.C = []
#             self.t = 0
#             self.t_explore = kwargs['explore']
#             self.alpha = kwargs['alpha']
#             self.uniform = (kwargs['method'] == 'uniform')
            
#             if kwargs['method'] == 'uniform':
#                 self.shuffler = UniformRankingSampler(np.empty(self.n_documents,
#                                                                dtype='float64'),
#                                                       random_state=self.random_state)

#             elif kwargs['method'] == 'ucb':
#                 self.shuffler = CascadeUCB1(self.n_documents, alpha=self.alpha,
#                                             first_click=kwargs['first_click'],
#                                             random_state=self.random_state)

#             elif kwargs['method'] == 'kl-ucb':
#                 self.shuffler = CascadeKL_UCB(self.n_documents, alpha=self.alpha,
#                                               first_click=kwargs['first_click'],
#                                               random_state=self.random_state)
#             else:
#                 raise ValueError('unrecognized value for \'method\' option: %s'
#                                  % kwargs['method'])

#         except KeyError as e:
#             raise ValueError('missing %s argument' % e)

#         # Validate the type of the feedback model.
#         if not isinstance(self.feedback_model,
#                           ClickLambdasAlgorithm.RefinedSkipClickLambdasAlgorithm):
#             raise ValueError('expected RefinedSkipClickLambdasAlgorithm for '
#                              'feedback_model but received %s'
#                              % type(self.feedback_model).__name__)

#     @classmethod
#     def update_parser(cls, parser):
#         super(RelativeRankingAlgorithm, cls).update_parser(parser)
#         parser.add_argument('-m', '--method', choices=['uniform', 'ucb', 'kl-ucb'],
#                             required=True, help='specify bandit algorithm used '
#                            'for ranking preselection')
#         parser.add_argument('-a', '--alpha', type=float, default=0.51,
#                             required=False, help='alpha parameter')
#         parser.add_argument('-e', '--explore', type=int, default=10000, help='the number '
#                             'of time steps allocated for initial exploration')
#         parser.add_argument('-c', '--first-click', action='store_true',
#                             required=False, help='consider feedback only up to '
#                             'the first click instead of the whole list (relevant '
#                             ' only when `ucb` or `kl-ucb` is used for ranking '
#                             'preselection)')

#     @classmethod
#     def getName(cls):
#         '''
#         Returns the name of the algorithm.
#         '''
#         return 'RelativeRankingAlgorithm'

#     def get_ranking(self, ranking):
#         # Get the required statistics from the feedback model.
#         Lambdas, N = self.feedback_model.statistics()

#         # Number of query documents and cutoff are available field variables.
#         L = self.n_documents
#         K = self.cutoff

#         # Keep track of time inside the model. It is guaranteed that the pair
#         # of methods get_ranking and set_feedback is going to be called
#         # in this order each time step.
#         self.t += 1

#         # Sanity check that the arrays are in order ijkl.
#         if Lambdas.shape != (L, L, K, K):
#             raise ValueError('misordered dimension in lambdas and counts')

#         if self.t < self.t_explore:
#             if self.uniform:
#                 self.shuffler.sample(ranking)
#             else:
#                 self.shuffler.get_ranking(ranking)
#         elif self.t == self.t_explore:
#             if self.uniform:
#                 Lambda_ij = Lambdas
#                 Lambda_ji = np.swapaxes(Lambda_ij, 0, 1)

#                 N_ij = N
#                 N_ji = np.swapaxes(N_ij, 0, 1)
    
#                 P = Lambda_ij / N_ij - Lambda_ji / N_ji
    
#                 self.C = np.argsort(P.sum(axis=(1, 2, 3)))
#             else:
#                 self.C = self.shuffler.get_ranking()

#             ranking[:K] = self.C[:K]
#             self.feedback_model.reset()
#         else:
#             # Lambda_ij is the same as Lambdas
#             Lambda_ij = Lambdas

#             # Lambda_ji is the transpose of Lambda_ij. This operation
#             # is very cheap in NumPy >= 1.10 because only a view needs
#             # to be created.
#             Lambda_ji = np.swapaxes(Lambda_ij, 0, 1)

#             # N_ij is the same as N.
#             N_ij = N

#             # N_ji is the transpose of N_ij. Similarly to construction
#             # of Lambda_ji this can turn out to be very cheap.
#             N_ji = np.swapaxes(N_ij, 0, 1)

#             # P is the frequentist mean.
#             P = Lambda_ij / N_ij - Lambda_ji / N_ji

#             # C is the size of the confidence interval
#             C = (np.sqrt(self.alpha * np.log(self.t) / N_ij) + 
#                  np.sqrt(self.alpha * np.log(self.t) / N_ji))

#             # Get LCB.
#             LCB = P - C

#             # The partial order.
#             P_t = (LCB > 0).any(axis=(2, 3))

#             # Incorrectly ordered document indices.
#             I = np.flatnonzero(np.r_[P_t[self.C[1:K], self.C[:(K - 1)]],
#                                      P_t[self.C[K:], self.C[K - 1]]])

#             if len(I) > 0:
#                 k = I.min()
                
#                 if k < K - 1:
#                     self.C[k], self.C[k + 1] = self.C[k + 1], self.C[k]

#                     self.feedback_model.lambdas[:, :, k:, k:] = 1
#                     self.feedback_model.counts[:, :, k:, k:] = 2
#                     # Sanity check that the feedback model statistics have changed.
#                     if (self.feedback_model.statistics()[0][:, :, k:, k:] != 1).any():
#                         raise ValueError, 'Feedback model has not changed as was expected!!!'
                
#                 else:
#                     self.C[K - 1], self.C[k + 1] = self.C[k + 1], self.C[K - 1]

#             ranking[:K] = self.C[:K]
            
#             # Non-conforming pairwise document orderings.
#             N = np.flatnonzero(~np.r_[P_t[self.C[:(K - 1)], self.C[1:K]],
#                                       P_t[self.C[K - 1], self.C[K:]]])

#             if len(N) > 0:
#                 k = self.random_state.choice(N)

#                 if k < K - 1:
#                     if self.random_state.rand() < 0.5:
#                         ranking[k] = self.C[k + 1]
#                         ranking[k + 1] = self.C[k]
#                 else:
#                     if self.random_state.rand() < 0.5:
#                         ranking[K - 2] = self.C[k + 1]
#                     else:
#                         ranking[K - 2] = self.C[K - 1]
#                         ranking[K - 1] = self.C[k + 1]

#     def set_feedback(self, ranking, clicks):
#         if self.t < self.t_explore:
#             if not self.uniform:
#                 self.shuffler.set_feedback(ranking, clicks)
#         super(RelativeRankingAlgorithm, self).set_feedback(ranking, clicks)


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

    def getName(self):
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

    def getName(self):
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

        indicator = np.zeros(self.n_documents, dtype="bool")

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


class RankingBanditsGangAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(RankingBanditsGangAlgorithm, self).__init__(*args, **kwargs)
        try:
            method = kwargs['method']
            if method == 'UCB':
                self.rankers = [CascadeUCB1(self.n_documents, kwargs['alpha'],
                                            first_click=kwargs['first_click'],
                                            random_state=self.random_state)
                                for _ in xrange(self.cutoff)]
            elif method == 'KL-UCB':
                self.rankers = [CascadeKL_UCB(self.n_documents, 
                                              first_click=kwargs['first_click'],
                                              random_state=self.random_state)
                                for _ in xrange(self.cutoff)]
            else:
                raise ValueError('unknown ranking method: %s' % method)

            # Array used for random shuffling of the bandits.
            self.indices = np.arange(self.cutoff, dtype='int32')

            # Auxiliary array for bandit rankings.
            self.__tmp_rankings = np.empty((self.cutoff, self.n_documents),
                                           dtype='int32')

            # The position of documents within each bandits
            # recommendation list.
            self.__tmp_indices = np.empty(self.cutoff, dtype='int32')

            # Feedback for the bandits derived from clicks.
            self.__tmp_feedback = np.zeros(self.cutoff, dtype='int32')

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(RankingBanditsGangAlgorithm, cls).update_parser(parser)
        parser.add_argument('-m', '--method', choices=['UCB', 'KL-UCB'],
                            required=True, help='specify bandit algorithm')
        parser.add_argument('-a', '--alpha', type=float, default=1.5,
                            required=False, help='alpha parameter for UCB '
                            '(used only if "--method UCB" is specified)')
        parser.add_argument('-f', '--first-click', action='store_true',
                            required=False, help='consider feedback up to '
                            'the last click instead of the first (default)')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'RankingBanditsGangAlgorithm'

    def get_ranking(self, ranking):
        '''
        This method combines rankings proposed by each ranking bandit.
        The bandits are ordered randomly and a document not yet appearing 
        in the output ranking is picked from each.
        '''
        # Shuffle the bandits randomly...
        self.random_state.shuffle(self.indices)
        # ... and in that order ...
        for k, i in enumerate(self.indices):
            # ... let each bandit yield a document ...
            for j, d in enumerate(self.rankers[i].get_ranking(self.__tmp_rankings[i])):
                # ...  that has not appeared in the output ranking yet.
                if d not in ranking[:k]:
                    ranking[k] = d
                    self.__tmp_indices[i] = j
                    break

    def set_feedback(self, ranking, clicks):
        for i, c in zip(self.indices, clicks):
            j = self.__tmp_indices[i]
            self.__tmp_feedback[j] = c
            self.rankers[i].set_feedback(self.__tmp_rankings[i],
                                         self.__tmp_feedback[:(j + 1)])
            self.__tmp_feedback[j] = 0

    def cleanup(self):
        print 'RankingBanditsGangAlgorithm.cleanup:'
        print self.__tmp_rankings


class StackedRankingBanditsAlgorithm(BaseRankingBanditAlgorithm):

    def __init__(self, *args, **kwargs):
        super(StackedRankingBanditsAlgorithm, self).__init__(*args, **kwargs)
        try:
            method = kwargs['method']
            if method == 'UCB':
                self.rankers = [CascadeUCB1(self.n_documents, kwargs['alpha'],
                                            first_click=kwargs['first_click'],
                                            random_state=self.random_state)
                                for _ in xrange(self.cutoff)]
                self.preranker = CascadeUCB1(self.cutoff, kwargs['alpha'],
                                             first_click=kwargs['first_click'],
                                             random_state=self.random_state)
            elif method == 'KL-UCB':
                self.rankers = [CascadeKL_UCB(self.n_documents,
                                              first_click=kwargs['first_click'],
                                              random_state=self.random_state)
                                for _ in xrange(self.cutoff)]
                self.preranker = CascadeKL_UCB(self.cutoff,
                                               first_click=kwargs['first_click'],
                                               random_state=self.random_state)
            else:
                raise ValueError('unknown ranking method: %s' % method)

            self.first_click_feedback = kwargs['first_click']

            # Array used for random shuffling of the bandits.
            self.indices = np.arange(self.cutoff, dtype='int32')

            # Auxiliary array for bandit rankings.
            self.__tmp_rankings = np.empty((self.cutoff, self.n_documents),
                                           dtype='int32')

            # The position of documents within each bandits
            # recommendation list.
            self.__tmp_indices = np.empty(self.cutoff, dtype='int32')

            # Feedback for the bandits derived from clicks.
            self.__tmp_feedback = np.zeros(self.cutoff, dtype='int32')

        except KeyError as e:
            raise ValueError('missing %s argument' % e)

    @classmethod
    def update_parser(cls, parser):
        super(StackedRankingBanditsAlgorithm, cls).update_parser(parser)
        parser.add_argument('-m', '--method', choices=['UCB', 'KL-UCB'],
                            required=True, help='specify bandit algorithm')
        parser.add_argument('-a', '--alpha', type=float, default=1.5,
                            required=False, help='alpha parameter for UCB '
                            '(used only if "--method UCB" is specified)')
        parser.add_argument('-f', '--first-click', action='store_true',
                            required=False, help='consider feedback only up to '
                            'the first click instead of the whole feedback list '
                            '(default)')

    def getName(self):
        '''
        Returns the name of the algorithm.
        '''
        return 'StackedRankingBanditsAlgorithm'

    def get_ranking(self, ranking):
        '''
        This method combines rankings proposed by each ranking bandit.
        The bandits are ordered randomly and a document not yet appearing 
        in the output ranking is picked from each.
        '''
        # Shuffle the bandits randomly...
        self.preranker.get_ranking(self.indices)

        # ... and in that order ...
        for k, i in enumerate(self.indices):
            # ... let each bandit yield a document ...
            for j, d in enumerate(self.rankers[i].get_ranking(self.__tmp_rankings[i])):
                # ...  that has not appeared in the output ranking yet.
                if d not in ranking[:k]:
                    ranking[k] = d
                    self.__tmp_indices[i] = j
                    break

    def set_feedback(self, ranking, clicks):
        self.preranker.set_feedback(self.indices, clicks)

        clicks_cutoff = clicks.shape[0]
        
        if self.first_click_feedback:
            clicked_ranks = clicks.nonzero()[0]
            if len(clicked_ranks) > 0:
                clicks_cutoff = clicked_ranks[0] + 1    

        for i, c in zip(self.indices, clicks[:clicks_cutoff]):
            j = self.__tmp_indices[i]
            self.__tmp_feedback[j] = c
            self.rankers[i].set_feedback(self.__tmp_rankings[i],
                                         self.__tmp_feedback[:(j + 1)])
            self.__tmp_feedback[j] = 0

    # def cleanup(self):
    #     print 'RankingBanditsGangAlgorithm.cleanup:'
    #     print self.__tmp_rankings
