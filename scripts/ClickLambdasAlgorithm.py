# -*- coding: utf-8 -*-

import sys
import inspect
import numpy as np


def get_available_algorithms():
    algorithms = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if (not name.startswith('Base') and name.endswith('Algorithm')):
            algorithms.append(name)
    return algorithms


class BaseClickLambdasAlgorithm(object):
    '''
    Base class for implementation of an algorithm responsible for calculation
    and updates of what we call "click-based lambdas" from the feedback
    (clicks) on an impression of a particular (fixed) query.
    '''
    def __init__(self, n_documents, cutoff):
        self.n_documents = n_documents
        self.cutoff = cutoff

    def update(self, ranking, clicks):
        '''
        Update lambdas using the clicks from a simulated user on the ranking.

        Parameters
        ----------
        ranking : array, shape = [n_documents]
            A list of document IDs (query impression).

        clicks : array, shape = [n_documents]
            Click feedback produced by a simulated user on `ranking`.
        '''
        pass

    def reset(self):
        '''
        Reset the lambdas and impression counts.
        '''
        pass

    def statistics(self):
        '''
        Returns click-based lambdas and impression counts. It is expected that
        extension classes are going to return more statistics then just these
        two.

        Returns
        -------
        lambdas : array, shape = [n_documents, n_documents]
            A square matrix of lambdas.

        counts : array, shape = [n_documents, n_documents]
            A square matrix of counts.
        '''
        pass


class SkipClickLambdasAlgorithm(BaseClickLambdasAlgorithm):

    def __init__(self, n_documents, cutoff):
        super(SkipClickLambdasAlgorithm, self).__init__(n_documents, cutoff)
        self.lambdas = np.empty((n_documents, n_documents), dtype='float64')
        self.counts = np.empty((n_documents, n_documents), dtype='float64')
        self.reset()

    def update(self, ranking, clicks):
        for i in range(self.cutoff - 1):
            d_i = ranking[i]
            for j in range(i + 1, self.cutoff):
                d_j = ranking[j]
                if clicks[i] < clicks[j]:
                    self.lambdas[d_j, d_i] += 1.0
                self.counts[d_j, d_i] += 1.0

    def reset(self):
        self.lambdas.fill(1.0)
        self.counts.fill(2.0)

    def statistics(self):
        return self.lambdas, self.counts


class RefinedSkipClickLambdasAlgorithm(BaseClickLambdasAlgorithm):

    def __init__(self, n_documents, cutoff):
        super(RefinedSkipClickLambdasAlgorithm, self).__init__(n_documents, cutoff)
        self.lambdas = np.empty((n_documents, n_documents, cutoff, cutoff),
                                dtype='float64')
        self.counts = np.empty((n_documents, n_documents, cutoff, cutoff),
                               dtype='float64')
        self.reset()

    def update(self, ranking, clicks):
        for i in range(self.cutoff - 1):
            d_i = ranking[i]
            for j in range(i + 1, self.cutoff):
                d_j = ranking[j]
                if clicks[i] < clicks[j]:
                    self.lambdas[d_j, d_i, j, i] += 1.0
                self.counts[d_j, d_i, j, i] += 1.0

    def reset(self):
        self.lambdas.fill(1.0)
        self.counts.fill(2.0)

    def statistics(self):
        return self.lambdas, self.counts