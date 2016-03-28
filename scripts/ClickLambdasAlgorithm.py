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
    def __init__(self, n_documents):
        self.lambdas = np.zeros((n_documents, n_documents), dtype='float64')
        self.counts = np.zeros((n_documents, n_documents), dtype='float64')

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

    def __init__(self, n_documents):
        super(SkipClickLambdasAlgorithm, self).__init__(n_documents)
        self.n_viewed = np.zeros((n_documents, n_documents), dtype='float64')

    def update(self, ranking, clicks):
        if clicks.any():
            last_click_rank = np.where(clicks)[0][-1]
        else:
            last_click_rank = 0

        n_documents = len(ranking)

        for i in range(n_documents - 1):
            d_i = ranking[i]

            for j in range(i + 1, n_documents):
                d_j = ranking[j]

                if clicks[i] < clicks[j]:
                    self.lambdas[d_j, d_i] += 1.0

                if j <= last_click_rank:
                    self.n_viewed[d_i, d_j] += 1.0
                    self.n_viewed[d_j, d_i] += 1.0

                    if clicks[i] == clicks[j]:
                        self.lambdas[d_i, d_j] += 0.5
                        self.lambdas[d_j, d_i] += 0.5

                self.counts[d_j, d_i] += 1.0

    def statistics(self):
        return self.lambdas, self.counts, self.n_viewed