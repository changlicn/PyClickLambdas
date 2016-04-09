# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

from cpython cimport Py_INCREF, PyObject

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport calloc, free, qsort
from libc.math cimport exp, log, sqrt
from libc.string cimport memcpy

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t
ctypedef np.uint32_t  UINT_t

from numpy import int32   as INT
from numpy import uint32  as UINT
from numpy import float64 as DOUBLE

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF # alias 2**31 - 1


cdef np.ndarray __wrap_in_1d_double(object base, np.intp_t shape, DOUBLE_t * values):
    '''
    Wraps the given C array into NumPy array. The array keeps
    a reference to the Python object (`base`), which manages
    the underlying memory.
    '''
    cdef np.ndarray array = np.PyArray_SimpleNewFromData(1, 
                                                         &shape,
                                                         np.NPY_DOUBLE,
                                                         values)
    Py_INCREF(base)
    array.base = <PyObject*> base
    return array


cdef inline UINT_t rand_r(UINT_t *seed) nogil:
    seed[0] ^= <UINT_t> (seed[0] << 13)
    seed[0] ^= <UINT_t> (seed[0] >> 17)
    seed[0] ^= <UINT_t> (seed[0] << 5)

    return (seed[0] & <UINT_t> RAND_R_MAX)


cdef inline INT_t rand_choice(DOUBLE_t *p, INT_t sz, UINT_t *seed, DOUBLE_t p_sum=1.0) nogil:
    cdef INT_t    i = 0
    cdef DOUBLE_t c = 0.0
    cdef UINT_t   r = rand_r(seed)
    # argmin_{i}: sum_{j=0}{i}p[j] > uniform(0, 1]
    while i < sz:
        c += p[i]
        if p_sum * r <= (<UINT_t> (c * RAND_R_MAX)):
            break
        i += 1
    return i


cdef inline DOUBLE_t logsumexp(DOUBLE_t *a, INT_t sz) nogil:
    cdef INT_t    i
    cdef DOUBLE_t sumexp = 0.0, amax = a[0]

    for i in range(sz):
        if amax < a[i]:
            amax = a[i]

    for i in range(sz):
        sumexp += exp(a[i] - amax)

    return amax + log(sumexp)


cdef inline void shuffle(INT_t * array, INT_t size, UINT_t *seed) nogil:
    cdef INT_t i, j
    for i in range(size - 1):
        j = i + (rand_r(seed) % (size - i))
        array[i], array[j] = array[j], array[i]


cdef class UniformRankingSampler(object):
    cdef DOUBLE_t* S
    cdef INT_t     L

    cdef unsigned int rand_r_state

    def __cinit__(self, np.ndarray[DOUBLE_t, ndim=1] scores, random_state=None):
        if (scores < 0.0).any():
            raise ValueError('scores must be non-negative')

        self.L = scores.size
        self.S = <DOUBLE_t*> calloc(self.L, sizeof(DOUBLE_t))
        
        if self.S == NULL:
            free(self.S)
            raise MemoryError()

        memcpy(self.S, scores.data, self.L * sizeof(DOUBLE_t))

        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)

    def __dealloc__(self):
        free(self.S)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (UniformRankingSampler, (__wrap_in_1d_double(self, self.L, self.S),
                                        self.random_state))

    def sample(self, np.ndarray[INT_t, ndim=1] out=None):
        ''' 
        Produces a uniformly random rankings.

        Parameters
        ----------
        out : array-like, shape = [self.L,], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef np.ndarray[INT_t, ndim=1] _ranking = \
            np.empty(self.L, dtype=INT) if out is None else out

        cdef INT_t *ranking = <INT_t*>_ranking.data
        cdef INT_t  i

        if _ranking.size != self.L:
            raise ValueError('out must be 1-d array of size %d' % self.L)

        for i in range(self.L):
            ranking[i] = i

        shuffle(ranking, self.L, &self.rand_r_state)

        return _ranking


cdef class MultinomialRankingSampler(object):
    cdef DOUBLE_t *S
    cdef INT_t    *D
    cdef INT_t     L

    cdef unsigned int rand_r_state

    def __cinit__(self, np.ndarray[DOUBLE_t, ndim=1] scores, random_state=None):
        cdef int i
        cdef DOUBLE_t scores_sum

        scores_sum = scores.sum()

        if scores_sum <= 0.0:
            raise ValueError('scores do not sum to 1.0')

        if (scores < 0.0).any():
            raise ValueError('scores must be non-negative')

        self.L = scores.size
        self.S = <DOUBLE_t*> calloc(self.L, sizeof(DOUBLE_t))
        self.D = <INT_t*> calloc(self.L, sizeof(INT_t))

        if self.S == NULL or self.D == NULL:
            free(self.S)
            free(self.D)
            raise MemoryError()

        memcpy(self.S, scores.data, self.L * sizeof(DOUBLE_t))

        # Renormalize the scores to sum to 1.0 and prepare document ranking.
        for i in range(self.L):
            self.S[i] /= scores_sum
            self.D[i] = i

        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)

    def __dealloc__(self):
        free(self.S)
        free(self.D)

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (MultinomialRankingSampler,
                (__wrap_in_1d_double(self, self.L, self.S), self.rand_r_state))

    def sample(self, np.ndarray[INT_t, ndim=1] out=None):
        '''
        Produces a random rankings by sampling with replacement from a
        multinomial distribution induced by document relevances.

        Parameters
        ----------
        out : array-like, shape = [self.L,], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef np.ndarray[INT_t, ndim=1] _ranking = \
            np.empty(self.L, dtype=INT) if out is None else out

        cdef INT_t   *ranking = <INT_t*>_ranking.data
        cdef INT_t    i, j
        cdef DOUBLE_t Ssum = 1.0

        if _ranking.size != self.L:
            raise ValueError('out must be 1-d array of size %d' % self.L)

        for i in range(self.L):
            # Sample a random document...
            j = i + rand_choice(self.S + i, self.L - i, &self.rand_r_state, p_sum=Ssum)
            # ... put it into the ranking...
            ranking[i] = self.D[j]
            # ... and make the changes needed for the next iteration.
            Ssum -= self.S[j]
            self.D[i], self.D[j] = self.D[j], self.D[i]
            self.S[i], self.S[j] = self.S[j], self.S[i]

        return _ranking


cdef class SoftmaxRankingSampler(object):
    cdef DOUBLE_t* S
    cdef DOUBLE_t* P
    cdef DOUBLE_t* C
    cdef INT_t*    D
    cdef INT_t     L
    cdef DOUBLE_t  G
    
    cdef unsigned int rand_r_state

    def __cinit__(self, np.ndarray[DOUBLE_t, ndim=1] scores, gamma=0.0, random_state=None):
        if (scores < 0.0).any():
            raise ValueError('scores must be non-negative')

        self.L = scores.size
        self.S = <DOUBLE_t*> calloc(self.L, sizeof(DOUBLE_t))
        self.P = <DOUBLE_t*> calloc(self.L, sizeof(DOUBLE_t))
        self.C = <DOUBLE_t*> calloc(self.L, sizeof(DOUBLE_t))
        self.D = <INT_t*>    calloc(self.L, sizeof(INT_t))

        if (self.S == NULL or self.P == NULL or self.C == NULL or self.D == NULL):
            free(self.S)
            free(self.P)
            free(self.C)
            free(self.D)
            raise MemoryError()

        memcpy(self.S, scores.data, self.L * sizeof(DOUBLE_t))

        self.G = gamma
        
        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)
    
    def __dealloc__(self):
        free(self.S)
        free(self.P)
        free(self.C)
        free(self.D)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (SoftmaxRankingSampler, (__wrap_in_1d_double(self, self.L, self.S),
                                 self.G, self.random_state))

    def sample(self, np.ndarray[INT_t, ndim=1] out=None):
        ''' 
        Produces a ranking based on the document scores using softmax function.

        Parameters
        ----------
        out : array-like, shape = [self.L,], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef np.ndarray[INT_t, ndim=1] _ranking = \
            np.empty(self.L, dtype=INT) if out is None else out

        cdef INT_t *  ranking = <INT_t*>_ranking.data
        cdef INT_t    i, j
        cdef DOUBLE_t nC

        if _ranking.size != self.L:
            raise ValueError('out must be 1-d array of size %d' % self.L)

        for i in range(self.L):
            self.C[i] = self.S[i]
            self.D[i] = i

        for i in range(self.L - 1):            
            # log of the normalizing constant in the induced Gibbs distribution.
            nC = logsumexp(self.C + i, self.L - i)
            # Probability of picking the next document in the ranking.
            for j in range(i, self.L):
                self.P[j] = (1.0 - self.G) * exp(self.C[j] - nC) + self.G / (self.L - i)
            # Sample a random document...
            j = i + rand_choice(self.P + i, self.L - i, &self.rand_r_state)
            # ... and put it into the ranking.
            ranking[i] = self.D[j]
            # Update the structures for the next iteration.
            self.C[j] = self.C[i]
            self.D[j] = self.D[i]

        ranking[self.L - 1] = self.D[self.L - 1]

        return _ranking