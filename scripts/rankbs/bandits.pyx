# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cpython cimport Py_INCREF, PyObject

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport malloc, calloc, free, qsort
from libc.math cimport exp, log, sqrt
from libc.string cimport memcpy

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t
ctypedef np.uint32_t  UINT_t

from numpy import float64 as DOUBLE
from numpy import int32   as INT
from numpy import uint32  as UINT

cdef DOUBLE_t DOUBLE_EPSILON = np.finfo('float64').resolution

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF # alias 2**31 - 1


cdef struct UCB_info_t:
    DOUBLE_t ucb
    INT_t    nonce
    INT_t    index


cdef np.ndarray __wrap_in_1d_double(object base, np.intp_t shape, DOUBLE_t * values):
        '''
        Wraps the given C array into NumPy array. The array keeps
        a reference to the Python object (`base`), which manages
        the underlying memory.
        '''
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(1, &shape, np.NPY_DOUBLE, values)
        Py_INCREF(base)
        array.base = <PyObject*> base
        return array
    

cdef int ucb_info_compare(const void * x, const void * y) nogil:
    cdef UCB_info_t *xx = <UCB_info_t*> x
    cdef UCB_info_t *yy = <UCB_info_t*> y

    if xx.ucb - yy.ucb == 0.0:
        return xx.nonce - yy.nonce
    else:
        return 1 if xx.ucb - yy.ucb < 0.0 else -1


cdef inline UINT_t rand_r(UINT_t *seed) nogil:
    seed[0] ^= <UINT_t> (seed[0] << 13)
    seed[0] ^= <UINT_t> (seed[0] >> 17)
    seed[0] ^= <UINT_t> (seed[0] << 5)

    return (seed[0] & <UINT_t> RAND_R_MAX)


cdef inline INT_t rand_choice(DOUBLE_t *p, INT_t sz, UINT_t *seed) nogil:
    cdef INT_t    i = 0
    cdef DOUBLE_t c = 0.0
    cdef UINT_t   r = rand_r(seed)

    # argmin_{i}: sum_{j=0}{i}p[j] > uniform(0, 1]
    while i < sz - 1:
        c += p[i]
        if r <= (<UINT_t> (c * RAND_R_MAX)):
            break
        i += 1

    return i


cdef inline void shuffle(INT_t * array, INT_t size, UINT_t *seed) nogil:
    cdef INT_t i, j
    for i in range(size - 1):
        j = i + (rand_r(seed) % (size - i))
        array[i], array[j] = array[j], array[i]


cdef DOUBLE_t KLdivergence(DOUBLE_t p, DOUBLE_t q) nogil:
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


cdef DOUBLE_t dKLdivergence(DOUBLE_t p, DOUBLE_t q) nogil:
    return -(1.0 * p / q) + (1.0 - p) / (1.0 - q)


cdef inline DOUBLE_t logsumexp(DOUBLE_t *a, INT_t sz) nogil:
    cdef INT_t    i
    cdef DOUBLE_t sumexp = 0.0, amax = a[0]

    for i in range(sz):
        if amax < a[i]:
            amax = a[i]

    for i in range(sz):
        sumexp += exp(a[i] - amax)

    return amax + log(sumexp)


cdef class CascadeUCB1(object):
    cdef readonly INT_t     L
    cdef readonly INT_t     t
    cdef DOUBLE_t*          S
    cdef DOUBLE_t*          N
    cdef UCB_info_t*        U
    cdef readonly DOUBLE_t  alpha
    cdef readonly bint      first_click
    cdef readonly UINT_t    rand_r_state

    property wins:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.S)
    
    property pulls:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.N)

    property means:
        def __get__(self):
            return self.wins / self.pulls
    
    def __cinit__(self, INT_t L, DOUBLE_t alpha=1.5, bint first_click=True, object random_state=None):
        self.L = L
        self.t = 0
        self.S = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.N = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.U = <UCB_info_t*> malloc(self.L * sizeof(UCB_info_t))
        self.alpha = alpha
        self.first_click = first_click
        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)
    
    def __dealloc__(self):
        free(self.S)
        free(self.N)
        free(self.U)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (CascadeUCB1, (self.L, self.alpha, self.first_click), self.__getstate__())

    def __setstate__(self, d):
        self.t = d['t']
        memcpy(self.S, (<np.ndarray> d['S']).data, self.L * sizeof(DOUBLE_t))
        memcpy(self.N, (<np.ndarray> d['N']).data, self.L * sizeof(DOUBLE_t))
        self.rand_r_state = d['rand_r_state']

    def __getstate__(self):
        d = {}
        d['t'] = self.t
        d['S'] = __wrap_in_1d_double(self, self.L, self.S)
        d['N'] = __wrap_in_1d_double(self, self.L, self.N)
        d['rand_r_state'] = self.rand_r_state
        return d

    def get_ranking(self, np.ndarray[INT_t, ndim=1] ranking=None):
        ''' 
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [self.L], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef INT_t index
        cdef np.ndarray[INT_t, ndim=1] indices = np.empty(self.L, dtype=INT) if ranking is None else ranking

        if indices.size != self.L:
            raise ValueError('ranking array must be 1-d integer array of size %d' % self.L)

        if self.t < self.L:                
            for index in range(self.L):
                indices[index] = index
                    
            indices[0], indices[self.t] = indices[self.t], indices[0]
                
            shuffle(<INT_t*> indices.data + 1, self.L - 1, &self.rand_r_state)
            
        else:
            for index in range(self.L):
                self.U[index].ucb = self.S[index] / self.N[index] + sqrt(self.alpha * log(self.t) / self.N[index])
                self.U[index].nonce = rand_r(&self.rand_r_state)
                self.U[index].index = index

            qsort(self.U, self.L, sizeof(UCB_info_t), ucb_info_compare)

            for index in range(self.L):
                indices[index] = self.U[index].index

        return indices

    def set_feedback(self, np.ndarray[INT_t, ndim=1] ranking, np.ndarray[INT_t, ndim=1] clicks):
        ''' 
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        cdef INT_t i, last_i = clicks.size - 1

        if ranking.size < clicks.size:
            raise ValueError('clicks array size cannot be larger than ranking array size')

        for i in range(clicks.size):
            if clicks[i] == 1:
                last_i = i
                if self.first_click:
                    break
        
        self.t += 1
        for i in range(last_i + 1):
            self.S[ranking[i]] += clicks[i]
            self.N[ranking[i]] += 1.0


cdef class CascadeKL_UCB(object):
    cdef readonly INT_t     L
    cdef readonly INT_t     t
    cdef DOUBLE_t*          S
    cdef DOUBLE_t*          N
    cdef UCB_info_t*        U
    cdef readonly bint      first_click
    cdef readonly UINT_t    rand_r_state

    property wins:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.S)
    
    property pulls:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.N)

    property means:
        def __get__(self):
            return self.wins / self.pulls
    
    def __cinit__(self, INT_t L, first_click=True, object random_state=None):
        self.L = L
        self.t = 0
        self.S = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.N = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.U = <UCB_info_t*> malloc(self.L * sizeof(UCB_info_t))
        self.first_click = first_click
        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)
    
    def __dealloc__(self):
        free(self.S)
        free(self.N)
        free(self.U)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (CascadeKL_UCB, (self.L, self.first_click), self.__getstate__())

    def __setstate__(self, d):
        self.t = d['t']
        memcpy(self.S, (<np.ndarray> d['S']).data, self.L * sizeof(DOUBLE_t))
        memcpy(self.N, (<np.ndarray> d['N']).data, self.L * sizeof(DOUBLE_t))
        self.rand_r_state = d['rand_r_state']

    def __getstate__(self):
        d = {}
        d['t'] = self.t
        d['S'] = __wrap_in_1d_double(self, self.L, self.S)
        d['N'] = __wrap_in_1d_double(self, self.L, self.N)
        d['rand_r_state'] = self.rand_r_state
        return d

    @staticmethod
    cdef DOUBLE_t compute_ucb_bisection(DOUBLE_t p, DOUBLE_t e) nogil:
        '''
        Find q = argmax{q in [p, 1]: KL(p, q) <= e}.
        '''
        cdef DOUBLE_t kld, low, high, q
        
        p = p if p < (1 - DOUBLE_EPSILON) else (1 - DOUBLE_EPSILON)
        p = p if p > DOUBLE_EPSILON else DOUBLE_EPSILON
        
        low  = p
        high = 1.0
        
        while (high - low) > DOUBLE_EPSILON:
            q = (low + high) / 2.0
            kld = KLdivergence(p, q)
            if kld < e:
                low = q
            else:
                high = q

        return (low + high) / 2.0        

    @staticmethod
    cdef DOUBLE_t compute_ucb_newton(DOUBLE_t p, DOUBLE_t e) nogil:
        '''
        Find q = argmax{q in [p, 1]: KL(p, q) <= e}.
        '''
        cdef DOUBLE_t kld, q, prev_q

        p = p if p < (1 - 2 * DOUBLE_EPSILON) else (1 - 2 * DOUBLE_EPSILON)
        p = p if p > DOUBLE_EPSILON else DOUBLE_EPSILON

        prev_q = (1 + DOUBLE_EPSILON)
        q = (1 - DOUBLE_EPSILON)

        if KLdivergence(p, q) > e:
            while (prev_q - q) > DOUBLE_EPSILON:
                prev_q = q
                q += (e - KLdivergence(p, q)) / dKLdivergence(p, q)

        return q
    
    @staticmethod
    def compute_ucb(DOUBLE_t p, DOUBLE_t e, bint newton=True):
        return CascadeKL_UCB.compute_ucb_newton(p, e) if newton else CascadeKL_UCB.compute_ucb_bisection(p, e)
    
    def get_ranking(self, np.ndarray[INT_t, ndim=1] ranking=None):
        ''' 
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [self.L], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef INT_t index
        cdef np.ndarray[INT_t, ndim=1] indices = np.empty(self.L, dtype=INT) if ranking is None else ranking

        if indices.size != self.L:
            raise ValueError('ranking array must be 1-d integer array of size %d' % self.L)
        
        if self.t < self.L:                
            for index in range(self.L):
                indices[index] = index
                    
            indices[0], indices[self.t] = indices[self.t], indices[0]
                
            shuffle(<INT_t*> indices.data + 1, self.L - 1, &self.rand_r_state)
            
        else:
            for index in range(self.L):
                self.U[index].ucb = CascadeKL_UCB.compute_ucb_newton(self.S[index] / self.N[index], log(self.t) / self.N[index])
                self.U[index].nonce = rand_r(&self.rand_r_state)
                self.U[index].index = index

            qsort(self.U, self.L, sizeof(UCB_info_t), ucb_info_compare)

            for index in range(self.L):
                indices[index] = self.U[index].index

        return indices

    def set_feedback(self, np.ndarray[INT_t, ndim=1] ranking, np.ndarray[INT_t, ndim=1] clicks):
        ''' 
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        cdef INT_t i, last_i = clicks.size - 1

        if ranking.size < clicks.size:
            raise ValueError('clicks array size cannot be larger than ranking array size')

        for i in range(clicks.size):
            if clicks[i] == 1:
                last_i = i
                if self.first_click:
                    break
        
        self.t += 1
        for i in range(last_i + 1):
            self.S[ranking[i]] += clicks[i]
            self.N[ranking[i]] += 1.0


cdef class CascadeLambdaMachine(object):
    cdef readonly INT_t     K
    cdef readonly INT_t     L
    cdef readonly INT_t     t
    cdef public INT_t       burnin
    cdef UCB_info_t*        U
    cdef DOUBLE_t*          _lambdas
    cdef DOUBLE_t*          _d_lambdas
    cdef readonly DOUBLE_t  sigma
    cdef readonly UINT_t    rand_r_state
    
    property means:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self._lambdas)
    
    def __cinit__(self, INT_t L, INT_t burnin=0, DOUBLE_t sigma=1.0, object random_state=None):
        self.L = L
        self.t = 0
        self.burnin = burnin
        self.U = <UCB_info_t*> malloc(self.L * sizeof(UCB_info_t))
        self._lambdas = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self._d_lambdas = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.sigma = sigma
        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)
    
    def __dealloc__(self):
        free(self.U)
        free(self._lambdas)
        free(self._d_lambdas)
        
    def __reduce__(self):
        return (CascadeLambdaMachine, (self.L, self.burnin, self.sigma), self.__getstate__())

    def __setstate__(self, d):
        self.t = d['t']
        memcpy(self._lambdas, (<np.ndarray> d['_lambdas']).data, self.L * sizeof(DOUBLE_t))
        self.rand_r_state = d['rand_r_state']

    def __getstate__(self):
        d = {}
        d['t'] = self.t        
        d['_lambdas'] = __wrap_in_1d_double(self, self.L, self._lambdas)
        d['rand_r_state'] = self.rand_r_state
        return d
    
    def get_ranking(self, np.ndarray[INT_t, ndim=1] ranking=None):
        ''' 
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [self.L], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef INT_t index
        cdef np.ndarray[INT_t, ndim=1] indices = np.empty(self.L, dtype=INT) if ranking is None else ranking

        if indices.size != self.L:
            raise ValueError('ranking array must be 1-d integer array of size %d' % self.L)
        
        if self.t < self.burnin:
            for index in range(self.L):
                indices[index] = index

            shuffle(<INT_t*> indices.data , self.L, &self.rand_r_state)

        else:
            for index in range(self.L):
                self.U[index].ucb = self._lambdas[index]
                self.U[index].nonce = rand_r(&self.rand_r_state)
                self.U[index].index = index

            qsort(self.U, self.L, sizeof(UCB_info_t), ucb_info_compare)

            for index in range(self.L):
                indices[index] = self.U[index].index

        return indices

    def set_feedback(self, np.ndarray[INT_t, ndim=1] ranking, np.ndarray[INT_t, ndim=1] clicks):
        ''' 
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        cdef INT_t i, j, fc = -1
        cdef DOUBLE_t lambda_i_j

        if ranking.size < clicks.size:
            raise ValueError('clicks array size cannot be larger than ranking array size')
        
        self.t += 1
        
        # Find first clicked document rank.
        for i in range(clicks.shape[0]):
            if clicks[i] == 1:
                fc = i + 1
                break
                
        # Zero clicks.
        if fc == -1:
            return

        i = fc - 1

        # for j in range(fc):
        #     if clicks[j] != 1:
        #         self._lambdas[ranking[i]] += 1.0
        #         self._lambdas[ranking[j]] -= 1.0

        # For each document pair (not-clicked/clicked) compute
        # update for its documents lambdas.
        for i in range(fc):
            # Document d_i was clicked...
            if clicks[i] == 1:
                for j in range(i):
                    # ... and d_j was not (assuming it was observed!).
                    if clicks[j] != 1:
                        lambda_i_j = self.sigma / (1.0 + exp(self.sigma * (self._lambdas[ranking[i]] - self._lambdas[ranking[j]])))
                        self._d_lambdas[i] += lambda_i_j
                        self._d_lambdas[j] -= lambda_i_j
       
        for i in range(fc):
            self._lambdas[ranking[i]] += self._d_lambdas[i]
            self._d_lambdas[i] = 0.0


cdef class CascadeThompsonSampler(object):
    cdef readonly INT_t     L
    cdef readonly INT_t     t
    cdef DOUBLE_t*          S
    cdef DOUBLE_t*          N
    cdef UCB_info_t*        U
    cdef readonly DOUBLE_t  alpha
    cdef readonly DOUBLE_t  beta
    cdef readonly INT_t     I
    cdef readonly object    random_state

    property wins:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.S)
    
    property pulls:
        def __get__(self):
            return self.wins + __wrap_in_1d_double(self, self.L, self.N)

    property means:
        def __get__(self):
            return self.wins / self.pulls
    
    def __cinit__(self, L, alpha=1.0, beta=1.0, random_state=None):
        cdef INT_t i
        self.L = L
        self.S = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        for i in range(L):
            self.S[i] = alpha
        self.N = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        for i in range(L):
            self.N[i] = beta
        self.U = <UCB_info_t*> malloc(self.L * sizeof(UCB_info_t))
        self.alpha = alpha
        self.beta = beta
        if random_state is None:
            self.random_state = np.random.RandomState(np.random.randint(1, RAND_R_MAX))
        else:
            self.random_state = np.random.RandomState(random_state.randint(1, RAND_R_MAX))
    
    def __dealloc__(self):
        free(self.S)
        free(self.N)
        free(self.U)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (CascadeThompsonSampler, (self.L, self.alpha, self.beta), self.__getstate__())

    def __setstate__(self, d):
        memcpy(self.S, (<np.ndarray> d['S']).data, self.L * sizeof(DOUBLE_t))
        memcpy(self.N, (<np.ndarray> d['N']).data, self.L * sizeof(DOUBLE_t))
        self.random_state = d['random_state']

    def __getstate__(self):
        d = {}
        d['S'] = __wrap_in_1d_double(self, self.L, self.S)
        d['N'] = __wrap_in_1d_double(self, self.L, self.N)
        d['random_state'] = self.random_state
        return d

    def get_ranking(self, np.ndarray[INT_t, ndim=1] ranking=None):
        ''' 
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [self.L], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef INT_t index
        cdef np.ndarray[INT_t, ndim=1] indices = np.empty(self.L, dtype=INT) if ranking is None else ranking

        if indices.size != self.L:
            raise ValueError('ranking array must be 1-d integer array of size %d' % self.L)

        for index in range(self.L):
            self.U[index].ucb = self.random_state.beta(self.S[index], self.N[index])
            self.U[index].nonce = 0 # The probability of drawing the same number above is 0.
            self.U[index].index = index

        qsort(self.U, self.L, sizeof(UCB_info_t), ucb_info_compare)

        for index in range(self.L):
            indices[index] = self.U[index].index

        return indices

    def set_feedback(self, np.ndarray[INT_t, ndim=1] ranking, np.ndarray[INT_t, ndim=1] clicks):
        ''' 
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        cdef INT_t i

        if ranking.size < clicks.size:
            raise ValueError('clicks array size cannot be larger than ranking array size')
        
        for i in range(clicks.shape[0]):
            self.S[ranking[i]] += clicks[i]
            self.N[ranking[i]] += 1.0
            if clicks[i] == 1: break;


cdef class CascadeExp3(object):
    cdef readonly INT_t     L
    cdef readonly INT_t     t
    cdef DOUBLE_t*          S
    cdef DOUBLE_t*          P
    cdef INT_t*             D
    cdef DOUBLE_t*          C
    cdef readonly DOUBLE_t  gamma
    cdef readonly UINT_t    rand_r_state

    property means:
        def __get__(self):
            return __wrap_in_1d_double(self, self.L, self.S)
    
    def __cinit__(self, L, gamma=0.01, random_state=None):
        self.S = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.P = <DOUBLE_t*> calloc(L, sizeof(DOUBLE_t))
        self.L = L
        self.t = 0
        self.gamma = gamma
        self.D = <INT_t*> malloc(self.L * sizeof(INT_t))
        self.C = <DOUBLE_t*> malloc(self.L * sizeof(DOUBLE_t))
        if random_state is None:
            self.rand_r_state = np.random.randint(1, RAND_R_MAX)
        else:
            self.rand_r_state = random_state.randint(1, RAND_R_MAX)
    
    def __dealloc__(self):
        free(self.S)
        free(self.P)
        free(self.D)
        free(self.C)

    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (CascadeExp3, (self.L, self.gamma), self.__getstate__())

    def __setstate__(self, d):
        memcpy(self.S, (<np.ndarray> d['S']).data, self.L * sizeof(DOUBLE_t))
        memcpy(self.P, (<np.ndarray> d['P']).data, self.L * sizeof(DOUBLE_t))
        self.t = d['t']
        self.rand_r_state = d['rand_r_state']

    def __getstate__(self):
        d = {}
        d['S'] = __wrap_in_1d_double(self, self.L, self.S)
        d['P'] = __wrap_in_1d_double(self, self.L, self.P)
        d['t'] = self.t
        d['rand_r_state'] = self.rand_r_state
        return d

    def get_ranking(self, np.ndarray[INT_t, ndim=1] ranking=None):
        ''' 
        Produces a ranking based on the current state of the model.

        Parameters
        ----------
        ranking : array of ints, shape = [self.L], optional
              Optional output array, which can speed up the computation
              because no array is created during the call.
        '''
        cdef np.ndarray[INT_t, ndim=1] _ranking = np.empty(self.L, dtype=INT) if ranking is None else ranking

        cdef INT_t *  ranking_ptr = <INT_t*>_ranking.data
        cdef INT_t    i, j
        cdef DOUBLE_t nC

        if _ranking.size != self.L:
            raise ValueError('ranking array must be 1-d integer array of size %d' % self.L)

        for i in range(self.L):
            self.C[i] = self.S[i]        
            self.D[i] = i

        for i in range(self.L - 1):
            # log of the normalizing constant in the induced Gibbs distribution.
            nC = logsumexp(self.C + i, self.L - i)
            # Probability of picking the next document in the ranking.
            for j in range(i, self.L):
                self.P[j] = (1.0 - self.gamma) * exp(self.C[j] - nC) + self.gamma / (self.L - i)
            # Sample a random document...
            j = i + rand_choice(self.P + i, self.L - i, &self.rand_r_state)
            # ... and put it into the ranking.
            ranking_ptr[i] = self.D[j]
            # Update the structures for the next iteration.
            self.C[j] = self.C[i]
            self.D[j] = self.D[i]

        self.P[self.L - 1] = 1.0
        ranking_ptr[self.L - 1] = self.D[self.L - 1]

        return _ranking

    def set_feedback(self, np.ndarray[INT_t, ndim=1] ranking, np.ndarray[INT_t, ndim=1] clicks):
        ''' 
        Update model parameters based on clicks. The ranking is assumed coming
        from a preceding call to `self.advance` method.

        Parameters
        ----------
        ranking : array of ints
            The ranking produced by a preceding call to `self.advance`.

        clicks : array of ints
            The binary indicator array marking the ranks that received
            a click from the user.
        '''
        cdef INT_t i

        if ranking.size < clicks.size:
            raise ValueError('clicks array size cannot be larger than ranking array size')

        if not clicks.any():
            return

        for i in range(clicks.shape[0]):
            if clicks[i]:
                self.S[ranking[i]] += self.gamma / (self.L - i) / self.P[i]

        # for i in range(clicks.shape[0]):
        #     if clicks[i]: break
        #     self.S[ranking[i]] -= self.gamma / (self.L - i) / self.P[i]