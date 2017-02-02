import numpy as np


class ClickModelCombinator(object):
    def __init__(self, models, ps, name=None, random_state=None):
        '''
        Combines the given user behaviour models such that
        each model (models[i]) is used in proportion to 
        the corresponding probability (ps[i]).
        
        Note: coefs need to sum to 1.0

        Parameters
        ----------
        models : array of click models, shape=(n,)
            List of models to combine.

        ps : array, shape=(n,)
            List of probabilities that determine
            how often (on average) the corresponding
            click model is used to generate feedback.

        name : str, optiona, default=None
            The name that will override the name
            generated automatically if name is None.
        '''
        self.models = np.array(models, dtype='O')
        self.ps  = np.array(ps, dtype='f4')

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)

        if not np.isclose(self.ps.sum(), 1.0):
            raise ValueError('sum of ps does not equal 1.0')

    def getName(self):
        ps = map(lambda p: `round(p, 2)`, self.ps)
        ms = map(lambda m: m.getName(), self.models)
        return '+'.join([p + '*' +  m for p, m in zip(ps, ms)])

    def get_ideal_ranking(self, cutoff=-1, satisfied=False):
        # XXX: What to return here when the ideal rankings of individual
        #      click models do not match? --- Currently, the ranking of
        #      the 1st model is returned.
        ideal_ranking = self.models[0].get_ideal_ranking(cutoff=cutoff, satisfied=satisfied)
        for m in self.models[1:]:
            if (ideal_ranking != m.get_ideal_ranking(cutoff=cutoff, satisfied=satisfied)).any():
                raise ValueError('ideal rankings of the combined models do not match')
        return ideal_ranking

    def get_clicks(self, ranked_documents, labels, cutoff=2**31-1):
        return self.random_state.choice(self.models, p=self.ps).get_clicks(ranked_documents, labels, cutoff=cutoff)

    def __setattr__(self, name, value):
        if name == 'seed':
            if not isinstance(value, int):
                raise ValueError('seed must be an integer')
            self.random_state = np.random.RandomState(value)
        super(ClickModelCombinator, self).__setattr__(name, value)

    def get_clickthrough_rate(self, ranked_documents, labels, cutoff=2**31-1, relative=False):
        return np.sum([p * m.get_clickthrough_rate(ranked_documents, labels, cutoff, relative) for p, m in zip(self.ps, self.models)])
