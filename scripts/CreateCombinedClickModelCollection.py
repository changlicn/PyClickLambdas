#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle

from users import ClickModelCombinator

# Load the 60 queries trained on Yandex dataset.
MQD = np.load('data/60Q/model_query_collection.pkl')

# The selected queries for which we want to create
# the new collection.
Qs = map(str, [104183, 11527, 128292, 46254, 218954, 89951])

# Use this to choose all queries:
# Q = MQD[MQD.keys()[0]].keys()

# The selected click models to be combined and 
# corresponding probabilities with which they
# will be used to generated feedback.
Ms = ['PBM', 'CM']
Ps = [0.5, 0.5]

new_MQD = {}

for q in Qs:
    new_MQD[q] = { 'model': ClickModelCombinator([MQD[m][q]['model'] for m in Ms], Ps),
                   # XXX: This need to be set to something, but I do not
                   # see a reasonable value for it.
                   'relevances': np.zeros(10, dtype='f4'),
                   'query': q }

# Give the new click model a name. THIS IS IMPORTANT
# because it will be used to refer to the click model
# in the experiments.
new_MQD = { 'PBM+CM': new_MQD }

with open('./data/PBM+CM_model_query_collection.pkl', 'wb') as ofile:
    pickle.dump(new_MQD, ofile, protocol=-1)
