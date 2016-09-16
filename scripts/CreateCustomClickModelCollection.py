#!/usr/bin/python2
# -*- coding: utf-8 -*-

'''
Creates click models with parameters set by hand.
'''

import numpy as np
import cPickle as pickle

from users import CascadeModel
from users import DependentClickModel
from users import ClickChainUserModel
from users import PositionBasedModel


if __name__ == '__main__':
    #===============================================================================
    # "Model-Query-Data" dictionary keeps track of the data
    # for each of (click model, query) pair.
    #===============================================================================
    MQD = {}
    
    # Every document has its own attractiveness. This attractiveness is shared
    # across all the considered click models.
    r1 = np.array([(0.5)**x for x in range(10)])
    r1 /= r1.sum()
    
    r2 = np.array([(1.0 / (x + 1)) for x in range(10)])
    r2 /= r2.sum()
    
    r3 = np.array([(10.0 - x) for x in range(10)])
    r3 /= r3.sum()
    
    r4 = np.array([1.0 / np.log(1 + np.log((x + 1) * np.e)) for x in range(10)]);
    r4 /= r4.sum()
    
    # The document relevances for each query.
    Rs = [r1, r2, r3, r4]
    
    MQD['CM'] = {}
    for query, relevances in enumerate(Rs, 1):
        MQD['CM'][`query`] = {}        
        MQD['CM'][`query`]['model'] = CascadeModel(relevances)
        MQD['CM'][`query`]['query'] = `query`
        MQD['CM'][`query`]['relevances'] = relevances
    
    MQD['PBM'] = {}
    for query, relevances in enumerate(Rs, 1):
        examination_probabilities = np.array([(10.0 - x) / 10. for x in range(10)])
        MQD['PBM'][`query`] = {}
        MQD['PBM'][`query`]['model'] = PositionBasedModel(relevances, examination_probabilities)        
        MQD['PBM'][`query`]['query'] = `query`
        MQD['PBM'][`query`]['relevances'] = relevances
    
    MQD['DCM'] = {}
    for query, relevances in enumerate(Rs, 1):
        abandonment_probabilities = np.array([0.65 * (0.9)**i for i in range(10)])
        MQD['DCM'][`query`] = {}        
        MQD['DCM'][`query`]['model'] = DependentClickModel(relevances, abandonment_probabilities)
        MQD['DCM'][`query`]['query'] = `query`
        MQD['DCM'][`query`]['relevances'] = relevances
        
    MQD['CCM'] = {}
    for query, relevances in enumerate(Rs, 1):
        MQD['CCM'][`query`] = {}
        # Parameters of the model: probability of continuation after no click, after clicking
        # and finding document not relevant, and continuing after clicking on relevant document.
        MQD['CCM'][`query`]['model'] = ClickChainUserModel(relevances, 1.0 - 1e-4, 0.95, 1e-4)
        MQD['CCM'][`query`]['query'] = `query`
        MQD['CCM'][`query`]['relevances'] = relevances
            
    with open('./data/model_query_collection_custom.pkl', 'wb') as ofile:
        pickle.dump(MQD, ofile, protocol=-1)
