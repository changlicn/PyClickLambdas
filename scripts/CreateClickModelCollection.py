# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle

from itertools import groupby

from users import CascadeUserModel
from users import PositionBasedModel
from users import DependentClickModel
from users import ClickChainUserModel
from users import UserBrowsingModel


def load_session_queries(source='./data/search_sessions.pkl'):
    '''
    Load the pre-selected queries.
    '''
    # Make the random selection of queries preproducible.
    random_state = np.random.RandomState(42)

    query_sessions = []

    # Load query sessions which...
    with open(source, 'r') as sessions_file:
        # ... have at least 10 documents...
        sessions = filter(lambda session: len(session.web_results) >= 10,
                          pickle.load(sessions_file))
        # ... group them by query ID...
        qid = lambda session: session.query
        for query, group in groupby(sorted(sessions, key=qid), key=qid):
            # ... and randomly pick a SERP for each query.
            query_sessions.append(random_state.choice(list(group)))

    return query_sessions


def get_click_model_for_session_query(session, click_model_type, seed=None):
    '''
    Load parameters of a click model for a query in the session.
    
    Parameters
    ----------
    session : dict
        A query session.
    
    click_model_type : str
        The click model to load. Supported values are: 'CM', 'PBM', 'DCM'
        'DBN', 'CCM', 'UBM'.
    
    seed : int
        A seed for random number generators in the click models. Can
        be used to get reproducible results.
    
    Returns
    -------
    click_model : click model
        The click model with parameters loaded specificaly
        for the query.
    
    relevance_scores : array of float, shape = [n_documents]
        The relevance scores or the so-called attractivenesses
        of documents estimated for the click model.
    '''
    if click_model_type not in ['CM', 'PBM', 'DCM', 'DBN', 'CCM', 'UBM']:
        raise ValueError('unkown click model: %s' % click_model_type)

    # Load the trained parameters for the click model.
    with open('./data/%s_click_model_10_queries_1M_sessions.pkl'
              % click_model_type) as cmp_file:
        click_model_params = pickle.load(cmp_file)

    qid = session.query
    docids = [doc.id for doc in session.web_results]
    
    if click_model_type == 'CM':
        if click_model_params[0][0] != 'attr':
            raise ValueError('given parameters are not for CM model')

        click_model = CascadeUserModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                       [1.0] * len(docids), abandon_proba=0.0, seed=seed)

        relevance_scores = click_model.click_proba
    
    if click_model_type == 'PBM':
        if (click_model_params[0][0] != 'attr' or
            click_model_params[1][0] != 'exam'):
            raise ValueError('given parameters are not for PBM model')

        # There is no interface method to find this out.
        max_rank = len(click_model_params[1][1]._container)
        
        click_model = PositionBasedModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                         [click_model_params[1][1].get(rank).value() for rank in range(max_rank)],
                                         seed=seed)
        
        relevance_scores = click_model.click_proba
    
    if click_model_type == 'DCM':
        if (click_model_params[0][0] != 'attr' or
            click_model_params[1][0] != 'cont'):
            raise ValueError('given parameters are not for DCM model')
        
        # There is no interface method to find this out.
        max_rank = len(click_model_params[1][1]._container)

        click_model = DependentClickModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                          [1.0 - click_model_params[1][1].get(rank).value() for rank in range(max_rank)],
                                          seed=seed)

        relevance_scores = click_model.click_proba

    if click_model_type == 'DBN':
        if (click_model_params[0][0] != 'attr' or 
            click_model_params[1][0] != 'sat' or 
            click_model_params[2][0] != 'cont'):
            raise ValueError('given parameters are not for DBN model')

        click_model = CascadeUserModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                       [click_model_params[1][1].get(qid, docid).value() for docid in docids],
                                       abandon_proba=(1 - click_model_params[2][1].get().value()),
                                       seed=seed)

        relevance_scores = click_model.click_proba

    if click_model_type == 'CCM':
        if (click_model_params[0][0] != 'attr' or
            click_model_params[1][0] != 'cont_noclick' or
            click_model_params[2][0] != 'cont_click_nonrel' or
            click_model_params[3][0] != 'cont_click_rel'):
            raise ValueError('given parameters are not for CCM model')

        click_model = ClickChainUserModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                          click_model_params[1][1].get().value(),
                                          click_model_params[2][1].get().value(),
                                          click_model_params[3][1].get().value(),
                                          seed=seed)

        relevance_scores = click_model.p_attraction

    if click_model_type == 'UBM':
        if (click_model_params[0][0] != 'attr' or 
            click_model_params[1][0] != 'exam'):
            raise ValueError('given parameters are not for UBM model')

        # There is no interface method to find this out.
        max_rank = len(click_model_params[1][1]._container)

        click_model = UserBrowsingModel([click_model_params[0][1].get(qid, docid).value() for docid in docids],
                                        [[click_model_params[1][1].get(rank, prev_rank).value() for prev_rank in range(max_rank)] for rank in range(max_rank)],
                                        seed=seed)

        relevance_scores = click_model.p_attraction

    return click_model, relevance_scores


if __name__ == '__main__':
    #===============================================================================
    # "Model-Query-Data" dictionary keeps track of the data
    # for each of (click model, query) pair
    #===============================================================================
    MQD = {}

    for session in load_session_queries():
        for click_model_type in ['CM', 'PBM', 'DCM', 'CCM', 'DBN', 'UBM']:
            # Load click model and get ideal ranking for the documents
            # and their relevances (estimated attractiveness).
            model, relevances = get_click_model_for_session_query(
                                                session, click_model_type,
                                                seed=None)
        
            # Instantiate an entry for the current click model.
            if click_model_type not in MQD:
                MQD[click_model_type] = {}
        
            QD = MQD[click_model_type]
        
            # Instantiate an entry for the current query.
            if session.query not in QD:
                QD[session.query] = {}
        
            data = QD[session.query]
        
            data['model'] = model
            data['query'] = session.query
            data['relevances'] = relevances

    with open('./data/model_query_collection.pkl', 'wb') as ofile:
        pickle.dump(MQD, ofile, protocol=-1)
