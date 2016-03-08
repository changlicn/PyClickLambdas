import os
import sys

import numpy as np
import cPickle as pickle

from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.click_models.DCM import DCM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.CCM import CCM
from pyclick.click_models.UBM import UBM
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


if __name__ == "__main__":
    search_sessions_path = './data/YandexClicks.txt'
    click_model_output_path = './data'

    models_to_train = [('CM',  'CM_click_model_10_queries_1M_sessions'),
                       ('PBM', 'PBM_click_model_10_queries_1M_sessions'),
                       ('DCM', 'DCM_click_model_10_queries_1M_sessions'),
                       ('DBN', 'DBN_click_model_10_queries_1M_sessions'),
                       ('CCM', 'CCM_click_model_10_queries_1M_sessions'),
                       ('UBM', 'UBM_click_model_10_queries_1M_sessions')]

    if len(sys.argv) > 1:
        models_to_train = [mtt for mtt in models_to_train if mtt[0] == sys.argv[1]]

    if len(models_to_train) == 0:
        raise ValueError('no model to train has been specified')

    if not os.path.exists('./data/query_stats.pkl'):
        query_stats = YandexRelPredChallengeParser().query_statistics(search_sessions_path)

        with open('./data/query_stats.pkl', 'wb') as fout:
            pickle.dump(query_stats, fout, protocol=-1)
    else:
        with open('./data/query_stats.pkl') as fin:
            query_stats = pickle.load(fin)

    # For reproducible results the seed is fixed to 42.
    random_state = np.random.RandomState(42)

    if not os.path.exists('./data/search_sessions.pkl'):
        # Apply a simple filter on the queries:
        #   - min. # of impressions 500
        #   - min. # of clicks 100
        #   - at least 0.01 click per query on average
        filtered_queries = np.vstack(filter(lambda x: x[1] >= 500 and x[2] >= 100 and 1.0 * x[2] / x[1] > 0.01, query_stats))
        filtered_queries = filtered_queries[random_state.choice(filtered_queries.shape[0], 10)][:,0]

        search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, query_ids=map(str, filtered_queries))

        with open('./data/search_sessions.pkl', 'wb') as fout:
            pickle.dump(search_sessions, fout, protocol=-1)
    else:
        with open('./data/search_sessions.pkl') as fin:
            search_sessions = pickle.load(fin)

        # Because we are interested mostly in getting parameter estimates
        # for theses session queries, they are also used as 'hold-out set'.
        holdout_search_sessions = list(search_sessions)

        n_search_sessions = 1000000

        # Reads `n_search_session` number of sessions from the Yandex dataset ...
        holdout_search_sessions = Utils.filter_sessions(YandexRelPredChallengeParser().parse(search_sessions_path, sessions_max=n_search_sessions),
                                                        Utils.get_unique_queries(search_sessions),
                                                        operation='remove')

        # ... and merges them with `search_sessions` while taking care of not
        # including any of these sessions twice.
        search_sessions.extend(holdout_search_sessions)

    queries = Utils.get_unique_queries(search_sessions)

    print "---------------------------------------------------------"
    print "Training on %d search sessions (%d unique queries)." % (len(search_sessions), len(queries))
    print "---------------------------------------------------------"

    for model_name, model_filename in models_to_train:
        # Makes sure the output file for model parameters
        # does not exist (overwrite protection).
        if os.path.exists(os.path.join(click_model_output_path, model_filename + '.pkl')):
            print ('ERROR: output file (%s) for trained %s model already exists -- skipping'
                   % (model_filename, model_name))
            continue

        # ====================================================================
        # CM Click Model
        # ====================================================================

        if model_name == 'CM':
            click_model = CM()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr])]
    
        # ====================================================================
        # PBM Click Model
        # ====================================================================

        if model_name == 'PBM':
            click_model = PBM()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr]),
                                  ('exam', click_model.params[click_model.param_names.exam])]

        # ====================================================================
        # DCM Click Model
        # ====================================================================

        if model_name == 'DCM':
            click_model = DCM()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr]),
                                  ('cont', click_model.params[click_model.param_names.cont])]    

        # ====================================================================
        # DBN Click Model
        # ====================================================================

        if model_name == 'DBN':
            click_model = DBN()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr]),
                                  ('sat',  click_model.params[click_model.param_names.sat]),
                                  ('cont',  click_model.params[click_model.param_names.cont])]
    
        # ====================================================================
        # CCM Click Model
        # ====================================================================

        if model_name == 'CCM':
            click_model = CCM()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr]),
                                  ('cont_noclick', click_model.params[click_model.param_names.cont_noclick]),
                                  ('cont_click_nonrel',  click_model.params[click_model.param_names.cont_click_nonrel]),
                                  ('cont_click_rel',  click_model.params[click_model.param_names.cont_click_rel])]
    
        # ====================================================================
        # UBM Click Model
        # ====================================================================

        if model_name == 'UBM':
            click_model = UBM()
            click_model.train(search_sessions, holdout_search_sessions)

            click_model_params = [('attr', click_model.params[click_model.param_names.attr]),
                                  ('exam', click_model.params[click_model.param_names.exam])]

        # ====================================================================
        # Save click model parameters
        # ====================================================================

        with open(os.path.join(click_model_output_path, model_filename + '.pkl'), 'wb') as fout:
            pickle.dump(click_model_params, fout, protocol=-1)

        del click_model, click_model_params
