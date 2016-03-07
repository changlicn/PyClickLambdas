#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

from collections import defaultdict

__author__ = 'Ilya Markov, Bart Vredebregt, Nick de Wolf'


class YandexRelPredChallengeParser:
    """
    A parser for the publicly available dataset, released by Yandex (https://www.yandex.com)
    for the Relevance Prediction Challenge (http://imat-relpred.yandex.ru/en).
    """

    @staticmethod
    def parse(sessions_filename, sessions_max=None, query_ids=None):
        """
        Parses search sessions, formatted according to the Yandex Relevance Prediction Challenge (RPC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.

        An RPC file contains lines of two formats:
        1. Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action
        SessionID TimePassed TypeOfAction URLID

        :param sessions_filename: The name of the file with search sessions formatted according to RPC.
        :param sessions_max: The maximum number of search sessions to return.
        If not set, all search sessions are parsed and returned.

        :returns: A list of parsed search sessions, wrapped into SearchSession objects.
        """
        sessions_file = open(sessions_filename, "r")
        sessions = []

        if query_ids is not None:
            query_ids = set(query_ids)

        for lineno, line in enumerate(sessions_file, 1):
            if (sessions_max is not None) and (len(sessions) >= sessions_max):
                break

            entry_array = line.strip().split("\t")

            # If the entry has 6 or more elements it is a query
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                task = None
                query = entry_array[3]
                # Ignore queries which are listed in `query_ids` (if given).
                if (query_ids is not None) and (query not in query_ids):
                    continue
                task = entry_array[0]
                results = entry_array[5:]
                session = TaskCentricSearchSession(task, query)

                for result in results:
                    result = SearchResult(result, 0)
                    session.web_results.append(result)

                sessions.append(session)

            # If the entry has 4 elements it is a click
            elif len(entry_array) == 4 and entry_array[2] == "C":
                if entry_array[0] == task:
                    clicked_result = entry_array[3]
                    if clicked_result in results:
                        index = results.index(clicked_result)
                        session.web_results[index].click = 1

            # Else it is an unknown data format so leave it out
            else:
                continue

            if lineno % 1000000 == 0:
                print 'progress: %2.2f' % ((lineno / 340796067.0) * 100.0)

        return sessions

    @staticmethod
    def query_statistics(sessions_filename):
        """
        Parses search sessions, formatted according to the Yandex Relevance Prediction Challenge (RPC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.

        An RPC file contains lines of two formats:
        1. Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action
        SessionID TimePassed TypeOfAction URLID

        :param sessions_filename: The name of the file with search sessions formatted according to RPC.
        :param sessions_max: The maximum number of search sessions to return.
        If not set, all search sessions are parsed and returned.

        :returns: A list of parsed search sessions, wrapped into SearchSession objects.
        """
        query_id = None
        query_n_impressions = defaultdict(int)
        query_n_clicks = defaultdict(int)

        with open(sessions_filename, "r") as sessions_file:
            for lineno, line in enumerate(sessions_file, 1):
                entry_array = line.strip().split("\t")

                # If the entry has 6 or more elements it is a query
                if len(entry_array) >= 6 and entry_array[2] == "Q":
                    query_id = int(entry_array[3])
                    query_n_impressions[query_id] += 1

                # If the entry has 4 elements it is a click
                elif len(entry_array) == 4 and entry_array[2] == "C":
                    query_n_clicks[query_id] += 1

                # Else it is an unknown data format so leave it out
                else:
                    continue

                if lineno % 1000000 == 0:
                    print 'Progress: %2.2f' % ((lineno / 340796067.0) * 100.0)

        stats = []

        for query_id in query_n_impressions:
            stats.append((query_id, query_n_impressions[query_id], query_n_clicks[query_id]))

        return np.array(stats, dtype='int64')
