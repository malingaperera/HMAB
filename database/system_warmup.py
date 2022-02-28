import logging
from importlib import reload

import constants
import shared.configs_v2 as configs
from database import sql_connection, sql_helper_v3 as sql_helper
from shared import helper_v2 as helper


class SysWarm:
    @staticmethod
    def run():
        reload(configs)
        logging.basicConfig(
            filename=helper.get_experiment_folder_path(configs.experiment_id) + configs.experiment_id + '.log',
            filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(constants.LOGGING_LEVEL)
        logging.info("Starting System warming-up")
        queries_start = 0
        queries_end = configs.queries_start_list[0]
        queries = helper.get_queries_v2()
        connection = sql_connection.get_sql_connection()
        loading_bar = '|'
        bar_size = queries_end/100
        current_bar = 0
        run_analytical = True
        run_transactional = False

        execution_cost = 0
        analytical_cost = 0
        transactional_cost = 0
        print(loading_bar*100)

        count = 0
        query_times = {}
        query_counts = {}
        is_analytical = {}
        for query in queries[queries_start:queries_end]:
            count += 1
            if count / bar_size > current_bar:
                print(loading_bar, end='')
                current_bar += 1
            if query['query_string'].strip().startswith('select'):
                if run_analytical:
                    query_plan = sql_helper.execute_query_v2(connection, query['query_string'], False)
                    cost = query_plan[constants.COST_TYPE_CURRENT_EXECUTION] if query_plan else 0
                else:
                    cost = 0
                analytical_cost += cost
            else:
                if run_transactional:
                    query_plan = sql_helper.execute_query_v2(connection, query['query_string'], False)
                    cost = query_plan[constants.COST_TYPE_CURRENT_EXECUTION] if query_plan else 0
                else:
                    cost = 0
                transactional_cost += cost
            execution_cost += cost
            if query['id'] in query_times:
                query_times[query['id']] += cost
                query_counts[query['id']] += 1
            else:
                query_times[query['id']] = cost
                query_counts[query['id']] = 1
                is_analytical[query['id']] = query['query_string'].strip().startswith('select')

        for q_id, q_time in query_times.items():
            logging.info(
                f"Query {q_id}: \tanalytical-{is_analytical[q_id]} \tcount-{query_counts[q_id]} \tcost-{q_time}")
        logging.info("Execution cost: " + str(execution_cost))
        logging.info("Analytical cost: " + str(analytical_cost))
        logging.info("Transactional cost: " + str(transactional_cost))

        connection.close()
        return
