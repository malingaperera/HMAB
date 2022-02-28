import configparser
import datetime
import logging
import os
import subprocess
import uuid
from importlib import reload

import constants
import shared.configs_v2 as configs
from database import sql_connection, sql_helper_v3 as sql_helper
from shared import helper_v2 as helper


class DTARunner:

    def __init__(self, ta_runs, workload_type='optimal', uniform=False, rep=1):
        # workload_types: 'full', 'current', 'optimal', 'last_run'
        db_config = configparser.ConfigParser()
        db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
        db_type = db_config['SYSTEM']['db_type']
        self.server = db_config[db_type]['server']
        self.database = db_config[db_type]['database']
        self.connection = sql_connection.get_sql_connection()
        self.workload_file_current = constants.ROOT_DIR + constants.WORKLOADS_FOLDER + '\\temp_workload_current.sql'
        self.workload_file_optimal = constants.ROOT_DIR + constants.WORKLOADS_FOLDER + '\\temp_workload_optimal.sql'
        self.workload_file_full = constants.ROOT_DIR + constants.WORKLOADS_FOLDER + '\\temp_workload_full.sql'
        self.workload_file_last_run = constants.ROOT_DIR + constants.WORKLOADS_FOLDER + '\\temp_workload_last_run.sql'
        self.workload_type = workload_type
        self.ta_runs = ta_runs
        self.queries = helper.get_queries_v2()
        self.uniform = uniform
        self.rep = rep

    def run(self):
        reload(configs)
        # resets the workload file
        workload_file = open(self.workload_file_current, 'w')
        workload_file.close()
        workload_file_last_run = open(self.workload_file_last_run, 'w')
        workload_file_last_run.close()
        workload_file_optimal = open(self.workload_file_optimal, 'w')
        workload_file_optimal.close()
        results = []

        # setting up logging
        logging.basicConfig(
            filename=helper.get_experiment_folder_path(configs.experiment_id) + configs.experiment_id + '.log',
            filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(constants.LOGGING_LEVEL)
        logging.info(f"============= Starting TA session: {self.workload_type} =============")
        table_list = []
        logging.info(f"Table list: {table_list}")

        next_workload_shift = 0
        previous_workload_shift = 0
        queries_start = configs.queries_start_list[next_workload_shift]
        queries_end = configs.queries_end_list[next_workload_shift]

        # these variables will hold the overall values for each cost
        execution_cost = 0.0
        recommendation_cost = 0.0
        apply_cost = 0.0

        for i in range(configs.rounds):
            logging.info("Round :" + str(i))
            start_time_round = datetime.datetime.now()
            execution_cost_round = 0
            recommend_cost_round = 0
            transactional_cost_round = 0
            analytical_cost_round = 0
            apply_cost_round = 0
            # sql_helper.restart_sql_server()
            # self.connection = sql_connection.get_sql_connection()
            # sql_helper.get_last_restart_time(self.connection)

            # soon after a workload shift we call dta if run_on_workload_shift is True
            if i in set(self.ta_runs):
                recommend_cost_round, recommendation_output_file = self.get_recommendations(previous_workload_shift, table_list)
                if os.path.isfile(recommendation_output_file):
                    apply_cost_round = self.implement_recommendations_v2(recommendation_output_file)
                    # reset last run file which keeps queries from last invoke of TA
                workload_file_last_run = open(self.workload_file_last_run, 'w')
                workload_file_last_run.close()

            # check if workload shift is required
            if i == configs.workload_shifts[next_workload_shift]:
                queries_start = configs.queries_start_list[next_workload_shift]
                queries_end = configs.queries_end_list[next_workload_shift]
                previous_workload_shift = next_workload_shift
                if len(configs.workload_shifts) > next_workload_shift + 1:
                    next_workload_shift += 1

            # executing the queries, we will write the queries the workload file after execution, this work as the
            # workload that we have saw up to now
            query_times = {}
            query_counts = {}
            is_analytical = {}
            with open(self.workload_file_current, 'a+') as workload_file, \
                    open(self.workload_file_optimal, 'w+') as workload_file_optimal, \
                    open(self.workload_file_last_run, 'a+') as workload_file_last_run:
                for query in self.queries[queries_start:queries_end]:
                    query_string = query['query_string'].lower()
                    query_plan = sql_helper.execute_query_v2(self.connection, query_string)
                    if constants.LOG_XML:
                        helper.log_query_xml(configs.experiment_id, query['id'], query_plan.xml, i, constants.COMPONENT_TA)
                    if query_plan and (len(query_plan.non_clustered_view_usages) > 0 or len(query_plan.clustered_view_usages))> 0:
                        logging.error(f"Query {query['id']} used views")
                    if query_plan:
                        cost = query_plan[constants.COST_TYPE_CURRENT_EXECUTION]
                    else:
                        logging.error(f"Query {query['id']} don't have a query plan")
                        cost = 0
                    if query['id'] in query_times:
                        query_times[query['id']] += cost
                        query_counts[query['id']] += 1
                    else:
                        query_times[query['id']] = cost
                        query_counts[query['id']] = 1
                        is_analytical[query['id']] = query_string.strip().startswith('select') or query_string.strip().startswith('with')

                    execution_cost_round += cost
                    if query_string.strip().startswith('select') or query_string.strip().startswith('with'):
                        analytical_cost_round += cost
                    else:
                        transactional_cost_round += cost
                    workload_file.write(query_string)
                    workload_file.write('\n\n\n')
                    workload_file_optimal.write(query_string)
                    workload_file_optimal.write('\n\n\n')
                    workload_file_last_run.write(query_string)
                    workload_file_last_run.write('\n\n\n')
            workload_file_optimal.close()
            workload_file.close()
            workload_file_last_run.close()

            for q_id, q_time in query_times.items():
                logging.info(
                    f"Query {q_id}: \tanalytical-{is_analytical[q_id]} \tcount-{query_counts[q_id]} \tcost-{q_time}")

            execution_cost += execution_cost_round
            recommendation_cost += recommend_cost_round
            apply_cost += apply_cost_round
            batch_time = execution_cost_round + apply_cost_round + recommend_cost_round
            results.append([i, constants.MEASURE_BATCH_TIME, batch_time])
            results.append([i, constants.MEASURE_QUERY_EXECUTION_COST, execution_cost_round])
            results.append([i, constants.MEASURE_INDEX_CREATION_COST, apply_cost_round])
            results.append([i, constants.MEASURE_INDEX_RECOMMENDATION_COST, recommend_cost_round])
            results.append([i, constants.MEASURE_ANALYTICAL_EXECUTION_COST, analytical_cost_round])
            results.append([i, constants.MEASURE_TRANSACTIONAL_EXECUTION_COST, transactional_cost_round])
            logging.info("Execution cost: " + str(execution_cost_round))
            logging.info(f"Time taken for analytical queries: {analytical_cost_round}")
            logging.info(f"Time taken for transactional queries: {transactional_cost_round}")

        self.connection.close()
        total_workload_time = recommendation_cost + apply_cost + execution_cost
        logging.info("Total workload time: " + str(total_workload_time) + "s")

        # Removing the indexes
        connection = sql_connection.get_sql_connection()
        sql_helper.remove_all_non_clustered(connection, constants.SCHEMA_NAME)
        sql_helper.drop_all_dta_statistics(connection)
        sql_connection.close_sql_connection(connection)
        sql_helper.clean_up_routine(sql_connection)
        self.connection = sql_connection.get_sql_connection()

        return results, total_workload_time

    def implement_recommendations_v2(self, recommendation_output_file):
        # Reading the recommendation file
        with open(recommendation_output_file, encoding="utf-16") as f:
            query_lines = f.readlines()
        sql = ' '.join(query_lines)
        sql = sql.replace('go\n', ';')
        time_apply = 0
        if sql:
            queries = sql.split(';')
            for query in queries[1:]:
                if not query.isspace():
                    if "drop " in query.lower():
                        try:
                            sql_helper.simple_execute(self.connection, query)
                        except:
                            # for some reason, when dropping the indices in MVs, TA suggest the clustered index first
                            # When you drop it first, yuo get a error if you try to drop the non-clustered index on
                            # same MV
                            print("Coudn't drop", query)
                    elif " index " in query.lower():
                        time_apply += sql_helper.create_index_v2(self.connection, query)
                    elif " view " in query.lower():
                        sql_helper.create_view_v1(self.connection, query)
                    # elif " statistics " in query.lower():
                    #     time_apply += sql_helper.create_statistics(self.connection, query)
            logging.info("Time taken to apply the recommendations: " + str(time_apply) + "s")
            logging.info("Size taken by the config: " + str(sql_helper.get_current_pds_size(self.connection)) + "MB")
        return time_apply

    def get_recommendations(self, workload_shift, table_list):
        # Tuning Parameters
        reload(configs)
        max_memory = configs.max_memory
        session_name = configs.experiment_id + self.workload_type + "_" + str(uuid.uuid4())
        experiment_folder_path = helper.get_experiment_folder_path(configs.experiment_id)
        recommendation_output_file = experiment_folder_path + configs.experiment_id + "_" + str(workload_shift) + "_" + \
            str(self.rep) + "_" + self.workload_type + "_dta_recommendation.sql"
        session_output_xml_file = experiment_folder_path + configs.experiment_id + "_" + str(workload_shift) + "_" + \
            str(self.rep) + "_" + self.workload_type + "_dta_session_output.xml"
        if self.workload_type == constants.TA_WORKLOAD_TYPE_FULL:
            self.generate_full_workload_file()
            workload_file = self.workload_file_full
        elif self.workload_type == constants.TA_WORKLOAD_TYPE_CURRENT:
            workload_file = self.workload_file_current
        elif self.workload_type == constants.TA_WORKLOAD_TYPE_OPTIMAL:
            workload_file = self.workload_file_optimal
        elif self.workload_type == constants.TA_WORKLOAD_TYPE_SCHEDULE:
            workload_file = self.workload_file_last_run
        else:
            workload_file = self.workload_file_optimal

        # building the command, workload file should be up-to-date with all the queries that we have executed so far
        # add -A 60 if you want to limit the tuning time to 60 mins
        # add -fa IDX_IV to tune clustered, non-clustered indices and views, NCL_IDX for just non-clustered indices,
        # IV for view only
        # -fk CL_IDX - keep clustered index, NONE - Keep none
        command = f"dta -S {self.server} -E -D {self.database} -d {self.database} " \
                  f"-if \"{workload_file}\" -s {session_name}_{workload_shift} " \
                  f"-of \"{recommendation_output_file}\" " \
                  f"-ox \"{session_output_xml_file}\" -F " \
                  f"-B {max_memory} -fp NONE "

        if configs.pds_selection == constants.VIEW_ONLY:
            command = command + "-fa IV -fk NONE "
        elif configs.pds_selection == constants.INDICES_ONLY:
            command = command + "-fa NCL_IDX -fk CL_IDX "
        elif configs.pds_selection == constants.VIEW_AND_INDICES:
            command = command + "-fa IDX_IV -fk CL_IDX "
        else:
            command = command + "-fa IDX_IV -fk CL_IDX "

        # list of tables that we need to tune
        if len(table_list) > 0:
            table_list_str = ', '.join(table_list)
            command = command + "-Tl " + table_list_str

        sql_connection.close_sql_connection(self.connection)
        start_time_execute = datetime.datetime.now()
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, shell=True, stdout=devnull)
        end_time_execute = datetime.datetime.now()
        time_recommend = (end_time_execute - start_time_execute).total_seconds()
        logging.info("Time taken by tuning adviser for recommendation generation: " + str(time_recommend) + "s")
        self.connection = sql_connection.get_sql_connection()
        return time_recommend, recommendation_output_file

    def generate_full_workload_file(self):
        next_workload_shift = 0
        queries_start = configs.queries_start_list[next_workload_shift]
        queries_end = configs.queries_end_list[next_workload_shift]
        with open(self.workload_file_full, 'w+') as workload_file:
            for i in range(configs.rounds):
                # check if workload shift is required
                if i == configs.workload_shifts[next_workload_shift]:
                    queries_start = configs.queries_start_list[next_workload_shift]
                    queries_end = configs.queries_end_list[next_workload_shift]
                    if len(configs.workload_shifts) > next_workload_shift + 1:
                        next_workload_shift += 1

                # executing the queries, we will write the queries the workload file after execution, this work as the
                # workload that we have saw up to now
                for query in self.queries[queries_start:queries_end]:
                    query_string = query['query_string']
                    workload_file.write(query_string)
                    workload_file.write('\n\n\n')
                    workload_file.flush()
            workload_file.close()

