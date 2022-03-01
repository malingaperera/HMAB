import datetime
import logging
import operator
import pprint
from importlib import reload

import numpy
from pandas import DataFrame

import bandits.bandit_c2ucb_v1 as bandits
import bandits.bandit_helper_v1 as bandit_helper
import constants as constants
import database.sql_connection as sql_connection
import shared.configs_v2 as configs
import shared.helper_v2 as helper
from bandits.experiment_report import ExpReport
from bandits.oracleMV_v3 import OracleV1 as OracleMV
from bandits.oracle_super import OracleV1 as OracleS
from bandits.oracle_v1 import OracleV1 as Oracle
from bandits.query_v1 import Query
from database.sql_helper_factory import SQLHelperFactory

# First per table bandit

sql_helper = SQLHelperFactory.get_sql_helper()


class BaseSimulator:
    def __init__(self):
        # configuring the logger
        logging.basicConfig(
            filename=helper.get_experiment_folder_path(configs.experiment_id) + configs.experiment_id + '.log',
            filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(logging.DEBUG)

        # Get the query List
        self.queries = helper.get_queries_v2()
        self.connection = sql_connection.get_sql_connection()
        self.query_properties = helper.get_query_properties()
        self.query_obj_store = {}
        reload(bandit_helper)


class Simulator(BaseSimulator):

    def run(self):
        pp = pprint.PrettyPrinter()
        reload(configs)
        results = []
        logging.info("Logging configs...\n")
        helper.log_configs(logging, configs)
        logging.info("Logging constants...\n")
        helper.log_configs(logging, constants)
        logging.info("Starting MAB...\n")
        table_list = sql_helper.get_table_list(self.connection, constants.SMALL_TABLE_IGNORE)
        logging.info(f"tuning for tables: {table_list}")
        mv = 'MV'
        with_mv = configs.pds_selection in {constants.VIEW_AND_INDICES, constants.VIEW_ONLY}
        super_static_context_size = 2
        hyp_check_rounds = 25

        # reset hyp query log
        hyp_file_path = helper.get_experiment_folder_path(configs.experiment_id) + configs.experiment_id + '_hyp.sql'

        # Get all the columns from the database
        bandits_dict = {}
        columns = {}
        column_counts = {}

        configs.max_memory -= int(sql_helper.get_current_pds_size(self.connection))
        # Creating bandits for tables
        cluster_id = 1
        for table_name in table_list:
            columns[table_name], column_counts[table_name] = sql_helper.get_table_columns(self.connection, table_name)
            context_size = column_counts[table_name] * (
                    1 + constants.CONTEXT_UNIQUENESS + constants.CONTEXT_INCLUDES) + constants.STATIC_CONTEXT_SIZE

            # Create oracle and the bandit
            oracle = Oracle(configs.max_memory)
            bandits_dict[table_name] = bandits.C3UCB(context_size, configs.input_alpha, configs.input_lambda, oracle,
                                                     cluster_id)
        cluster_id += 1

        tables = []
        all_columns, number_of_columns = None, None
        if with_mv:
            # Creating bandit for MVs
            all_columns, number_of_columns = sql_helper.get_all_columns(self.connection)
            tables = sql_helper.get_tables(self.connection)
            context_size = number_of_columns + len(tables) + 4
            bandits_dict[mv] = bandits.C3UCB(context_size, configs.input_alpha, configs.input_lambda,
                                             OracleMV(configs.max_memory), cluster_id)

        # Creating super bandit
        number_of_clusters = cluster_id + 1
        oracle_s = OracleS(configs.max_memory)
        super_bandit = bandits.C3UCB(number_of_clusters + super_static_context_size, configs.input_alpha, configs.input_lambda, oracle_s)

        # Running the bandit for T rounds and gather the reward
        arm_selection_count = {}
        chosen_arms_last_round = {}
        next_workload_shift = 0
        queries_start = configs.queries_start_list[next_workload_shift]
        queries_end = configs.queries_end_list[next_workload_shift]
        query_obj_additions = []
        total_time = 0.0

        for t in range(configs.rounds):
            logging.info(f"round: {t}")
            start_time_round = datetime.datetime.now()
            # At the start of the round we will read the applicable set for the current round. This is a workaround
            # used to demo the dynamic query flow. We read the queries from the start and move the window each round

            # check if workload shift is required
            if t == configs.workload_shifts[next_workload_shift]:
                queries_start = configs.queries_start_list[next_workload_shift]
                queries_end = configs.queries_end_list[next_workload_shift]
                if len(configs.workload_shifts) > next_workload_shift + 1:
                    next_workload_shift += 1

            # New set of queries in this batch, required for query execution
            queries_current_batch = self.queries[queries_start:queries_end]

            # Adding new queries to the query store
            query_obj_list_current = []
            for n in range(len(queries_current_batch)):
                query = queries_current_batch[n]
                query_id = query['id']
                if query_id in self.query_obj_store:
                    query_obj_in_store = self.query_obj_store[query_id]
                    query_obj_in_store.frequency += 1
                    query_obj_in_store.last_seen = t
                    query_obj_in_store.query_strings.append(query['query_string'])
                    if query_obj_in_store.first_seen == -1:
                        query_obj_in_store.first_seen = t
                else:
                    query = Query(self.connection, query_id, query['query_string'], query['predicates'],
                                  query['payload'], t)
                    # query.context = bandit_helper.get_query_context_v1(query, all_columns, number_of_columns)
                    self.query_obj_store[query_id] = query
                query_obj_list_current.append(self.query_obj_store[query_id])

            # This list contains all past queries, we don't include new queries seen for the first time.
            query_obj_list_past = []
            query_obj_list_new = []
            for key, obj in self.query_obj_store.items():
                if t - obj.last_seen <= constants.QUERY_MEMORY and 0 <= obj.first_seen < t:
                    query_obj_list_past.append(obj)
                elif t - obj.last_seen > constants.QUERY_MEMORY:
                    obj.first_seen = -1
                elif obj.first_seen == t:
                    query_obj_list_new.append(obj)

            # We don't want to reset in the first round, if there is new additions or removals we identify a
            # workload change
            if t > 0 and len(query_obj_additions) > 0:
                workload_change = len(query_obj_additions) / len(query_obj_list_past)
                for table_name in table_list:
                    bandits_dict[table_name].workload_change_trigger(workload_change)
                if with_mv:
                    bandits_dict[mv].workload_change_trigger(workload_change)

            # this rounds new will be the additions for the next round
            query_obj_additions = query_obj_list_new

            # Get the predicates, frequent table subsets for queries and Generate index and view arms for each query
            index_arms = {}
            frequent_table_subsets = {}
            if with_mv:
                frequent_table_subsets = bandit_helper.gen_frq_table_subsets(self.connection, query_obj_list_past, tables,
                                                                             self.query_properties)
            for i in range(len(query_obj_list_past)):
                bandit_arms_tmp = bandit_helper.gen_arms_from_predicates_v2(self.connection, query_obj_list_past[i])
                bandit_arms_mv = {}
                if with_mv:
                    bandit_arms_mv = bandit_helper.gen_mv_arms_from_predicates_v3(self.connection, query_obj_list_past[i],
                                                                                  tables, frequent_table_subsets,
                                                                                  self.query_properties, True)
                    # bandit_arms_mv = bandit_helper.finalizing_mv_arms(self.connection, bandit_arms_mv)
                # Adding index arms
                for key, index_arm in bandit_arms_tmp.items():
                    table_name = index_arm.table_name
                    if table_name not in index_arms:
                        index_arms[table_name] = {}
                    if key not in index_arms[table_name]:
                        index_arm.query_ids = set()
                        index_arm.query_ids_backup = set()
                        index_arm.clustered_index_time = 1
                        index_arms[table_name][key] = index_arm
                    # index_arm.clustered_index_time += max(
                    #     query_obj_list_past[i].table_scan_times[index_arm.table_name]) if \
                    #     query_obj_list_past[i].table_scan_times[index_arm.table_name] else 0
                    index_arm.clustered_index_time += query_obj_list_past[i].original_running_time
                    index_arms[table_name][key].query_ids.add(index_arm.query_id)
                    index_arms[table_name][key].query_ids_backup.add(index_arm.query_id)

                if with_mv:
                    # Adding MV arms
                    for key, index_arm in bandit_arms_mv.items():
                        table_name = index_arm.table_name
                        if table_name not in index_arms:
                            index_arms[table_name] = {}
                        if key not in index_arms[table_name]:
                            index_arm.query_ids = set()
                            index_arm.query_ids_backup = set()
                            index_arm.clustered_index_time = 1
                            index_arms[table_name][key] = index_arm
                        for ta in index_arm.table_names:
                            index_arm.clustered_index_time += max(
                                query_obj_list_past[i].table_scan_times[ta]) if \
                                query_obj_list_past[i].table_scan_times[ta] else 0
                        index_arms[table_name][key].query_ids.add(index_arm.query_id)
                        index_arms[table_name][key].query_ids_backup.add(index_arm.query_id)

            # set the index arms at the bandit
            if mv in index_arms:
                bandit_helper.finalizing_mv_arms(self.connection, index_arms[mv], self.query_properties, configs.max_memory)
            chosen_arms = {}
            chosen_arm_ids = {}
            for table_name in table_list:
                index_arms_for_table = {}
                if table_name in index_arms:
                    index_arms_for_table = index_arms[table_name]

                index_arm_list = list(index_arms_for_table.values())
                logging.info(f"Generated {len(index_arm_list)} arms for table {table_name}")
                bandits_dict[table_name].set_arms(index_arm_list)

                # creating the context, here we pass all the columns in the database
                context_vectors_v1 = bandit_helper.get_name_encode_cv_v2(index_arms_for_table, columns[table_name], column_counts[table_name], constants.CONTEXT_UNIQUENESS, constants.CONTEXT_INCLUDES)
                context_vectors_v2 = bandit_helper.get_derived_value_cv_v4(self.connection, index_arms_for_table, query_obj_list_past, chosen_arms_last_round, constants.INDEX_INCLUDES)

                context_vectors = []
                for i in range(len(context_vectors_v1)):
                    context_vectors.append(
                        numpy.array(list(context_vectors_v2[i]) + list(context_vectors_v1[i]), ndmin=2))
                # getting the super arm from the bandit
                chosen_arm_ids[table_name] = bandits_dict[table_name].select_arm(context_vectors, t)

                # get objects for the chosen set of arm ids
                if chosen_arm_ids[table_name]:
                    for (arm_id, ucb) in chosen_arm_ids[table_name]:
                        index_name = index_arm_list[arm_id].index_name
                        if table_name not in chosen_arms:
                            chosen_arms[table_name] = {}
                        chosen_arms[table_name][index_name] = (index_arm_list[arm_id], arm_id, ucb)

            # setting arms for MV bandit
            if with_mv:
                index_arm_list = {}
                index_arms_for_table = {}
                if mv in index_arms:
                    index_arm_list = list(index_arms[mv].values())
                    index_arms_for_table = index_arms[mv]

                logging.info(f"Generated {len(index_arm_list)} arms")
                bandits_dict[mv].set_arms(index_arm_list)

                context_vectors = bandit_helper.get_view_encode_cv_v1(self.connection, index_arms_for_table, all_columns,
                                                                      number_of_columns, chosen_arms_last_round)
                chosen_arm_ids[mv] = bandits_dict[mv].select_arm(context_vectors, t)
                chosen_arms[mv] = {}
                if chosen_arm_ids[mv]:
                    for (arm_id, ucb) in chosen_arm_ids[mv]:
                        index_name = index_arm_list[arm_id].index_name
                        chosen_arms[mv][index_name] = (index_arm_list[arm_id], arm_id, ucb)

            super_arm_list, super_context, original_map = bandit_helper.get_super_bandit_context(self.connection, chosen_arms, chosen_arms_last_round, super_static_context_size, number_of_clusters)
            super_bandit.set_arms(super_arm_list)
            super_chosen_arm_ids, mv_size_weight, index_size_weight = super_bandit.select_super_arm_v2(super_context)
            super_chosen_per_table = {}
            for c_id in super_chosen_arm_ids:
                c_t, c_aid = original_map[c_id]
                if c_t not in super_chosen_per_table:
                    super_chosen_per_table[c_t] = []
                super_chosen_per_table[c_t].append(c_aid)

            super_chosen_arms = {}
            used_memory = 0
            if super_chosen_arm_ids:
                for arm_id in super_chosen_arm_ids:
                    index_name = super_arm_list[arm_id].index_name
                    super_chosen_arms[index_name] = super_arm_list[arm_id]
                    used_memory = used_memory + super_arm_list[arm_id].memory
                    if index_name in arm_selection_count:
                        arm_selection_count[index_name] += 1
                    else:
                        arm_selection_count[index_name] = 1

            # finding the difference between last round and this round
            keys_last_round = set(chosen_arms_last_round.keys())
            keys_this_round = set(super_chosen_arms.keys())
            key_intersection = keys_last_round & keys_this_round
            key_additions = keys_this_round - key_intersection
            key_deletions = keys_last_round - key_intersection
            logging.info(f"Selected: {keys_this_round}")
            logging.debug(f"Added: {key_additions}")
            logging.debug(f"Removed: {key_deletions}")

            added_arms = {}
            deleted_arms = {}
            for key in key_additions:
                added_arms[key] = super_chosen_arms[key]
            for key in key_deletions:
                deleted_arms[key] = chosen_arms_last_round[key]

            start_time_create_query = datetime.datetime.now()
            sql_helper.drop_v7(self.connection, constants.SCHEMA_NAME, deleted_arms)

            hyp_cost = 0
            useless = set()
            if t < hyp_check_rounds:
                hyp_query_plans, _ = sql_helper.hyp_check_config(self.connection, constants.SCHEMA_NAME,
                                                                        added_arms, query_obj_list_current, hyp_file_path)
                hyp_cost = sql_helper.get_hyp_cost(self.connection, hyp_file_path)
                start_time_hyp_reward = datetime.datetime.now()
                hyp_arm_rewards = bandit_helper.calculate_hyp_reward(query_obj_list_current, hyp_query_plans)
                end_time_hyp_reward = datetime.datetime.now()
                hyp_cost += (end_time_hyp_reward - start_time_hyp_reward).total_seconds()
                useless = set(added_arms.keys()) - set(hyp_arm_rewards.keys())
                for a_id in useless:
                    logging.info(f"Suggestion Removed {a_id}")
                    del added_arms[a_id]
                    del super_chosen_arms[a_id]

            result = sql_helper.create_query_v7(self.connection, constants.SCHEMA_NAME, added_arms, deleted_arms, query_obj_list_current)
            execution_cost, creation_costs, query_plans, cost_analytical, cost_transactional = result
            arm_rewards = bandit_helper.calculate_reward(creation_costs, query_obj_list_current, query_plans)
            if constants.LOG_XML:
                helper.log_query_xmls(configs.experiment_id, query_obj_list_current, query_plans, t, constants.COMPONENT_MAB)
            end_time_create_query = datetime.datetime.now()
            creation_cost = sum(creation_costs.values())

            super_bandit.update_super_v3(super_chosen_arm_ids, arm_rewards, useless, mv_size_weight, index_size_weight)

            for table_name in table_list:
                arm_ids = super_chosen_per_table[table_name] if (table_name in super_chosen_per_table) else []
                bandits_dict[table_name].update(arm_ids, arm_rewards, useless, mv_size_weight, index_size_weight)

            if with_mv:
                arm_ids = super_chosen_per_table[mv] if (mv in super_chosen_per_table) else []
                bandits_dict[mv].update(arm_ids, arm_rewards, useless, mv_size_weight, index_size_weight)

            # keeping track of queries that we saw last time
            chosen_arms_last_round = super_chosen_arms

            if t == (configs.rounds - 1):
                sql_helper.bulk_drop(self.connection, constants.SCHEMA_NAME, super_chosen_arms)

            end_time_round = datetime.datetime.now()
            current_config_size = float(sql_helper.get_current_pds_size(self.connection))
            logging.info("Size taken by the config: " + str(current_config_size) + "MB")
            # Adding information to the results array
            actual_round_number = t
            recommendation_time = (end_time_round - start_time_round).total_seconds() + hyp_cost - (
                        end_time_create_query - start_time_create_query).total_seconds()
            logging.info("Recommendation cost: " + str(recommendation_time) + ", Hyp Component: " + str(hyp_cost))
            total_round_time = creation_cost + execution_cost + recommendation_time
            results.append([actual_round_number, constants.MEASURE_BATCH_TIME, total_round_time])
            results.append([actual_round_number, constants.MEASURE_INDEX_CREATION_COST, creation_cost])
            results.append([actual_round_number, constants.MEASURE_QUERY_EXECUTION_COST, execution_cost])
            results.append(
                [actual_round_number, constants.MEASURE_INDEX_RECOMMENDATION_COST, recommendation_time])
            results.append([actual_round_number, constants.MEASURE_MEMORY_COST, current_config_size])
            results.append([actual_round_number, constants.MEASURE_ANALYTICAL_EXECUTION_COST, cost_analytical])
            results.append([actual_round_number, constants.MEASURE_TRANSACTIONAL_EXECUTION_COST, cost_transactional])

            total_time += total_round_time

            print(f"current total {t}: ", total_time)
        logging.info("Time taken by bandit for " + str(configs.rounds) + " rounds: " + str(total_time))
        logging.info("\n\nIndex Usage Counts:\n" + pp.pformat(
            sorted(arm_selection_count.items(), key=operator.itemgetter(1), reverse=True)))
        sql_connection.close_sql_connection(self.connection)
        sql_helper.clean_up_routine(sql_connection)
        self.connection = sql_connection.get_sql_connection()
        return results, total_time


if __name__ == "__main__":
    # Running MAB
    exp_report_mab = ExpReport(configs.experiment_id, constants.COMPONENT_MAB, configs.reps, configs.rounds)
    for r in range(configs.reps):
        simulator = Simulator()
        sim_results, total_workload_time = simulator.run()
        temp = DataFrame(sim_results, columns=[constants.DF_COL_BATCH, constants.DF_COL_MEASURE_NAME,
                                               constants.DF_COL_MEASURE_VALUE])
        temp.append([-1, constants.MEASURE_TOTAL_WORKLOAD_TIME, total_workload_time])
        temp[constants.DF_COL_REP] = r
        exp_report_mab.add_data_list(temp)

    # plot line graphs
    helper.plot_exp_report(configs.experiment_id, [exp_report_mab],
                           (constants.MEASURE_BATCH_TIME, constants.MEASURE_QUERY_EXECUTION_COST))
