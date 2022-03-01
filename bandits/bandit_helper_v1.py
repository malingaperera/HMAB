import itertools
import numpy

import constants as constants
from bandits.bandit_arm_v1 import BanditArm
from database.qplan.write import WriteQueryPlan, InsertQueryPlan, DeleteQueryPlan, UpdateQueryPlan
from bandits.bandit_arm_MV_v1 import BanditArmMV
import database.sql_helper_v3 as sql_helper

bandit_arm_store = {}
table_scan_times = sql_helper.get_table_scan_times_structure()
table_scan_times_hyp = sql_helper.get_table_scan_times_structure()

# ========================== Arm Generation ==========================


def gen_arms_from_predicates_v2(connection, query_obj):
    """
    This method take predicates (a dictionary of lists) as input and creates the generate arms for all possible
    column combinations

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """
    bandit_arms = {}
    predicates = query_obj.predicates
    payloads = query_obj.payload
    query_id = query_obj.id
    tables = sql_helper.get_tables(connection)
    for table_name, table_predicates in predicates.items():
        table = tables[table_name]
        includes = []
        if table_name in payloads:
            includes = list(set(payloads[table_name]) - set(table_predicates))
        if table.table_row_count < constants.SMALL_TABLE_IGNORE:
            continue
        col_permutations = []
        if len(table_predicates) > 6:
            table_predicates = table_predicates[0:6]
        for j in range(1, (len(table_predicates) + 1)):
            col_permutations = col_permutations + list(itertools.permutations(table_predicates, j))
        for col_permutation in col_permutations:
            arm_id = BanditArm.get_arm_id(col_permutation, table_name)
            table_row_count = table.table_row_count
            # arm_value = (1 - query_obj.selectivity[table_name]) * (
            #             len(col_permutation) / len(table_predicates)) * table_row_count
            if arm_id in bandit_arm_store:
                bandit_arm = bandit_arm_store[arm_id]
                bandit_arm.query_id = query_id
                # if query_id in bandit_arm.arm_value:
                #     bandit_arm.arm_value[query_id] += arm_value
                #     bandit_arm.arm_value[query_id] /= 2
                # else:
                #     bandit_arm.arm_value[query_id] = arm_value
            else:
                size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                 table_name, col_permutation)
                bandit_arm = BanditArm(col_permutation, table_name, size, table_row_count)
                bandit_arm.query_id = query_id
                if len(col_permutation) == len(table_predicates):
                    bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                    if len(includes) == 0:
                        bandit_arm.is_include = 1
                # bandit_arm.arm_value[query_id] = arm_value
                bandit_arm_store[arm_id] = bandit_arm
            if bandit_arm not in bandit_arms:
                bandit_arms[arm_id] = bandit_arm

    for table_name, table_payloads in payloads.items():
        if table_name not in predicates:
            table = tables[table_name]
            if table.table_row_count < constants.SMALL_TABLE_IGNORE:
                continue
            col_permutation = table_payloads
            arm_id = BanditArm.get_arm_id(col_permutation, table_name, no_include=True)
            table_row_count = table.table_row_count
            # arm_value = 0.001 * table_row_count
            if arm_id in bandit_arm_store:
                bandit_arm = bandit_arm_store[arm_id]
                bandit_arm.query_id = query_id
                # if query_id in bandit_arm.arm_value:
                #     bandit_arm.arm_value[query_id] += arm_value
                #     bandit_arm.arm_value[query_id] /= 2
                # else:
                #     bandit_arm.arm_value[query_id] = arm_value
            else:
                size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                 table_name, col_permutation)
                bandit_arm = BanditArm(col_permutation, table_name, size, table_row_count)
                bandit_arm.query_id = query_id
                bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                bandit_arm.is_include = 1
                # bandit_arm.arm_value[query_id] = arm_value
                bandit_arm_store[arm_id] = bandit_arm
            if bandit_arm not in bandit_arms:
                bandit_arms[arm_id] = bandit_arm

    if constants.INDEX_INCLUDES:
        for table_name, table_predicates in predicates.items():
            table = tables[table_name]
            if table.table_row_count < constants.SMALL_TABLE_IGNORE:
                continue
            includes = []
            if table_name in payloads:
                includes = sorted(list(set(payloads[table_name]) - set(table_predicates)))
            if includes:
                col_permutations = list(itertools.permutations(table_predicates, len(table_predicates)))
                for col_permutation in col_permutations:
                    arm_id_with_include = BanditArm.get_arm_id(col_permutation, table_name, includes)
                    table_row_count = table.table_row_count
                    # arm_value = (1 - query_obj.selectivity[table_name]) * table_row_count
                    if arm_id_with_include not in bandit_arm_store:
                        size_with_includes = sql_helper.get_estimated_size_of_index_v1(connection,
                                                                                       constants.SCHEMA_NAME,
                                                                                       table_name,
                                                                                       col_permutation + tuple(
                                                                                           includes))
                        bandit_arm = BanditArm(col_permutation, table_name, size_with_includes, table_row_count,
                                               includes)
                        bandit_arm.is_include = 1
                        bandit_arm.query_id = query_id
                        bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                        # bandit_arm.arm_value[query_id] = arm_value
                        bandit_arm_store[arm_id_with_include] = bandit_arm
                    else:
                        bandit_arm_store[arm_id_with_include].query_id = query_id
                        # if query_id in bandit_arm_store[arm_id_with_include].arm_value:
                        #     bandit_arm_store[arm_id_with_include].arm_value[query_id] += arm_value
                        #     bandit_arm_store[arm_id_with_include].arm_value[query_id] /= 2
                        # else:
                        #     bandit_arm_store[arm_id_with_include].arm_value[query_id] = arm_value
                    bandit_arms[arm_id_with_include] = bandit_arm_store[arm_id_with_include]
    return bandit_arms


def gen_frq_table_subsets(connection, query_objs, all_tables, query_properties):
    frq_table_subsets = {}
    total_workload_time = 0
    min_rows = sum([x.table_row_count for x in all_tables.values()]) * 0.1
    for query_obj in query_objs:
        total_workload_time += query_obj.original_running_time
        table_set = query_properties['tables'][query_obj.id]
        for i in range(2, len(table_set)+1):
            table_subsets = list(itertools.combinations(table_set, i))
            for table_subset in table_subsets:
                if can_be_joined(table_subset, query_properties['joins'][query_obj.id]):
                    if table_subset in frq_table_subsets:
                        frq_table_subsets[table_subset] += query_obj.original_running_time
                    else:
                        frq_table_subsets[table_subset] = query_obj.original_running_time
    subset_list = list(frq_table_subsets.keys())
    for table_subset in subset_list:
        if frq_table_subsets[table_subset] < total_workload_time * 0.05:
            del frq_table_subsets[table_subset]
        elif sum(map(lambda x: all_tables[x].table_row_count, table_subset)) < min_rows:
            del frq_table_subsets[table_subset]

    frq_table_subsets = dict(sorted(frq_table_subsets.items(), key=lambda item: item[1], reverse=True))
    return frq_table_subsets


def can_be_joined(table_subset, query_joins):
    tables_can_be_joined = set()
    join_count = 0
    for key in query_joins:
        if key[0] in table_subset and key[1] in table_subset:
            join_count += 1
            tables_can_be_joined.add(key[0])
            tables_can_be_joined.add(key[1])
    if set(table_subset) == tables_can_be_joined and join_count >= len(table_subset) - 1:
        return True
    else:
        return False


def gen_mv_arms_from_predicates_v3(connection, query_obj, tables, frequent_table_subsets, query_properties, include_group_by):
    """
    This method take predicates (a dictionary of lists) as input and creates the generate arms for all possible
    column combinations

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """

    bandit_arms = {}
    query_id = query_obj.id
    group_by_possible = include_group_by and query_properties['gb_payload'][query_id]

    for table_subset in frequent_table_subsets:
        if set(table_subset).issubset(set(query_properties['tables'][query_id])) and \
                can_be_joined(table_subset, query_properties['joins'][query_id]):
            group_bys = {False}
            if group_by_possible and set(table_subset) == set(query_properties['tables'][query_id]):
                group_bys.add('True')
            for group_by in group_bys:
                arm_id = BanditArmMV.get_arm_id(query_id, table_subset, group_by)
                if arm_id in bandit_arm_store:
                    bandit_arm = bandit_arm_store[arm_id]
                    bandit_arm.query_ids.add(query_id)
                    if arm_id not in bandit_arms:
                        bandit_arms[arm_id] = bandit_arm
                else:
                    arm_payload = {}
                    payload_list = []
                    indexed_columns_equal = []
                    indexed_columns_range = []
                    indexed_columns_pk = []
                    group_by_columns_renamed = []
                    group_by_columns_original = []
                    columns_pk = []
                    for table_name in table_subset:
                        if table_name in query_properties['payload'][query_id]:
                            arm_payload[table_name] = query_properties['payload'][query_id][table_name]
                            for payload_column in query_properties['payload'][query_id][table_name]:
                                if payload_column in tables[table_subset[0]].pk_columns:
                                    columns_pk.append(payload_column)
                                if group_by:
                                    if query_properties['payload'][query_id][table_name][payload_column] != 'PL':
                                        payload_list.append(f"\t{table_name}.{payload_column} as {table_name}_{payload_column}")
                                        group_by_columns_renamed.append(f"{table_name}_{payload_column}")
                                        group_by_columns_original.append(f"{table_name}.{payload_column}")
                                else:
                                    payload_list.append(f"\t{table_name}.{payload_column} as {table_name}_{payload_column}")
                                if query_properties['payload'][query_id][table_name][payload_column] == 'EQ':
                                    indexed_columns_equal.append(f"{table_name}_{payload_column}")
                                elif query_properties['payload'][query_id][table_name][payload_column] in {'GT', 'LT', 'GE', 'LE'}:
                                    indexed_columns_range.append(f"{table_name}_{payload_column}")
                                else:
                                    if payload_column in tables[table_subset[0]].pk_columns:
                                        indexed_columns_pk.append(f"{table_name}_{payload_column}")

                    if set(columns_pk) != set(tables[table_subset[0]].pk_columns) and not group_by:
                        for payload_column in set(tables[table_subset[0]].pk_columns).difference(set(columns_pk)):
                            payload_list.append(f"\t{table_subset[0]}.{payload_column} as {table_subset[0]}_{payload_column}")
                            indexed_columns_pk.append(f"{table_subset[0]}_{payload_column}")
                    if group_by:
                        cn = 1
                        for table_name in table_subset:
                            if table_name in query_properties['gb_payload'][query_id]:
                                for payload_column in query_properties['gb_payload'][query_id][table_name]:
                                    payload_list.append(f"\t{payload_column} as {table_name}_exp{cn}")
                                    cn += 1
                    if payload_list:
                        if group_by:
                            payload_list.append("\tcount_big(*) as big_count_col")
                        join_list = []
                        arm_joins = {}
                        for (table1, table2), join_columns in query_properties['joins'][query_id].items():
                            if table1 in table_subset and table2 in table_subset:
                                arm_joins[(table1, table2)] = join_columns
                                for (column1, column2) in join_columns:
                                    join_list.append(f"{table1}.{column1} = {table2}.{column2}")

                        bandit_arm = BanditArmMV(query_id, arm_payload, arm_joins, table_subset,
                                                 group_by_columns_original)

                        bandit_arm.set_view_query_components(payload_list, table_subset, join_list, group_by_columns_original)
                        bandit_arm.set_index_query_components(indexed_columns_equal, indexed_columns_range, indexed_columns_pk, group_by_columns_renamed)

                        bandit_arm_store[arm_id] = bandit_arm
                        if arm_id not in bandit_arms:
                            bandit_arms[arm_id] = bandit_arm

    return bandit_arms


def finalizing_mv_arms(connection, bandit_arms_mv, query_properties, max_memory):
    ids_to_delete = []
    connection.timeout = 60
    for arm_id, arm in bandit_arms_mv.items():
        if arm.memory == -1:
            ids_to_delete.append(arm_id)
        elif arm.memory is None:
            view_query, count_query = get_mv_arm_view_query(arm.index_name, arm.view_query_comps)
            index_query = get_mv_arm_index_query(arm.index_name, arm.index_query_comps)
            dim_tables = query_properties['dim_tables']
            count_query_id = tuple(sorted(set(arm.table_names).difference(dim_tables)))
            size = sql_helper.get_estimated_size_of_mv_v2(connection, arm.payload,
                                                          view_query[view_query.lower().index("select"):],
                                                          count_query, count_query_id, arm.group_by)
            arm.view_query = view_query
            arm.index_query = index_query
            arm.memory = size
            if size <= 0 or size > 5000:
                ids_to_delete.append(arm_id)
    for arm_id in ids_to_delete:
        del bandit_arms_mv[arm_id]
    connection.timeout = 0


def get_mv_arm_view_query(arm_id, view_comps):
    view_query = ""
    view_query += f"CREATE VIEW dbo.{arm_id} WITH SCHEMABINDING AS \n"
    view_query += "SELECT\n"
    view_query += ",\n".join(view_comps["payload_list"])
    view_query += "\n"
    view_query_from = ""
    view_query_from += "FROM " + ", ".join(['dbo.' + s for s in view_comps["table_subset"]]) + "\n"
    view_query_from += "WHERE "
    view_query_from += "\n AND ".join(view_comps["join_list"])
    if view_comps["group_by_columns_original"]:
        view_query_from += "\nGROUP BY " + ", ".join(view_comps["group_by_columns_original"])

    count_query = "SELECT COUNT_BIG(1) as C1 \n" + view_query_from
    if view_comps["group_by_columns_original"]:
        count_query = "SELECT COUNT_BIG(*) FROM ( " + count_query + ") d1"
    count_query += ';'
    view_query += view_query_from + ';'

    return view_query, count_query


def get_mv_arm_index_query(arm_id, index_comps):
    index_query = ""
    index_query += f"CREATE UNIQUE CLUSTERED INDEX {arm_id}_ci ON dbo.{arm_id}\n(\n"
    indexed_columns_list = []
    if not index_comps["group_by_columns_renamed"]:
        if index_comps["indexed_columns_equal"]:
            indexed_columns_list.append(",\n".join(index_comps["indexed_columns_equal"]))
        if index_comps["indexed_columns_range"]:
            indexed_columns_list.append(",\n".join(index_comps["indexed_columns_range"]))
        if index_comps["indexed_columns_pk"]:
            indexed_columns_list.append(",\n".join(index_comps["indexed_columns_pk"]))
    else:
        indexed_columns_list = index_comps["group_by_columns_renamed"]

    index_query += ",\n".join(indexed_columns_list)
    index_query += ');'
    return index_query


# ========================== Context Vectors ==========================

def get_predicate_position(arm, predicate, table_name):
    """
    Returns float between 0 and 1  if a arm includes a predicate for the the correct table

    :param arm: bandit arm
    :param predicate: given predicate
    :param table_name: table name
    :return: float [0, 1]
    """
    for i in range(len(arm.index_cols)):
        if table_name == arm.table_name and predicate == arm.index_cols[i]:
            return i
    return -1


def get_context_vector_v2(bandit_arm, all_columns, context_size, uniqueness=0, includes=False):
    """
    Return the context vector for a given arm, and set of predicates. Size of the context vector will depend on
    the arm and the set of predicates (for now on predicates)

    :param bandit_arm: bandit arm
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: a context vector
    """
    context_vectors = {}
    for j in range(uniqueness):
        context_vectors[j] = numpy.zeros((context_size, 1), dtype=float)
    left_over_context = numpy.zeros((context_size, 1), dtype=float)
    include_context = numpy.zeros((context_size, 1), dtype=float)

    if len(bandit_arm.name_encoded_context) > 0:
        context_vector = bandit_arm.name_encoded_context
    else:
        i = 0
        for table_name in all_columns:
            for k in range(len(all_columns[table_name])):
                column_position_in_arm = get_predicate_position(bandit_arm, all_columns[table_name][k], table_name)
                if column_position_in_arm >= 0:
                    if column_position_in_arm < uniqueness:
                        context_vectors[column_position_in_arm][i] = 1
                    else:
                        left_over_context[i] = 1 / (10 ** column_position_in_arm)
                elif all_columns[table_name][k] in bandit_arm.include_cols:
                    include_context[i] = 1
                i += 1

        full_list = []
        for j in range(uniqueness):
            full_list = full_list + list(context_vectors[j])
        full_list = full_list + list(left_over_context)
        if includes:
            full_list = full_list + list(include_context)
        context_vector = numpy.array(full_list, ndmin=2, dtype=float)
        bandit_arm.name_encoded_context = context_vector
    return context_vector


def get_context_vector_mv_v1(connection, bandit_arm, all_columns, number_of_columns, chosen_arms_last_round):
    """
    Return the context vector for a given arm, and set of predicates. Size of the context vector will depend on
    the arm and the set of predicates (for now on predicates)

    :param connection: SQL connection
    :param bandit_arm: bandit arm
    :param all_columns: predicate dict(list)
    :param number_of_columns: number of columns in the database
    :param chosen_arms_last_round: Already created arms
    :return: a context vector
    """

    if len(bandit_arm.name_encoded_context) > 0:
        context_vector = bandit_arm.name_encoded_context
    else:
        table_names = list(all_columns.keys())
        table_names.sort()
        table_context = numpy.zeros((len(table_names), 1), dtype=float)
        for i, table_name in enumerate(table_names):
            if table_name in bandit_arm.table_names:
                table_context[i] = 1

        i = 0
        column_context = numpy.zeros((number_of_columns, 1), dtype=float)
        for table_name in all_columns:
            for k in range(len(all_columns[table_name])):
                if table_name in bandit_arm.payload and all_columns[table_name][k] in bandit_arm.payload[table_name]:
                    column_context[i] = 1
                i += 1

        database_size = sql_helper.get_database_size(connection)
        keys_last_round = set(chosen_arms_last_round.keys())
        if bandit_arm.index_name not in keys_last_round:
            index_size = bandit_arm.memory
        else:
            index_size = 0

        additional_context = numpy.zeros((4, 1), dtype=float)
        additional_context[1] = index_size / database_size
        if bandit_arm.group_by:
            additional_context[2] = 1
        if bandit_arm.filter_by:
            additional_context[3] = 1
        full_list = list(additional_context) + list(table_context) + list(column_context)

        context_vector = numpy.array(full_list, ndmin=2, dtype=float)
        bandit_arm.name_encoded_context = context_vector
    return context_vector


def get_super_arm_context_v1(connection, arm, ucb, chosen_arms_last_round, static_context_size, number_of_clusters):
    database_size = sql_helper.get_database_size(connection)
    context = numpy.zeros((static_context_size + number_of_clusters, 1), dtype=float)
    keys_last_round = set(chosen_arms_last_round.keys())
    if arm.index_name not in keys_last_round:
        index_size = arm.memory
    else:
        index_size = 0
    context[static_context_size + arm.bandit_cluster] = ucb

    if type(arm).__name__ == 'BanditArmMV':
        context[0] = index_size / database_size
    else:
        context[1] = index_size / database_size
    return context


def get_name_encode_cv_v2(bandit_arm_dict, all_columns, context_size, uniqueness=0, includes=False):
    """
    Return the context vectors for a given arms, and set of predicates.

    :param bandit_arm_dict: bandit arms
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: list of context vectors
    """
    context_vectors = []
    for key, bandit_arm in bandit_arm_dict.items():
        context_vector = get_context_vector_v2(bandit_arm, all_columns, context_size, uniqueness, includes)
        context_vectors.append(context_vector)

    return context_vectors


def get_derived_value_cv_v4(connection, bandit_arm_dict, query_obj_list, chosen_arms_last_round,
                            with_includes):
    """
    Similar to the v2, but it don't have the is_include part

    :param connection: SQL connection
    :param bandit_arm_dict: bandit arms
    :param query_obj_list: list of queries
    :param chosen_arms_last_round: Already created arms
    :param with_includes: have is include feature, note if includes are added to encode part we don't need it here.
    :return: list of context vectors
    """
    context_vectors = []
    high_reward_value = 2
    high_reward_threshold = 100     # depends on DB size
    database_size = sql_helper.get_database_size(connection)
    for key, bandit_arm in bandit_arm_dict.items():
        keys_last_round = set(chosen_arms_last_round.keys())
        is_high_reward_arm = high_reward_value if (bandit_arm.clustered_index_time > high_reward_threshold and bandit_arm.is_include) else 0
        if bandit_arm.index_name not in keys_last_round:
            index_size = bandit_arm.memory
        else:
            index_size = 0
        context_vector = numpy.array([
            is_high_reward_arm,
            index_size/database_size,
            bandit_arm.is_include if with_includes else 0
        ], ndmin=2).transpose()
        context_vectors.append(context_vector)

    return context_vectors


def get_view_encode_cv_v1(connection, bandit_arm_dict, all_columns, number_of_columns, chosen_arms_last_round):
    """
    Return the context vectors for a given view.

    :param connection: SQL connection
    :param bandit_arm_dict: bandit arms
    :param all_columns: predicate dict(list)
    :param number_of_columns: number of columns in the database
    :param chosen_arms_last_round: Already created arms
    :return: list of context vectors
    """
    context_vectors = []
    for key, bandit_arm in bandit_arm_dict.items():
        context_vector = get_context_vector_mv_v1(connection, bandit_arm, all_columns, number_of_columns,
                                                  chosen_arms_last_round)
        context_vectors.append(context_vector)

    return context_vectors


def get_super_bandit_context(connection, chosen_arms, chosen_arms_last_round, static_context_size, number_of_clusters):
    original_map = []
    arm_list = []
    context_list = []
    for table, super_arm in chosen_arms.items():
        for key, (arm, arm_id, ucb) in super_arm.items():
            if arm.memory > 0:
                arm_list.append(arm)
                context_list.append(get_super_arm_context_v1(connection, arm, ucb, chosen_arms_last_round, static_context_size, number_of_clusters))
                original_map.append((table, arm_id))
    return arm_list, context_list, original_map

# ========================== Reward Calculation ==========================


class Reward:
    index_name = ""
    execution = 0
    maintenance = 0
    creation = 0
    offset = 0
    queries = None


def calculate_reward(creation_cost, queries, query_plans):
    arm_rewards = {}
    for i in range(len(queries)):
        # Get the reward from each query for used indices, query_plan can be None, when query failed
        if query_plans[i]:
            query_rewards = get_query_rewards(queries[i], query_plans[i])

            # Add them to full set
            for index_name, reward in query_rewards.items():
                if index_name not in arm_rewards:
                    arm_rewards[index_name] = Reward()
                    arm_rewards[index_name].execution = reward.execution
                    arm_rewards[index_name].maintenance = reward.maintenance
                    arm_rewards[index_name].offset = reward.offset
                    arm_rewards[index_name].queries = {queries[i].id, }
                else:
                    arm_rewards[index_name].execution += reward.execution
                    arm_rewards[index_name].maintenance += reward.maintenance
                    arm_rewards[index_name].offset += reward.offset
                    arm_rewards[index_name].queries.add(queries[i].id)

    for index_name in creation_cost:
        if index_name in arm_rewards:
            arm_rewards[index_name].creation += -1 * creation_cost[index_name]
        else:
            arm_rewards[index_name] = Reward()
            arm_rewards[index_name].creation = -1 * creation_cost[index_name]
    return arm_rewards


def get_query_rewards(query, query_plan):
    """
    Return the reward for a query

    :param query: query object
    :param query_plan: query plan object
    :return:
    """
    rewards = {}
    non_clustered_index_usages = query_plan.non_clustered_index_usages
    clustered_index_usages = query_plan.clustered_index_usages
    clustered_view_usages = query_plan.clustered_view_usages
    non_clustered_view_usages = query_plan.non_clustered_view_usages
    look_ups = {}
    if clustered_index_usages:
        # Registers the clustered index scan times, and return the look-ips
        look_ups = process_clustered_index_scans(clustered_index_usages, query)

    if non_clustered_index_usages:
        # Again there might be more than one scan of the same index in same query,
        # so most probably one is catered from cache. So we calculate the max reward that can be given for a table
        # if there are more than one index on that table, those indices have to share the reward
        total_cost_tables = {}
        reward_table_count = {}
        for node_id, nc_index_use in non_clustered_index_usages.items():
            table_name = nc_index_use.indices[0].table
            scan_cost = nc_index_use[constants.COST_TYPE_CURRENT_EXECUTION]
            if table_name in total_cost_tables:
                total_cost_tables[table_name] += scan_cost
                reward_table_count[table_name] += 1
            else:
                total_cost_tables[table_name] = scan_cost
                reward_table_count[table_name] = 1

        for node_id, nc_index_use in non_clustered_index_usages.items():
            # nc_index_use = non_clustered_index_usages[node_id]
            index_name = nc_index_use.indices[0].index
            table_name = nc_index_use.indices[0].table
            total_cost_table = total_cost_tables[table_name]

            # we register index scan times for a table and query, this can be used if we didn't see any table scans
            if len(query.index_scan_times[table_name]) < constants.TABLE_SCAN_TIME_LENGTH:
                query.index_scan_times[table_name].append(total_cost_table)

            est_table_scan_time = get_est_table_scan_time(query, table_name)

            reward = (est_table_scan_time - total_cost_table)/reward_table_count[table_name]

            # reduce the cost on the look-up
            if node_id in look_ups:
                reward -= look_ups[node_id]

            if index_name not in rewards:
                rewards[index_name] = Reward()
                rewards[index_name].index_name = index_name
                rewards[index_name].execution = reward
            else:
                rewards[index_name].execution += reward

    # Calculate reward for index updates
    index_writes = get_index_writes(query_plan)
    if index_writes:
        for index_write in index_writes:
            index_count = len(index_write.indices)
            reward = -1 * index_write[constants.COST_TYPE_CURRENT_EXECUTION] / index_count
            for index in index_write.indices:
                index_name = index.index
                if index_name not in rewards:
                    rewards[index_name] = Reward()
                    rewards[index_name].index_name = index_name
                    rewards[index_name].maintenance = reward
                else:
                    rewards[index_name].maintenance += reward

    # Calculate the rewards for clustered index views
    if clustered_view_usages:
        for node_id, c_view_use in clustered_view_usages.items():
            est_table_scan_time = 0
            for table_name in c_view_use.view_tables:
                est_table_scan_time += get_est_table_scan_time(query, table_name)
            view_name = c_view_use.indices[0].table
            if view_name not in rewards:
                rewards[view_name] = Reward()
                rewards[view_name].index_name = view_name
                rewards[view_name].execution = est_table_scan_time - c_view_use[constants.COST_TYPE_CURRENT_EXECUTION]
            else:
                rewards[view_name].execution += est_table_scan_time - c_view_use[constants.COST_TYPE_CURRENT_EXECUTION]

    # Calculate the rewards for non-clustered index views
    if non_clustered_view_usages:
        for node_id, nc_view_use in non_clustered_view_usages.items():
            est_table_scan_time = 0
            for table_name in nc_view_use.view_tables:
                est_table_scan_time += get_est_table_scan_time(query, table_name)
            view_name = nc_view_use.indices[0].table
            if view_name not in rewards:
                rewards[view_name] = Reward()
                rewards[view_name].index_name = view_name
                rewards[view_name].execution = est_table_scan_time - nc_view_use[constants.COST_TYPE_CURRENT_EXECUTION]
            else:
                rewards[view_name].execution += est_table_scan_time - nc_view_use[constants.COST_TYPE_CURRENT_EXECUTION]

    # compute the unclaimed rewards
    if len(rewards.keys()) > 0:
        current_running_time = query_plan[constants.COST_TYPE_CURRENT_EXECUTION]
        original_running_time = query.original_running_time
        claimed_reward = 0
        for index_name, reward in rewards.items():
            claimed_reward += reward.execution + reward.maintenance
        unclaimed_reward = original_running_time - current_running_time - claimed_reward
        unclaimed_reward = unclaimed_reward / len(rewards.keys())
        for index_name, reward in rewards.items():
            reward.offset = unclaimed_reward

    return rewards


def calculate_hyp_reward(queries, query_plans):
    arm_rewards = {}
    for i in range(len(queries)):
        # Get the reward from each query for used indices, query_plan can be None, when query failed
        if query_plans[i]:
            query_rewards = get_hyp_query_rewards(queries[i], query_plans[i])

            # Add them to full set
            for index_name, reward in query_rewards.items():
                if index_name not in arm_rewards:
                    arm_rewards[index_name] = Reward()
                    arm_rewards[index_name].execution = reward.execution
                    arm_rewards[index_name].maintenance = reward.maintenance
                    arm_rewards[index_name].offset = reward.offset
                    arm_rewards[index_name].queries = {queries[i].id, }
                else:
                    arm_rewards[index_name].execution += reward.execution
                    arm_rewards[index_name].maintenance += reward.maintenance
                    arm_rewards[index_name].offset += reward.offset
                    arm_rewards[index_name].queries.add(queries[i].id)

    return arm_rewards


def get_hyp_query_rewards(query, query_plan):
    """
    Return the reward for a query

    :param query: query object
    :param query_plan: query plan object
    :return:
    """
    rewards = {}
    non_clustered_index_usages = query_plan.non_clustered_index_usages
    clustered_index_usages = query_plan.clustered_index_usages
    clustered_view_usages = query_plan.clustered_view_usages
    non_clustered_view_usages = query_plan.non_clustered_view_usages
    look_ups = {}
    if clustered_index_usages:
        # Registers the clustered index scan times, and return the look-ips
        look_ups = process_hyp_clustered_index_scans(clustered_index_usages, query)

    if non_clustered_index_usages:
        # Again there might be more than one scan of the same index in same query,
        # so most probably one is catered from cache. So we calculate the max reward that can be given for a table
        # if there are more than one index on that table, those indices have to share the reward
        total_cost_tables = {}
        reward_table_count = {}
        for node_id, nc_index_use in non_clustered_index_usages.items():
            table_name = nc_index_use.indices[0].table
            scan_cost = nc_index_use[constants.COST_TYPE_SUB_TREE_COST]
            if table_name in total_cost_tables:
                total_cost_tables[table_name] += scan_cost
                reward_table_count[table_name] += 1
            else:
                total_cost_tables[table_name] = scan_cost
                reward_table_count[table_name] = 1

        for node_id, nc_index_use in non_clustered_index_usages.items():
            # nc_index_use = non_clustered_index_usages[node_id]
            index_name = nc_index_use.indices[0].index
            table_name = nc_index_use.indices[0].table
            total_cost_table = total_cost_tables[table_name]

            # we register index scan times for a table and query, this can be used if we didn't see any table scans
            if len(query.index_scan_times_hyp[table_name]) < constants.TABLE_SCAN_TIME_LENGTH:
                query.index_scan_times_hyp[table_name].append(total_cost_table)

            est_table_scan_time = get_est_hyp_table_scan_time(query, table_name)

            reward = (est_table_scan_time - total_cost_table)/reward_table_count[table_name]

            # reduce the cost on the look-up
            if node_id in look_ups:
                reward -= look_ups[node_id]

            if index_name not in rewards:
                rewards[index_name] = Reward()
                rewards[index_name].index_name = index_name
                rewards[index_name].execution = reward
            else:
                rewards[index_name].execution += reward

    # Calculate reward for index updates
    index_writes = get_index_writes(query_plan)
    if index_writes:
        for index_write in index_writes:
            index_count = len(index_write.indices)
            reward = -1 * index_write[constants.COST_TYPE_SUB_TREE_COST] / index_count
            for index in index_write.indices:
                index_name = index.index
                if index_name not in rewards:
                    rewards[index_name] = Reward()
                    rewards[index_name].index_name = index_name
                    rewards[index_name].maintenance = reward
                else:
                    rewards[index_name].maintenance += reward

    # Calculate the rewards for clustered index views
    if clustered_view_usages:
        for node_id, c_view_use in clustered_view_usages.items():
            est_table_scan_time = 0
            for table_name in c_view_use.view_tables:
                est_table_scan_time += get_est_hyp_table_scan_time(query, table_name)
            view_name = c_view_use.indices[0].table
            if view_name not in rewards:
                rewards[view_name] = Reward()
                rewards[view_name].index_name = view_name
                rewards[view_name].execution = est_table_scan_time - c_view_use[constants.COST_TYPE_SUB_TREE_COST]
            else:
                rewards[view_name].execution += est_table_scan_time - c_view_use[constants.COST_TYPE_SUB_TREE_COST]

    # Calculate the rewards for non-clustered index views
    if non_clustered_view_usages:
        for node_id, nc_view_use in non_clustered_view_usages.items():
            est_table_scan_time = 0
            for table_name in nc_view_use.view_tables:
                est_table_scan_time += get_est_hyp_table_scan_time(query, table_name)
            view_name = nc_view_use.indices[0].table
            if view_name not in rewards:
                rewards[view_name] = Reward()
                rewards[view_name].index_name = view_name
                rewards[view_name].execution = est_table_scan_time - nc_view_use[constants.COST_TYPE_SUB_TREE_COST]
            else:
                rewards[view_name].execution += est_table_scan_time - nc_view_use[constants.COST_TYPE_SUB_TREE_COST]

    # compute the unclaimed rewards
    if len(rewards.keys()) > 0:
        current_running_time = query_plan[constants.COST_TYPE_SUB_TREE_COST]
        original_hyp_running_time = query.original_hyp_running_time
        claimed_reward = 0
        for index_name, reward in rewards.items():
            claimed_reward += reward.execution + reward.maintenance
        unclaimed_reward = original_hyp_running_time - current_running_time - claimed_reward
        unclaimed_reward = unclaimed_reward / len(rewards.keys())
        for index_name, reward in rewards.items():
            reward.offset = unclaimed_reward

    return rewards


def get_index_writes(query_plan):
    """
    Return the index updates based on the query plan type

    :param query_plan: Query plan object
    :return: index update operator or None
    """
    if isinstance(query_plan, WriteQueryPlan):
        if isinstance(query_plan, InsertQueryPlan):
            return query_plan.index_inserts
        elif isinstance(query_plan, DeleteQueryPlan):
            return query_plan.index_deletes
        elif isinstance(query_plan, UpdateQueryPlan):
            return query_plan.index_updates
    return None


def process_clustered_index_scans(clustered_index_usages, query):
    """
    Registers the clustered index scan times in the query as well as in global array, return the lookups

    :param clustered_index_usages: clustered index scans in a query
    :param query: query object
    :return: look-up dict
    """
    max_table_scan = {}
    look_ups = {}
    for node_id, c_index_use in clustered_index_usages.items():
        # if there is more than one scan for a table, ones after first is possibly given from cache.
        # So we take the the um of access times for a table
        table_name = c_index_use.indices[0].table
        scan_cost = c_index_use[constants.COST_TYPE_CURRENT_EXECUTION]
        if table_name not in max_table_scan:
            max_table_scan[table_name] = scan_cost
        else:
            max_table_scan[table_name] += scan_cost

        if c_index_use.is_lookup and c_index_use.lookup_parent:
            # if this is a look-up, then cost should go to the index which triggered the look-up.
            # There can be 2+ look-ups with different costs for same index (some from cache),
            # still we reduce same amount from the non-clustered index reward which triggered the look-up.
            parent_node_id = c_index_use.lookup_parent
            if parent_node_id in look_ups:
                look_ups[parent_node_id] += scan_cost
            else:
                look_ups[parent_node_id] = scan_cost

    for table_name, scan_time in max_table_scan.items():
        # We add the max time to the arrays
        if len(query.table_scan_times[table_name]) < constants.TABLE_SCAN_TIME_LENGTH:
            query.table_scan_times[table_name].append(scan_time)
            table_scan_times[table_name].append(scan_time)

    return look_ups


def process_hyp_clustered_index_scans(clustered_index_usages, query):
    """
    Registers the clustered index scan times in the query as well as in global array, return the lookups

    :param clustered_index_usages: clustered index scans in a query
    :param query: query object
    :return: look-up dict
    """
    max_table_scan = {}
    look_ups = {}
    for node_id, c_index_use in clustered_index_usages.items():
        # if there is more than one scan for a table, ones after first is possibly given from cache.
        # So we take the the um of access times for a table
        table_name = c_index_use.indices[0].table
        scan_cost = c_index_use[constants.COST_TYPE_SUB_TREE_COST]
        if table_name not in max_table_scan:
            max_table_scan[table_name] = scan_cost
        else:
            max_table_scan[table_name] += scan_cost

        if c_index_use.is_lookup and c_index_use.lookup_parent:
            # if this is a look-up, then cost should go to the index which triggered the look-up.
            # There can be 2+ look-ups with different costs for same index (some from cache),
            # still we reduce same amount from the non-clustered index reward which triggered the look-up.
            parent_node_id = c_index_use.lookup_parent
            if parent_node_id in look_ups:
                look_ups[parent_node_id] += scan_cost
            else:
                look_ups[parent_node_id] = scan_cost

    for table_name, scan_time in max_table_scan.items():
        # We add the max time to the arrays
        if len(query.table_scan_times_hyp[table_name]) < constants.TABLE_SCAN_TIME_LENGTH:
            query.table_scan_times_hyp[table_name].append(scan_time)
            table_scan_times_hyp[table_name].append(scan_time)

    return look_ups


def get_est_table_scan_time(query, table_name):
    """
    We estimate the table scan time from the clustered index scan if available
    If not we estimate it by the max index scan time

    :param query: query object
    :param table_name: String, name of the table
    :return: Float, estimated table scan time
    """
    q_table_scan_times = query.table_scan_times[table_name]
    q_index_scan_times = query.index_scan_times[table_name]
    if len(q_table_scan_times) > 0:
        # We can use the average here as well.
        return max(q_table_scan_times)
    else:
        return max(q_index_scan_times)


def get_est_hyp_table_scan_time(query, table_name):
    """
    We estimate the table scan time from the clustered index scan if available
    If not we estimate it by the max index scan time

    :param query: query object
    :param table_name: String, name of the table
    :return: Float, estimated table scan time
    """
    q_table_scan_times = query.table_scan_times_hyp[table_name]
    q_index_scan_times = query.index_scan_times_hyp[table_name]
    if len(q_table_scan_times) > 0:
        # We can use the average here as well.
        return max(q_table_scan_times)
    else:
        return max(q_index_scan_times)
