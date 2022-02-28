import configparser
import copy
import datetime
import logging
import os
import subprocess
import time
from collections import defaultdict

import constants
from database.column import Column
from database.qplan.query_plan import QueryPlan
from database.table import Table

db_config = configparser.ConfigParser()
db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
db_type = db_config['SYSTEM']['db_type']
database = db_config[db_type]['database']

table_scan_times_hyp = copy.deepcopy(constants.TABLE_SCAN_TIMES[database[:-4]])
table_scan_times = copy.deepcopy(constants.TABLE_SCAN_TIMES[database[:-4]])

tables_global = None
pk_columns_dict = {}
count_numbers = {}
cache_hits = 0


# ############################# TA functions #############################


def create_index_v2(connection, query):
    """
    Create an index on the given table

    :param connection: sql_connection
    :param query: query for index creation
    """
    cursor = connection.cursor()
    cursor.execute("SET STATISTICS XML ON")
    cursor.execute(query)
    stat_xml = cursor.fetchone()[0]
    cursor.execute("SET STATISTICS XML OFF")
    connection.commit()

    # Return the current reward
    query_plan = QueryPlan.get_plan(stat_xml)
    return query_plan[constants.COST_TYPE_CURRENT_EXECUTION]


def create_view_v1(connection, query):
    """
    Create an index on the given table

    :param connection: sql_connection
    :param query: query for index creation
    """
    cursor = connection.cursor()
    cursor.execute("SET STATISTICS XML ON")
    cursor.execute(query)
    cursor.execute("SET STATISTICS XML OFF")
    connection.commit()

    return 0


def simple_execute(connection, query):
    """
    Drops the index on the given table with given name

    :param connection: sql_connection
    :param query: query to execute
    :return:
    """
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    logging.debug(query)


# ############################# Core MAB functions ##############################

def drop_v7(connection, schema_name, arm_list_to_delete):
    bulk_drop(connection, schema_name, arm_list_to_delete)


def create_query_v7(connection, schema_name, arm_list_to_add, arm_list_to_delete, queries):
    """
    This method aggregate few functions of the sql helper class.
        1. This method create the indexes related to the given bandit arms
        2. Execute all the queries in the given list
        3. Clean (drop) the created indexes
        4. Finally returns the cost taken to run all the queries

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param arm_list_to_add: arms that need to be added in this round
    :param arm_list_to_delete: arms that need to be removed in this round
    :param queries: queries that should be executed
    :return:
    """
    if tables_global is None:
        get_tables(connection)
    creation_cost = bulk_create(connection, schema_name, arm_list_to_add)
    execute_cost = 0
    execute_cost_transactional = 0
    execute_cost_analytical = 0
    query_plans = []
    query_times = {}
    query_counts = {}
    is_analytical = {}
    for query in queries:
        query_plan = execute_query_v2(connection, query.get_query_string())
        if query_plan:
            cost = query_plan[constants.COST_TYPE_CURRENT_EXECUTION]
            if query.id in query_times:
                query_times[query.id] += cost
                query_counts[query.id] += 1
            else:
                query_times[query.id] = cost
                query_counts[query.id] = 1
                is_analytical[query.id] = query.is_analytical
            execute_cost += cost
            if query.is_analytical:
                execute_cost_analytical += cost
            else:
                execute_cost_transactional += cost
            if query.first_seen == query.last_seen:
                query.original_running_time = cost

        query_plans.append(query_plan)

    for q_id, q_time in query_times.items():
        logging.info(f"Query {q_id}: \tanalytical-{is_analytical[q_id]} \tcount-{query_counts[q_id]} \tcost-{q_time}")
    logging.info(f"Index creation cost: {sum(creation_cost.values())}")
    logging.info(f"Time taken to run the queries: {execute_cost}")
    logging.info(f"Time taken for analytical queries: {execute_cost_analytical}")
    logging.info(f"Time taken for transactional queries: {execute_cost_transactional}")

    return execute_cost, creation_cost, query_plans, execute_cost_analytical, execute_cost_transactional


def bulk_create(connection, schema_name, bandit_arm_list):
    """
        This uses create_index method to create multiple indexes at once. This is used when a super arm is pulled

        :param connection: sql_connection
        :param schema_name: name of the database schema
        :param bandit_arm_list: list of BanditArm objects
        :return: cost (regret)
    """
    cost = {}
    for name, bandit_arm in bandit_arm_list.items():
        if type(bandit_arm).__name__ == 'BanditArmMV':
            cost[name] = create_view(connection, bandit_arm.index_name, bandit_arm.view_query,
                                     bandit_arm.index_query)
            if cost[name]:
                set_arm_size_mv(connection, bandit_arm)
        else:
            cost[name] = create_index_v1(connection, schema_name, bandit_arm.table_name, bandit_arm.index_cols,
                                         bandit_arm.index_name,
                                         bandit_arm.include_cols)
            set_arm_size(connection, bandit_arm)
    return cost


def bulk_drop(connection, schema_name, bandit_arm_list, file=None):
    """
    Drops the index for all given bandit arms

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param bandit_arm_list: list of bandit arms
    :return:
    """
    for name, bandit_arm in bandit_arm_list.items():
        if type(bandit_arm).__name__ == 'BanditArmMV':
            drop_view(connection, schema_name, name, file)
        else:
            drop_index(connection, schema_name, bandit_arm.table_name, bandit_arm.index_name, file)


def create_index_v1(connection, schema_name, tbl_name, col_names, idx_name, include_cols=()):
    """
    Create an index on the given table

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param tbl_name: name of the database table
    :param col_names: string list of column names
    :param idx_name: name of the index
    :param include_cols: columns that needed to added as includes
    """
    if include_cols:
        query = f"CREATE NONCLUSTERED INDEX {idx_name} ON {schema_name}.{tbl_name} ({', '.join(col_names)})" \
            f" INCLUDE ({', '.join(include_cols)})"
    else:
        query = f"CREATE NONCLUSTERED INDEX {idx_name} ON {schema_name}.{tbl_name} ({', '.join(col_names)})"
    cursor = connection.cursor()
    cursor.execute("SET STATISTICS XML ON")
    cursor.execute(query)
    stat_xml = cursor.fetchone()[0]
    cursor.execute("SET STATISTICS XML OFF")
    connection.commit()
    logging.info(f"Added: {idx_name}")
    logging.debug(query)

    # Return the current reward
    query_plan = QueryPlan.get_plan(stat_xml)
    return query_plan[constants.COST_TYPE_CURRENT_EXECUTION]


def drop_index(connection, schema_name, tbl_name, idx_name, file=None):
    """
    Drops the index on the given table with given name

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param tbl_name: name of the database table
    :param idx_name: name of the index
    :return:
    """
    query = f"DROP INDEX {schema_name}.{tbl_name}.{idx_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    if file:
        file.write(query + '\n')
    connection.commit()
    logging.info(f"removed: {idx_name}")
    logging.debug(query)


def create_view(connection, view_name, view_query, index_query):
    # Initial view creation operation has a negligible cost. So the returned cost is from the clustered index creation.

    # creates the view
    cursor = connection.cursor()
    cursor.execute(view_query)
    connection.commit()

    # creates the clustered index
    try:
        cursor.execute("SET STATISTICS XML ON")
        cursor.execute(index_query)
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET STATISTICS XML OFF")
        connection.commit()
        logging.info(f"Added and Created Clustered Index on: {view_name}")
        # Return the current reward
        query_plan = QueryPlan.get_plan(stat_xml)
        return query_plan[constants.COST_TYPE_CURRENT_EXECUTION]
    except:
        logging.error(f"Failed index creation: {view_name}")
        logging.error(f"{view_query}")
        logging.error(f"{index_query}")
        return 0


def drop_view(connection, schema_name, view_name, file=None):
    """
    Drops the index on the given table with given name

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param view_name: name of the view
    :return:
    """
    query = f"DROP VIEW {schema_name}.{view_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    if file:
        file.write(query+'\n')
    connection.commit()
    logging.info(f"removed: {view_name}")


def execute_query_v2(connection, query, print_exc=True):
    """
    This executes the given query and return the time took to run the query. This Clears the cache and executes
    the query and return the time taken to run the query. This return the 'elapsed time' by default.
    However its possible to get the cpu time by setting the is_cpu_time to True

    :param connection: sql_connection
    :param query: query that need to be executed
    :param print_exc: print the exception, True or False
    :return: time taken for the query
    """
    try:
        cursor = connection.cursor()
        cursor.execute("CHECKPOINT;")
        cursor.execute("DBCC DROPCLEANBUFFERS;")
        cursor.execute("SET STATISTICS XML ON")
        cursor.execute(query)
        cursor.nextset()
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET STATISTICS XML OFF")
        connection.commit()
        return QueryPlan.get_plan(stat_xml)
    except Exception as e:
        return None


# ############################# Hyp MAB functions ##############################

def hyp_check_config(connection, schema_name, arm_list_to_add, queries, file_path):
    """
    This method aggregate few functions of the sql helper class.
        1. This method create the indexes related to the given bandit arms
        2. Execute all the queries in the given list
        3. Clean (drop) the created indexes
        4. Finally returns the time taken to run all the queries

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param arm_list_to_add: new arms considered in this round
    :param super_arm_list: complete arm list
    :param queries: queries that should be executed
    :return:
    """
    cost = 0
    file = open(file_path, 'w')
    if tables_global is None:
        get_tables(connection)
    cost += hyp_bulk_create(connection, schema_name, arm_list_to_add, file)
    query_plans = []
    hyp_enable_index(connection, file)
    for query in queries:
        query_plan, exe_cost = hyp_execute_query_v2(connection, query.get_query_string(hyp=True), file)
        query_plans.append(query_plan)
        cost += exe_cost
        if query.first_seen == query.last_seen:
            query.original_hyp_running_time = query_plan.sub_tree_cost
    bulk_drop(connection, schema_name, arm_list_to_add, file)
    file.close()

    return query_plans, cost


def get_hyp_cost(connection, file):
    """
    This method aggregate few functions of the sql helper class.
        1. This method create the indexes related to the given bandit arms
        2. Execute all the queries in the given list
        3. Clean (drop) the created indexes
        4. Finally returns the time taken to run all the queries

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param arm_list_to_add: new arms considered in this round
    :param super_arm_list: complete arm list
    :param queries: queries that should be executed
    :return:
    """
    try:
        f = open(file, 'r')
        query = f.read()
        f.close()
        cursor = connection.cursor()
        cursor.execute("SET AUTOPILOT ON")
        start_time = datetime.datetime.now()
        cursor.execute(query)
        end_time = datetime.datetime.now()
        cursor.execute("SET AUTOPILOT OFF")
        connection.commit()
        return (end_time - start_time).total_seconds()
    except Exception as e:
        print("Exception when executing hyp cost")
        return 0


def hyp_bulk_create(connection, schema_name, bandit_arm_list, file):
    """
        This uses create_index method to create multiple indexes at once. This is used when a super arm is pulled

        :param connection: sql_connection
        :param schema_name: name of the database schema
        :param bandit_arm_list: list of BanditArm objects
        :return: cost (regret)
    """
    cost = 0
    for name, bandit_arm in bandit_arm_list.items():
        if type(bandit_arm).__name__ == 'BanditArmMV':
            cost += hyp_create_view(connection, bandit_arm.index_name, bandit_arm.view_query,
                                         bandit_arm.index_query, file)
        else:
            cost += hyp_create_index_v1(connection, schema_name, bandit_arm.table_name, bandit_arm.index_cols,
                                             bandit_arm.index_name, file, bandit_arm.include_cols)
    return cost


def hyp_create_index_v1(connection, schema_name, tbl_name, col_names, idx_name, file, include_cols=()):
    """
    Create an hypothetical index on the given table

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param tbl_name: name of the database table
    :param col_names: string list of column names
    :param idx_name: name of the index
    :param include_cols: columns that needed to be added as includes
    """
    if include_cols:
        query = f"CREATE NONCLUSTERED INDEX {idx_name} ON {schema_name}.{tbl_name} ({', '.join(col_names)}) " \
                f"INCLUDE ({', '.join(include_cols)}) WITH STATISTICS_ONLY = -1"
    else:
        query = f"CREATE NONCLUSTERED INDEX {idx_name} ON {schema_name}.{tbl_name} ({', '.join(col_names)}) " \
                f"WITH STATISTICS_ONLY = -1"
    cursor = connection.cursor()
    start_time = datetime.datetime.now()
    cursor.execute(query)
    file.write(query + '\n')
    end_time = datetime.datetime.now()
    connection.commit()
    logging.debug(query)
    logging.info(f"Added HYP: {idx_name}")
    return (end_time - start_time).total_seconds()


def hyp_create_view(connection, view_name, view_query, index_query, file):
    # Initial view creation operation has a negligible cost. So the returned cost is from the clustered index creation.

    # creates the view
    cursor = connection.cursor()
    cursor.execute(view_query)
    connection.commit()

    # creates the clustered index
    cursor.execute("SET STATISTICS XML ON")
    start_time = datetime.datetime.now()
    cursor.execute(index_query[:-1] + " WITH STATISTICS_ONLY = -1;")
    file.write(index_query[:-1] + " WITH STATISTICS_ONLY = -1;\n")
    end_time = datetime.datetime.now()
    cursor.execute("SET STATISTICS XML OFF")
    connection.commit()
    logging.info(f"Added and Created Hyp Clustered Index on: {view_name}")

    return (end_time - start_time).total_seconds()


def hyp_enable_index(connection, file):
    """
    This enables the hypothetical indexes for the given connection. This will be enabled for a given connection and all
    hypothetical queries must be executed via the same connection
    :param connection: connection for which hypothetical indexes will be enabled
    """
    query = f'''SELECT dbid = Db_id(),
                    objectid = object_id,
                    indid = index_id, type
                FROM   sys.indexes
                WHERE  is_hypothetical = 1;'''
    cursor = connection.cursor()
    cursor.execute(query)
    result_rows = cursor.fetchall()
    for result_row in result_rows:
        if result_row[3] == 2:
            query_2 = f"DBCC AUTOPILOT(0, {result_row[0]}, {result_row[1]}, {result_row[2]})"
            cursor.execute(query_2)
            file.write(query_2 + ';\n')
        elif result_row[3] == 1:
            query_2 = f"DBCC AUTOPILOT(6, {result_row[0]}, {result_row[1]}, {result_row[2]})"
            cursor.execute(query_2)
            file.write(query_2 + ';\n')
        else:
            print('Unknown index type')


def hyp_execute_query_v2(connection, query, file, print_exc=True):
    """
    This executes the given query and return the time took to run the query. This Clears the cache and executes
    the query and return the time taken to run the query. This return the 'elapsed time' by default.
    However its possible to get the cpu time by setting the is_cpu_time to True

    :param connection: sql_connection
    :param query: query that need to be executed
    :param print_exc: print the exception, True or False
    :return: time taken for the query
    """
    try:
        cursor = connection.cursor()
        cursor.execute("CHECKPOINT;")
        cursor.execute("DBCC DROPCLEANBUFFERS;")
        cursor.execute("SET STATISTICS XML ON")
        cursor.execute("SET AUTOPILOT ON")
        start_time = datetime.datetime.now()
        cursor.execute(query)
        file.write(query + '\n')
        end_time = datetime.datetime.now()
        stat_xml = cursor.fetchone()[0]
        cursor.execute("SET AUTOPILOT OFF")
        cursor.execute("SET STATISTICS XML OFF")
        connection.commit()
        return QueryPlan.get_plan(stat_xml), (end_time - start_time).total_seconds()
    except Exception as e:
        return None


# ############################# Helper function ##############################

def get_table_row_count(connection, schema_name, tbl_name):
    row_query = f'''SELECT SUM (Rows)
                        FROM sys.partitions
                        WHERE index_id IN (0, 1)
                        And OBJECT_ID = OBJECT_ID('{schema_name}.{tbl_name}');'''
    cursor = connection.cursor()
    cursor.execute(row_query)
    row_count = cursor.fetchone()[0]
    return row_count


def get_all_columns(connection):
    """
    Get all column in the database of the given connection. Note that the connection here is directly pointing to a
    specific database of interest

    :param connection: Sql connection
    :return: dictionary of lists - columns, number of columns
    """
    query = '''SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS;'''
    columns = defaultdict(list)
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        columns[result[0]].append(result[1])

    return columns, len(results)


def get_table_columns(connection, table_name):
    """
    Get all column in the table of the given connection and table. Note that the connection here is directly pointing to a
    specific database of interest

    :param connection: Sql connection
    :param table_name: table of interest
    :return: dictionary of lists - columns, number of columns
    """
    query = f'''SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table_name}';'''
    columns = defaultdict(list)
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        columns[result[0]].append(result[1])

    return columns, len(results)


def get_current_pds_size(connection):
    """
    Get the current size of all the physical design structures
    :param connection: SQL Connection
    :return: size of all the physical design structures in MB
    """
    query = '''SELECT (SUM(s.[used_page_count]) * 8)/1024.0 AS size_mb FROM sys.dm_db_partition_stats AS s'''
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchone()[0]


def get_primary_key(connection, schema_name, table_name):
    """
    Get Primary key of a given table. Note tis might not be in order (not sure)
    :param connection: SQL Connection
    :param schema_name: schema name of table
    :param table_name: table name which we want to find the PK
    :return: array of columns
    """
    if table_name in pk_columns_dict:
        pk_columns = pk_columns_dict[table_name]
    else:
        pk_columns = []
        query = f"""SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + QUOTENAME(CONSTRAINT_NAME)), 'IsPrimaryKey') = 1
                AND TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = '{schema_name}'"""
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        for result in results:
            pk_columns.append(result[0])
        pk_columns_dict[table_name] = pk_columns
    return pk_columns


def get_column_data_length_v2(connection, table_name, col_names):
    """
    get the data length of given set of columns
    :param connection: SQL Connection
    :param table_name: Name of the SQL table
    :param col_names: array of columns
    :return:
    """
    tables = get_tables(connection)
    varchar_count = 0
    column_data_length = 0

    for column_name in col_names:
        column = tables[table_name].columns[column_name]
        if column.column_type == 'varchar':
            varchar_count += 1
        column_data_length += column.column_size if column.column_size else 0

    if varchar_count > 0:
        variable_key_overhead = 2 + varchar_count * 2
        return column_data_length + variable_key_overhead
    else:
        return column_data_length


def get_max_column_data_length_v2(connection, table_name, col_names):
    tables = get_tables(connection)
    column_data_length = 0
    for column_name in col_names:
        column = tables[table_name].columns[column_name]
        column_data_length += column.max_column_size if column.max_column_size else 0
    return column_data_length


def get_columns(connection, table_name):
    """
    Get all the columns in the given table

    :param connection: sql connection
    :param table_name: table name
    :return: dictionary of columns column name as the key
    """
    columns = {}
    cursor = connection.cursor()
    data_type_query = f"""SELECT COLUMN_NAME, DATA_TYPE, COL_LENGTH( '{table_name}' , COLUMN_NAME)
                          FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE 
                            TABLE_NAME = '{table_name}'"""
    cursor.execute(data_type_query)
    results = cursor.fetchall()
    variable_len_query = 'SELECT '
    variable_len_select_segments = []
    variable_len_inner_segments = []
    varchar_ids = []
    for result in results:
        col_name = result[0]
        column = Column(table_name, col_name, result[1])
        column.set_max_column_size(int(result[2]))
        if result[1] != 'varchar':
            column.set_column_size(int(result[2]))
        else:
            varchar_ids.append(col_name)
            variable_len_select_segments.append(f'''AVG(DL_{col_name})''')
            variable_len_inner_segments.append(f'''DATALENGTH({col_name}) DL_{col_name}''')
        columns[col_name] = column

    if len(varchar_ids) > 0:
        variable_len_query = variable_len_query + ', '.join(
            variable_len_select_segments) + ' FROM (SELECT TOP (1000) ' + ', '.join(
            variable_len_inner_segments) + f' FROM {table_name}) T'
        cursor.execute(variable_len_query)
        result_row = cursor.fetchone()
        for i in range(0, len(result_row)):
            columns[varchar_ids[i]].set_column_size(result_row[i])

    return columns


def get_tables(connection):
    """
    Get all tables as Table objects
    :param connection: SQL Connection
    :return: Table dictionary with table name as the key
    """
    global tables_global
    if tables_global is not None:
        return tables_global
    else:
        tables = {}
        get_tables_query = """SELECT TABLE_NAME
                                FROM INFORMATION_SCHEMA.TABLES
                                WHERE TABLE_TYPE = 'BASE TABLE'"""
        cursor = connection.cursor()
        cursor.execute(get_tables_query)
        results = cursor.fetchall()
        for result in results:
            table_name = result[0]
            row_count = get_table_row_count(connection, constants.SCHEMA_NAME, table_name)
            pk_columns = get_primary_key(connection, constants.SCHEMA_NAME, table_name)
            tables[table_name] = Table(table_name, row_count, pk_columns)
            tables[table_name].set_columns(get_columns(connection, table_name))
        tables_global= tables
    return tables_global


def get_table_list(connection, min_row_count):
    """
    Get the list of names of tables with more rows then the minimum row count

    :param connection: SQL Connection
    :param min_row_count: minimum row count
    :return: table name list (list of Strings)
    """
    global tables_global
    if tables_global is None:
        tables = {}
        get_tables_query = """SELECT TABLE_NAME
                                    FROM INFORMATION_SCHEMA.TABLES
                                    WHERE TABLE_TYPE = 'BASE TABLE'"""
        cursor = connection.cursor()
        cursor.execute(get_tables_query)
        results = cursor.fetchall()
        for result in results:
            table_name = result[0]
            row_count = get_table_row_count(connection, constants.SCHEMA_NAME, table_name)
            pk_columns = get_primary_key(connection, constants.SCHEMA_NAME, table_name)
            tables[table_name] = Table(table_name, row_count, pk_columns)
            tables[table_name].set_columns(get_columns(connection, table_name))
        tables_global = tables

    table_name_list = []
    for table_name, table in tables_global.items():
        if table.table_row_count > min_row_count:
            table_name_list.append(table_name)
    return table_name_list


def get_estimated_size_of_mv_v2(connection, payload, mv_query, count_query, count_query_id, is_gb):
    """
    This helper method can be used to get a estimate size for a index. This simply multiply the column sizes with a
    estimated row count (need to improve further)

    :param connection: sql_connection
    :param payload: payload of the MV query
    :param mv_query: query used to create the MV
    :return: estimated size in MB
    """
    # Except for the estimated number of rows, rest of the calculation looks accurate.
    # Bit hard to think of a better way to estimate the number of rows
    # estimated_rows = QueryPlan.get_plan(get_query_plan_xml(connection, mv_query)).estimated_rows
    global count_numbers
    global cache_hits
    if count_query_id in count_numbers and not is_gb:
        estimated_rows = count_numbers[count_query_id]
        cache_hits += 1
    else:
        try:
            cursor = connection.cursor()
            cursor.execute(count_query)
            result = cursor.fetchone()
            estimated_rows = result[0]
            if not is_gb:
                count_numbers[count_query_id] = estimated_rows
        except:
            if not is_gb:
                count_numbers[count_query_id] = -1
            estimated_rows = -1

    if estimated_rows > 0:
        tables = set(payload.keys())
        total_row_length = 0
        header_size = 4
        nullable_buffer = 2
        for tbl_name in tables:
            columns = set()
            if tbl_name in payload:
                columns = columns.union(payload[tbl_name])
            key_columns_length = get_column_data_length_v2(connection, tbl_name, columns)
            total_row_length += key_columns_length

        rows_per_page = 8096/(total_row_length + nullable_buffer + header_size)
        number_of_leafs = estimated_rows/rows_per_page

        return (8192 * number_of_leafs) / float(1024 * 1024)
    else:
        return -1


def get_estimated_size_of_index_v1(connection, schema_name, tbl_name, col_names):
    """
    This helper method can be used to get a estimate size for a index. This simply multiply the column sizes with a
    estimated row count (need to improve further)

    :param connection: sql_connection
    :param schema_name: name of the database schema
    :param tbl_name: name of the database table
    :param col_names: string list of column names
    :return: estimated size in MB
    """
    table = get_tables(connection)[tbl_name]
    header_size = 6
    nullable_buffer = 2
    primary_key = get_primary_key(connection, schema_name, tbl_name)
    primary_key_size = get_column_data_length_v2(connection, tbl_name, primary_key)
    col_not_pk = tuple(set(col_names) - set(primary_key))
    key_columns_length = get_column_data_length_v2(connection, tbl_name, col_not_pk)
    index_row_length = header_size + primary_key_size + key_columns_length + nullable_buffer
    row_count = table.table_row_count
    estimated_size = row_count * index_row_length
    estimated_size = estimated_size/float(1024*1024)
    max_column_length = get_max_column_data_length_v2(connection, tbl_name, col_names)
    if max_column_length > 1700:
        print(f'Index going past 1700: {col_names}')
        estimated_size = 99999999
    return estimated_size


def get_query_plan_xml(connection, query):
    """
    This returns the XML query plan of  the given query

    :param connection: sql_connection
    :param query: sql query for which we need the query plan
    :return: XML query plan as a String
    """
    cursor = connection.cursor()
    cursor.execute("SET SHOWPLAN_XML ON;")
    cursor.execute(query)
    query_plan_xml = cursor.fetchone()[0]
    cursor.execute("SET SHOWPLAN_XML OFF;")
    return query_plan_xml


def remove_all_non_clustered(connection, schema_name):
    """
    Removes all non-clustered indexes from the database
    :param connection: SQL Connection
    :param schema_name: schema name related to the index
    """
    query = """select i.name as index_name, t.name as table_name
                from sys.indexes i, sys.tables t
                where i.object_id = t.object_id and i.type_desc = 'NONCLUSTERED'"""
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        if not result[0].startswith('UQ_'):
            drop_index(connection, schema_name, result[1], result[0])


def get_table_scan_times_structure():
    query_table_scan_times = copy.deepcopy(constants.TABLE_SCAN_TIMES[database[:-4]])
    return query_table_scan_times


def drop_all_dta_statistics(connection):
    query_get_stat_names = """SELECT OBJECT_NAME(s.[object_id]) AS TableName, s.[name] AS StatName
                                FROM sys.stats s
                                WHERE OBJECTPROPERTY(s.OBJECT_ID,'IsUserTable') = 1 AND s.name LIKE '_dta_stat%';"""
    cursor = connection.cursor()
    cursor.execute(query_get_stat_names)
    results = cursor.fetchall()
    for result in results:
        drop_statistic(connection, result[0], result[1])
    logging.info("Dropped all dta statistics")


def drop_statistic(connection, table_name, stat_name):
    query = f"DROP STATISTICS {table_name}.{stat_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    cursor.commit()


def set_arm_size(connection, bandit_arm):
    query = f"""SELECT (SUM(s.[used_page_count]) * 8)/1024 AS IndexSizeMB
                FROM sys.dm_db_partition_stats AS s
                INNER JOIN sys.indexes AS i ON s.[object_id] = i.[object_id]
                    AND s.[index_id] = i.[index_id]
                WHERE i.[name] = '{bandit_arm.index_name}'
                GROUP BY i.[name]
                ORDER BY i.[name]
        """
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    bandit_arm.memory = result[0]
    return bandit_arm


def set_arm_size_mv(connection, bandit_arm):
    """
    Returns the size of the MV in MB
    :param connection: SQL Connection
    :param bandit_arm: Bandit arm for the MV
    :return: int size (in MB)
    """
    query = f"""
        SELECT 
            (SUM(a.total_pages) * 8)/1024 AS TotalSpaceMB
        FROM 
            sys.views v
        INNER JOIN      
            sys.indexes i ON v.OBJECT_ID = i.object_id
        INNER JOIN 
            sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
        INNER JOIN 
            sys.allocation_units a ON p.partition_id = a.container_id
        WHERE 
            v.Name = '{bandit_arm.index_name}' --View name only, not 'schema.viewname'
            AND
            i.index_id = 1   -- clustered index, remove this to see all indexes
        GROUP BY 
            v.NAME, i.object_id, i.index_id, i.name, p.Rows
        """
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    bandit_arm.memory = result[0]
    return bandit_arm


def restart_sql_server():
    command1 = f"net stop mssqlserver"
    command2 = f"net start mssqlserver"
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command1, shell=True, stdout=devnull)
        time.sleep(45)
        subprocess.run(command2, shell=True, stdout=devnull)
        time.sleep(15)
    logging.info("Server Restarted")
    return


def get_database_size(connection):
    database_size = 10240
    try:
        query = "exec sp_spaceused @oneresultset = 1;"
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        database_size = float(result[4].split(" ")[0])/1024
    except Exception as e:
        logging.error("Exception when get_database_size: " + str(e))
    return database_size


def clean_up_routine(sql_connection):
    # restart server. We need to do this before restore to remove all connections
    if constants.SERVER_RESTART:
        restart_sql_server()

    master_connection = sql_connection.get_master_sql_connection()

    # restore the backup
    if constants.RESTORE_BACKUP:
        restore_database_snapshot(master_connection)

    sql_connection.close_sql_connection(master_connection)


def create_database_snapshot(master_connection):
    """
    This create ta database snapshot, we need to create a snapshot when we setup experiment
    local: C:\Program Files\Microsoft SQL Server\MSSQL14.MSSQLSERVER\MSSQL\Backup
    server: default

    :param master_connection: connection to the master DB
    """
    ss_name = f"{database}_snapshot"
    ss_location = "E:\Microsoft SQL Server Data\MSSQL14.MSSQLSERVER\MSSQL\Backup"
    create_ss_query = f"""CREATE DATABASE {ss_name} ON  
                        ( NAME = {database}, FILENAME =   
                        '{ss_location}\\{ss_name}.ss' )  
                        AS SNAPSHOT OF {database};  
                        """
    cursor = master_connection.cursor()
    cursor.execute(create_ss_query)
    while cursor.nextset():
        pass
    logging.info("DB snapshot created")
    print(create_ss_query)


def restore_database_snapshot(master_connection):
    """
    This restores the database snapshot, we have to make sure there is only one snapshot at a time.

    :param master_connection: connection to the master DB
    """
    ss_name = f"{database}_snapshot"
    restore_ss_query = f"""RESTORE DATABASE {database} from   
                        DATABASE_SNAPSHOT = '{ss_name}';  
                        """
    cursor = master_connection.cursor()
    cursor.execute(restore_ss_query)
    while cursor.nextset():
        pass
    logging.info("DB snapshot restored")
