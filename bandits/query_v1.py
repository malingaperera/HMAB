from database.sql_helper_factory import SQLHelperFactory
sql_helper = SQLHelperFactory.get_sql_helper()


class Query:
    def __init__(self, connection, query_id, query_string, predicates, payloads, time_stamp=0):
        query_string = query_string.lower()
        self.id = query_id
        self.predicates = predicates
        self.payload = payloads
        self.group_by = {}
        # self.selectivity = sql_helper.get_selectivity_v3(connection, query_string, self.predicates)
        self.query_strings = [query_string]
        self.frequency = 1
        self.last_seen = time_stamp
        self.first_seen = time_stamp
        self.table_scan_times = sql_helper.get_table_scan_times_structure()
        self.index_scan_times = sql_helper.get_table_scan_times_structure()
        self.table_scan_times_hyp = sql_helper.get_table_scan_times_structure()
        self.index_scan_times_hyp = sql_helper.get_table_scan_times_structure()
        self.context = None
        self.next_execution = 0
        self.original_running_time = 0
        self.original_hyp_running_time = 0
        if query_string.strip().startswith('select') or query_string.strip().startswith('with'):
            self.is_analytical = True
        else:
            self.is_analytical = False
            if type(self.id) != str:
                raise Exception('Assumption failed')

    def __hash__(self):
        return self.id

    def get_query_string(self, hyp=False):
        query_string = self.query_strings[self.next_execution]
        if not hyp:
            self.next_execution += 1
        return query_string
