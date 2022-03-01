class BanditArmMV:
    def __init__(self, query_id, payload, join_cols, table_names, group_by_columns):
        group_by = True if group_by_columns else False
        self.schema_name = 'dbo'
        self.table_names = table_names
        self.table_name = 'MV'
        self.payload = payload
        self.join_cols = join_cols
        self.index_name = self.get_arm_id(query_id, table_names, group_by)
        self.memory = None
        self.name_encoded_context = []
        self.index_usage_last_batch = 0
        self.query_id = query_id
        self.query_ids = set()
        self.query_ids_backup = set()
        self.clustered_index_time = 0
        self.creation_query = None
        self.view_query = None
        self.index_query = None
        self.group_by = group_by
        self.filter_by = False
        self.bandit_cluster = None
        self.group_by_columns = group_by_columns
        self.view_query_comps = None
        self.index_query_comps = None

    def __eq__(self, other):
        return self.index_name == other.index_name

    def __hash__(self):
        return hash(self.index_name)

    def __le__(self, other):
        if len(self.payload) > len(other.payload):
            return False
        else:
            for i in range(len(self.payload)):
                if self.payload[i] != other.payload[i]:
                    return False
            return True

    def __str__(self):
        return self.index_name

    def set_view_query_components(self, payload_list, table_subset, join_list, group_by_columns_original):
        self.view_query_comps = {"payload_list": payload_list, "table_subset": table_subset, "join_list": join_list,
                                 "group_by_columns_original": group_by_columns_original}

    def set_index_query_components(self, indexed_columns_equal, indexed_columns_range, indexed_columns_pk,
                                   group_by_columns_renamed):
        self.index_query_comps = {"indexed_columns_equal": indexed_columns_equal,
                                  "indexed_columns_range": indexed_columns_range, "indexed_columns_pk": indexed_columns_pk,
                                  "group_by_columns_renamed": group_by_columns_renamed}

    @staticmethod
    def get_arm_id(query_id, table_names, group_by):
        if group_by:
            arm_id = 'mv_' + str(query_id) + '_' + '_'.join(table_names).lower() + '_GB'
        else:
            arm_id = 'mv_' + str(query_id) + '_' + '_'.join(table_names).lower()
        arm_id = arm_id[:127]
        return arm_id


