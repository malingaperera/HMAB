import constants


class IndexUse:
    def __init__(self, node_id, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum, est_cpu,
                 sub_tree_cost, act_rows_output, est_rows_output):
        """
        Object to keep the measurements from index use

        :param node_id: ID of the node in xml
        :param act_elapsed_max: Float, max actual elapsed time taken from all the threads
        :param act_elapsed_sum: Float, sum of actual elapsed times taken from all the threads
        :param est_elapsed: Float, elapsed time calculated based on the query elapsed time and element sub-tree
        cost percentage
        :param act_cpu_max: Float, actual cpu time
        :param sub_tree_cost: Float, sub-tree cost
        :param act_rows_output: Int, actual output row count
        :param est_rows_output: Int, estimate output row count
        """
        self.node_id = node_id
        self.indices = []
        self.act_elapsed_max = act_elapsed_max
        self.act_elapsed_sum = act_elapsed_sum
        self.est_elapsed = est_elapsed
        self.act_cpu_max = act_cpu_max
        self.act_cpu_sum = act_cpu_sum
        self.est_cpu = est_cpu
        self.sub_tree_cost = sub_tree_cost
        self.act_rows_output = act_rows_output
        self.est_rows_output = est_rows_output

    def __getitem__(self, *args):
        if isinstance(*args, str):
            if args == constants.COST_TYPE_ELAPSED_TIME:
                return self.act_elapsed_max
            elif args == constants.COST_TYPE_CPU_TIME:
                return self.act_cpu_max
            elif args == constants.COST_TYPE_SUB_TREE_COST:
                return self.sub_tree_cost
            else:
                return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]

    def add_index(self, table, index, index_kind):
        """
        Adds a index to the operator
        """
        index = Index(table, index, index_kind)
        self.indices.append(index)


class IndexRead(IndexUse):

    def __init__(self, node_id, table, index, index_kind, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum, est_cpu,
                 sub_tree_cost, act_rows_read, act_rows_output, est_rows_read, est_rows_output, table_cardinality):
        """
        Object to keep the measurements from index use

        :param table: String, Table of the index
        :param index: String, Index used
        :param index_kind: String, Kind of the index Clustered / NonClustered
        :param act_rows_read: Int, Actual rows read, there is no estimated value for rows read.
        :param est_rows_read: Int, estimate read row count
        :param table_cardinality: Float, cardinality of the table
        """
        IndexUse.__init__(self, node_id, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum,
                          est_cpu, sub_tree_cost, act_rows_output, est_rows_output)
        self.add_index(table, index, index_kind)
        self.act_rows_read = act_rows_read
        self.est_rows_read = est_rows_read
        self.table_cardinality = table_cardinality
        self.is_lookup = False
        self.lookup_parent = None

    def set_look_up(self, parent):
        """

        :param parent: parent ID
        :return: None
        """
        self.is_lookup = True
        self.lookup_parent = parent


class IndexWrite(IndexUse):

    def __init__(self, node_id, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum, est_cpu,
                 sub_tree_cost, act_rows_output, est_rows_output):
        """
        Object to keep the measurements from index write. We don't have separate operators for all index writes, but one
        operator that will include all index writes.

        """
        IndexUse.__init__(self, node_id, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum,
                          est_cpu, sub_tree_cost, act_rows_output, est_rows_output)


class ViewRead(IndexRead):
    def __init__(self, node_id, table, index, index_kind, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max,
                 act_cpu_sum, est_cpu, sub_tree_cost, act_rows_read, act_rows_output, est_rows_read, est_rows_output,
                 table_cardinality, view_tables):
        """
        Object to keep the measurements from index use

        :param table: String, Table of the index
        :param index: String, Index used
        :param index_kind: String, Kind of the index Clustered / NonClustered
        :param act_rows_read: Int, Actual rows read, there is no estimated value for rows read.
        :param est_rows_read: Int, estimate read row count
        :param table_cardinality: Float, cardinality of the table
        """
        IndexRead.__init__(self, node_id, table, index, index_kind, act_elapsed_max, act_elapsed_sum, est_elapsed,
                           act_cpu_max, act_cpu_sum, est_cpu,
                           sub_tree_cost, act_rows_read, act_rows_output, est_rows_read, est_rows_output,
                           table_cardinality)
        self.view_tables = view_tables


class Index:
    def __init__(self, table, index, index_kind):
        self.table = table
        self.index = index
        self.index_kind = index_kind


