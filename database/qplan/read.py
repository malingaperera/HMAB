from database.qplan.index_use import IndexRead, ViewRead
from shared import helper_v2

po_read = {'Index Seek', 'Index Scan', 'Clustered Index Scan', 'Clustered Index Seek'}
po_insert = {'Clustered Index Insert'}


class ReadQueryPlan:

    def __init__(self, xml_string):
        self.act_elapsed_max = 0
        self.est_elapsed_time = 0
        self.est_cpu_time = 0
        self.act_cpu_sum = 0
        self.non_clustered_index_usages = {}
        self.clustered_index_usages = {}
        self.clustered_view_usages = {}
        self.non_clustered_view_usages = {}
        self.xml = xml_string

        self.root = helper_v2.strip_namespace(xml_string)
        stmt_simple = self.root.find('.//StmtSimple')

        self.estimated_rows = float(stmt_simple.attrib.get('StatementEstRows'))
        self.sub_tree_cost = float(stmt_simple.attrib.get('StatementSubTreeCost'))

        query_stats = self.root.find('.//QueryTimeStats')
        if query_stats is not None:
            self.act_cpu_sum = float(query_stats.attrib.get('CpuTime')) / 1000
            self.act_elapsed_max = float(query_stats.attrib.get('ElapsedTime')) / 1000
            self.est_cpu_time = self.act_cpu_sum
            self.est_elapsed_time = self.act_elapsed_max

        self.rel_ops = self.root.findall('.//RelOp')

        # Get the sum of sub tree cost for physical operations (assumption: sub tree cost is dominated by the physical
        # operations). Note we include the index insert as well (but not update or delete)
        self.total_sub_tree_cost = 0
        self.total_actual_elapsed_max = 0
        self.total_actual_elapsed_sum = 0
        for rel_op in self.rel_ops:
            temp_act_elapsed_time = 0
            if rel_op.attrib.get('PhysicalOp') in (po_read.union(po_insert)):
                self.total_sub_tree_cost += float(rel_op.attrib.get('EstimatedTotalSubtreeCost'))
                runtime_thread_information = rel_op.findall('.//RunTimeCountersPerThread')
                for thread_info in runtime_thread_information:
                    thread_elapsed_sec = int(thread_info.attrib.get('ActualElapsedms')) / 1000
                    self.total_actual_elapsed_sum += thread_elapsed_sec
                    temp_act_elapsed_time = max(int(thread_elapsed_sec) if thread_elapsed_sec is not None else 0,
                                                temp_act_elapsed_time)
                    self.total_actual_elapsed_max += temp_act_elapsed_time

        for rel_op in self.rel_ops:
            if rel_op.attrib.get('PhysicalOp') in po_read:
                # Getting information from rel-op level
                node_id, est_cpu_time, est_elapsed_time, est_rows_output, est_rows_read, po_subtree_cost, table_cardinality = self.get_rel_op_info(rel_op)

                # Getting information from the thread level
                act_cpu_max, act_cpu_sum, act_elapsed_max, act_elapsed_sum, act_rows_output, act_rows_read = ReadQueryPlan.get_thread_info(rel_op)

                # Getting index information
                index_obj = rel_op.find('./IndexScan/Object')
                index, table, index_kind = ReadQueryPlan.get_index_object_info(index_obj)

                if index_kind in {'ViewNonClustered', 'ViewClustered'}:
                    view_tables = []
                    view_table_objs = rel_op.findall('./IndexScan/IndexedViewInfo/Object')
                    for view_table_obj in view_table_objs:
                        view_tables.append(view_table_obj.attrib.get('Table').strip("[]"))
                    index_use = ViewRead(node_id, table, index, index_kind, act_elapsed_max, act_elapsed_sum,
                                         est_elapsed_time, act_cpu_max, act_cpu_sum, est_cpu_time, po_subtree_cost,
                                         act_rows_read, act_rows_output, est_rows_read, est_rows_output,
                                         table_cardinality, view_tables)
                else:
                    index_use = IndexRead(node_id, table, index, index_kind, act_elapsed_max, act_elapsed_sum,
                                          est_elapsed_time, act_cpu_max, act_cpu_sum, est_cpu_time, po_subtree_cost,
                                          act_rows_read, act_rows_output, est_rows_read, est_rows_output,
                                          table_cardinality)
                if rel_op.attrib.get('PhysicalOp') in {'Index Seek', 'Index Scan'}:
                    if index_kind == 'ViewNonClustered':
                        self.non_clustered_view_usages[node_id] = index_use
                    else:
                        self.non_clustered_index_usages[node_id] = index_use
                elif rel_op.attrib.get('PhysicalOp') in {'Clustered Index Scan', 'Clustered Index Seek'}:
                    if index_kind == 'ViewClustered':
                        self.clustered_view_usages[node_id] = index_use
                    else:
                        if rel_op.attrib.get('PhysicalOp') == 'Clustered Index Seek':
                            is_lookup = self.get_attr(rel_op.find('./IndexScan'), 'Lookup', False) == 'true'
                            if is_lookup:
                                node_id = rel_op.attrib.get('NodeId')
                                neighbours = self.root.findall(f".//RelOp[@NodeId='{node_id}']/..//RelOp")
                                count_non_clustered = 0
                                neighbour = None
                                for ops in neighbours:
                                    if ops.attrib.get('PhysicalOp') in {'Index Seek', 'Index Scan'}:
                                        count_non_clustered += 1
                                        neighbour = ops
                                if count_non_clustered == 1:
                                    index_use.set_look_up(int(neighbour.attrib.get('NodeId')))
                                else:
                                    print("ERROR: Lookup assumption failed")
                        self.clustered_index_usages[node_id] = index_use

    def __getitem__(self, *args):
        if isinstance(*args, str):
            return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]

    def get_rel_op_info(self, rel_op):
        """
        Get the basic attributes in the rel-op
        :param rel_op: xml element
        :return: tuple
        """
        node_id = int(rel_op.attrib.get('NodeId'))
        est_rows_read = float(ReadQueryPlan.get_attr(rel_op, 'EstimatedRowsRead'))
        est_rows_output = float(rel_op.attrib.get('EstimateRows'))
        po_subtree_cost = float(rel_op.attrib.get('EstimatedTotalSubtreeCost'))
        table_cardinality = rel_op.attrib.get('TableCardinality')
        # For delete and update with primary key in where clause total sub tree cost can be 0
        if self.total_sub_tree_cost == 0:
            self.total_sub_tree_cost += po_subtree_cost
        est_elapsed_time = float(self.act_elapsed_max) * (po_subtree_cost / self.total_sub_tree_cost)
        est_cpu_time = float(self.act_cpu_sum) * (po_subtree_cost / self.total_sub_tree_cost)
        return node_id, est_cpu_time, est_elapsed_time, est_rows_output, est_rows_read, po_subtree_cost, table_cardinality

    @staticmethod
    def get_index_object_info(index_obj):
        """
        Strips index object information

        :param index_obj: xml element
        :return: tuple
        """
        index = index_obj.attrib.get('Index').strip("[]")
        table = index_obj.attrib.get('Table').strip("[]")
        index_kind = index_obj.attrib.get('IndexKind')
        return index, table, index_kind

    @staticmethod
    def get_thread_info(rel_op, child_max=0):
        """
        Get thread info in a rel-op
        :param rel_op: xml element
        :param child_max: maximum thread elapsed time from child nodes
        :return: summarised thread information
        """
        act_rows_read = 0
        act_rows_output = 0
        act_elapsed_max = 0
        act_elapsed_sum = 0
        act_cpu_max = 0
        act_cpu_sum = 0
        runtime_thread_information = rel_op.findall('*/./RunTimeCountersPerThread')
        for thread_info in runtime_thread_information:
            thread_elapsed_sec = int(thread_info.attrib.get('ActualElapsedms')) / 1000
            thread_cpu_sec = int(thread_info.attrib.get('ActualCPUms')) / 1000

            act_elapsed_sum += thread_elapsed_sec - child_max
            act_cpu_sum += thread_cpu_sec

            act_elapsed_max = max(thread_elapsed_sec if thread_elapsed_sec is not None else 0,
                                  act_elapsed_max)
            act_cpu_max = max(thread_cpu_sec if thread_cpu_sec is not None else 0,
                              act_cpu_max)

            act_rows_read += int(ReadQueryPlan.get_attr(thread_info, 'ActualRowsRead'))
            act_rows_output += int(thread_info.attrib.get('ActualRows'))
        return act_cpu_max, act_cpu_sum, act_elapsed_max - child_max, act_elapsed_sum, act_rows_output, act_rows_read

    @staticmethod
    def get_attr(element, attr_name, default=0):
        """
        Use this when you are not sure of the attribute availability
        :param element: xml tree element
        :param attr_name: String, Attribute Name
        :param default: Default value when the attribute is not available
        :return: String, attribute value
        """
        value = element.attrib.get(attr_name)
        value = value if value is not None else default
        return value

    @staticmethod
    def get_child_max_elapsed(element):
        max_act_elapsed_max = 0
        rel_op_children = element.findall('*/RelOp')
        for child in rel_op_children:
            act_cpu_max, act_cpu_sum, act_elapsed_max, act_elapsed_sum, act_rows_output, act_rows_read = ReadQueryPlan.get_thread_info(child)
            if max_act_elapsed_max < act_elapsed_max:
                max_act_elapsed_max = act_elapsed_max

        return max_act_elapsed_max





