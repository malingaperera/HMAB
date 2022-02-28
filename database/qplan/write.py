from database.qplan.read import ReadQueryPlan
from database.qplan.index_use import IndexWrite


class WriteQueryPlan(ReadQueryPlan):

    def __init__(self, xml_string, po_set, xml_path):
        ReadQueryPlan.__init__(self, xml_string)
        self.po_set = po_set
        self.xml_path = xml_path

    def get_index_uses(self):
        index_uses = []
        for rel_op in self.rel_ops:
            if rel_op.attrib.get('PhysicalOp') in self.po_set:
                # Getting information from rel-op level
                node_id, est_cpu, est_elapsed, est_rows_out, est_rows_read, subtree_cost, tab_cardinality = self.get_rel_op_info(rel_op)

                # Getting child information
                child_act_elapsed_max = self.get_child_max_elapsed(rel_op)

                # Getting information from the thread level
                act_cpu_max, act_cpu_sum, act_elapsed_max, act_elapsed_sum, act_rows_out, act_rows_read = self.get_thread_info(rel_op, child_act_elapsed_max)

                index_use = IndexWrite(node_id, act_elapsed_max, act_elapsed_sum, est_elapsed, act_cpu_max, act_cpu_sum,
                                       est_cpu, subtree_cost, act_rows_out, est_rows_out)
                index_use_objects = rel_op.findall(self.xml_path)
                for index_use_object in index_use_objects:
                    index, table, index_kind = ReadQueryPlan.get_index_object_info(index_use_object)
                    index_use.add_index(table, index, index_kind)
                index_uses.append(index_use)
        return index_uses


class InsertQueryPlan(WriteQueryPlan):

    def __init__(self, xml_string):
        WriteQueryPlan.__init__(self, xml_string, po_set={'Clustered Index Insert', 'Index Insert'}, xml_path='./*/Object')
        self.index_inserts = self.get_index_uses()


class DeleteQueryPlan(WriteQueryPlan):

    def __init__(self, xml_string):
        WriteQueryPlan.__init__(self, xml_string, po_set={'Clustered Index Delete', 'Index Delete'}, xml_path='./*/Object')
        self.index_deletes = self.get_index_uses()


class UpdateQueryPlan(WriteQueryPlan):

    def __init__(self, xml_string):
        WriteQueryPlan.__init__(self, xml_string, po_set={'Clustered Index Update', 'Index Update'}, xml_path='./*/Object')
        self.index_updates = self.get_index_uses()

