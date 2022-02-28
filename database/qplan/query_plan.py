from shared import helper_v2
from database.qplan.read import ReadQueryPlan
from database.qplan.write import UpdateQueryPlan, InsertQueryPlan, DeleteQueryPlan


class QueryPlan:

    @staticmethod
    def get_plan(xml_string):
        root = helper_v2.strip_namespace(xml_string)
        stmt_simple = root.find('.//StmtSimple')
        statement_text = str(stmt_simple.attrib.get('StatementText')).lower()
        if statement_text.strip().startswith('select') or statement_text.strip().startswith('with'):
            base_query_plan = ReadQueryPlan(xml_string)
        elif statement_text.strip().startswith('insert'):
            base_query_plan = InsertQueryPlan(xml_string)
        elif statement_text.strip().startswith('delete'):
            base_query_plan = DeleteQueryPlan(xml_string)
        elif statement_text.strip().startswith('update'):
            base_query_plan = UpdateQueryPlan(xml_string)
        else:
            raise ValueError(format)
        return base_query_plan
