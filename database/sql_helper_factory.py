import configparser
import constants

db_config = configparser.ConfigParser()
db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
db_type = db_config['SYSTEM']['db_type']


class SQLHelperFactory:
    @staticmethod
    def get_sql_helper():
        if db_type == 'MSSQL':
            import database.sql_helper_v3 as sql_helper
        elif db_type == 'PG':
            import database.sql_helper_postgres_v1 as sql_helper
        else:
            import database.sql_helper_v3 as sql_helper
        return sql_helper
