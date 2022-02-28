import pyodbc
import configparser
import psycopg2

import constants


def get_sql_connection():
    """
    This method simply returns the sql connection based on the DB type and the connection settings
    defined in the db.conf
    :return: connection
    """

    # Reading the Database configurations
    db_config = configparser.ConfigParser()
    db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
    db_type = db_config['SYSTEM']['db_type']
    server = db_config[db_type]['server']
    database = db_config[db_type]['database']

    if db_type == 'MSSQL':
        driver = db_config[db_type]['driver']
        return pyodbc.connect(
            r'Driver=' + driver + ';Server=' + server + ';Database=' + database + ';Trusted_Connection=yes;')
    elif db_type == 'PG':
        user = db_config[db_type]['user']
        password = db_config[db_type]['password']
        return psycopg2.connect(host=server, database=database, user=user, password=password)


def get_master_sql_connection():
    """
    This method simply returns the sql connection based on the DB type and the connection settings
    defined in the db.conf
    :return: connection
    """

    # Reading the Database configurations
    db_config = configparser.ConfigParser()
    db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
    db_type = db_config['SYSTEM']['db_type']
    server = db_config[db_type]['server']
    database = 'master'
    driver = db_config[db_type]['driver']

    return pyodbc.connect(
        r'Driver=' + driver + ';Server=' + server + ';Database=' + database + ';Trusted_Connection=yes;',autocommit=True)


def close_sql_connection(connection):
    """
    Take care of the closing process of the SQL connection
    :param connection: sql_connection
    :return: operation status
    """
    return connection.close()
