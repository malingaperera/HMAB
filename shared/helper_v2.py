import configparser
import json
import os

import pandas as pd
import seaborn as sns
from pandas import DataFrame
import xml.etree.ElementTree as ET
from io import StringIO

import constants


def get_experiment_folder_path(experiment_id):
    """
    Get the folder location of the experiment
    :param experiment_id: name of the experiment
    :return: file path as string
    """
    experiment_folder_path = constants.ROOT_DIR + constants.EXPERIMENT_FOLDER + '\\' + experiment_id + '\\'
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    return experiment_folder_path


def get_queries_v2():
    """
    Read all the queries in the queries pointed by the QUERY_DICT_FILE constant
    :return: list of queries
    """
    # Reading the configuration for given experiment ID
    exp_config = configparser.ConfigParser()
    exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)

    # experiment id for the current run
    experiment_id = exp_config['general']['run_experiment']
    workload_file = str(exp_config[experiment_id]['workload_file'])

    queries = []
    with open(constants.ROOT_DIR + workload_file) as f:
        line = f.readline()
        while line:
            queries.append(json.loads(line))
            line = f.readline()
    return queries


def get_query_properties():
    """
    Read all the queries in the queries pointed by the QUERY_DICT_FILE constant
    :return: list of queries
    """
    # Reading the configuration for given experiment ID
    exp_config = configparser.ConfigParser()
    exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)

    # experiment id for the current run
    experiment_id = exp_config['general']['run_experiment']
    query_parser_file = str(exp_config[experiment_id]['query_parser_file'])
    query_properties = __import__(query_parser_file, fromlist=[query_parser_file.split('.')[-1]]).query_properties
    return query_properties


def plot_exp_report(exp_id, exp_report_list, measurement_names, log_y=False):
    """
    Creates a plot for several experiment reports
    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :param measurement_names: What measurement that we will use for y
    :param log_y: draw y axis in log scale
    """
    for measurement_name in measurement_names:
        comps = []
        final_df = DataFrame()
        for exp_report in exp_report_list:
            df = exp_report.data
            df[constants.DF_COL_COMP_ID] = exp_report.component_id
            final_df = pd.concat([final_df, df])
            comps.append(exp_report.component_id)

        final_df = final_df[final_df[constants.DF_COL_MEASURE_NAME] == measurement_name]
        # Error style = 'band' / 'bars'
        sns_plot = sns.relplot(x=constants.DF_COL_BATCH, y=constants.DF_COL_MEASURE_VALUE, hue=constants.DF_COL_COMP_ID,
                               kind="line", ci="sd", data=final_df, err_style="band")
        if log_y:
            sns_plot.set(yscale="log")
        plot_title = measurement_name + " Comparison"
        sns_plot.set(xlabel=constants.DF_COL_BATCH, ylabel=measurement_name)
        sns_plot.savefig(get_experiment_folder_path(exp_id) + plot_title + '.png')


def create_comparison_tables(exp_id, exp_report_list):
    """
    Create a CSV with numbers that are important for the comparison

    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :return:
    """
    final_df = DataFrame(
        columns=[constants.DF_COL_COMP_ID, constants.DF_COL_BATCH_COUNT, constants.MEASURE_HYP_BATCH_TIME,
                 constants.MEASURE_INDEX_RECOMMENDATION_COST, constants.MEASURE_INDEX_CREATION_COST,
                 constants.MEASURE_QUERY_EXECUTION_COST, constants.MEASURE_ANALYTICAL_EXECUTION_COST,
                 constants.MEASURE_TRANSACTIONAL_EXECUTION_COST, constants.MEASURE_TOTAL_WORKLOAD_TIME])

    for exp_report in exp_report_list:
        data = exp_report.data
        component = exp_report.component_id
        rounds = exp_report.batches_per_rep
        reps = exp_report.reps

        # Get information from the data frame
        hyp_batch_time = get_avg_measure_value(data, constants.MEASURE_HYP_BATCH_TIME, reps)
        recommend_time = get_avg_measure_value(data, constants.MEASURE_INDEX_RECOMMENDATION_COST, reps)
        creation_time = get_avg_measure_value(data, constants.MEASURE_INDEX_CREATION_COST, reps)
        elapsed_time = get_avg_measure_value(data, constants.MEASURE_QUERY_EXECUTION_COST, reps)
        analytical_time = get_avg_measure_value(data, constants.MEASURE_ANALYTICAL_EXECUTION_COST, reps)
        transactional_time = get_avg_measure_value(data, constants.MEASURE_TRANSACTIONAL_EXECUTION_COST, reps)
        total_workload_time = get_avg_measure_value(data, constants.MEASURE_BATCH_TIME, reps) + hyp_batch_time

        # Adding to the final data frame
        final_df.loc[len(final_df)] = [component, rounds, hyp_batch_time, recommend_time, creation_time, elapsed_time,
                                       analytical_time, transactional_time, total_workload_time]

    final_df.round(4).to_csv(get_experiment_folder_path(exp_id) + 'comparison_table.csv')


def get_avg_measure_value(data, measure_name, reps):
    return (data[data[constants.DF_COL_MEASURE_NAME] == measure_name][constants.DF_COL_MEASURE_VALUE].sum())/reps


def change_experiment(exp_id):
    """
    Programmatically change the experiment

    :param exp_id: id of the new experiment
    """
    exp_config = configparser.ConfigParser()
    exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)
    exp_config['general']['run_experiment'] = exp_id
    with open(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG, 'w') as configfile:
        exp_config.write(configfile)


def log_configs(logging, module):
    for variable in dir(module):
        if not variable.startswith('__'):
            logging.info(str(variable) + ': ' + str(getattr(module, variable)))


def strip_namespace(xml_string):
    """
    Return the Element tree while striping the namespace
    :param xml_string: String
    :return: xml.etree.ElementTree object
    """
    it = ET.iterparse(StringIO(xml_string))
    for _, el in it:
        prefix, has_namespace, postfix = el.tag.partition('}')
        if has_namespace:
            el.tag = postfix  # strip all namespaces
    root = it.root
    return root


def xml_to_obj(element):
    """
    Return a python object for a given xml ElementTree
    :param element: xml.etree.ElementTree object
    :return: python object
    """
    name = element.tag

    py_type = type(name, (object,), {})
    py_obj = py_type()

    for attr in element.attrib.keys():
        setattr(py_obj, attr, element.get(attr))

    if element.text and element.text != '' and element.text != ' ' and element.text != '\n':
        setattr(py_obj, 'text', element.text)

    for cn in element:
        if not hasattr(py_obj, cn.tag):
            setattr(py_obj, cn.tag, xml_to_obj(cn))
        else:
            temp = getattr(py_obj, cn.tag)
            if type(temp) != list:
                setattr(py_obj, cn.tag, [])
                getattr(py_obj, cn.tag).append(temp)
            getattr(py_obj, cn.tag).append(xml_to_obj(cn))

    return py_obj


def log_query_xmls(experiment_id, query_objs, query_plans, round_num, component):
    for i in range(len(query_objs)):
        q_id = query_objs[i].id
        log_query_xml(experiment_id, q_id, query_plans[i].xml, round_num, component)


def log_query_xml(experiment_id, q_id, query_xml, round_num, component):
    xml_folder = get_experiment_folder_path(experiment_id) + 'xml' + '\\'
    if not os.path.exists(xml_folder):
        os.makedirs(xml_folder)
    file_name = component + '_' + str(round_num) + '_' + str(q_id) + '.' + constants.XML_FORMAT
    file_path = xml_folder + file_name
    with open(file_path, 'w') as f:
        f.write(query_xml)


def pretty_print(obj, indent=0):
    for k, v in obj.__dict__.items():
        if isinstance(v, list):
            for i in v:
                if '__dict__' in dir(i):
                    print(' ' * indent + k + ': ')
                    pretty_print(i, indent+4)
                else:
                    print(' ' * indent + k + ': ' + str(i))
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if '__dict__' in dir(v2):
                    print(' ' * indent + str(k) + ': ')
                    print(' ' * (indent + 4) + str(k2) + ': ')
                    pretty_print(v2, indent+4)
                else:
                    print(' ' * indent + str(k2) + ': ' + str(v2))
        else:
            if '__dict__' in dir(v) or isinstance(v, dict):
                pretty_print(v, indent+4)
            else:
                print(' ' * indent + k + ': ' + str(v))
