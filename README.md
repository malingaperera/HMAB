
# HMAB : Self-Driving Hierarchy of Bandits for Integrated Physical Database Design Tuning  
  
## Steps:

 1. **Setup the Database** - In this example we use 10GB TPC-DS http://www.tpc.org/tpcds/
 2. **Set up the DB config file** (config/db.conf). We have implemented the MSSQL DB layer, for any other DB you have to implement it. Set the 'server'  and 'database'.

    [SYSTEM]  
    db_type = MSSQL  
      
    [MSSQL]  
    server = SERVER_NAME  
    database = TPCDS
    driver = {SQL Server}  
      
    [PG]  
    server = localhost  
    database = pgtpch_001  
    user = postgres  
    password = password

3. **Create the workload**. In our exmple we use generators in TPC-DS. Our solution does not provide query parsing. So we have used in-house scripts to parse the querys and generate the workload file (resources/workloads/ds_static_100.json) and query property file (resources/query_properties/tpc_ds.py).
4. **Create the experiment config** (config/db.conf)

    [example_tpc_ds]
    reps = 1  # How many repetitions\
    rounds = 25 # How many rounds in one repetition\
    hyp_rounds = 0 # ignore\
    workload_shifts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] # ignore\
    queries_start = [0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782, 1881, 1980, 2079, 2178, 2277, 2376, 2475, 2574, 2673, 2772, 2871]\ 
    queries_end = [99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782, 1881, 1980, 2079, 2178, 2277, 2376, 2475, 2574, 2673, 2772, 2871, 2970] # query start and end mark the workload for each round\
    ta_runs = [1] # PDTool invocation rounds\
    ta_workload = optimal # ignore\
    workload_file = \resources\workloads\ds_static_100.json\ 
    query_parser_file = resources.query_properties.tpc_ds\
    config_shifts = [0] # ignore\
    config_start = [0] # ignore\
    config_end = [0] # ignore\
    max_memory = 25000 # memory budget in MB\
    input_alpha = 1 # exploration boost factor\
    input_lambda = 0.5\
    time_weight = 5 # ignore\
    memory_weight = 0 # ignore\
    components = ["MAB", "TA_OPTIMAL", "NO_INDEX"] # baselines\
    mab_versions = ["simulation.sim_c3ucb_vF"] # ignore\
    pds_selection = VIEW_AND_INDICES # use VIEW_AND_INDICES or INDEX_ONLY

5. **Run experiment**. Set the experiment ID and run simulation/sim_run_experiment.py
6. **Experiment results and Graphs**. All experiment results can be found in the experiments folder, under a sub folder by the name of your experiment (example_tpc_ds)
7. **CSV Export**. A complete result export as a CSV (including component-wise breakdown per round) can be generated running shared/generate_csv_files.py
