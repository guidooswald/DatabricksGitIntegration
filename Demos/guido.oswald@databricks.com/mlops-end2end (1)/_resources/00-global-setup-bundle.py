# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # Technical Setup notebook. Hide this cell results
# MAGIC Initialize dataset to the current user and cleanup data when reset_all_data is set to true
# MAGIC 
# MAGIC Do not edit

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text("db_prefix", "retail", "Database prefix")
dbutils.widgets.text("min_dbr_version", "9.1", "Min required DBR version")

# COMMAND ----------

from delta.tables import *
import pandas as pd
import logging
from pyspark.sql.functions import to_date, col, regexp_extract, rand, to_timestamp, initcap, sha1
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType, input_file_name
import re


# VERIFY DATABRICKS VERSION COMPATIBILITY ----------

try:
  min_required_version = dbutils.widgets.get("min_dbr_version")
except:
  min_required_version = "9.1"

version_tag = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
version_search = re.search('^([0-9]*\.[0-9]*)', version_tag)
assert version_search, f"The Databricks version can't be extracted from {version_tag}, shouldn't happen, please correct the regex"
current_version = float(version_search.group(1))
assert float(current_version) >= float(min_required_version), f'The Databricks version of the cluster must be >= {min_required_version}. Current version detected: {current_version}'
assert "ml" in version_tag.lower(), f"The Databricks ML runtime must be used. Current version detected doesn't contain 'ml': {version_tag} "

import mlflow
import mlflow.sklearn

#force the experiment to the demos. Required to launch as a batch
def init_experiment_for_batch(demo_name, experiment_name):
  pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
  url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
  import requests
  xp_root_path = f"/demos/experiments/{demo_name}"
  requests.post(f"{url}/api/2.0/workspace/mkdirs", headers = {"Accept": "application/json", "Authorization": f"Bearer {pat_token}"}, json={ "path": xp_root_path})
  xp = f"{xp_root_path}/{experiment_name}"
  print(f"Using common experiment under {xp}")
  mlflow.set_experiment(xp)
  return mlflow.get_experiment_by_name(xp)

def get_cloud_name():
  return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()

# COMMAND ----------

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

db_prefix = dbutils.widgets.get("db_prefix")

dbName = db_prefix+"_"+current_user_no_at
cloud_storage_path = f"/Users/{current_user}/demos/{db_prefix}"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)

spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

print("using cloud_storage_path {}".format(cloud_storage_path))

# COMMAND ----------

def display_slide(slide_id, slide_number):
  displayHTML(f'''
  <div style="width:1150px; margin:auto">
  <iframe
    src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}"
    frameborder="0"
    width="1150"
    height="683"
  ></iframe></div>
  ''')

# COMMAND ----------

import time
def get_active_streams(start_with = ""):
    return [s for s in spark.streams.active if len(start_with) == 0 or (s.name is not None and s.name.startswith(start_with))]
  
# Function to stop all streaming queries 
def stop_all_streams(start_with = ""):
  streams = get_active_streams(start_with)
  if len(streams) > 0:
    print(f"Stopping {stream_count} streams")
    for s in streams:
        try:
            s.stop()
        except:
            pass
    print(f"All stream stopped (starting with: {start_with}.")

    
def wait_for_all_stream(start_with = ""):
  actives = get_active_streams(start_with)
  if len(actives) > 0:
    print(f"{len(actives)} streams still active, waiting... ({[s.name for s in actives]})")
  while len(actives) > 0:
    spark.streams.awaitAnyTermination()
    time.sleep(1)
    actives = get_active_streams(start_with)
  print("All streams completed.")
  
#Return true if the folder is empty or does not exists
def is_folder_empty(folder):
  try:
    return len(dbutils.fs.ls(folder)) == 0
  except:
    return True

# COMMAND ----------

from pyspark.sql.functions import col
import mlflow

import databricks
from datetime import datetime

def get_automl_run(name):
  #get the most recent automl run
  df = spark.table("hive_metastore.dbdemos_metadata.automl_experiment").filter(col("name") == name).orderBy(col("date").desc()).limit(1)
  return df.collect()

#Get the automl run information from the hive_metastore.dbdemos_metadata.automl_experiment table. 
#If it's not available in the metadata table, start a new run with the given parameters
def get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes, move_to_production = False):
  spark.sql("create database if not exists hive_metastore.dbdemos_metadata")
  spark.sql("create table if not exists hive_metastore.dbdemos_metadata.automl_experiment (name string, date string)")
  result = get_automl_run(name)
  if len(result) == 0:
    print("No run available, start a new Auto ML run, this will take a few minutes...")
    start_automl_run(name, model_name, dataset, target_col, timeout_minutes, move_to_production)
    return (False, get_automl_run(name))
  return (True, result[0])


#Start a new auto ml classification task and save it as metadata.
def start_automl_run(name, model_name, dataset, target_col, timeout_minutes = 5, move_to_production = False):
  from databricks import automl
  automl_run = databricks.automl.classify(
    dataset = dataset,
    target_col = target_col,
    timeout_minutes = timeout_minutes
  )
  experiment_id = automl_run.experiment.experiment_id
  path = automl_run.experiment.name
  data_run_id = mlflow.search_runs(experiment_ids=[automl_run.experiment.experiment_id], filter_string = "tags.mlflow.source.name='Notebook: DataExploration'").iloc[0].run_id
  exploration_notebook_id = automl_run.experiment.tags["_databricks_automl.exploration_notebook_id"]
  best_trial_notebook_id = automl_run.experiment.tags["_databricks_automl.best_trial_notebook_id"]

  cols = ["name", "date", "experiment_id", "experiment_path", "data_run_id", "best_trial_run_id", "exploration_notebook_id", "best_trial_notebook_id"]
  spark.createDataFrame(data=[(name, datetime.today().isoformat(), experiment_id, path, data_run_id, automl_run.best_trial.mlflow_run_id, exploration_notebook_id, best_trial_notebook_id)], schema = cols).write.mode("append").option("mergeSchema", "true").saveAsTable("hive_metastore.dbdemos_metadata.automl_experiment")
  #Create & save the first model version in the MLFlow repo (required to setup hooks etc)
  model_registered = mlflow.register_model(f"runs:/{automl_run.best_trial.mlflow_run_id}/model", model_name)
  set_experiment_permission(path)
  if move_to_production:
    client = mlflow.tracking.MlflowClient()
    print("registering model version "+model_registered.version+" as production model")
    client.transition_model_version_stage(name = model_name, version = model_registered.version, stage = "Production", archive_existing_versions=True)
  return get_automl_run(name)

#Generate nice link for the given auto ml run
def display_automl_link(name, model_name, dataset, target_col, timeout_minutes = 5, move_to_production = False):
  from_cache, r = get_automl_run_or_start(name, model_name, dataset, target_col, timeout_minutes, move_to_production)
  if from_cache:
    html = f"""For exploratory data analysis, open the <a href="/#notebook/{r["exploration_notebook_id"]}">data exploration notebook</a><br/><br/>"""
    html += f"""To view the best performing model, open the <a href="/#notebook/{r["best_trial_notebook_id"]}">best trial notebook</a><br/><br/>"""
    html += f"""To view details about all trials, navigate to the <a href="/#mlflow/experiments/{r["experiment_id"]}/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false">MLflow experiment</>"""
    displayHTML(html)

def reset_automl_run(model_name):
  spark.sql(f"delete from hive_metastore.dbdemos_metadata.automl_experiment where name='{model_name}'")

#Once the automl experiment is created, we assign CAN MANAGE to all users as it's shared in the workspace
def set_experiment_permission(experiment_path):
  url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply("api_url")
  import requests
  pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
  headers =  {"Authorization": "Bearer " + pat_token, 'Content-type': 'application/json'}
  status = requests.get(url+"/api/2.0/workspace/get-status", params = {"path": experiment_path}, headers=headers).json()
  #Set can manage to all users to the experiment we created as it's shared among all
  params = {"access_control_list": [{"group_name": "users","permission_level": "CAN_MANAGE"}]}
  permissions = requests.patch(f"{url}/api/2.0/permissions/experiments/{status['object_id']}", json = params, headers=headers)
  if permissions.status_code != 200:
    print("ERROR: couldn't set permission to all users to the autoML experiment")

  #try to find the experiment id
  result = re.search(r"_([a-f0-9]{8}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{12})_", experiment_path)
  if result is not None and len(result.groups()) > 0:
    ex_id = result.group(0)
  else:
    print(f"WARN: couldn't get tje experiment id from path: {experiment_path}")
    ex_id = experiment_path[experiment_path.rfind('/')+1:]

  path = experiment_path
  path = path[:path.rfind('/')]+"/"
  #List to get the folder with the notebooks from the experiment
  folders = requests.get(url+"/api/2.0/workspace/list", params = {"path": path}, headers=headers).json()
  for f in folders['objects']:
    if f['object_type'] == 'DIRECTORY' and ex_id in f['path']:
        #Set the permission of the experiment notebooks to all
        permissions = requests.patch(f"{url}/api/2.0/permissions/directories/{f['object_id']}", json = params, headers=headers)
        if permissions.status_code != 200:
          print("ERROR: couldn't set permission to all users to the autoML experiment notebooks")

# COMMAND ----------

#Remove warnings
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', UserWarning)
    # simulate import of module giving SyntaxWarning
    warnings.warn('bad', SyntaxWarning)
