# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `demo-mlops-end2end-guido_oswald` from the dropdown menu ([open cluster configuration](https://adb-984752964297111.11.azuredatabricks.net/#setting/clusters/0309-121348-fapspjed/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('mlops-end2end')` or re-install the demo: `dbdemos.install('mlops-end2end')`*

# COMMAND ----------

dbutils.widgets.dropdown("force_refresh_automl", "false", ["false", "true"], "Restart AutoML run")

# COMMAND ----------

# MAGIC %md
# MAGIC # Churn Prediction Feature Engineering
# MAGIC Our first step is to analyze the data and build the features we'll use to train our model. Let's see how this can be done.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-1.png" width="1200">
# MAGIC 
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=984752964297111&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F01_feature_engineering&uid=6066264009754637">
# MAGIC <!-- [metadata={"description":"MLOps end2end workflow: Feature engineering",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["feature store"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# Read into Spark
telcoDF = spark.table("churn_bronze_customers")
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Koalas
# MAGIC 
# MAGIC Because our Data Scientist team is familiar with Pandas, we'll use `koalas` to scale `pandas` code. The Pandas instructions will be converted in the spark engine under the hood and distributed at scale.
# MAGIC 
# MAGIC *Note: Starting from `spark 3.2`, koalas is builtin and we can get an Pandas Dataframe using `to_pandas_on_spark`.*

# COMMAND ----------

# DBTITLE 1,Define featurization function
from databricks.feature_store import feature_table
import pyspark.pandas as ps

def compute_churn_features(data):
  
  # Convert to koalas
  data = data.pandas_api()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phone_service', 'multiple_lines', 'internet_service',
                                 'online_security', 'online_backup', 'device_protection',
                                 'tech_support', 'streaming_tv', 'streaming_movies',
                                 'contract', 'paperless_billing', 'payment_method'], dtype = 'int64')
  
  # Convert label to int and rename column
  data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})
  data = data.astype({'churn': 'int32'})
  
  # Clean up column names
  data.columns = [re.sub(r'[\(\)]', ' ', name).lower() for name in data.columns]
  data.columns = [re.sub(r'[ -]', '_', name).lower() for name in data.columns]

  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Write to Feature Store (Optional)
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC 
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC 
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC 
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC 
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

churn_features_df = compute_churn_features(telcoDF)

try:
  #drop table if exists
  fs.drop_table(f'{dbName}.churn_features')
except:
  pass
#Note: You might need to delete the FS table using the UI
churn_feature_table = fs.create_table(
  name=f'{dbName}.churn_features',
  primary_keys='customer_id',
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the churn_bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=churn_features_df.to_spark(), name=f'{dbName}.churn_features', mode='overwrite')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC 
# MAGIC Databricks simplify model creation and MLOps. However, bootstraping new ML projects can still be long and inefficient. 
# MAGIC 
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC 
# MAGIC 
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC 
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC 
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC 
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC 
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`churn_features`)
# MAGIC 
# MAGIC Our prediction target is the `churn` column.
# MAGIC 
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC 
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

# DBTITLE 1,We have already started a run for you, you can explore it here:
force_refresh = dbutils.widgets.get("force_refresh_automl") == "true"
display_automl_churn_link(force_refresh = force_refresh)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Using the generated notebook to build our model
# MAGIC 
# MAGIC Next step: [Explore the generated Auto-ML notebook]($./02_automl_baseline)
