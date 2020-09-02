# Databricks notebook source
path = "/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2009-01.csv.gz"

logDF = (spark
  .read
  .option("header", True)
  .csv(path)
)

display(logDF)

# COMMAND ----------

# MAGIC %scala
# MAGIC val path = "/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2009-02.csv.gz"
# MAGIC 
# MAGIC val logDF = spark
# MAGIC   .read
# MAGIC   .option("header", true)
# MAGIC   .csv(path)
# MAGIC   .sample(withReplacement=false, fraction=0.3, seed=3) // using a sample to reduce data size
# MAGIC 
# MAGIC display(logDF)