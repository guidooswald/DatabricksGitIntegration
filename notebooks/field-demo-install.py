# Databricks notebook source
# %pip install https://github.com/databricks-demos/dbdemos/raw/main/release/dbdemos-0.1-py3-none-any.whl --force

# COMMAND ----------

# MAGIC %pip install dbdemos

# COMMAND ----------

import dbdemos
dbdemos.list_demos()

# COMMAND ----------

dbdemos.install('delta-lake')

# COMMAND ----------

dbdemos.install('uc-01-acl')

# COMMAND ----------

dbdemos.install('uc-02-external-location')

# COMMAND ----------

dbdemos.install('uc-03-data-lineage')

# COMMAND ----------

dbdemos.install('uc-04-audit-log')

# COMMAND ----------

dbdemos.install('uc-05-upgrade')

# COMMAND ----------

dbdemos.install('dlt-cdc')

# COMMAND ----------

dbdemos.install('mlops-end2end', overwrite=True)

# COMMAND ----------

dbdemos.install('lakehouse-retail-c360')

# COMMAND ----------

dbdemos.install('streaming-sessionization')

# COMMAND ----------

dbdemos.install('identity-pk-fk')

# COMMAND ----------

dbdemos.install('llm-dolly-chatbot')
