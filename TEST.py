# Databricks notebook source
# MAGIC %md
# MAGIC # Test Notebook (local)

# COMMAND ----------

# MAGIC %pip install ftplib pandas
# MAGIC
# MAGIC from ftplib import FTP
# MAGIC import pandas as pd
# MAGIC from io import StringIO
# MAGIC
# MAGIC # FTP server details
# MAGIC ftp_server = 'ftp.example.com'
# MAGIC ftp_user = 'username'
# MAGIC ftp_password = 'password'
# MAGIC file_path = '/path/to/your/file.csv'
# MAGIC
# MAGIC # Connect to the FTP server
# MAGIC ftp = FTP(ftp_server)
# MAGIC ftp.login(user=ftp_user, passwd=ftp_password)
# MAGIC
# MAGIC # Retrieve the CSV file
# MAGIC csv_data = StringIO()
# MAGIC ftp.retrbinary(f'RETR {file_path}', csv_data.write)
# MAGIC csv_data.seek(0)
# MAGIC
# MAGIC # Load the CSV data into a pandas DataFrame
# MAGIC df = pd.read_csv(csv_data)
# MAGIC
# MAGIC # Convert pandas DataFrame to Spark DataFrame
# MAGIC spark_df = spark.createDataFrame(df)
# MAGIC
# MAGIC # Write the Spark DataFrame to a Delta Lake table
# MAGIC spark_df.write.format("delta").mode("overwrite").save("/delta/table/path")
# MAGIC
# MAGIC # Display the Delta Lake table
# MAGIC display(spark.read.format("delta").load("/delta/table/path"))

# COMMAND ----------

1+1

# COMMAND ----------

# MAGIC %sh
# MAGIC databricks --version
# MAGIC python --version
# MAGIC pip --version
# MAGIC ls -al
# MAGIC whoami
# MAGIC pwd
# MAGIC cat /etc/os-release

# COMMAND ----------

1+1

# COMMAND ----------

import sklearn as sk

sk.__version__

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

#dbutils.fs.ls('dbfs:/mnt/data/export_models/')
dbutils.fs.ls('dbfs:/mnt/Users')

# COMMAND ----------

num_terms = 1000000
pi = 0
for i in range(num_terms):
    denominator = 2 * i + 1
    term = 4 / denominator * (-1) ** i
    pi += term

print(pi)

# COMMAND ----------

import math

pi = round(math.pi, 10)
print(pi)

# COMMAND ----------

# generate a random string of 10 characters
import uuid
uuid.uuid4().hex[:10]


# COMMAND ----------

spark.conf.set("spark.securitymanager.enabled", False)
display(spark.sql("show tables").filter("upper(tableName) like '%G%'").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## header 2

# COMMAND ----------

# MAGIC %sql
# MAGIC show catalogs like 'guido*';

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog guido;

# COMMAND ----------

# MAGIC %md
# MAGIC ## onother headline comes here...

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT current_catalog();

# COMMAND ----------

# MAGIC %sql
# MAGIC show databases;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from samples.nyctaxi.trips limit 50;

# COMMAND ----------

df = spark.table("samples.nyctaxi.trips").limit(10)
display(df)

# COMMAND ----------

# MAGIC %pip install -U sentence-transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# COMMAND ----------

#load pretrained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#the sentences to encode
sentences = [
  "This is a sentence",
  "This is another sentence",
  "This is a third sentence"
  ]

#calculate embeddinggs calling sentences.encode
embeddings = model.encode(sentences)
print(embeddings.shape)

# COMMAND ----------

from faker import Faker
import pandas as pd

# Initialize Faker
fake = Faker()

# Generate fake data for a CRM database
def generate_customers(n=1000):
    customers = [{
        "customer_id": i,
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "email": fake.email(),
        "phone": fake.phone_number()
    } for i in range(1, n + 1)]
    return pd.DataFrame(customers)

def generate_orders(n=300, customer_count=1000):
    orders = [{
        "order_id": i,
        "customer_id": fake.random_int(min=1, max=customer_count),
        "order_date": fake.date_this_year(),
        "amount": round(fake.random_number(digits=5), 2)
    } for i in range(1, n + 1)]
    return pd.DataFrame(orders)

def generate_interactions(n=500, customer_count=1000):
    interactions = [{
        "interaction_id": i,
        "customer_id": fake.random_int(min=1, max=customer_count),
        "interaction_date": fake.date_this_year(),
        "channel": fake.random_element(elements=("Email", "Phone", "In Person", "Social Media")),
        "notes": fake.text(max_nb_chars=200)
    } for i in range(1, n + 1)]
    return pd.DataFrame(interactions)

# Create DataFrames
customers_df = generate_customers()
orders_df = generate_orders()
interactions_df = generate_interactions()

# Convert DataFrames to Spark DataFrames
customers_sdf = spark.createDataFrame(customers_df)
orders_sdf = spark.createDataFrame(orders_df)
interactions_sdf = spark.createDataFrame(interactions_df)

# Display the Spark DataFrames
display(customers_sdf)
display(orders_sdf)
display(interactions_sdf)

# Save the Spark DataFrames to the guido.default database
customers_sdf.write.format("delta").mode("overwrite").saveAsTable("guido.default.customers")
orders_sdf.write.format("delta").mode("overwrite").saveAsTable("guido.default.orders")
interactions_sdf.write.format("delta").mode("overwrite").saveAsTable("guido.default.interactions")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd

# Load your model from MLflow
logged_model = mlflow.pyfunc.load_model(model_uri="models:/basic_rag_bot_Content_discovery/1")

examples =  {
    "request": [
      {
      # Recommended `messages` format
        "messages": [{
          "role": "user",
          "content": "Spark is a data analytics framework."
        }],
      },
      # Primitive string format
      # Note: Using a primitive string is discouraged. The string will be wrapped in the
      # OpenAI messages format before being passed to your agent.
      "How do I convert a Spark DataFrame to Pandas?"
    ],
    "response": [
        "Spark is a data analytics framework.",
        "This is not possible as Spark is not a panda.",
    ],
    "retrieved_context": [ # Optional, needed for judging groundedness.
        [{"doc_uri": "doc1.txt", "content": "In 2013, Spark, a data analytics framework, was open sourced by UC Berkeley's AMPLab."}],
        [{"doc_uri": "doc2.txt", "content": "To convert a Spark DataFrame to Pandas, you can use toPandas()"}],
    ],
    "expected_response": [ # Optional, needed for judging correctness.
        "Spark is a data analytics framework.",
        "To convert a Spark DataFrame to Pandas, you can use the toPandas() method.",
    ]
}

result = mlflow.evaluate(
    data=pd.DataFrame(examples),    # Your evaluation set
    model=logged_model.model_uri, # If you have an MLFlow model. `retrieved_context` and `response` will be obtained from calling the model.
    model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
)

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(result.tables['eval_results'])
