-- Databricks notebook source
--test test
SET test.nrows = 1000000;
SET test.npartitions = 512;
SELECT * FROM RANGE(0, ${test.nrows}, 1, ${test.npartitions})  -- start, end, step, numPartitions

-- COMMAND ----------

select * from us500 limit 20;

-- COMMAND ----------

select distinct(PROVIDER),BORO from nyc_free_wifi

-- COMMAND ----------

select count(rating) from guido.amazon20k group by rating;