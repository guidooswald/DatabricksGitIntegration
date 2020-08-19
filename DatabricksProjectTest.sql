-- Databricks notebook source
SET test.nrows = 1000000;
SET test.npartitions = 512;
SELECT * FROM RANGE(0, ${test.nrows}, 1, ${test.npartitions})  -- start, end, step, numPartitions

-- COMMAND ----------

select * from books