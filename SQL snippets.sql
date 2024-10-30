-- Databricks notebook source
CREATE TABLE guido.demo.g_patient_d (
  patient_sk BIGINT GENERATED ALWAYS AS IDENTITY COMMENT 'Primary Key (ID)',
  last_name STRING NOT NULL COMMENT 'Last name of the person',
  gender STRUCT<cd:STRING, desc:STRING> COMMENT 'Patient gender',
  birth_date TIMESTAMP COMMENT 'Birth date and time',
  ethnicity STRUCT<cd:STRING, desc:STRING> COMMENT 'Code for ethnicity',
  languages_spoken ARRAY<STRUCT<cd:STRING, desc:STRING>> COMMENT 'Ordered list of known languages (first = preferred)',
  patient_contact ARRAY<STRUCT<contact_info_type:STRING, contact_info:STRING, preferred_flag:BOOLEAN>> COMMENT 'Contact information',
  patient_mrn STRING COMMENT 'Patient medical record number',
  other_identifiers MAP<STRING, STRING> COMMENT 'Identifier type (passport number, license number except mrn, ssn) and value',
  uda MAP<STRING, STRING> COMMENT 'User Defined Attributes',
  source_ref STRING COMMENT 'Unique reference to the source record',
  effective_start_date TIMESTAMP COMMENT 'SCD2 effective start date for version',
  effective_end_date TIMESTAMP COMMENT 'SCD2 effective end date for version',
  g_process_id STRING COMMENT 'Process ID for record inserted',

  CONSTRAINT g_patient_d_pk PRIMARY KEY(patient_sk)
)
COMMENT 'Patient dimension'
CLUSTER BY (last_name, gender.cd, birth_date, patient_mrn)
TBLPROPERTIES (
  'delta.deletedFileRetentionDuration' = 'interval 30 days',
  'delta.columnMapping.mode' = 'name',
  'delta.dataSkippingStatsColumns' = 'patient_sk,last_name,gender',
  'delta.enableChangeDataFeed' = false
);
