# Databricks notebook source
# MAGIC %md # Running Code

# COMMAND ----------

# MAGIC %md First and foremost, the Jupyter Notebook is an interactive environment for writing and running code. The notebook is capable of running code in a wide range of languages. However, each notebook is associated with a single kernel.  This notebook is associated with the IPython kernel, therefore runs Python code.

# COMMAND ----------

# MAGIC %md ## Code cells allow you to enter and run code

# COMMAND ----------

# MAGIC %md Run a code cell using `Shift-Enter` or pressing the <button class='btn btn-default btn-xs'><i class="icon-step-forward fa fa-play"></i></button> button in the toolbar above:

# COMMAND ----------

a = 10

# COMMAND ----------

print(a)

# COMMAND ----------

# MAGIC %md There are two other keyboard shortcuts for running code:
# MAGIC
# MAGIC * `Alt-Enter` runs the current cell and inserts a new one below.
# MAGIC * `Ctrl-Enter` run the current cell and enters command mode.

# COMMAND ----------

# MAGIC %md ## Managing the Kernel

# COMMAND ----------

# MAGIC %md Code is run in a separate process called the Kernel.  The Kernel can be interrupted or restarted.  Try running the following cell and then hit the <button class='btn btn-default btn-xs'><i class='icon-stop fa fa-stop'></i></button> button in the toolbar above.

# COMMAND ----------

import time
time.sleep(10)

# COMMAND ----------

# MAGIC %md If the Kernel dies you will be prompted to restart it. Here we call the low-level system libc.time routine with the wrong argument via
# MAGIC ctypes to segfault the Python interpreter:

# COMMAND ----------

import sys
from ctypes import CDLL
# This will crash a Linux or Mac system
# equivalent calls can be made on Windows

# Uncomment these lines if you would like to see the segfault

dll = 'dylib' if sys.platform == 'darwin' else 'so.6'
libc = CDLL("libc.%s" % dll) 
libc.time(-1)  # BOOM!!

# COMMAND ----------

# MAGIC %md ## Cell menu

# COMMAND ----------

# MAGIC %md The "Cell" menu has a number of menu items for running code in different ways. These includes:
# MAGIC
# MAGIC * Run and Select Below
# MAGIC * Run and Insert Below
# MAGIC * Run All
# MAGIC * Run All Above
# MAGIC * Run All Below

# COMMAND ----------

# MAGIC %md ## Restarting the kernels

# COMMAND ----------

# MAGIC %md The kernel maintains the state of a notebook's computations. You can reset this state by restarting the kernel. This is done by clicking on the <button class='btn btn-default btn-xs'><i class='fa fa-repeat icon-repeat'></i></button> in the toolbar above.

# COMMAND ----------

# MAGIC %md ## sys.stdout and sys.stderr

# COMMAND ----------

# MAGIC %md The stdout and stderr streams are displayed as text in the output area.

# COMMAND ----------

print("hi, stdout")

# COMMAND ----------

from __future__ import print_function
import sys
print('hi, stderr', file=sys.stderr)

# COMMAND ----------

# MAGIC %md ## Output is asynchronous

# COMMAND ----------

# MAGIC %md All output is displayed asynchronously as it is generated in the Kernel. If you execute the next cell, you will see the output one piece at a time, not all at the end.

# COMMAND ----------

import time, sys
for i in range(8):
    print(i)
    time.sleep(0.5)

# COMMAND ----------

# MAGIC %md ## Large outputs

# COMMAND ----------

# MAGIC %md To better handle large outputs, the output area can be collapsed. Run the following cell and then single- or double- click on the active area to the left of the output:

# COMMAND ----------

for i in range(50):
    print(i)

# COMMAND ----------

# MAGIC %md Beyond a certain point, output will scroll automatically:

# COMMAND ----------

for i in range(500):
    print(2**i - 1)
