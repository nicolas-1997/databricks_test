# Databricks notebook source
# MAGIC %md
# MAGIC #### Procesamiento de datos y EDA

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

raw_covid = spark.read.csv("/mnt/coviddl1/raw/covid_population.csv", sep=',', header=True)

# COMMAND ----------

raw_covid.head(12)

# COMMAND ----------

from pyspark.sql.functions import *
raw_covid_sel = raw_covid.select(col("date"),
                                 col("state"),
                                 col("place_type").alias("place"),
                                 col("deaths"),
                                 col("estimated_population_2019").alias("population"),
                                 col("death_rate"))

raw_covid_sel.show()

# COMMAND ----------

raw_covid.createOrReplaceTempView("covid")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM covid

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE covid

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date, state, deaths FROM covid WHERE state in ("MG", "RJ", "SP") and place_type = "state"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT state, deaths FROM covid WHERE place_type = "state" AND is_last = "True" ORDER BY deaths DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date, deaths FROM covid WHERE state = "MG" and place_type = "state" ORDER BY date DESC

# COMMAND ----------

query = """
  SELECT date, deaths FROM covid WHERE state = "MG" and place_type = "state" ORDER BY date DESC
"""

df = spark.sql(query)
df = df.toPandas()

# COMMAND ----------

df_cov = spark.createDataFrame(df)

# COMMAND ----------

df_cov.write.format("com.databricks.spark.csv").option("header","true").option("delimiter", ",").mode("overwrite").save("/mnt/coviddl1/processed/population")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Machine Learning con Spark

# COMMAND ----------

import pandas as pd
import logging

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)


query = """
  SELECT string(date) as ds, int(deaths) as y FROM covid WHERE state = "MG" and place_type = "state" order by date
"""

df = spark.sql(query)
df = df.toPandas()
display(df)

# COMMAND ----------

from prophet import Prophet

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)

fig1 = m.plot(forecast)

# COMMAND ----------

forecast.head()

# COMMAND ----------

forecast = spark.createDataFrame(forecast)

# COMMAND ----------

forecast.createOrReplaceTempView("forecast")

output = spark.sql("SELECT ds,yhat FROM forecast")

# COMMAND ----------

output.write.format("com.databricks.spark.csv").option("header","true").option("delimiter", ",").mode("overwrite").save("/mnt/coviddl1/processed/population")
