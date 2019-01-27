from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# creating a spark session
spark = SparkSession.builder.appName("Data wrangling with Spark SQL").getOrCreate()

# setting the directory for the data
data_dir = 'C:/Users/John/PycharmProjects/customer-attrition/data/'

# Reading in the JSON file as a spark dataframe
df = spark.read.json(data_dir + 'sparkify_log_small.json')

df.take(1)

df.printSchema()

df.createOrReplaceTempView('df_table')

spark.sql('SELECT * FROM df_table LIMIT 20').show()

spark.sql('''
SELECT * 
FROM df_table 
LIMIT 2
''').show()

spark.sql('''
    SELECT DISTINCT page
    FROM df_table
    ORDER BY page ASC 
''').show()

# user defined function has to be registered in spark sql
spark.udf.register("get_hour", lambda x: datetime.datetime.fromtimestamp(x/1000.0).hour, IntegerType())

spark.sql('''
    SELECT *, get_hour(ts) AS hour
    FROM df_table
    LIMIT 1    
''').show()

df.show(20)

spark.sql('''
    SELECT get_hour(ts) as hour, COUNT(userId)
    FROM df_table
    WHERE page = 'NextSong'
    GROUP BY hour
    ORDER BY hour ASC
''').show()

songs_in_hour = spark.sql('''
    SELECT get_hour(ts) as hour, COUNT(*) as plays_per_hour
    FROM df_table
    WHERE page = 'NextSong'
    GROUP BY hour
    ORDER BY cast(hour as int) ASC    
''')

songs_in_hour.show()

songs_in_hour = spark.sql('''
    SELECT get_hour(T.ts) as hour, COUNT(*) as plays_per_hour
    FROM df_table as T
    WHERE T.page = 'NextSong'
    GROUP BY hour
    ORDER BY cast(hour as int) ASC
''')

songs_in_hour.show()

songs_in_hour_pd = songs_in_hour.toPandas()
print(songs_in_hour_pd)

spark.sql('''
    SELECT *
    FROM (SELECT DISTINCT page
    FROM df_table
    WHERE userId = "") AS blank_pages
    RIGHT JOIN (SELECT DISTINCT page
    FROM df_table) AS all_pages
    ON blank_pages.page = all_pages.page
    WHERE blank_pages.page IS NULL
''').show()















