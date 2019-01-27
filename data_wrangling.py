from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, desc, asc
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.types import StringType, IntegerType

import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# We are creating a spark session and giving a name for the app. The app name could be anything
spark = SparkSession.builder.appName('data wrangling').getOrCreate()

data_dir = 'C:/Users/John/PycharmProjects/sparkdemo/data/'
# Reading in the JSON file as a dataframe
user_log = spark.read.json(data_dir + 'sparkify_log_small.json')

# Taking a look at the first 5 records
user_log.take(5)

# To easily see the columns in the data frame we can use printschema
user_log.printSchema()

# This command only shows the column name and its data type
user_log.describe()

# chaining a show() method along with describe() will give show the actual statistic
user_log.describe().show()

# To see for a single column
user_log.describe('artist').show()
user_log.describe('sessionID').show()

# to check how many rows are in the dataframe, we can use count() method
user_log.count()

user_log.select("length").show()

user_log.select('page').drop_duplicates().sort('page').show()

user_log.take(1)

user_log.select(['userId', 'firstName', 'page', 'song']).where(user_log.userId == '1046').collect()

get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x/1000.0).hour)

user_log = user_log.withColumn('hour',get_hour(user_log.ts))

user_log.head()

songs_in_hour = user_log.filter(user_log.page == 'NextSong').groupby(user_log.hour).count().orderBy(user_log.hour.cast('float'))

songs_in_hour.show()

songs_in_hour_pd = songs_in_hour.toPandas()

plt.scatter(songs_in_hour_pd['hour'], songs_in_hour_pd['count']);
plt.xlim(-1, 24);
plt.ylim(0, 1.2 * max(songs_in_hour_pd['count']))
plt.xlabel('Hour')
plt.ylabel('Songs played')


user_log_valid = user_log.dropna(how = 'any', subset = ['userId', 'sessionId'])

user_log.select('userId').drop_duplicates().sort('userId').show()

# user_log_valid = user_log_valid.where(user_log_valid.userId != "")
user_log_valid = user_log_valid.filter(user_log_valid['userId'] != "")
user_log_valid.count()

# user_log_valid.filter(user_log_valid.page == 'Submit Downgrade').show()
user_log_valid.filter("page == 'Submit Downgrade'").show()

user_log_valid.select(["userId","firstname","page", "level", "song"]).filter("userId == 1138").collect()

flag_downgrade_event = udf(lambda x: 1 if x == 'Submit Downgrade' else 0)

user_log_valid = user_log_valid.withColumn('downgraded', flag_downgrade_event('page'))

user_log_valid.head()

# user_log_valid.select(["userId","firstname","page", "level", "song", "ts"]).show()

from pyspark.sql import Window
windowval = Window.partitionBy('userId').orderBy(desc('ts')).rangeBetween(Window.unboundedPreceding, 0)

user_log_valid = user_log_valid.withColumn('phase', Fsum('downgraded').over(windowval))

user_log_valid.select(["userId","firstname", "ts", "page", "level", "phase"]).where(user_log.userId == "1138").sort("ts").collect()

