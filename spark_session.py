from pyspark import SparkContext, SparkConf

# SparkContext connects the application with the clusters. It is the main entry point of all spark functionalities.

#configure = SparkConf().setAppName('name').setMaster('IP Address')

configure = SparkConf().setAppName('name').setMaster('local')

sc = SparkContext(conf = configure)

# To read dataframes
from pyspark.sql import SparkSession

SparkContext().stop()