from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('our first spark SQL example').getOrCreate()

spark.sparkContext.getConf().getAll()

data_dir = 'C:/Users/John/PycharmProjects/sparkdemo/data/'

path = data_dir + 'sparkify_log_small.json'

# loading the JSON file
user_log = spark.read.json(path)

user_log.printSchema()

user_log.describe()

user_log.show(n = 1)

user_log.take(5)

out_path = data_dir+'user_log_file.csv'

user_log.write.save(out_path, format = 'csv', header = True)

user_log_2 = spark.read.csv(out_path, header = True)

user_log_2.printSchema()

user_log_2.take(2)

