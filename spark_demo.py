from pyspark import SparkConf, SparkContext

# sc = SparkContext(master = "local", appName="Spark Demo")
# print(sc.textFile("C:\\deckofcards.txt").first())

sc = SparkContext(master = 'local', appName="maps_and_lazy_evaluation_example")

log_of_songs = [
    "Despacito",
    "Nice for what",
    "No tears left to cry",
    "Despacito",
    "Havana",
    "In my feelings",
    "Nice for what",
    "Despacito",
    "All the stars"
]

# parallelize the log_of_songs to use with Spark
distributed_song_log = sc.parallelize(log_of_songs)

def convert_song_to_lowercase(song):
    return song.lower()

# lazy evaluation. Spark does not actually execute the map unless it needs to
distributed_song_log.map(convert_song_to_lowercase)

# To get Spark to actually run the map step, you need to use an "action".
# One available action is the collect method. The collect() method takes the results
# from all of the clusters and "collects" them into a single list on the master node.
distributed_song_log.map(convert_song_to_lowercase).collect()

# Spark never manipulates the original dataset.It just manipulates the copy.
distributed_song_log.collect()

distributed_song_log.map(lambda x: x.lower()).collect()

