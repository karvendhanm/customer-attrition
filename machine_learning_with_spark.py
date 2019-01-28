from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, avg, col, concat, desc, explode, lit, min, max, split
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import re


data_dir = 'C:/Users/John/PycharmProjects/customer-attrition/data/'

spark = SparkSession.builder.appName("spark machine learning").getOrCreate()

df = spark.read.json(data_dir + 'Train_onetag_small.json')
df.persist()
#df.describe() # TODO have to understand why df.describe() and df.persist() give different datatypes for the same column.
df.head()


regexTokenizer = RegexTokenizer(inputCol='Body', outputCol='words', pattern="\\W")
df = regexTokenizer.transform(df)
df.head()

body_length = udf(lambda x: len(x), IntegerType())
df = df.withColumn('BodyLength', body_length('words'))
df.head()

number_of_paragraphs = udf(lambda x: len(re.findall('</p>',x)), IntegerType())
number_of_links = udf(lambda x: len(re.findall('</a>',x)), IntegerType())

df = df.withColumn('NumParagraphs', number_of_paragraphs('Body'))
df = df.withColumn('NumLinks', number_of_links('Body'))
df.head()

assembler = VectorAssembler(inputCols = ['BodyLength', 'NumParagraphs', 'NumLinks'], outputCol = "NumFeatures")
df = assembler.transform(df)
df.head()

scaler = Normalizer(inputCol = 'NumFeatures', outputCol ='ScaledNumFeatures' )
df = scaler.transform(df)
df.head(2)

scaler2 = StandardScaler(inputCol = 'NumFeatures', outputCol ='ScaledNumFeatures2', withStd = True)
scalerModel = scaler2.fit(df)
df = scalerModel.transform(df)
df.head(2)

## Text Preprocessing

cv = CountVectorizer(inputCol='words', outputCol='TF', vocabSize = 1000)
cvmodel = cv.fit(df)
df = cvmodel.transform(df)
df.take(2)

cvmodel.vocabulary

# seeing the least common words
cvmodel.vocabulary[::-1] # the list invertes

cvmodel.vocabulary[-10:]

idf = IDF(inputCol = 'TF', outputCol = 'TFIDF')
idfmodel = idf.fit(df)
df = idfmodel.transform(df)
df.head()

indexer = StringIndexer(inputCol = 'oneTag', outputCol = 'label')
df = indexer.fit(df).transform(df)
df.head()

df = df.withColumn('Joined_Body_Title', concat(col('Body'),lit(" "), col('Title')))

regexTokenizer = RegexTokenizer(inputCol='Joined_Body_Title', outputCol='Joined_Body_Title_words', pattern="\\W")
df = regexTokenizer.transform(df)
df.head()

body_length = udf(lambda x: len(x), IntegerType())
df = df.withColumn('Length_Body_Title', body_length('Joined_Body_Title_words'))

assembler1 = VectorAssembler(inputCols = ['Length_Body_Title'], outputCol = "NumFeatures_Body_Title")
df = assembler1.transform(df)

scaler3 = Normalizer(inputCol = 'NumFeatures_Body_Title', outputCol ='ScaledNumFeatures_Body_Title' )
df = scaler3.transform(df)

scaler4 = StandardScaler(inputCol = 'NumFeatures_Body_Title', outputCol ='ScaledNumFeatures_Body_Title3', withStd = True, withMean = True)
scalerModel1 = scaler4.fit(df)
df = scalerModel1.transform(df)

minmaxscaler = MinMaxScaler(inputCol = 'NumFeatures_Body_Title', outputCol ='ScaledMinMax')
minmaxscaler_Model = minmaxscaler.fit(df)
df = minmaxscaler_Model.transform(df)









