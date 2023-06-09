#
# Before running the docker container
# Please make sure to unzip the archive.zip
# You may need to edit the file path
#

# Set up Logging
import logging
from datetime import date, datetime
import sys
today = date.today()
logFilePath = 'logs/' + str(today) + '-pipeline.log'

'''
logging.basicConfig(filename=logFilePath, 
                    #filemode='w',
                    level=logging.INFO, 
                    format='%(asctime)s, %(name)s %(levelname)s: %(message)s')
'''

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create file handler and set level to INFO
file_handler = logging.FileHandler(logFilePath)
file_handler.setLevel(logging.INFO)

# Create stream handler and set level to INFO
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Set format for log messages
formatter = logging.Formatter('%(asctime)s, %(name)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#######################
#   Data Ingestion    #
#######################


import glob
# Get the lists of all CSV files in the directory
etf_list = glob.glob("etfs/*.csv")
#print(etf_list)
logger.info(f'A total of {len(etf_list)} ETFs found.')

stock_list = glob.glob("stocks/*.csv")
#print(stock_list)
logger.info(f'A total of {len(stock_list)} stocks found.')

from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name


# 
# Create a SparkSession
# 
spark = SparkSession.builder.appName("pipeline").getOrCreate()
logger.info('Loading data from CSV')
#
# Read all CSV files as a Spark DataFrame and add a new column with the file name
#
df = spark.read.csv(etf_list, header=True).withColumn("Symbol", input_file_name())

#
# Combine stocks and etfs data
#
df = df.union(spark.read.csv(stock_list, header=True).withColumn("Symbol", input_file_name()))

#print(f'Total of Rows: {df.count()}, Columns: {len(df.columns)} combined.')
logger.info(f'Combined all CSV files into a Spark DataFrame with Rows: {df.count()}, Columns: {len(df.columns)}')

#
# Extract the filename from the "Symbol" column in the original DataFrame (df)
#
from pyspark.sql.functions import regexp_extract
df = df.withColumn("Symbol", regexp_extract(df["Symbol"], r"([^/]+)\.csv$", 1))
#df.show()
logger.info('Extracted Symbol from path')

#
# Change data type of columns 
#
from pyspark.sql.functions import col
df = df.withColumn("Open", col('Open').cast('float')) \
    .withColumn("High", col('High').cast('float')) \
    .withColumn("Low", col('Low').cast('float')) \
    .withColumn("Close", col('Close').cast('float')) \
    .withColumn("Adj Close", col('Adj Close').cast('float')) \
    .withColumn("Volume", col('Volume').cast('int'))
#df.printSchema()
logger.info('Changed the data type of specific columns')

#
# Read the metadata CSV file into a Spark DataFrame and select only the relevant columns
#
metadata_df = spark.read.csv("symbols_valid_meta.csv", header=True)
metadata_df = metadata_df.select("Symbol", "Security Name")
#metadata_df.show()
logger.info('Read metadata CSV file')

#
# Join the original DataFrame (df) with the metadata DataFrame on the "Symbol" column
#
df = df.join(metadata_df, on=["Symbol"], how="left")
#df.show(10)
#df.printSchema()
logger.info('Security Name from metadata joined to etfs-stocks DataFrame')

#
# Re-order and rename columns
#
df = df.select('Symbol','Security Name',
 'Date',
 'Open',
 'High',
 'Low',
 'Close',
 'Adj Close',
 'Volume')
df.show()
logger.info('Columns re-ordered')

df = df.withColumnRenamed("Security Name", "Security_Name")
df = df.withColumnRenamed("Adj Close", "Adj_Close")
#df.show()
logger.info('Spaces in column names removed')

#
# Save as Parquet
#
filename = str(today)+"-etfs_stocks.parquet"
df.write.parquet(filename, mode="overwrite")
logger.info(f'DataFrame saved as {filename}')

#######################
# Feature Engineering #
#######################

#import pyspark.sql
from pyspark.sql.functions import mean
from pyspark.sql.window import Window

#
# Calculate moving average
#
# Define the window specification
windowSpec = (
    Window()
    .partitionBy("Symbol")
    .orderBy(col("Date").cast("timestamp").cast("long"))
    .rangeBetween(-29*86400, 0)                          # 86400 is the number of seconds a day
)

# Calculate the rolling 30-day median of the Adj_Close column
df2 = df.withColumn("vol_moving_avg", mean("Volume").over(windowSpec))

# Show the resulting DataFrame
#df2.show()
#df2.printSchema()
logger.info('Moving average of volume calculated')

#
# Calculate rolling median
#
from pyspark.sql.functions import udf, collect_list
import numpy as np
from pyspark.sql.types import FloatType

median_udf = udf(lambda x: float(np.median(x)), FloatType())

df2 = df2.withColumn("list", collect_list("Adj_Close").over(windowSpec)) \
  .withColumn("adj_close_rolling_med", median_udf("list"))

df2 = df2.drop("list")
# Show the resulting DataFrame
df2.show()
df2.printSchema()
logger.info('Rolling median of Adj Close calculated')

#
# Save as Parquet
#
filename = str(today)+"-etfs_stocks_2.parquet"
df2.repartition(1).write.parquet(filename, mode="overwrite")
logger.info('DataFrame saved as a Parquet file')

############################
#  Integrate ML Training   #
############################

spark = SparkSession.builder.appName("pipeline").getOrCreate()

# Read from Parquet
#parDF=spark.read.parquet('/content/drive/MyDrive/RiskThinkingAI/etfs2.parquet')
#parDF=spark.read.parquet('/content/etfs2.parquet/etfs_stocks_2.parquet')
parDF = df2

parDF = parDF.na.drop()
# Reference: https://hackernoon.com/building-a-machine-learning-model-with-pyspark-a-step-by-step-guide-1z2d3ycd

required_features = ['vol_moving_avg', 'adj_close_rolling_med']

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(parDF)
transformed_data.show(10)
transformed_data = transformed_data.select(['features', 'Volume'])

# Split the data
splits = transformed_data.randomSplit([0.8, 0.2])
training_data = splits[0]
test_data = splits[1]

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol='Volume', 
                      featuresCol='features',
                      maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(training_data)
rf_predictions = model.transform(test_data)

#print("Coefficients: " + str(model.coefficients))
#print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = model.summary
#print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
#print("r2: %f" % trainingSummary.r2)

infoString = 'Model trained using PySpark ML\n'\
            f'With Coefficients: {model.coefficients}\n'\
            +f'With Intercept: {model.intercept}\n'\
            +f'With RMSE: {trainingSummary.rootMeanSquaredError}\n'\
            +f'With r2: {trainingSummary.r2}\n'\
            +f'Number of Iterations: {trainingSummary.totalIterations}'

#print(infoString)

logger.info(infoString)


now = datetime.now().strftime("%Y%m%d-%H%M%S")
#print(now)
# save the model to disk
modelPath = 'trained-models/' + str(now) + '-ps-lr-model'
#joblib.dump(model, modelFileName)
model.save(modelPath)
logger.info(f'Model saved as {modelPath}')


logger.info('Pipeline ended')