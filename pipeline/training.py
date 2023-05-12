

################################################################################
#                                                                              #
#    Set up log                                                                #
#                                                                              #
################################################################################


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



logger.info("Started training pipeline...")

logger.info("Started Spark ML training pipeline...")


################################################################################
#                                                                              #
#    Train Spark ML model                                                      #
#                                                                              #
################################################################################


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('pipeline').getOrCreate()


datasetPath = 'D:\RiskThinkingAI\etfs_stocks_2.parquet'
# Read from Parquet
#parDF=spark.read.parquet('/content/drive/MyDrive/RiskThinkingAI/etfs2.parquet')
df=spark.read.parquet(datasetPath)
logger.info("Data loaded as Spark DataFrame")

df = df.na.drop()
# Reference: https://hackernoon.com/building-a-machine-learning-model-with-pyspark-a-step-by-step-guide-1z2d3ycd

# Use a sample of the dataset to train the model
df = df.sample(0.05)

required_features = ['vol_moving_avg', 'adj_close_rolling_med']

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(df)
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
#trainingSummary.residuals.show()
#print("numIterations: %d" % trainingSummary.totalIterations)
#print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))


infoString = 'Model trained using PySpark ML\n'\
            f'With Coefficients: {model.coefficients}\n'\
            +f'With Intercept: {model.intercept}\n'\
            +f'With RMSE: {trainingSummary.rootMeanSquaredError}\n'\
            +f'With r2 score: {trainingSummary.r2}\n'\
            +f'Number of Iterations: {trainingSummary.totalIterations}'

#print(infoString)
logger.info(infoString)


now = datetime.now().strftime("%Y%m%d-%H%M%S")
#print(now)
# save the model to disk
modelPath = 'trained-models/' + str(now) + '-ps-lr-model'
#joblib.dump(model, modelFileName)
model.save(modelPath)
#print(f'Model saved as {modelPath}')
logger.info(f'Model saved as {modelPath}')
spark.stop()


##################################################################################################################################
##################################################################################################################################

################################################################################
#                                                                              #
#    Train Scikit-learn models                                                 #
#                                                                              #
################################################################################

logger.info("Started Scikit-learn training pipeline...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select features and target
features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'


data = pd.read_parquet(datasetPath, columns=[features].append('Volume'))
infoString = 'Loaded dataset as Pandas DataFrame'
#print(infoString)
logger.info(infoString)

# Remove rows with NaN values
data.dropna(inplace=True)

data = data.sample(1000)


X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)


#print('Start training...')
logger.info('Start training...')

# Train the model
model.fit(X_train, y_train)
#print('Finish training...')
logger.info('Finish training...')

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error and Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

infoString = 'Model trained using Scikit-learn\n'\
            f'With mean_absolute_error: {mae}\n'\
            f'With RMSE: {rmse}\n'\
            +f'With r2 score: {r2}\n'

#print(infoString)
logger.info(infoString)



import joblib
now = datetime.now().strftime("%Y%m%d-%H%M%S")
#print(now)
# save the model to disk
modelPath = 'trained-models/' + str(now) + '-sk-lr-model'
joblib.dump(model, modelPath)
logger.info(f'Model saved as {modelPath}')

logger.info('Training pipeline ended')
