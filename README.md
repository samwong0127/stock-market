## Pipelining ETFs and stocks data
A work sample for the role of a Data Engineer

https://github.com/RiskThinking/work-samples/blob/main/Data-Engineer.md


## Overview
This project is aimed to develop a data pipeline to process a large dataset with Spark as parallelization. There are 4 steps in the pipeline 
1. Data Ingestion: Ingest and process the raw data to a structured format for easy reading.
2. Feature Engineering: Build some feature engineering on top of the dataset from Step 1
3. Integrate ML training: Train Machine Learning model with the dataset from Step 2
4. Model Serving: Serve the trained model with API
The stock and ETF dataset used can be downloaded from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?resource=download.  This project is assisted by ChatGPT and the full interaction is attached to `ChatGPT-interactions/`. All tasks are developed with Google Colab and details can be found in `StockMarket.ipynb`. Steps 1 - 3 are also dockerized and combined under `pipeline/` and Step 4 is accomplished by running a Python Notebook (`ModelServing.ipynb`) with Google Colab.



### Q1. Setup a data structure to retain all data from ETFs and stocks with the following columns.

#### Idea 1a
1. Use a SQL database to store all the data (import all data into the database with Python, parallelize imports by Spark?)
2. Convert all data as a Parquet file by pyspark SQL (https://stackoverflow.com/questions/41498672/how-to-convert-an-500gb-sql-table-into-apache-parquet)

#### Idead 1b
1. Load all data into a dataframe with the given columns with Python, parallelize import by Spark (Needs heavy RAMs)
2. Convert dataframe to Parquet

### Q2. Feature Engineering
Tasks:
1. Calculate the moving average of the trading volume (Volume) of 30 days per each stock and ETF, and retain it in a newly added column vol_moving_avg.
2. Similarly, calculate the rolling median and retain it in a newly added column adj_close_rolling_med.
3. Retain the resulting dataset into the same format as Problem 1, but in its own stage/directory distinct from the first.
4. (Bonus) Write unit tests for any relevant logic.

#### Idea 2a
1. We can calculate the moving average with Python directly by it's time consuming. We can speed up the process with Spark SQL as in COMP4442 project and the following method https://stackoverflow.com/questions/45806194/pyspark-rolling-average-using-timeseries-data

### Q3. Integrate ML Training

#### Idea 3a
1. Use the code given (Out of Memory Error)

#### Idea 3b
1. Use the ML models in PySpark
https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression

Linear Regression Predictions using PySpark

https://medium.com/@solom1913/linear-regression-predictions-using-pyspark-d0f283167040

https://stackoverflow.com/questions/50453428/how-do-i-download-multiple-files-or-an-entire-folder-from-google-colab



### Q4. Model Serving

#### Idea 4a
1. Deploy the API server with the model through a suitable service

#### Idea 4b
1. Run API with Google Colab 



### Dockerize

https://medium.com/hashmapinc/building-ml-pipelines-654daf4f23dc
