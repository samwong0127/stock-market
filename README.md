# Pipelining ETFs and stocks data
A work sample for the role of a Data Engineer

Inspired by https://github.com/RiskThinking/work-samples/blob/main/Data-Engineer.md

# Overview
This project is aimed to develop a data pipeline to process a large dataset with Spark as parallelization. There are 4 steps in the pipeline 
1. Data Ingestion: Ingest and process the raw data to a structured format for easy reading.
2. Feature Engineering: Build some feature engineering on top of the dataset from Step 1
3. Integrate ML training: Train Machine Learning model with the dataset from Step 2
4. Model Serving: Serve the trained model with API

### DAG
![DAG](/pictures/DAG.png)

The stock and ETF dataset used can be downloaded from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?resource=download. This project is assisted by ChatGPT and the full interaction is attached to `ChatGPT-interactions/`. Most of the tasks are developed with Google Colab and their drafts can be found in `StockMarket.ipynb`. Steps 1 - 3 are also dockerized and combined under `pipeline/`. In step 3, the ML model can be trained by running `pipeline/training.py` with dependencies in `pipeline/requirements.txt` and the dataset from Step 2. Step 4 is accomplished by running a Python Notebook (`model-serving/ModelServing-GoogleColab.ipynb`) with Google Colab or a local API server.

# Data Ingestion
Spark DataFrame is used to set up a data structure to retain all data from ETFs and stocks because it is faster and consumes less memory than Pandas DataFrame. The lists of paths of CSV files are created and read into a DataFrame together with their symbols. Their security names, column types, column names are added and changed afterwards. After ingesting the data, the resulting DataFrame is saved as in a Parquet format. In addition, the dataset can be maintained easier by saving into a SQL database. (Link to dataset: https://drive.google.com/file/d/1NKYjKmdH_B3LXxW_f3P3xSTxKqhOprF-/view?usp=share_link)

# Feature Engineering
The moving average and rolling median are calculated by "rolling" method combined by `orderBy()`, `window()`, and `partitionBy()` functions. New columns: `vol_moving_avg` and `adj_close_rolling_med` are added and saved as a new file. (Link to dataset: https://drive.google.com/file/d/19o5KhXQ2onOH-5wwRRYSzHvdZmNi4uCK/view?usp=share_link)

**References:**

https://stackoverflow.com/questions/33207164/spark-window-functions-rangebetween-dates

https://stackoverflow.com/questions/45806194/pyspark-rolling-average-using-timeseries-data

# Integrate ML Training
To train a Machine Learning model to predict the `Volume` of the stock market, ML libraries such as Scikit-learn are considered. However, the RAM of free-of-charge version of Google Colab is limited and it ran out of memory while training the ML model with the whole dataset. As a demonstration, a model trained with a sample of the dataset is loaded by the local API in **Model Serving**.

Meanwhile, there are built-in ML models in PySpark with distributed implementations and are fully integrated with Spark's distributed computing feature. The trained model can be saved with built-in `save()` function and can be retreived by E.g. `pyspark.ml.regression.LinearRegressionModel.load(path)`.

-----
In addition, the data ingestion, feature engineering, and ML training process can be integrated into a data pipeline using **Apache Airflow**.

-----

**References:**

https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression

https://medium.com/@solom1913/linear-regression-predictions-using-pyspark-d0f283167040

# Model Serving

## By local API
The model used in this case is a Random Forest Regressor model trained with a sample of the dataset as a demonstration. The local server can be run by `python api.py` and it will load the model from `trained-models` folder. The responses from the API are shown in `StockMarket.ipynb` and dependencies of the API are listed in `model-serving\requirements.txt`.


## By Google Colab
To serve the trained model, a `/predict` API is run by Google Colab and users can access the API by visiting the link printed and appending required parameters such as `/predict?vol_moving_avg=12345&adj_close_rolling_med=25`. The model used in this case is a Linear Regression model trained with the whole dataset as a demonstration.

**Index page**

![Hello World](/pictures/model-serving-1.jpg)

**API serving**

![/predict](/pictures/model-serving-2.jpg)


**References:**

https://brightersidetech.com/running-flask-apps-in-google-colab/

https://www.geeksforgeeks.org/get-value-of-a-particular-cell-in-pyspark-dataframe/


# Other references

Save ChatGPT as MD file

https://www.reddit.com/r/ChatGPT/comments/zm237o/save_your_chatgpt_conversation_as_a_markdown_file/

Download folder from Google Colab

https://stackoverflow.com/questions/50453428/how-do-i-download-multiple-files-or-an-entire-folder-from-google-colab

Dockerization

https://medium.com/hashmapinc/building-ml-pipelines-654daf4f23dc

