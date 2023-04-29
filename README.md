# stock-market
A work sample for the role of a Data Engineer

https://github.com/RiskThinking/work-samples/blob/main/Data-Engineer.md

https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?resource=download
### Q1. Setup a data structure to retain all data from ETFs and stocks with the following columns.

#### Idea 1 
1. Use a SQL database to store all the data (import all data into the database with Python, parallelize imports by Spark?)
2. Convert all data as a Parquet file by pyspark SQL (https://stackoverflow.com/questions/41498672/how-to-convert-an-500gb-sql-table-into-apache-parquet)

#### Idead 2
1. Load all data into a dataframe with the given columns with Python, parallelize import by Spark (Needs heavy RAMs)
2. Convert dataframe to Parquet

### Q2. Feature Engineering
Tasks:
1. Calculate the moving average of the trading volume (Volume) of 30 days per each stock and ETF, and retain it in a newly added column vol_moving_avg.
2. Similarly, calculate the rolling median and retain it in a newly added column adj_close_rolling_med.
3. Retain the resulting dataset into the same format as Problem 1, but in its own stage/directory distinct from the first.
4. (Bonus) Write unit tests for any relevant logic.
