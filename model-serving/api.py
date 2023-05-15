# References:
# https://brightersidetech.com/running-flask-apps-in-google-colab/
# https://www.geeksforgeeks.org/get-value-of-a-particular-cell-in-pyspark-dataframe/

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained Spark ML model
"""
#from pyspark.sql import SparkSession
# Create a SparkSession
#spark = SparkSession.builder.appName('api').getOrCreate()

#from pyspark.ml.feature import VectorAssembler
#required_features = ['vol_moving_avg', 'adj_close_rolling_med']
#assembler = VectorAssembler(inputCols=required_features, outputCol='features')

#from google.colab.output import eval_js
#print("Visit below link to access the API")
#print("---------------------------------------------")
#print(eval_js("google.colab.kernel.proxyPort(5000)"))
#print("---------------------------------------------\n")

# Load the model
from pyspark.ml.regression import LinearRegressionModel
#model = LinearRegressionModel.load(r"trained-models/20230512-114220-lr-model")
#model = LinearRegressionModel.load(r"trained-models\2023-05-12-lr-model")

try:
    model = LinearRegressionModel.load('trained-models/20230512-114220-lr-model')
    print('Spark ML model loaded.')
except:
    model = joblib.load(r'trained-models\20230512-132900-sk-lr-model')
    print('Scikit-learn ML model loaded.')
    #spark.stop()
"""

# Load trained Scikit-learn model
model = joblib.load(r'trained-models\20230512-132900-sk-lr-model')
#print('Scikit-learn ML model loaded.')

def volume_prediction(model, data):

    df = pd.DataFrame(data=data)
    #print(df)
    #apiData = spark.createDataFrame([data], required_features)

    #apiData = assembler.transform(apiData)
    #apiData = apiData.select(['features'])

    prediction = model.predict(df)
    #print(f'Prediction: {int(prediction.collect()[0][1])}')
    #print(prediction[0])
    #return int(prediction[0])
    return {'volume': int(prediction[0])}



@app.route('/')
def home():
    return "Hello World"

"""
@app.route('/test')
def test():
  return jsonify({'test': 'You can access test API'}), 200
"""
@app.route('/predict')
def get_volume():
    vol_moving_avg = request.args.get('vol_moving_avg')
    adj_close_rolling_med = request.args.get('adj_close_rolling_med')
    #print(vol_moving_avg)
    #print(adj_close_rolling_med)
    if not vol_moving_avg or not adj_close_rolling_med:
        return jsonify({'error': 'You need to supply both vol_moving_avg and adj_close_rolling_med'}), 400

    #print(vol_moving_avg)
    #print(adj_close_rolling_med)
    #data = [float(vol_moving_avg), float(adj_close_rolling_med)]
    data = {'vol_moving_avg': [float(vol_moving_avg)], 'adj_close_rolling_med': [float(adj_close_rolling_med)]}
    return jsonify({
        **volume_prediction(model, data),
    }),200

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)