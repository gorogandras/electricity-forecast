from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.eForecaster.pipeline.prediction import PredictionPipeline


app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def training():
    os.system("python3 main.py")
    return "Training successful!"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            moment_to_pred = [[str(request.form['moment_to_pred'])]]
            df = pd.DataFrame(moment_to_pred, columns=['datetime'])
            df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
            df2 = pd.DataFrame()

            df2['hour'] = df['datetime'].dt.hour
            df2['dayofweek'] = df['datetime'].dt.dayofweek
            df2['quarter']  = df['datetime'].dt.quarter
            df2['month']  = df['datetime'].dt.month
            df2['year']  = df['datetime'].dt.year
            df2['dayofyear']  = df['datetime'].dt.dayofyear
            df2['minute']  = df['datetime'].dt.minute
            #data = [hour[0], dayofweek[0], quarter[0], month[0], year[0], dayofyear[0], minute[0]]

            obj = PredictionPipeline()
            predict = obj.predict(df2)

            return render_template('results.html', prediction = str(predict))
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)