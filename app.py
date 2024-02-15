from flask import Flask, render_template, request, url_for, send_from_directory
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
            obj = PredictionPipeline()
            datetime_df = obj.create_datetime_df(moment_to_pred)
            predict = obj.predict(datetime_df)
            
            return render_template('results.html', prediction = round(predict[0]))
        
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')

@app.route("/artifacts/plotting/scatterplot.png")
def serve_image():
    return send_from_directory("artifacts", "plotting/scatterplot.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)