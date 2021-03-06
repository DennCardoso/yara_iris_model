from flask import Flask, render_template, url_for, request
from flask_material import Material 
import regression as reg

import pandas as pd 
import numpy as np 

import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    return render_template('preview.html')

@app.route('/training', methods=['GET'])
def training_model():
    reg.logisticRegModel()
    return 'Training Finished - Create pkl File'

@app.route('/predict', methods=['POST'])
def analyze():
    print(request.is_json)
    content = request.get_json()
    sepal_length = content['sepal_length']
    sepal_width = content['sepal_width']
    petal_length = content['petal_length']
    petal_width = content['petal_width']
    
    #Clean the data by convert from unicode to float
    sample_data = [sepal_length, sepal_width, petal_length, petal_width]
    clean_data = [float(i) for i in sample_data]
    
    # reshape the data as a Sample not Individual Features
    ex1 = np.array(clean_data).reshape(1,-1)
    
    #reload de model
    logit_model = joblib.load('data/model.pkl')
    result_prediction = logit_model.predict(ex1)
    
    result = 'the IRIS for this vector is: '+ result_prediction[0]

    return result

if __name__ == '__main__':
    app.run(debug=True)