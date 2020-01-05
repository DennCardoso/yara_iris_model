from flask import Flask, render_template, url_for, request
from flask_material import Material 

import pandas as pd 
import numpy as np 

from sklearn.externals import joblib 

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.hmtl")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html", df_view= df)

@app.route('/', methods=['POST'])
def analyze():
    if request.method == 'POST':
        petal_length = request.form['petal_length']
        sepal_length = request.form['sepal_length']
        petal_width = request.form['petal_width']
        sepal_width = request.form['sepal_width']

        #Clean the data by convert from unicode to float
        sample_data = [sepal_length, sepal_width, petal_length, petal_width]
        clean_data = [float(i) for i in sample_data]

        # reshape the data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

        #reload de model
        logit_model = joblib.load('data/model.pkl')
        result_prediction = logit_model.predict(ex1)

        return render_template('index_html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
        result_prediction=result_prediction)


if __name__ == '__main__':
    app.run(debug=True)