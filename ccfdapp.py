import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
rgs_model = pickle.load(open('Logisticregression_model.pkl', 'rb'))

rfc_model = pickle.load(open('Randomforest_model.pkl', 'rb'))

nbs_model = pickle.load(open('Naivebayes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('ccfd_home_page.html')

@app.route('/Randomforest',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    myinput = request.form['Transaction'].split()
    del myinput[-1]
    input_values = []
    input_values.append(myinput)
    result = rfc_model.predict(input_values)

    if result[0]==1:
        return render_template('ccfd_home_page.html', pred='This transaction is ',result='Fraud')
    if result[0]==0:
        return render_template('ccfd_home_page.html', pred='This transaction is ',result='Normal')
 
    ''' http://localhost:8888/notebooks/Desktop/testing%20ccfd/Handling%20Imbalanced%20Data-%20Under%20Sampling.ipynb'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = rgs_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)