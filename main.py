from logging import debug
from flask import Flask, render_template, request

app = Flask(__name__)

import joblib
model = joblib.load('hiring_model.pkl')


@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/predict',methods = ['POST'])
def predict():
    exp = request.form.get('experience')
    score = request.form.get('test_score')
    int_score = request.form.get('interview_score')
    
    prediction = model.predict([[int(exp),int(score),int(int_score)]])
    output = round(prediction[0],2)
    return render_template('base.html',prediction_text=f'\nthe predicted salary is ${output}')

if __name__=='__main__':
    app.run(debug=True)