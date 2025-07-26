from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
# Load the model and scaler
model = joblib.load('D:\Heart disease\model\heart_disease_model.pkl')
scaler = joblib.load('D:\Heart disease\model\scaler.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'age': int(request.form['age']),
        'sex': int(request.form['sex']),
        'cp': int(request.form['cp']),
        'trestbps': int(request.form['trestbps']),
        'chol': int(request.form['chol']),
        'fbs': int(request.form['fbs']),
        'restecg': int(request.form['restecg']),
        'thalach': int(request.form['thalach']),
        'exang': int(request.form['exang']),
        'oldpeak': float(request.form['oldpeak']),
        'slope': int(request.form['slope']),
        'ca': int(request.form['ca']),
        'thal': int(request.form['thal'])
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    result = "⚠️ Person is likely to have heart disease." if prediction[0] == 1 else "✅ Person is unlikely to have heart disease."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
