from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler from local files
clf = joblib.load("kmeans_model.joblib")
scaler = joblib.load("scaler.joblib")  # Use the saved StandardScaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    gender = request.form.get('gender')
    age = float(request.form.get('age'))
    income = float(request.form.get('income'))
    score = float(request.form.get('score'))

    # Encode gender
    gender_encoded = 1 if gender.lower() == "male" else 0

    # Create input as array (in expected order)
    input_data = np.array([[gender_encoded, age, income, score]])

    # Apply standard scaling using loaded scaler
    scaled_input = scaler.transform(input_data)

    # Predict cluster
    prediction = clf.predict(scaled_input)[0]

    return render_template('index.html', cluster=prediction)

if __name__ == '__main__':
    app.run(debug=True)
