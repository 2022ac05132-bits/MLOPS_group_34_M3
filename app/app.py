# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('app/best_random_forest_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert input data to numpy array
    predictions = []
    for val in data['input']:
        input_data = np.array(val).reshape(1, -1)
    # Make prediction
        prediction = model.predict(input_data)
        predictions.append(int(prediction[0]))
    return jsonify({'prediction': predictions })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
