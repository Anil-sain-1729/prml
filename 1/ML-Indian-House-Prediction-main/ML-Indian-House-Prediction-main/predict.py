import pickle
import logging
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS  # Import CORS
from waitress import serve

logging.basicConfig(level=logging.INFO)

output_file = 'model.bin'

# Load the model
with open(output_file, 'rb') as f_in:  # replace w with r for reading
    dv, model = pickle.load(f_in)

app = Flask('House Price')

# Enable CORS for all routes
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_data = dv.transform(data)
    y_pred = model.predict(X_data)

    result = {
        'price': int(y_pred)
    }
    return jsonify(result)

@app.route("/")
def form():
    return render_template("index.html")


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)
