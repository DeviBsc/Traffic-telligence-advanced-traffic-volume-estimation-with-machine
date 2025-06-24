import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('G:/AI&ML/ML projects/Traffic_volume/model.pkl', 'rb'))
scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")  # Rendering the home page

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = np.array([input_feature])

    # Define the feature names
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
             'hours', 'minutes', 'seconds']

    # Create DataFrame and scale the data
    data = pd.DataFrame(features_values, columns=names)
    data = scale.fit_transform(data)
    data = pd.DataFrame(data, columns=names)

    # Predicting using the loaded model
    prediction = model.predict(data)

    # Return the result to the HTML page
    text = "Estimated Traffic Volume is : "
    return render_template("index.html", prediction_text=text + str(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)
