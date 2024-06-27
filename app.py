from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

# Load the dataset for location options
dataset = pd.read_csv('Cleaned_Data.csv')
locations = dataset['Address'].unique()  # Extract unique locations

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    bedrooms = int(request.form['bedrooms'])
    washrooms = int(request.form['washrooms'])
    area = float(request.form['area'])

    # Prepare features for prediction
    features = np.array([[location, bedrooms, washrooms, area]])
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'Estimated House Price: {prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
