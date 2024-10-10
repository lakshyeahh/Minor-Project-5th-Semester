from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the XGBoost model
xgboost_model = joblib.load('xgboost_model.pkl')

# Load the RandomForest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Load the MLPClassifier model
mlp_classifier_model = joblib.load('mlp_classifier_model.pkl')

# Route for the XGBoost model
@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    # Get the data from the request
    data = request.get_json(force=True)
    
    # Convert the data into a NumPy array
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Make the prediction
    prediction = xgboost_model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Route for the RandomForest model
@app.route('/predict/randomforest', methods=['POST'])
def predict_randomforest():
    # Get the data from the request
    data = request.get_json(force=True)
    
    # Convert the data into a NumPy array
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Make the prediction
    prediction = random_forest_model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(round(prediction[0]))})

# Route for the MLPClassifier model
@app.route('/predict/mlp', methods=['POST'])
def predict_mlp():
    # Get the data from the request
    data = request.get_json(force=True)
    
    # Convert the data into a NumPy array
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Make the prediction
    prediction = mlp_classifier_model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
