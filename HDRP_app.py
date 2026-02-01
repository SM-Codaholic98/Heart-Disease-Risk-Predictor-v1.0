import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
from tensorflow.keras.models import load_model
import socket

app = Flask(__name__)

# --- Load Models & Scaler ---
print("⏳ Loading models...")

# 1. Load ML Model (CatBoost)
try:
    ml_model = pickle.load(open('CatBoost_ML_Model.pkl', 'rb'))
    print("✅ ML Model (CatBoost) loaded.")
except Exception as e:
    ml_model = None
    print(f"❌ Error loading ML model: {e}")

# 2. Load DL Model (DNN)
try:
    dl_model = load_model('DNN_Model.keras')
    print("✅ DL Model (DNN) loaded.")
except Exception as e:
    dl_model = None
    print(f"❌ Error loading DL model: {e}")

# 3. Load Scaler (Required for DL)
try:
    scaler = joblib.load('scaler.gz')
    print("✅ Scaler loaded.")
except Exception as e:
    scaler = None
    print(f"❌ Error loading Scaler: {e}")


def get_form_data(form):
    """Helper to extract data from form and convert to numpy array."""
    data = [
        float(form['age']),
        float(form['sex']),
        float(form['cp']),
        float(form['trestbps']),
        float(form['chol']),
        float(form['fbs']),
        float(form['restecg']),
        float(form['thalach']),
        float(form['exang']),
        float(form['oldpeak']),
        float(form['slope'])
    ]
    return np.array([data])


# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html', page_class='bg-home')

@app.route('/predict_ml', methods=['GET', 'POST'])
def predict_ml():
    prediction_text = None
    result_color = "white"

    if request.method == 'POST':
        if ml_model:
            try:
                features = get_form_data(request.form)
                prediction = ml_model.predict(features)[0]
                
                if prediction == 1:
                    prediction_text = "High Risk of Heart Disease"
                    result_color = "#ff4c4c"
                else:
                    prediction_text = "Low Risk (Healthy)"
                    result_color = "#00ffff"
            except Exception as e:
                prediction_text = f"Error: {e}"
        else:
            prediction_text = "Error: ML Model file not found."

    return render_template('predict.html', 
                           model_type="Machine Learning (CatBoost)",
                           page_class='bg-ml',
                           prediction_text=prediction_text,
                           result_color=result_color)

@app.route('/predict_dl', methods=['GET', 'POST'])
def predict_dl():
    prediction_text = None
    result_color = "white"

    if request.method == 'POST':
        if dl_model and scaler:
            try:
                features = get_form_data(request.form)
                
                # DL Specific: Scale and Reshape
                features_scaled = scaler.transform(features)
                features_reshaped = features_scaled.reshape(1, 11, 1)
                
                prediction_prob = dl_model.predict(features_reshaped)[0][0]
                prediction_class = 1 if prediction_prob > 0.5 else 0
                
                if prediction_class == 1:
                    prediction_text = f"High Risk ({prediction_prob:.2%} Probability)"
                    result_color = "#ff4c4c"
                else:
                    prediction_text = f"Low Risk ({prediction_prob:.2%} Probability)"
                    result_color = "#00ffff"
            except Exception as e:
                prediction_text = f"Error: {e}"
        else:
            prediction_text = "Error: DL Model or Scaler not found."

    return render_template('predict.html', 
                           model_type="Deep Neural Network (DNN Model)",
                           page_class='bg-dl',
                           prediction_text=prediction_text,
                           result_color=result_color)

if __name__ == "__main__":
    # Get the local IP address of this machine
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🚀 App is running! Open this URL on any device connected to the SAME WiFi:")
    print(f"👉 http://{local_ip}:5000\n")
    
    # host='0.0.0.0' allows external connections
    app.run(host='0.0.0.0', port=5000, debug=True)