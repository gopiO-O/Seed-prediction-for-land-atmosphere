import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

loaded_data = joblib.load('model_and_scaler.pkl')
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']

def get_user_input_and_predict():
    temperature = float(input("Enter temperature: "))
    ph = float(input("Enter pH level: "))
    humidity = float(input("Enter humidity: "))
    rainfall = float(input("Enter rainfall: "))
    N = float(input("Enter nitrogen content (N): "))
    P = float(input("Enter phosphorus content (P): "))
    K = float(input("Enter potassium content (K): "))
    
    user_data = pd.DataFrame([[temperature, ph, humidity, rainfall, N, P, K]],
                             columns=['temperature', 'ph', 'humidity', 'rainfall', 'N', 'P', 'K'])
    
    user_data_transformed = loaded_scaler.transform(user_data)
    
    prediction = loaded_model.predict(user_data_transformed)
    
    print(f"The predicted crop is: {prediction[0]}")

get_user_input_and_predict()
