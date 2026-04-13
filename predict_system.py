 import pandas as pd
import joblib
import numpy as np

# Load models
anomaly_model = joblib.load("xgb_anomaly_model.pkl")
efficiency_model = joblib.load("xgb_efficiency_model.pkl")

# New input data (without Detected_Anomaly)
new_data = {
    'Anomaly_Severity_Level': 1,
    'Affected_Area_%': 12,
    'Best_Efficiency_%': 100,
    'Solar_Irradiance_W_m2': 850,
    'Ambient_Temperature_C': 32,
    'Panel_Temperature_C': 45,
    'Humidity_%': 60,
    'Wind_Speed_mps': 3,
    'Rainfall_mm': 0,
    'Voltage_V': 230,
    'Current_A': 8,
    'Energy_Generated_kWh': 4.2,
    'Panel_Type': 1,
    'Season': 2,
    'Month': 6,
    'Day_of_Year': 150,
    'Temp_Diff': 13,
    'Electrical_Load': 1840
}

df_new = pd.DataFrame([new_data])

# STEP 1: Predict anomaly
predicted_anomaly = anomaly_model.predict(df_new)

# Add anomaly to input
df_new['Detected_Anomaly'] = predicted_anomaly

# STEP 2: Reorder columns exactly as training
df_new = df_new[efficiency_model.get_booster().feature_names]

# STEP 3: Predict efficiency
predicted_efficiency = efficiency_model.predict(df_new)

print("=== FINAL PREDICTION RESULT ===")
print("Predicted Anomaly Class:", predicted_anomaly[0])
print("Predicted Efficiency %:", round(predicted_efficiency[0], 2))