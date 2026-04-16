from unittest import result

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

car_data = pd.read_csv("../../resources/ElectricCarData_Clean.csv")

# Handle null values
car_data = car_data.replace("-", np.nan)
car_data["FastCharge_KmH"] = pd.to_numeric(car_data["FastCharge_KmH"], errors="coerce")
car_data.dropna(inplace=True)

# Convert categorical columns to numeric
car_data = pd.get_dummies(car_data, columns=['Brand', 'PowerTrain', 'PlugType'], drop_first=True)
car_data["RapidCharge"] = (car_data["RapidCharge"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}))

# Drop unused columns and fetch the target
X = car_data.drop(columns=['PriceEuro', 'Model', 'BodyStyle', 'Segment','TopSpeed_KmH'])
y = car_data['PriceEuro']

# Scaling the data
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mae')

# Train
history = model.fit(X_scaled, y_scaled, batch_size=256, epochs=50, verbose=1)

# Predict (example: using first three rows)
predicted_scaled = model.predict(X_scaled[:7])

# Convert back to original price scale
predicted_actual = y_scaler.inverse_transform(predicted_scaled)

print(predicted_actual.flatten())
