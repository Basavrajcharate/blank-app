
import streamlit as st
import pennylane as qml
from pennylane import numpy as np
import numpy as npy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Quantum setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

def variational_block(weights):
    for i in range(n_qubits):
        qml.Rot(*weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def circuit(x, weights):
    feature_map(x[:n_qubits])
    variational_block(weights)
    return qml.expval(qml.PauliZ(0))

# Load weights
try:
    weights = np.load("qnn_aqi_weights.npy", allow_pickle=True)
except:
    st.warning("⚠️ Using random weights. Upload real model file for true results.")
    weights = np.random.randn(n_qubits, 3, requires_grad=True)

# UI
st.title("⚛️ Quantum AQI Predictor")
st.subheader("Enter past 14 days AQI values to predict the next 7 days")

aqi_input = st.text_area("Enter 14 AQI values (one per line)", "135
138
140
145
142
141
138
136
134
137
139
141
140
143")
try:
    aqi_values = [float(x) for x in aqi_input.strip().split('\n')]
    assert len(aqi_values) == 14
except:
    st.error("Please enter exactly 14 numeric values.")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, np.pi))
input_scaled = scaler.fit_transform(np.array(aqi_values).reshape(-1,1)).flatten()

def predict_next_days(last_window, weights, days=7):
    predictions = []
    current_input = last_window.copy()
    for _ in range(days):
        x_scaled = scaler.transform(np.array(current_input[-14:]).reshape(-1,1)).flatten()
        y_pred = circuit(x_scaled, weights)
        y_real = scaler.inverse_transform([[y_pred]])[0][0]
        predictions.append(y_real)
        current_input.append(y_real)
    return predictions

next_week_pred = predict_next_days(aqi_values.copy(), weights)

st.success("Prediction complete!")
st.write("### 📅 Next 7 Days AQI:")
for i, val in enumerate(next_week_pred, 1):
    st.write(f"Day {i}: {val:.2f}")

st.write("### 📈 AQI Trend")
fig, ax = plt.subplots()
days = list(range(-13, 1)) + list(range(1, 8))
values = aqi_values + next_week_pred
ax.plot(days, values, marker='o')
ax.axvline(x=0, linestyle='--', color='gray')
ax.set_xlabel("Days (0 = Today)")
ax.set_ylabel("AQI")
ax.set_title("AQI Forecast")
st.pyplot(fig)
