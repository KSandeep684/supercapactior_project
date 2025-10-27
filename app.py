import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- DATASET GENERATION (Synthetic example) ---
def generate_dataset():
    np.random.seed(42)
    data = {
        'surface_area': np.random.uniform(1000, 2000, 200),
        'pore_size': np.random.uniform(2.0, 5.0, 200),
        'doping_level': np.random.uniform(0.05, 0.2, 200),
        'voltage_window': np.random.uniform(2.5, 3.5, 200),
    }
    # Mock output: weighted sum + noise
    data['capacitance'] = (
        0.13 * data['surface_area']
        + 12 * data['pore_size']
        + 180 * data['doping_level']
        + 75 * data['voltage_window']
        + np.random.normal(0, 20, 200)
    )
    return pd.DataFrame(data)

df = generate_dataset()

# --- TRAINING MODEL ---
features = ['surface_area', 'pore_size', 'doping_level', 'voltage_window']
X = df[features].values
y = df['capacitance'].values
model = xgb.XGBRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

st.set_page_config(layout="wide")
st.title("Supercapacitor Capacitance Predictor + AI Recommendation")
st.markdown("Enter the supercapacitor properties below to predict capacitance and get AI-driven expert insights.")

surface_area = st.number_input('Surface Area', value=1200.0)
pore_size = st.number_input('Pore Size', value=2.5)
doping_level = st.number_input('Doping Level', value=0.12)
voltage_window = st.number_input('Voltage Window', value=2.8)

if st.button("Predict & AI Recommend"):
    input_data = np.array([[surface_area, pore_size, doping_level, voltage_window]])
    cap_pred = model.predict(input_data)[0]
    st.success(f"Predicted Capacitance: {cap_pred:.2f} F/g")

    # --- AI Recommendation using Gemini ---
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini = genai.GenerativeModel('gemini-1.5-pro')  # correct model name
        prompt = (
            f"You are an expert in supercapacitors. "
            f"Given: Surface area={surface_area}, Pore size={pore_size}, Doping level={doping_level}, Voltage window={voltage_window}, "
            f"Predicted capacitance={cap_pred:.2f} F/g. "
            "Write a performance summary (1 line), 2 key observations (strength/weakness), and 1 real-world application suggestion."
        )
        response = gemini.generate_content(prompt)
        st.markdown("### AI Recommendation")
        st.info(response.text)
    except Exception as e:
        st.error("AI Recommendation failed. Please check your API key and Gemini model name. Error: " + str(e))

st.divider()
st.subheader("Preview of Training Data")
st.dataframe(df.head(20))

# Optional: plot the model fit vs target
if st.checkbox("Show Model Fit Plot"):
    plt.figure(figsize=(6,3))
    plt.scatter(y, model.predict(X), alpha=0.6)
    plt.xlabel("True Capacitance (F/g)")
    plt.ylabel("Predicted Capacitance (F/g)")
    plt.title("XGBoost Model Fit")
    st.pyplot(plt)
