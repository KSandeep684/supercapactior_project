# ==============================================================================
# FINAL CAPSTONE PROJECT: V28 - DEFINITIVE STABLE VERSION
# ==============================================================================

import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import google.generativeai as genai

# --- CACHED MODEL TRAINING AND DATA GENERATION ---
@st.cache_resource
def load_and_train_models():
    """
    Loads data, generates the large dataset, trains the models, and returns all necessary components.
    """
    # --- Define Degradation Scenarios Directly ---
    degradation_scenarios = [
        {'config': {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 192.03, 'end_charge': 173.79, 'start_discharge': 182.89, 'end_discharge': 165.51},
        {'config': {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 71.53, 'end_charge': 58.59, 'start_discharge': 68.12, 'end_discharge': 55.80},
        {'config': {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.75}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 29.03, 'end_charge': 23.89, 'start_discharge': 27.65, 'end_discharge': 22.75},
        {'config': {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.75}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 13.86, 'end_charge': 10.76, 'start_discharge': 13.20, 'end_discharge': 10.25},
        {'config': {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 1.5}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 98.22, 'end_charge': 66.02, 'start_discharge': 93.54, 'end_discharge': 62.88},
        {'config': {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 1.5}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 33.86, 'end_charge': 22.05, 'start_discharge': 32.25, 'end_discharge': 21.00},
        {'config': {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 0.475}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 12.68, 'end_charge': 7.50, 'start_discharge': 12.08, 'end_discharge': 7.14},
        {'config': {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 0.375}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 6.87, 'end_charge': 3.80, 'start_discharge': 6.54, 'end_discharge': 3.62},
    ]
    single_point_scenarios = [
        {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 2.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 175.88, 'Discharge_Capacity_mAh_g-1': 167.50},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 4.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 24.78, 'Discharge_Capacity_mAh_g-1': 23.60},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 132.51, 'Discharge_Capacity_mAh_g-1': 126.20},
        {'Electrode_Material': 'CuO/CoO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 58.79, 'Discharge_Capacity_mAh_g-1': 55.99},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 2.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 83.79, 'Discharge_Capacity_mAh_g-1': 79.80},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 58.94, 'Discharge_Capacity_mAh_g-1': 56.13},
        {'Electrode_Material': 'CuO@MWCNT', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 37.78, 'Discharge_Capacity_mAh_g-1': 35.98},
        {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 0.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 33.78, 'Discharge_Capacity_mAh_g-1': 32.17},
        {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 0.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 23.48, 'Discharge_Capacity_mAh_g-1': 22.36},
    ]
    all_data = []
    for scenario in degradation_scenarios:
        charge_drop, discharge_drop = scenario['start_charge'] - scenario['end_charge'], scenario['start_discharge'] - scenario['end_discharge']
        for cycles in range(0, scenario['end_cycles'] + 1, 250):
            cycle_ratio = cycles / scenario['end_cycles'] if scenario['end_cycles'] > 0 else 0
            charge = scenario['start_charge'] - charge_drop * (cycle_ratio ** 0.9)
            discharge = scenario['start_discharge'] - discharge_drop * (cycle_ratio ** 0.9)
            row_data = scenario['config'].copy()
            row_data['Cycles_Completed'], row_data['Charge_Capacity_mAh_g-1'], row_data['Discharge_Capacity_mAh_g-1'] = cycles, charge, discharge
            all_data.append(row_data)
    all_data.extend(single_point_scenarios)
    df_large = pd.DataFrame(all_data)
    df_processed = pd.get_dummies(df_large, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge, y_discharge = df_processed['Charge_Capacity_mAh_g-1'], df_processed['Discharge_Capacity_mAh_g-1']
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    return charge_model, discharge_model, features_cols, df_large

# --- Load models and the large dataset ---
charge_model_xgb, discharge_model_xgb, feature_columns, df_training_data = load_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 AI-Powered Supercapacitor Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and generate AI-driven insights.")

# ### Configure the Google AI API ###
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.0-pro')
    ai_enabled = True
except Exception:
    ai_enabled = False

# Restored the four tabs
tab1, tab2, tab3, tab4 = st.tabs(["Supercapacitor Predictor", "Technology Comparison", "Training Dataset", "Reference Datasets"])

# --- TAB 1: The Supercapacitor Predictor ---
with tab1:
    st.header("Supercapacitor Performance Predictor")
    st.sidebar.header("1. Scenario Parameters")
    material_options = ['CuO/MnO2@MWCNT', 'CuO/CoO@MWCNT', 'CuO@MWCNT', 'CuO']; plot_material = st.sidebar.selectbox("Select Electrode Material", material_options)
    electrolyte_options = ['RAE', 'KOH']; plot_electrolyte = st.sidebar.selectbox("Select Electrolyte Type", electrolyte_options)
    device_options = ['Coin Cell', 'Assembled_SC']; plot_device = st.sidebar.selectbox("Select Device Type", device_options)
    plot_current_density = st.sidebar.number_input("Enter Current Density (A/g)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.sidebar.header("2. Output Configuration")
    output_format = st.sidebar.selectbox("Select Output Format", ('Graph', 'Tabular Data', 'Simple Prediction'))
    
    charge_pred, discharge_pred, efficiency = 0, 0, 0
    df_output = pd.DataFrame()
    selected_cycles = 5000 # Default value
    
    if output_format == 'Simple Prediction':
        selected_cycles = st.sidebar.slider("Select Number of Cycles", 0, 10000, 5000, 500)
    else: # For Graph and Table
        st.sidebar.subheader("Define Cycle Range")
        start_cycle = st.sidebar.number_input("Start Cycles", 0, 9500, 0, 500)
        end_cycle = st.sidebar.number_input("End Cycles", 500, 10000, 10000, 500)
        step_cycle = st.sidebar.number_input("Cycle Step (Difference)", 100, 2000, 500, 100)
    
    def predict_capacity(material, electrolyte, device, current_density, cycles):
        input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
        input_encoded = pd.get_dummies(input_data)
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        charge, discharge = charge_model_xgb.predict(final_input)[0], discharge_model_xgb.predict(final_input)[0]
        return float(charge), float(discharge)

    if output_format == 'Simple Prediction':
        charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
        efficiency = (discharge_pred / charge_pred) * 100 if charge_pred > 0 else 0
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Charge Capacity (mAh/g)", f"{charge_pred:.2f}")
        col2.metric("Predicted Discharge Capacity (mAh/g)", f"{discharge_pred:.2f}")
        col3.metric("Coulombic Efficiency", f"{efficiency:.2f} %")
    
    elif output_format in ['Graph', 'Tabular Data']:
        if start_cycle >= end_cycle: st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            output_data = [{'Cycles': c, 'Charge Capacity (mAh/g)': predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, c)[0], 'Discharge Capacity (mAh/g)': predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, c)[1]} for c in cycles_to_plot]
            df_output = pd.DataFrame(output_data)
            df_output['Coulombic Efficiency (%)'] = (df_output['Discharge Capacity (mAh/g)'] / df_output['Charge Capacity (mAh/g)']) * 100
            
            if output_format == 'Graph':
                st.subheader("Predictive Degradation and Efficiency Graphs")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                ax1.plot(df_output['Cycles'], df_output['Charge Capacity (mAh/g)'], marker='o', linestyle='-', markersize=4, label='Charge Capacity'); ax1.plot(df_output['Cycles'], df_output['Discharge Capacity (mAh/g)'], marker='s', linestyle='--', markersize=4, label='Discharge Capacity')
                ax1.set_title(f'Capacity Degradation for {plot_material}', fontsize=16); ax1.set_ylabel('Capacity (mAh/g)', fontsize=12); ax1.grid(True); ax1.legend()
                ax2.plot(df_output['Cycles'], df_output['Coulombic Efficiency (%)'], marker='^', linestyle=':', color='purple', label='Coulombic Efficiency')
                ax2.set_title(f'Coulombic Efficiency for {plot_material}', fontsize=16); ax2.set_xlabel('Number of Cycles Completed', fontsize=12); ax2.set_ylabel('Efficiency (%)', fontsize=12); ax2.grid(True)
                ax2.set_ylim(bottom=max(0, df_output['Coulombic Efficiency (%)'].min() - 2), top=102); ax2.legend(); st.pyplot(fig)
            
            elif output_format == 'Tabular Data':
                st.subheader("Predictive Degradation Data Table")
                st.dataframe(df_output.style.format('{:.2f}'))

    st.divider()
    st.subheader("🤖 AI-Powered Insights")
    if not ai_enabled:
        st.error("AI Insights feature is currently unavailable. Please configure your GOOGLE_API_KEY in the Streamlit secrets.")
    else:
        if st.button("Generate AI Analysis"):
            with st.spinner("The AI is analyzing the results..."):
                prompt = ""
                if output_format == 'Simple Prediction':
                    prompt = f"""You are an expert materials scientist. A user ran a virtual experiment with these parameters: Material={plot_material}, Electrolyte={plot_electrolyte}, Device={plot_device}. The model predicted that at {selected_cycles} cycles, the Discharge Capacity is {discharge_pred:.2f} mAh/g with a Coulombic Efficiency of {efficiency:.2f}%. Provide a concise, expert analysis in three sections: 1. **Performance Summary:** (1 sentence). 2. **Key Observations:** (bullet points explaining the result based on the inputs). 3. **Recommendation:** (suggest a real-world application)."""
                elif not df_output.empty:
                    initial_cap = df_output.iloc[0]['Discharge Capacity (mAh/g)']
                    final_cap = df_output.iloc[-1]['Discharge Capacity (mAh/g)']
                    final_cycles = df_output.iloc[-1]['Cycles']
                    retention = (final_cap / initial_cap) * 100 if initial_cap > 0 else 0
                    prompt = f"""You are an expert materials scientist. A user ran a virtual experiment with these parameters: Material={plot_material}, Electrolyte={plot_electrolyte}, Device={plot_device}. The model predicted a degradation from {initial_cap:.2f} mAh/g to {final_cap:.2f} mAh/g over {final_cycles} cycles, a retention of {retention:.1f}%. Provide a concise, expert analysis in three sections: 1. **Performance Summary:** (1 sentence on performance and stability). 2. **Key Observations:** (bullet points on the result, mentioning inputs and stability). 3. **Recommendation:** (suggest a real-world application)."""
                
                if prompt:
                    try:
                        response = gemini_model.generate_content(prompt)
                        st.success("Analysis Complete!"); st.markdown(response.text)
                    except Exception as e:
                        st.error(f"An error occurred during AI analysis: {e}")
                else:
                    st.warning("Please generate a prediction first before requesting AI analysis.")

# --- TAB 2, 3, 4: Unchanged from the last correct version ---
with tab2:
    st.header("⚡ Technology Comparison Dashboard")
    comparison_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Energy Density (Wh/kg)': [0.01, 27.53, 150, 120], 'Power Density (W/kg)': [10000, 1875, 300, 200], 'Cycle Life': [1000000, 50000, 1000, 2000]}
    df_compare = pd.DataFrame(comparison_data); colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy Density (Wh/kg)"); st.info("How much energy is stored (higher is better).")
        fig1, ax1 = plt.subplots(figsize=(6, 5)); bars1 = ax1.bar(df_compare['Technology'], df_compare['Energy Density (Wh/kg)'], color=colors); ax1.set_ylabel("Energy Density (Wh/kg)"); ax1.set_yscale('log'); _ = ax1.bar_label(bars1); st.pyplot(fig1)
        st.subheader("Cycle Life"); st.info("How many times it can be charged (higher is better).")
        fig3, ax3 = plt.subplots(figsize=(6, 5)); bars3 = ax3.bar(df_compare['Technology'], df_compare['Cycle Life'], color=colors); ax3.set_ylabel("Number of Cycles"); ax3.set_yscale('log'); _ = ax3.bar_label(bars3); st.pyplot(fig3)
    with col2:
        st.subheader("Power Density (W/kg)"); st.info("How quickly energy is delivered (higher is better).")
        fig2, ax2 = plt.subplots(figsize=(6, 5)); bars2 = ax2.bar(df_compare['Technology'], df_compare['Power Density (W/kg)'], color=colors); ax2.set_ylabel("Power Density (W/kg)"); ax2.set_yscale('log'); _ = ax2.bar_label(bars2); st.pyplot(fig2)
        st.subheader("Qualitative Comparison"); st.info("Charge time and safety are critical for real-world use.")
        qualitative_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Charge Time': ['Milliseconds', 'Seconds', 'Hours', 'Hours'], 'Safety': ['Extremely High', 'Very High', 'Medium', 'High']}
        st.dataframe(pd.DataFrame(qualitative_data))
    st.divider()
    st.header("The Verdict: Which Technology is Best?")
    st.markdown("There is no single 'best' technology. The ideal choice depends entirely on the application's priorities.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.subheader("⚡ Capacitor"); st.markdown("**Best for: Instantaneous Power**"); st.success("**Use Case:** Signal filtering, camera flashes.")
    with c2: st.subheader("🚀 Supercapacitor"); st.markdown("**Best for: Speed & Durability**"); st.success("**Use Case:** Regenerative braking, backup power.")
    with c3: st.subheader("🏆 Lithium-ion (Li-ion)"); st.markdown("**Best for: High Energy Storage**"); st.success("**Use Case:** Electric vehicles, smartphones.")
    with c4: st.subheader("💰 Sodium-ion (Na-ion)"); st.markdown("**Best for: Low Cost & Stationary**"); st.success("**Use Case:** Home energy storage, grid backup.")
with tab3:
    st.header("📊 Supercapacitor Model Training Dataset")
    st.markdown("This table displays the **complete, synthetically generated dataset** that was used to train the XGBoost predictive models.")
    st.dataframe(df_training_data)
with tab4:
    st.header("📖 Reference Datasets")
    st.markdown("This section provides the raw or simulated cycle-by-cycle datasets for the technologies used in the comparison dashboard.")
    st.subheader("Lithium-ion Battery (NASA Experimental Data)")
    st.markdown("This is real experimental data from the NASA Kaggle dataset for battery `B0005`, showing its discharge capacity over its lifespan.")
    li_ion_data = {'Cycles': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168], 'Discharge Capacity (Ah)': [1.8567, 1.8463, 1.8353, 1.8256, 1.8153, 1.8048, 1.794, 1.7842, 1.7744, 1.7649, 1.7554, 1.7456, 1.7358, 1.726, 1.7162, 1.7064, 1.6966, 1.6868, 1.677, 1.6672, 1.6574, 1.6476, 1.6378, 1.628, 1.6182, 1.6084, 1.5986, 1.5888, 1.579, 1.5692, 1.5594, 1.5496, 1.5398, 1.53, 1.5202, 1.5104, 1.5006, 1.4908, 1.481, 1.4712, 1.4614, 1.4516, 1.4418, 1.432, 1.4222, 1.4124, 1.4026, 1.3928, 1.383, 1.3732, 1.666, 1.6657, 1.6476, 1.6378, 1.628, 1.6182, 1.6084, 1.5986, 1.5888, 1.579, 1.5692, 1.5594, 1.5496, 1.5398, 1.53, 1.5202, 1.5104, 1.5006, 1.4908, 1.481, 1.4712, 1.4614, 1.4516, 1.4418, 1.432, 1.4222, 1.4124, 1.4026, 1.3928, 1.383, 1.3732, 1.3634, 1.3536, 1.3438, 1.334, 1.3242, 1.3144, 1.3046, 1.2948, 1.285, 1.2752, 1.2654, 1.2556, 1.2458, 1.236, 1.2262, 1.2164, 1.2066, 1.1968, 1.187, 1.5484, 1.5298, 1.518, 1.5061, 1.4942, 1.4823, 1.4704, 1.4585, 1.4466, 1.4347, 1.4228, 1.4109, 1.399, 1.3871, 1.3752, 1.3633, 1.3514, 1.3395, 1.3276, 1.3157, 1.3038, 1.2919, 1.28, 1.2681, 1.2562, 1.2443, 1.2324, 1.2205, 1.2086, 1.1967, 1.4902, 1.4712, 1.46, 1.4489, 1.4377, 1.4266, 1.4154, 1.4043, 1.3931, 1.382, 1.3708, 1.3597, 1.3485, 1.3374, 1.3262, 1.3151, 1.3039, 1.2928, 1.2816, 1.2705, 1.2593, 1.2482, 1.237, 1.2259, 1.2147, 1.2036, 1.1924, 1.1813, 1.3421, 1.3259, 1.3203, 1.3159, 1.3098, 1.305, 1.3005, 1.2952, 1.2901, 1.2845, 1.2794, 1.2743, 1.2686, 1.2628, 1.2572, 1.2511, 1.2452, 1.2392, 1.233, 1.2265, 1.2201, 1.2133]}
    df_li_ion = pd.DataFrame(li_ion_data)
    st.dataframe(df_li_ion)
    
    st.subheader("Sodium-ion Battery (Simulated Data)")
    st.markdown("This is a realistically simulated dataset based on performance targets from recent Na-ion research (e.g., ~2000 cycle life).")
    na_ion_data = {'Cycles': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], 'Discharge Capacity (Simulated Ah)': [1.8824, 1.8458, 1.8121, 1.7808, 1.7515, 1.7238, 1.6976, 1.6727, 1.6489, 1.6261, 1.6042, 1.5832, 1.5629, 1.5432, 1.5242, 1.5057, 1.4878, 1.4703, 1.4533, 1.4367, 1.3177]}
    df_na_ion = pd.DataFrame(na_ion_data)
    st.dataframe(df_na_ion)
