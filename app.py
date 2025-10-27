import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import google.generativeai as genai

# --- CACHED MODEL TRAINING AND DATA GENERATION ---
@st.cache_resource
def load_and_train_models():
    degradation_scenarios = [
        # Truncated, add all your scenarios here as originally authored
        {'config': {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 1.0}, 'start_cycles': 0, 'end_cycles': 5000, 'start_charge': 192.03, 'end_charge': 173.79, 'start_discharge': 182.89, 'end_discharge': 165.51},
        # ... (additional entries unchanged)
        {'config': {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Assembled_SC', 'Current_Density_Ag-1': 0.375}, 'start_cycles': 0, 'end_cycles': 10000, 'start_charge': 6.87, 'end_charge': 3.80, 'start_discharge': 6.54, 'end_discharge': 3.62},
    ]
    single_point_scenarios = [
        # Add all your single-point scenarios here
        {'Electrode_Material': 'CuO/MnO2@MWCNT', 'Electrolyte_Type': 'RAE', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 2.0, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 175.88, 'Discharge_Capacity_mAh_g-1': 167.50},
        # ...
        {'Electrode_Material': 'CuO', 'Electrolyte_Type': 'KOH', 'Device_Type': 'Coin Cell', 'Current_Density_Ag-1': 0.5, 'Cycles_Completed': 0, 'Charge_Capacity_mAh_g-1': 23.48, 'Discharge_Capacity_mAh_g-1': 22.36},
    ]
    all_data = []
    for scenario in degradation_scenarios:
        charge_drop = scenario['start_charge'] - scenario['end_charge']
        discharge_drop = scenario['start_discharge'] - scenario['end_discharge']
        for cycles in range(0, scenario['end_cycles'] + 1, 250):
            cycle_ratio = cycles / scenario['end_cycles'] if scenario['end_cycles'] > 0 else 0
            charge = scenario['start_charge'] - charge_drop * (cycle_ratio ** 0.9)
            discharge = scenario['start_discharge'] - discharge_drop * (cycle_ratio ** 0.9)
            row_data = scenario['config'].copy()
            row_data['Cycles_Completed'] = cycles
            row_data['Charge_Capacity_mAh_g-1'] = charge
            row_data['Discharge_Capacity_mAh_g-1'] = discharge
            all_data.append(row_data)
    all_data.extend(single_point_scenarios)
    df_large = pd.DataFrame(all_data)
    df_processed = pd.get_dummies(df_large, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge = df_processed['Charge_Capacity_mAh_g-1']
    y_discharge = df_processed['Discharge_Capacity_mAh_g-1']
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    return charge_model, discharge_model, features_cols, df_large

charge_model_xgb, discharge_model_xgb, feature_columns, df_training_data = load_and_train_models()

st.set_page_config(layout="wide")
st.title("🔋 AI-Powered Supercapacitor Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and generate AI-driven insights.")

# Set up Gemini (Google Generative AI) model
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')  # Fixed model name!
    ai_enabled = True
except Exception as e:
    st.error(f"Gemini AI API failed to initialize: {e}")
    ai_enabled = False

tab1, tab2, tab3, tab4 = st.tabs(["Supercapacitor Predictor", "General Comparison", "Training Dataset", "Detailed Comparison"])

# --- TAB 1: The Supercapacitor Predictor ---
with tab1:
    st.header("Supercapacitor Performance Predictor")
    st.sidebar.header("1. Scenario Parameters")
    material_options = ['CuO/MnO2@MWCNT', 'CuO/CoO@MWCNT', 'CuO@MWCNT', 'CuO']
    plot_material = st.sidebar.selectbox("Select Electrode Material", material_options)
    electrolyte_options = ['RAE', 'KOH']
    plot_electrolyte = st.sidebar.selectbox("Select Electrolyte Type", electrolyte_options)
    device_options = ['Coin Cell', 'Assembled_SC']
    plot_device = st.sidebar.selectbox("Select Device Type", device_options)
    plot_current_density = st.sidebar.number_input("Enter Current Density (A/g)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.sidebar.header("2. Output Configuration")
    output_format = st.sidebar.selectbox("Select Output Format", ('Graph', 'Tabular Data', 'Simple Prediction'))

    charge_pred, discharge_pred = 0, 0
    df_output = pd.DataFrame()
    selected_cycles = 5000

    if output_format == 'Simple Prediction':
        selected_cycles = st.sidebar.slider("Select Number of Cycles", 0, 10000, 5000, 500)
    else:
        st.sidebar.subheader("Define Cycle Range")
        start_cycle = st.sidebar.number_input("Start Cycles", 0, 9500, 0, 500)
        end_cycle = st.sidebar.number_input("End Cycles", 500, 10000, 10000, 500)
        step_cycle = st.sidebar.number_input("Cycle Step (Difference)", 100, 2000, 500, 100)

    def predict_capacity(material, electrolyte, device, current_density, cycles):
        input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
        input_encoded = pd.get_dummies(input_data)
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        charge = charge_model_xgb.predict(final_input)[0]
        discharge = discharge_model_xgb.predict(final_input)[0]
        return float(charge), float(discharge)

    if output_format == 'Simple Prediction':
        charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Charge Capacity (mAh/g)", f"{charge_pred:.2f}")
        col2.metric("Predicted Discharge Capacity (mAh/g)", f"{discharge_pred:.2f}")

    elif output_format in ['Graph', 'Tabular Data']:
        if start_cycle >= end_cycle:
            st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            output_data = []
            for c in cycles_to_plot:
                charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, c)
                output_data.append({'Cycles': c, 'Charge Capacity (mAh/g)': charge, 'Discharge Capacity (mAh/g)': discharge})
            df_output = pd.DataFrame(output_data)

            if output_format == 'Graph':
                st.subheader("Predictive Degradation Graph")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_output['Cycles'], df_output['Charge Capacity (mAh/g)'], marker='o', linestyle='-', markersize=4, label='Charge Capacity')
                ax.plot(df_output['Cycles'], df_output['Discharge Capacity (mAh/g)'], marker='s', linestyle='--', markersize=4, label='Discharge Capacity')
                ax.set_title(f'Capacity Degradation for {plot_material}', fontsize=16)
                ax.set_xlabel('Number of Cycles Completed', fontsize=12)
                ax.set_ylabel('Capacity (mAh/g)', fontsize=12)
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
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
                    prompt = f"""You are an expert materials scientist. A user ran a virtual experiment with these parameters: Material={plot_material}, Electrolyte={plot_electrolyte}, Device={plot_device}. The model predicted that at {selected_cycles} cycles, the Discharge Capacity is {discharge_pred:.2f} mAh/g. Provide a concise, expert analysis in three sections: 1. **Performance Summary:** (1 sentence). 2. **Key Observations:** (bullet points explaining the result based on the inputs). 3. **Recommendation:** (suggest a real-world application)."""
                elif not df_output.empty:
                    initial_cap = df_output.iloc[0]['Discharge Capacity (mAh/g)']
                    final_cap = df_output.iloc[-1]['Discharge Capacity (mAh/g)']
                    final_cycles = df_output.iloc[-1]['Cycles']
                    retention = (final_cap / initial_cap) * 100 if initial_cap > 0 else 0
                    prompt = f"""You are an expert materials scientist. A user ran a virtual experiment with these parameters: Material={plot_material}, Electrolyte={plot_electrolyte}, Device={plot_device}. The model predicted a degradation from {initial_cap:.2f} mAh/g to {final_cap:.2f} mAh/g over {final_cycles} cycles, a retention of {retention:.1f}%. Provide a concise, expert analysis in three sections: 1. **Performance Summary:** (1 sentence on performance and stability). 2. **Key Observations:** (bullet points on the result, mentioning inputs and stability). 3. **Recommendation:** (suggest a real-world application)."""
                try:
                    response = gemini_model.generate_content(prompt)
                    st.success("Analysis Complete!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred during AI analysis: {e}")

with tab2:
    st.header("⚡ General Technology Comparison Dashboard")
    comparison_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Energy Density (Wh/kg)': [0.01, 27.53, 150, 120], 'Power Density (W/kg)': [10000, 1875, 300, 200], 'Cycle Life': [1000000, 50000, 1000, 2000]}
    df_compare = pd.DataFrame(comparison_data)
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy Density (Wh/kg)")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        bars1 = ax1.bar(df_compare['Technology'], df_compare['Energy Density (Wh/kg)'], color=colors)
        ax1.set_ylabel("Energy Density (Wh/kg)")
        ax1.set_yscale('log')
        ax1.bar_label(bars1)
        st.pyplot(fig1)
        st.subheader("Cycle Life")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        bars3 = ax3.bar(df_compare['Technology'], df_compare['Cycle Life'], color=colors)
        ax3.set_ylabel("Number of Cycles")
        ax3.set_yscale('log')
        ax3.bar_label(bars3)
        st.pyplot(fig3)
    with col2:
        st.subheader("Power Density (W/kg)")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bars2 = ax2.bar(df_compare['Technology'], df_compare['Power Density (W/kg)'], color=colors)
        ax2.set_ylabel("Power Density (W/kg)")
        ax2.set_yscale('log')
        ax2.bar_label(bars2)
        st.pyplot(fig2)
        st.subheader("Qualitative Comparison")
        qualitative_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Charge Time': ['Milliseconds', 'Seconds', 'Hours', 'Hours'], 'Safety': ['Extremely High', 'Very High', 'Medium', 'High']}
        st.dataframe(pd.DataFrame(qualitative_data))
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("⚡ Capacitor")
        st.markdown("**Best for: Instantaneous Power**")
        st.success("**Use Case:** Signal filtering, camera flashes.")
    with c2:
        st.subheader("🚀 Supercapacitor")
        st.markdown("**Best for: Speed & Durability**")
        st.success("**Use Case:** Regenerative braking, backup power.")
    with c3:
        st.subheader("🏆 Lithium-ion (Li-ion)")
        st.markdown("**Best for: High Energy Storage**")
        st.success("**Use Case:** Electric vehicles, smartphones.")
    with c4:
        st.subheader("💰 Sodium-ion (Na-ion)")
        st.markdown("**Best for: Low Cost & Stationary**")
        st.success("**Use Case:** Home energy storage, grid backup.")

with tab3:
    st.header("📊 Supercapacitor Model Training Dataset")
    st.dataframe(df_training_data)

with tab4:
    st.header("⚙️ Detailed Technology Comparison")
    detailed_comparison_data = {
        'Metric': ["Specific Capacitance (F/g)", "Energy Density (Wh/kg)", "Power Density (W/kg)", "Cycle Life", "Charge Time", "Energy Storage Mechanism"],
        'Conventional Capacitor': ["Very Low (~0.0001)", "< 0.1", "> 10,000", "> 1,000,000", "Milliseconds", "Physical (Electric Field)"],
        "This Project's Supercapacitor": ["High (100 - 1,200)", "~27.5", "~1,875", "> 50,000", "Seconds to Minutes", "Physical & Chemical (EDLC + Faradaic)"],
        'Lithium-ion (Li-ion) Battery': ["N/A (Not a capacitor)", "~150", "~300", "~1,000", "Hours", "Chemical (Intercalation)"],
        'Sodium-ion (Na-ion) Battery': ["N/A (Not a capacitor)", "~120", "~200", "~2,000", "Hours", "Chemical (Intercalation)"]
    }
    df_detailed_compare = pd.DataFrame(detailed_comparison_data)
    st.subheader("Key Performance Metrics")
    st.table(df_detailed_compare.set_index('Metric'))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("#### Conventional Capacitor")
        st.markdown("- **Strength:** Extremely high power density (instant speed).\n- **Weakness:** Negligible energy storage.")
    with col2:
        st.warning("#### Supercapacitor")
        st.markdown("- **Strength:** Bridges the gap. High power, very long life, and more energy than a capacitor.\n- **Weakness:** Less energy than a battery.")
    with col3:
        st.error("#### Lithium-ion Battery")
        st.markdown("- **Strength:** High energy density (long runtime).\n- **Weakness:** Slower, shorter life, and higher cost/safety concerns.")
    with col4:
        st.success("#### Sodium-ion Battery")
        st.markdown("- **Strength:** Very low cost and high safety.\n- **Weakness:** Lower energy and power than Li-ion.")
