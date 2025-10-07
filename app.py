import streamlit as st
import requests
import json
import time
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# --- AI Integration: Configuration, Data, and API Call Function ---

# NOTE: For security in a production environment, the API_KEY is intentionally set to ""
# and is expected to be provided by the Canvas runtime or a Streamlit Secret.
API_KEY = "" 
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# Hardcoded performance data to guide the AI's reasoning
TECH_DATA = """
## Energy Storage Technology Performance Summary

| Technology | Storage Mechanism | Typical Energy Density (Wh/kg) | Power Density (W/kg) | Cycle Life (Cycles) | Key Strengths | Key Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lithium-Ion (Li-ion)** | Electrochemical (Intercalation) | 150 - 250 | Moderate (500 - 2,000) | Moderate (1,000 - 3,000) | Good balance, market maturity, high volumetric density. | Limited ultimate energy density, safety concerns. |
| **Lithium-Sulfur (Li-S)** | Electrochemical (Conversion) | 400 - 500 (Practical) | Moderate (500 - 1,500) | Low (100 - 500) | Extremely high theoretical energy density, low-cost materials. | Polysulfide shuttle effect, short cycle life, volume expansion. |
| **Supercapacitor (SC)** | Electrostatic (Double-layer) | 5 - 20 | Very High (up to 10,000+) | Excellent (> 1,000,000) | Ultra-fast charge/discharge, very long life, high efficiency. | Very low energy density (cannot run device for long). |
"""

def call_gemini_api(system_instruction, user_query):
    """Calls the Gemini API with exponential backoff and comprehensive error handling."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "tools": [{"google_search": {}}], # Enable Google Search Grounding for up-to-date data
    }
    
    # Simple Exponential Backoff for retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract generated text
            candidate = result.get('candidates', [])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'API response structure unexpected.')
            
            # Extract grounding sources
            sources = []
            grounding_metadata = candidate.get('groundingMetadata', {})
            attributions = grounding_metadata.get('groundingAttributions', [])
            
            if attributions:
                sources = [{
                    'uri': attr.get('web', {}).get('uri', 'N/A'),
                    'title': attr.get('web', {}).get('title', 'N/A')
                } for attr in attributions if attr.get('web', {}).get('uri')]

            return text, sources

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"An HTTP error occurred: {e}. Status: {response.status_code}. Please check your API key or deployment environment."
                print(error_msg)
                return error_msg, []
        except requests.exceptions.RequestException as e:
            error_msg = f"A network error occurred: {e}. Check your internet connection."
            print(error_msg)
            return error_msg, []
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(error_msg)
            return error_msg, []
            
    return "Failed to get a response from the API after multiple retries.", []


# --- Existing Model Training and Data Loading ---

@st.cache_resource
def load_and_train_models():
    """Generates synthetic data and trains XGBoost models for capacity prediction."""
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
            # Apply a non-linear (power law) degradation curve for realistic simulation
            charge = scenario['start_charge'] - charge_drop * (cycle_ratio ** 0.9) 
            discharge = scenario['start_discharge'] - discharge_drop * (cycle_ratio ** 0.9)
            row_data = scenario['config'].copy()
            row_data['Cycles_Completed'], row_data['Charge_Capacity_mAh_g-1'], row_data['Discharge_Capacity_mAh_g-1'] = cycles, charge, discharge
            all_data.append(row_data)
    
    all_data.extend(single_point_scenarios)
    df_large = pd.DataFrame(all_data)
    
    # Feature Engineering for model input
    df_processed = pd.get_dummies(df_large, columns=['Electrode_Material', 'Electrolyte_Type', 'Device_Type'])
    features_cols = df_processed.drop(columns=['Charge_Capacity_mAh_g-1', 'Discharge_Capacity_mAh_g-1']).columns
    y_charge, y_discharge = df_processed['Charge_Capacity_mAh_g-1'], df_processed['Discharge_Capacity_mAh_g-1']
    
    # Train XGBoost models
    charge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_charge)
    discharge_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(df_processed[features_cols], y_discharge)
    
    return charge_model, discharge_model, features_cols, df_large

# --- Load models and the large dataset ---
charge_model_xgb, discharge_model_xgb, feature_columns, df_training_data = load_and_train_models()

# --- WEB APPLICATION INTERFACE ---
st.set_page_config(layout="wide")
st.title("🔋 Supercapacitor & Battery Technology Analyzer")
st.markdown("A Capstone Project to predict supercapacitor performance and compare it against other energy storage technologies.")

# ### Define Tabs ###
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Supercapacitor Predictor", "General Comparison", "Training Dataset", "Detailed Comparison", "AI Advisor"])

# --- TAB 1: The Supercapacitor Predictor ---
with tab1:
    st.header("Supercapacitor Performance Predictor")
    
    # Sidebar inputs
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
    unit_choice = st.sidebar.radio("Select Output Units", ('mAh/g', 'C/g'))
    
    if output_format in ['Graph', 'Tabular Data']:
        value_type = st.sidebar.radio("Select Value Type", ('Absolute Values', 'Percentage Retention'))
        st.sidebar.subheader("Define Cycle Range")
        start_cycle = st.sidebar.number_input("Start Cycles", 0, 9500, 0, 500)
        end_cycle = st.sidebar.number_input("End Cycles", 500, 10000, 10000, 500)
        step_cycle = st.sidebar.number_input("Cycle Step (Difference)", 100, 2000, 500, 100)
    
    def predict_capacity(material, electrolyte, device, current_density, cycles):
        """Prepares input data, one-hot encodes, aligns with feature columns, and predicts."""
        input_data = pd.DataFrame({'Current_Density_Ag-1': [current_density], 'Cycles_Completed': [cycles], 'Electrode_Material': [material], 'Electrolyte_Type': [electrolyte], 'Device_Type': [device]})
        input_encoded = pd.get_dummies(input_data)
        # Reindex to ensure all 0-value feature columns from training are present
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        charge, discharge = charge_model_xgb.predict(final_input)[0], discharge_model_xgb.predict(final_input)[0]
        return float(charge), float(discharge)

    if output_format == 'Simple Prediction':
        st.subheader("Simple Prediction for a Single Point")
        selected_cycles = st.slider("Select Number of Cycles to Predict", 0, 10000, 5000, 500, key="slider_tab1")
        
        charge_pred, discharge_pred = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, selected_cycles)
        
        if unit_choice == 'C/g': 
            # Conversion from mAh/g to C/g (1 mAh/g = 3.6 C/g)
            charge_pred *= 3.6
            discharge_pred *= 3.6
            
        col1, col2 = st.columns(2)
        col1.metric("Predicted Charge Capacity", f"{charge_pred:.2f} {unit_choice}")
        col2.metric("Predicted Discharge Capacity", f"{discharge_pred:.2f} {unit_choice}")
        
    else:
        # Graph or Tabular Data generation
        if start_cycle >= end_cycle: 
            st.error("Error: 'Start Cycles' must be less than 'End Cycles'.")
        else:
            cycles_to_plot = list(range(start_cycle, end_cycle + 1, step_cycle))
            output_data = []
            
            initial_charge, initial_discharge = 1, 1
            if value_type == 'Percentage Retention': 
                # Get baseline capacity at 0 cycles
                initial_charge, initial_discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, 0)
            
            for cycle in cycles_to_plot:
                charge, discharge = predict_capacity(plot_material, plot_electrolyte, plot_device, plot_current_density, cycle)
                
                if value_type == 'Percentage Retention':
                    charge = (charge / initial_charge) * 100 if initial_charge > 0 else 0
                    discharge = (discharge / initial_discharge) * 100 if initial_discharge > 0 else 0
                
                output_data.append({'Cycles': cycle, 'Charge Capacity': charge, 'Discharge Capacity': discharge})
            
            df_output = pd.DataFrame(output_data)
            
            if unit_choice == 'C/g' and value_type == 'Absolute Values': 
                df_output['Charge Capacity'] *= 3.6
                df_output['Discharge Capacity'] *= 3.6
                
            ylabel = f'Capacity ({unit_choice})' if value_type == 'Absolute Values' else 'Capacity Retention (%)'
            
            if output_format == 'Graph':
                st.subheader(f"Predictive Degradation Graph ({value_type})")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(df_output['Cycles'], df_output['Charge Capacity'], marker='o', linestyle='-', markersize=4, label='Predicted Charge Capacity')
                ax.plot(df_output['Cycles'], df_output['Discharge Capacity'], marker='s', linestyle='--', markersize=4, label='Predicted Discharge Capacity')
                
                ax.set_title(f'Prediction for {plot_material} ({plot_electrolyte})', fontsize=16)
                ax.set_xlabel('Number of Cycles Completed', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                
                if value_type == 'Percentage Retention':
                    ax.set_yticks(np.arange(0, 101, 10))
                    ax.set_ylim(bottom=0, top=105)
                
                ax.grid(True)
                _ = ax.legend()
                st.pyplot(fig)
                
            elif output_format == 'Tabular Data':
                st.subheader(f"Predictive Degradation Data Table ({value_type})")
                st.dataframe(df_output.style.format({'Charge Capacity': '{:.2f}', 'Discharge Capacity': '{:.2f}', 'Cycles': '{}'}))

# --- TAB 2: The General Technology Comparison page ---
with tab2:
    st.header("⚡ General Technology Comparison Dashboard")
    st.markdown("This dashboard compares key performance metrics across the full spectrum of energy storage technologies.")
    
    comparison_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Energy Density (Wh/kg)': [0.01, 27.53, 150, 120], 'Power Density (W/kg)': [10000, 1875, 300, 200], 'Cycle Life': [1000000, 50000, 1000, 2000]}
    df_compare = pd.DataFrame(comparison_data)
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Density (Wh/kg)"); st.info("How much energy is stored (higher is better).")
        fig1, ax1 = plt.subplots(figsize=(6, 5)); 
        bars1 = ax1.bar(df_compare['Technology'], df_compare['Energy Density (Wh/kg)'], color=colors); 
        ax1.set_ylabel("Energy Density (Wh/kg)"); 
        ax1.set_yscale('log'); # Use log scale due to large difference in Capacitor energy
        _ = ax1.bar_label(bars1); 
        st.pyplot(fig1)
        
        st.subheader("Cycle Life"); st.info("How many times it can be charged (higher is better).")
        fig3, ax3 = plt.subplots(figsize=(6, 5)); 
        bars3 = ax3.bar(df_compare['Technology'], df_compare['Cycle Life'], color=colors); 
        ax3.set_ylabel("Number of Cycles"); 
        ax3.set_yscale('log'); # Use log scale due to large difference in Capacitor cycle life
        _ = ax3.bar_label(bars3); 
        st.pyplot(fig3)
        
    with col2:
        st.subheader("Power Density (W/kg)"); st.info("How quickly energy is delivered (higher is better).")
        fig2, ax2 = plt.subplots(figsize=(6, 5)); 
        bars2 = ax2.bar(df_compare['Technology'], df_compare['Power Density (W/kg)'], color=colors); 
        ax2.set_ylabel("Power Density (W/kg)"); 
        ax2.set_yscale('log'); # Use log scale due to large difference in Capacitor power
        _ = ax2.bar_label(bars2); 
        st.pyplot(fig2)
        
        st.subheader("Qualitative Comparison"); st.info("Charge time and safety are critical for real-world use.")
        qualitative_data = {'Technology': ['Conventional Capacitor', 'This Project\'s Supercapacitor', 'Lithium-ion (Li-ion)', 'Sodium-ion (Na-ion)'], 'Charge Time': ['Milliseconds', 'Seconds', 'Hours', 'Hours'], 'Safety': ['Extremely High', 'Very High', 'Medium', 'High']}
        st.dataframe(pd.DataFrame(qualitative_data))
        
    st.divider()
    
    st.header("The Verdict: Which Technology is Best?")
    st.markdown("There is no single 'best' technology. The ideal choice depends entirely on the application's priorities.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.subheader("⚡ Capacitor"); st.markdown("**Best for: Instantaneous Power**"); st.success("**Use Case:** Signal filtering, camera flashes.")
    with c2: st.subheader("🚀 Supercapacitor"); st.markdown("**Best for: Speed & Durability**"); st.success("**Use Case:** Regenerative braking, backup power.")
    with c3: st.subheader("🏆 Lithium-ion (Li-ion)"); st.markdown("**Best for: High Energy Storage**"); st.error("**Use Case:** Electric vehicles, smartphones.") 
    with c4: st.subheader("💰 Sodium-ion (Na-ion)"); st.markdown("**Best for: Low Cost & Stationary**"); st.success("**Use Case:** Home energy storage, grid backup.")

# --- TAB 3: The Training Dataset Viewer ---
with tab3:
    st.header("📊 Supercapacitor Model Training Dataset")
    st.markdown("This table displays the **complete, synthetically generated dataset** that was used to train the XGBoost predictive models.")
    st.dataframe(df_training_data)

# --- TAB 4: Detailed Comparison ---
with tab4:
    st.header("⚙️ Detailed Technology Comparison")
    st.markdown("This dashboard provides a detailed, side-by-side comparison of the key performance metrics for all four energy storage technologies.")

    # --- Data for detailed comparison ---
    detailed_comparison_data = {
        'Metric': [
            "Specific Capacitance (F/g)",
            "Energy Density (Wh/kg)",
            "Power Density (W/kg)",
            "Cycle Life",
            "Charge Time",
            "Energy Storage Mechanism"
        ],
        'Conventional Capacitor': [
            "Very Low (~0.0001)",
            "< 0.1",
            "> 10,000",
            "> 1,000,000",
            "Milliseconds",
            "Physical (Electric Field)"
        ],
        'This Project\'s Supercapacitor': [
            "High (100 - 1,200)",
            "~27.5",
            "~1,875",
            "> 50,000",
            "Seconds to Minutes",
            "Physical & Chemical (EDLC + Faradaic)"
        ],
        'Lithium-ion (Li-ion) Battery': [
            "N/A (Not a capacitor)",
            "~150",
            "~300",
            "~1,000",
            "Hours",
            "Chemical (Intercalation)"
        ],
        'Sodium-ion (Na-ion) Battery': [
            "N/A (Not a capacitor)",
            "~120",
            "~200",
            "~2,000",
            "Hours",
            "Chemical (Intercalation)"
        ]
    }
    df_detailed_compare = pd.DataFrame(detailed_comparison_data)

    st.subheader("Key Performance Metrics")
    st.markdown("This table highlights the fundamental trade-offs between these devices. Notice the orders-of-magnitude differences.")
    st.table(df_detailed_compare.set_index('Metric'))
    
    st.subheader("Core Differences Explained")
    
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

# --- TAB 5: AI Advisor ---
with tab5:
    st.header("💡 AI Energy Storage Recommendation Engine")
    st.markdown("Define your project's technical requirements, and the Gemini AI will recommend the best energy storage technology (Li-ion, Li-S, or Supercapacitor).")
    
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("1. Define Priorities")
        
        priority = st.selectbox(
            "Primary Goal/Priority",
            ["Maximize Range/Runtime (Energy Density)", "Maximize Power Output (Fast Charge/Discharge)", "Maximize Lifespan (Cycle Life)"],
            index=0,
            help="Select the single most important factor for your application."
        )
        
        cost_sensitivity = st.radio(
            "Cost Sensitivity",
            ["High", "Medium", "Low"],
            index=1,
            help="How critical is the initial manufacturing cost?"
        )
        
        application_description = st.text_area(
            "Application Context (e.g., Drone, Grid Storage, EV)",
            "A long-duration battery for a commercial delivery drone.",
            height=100
        )
    
    with col_b:
        st.subheader("2. Define Targets")

        required_energy_density = st.slider(
            "Target Energy Density (Wh/kg)",
            min_value=5, max_value=550, value=200, step=5,
            help="Energy stored per unit mass. Higher means longer device runtime or range."
        )
        
        required_power_density = st.slider(
            "Target Power Density (W/kg)",
            min_value=100, max_value=10000, value=1500, step=100,
            help="Power delivered per unit mass. Higher means faster charging/discharging or acceleration."
        )
        
        required_cycle_life = st.number_input(
            "Target Cycle Life (Minimum Cycles)",
            min_value=100, max_value=1000000, value=1000, step=100,
            help="The minimum number of full charge-discharge cycles required over the device's lifetime."
        )

    st.markdown("---")
    if st.button("Generate AI Recommendation 🚀", use_container_width=True, key="ai_recommendation_button"):
        
        # 3. Define System Instruction and User Query
        system_instruction = f"""
        You are a highly experienced Energy Storage Consultant specializing in Lithium-Ion, Lithium-Sulfur, and Supercapacitor technologies.
        Your task is to provide a single, concise recommendation and a detailed justification based on the user's requirements.
        
        Use the following technical data to ground your analysis. Do not mention this table directly in the final output, but use the facts within it:
        {TECH_DATA}
        
        Follow these steps:
        1. **Identify the single best technology** (Li-ion, Li-S, or Supercapacitor) that meets or comes closest to the user's **Primary Goal** while acknowledging necessary trade-offs for other requirements.
        2. **Explain the selected technology's working principle** (Electrostatic vs. Electrochemical) in simple terms.
        3. **Justify the recommendation** by comparing the chosen technology's performance (Energy Density, Power Density, Cycle Life) against the user's specific targets and the competing technologies.
        4. If a single technology cannot meet all needs (e.g., high power *and* high energy), suggest a **Hybrid Solution** (Battery + Supercapacitor).
        """
        
        user_query = f"""
        Analyze the best energy storage solution for the following application based on the requirements below.

        Application Context: {application_description}

        Primary Priority: {priority}
        Target Energy Density: {required_energy_density} Wh/kg
        Target Power Density: {required_power_density} W/kg
        Target Cycle Life: {required_cycle_life} cycles
        Cost Sensitivity: {cost_sensitivity}
        
        Provide the final recommendation and justification directly.
        """
        
        # 4. Call API and Display Results
        with st.spinner("Analyzing requirements and generating recommendation with Gemini..."):
            recommendation_text, sources = call_gemini_api(system_instruction, user_query)

        st.subheader("💡 AI Recommendation Result")
        st.info("The analysis below is tailored to your specific project priorities.")
        st.markdown(recommendation_text)
        
        if sources:
            st.subheader("Sources Consulted")
            for i, source in enumerate(sources):
                st.markdown(f"{i+1}. [{source['title']}]({source['uri']})")
        
        st.markdown("---")
