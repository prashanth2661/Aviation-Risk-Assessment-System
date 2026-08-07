import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.prediction import RiskEngine
from src.config import Config
st.set_page_config(
    page_title="Aviation Risk Assessment System",
    layout="wide"
)
@st.cache_resource
def load_engine():
    return RiskEngine()
try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you have run the training and evaluation scripts.")
    st.stop()
SCENARIOS = {
    "Custom (Manual)": {},
    "Standard Stable Approach": {
        "aircraft_type": "Boeing 747", "altitude_msl_ft": 1500, "vertical_speed_fpm": -500, 
        "airspeed_ias_kts": 145, "wind_speed_kts": 5, "runway_condition": "Dry", 
        "pitch_attitude_deg": 3, "roll_attitude_deg": 0, "gpws_alert": "None"
    },
    "High Sink Rate (SINK RATE!)": {
        "aircraft_type": "Airbus A350", "altitude_msl_ft": 800, "vertical_speed_fpm": -1250, 
        "airspeed_ias_kts": 165, "wind_speed_kts": 12, "runway_condition": "Wet", 
        "pitch_attitude_deg": -2, "roll_attitude_deg": 2, "gpws_alert": "None"
    },
    "Severe Storm / Icy Runway": {
        "aircraft_type": "Cessna 172", "altitude_msl_ft": 400, "vertical_speed_fpm": -700, 
        "airspeed_ias_kts": 75, "wind_speed_kts": 35, "runway_condition": "Icy", 
        "pitch_attitude_deg": 6, "roll_attitude_deg": 12, "gpws_alert": "None"
    },
    "Terrain Conflict (GPWS Alert)": {
        "aircraft_type": "UH-60 Black Hawk", "altitude_msl_ft": 250, "vertical_speed_fpm": -900, 
        "airspeed_ias_kts": 110, "wind_speed_kts": 15, "runway_condition": "Dry", 
        "pitch_attitude_deg": -1, "roll_attitude_deg": 0, "gpws_alert": "Terrain"
    }
}
st.sidebar.title("Simulation Controls")
selected_scenario = st.sidebar.selectbox("Select Flight Scenario", list(SCENARIOS.keys()))
preset = SCENARIOS[selected_scenario]
st.sidebar.divider()
st.sidebar.info("""
**System Architecture:**
- Stacking Ensemble (RandomForestClassifier + XGBClassifier + Support Vector Machine + GradientBoostingClassifier)
- Long Short-Term Memory Sequence Analysis
""")
st.title("Aviation Risk Assessment System")
st.markdown("### Real-time Flight Safety & Pilot Decision Support")
tab1, tab2, tab3 = st.tabs(["Manual Simulation", "Bulk CSV Analysis", "Model Performance"])
with tab1:
    st.subheader("Flight Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Aircraft & Environment")
        ac_type = st.selectbox("Aircraft Type", 
                               ['Boeing 747', 'Airbus A350', 'F-16 Fighting Falcon', 'UH-60 Black Hawk', 'Cessna 172'],
                               index=0 if "aircraft_type" not in preset else ['Boeing 747', 'Airbus A350', 'F-16 Fighting Falcon', 'UH-60 Black Hawk', 'Cessna 172'].index(preset["aircraft_type"]))
        
        runway_list = ['Dry', 'Wet', 'Icy', 'Contaminated']
        runway = st.selectbox("Runway Condition", runway_list, 
                              index=runway_list.index(preset.get("runway_condition", "Dry")))
        wind = st.slider("Wind Speed (kts)", 0, 60, preset.get("wind_speed_kts", 10))
    with col2:
        st.markdown("#### Performance")
        alt = st.number_input("Altitude MSL (ft)", value=preset.get("altitude_msl_ft", 1500))
        v_speed = st.number_input("Vertical Speed (fpm)", value=preset.get("vertical_speed_fpm", -500))
        ias = st.number_input("Airspeed (IAS kts)", value=preset.get("airspeed_ias_kts", 140))
    with col3:
        st.markdown("#### Alerts & Attitude")
        pitch = st.slider("Pitch Angle (deg)", -10, 20, preset.get("pitch_attitude_deg", 3))
        roll = st.slider("Roll Angle (deg)", -45, 45, preset.get("roll_attitude_deg", 0))
        gpws_list = ['None', 'Terrain', 'Runway']
        gpws = st.selectbox("GPWS Status", gpws_list, 
                            index=gpws_list.index(preset.get("gpws_alert", "None")))
    st.divider()
    input_data = {
        "aircraft_type": ac_type, "altitude_msl_ft": alt, "vertical_speed_fpm": v_speed,
        "airspeed_ias_kts": ias, "wind_speed_kts": wind, "runway_condition": runway,
        "pitch_attitude_deg": pitch, "roll_attitude_deg": roll, "gpws_alert": gpws,
        "tcas_alert": preset.get("tcas_alert", "None")
    }
    if st.button("EXECUTE RISK ANALYSIS", use_container_width=True):
        risk, conf, assist = engine.predict(input_data)
        color = "#00cc66" if risk == "Low" else "#ff9900" if risk == "Medium" else "#cc0000"
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
                <div style="background-color:{color}; padding:30px; border-radius:15px; color:white; text-align:center; border: 2px solid rgba(255,255,255,0.1);">
                    <p style="margin:0; font-size:18px; font-weight:bold; opacity:0.8;">PREDICTED RISK</p>
                    <h1 style="margin:0; font-size:48px;">{risk.upper()}</h1>
                    <p style="margin:0; font-size:20px;">{conf*100:.1f}% Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        with res_col2:
            st.subheader(f"Command: {assist['action']}")
            if assist['callouts'] != "MINIMA, CHECKED":
                st.error(f"**AURAL ALERT:** {assist['callouts']}")
            else:
                st.success(f"**STATUS:** {assist['callouts']}") 
            st.info(f"**GUIDANCE:** {assist['detail']}")
            with st.expander("Pilot Corrective Checklist"):
                for item in assist['checklist']:
                    st.write(f"• {item}")
with tab2:
    st.subheader("Batch Flight Processing")
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type="csv")
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if st.button("Run Bulk Analysis"):
            results = []            
            progress_bar = st.progress(0)
            for i, row in df_upload.iterrows():
                risk, conf, assist = engine.predict(row.to_dict())
                results.append({
                    "Predicted_Risk": risk,
                    "Confidence": f"{conf*100:.1f}%",
                    "Action": assist['action']
                })
                progress_bar.progress((i + 1) / len(df_upload))
            res_df = pd.DataFrame(results) 
            final_output = pd.concat([res_df, df_upload], axis=1)
            save_dir = os.path.join("reports", "batch_history")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"report_{timestamp}.csv")
            final_output.to_csv(save_path, index=False)
            st.success(f"Archived to: {save_path}")
            st.dataframe(final_output)
            csv_bytes = final_output.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv_bytes, f"report_{timestamp}.csv", "text/csv")
with tab3:
    st.subheader("Predictive Analytics Accuracy")
    metrics_path = os.path.join(Config.REPORTS_DIR, "model_performance.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.write("**Model Accuracy Comparison**")
            st.bar_chart(m_df.set_index("Model"))  
        with col_m2:
            st.write("**Detailed Metrics**")
            st.table(m_df.style.format({"Accuracy": "{:.2%}"}))  
    else:
        st.warning("No performance metrics found.")