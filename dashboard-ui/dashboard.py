import streamlit as st
import requests
import pandas as pd
import os

# --- Environment Variables for K8s ---
# gateway handles the live predictions
api_url = os.getenv("GATEWAY_URL", "http://api-gateway:8080/route/predict")

# database-service handles the history retrieval
db_history_url = os.getenv("DB_HISTORY_URL", "http://database-service:8000/history")

# --- Page Configuration ---
st.set_page_config(
    page_title="Factory Maintenance AI",
    page_icon="🛠️",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetric"] {
        background-color: #ffffff; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricValue"] div { color: #000000 !important; }
    [data-testid="stMetricLabel"] p { color: #000000 !important; }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] div { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛠️ Machine Health Predictor")

# --- Tabs for Navigation ---
tab1, tab2 = st.tabs(["🚀 Real-Time Diagnostic", "📜 Maintenance Logs"])

with tab1:
    st.info("Real-time diagnostic tool for predictive maintenance on the factory floor.")
    
    # --- Input Section ---
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.subheader("Machine Specs")
            m_type = st.selectbox("Machine Type", ["L", "M", "H"])
            wear = st.number_input("Tool Wear (min)", min_value=0, max_value=500, value=0)
        with col2:
            st.subheader("Temperature")
            air_temp = st.number_input("Air Temp (K)", value=300.0, step=0.1)
            proc_temp = st.number_input("Process Temp (K)", value=310.0, step=0.1)
        with col3:
            st.subheader("Operational")
            speed = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
            torque = st.slider("Torque (Nm)", 0.0, 100.0, 40.0)

    st.divider()
    _, center_col, _ = st.columns([2, 1, 2])
    with center_col:
        run_diagnostic = st.button("🚀 RUN DIAGNOSTIC", use_container_width=True)

    if run_diagnostic:
        payload = {
            "machine_type": m_type, "air_temperature": air_temp,
            "process_temperature": proc_temp, "rotational_speed": speed,
            "torque": torque, "tool_wear": wear
        }
        with st.spinner("Analyzing sensor telemetry..."):
            try:
                response = requests.post(api_url, json=payload, timeout=5)
                response.raise_for_status() 
                result = response.json()
                
                st.subheader("Diagnostic Results")
                res_col1, res_col2 = st.columns(2)

                if result["status"] == "Failure Detected":
                    res_col1.error(f"### ⚠️ {result['status']}")
                    res_col2.metric("Primary Cause", result["failure_cause"], delta="- CRITICAL")
                    st.warning(f"**Action Required:** Technical team should investigate **{result['failure_cause']}**.")
                else:
                    res_col1.success(f"### ✅ {result['status']}")
                    res_col2.metric("Machine State", "NOMINAL", delta="Optimal")
                
                st.info(f"Timestamp: {result['timestamp']}")
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")

with tab2:
    st.subheader("Historical Records")
    st.write("Below are the last 100 machine logs retrieved from the Database Service.")
    
    if st.button("🔄 Refresh History"):
        try:
            with st.spinner("Fetching logs..."):
                resp = requests.get(db_history_url, timeout=5)
                resp.raise_for_status()
                history_data = resp.json()
                
                if history_data:
                    df = pd.DataFrame(history_data)
                    # Styling the dataframe to highlight failures
                    def highlight_failure(val):
                        color = '#ff4b4b' if val == "Failure Detected" else 'transparent'
                        return f'background-color: {color}'
                    
                    st.dataframe(df.style.applymap(highlight_failure, subset=['status']), use_container_width=True)
                else:
                    st.info("The database is currently empty.")
        except Exception as e:
            st.error(f"Could not connect to Database Service: {e}")

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")
    # Simple health check visual
    for name, url in {"Gateway": api_url, "DB Service": db_history_url}.items():
        try:
            # Check the health endpoints (assuming they exist)
            h_url = url.replace("/route/predict", "/health").replace("/history", "/health")
            if requests.get(h_url, timeout=1).status_code == 200:
                st.success(f"● {name} Online")
            else: st.warning(f"● {name} Lagging")
        except: st.error(f"● {name} Offline")
    
    st.divider()
    if st.button("Reset Inputs"): st.rerun()