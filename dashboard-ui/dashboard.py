import streamlit as st
import requests
import pandas as pd
import os

api_url = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000/predict")

# --- Page Configuration ---
st.set_page_config(
    page_title="Factory Maintenance AI",
    page_icon="🛠️",
    layout="wide", # Wider layout feels more like a dashboard
)

# --- Custom CSS for Industrial Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    
    /* 1. Set the background card to White */
    [data-testid="stMetric"] {
        background-color: #ffffff; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0; /* Adds a light border for visibility on white background */
    }
    
    /* 2. Set the Number (Value) to Black */
    [data-testid="stMetricValue"] div {
        color: #000000 !important;
    }

    /* 3. Set the Label (Title) to Black/Dark Grey */
    [data-testid="stMetricLabel"] p {
        color: #000000 !important;
    }

    /* 4. Set the 'CRITICAL' / 'NOMINAL' (Delta) text to Black */
    [data-testid="stMetricDelta"] svg {
        display: none; /* Optional: hides the up/down arrow icon for a cleaner look */
    }
    [data-testid="stMetricDelta"] div {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🛠️ Machine Health Predictor")
st.info("Real-time diagnostic tool for predictive maintenance on the factory floor.")

# --- Sidebar: Configuration & Reference ---
with st.sidebar:
    st.header("Settings & Metadata")
    st.write("**Model Version:** v2.4.1-stable")
    st.write("**Environment:** Kubernetes Production")
    st.divider()
    if st.button("Reset Inputs"):
        st.rerun()

# --- Input Section ---
# Using a container to group inputs visually
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Machine Specs")
        m_type = st.selectbox("Machine Type", ["L", "M", "H"], help="L: Low, M: Medium, H: High quality grade")
        wear = st.number_input("Tool Wear (min)", min_value=0, max_value=500, value=0)

    with col2:
        st.subheader("Temperature")
        air_temp = st.number_input("Air Temp (K)", value=300.0, step=0.1)
        proc_temp = st.number_input("Process Temp (K)", value=310.0, step=0.1)

    with col3:
        st.subheader("Operational")
        speed = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
        torque = st.slider("Torque (Nm)", 0.0, 100.0, 40.0)

# --- Logic & API Call ---
st.divider()

# Center the button
_, center_col, _ = st.columns([2, 1, 2])

with center_col:
    run_diagnostic = st.button("🚀 RUN DIAGNOSTIC", use_container_width=True)

if run_diagnostic:
    payload = {
        "machine_type": m_type, 
        "air_temperature": air_temp,
        "process_temperature": proc_temp, 
        "rotational_speed": speed,
        "torque": torque, 
        "tool_wear": wear
    }

    with st.spinner("Analyzing sensor telemetry..."):
        try:
            # Tip: Replace 'localhost' with an environment variable for K8s deployment
            response = requests.post(api_url, json=payload, timeout=5)
            response.raise_for_status() 
            result = response.json()
            
            # --- Results Display ---
            st.subheader("Diagnostic Results")
            res_col1, res_col2 = st.columns(2)

            # 1. Handle Status and Icons
            if result["status"] == "Failure Detected":
                res_col1.error(f"### ⚠️ {result['status']}")
                
                # 2. Display the Specific Cause from the Specialist Model
                # We use a metric or a large subheader for the cause
                res_col2.metric("Primary Cause", result["failure_cause"], delta="- CRITICAL", delta_color="normal")
                
                st.warning(f"**Action Required:** Technical team should investigate **{result['failure_cause']}** immediately.")
                st.info(f"Report generated at: {result['timestamp']}")
            else:
                res_col1.success(f"### ✅ {result['status']}")
                res_col2.metric("Machine State", "NOMINAL", delta="Optimal")
                st.write(f"Last checked: {result['timestamp']}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Connection Error: {e}")
            st.info("Check if your FastAPI server is running (uvicorn main:app).")

# --- Footer/Data Preview ---
if 'payload' in locals():
    with st.expander("View Raw Input JSON"):
        st.json(payload)
else:
    with st.expander("View Raw Input JSON"):
        st.write("Run a diagnostic to see the request payload.")