import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px  # Added for advanced charting

# --- Environment Variables for K8s ---
base_url = os.getenv("GATEWAY_BASE_URL", "http://api-gateway:8080")

api_url = f"{base_url}/predict"
history_url = f"{base_url}/history"

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
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Diagnostic", "📜 Maintenance Logs", "📊 Analytics Dashboard"])

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
                resp = requests.get(history_url, timeout=5)
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

with tab3:
    st.subheader("📊 Fleet Analytics Dashboard")
    st.write("Visual breakdown of machine performance and failure correlations.")

    # Fetch data for charts
    try:
        resp = requests.get(history_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])

            # --- Row 1: Executive Summary ---
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 1. Fleet Health Status")
                # Pie Chart for Nominal vs Failure
                fig_pie = px.pie(df, names='status', hole=0.4, 
                                 color='status', 
                                 color_discrete_map={'Healthy': '#2ecc71', 'Failure Detected': '#e74c3c'})
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                st.markdown("### 2. Failure Distribution")
                # Filter only failures for the bar chart
                fail_df = df[df['failure_cause'] != True]
                if not fail_df.empty:
                    fig_bar = px.bar(fail_df['failure_cause'].value_counts().reset_index(), 
                                     x='count', y='failure_cause', orientation='h',
                                     labels={'count': 'Incidents', 'failure_cause': 'Reason'},
                                     color='failure_cause')
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.success("No failures recorded in history!")

            st.divider()

            # --- Row 2: Correlation Analysis ---
            col_c, col_d = st.columns(2)

            with col_c:
                st.markdown("### 3. Thermal Danger Zones")
                # Scatter Plot: Air vs Process Temp
                fig_scatter = px.scatter(df, x='air_temperature', y='process_temperature', 
                                         color='status', hover_data=['machine_type'],
                                         title="Air Temp vs Process Temp Correlation")
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_d:
                st.markdown("### 4. Mechanical Stress (Wear vs Torque)")
                # Bubble Chart: Tool Wear vs Torque
                fig_bubble = px.scatter(df, x='tool_wear', y='torque', 
                                        size='rotational_speed', color='status',
                                        title="Tool Wear vs Torque (Size=RPM)")
                st.plotly_chart(fig_bubble, use_container_width=True)

            st.divider()

            # --- Row 3: Telemetry Over Time ---
            st.markdown("### 5. Sensor Telemetry Drift (Recent Logs)")
            # Line Chart for Trends
            chart_df = df.sort_values("timestamp").tail(50) # Show last 50 for clarity
            st.line_chart(chart_df.set_index('timestamp')[['rotational_speed', 'torque']])
            
        else:
            st.warning("No data available to generate charts. Run some diagnostics first!")
            
    except Exception as e:
        st.error(f"Analytics Error: {e}")

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")

    for name, url in {"Gateway": f"{base_url}/health"}.items():
        try:
            if requests.get(url, timeout=1).status_code == 200:
                st.success(f"● {name} Online")
            else:
                st.warning(f"● {name} Lagging")
        except:
            st.error(f"● {name} Offline")

    st.divider()
    if st.button("Reset Inputs"):
        st.rerun()