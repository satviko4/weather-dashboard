"""
DASHBOARD.PY
(Run this script to launch your local dashboard)

This script will:
1. Create a simple web interface using Streamlit.
2. Load your pre-trained model and data on startup.
3. Provide a button to "Run 15-Day Forecast."
4. When clicked, it runs the full autoregressive forecast.
5. Displays the forecast in a table and interactive charts.
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Weather Forecast Dashboard", layout="wide")
st.title("Guwahati 15-Day Weather Forecast")

# --- Configuration (Must match your training script) ---
N_DAYS_FORECAST = 15
N_LAGS = 7
TARGET_VARS = [
    'max_temp_c', 
    'min_temp_c', 
    'total_precip_mm', 
    'max_wind_speed_ms', 
    'avg_humidity_percent'
]

# --- Caching: Load Model and Data Once ---
# @st.cache_resource ensures we don't reload the big model every time
@st.cache_resource
def load_model_and_features():
    """Loads the saved model and feature list."""
    try:
        model = joblib.load('weather_model.joblib')
        feature_cols = joblib.load('feature_columns.joblib')
        return model, feature_cols
    except FileNotFoundError:
        st.error("ERROR: Model files not found. Please run 'train_model.py' first.")
        return None, None

def load_history():
    """Loads the 7-day history file."""
    try:
        history_df = pd.read_csv('last_7_days_data.csv', index_col=0, parse_dates=True)
        return history_df
    except FileNotFoundError:
        st.error("ERROR: 'last_7_days_data.csv' not found. Please run 'train_model.py' first.")
        return None

# --- Core Forecasting Logic (from generate_forecast.py) ---
def generate_forecast(history_df, model, feature_cols):
    """Generates an N-day autoregressive forecast."""
    
    # Start forecasting from the day AFTER the last day in history
    start_date = history_df.index.max() + timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, periods=N_DAYS_FORECAST, freq='D')

    forecast_df = pd.DataFrame(index=forecast_dates, columns=TARGET_VARS, dtype=float)
    loop_history_df = history_df.copy()

    for current_date in forecast_dates:
        # 1. Create date features
        date_features = {
            'month': current_date.month,
            'day_of_year': current_date.dayofyear,
            'year': current_date.year
        }
        
        # 2. Create lag features from history
        lag_features = {}
        history_values = loop_history_df[TARGET_VARS]
        
        for var in TARGET_VARS:
            lags = history_values[var].values[-N_LAGS:][::-1]
            for i in range(1, N_LAGS + 1):
                lag_features[f'{var}_lag{i}'] = lags[i-1]
                
        # 3. Combine and order all features
        current_features_dict = {**date_features, **lag_features}
        current_features = pd.DataFrame([current_features_dict], columns=feature_cols)
        
        # 4. Make prediction
        prediction = model.predict(current_features)[0]
        
        # 5. Store prediction
        forecast_df.loc[current_date] = prediction
        
        # 6. Update history for the next loop
        new_row = pd.DataFrame([prediction], columns=TARGET_VARS, index=[current_date])
        loop_history_df = pd.concat([loop_history_df, new_row]).iloc[1:] # Drop oldest row

    return forecast_df

# --- Load Model and Data ---
model, feature_cols = load_model_and_features()
history_df = load_history()

if model and feature_cols and (history_df is not None):
    
    st.subheader("Current 7-Day History")
    st.write(f"The model will forecast 15 days *after* the last date shown here.")
    st.dataframe(history_df.style.format("{:.2f}"))
    
    # --- The "Run" Button ---
    if st.button("Generate 15-Day Forecast"):
        
        with st.spinner("Running autoregressive forecast..."):
            # 1. Run the forecast
            forecast_results = generate_forecast(history_df, model, feature_cols)
            
            # 2. Display the main data table
            st.subheader("Forecasted Weather (15 Days)")
            st.dataframe(forecast_results.style.format("{:.2f}"))
            
            # 3. Prepare data for charts
            chart_data = forecast_results.reset_index().rename(columns={'index': 'date'})

            # 4. Display Charts
            st.subheader("Visual Forecast")
            
            # Columns for side-by-side charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Temperature Forecast (°C)**")
                # Create a simple line chart for Max and Min Temp
                temp_chart_data = chart_data.melt(
                    id_vars=['date'], 
                    value_vars=['max_temp_c', 'min_temp_c'],
                    var_name='Temperature Type',
                    value_name='Temperature (°C)'
                )
                st.line_chart(temp_chart_data, x='date', y='Temperature (°C)', color='Temperature Type')

                st.write("**Max Wind Speed (m/s)**")
                st.line_chart(chart_data, x='date', y='max_wind_speed_ms', color="#FF4B4B")
            
            with col2:
                st.write("**Total Precipitation (mm)**")
                st.bar_chart(chart_data, x='date', y='total_precip_mm')

                st.write("**Average Humidity (%)**")
                st.line_chart(chart_data, x='date', y='avg_humidity_percent', color="#4BFF4B")

else:
    st.error("One or more essential files are missing. Please ensure all .joblib and .csv files are in the same directory.")