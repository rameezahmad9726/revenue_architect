import streamlit as st
import pandas as pd
import os
import joblib
from cleaner import DataJanitor
from model import RevenuePredictor

# 1. Page Config - The first thing the user sees
st.set_page_config(page_title="Revenue Architect Pro", layout="wide")

st.title("🚀 Intelligent Revenue Architect")
st.markdown("""
    **Production-Ready AI:** This system automates data scrubbing and uses 
    Machine Learning to forecast future customer value.
    ---
""")

# 2. Initialize our Logic Classes
janitor = DataJanitor()
predictor = RevenuePredictor()

# 3. Sidebar for Data Ingestion
st.sidebar.header("Data Control Center")
uploaded_file = st.sidebar.file_uploader("Upload Transactional CSV", type="csv")

if uploaded_file:
    # --- STAGE 1: INGESTION & CLEANING ---
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("🛠️ Stage 1: The Data Janitor")
    with st.spinner('Fixing messy data and enforcing schema...'):
        # This handles the '100invalid120' style errors automatically
        df_clean = janitor.deep_clean(df_raw)
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Raw Input (Messy)")
        st.write(df_raw.head())
    with col2:
        st.caption("Cleaned Output (Professional)")
        st.write(df_clean.head())

    # --- STAGE 2: MACHINE LEARNING ---
    st.markdown("---")
    st.subheader("🧠 Stage 2: Revenue Forecasting")
    
    if st.button("Run Prediction Engine"):
        with st.spinner('Calculating RFM Metrics & Running Random Forest...'):
            # Train model on current data to demonstrate the 'Brain'
            predictor.train(df_clean)
            
            # Generate features (Recency, Frequency, Monetary)
            features = predictor.engineer_features(df_clean)
            
            # Run the prediction
            preds = predictor.predict(features[['recency', 'frequency']])
            features['future_value_score'] = preds
            
        st.success("Analysis Complete!")

        # --- STAGE 3: PROFESSIONAL VISUALIZATION ---
        # We sort by highest predicted value first
        display_df = features.sort_values(by='future_value_score', ascending=False)
        
        # This fixes the 'concerning' decimals and adds currency symbols
        st.dataframe(
            display_df.style.format({
                'monetary': '${:,.2f}',
                'future_value_score': '${:,.2f}',
                'date': lambda x: x.strftime('%Y-%m-%d'), # This removes the time!
                'recency': '{:.0f} days',
                'frequency': '{:.0f} orders'
            }).background_gradient(cmap='Greens', subset=['future_value_score'])
        )
        
        # Download Link for the business user
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📩 Download Prediction Report",
            data=csv,
            file_name='revenue_forecast_report.csv',
            mime='text/csv',
        )
else:
    st.info("👋 Welcome! Please upload a CSV with 'customer_id', 'revenue', and 'date' columns to begin.")