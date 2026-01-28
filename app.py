import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from cleaner import DataJanitor
from model import RevenuePredictor
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_probability_alive_matrix

def run_prediction_engine(df):
    """
    Transform raw transaction data into RFM format and fit BG/NBD model.
    This matches the column names in your uploaded CSV.
    We create a summary table: Recency, Frequency, and T (Age of customer)
    """
    rfm_summary = summary_data_from_transaction_data(
        df, 
        customer_id_col='customer_id', 
        datetime_col='invoicedate', 
        monetary_value_col='price'
    )

    # Initialize and fit the BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T'])

    # Calculate Probability of being "Alive" (not churned)
    rfm_summary['prob_alive'] = bgf.conditional_probability_alive(
        rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T']
    )
    
    return rfm_summary, bgf

def segment_customer_loyalty(rfm_summary):
    """
    Separate customers into two groups based on purchase frequency.
    
    Parameters:
    -----------
    rfm_summary : DataFrame
        RFM summary table from the BG/NBD model
    
    Returns:
    --------
    loyal_customers : DataFrame
        Customers with frequency > 0 (repeat buyers)
    one_time_buyers : DataFrame
        Customers with frequency == 0 (one-time purchases only)
    """
    # 1. Identify Loyal (Repeat) Customers
    loyal_customers = rfm_summary[rfm_summary['frequency'] > 0].copy()
    
    # 2. Identify One-Time Buyers
    one_time_buyers = rfm_summary[rfm_summary['frequency'] == 0].copy()
    
    return loyal_customers, one_time_buyers

@st.cache_data
def convert_df_to_csv(df):
    """
    Convert DataFrame to CSV format for browser download.
    
    Parameters:
    -----------
    df : DataFrame
        The DataFrame to convert
    
    Returns:
    --------
    bytes
        CSV data encoded as UTF-8 bytes
    """
    return df.to_csv(index=True).encode('utf-8')

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
        with st.spinner('Calculating RFM Metrics & Running Predictive Analytics...'):
            # Train model on current data to demonstrate the 'Brain'
            predictor.train(df_clean)
            
            # Generate features (Recency, Frequency, Monetary)
            features = predictor.engineer_features(df_clean)
            
            # Run the prediction
            preds = predictor.predict(features[['recency', 'frequency']])
            features['future_value_score'] = preds
            
            # Run BG/NBD predictive engine
            results, model = run_prediction_engine(df_clean)
            
        st.success("Analysis Complete!")

        # --- Customer Vitality Analysis (BG/NBD Results) ---
        st.write("### 📊 Customer Vitality Score")
        st.caption("Probability that each customer is still 'alive' (active) based on BG/NBD model")
        vitality_df = results[['frequency', 'recency', 'T', 'monetary_value', 'prob_alive']].copy()
        vitality_df = vitality_df.sort_values(by='prob_alive', ascending=False)
        st.dataframe(
            vitality_df.style.format({
                'prob_alive': '{:.1%}',
                'monetary_value': '${:,.2f}',
                'frequency': '{:.0f}',
                'recency': '{:.0f}',
                'T': '{:.0f}'
            }).background_gradient(cmap='RdYlGn', subset=['prob_alive'])
        )

        # --- Probability Matrix Visualization ---
        st.write("### 🔥 Probability Matrix (Recency vs Frequency)")
        st.caption("This heatmap shows the probability of being alive based on purchase behavior")
        fig = plt.figure(figsize=(6, 4))
        plot_probability_alive_matrix(model)
        st.pyplot(fig)

        # --- Customer Loyalty Segmentation ---
        st.markdown("---")
        loyal_df, onetime_df = segment_customer_loyalty(results)

        st.write("## 📊 Customer Base Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Repeat Customers", len(loyal_df))
            st.caption("Customers with 2+ purchases")

        with col2:
            st.metric("One-Time Buyers", len(onetime_df))
            st.caption("Customers with only 1 purchase")

        # Visualizing the difference in "Probability of Being Alive"
        st.write("### 🧬 Health of Loyal Customers")
        st.write("Loyal customers have an average 'Probability of being Alive' of: " + 
                 f"{loyal_df['prob_alive'].mean():.2%}")

        # Show the top loyal customers who are "At Risk"
        st.write("### ⚠️ High-Value Customers At Risk")
        st.caption("Loyal customers with low churn probability (probability < 50%)")
        at_risk = loyal_df[loyal_df['prob_alive'] < 0.5].sort_values(by='monetary_value', ascending=False)
        if len(at_risk) > 0:
            st.dataframe(
                at_risk[['frequency', 'recency', 'monetary_value', 'prob_alive']].head(10).style.format({
                    'prob_alive': '{:.1%}',
                    'monetary_value': '${:,.2f}',
                    'frequency': '{:.0f}',
                    'recency': '{:.0f}'
                }).background_gradient(cmap='Reds', subset=['prob_alive'])
            )
            
            # Create the Download Button for At-Risk Customers
            csv_data = convert_df_to_csv(at_risk)
            
            st.download_button(
                label="📥 Download At-Risk Customer List for Marketing",
                data=csv_data,
                file_name='at_risk_customers.csv',
                mime='text/csv',
                help="Download this list to target these customers with a re-engagement email campaign."
            )
        else:
            st.success("✅ No high-value customers at risk! Your loyal customer base is healthy.")

        # --- STAGE 3: PROFESSIONAL VISUALIZATION ---
        st.markdown("---")
        st.write("### 💰 Revenue Forecast Report")
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