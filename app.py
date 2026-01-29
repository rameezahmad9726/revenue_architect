import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from cleaner import DataJanitor
from model import RevenuePredictor
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.plotting import plot_probability_alive_matrix

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Revenue Architect Pro", layout="wide")

# --- 2. HELPER FUNCTIONS ---

def run_prediction_engine(df):
    """
    Full Pipeline: RFM + BG/NBD (Vitality) + Gamma-Gamma (Monetary Value)
    """
    # 1. Transform to RFM
    rfm_summary = summary_data_from_transaction_data(
        df, 
        customer_id_col='customer_id', 
        datetime_col='invoicedate', 
        monetary_value_col='price'
    )

    # 2. Fit Vitality Model (BG/NBD)
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T'])

    # 3. Calculate Probability of being "Alive"
    rfm_summary['probability_alive'] = bgf.conditional_probability_alive(
        rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T']
    )

    # 4. Prepare for Monetary Model (Gamma-Gamma)
    # We need Average Order Value (AOV), not just sum.
    # Avoiding division by zero by adding 1 to frequency (total transactions)
    rfm_summary['avg_order_value'] = rfm_summary['monetary_value'] / (rfm_summary['frequency'] + 1)
    
    # Gamma-Gamma requires repeat buyers (frequency > 0) and positive value
    returning_customers = rfm_summary[
        (rfm_summary['frequency'] > 0) & 
        (rfm_summary['avg_order_value'] > 0)
    ]

    # 5. Fit Monetary Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_customers['frequency'], returning_customers['avg_order_value'])

    # 6. Predict CLV (The "Future Value")
    rfm_summary['predicted_clv'] = ggf.customer_lifetime_value(
        bgf, 
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
        rfm_summary['avg_order_value'],
        time=12, # 12 Months
        discount_rate=0.01
    )
    
    return rfm_summary, bgf, ggf

def render_dashboard(rfm):
    """
    Displays the High-Level Executive Summary
    """
    st.markdown("---")
    st.header("📈 Executive Dashboard")
    
    # KPI Calculations
    total_customers = len(rfm)
    active_customers = rfm[rfm['probability_alive'] > 0.5]
    num_active = len(active_customers)
    avg_clv = rfm['predicted_clv'].mean()
    total_clv = rfm['predicted_clv'].sum()
    churn_rate = (1 - (num_active / total_customers)) * 100

    # Layout: 4 Key Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("👥 Active Customers", f"{num_active}", delta=f"{num_active/total_customers:.1%} of Total")
    kpi2.metric("💸 Avg CLV (12M)", f"${avg_clv:.2f}")
    kpi3.metric("💰 Total Pipeline Value", f"${total_clv:,.0f}")
    kpi4.metric("📉 Churn Risk", f"{churn_rate:.1f}%", delta="-High Risk" if churn_rate > 50 else "Stable", delta_color="inverse")

    # Visual Health Check
    st.subheader("Customer Health Distribution")
    rfm['Status'] = rfm['probability_alive'].apply(lambda x: 'Active 🟢' if x >= 0.5 else 'At Risk 🔴')
    status_counts = rfm['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.bar(status_counts, x='Count', y='Status', orientation='h', color='Status',
                 color_discrete_map={'Active 🟢': '#2ecc71', 'At Risk 🔴': '#e74c3c'})
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

def render_simulator(rfm_summary):
    """
    The Strategic "What-If" Simulator
    """
    st.markdown("---")
    st.header("🔮 Strategic Revenue Simulator")

    # --- INPUTS ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Simulation Settings")
        strategy = st.radio("Choose Strategy:", ["🛡️ Defend: Rescue At-Risk", "🚀 Grow: Upsell Loyal"])
        threshold = st.slider("Define 'Alive' Probability Threshold", 0.0, 1.0, 0.5, 0.05)
        
        if "Defend" in strategy:
            impact_rate = st.slider("Recovery Success Rate (%)", 0, 50, 5)
            impact_label = "Recovery Rate"
        else:
            impact_rate = st.slider("Upsell Success Rate (%)", 0, 50, 5)
            impact_label = "AOV Increase"

    # --- CALCULATIONS ---
    if "Defend" in strategy:
        target_segment = rfm_summary[rfm_summary['probability_alive'] < threshold]
        file_prefix = "at_risk"
    else:
        target_segment = rfm_summary[rfm_summary['probability_alive'] >= threshold]
        file_prefix = "loyal_vip"

    total_potential_value = target_segment['predicted_clv'].sum()
    revenue_impact = total_potential_value * (impact_rate / 100)
    count = len(target_segment)

    # --- OUTPUTS ---
    with col2:
        st.subheader("📊 Projected Outcomes")
        st.metric(f"💰 Projected Revenue ({impact_label})", f"${revenue_impact:,.2f}", delta=f"{impact_rate}% Impact")
        st.info(f"Targeting **{count}** customers with total pipeline value of **${total_potential_value:,.2f}**")
        
        # EXPORT BUTTON
        st.markdown("### 📤 Take Action")
        csv = target_segment.to_csv().encode('utf-8')
        st.download_button(
            label=f"Download {file_prefix} List",
            data=csv,
            file_name=f"{file_prefix}_customers.csv",
            mime="text/csv"
        )

# --- 3. MAIN APP LAYOUT ---

st.title("🚀 Intelligent Revenue Architect (v2.0)")
st.markdown("**Production-Ready AI:** Automates data scrubbing, predicts CLV, and simulates ROI strategies.")

# Initialize Classes
janitor = DataJanitor()
predictor = RevenuePredictor()

# Initialize Session State
if 'rfm_summary' not in st.session_state:
    st.session_state.rfm_summary = None

# Sidebar
st.sidebar.header("Data Control Center")
uploaded_file = st.sidebar.file_uploader("Upload Transactional CSV", type="csv")

if uploaded_file:
    # --- STAGE 1: INGESTION ---
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("🛠️ Stage 1: The Data Janitor")
    with st.spinner('Fixing messy data and enforcing schema...'):
        df_clean = janitor.deep_clean(df_raw)
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Raw Input (Messy)")
        st.write(df_raw.head(3))
    with col2:
        st.caption("Cleaned Output")
        st.write(df_clean.head(3))

    # --- STAGE 2: PROCESSING (BUTTON) ---
    st.markdown("---")
    if st.button("🚀 Run Analysis Engine"):
        with st.spinner('Calculating RFM, Fitting BG/NBD, and Predicting CLV...'):
            # 1. Run the heavy math
            rfm, bgf, ggf = run_prediction_engine(df_clean)
            
            # 2. Save to Session State (So it survives slider interactions)
            st.session_state.rfm_summary = rfm
            st.session_state.bgf = bgf
            st.session_state.ggf = ggf
            st.success("Analysis Complete! Dashboard Generated.")

    # --- STAGE 3: DASHBOARD & SIMULATOR (PERSISTENT) ---
    if st.session_state.rfm_summary is not None:
        # Retrieve data from memory
        rfm = st.session_state.rfm_summary
        bgf = st.session_state.bgf
        
        # 1. Render Dashboard
        render_dashboard(rfm)
        
        # 2. Render Simulator
        render_simulator(rfm)
        
        # 3. Optional: Show Technical Model Plots
        with st.expander("Show Technical Model Diagnostics"):
            st.write("### Probability Alive Matrix")
            fig = plt.figure(figsize=(6, 4))
            plot_probability_alive_matrix(bgf)
            st.pyplot(fig)
        
        # --- 4. DETAILED DATA INSPECTOR (Customer Prediction Log) ---
        st.markdown("---")
        st.subheader("🔍 Customer Prediction Log")
        st.caption("View individual customer predictions. Columns are sortable.")

        # Format the dataframe for display
        display_df = rfm.copy()
        
        # We style it: Green = High Probability Alive, Red = Low Probability (Churn Risk)
        st.dataframe(
            display_df[['frequency', 'recency', 'T', 'monetary_value', 'probability_alive', 'predicted_clv']]
            .sort_values(by='probability_alive', ascending=False)
            .style.format({
                'probability_alive': '{:.1%}',      # e.g., 95.2%
                'predicted_clv': '${:,.2f}',        # e.g., $150.00
                'monetary_value': '${:,.2f}',
                'frequency': '{:.0f}',
                'recency': '{:.0f} days'
            })
            .background_gradient(cmap='RdYlGn', subset=['probability_alive'])  # Red/Yellow/Green gradient
        )
            
    else:
        st.info("👆 Click 'Run Analysis Engine' to generate insights.")

else:
    st.info("👋 Welcome! Please upload your transaction CSV to begin.")