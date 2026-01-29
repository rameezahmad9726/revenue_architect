import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from cleaner import DataJanitor
from model import RevenuePredictor
from lifetimes import BetaGeoFitter, GammaGammaFitter
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

def calculate_clv_with_gamma_gamma(rfm_summary, bgf):
    """
    Calculate Customer Lifetime Value using the Gamma-Gamma model.
    
    This combines:
    - BG/NBD model: Probability of being alive (churn prediction)
    - Gamma-Gamma model: Expected monetary value from transactions
    
    Parameters:
    -----------
    rfm_summary : DataFrame
        RFM summary from BG/NBD model with columns: frequency, recency, T, monetary_value
    bgf : BetaGeoFitter
        Fitted BG/NBD model
    
    Returns:
    --------
    rfm_summary : DataFrame
        Updated with avg_order_value and predicted_clv columns
    ggf : GammaGammaFitter
        Fitted Gamma-Gamma model
    """
    # Make a copy to avoid modifying the original
    rfm_summary = rfm_summary.copy()
    
    # 1. Prepare the Data for Gamma-Gamma
    # We need Average Order Value (AOV), not just the Sum.
    # We assume 'frequency' is repeat purchases, so total transactions = frequency + 1
    rfm_summary['avg_order_value'] = rfm_summary['monetary_value'] / (rfm_summary['frequency'] + 1)
    
    # 2. Filter for returning customers
    # Gamma-Gamma only works on customers with > 0 repeat transactions (frequency > 0)
    # and monetary value > 0
    returning_customers = rfm_summary[
        (rfm_summary['frequency'] > 0) & 
        (rfm_summary['avg_order_value'] > 0)
    ]
    
    # 3. Fit the Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_customers['frequency'], returning_customers['avg_order_value'])
    
    # 4. Estimate CLV for EVERY customer
    # This combines the "Vitality" (BG/NBD) with the "Wallet" (Gamma-Gamma)
    rfm_summary['predicted_clv'] = ggf.customer_lifetime_value(
        bgf,  # Your existing BG/NBD model
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
        rfm_summary['avg_order_value'],
        time=12,  # Predicting value over the next 12 months
        discount_rate=0.01
    )
    
    return rfm_summary, ggf

def render_dashboard(rfm):
    """
    Render an executive dashboard with key performance indicators.
    
    Parameters:
    -----------
    rfm : DataFrame
        RFM summary with prob_alive and predicted_clv columns
    """
    st.markdown("---")
    st.header("📈 Executive Dashboard")
    
    # --- KPI CALCULATIONS ---
    total_customers = len(rfm)
    
    # "Active" means probability > 0.5 (or whatever threshold you prefer)
    active_customers = rfm[rfm['prob_alive'] > 0.5]
    num_active = len(active_customers)
    
    # Financials
    avg_clv = rfm['predicted_clv'].mean()
    total_clv = rfm['predicted_clv'].sum()
    
    # Churn Rate (Percentage of customers < 0.5 probability)
    churn_rate = (1 - (num_active / total_customers)) * 100

    # --- LAYOUT: 4 KEY METRICS ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        label="👥 Active Customers",
        value=f"{num_active}",
        delta=f"{num_active/total_customers:.1%} of Total"
    )

    kpi2.metric(
        label="💸 Avg CLV (12M)",
        value=f"${avg_clv:.2f}",
        help="Average predicted value per customer over next 12 months"
    )

    kpi3.metric(
        label="💰 Total Pipeline Value",
        value=f"${total_clv:,.0f}",
        help="Total predicted revenue from all current customers"
    )

    kpi4.metric(
        label="📉 Churn Risk",
        value=f"{churn_rate:.1f}%",
        delta="-High Risk" if churn_rate > 50 else "Stable",
        delta_color="inverse"  # Red if high, Green if low
    )

    # --- VISUAL HEALTH CHECK ---
    # A quick breakdown of Customer Status
    st.subheader("Customer Health Distribution")
    
    # Categorize customers for the chart
    rfm_chart = rfm.copy()
    rfm_chart['Status'] = rfm_chart['prob_alive'].apply(lambda x: 'Active 🟢' if x >= 0.5 else 'At Risk 🔴')
    status_counts = rfm_chart['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    # Simple Bar Chart
    fig = px.bar(
        status_counts, 
        x='Count', 
        y='Status', 
        orientation='h', 
        color='Status',
        color_discrete_map={'Active 🟢': '#2ecc71', 'At Risk 🔴': '#e74c3c'},
        title="Active vs. At-Risk Composition"
    )
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


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

# Initialize session state for data persistence across Streamlit reruns
if 'rfm_summary' not in st.session_state:
    st.session_state.rfm_summary = None
    st.session_state.bgf = None
    st.session_state.ggf = None
    st.session_state.features = None

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
    
    if st.button("🚀 Run Analysis"):
        with st.spinner('Calculating RFM Metrics & Running Predictive Analytics...'):
            # Train model on current data to demonstrate the 'Brain'
            predictor.train(df_clean)
            
            # Generate features (Recency, Frequency, Monetary)
            features = predictor.engineer_features(df_clean)
            
            # Run the prediction
            preds = predictor.predict(features[['recency', 'frequency']])
            features['future_value_score'] = preds
            
            # Run BG/NBD predictive engine
            results, bgf = run_prediction_engine(df_clean)
            
            # Run Gamma-Gamma CLV model
            results, ggf = calculate_clv_with_gamma_gamma(results, bgf)
            
            # SAVE TO SESSION STATE - This persists across slider interactions
            st.session_state.rfm_summary = results
            st.session_state.bgf = bgf
            st.session_state.ggf = ggf
            st.session_state.features = features
            
        st.success("✅ Analysis Complete! Use the simulator below to explore scenarios.")

    # --- CHECK IF DATA EXISTS BEFORE SHOWING VISUALIZATIONS ---
    
    if st.session_state.rfm_summary is not None:
        # Retrieve data from session state
        results = st.session_state.rfm_summary
        features = st.session_state.features
        bgf = st.session_state.bgf
        
        # Display Executive Dashboard
        render_dashboard(results)
        
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
        plot_probability_alive_matrix(bgf)
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

        # --- REVENUE SIMULATOR (Strategic Analysis) ---
        st.markdown("---")
        st.header("🔮 Strategic Revenue Simulator")

        # --- 1. STRATEGY CONTROLS ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("⚙️ Simulation Settings")
            
            # Toggle: Are we playing Defense (Saving) or Offense (Growing)?
            strategy = st.radio(
                "Choose Your Strategy:",
                ["🛡️ Defend: Rescue At-Risk Customers", "🚀 Grow: Upsell Loyal Customers"],
                help="Switch between saving customers who are leaving vs. getting more value from loyal ones."
            )
            
            # Dynamic Threshold: You define what "Alive" means
            threshold = st.slider(
                "Define Probability Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Customers below this probability are 'At-Risk'. Customers above are 'Loyal'."
            )

            # The Lever: The percentage impact of your marketing team
            if "Defend" in strategy:
                impact_rate = st.slider("Recovery Success Rate (%)", 0, 50, 5)
                impact_label = "Recovery Rate"
            else:
                impact_rate = st.slider("Upsell Success Rate (%)", 0, 50, 5)
                impact_label = "AOV Increase"

        # --- 2. CALCULATION ENGINE ---

        # Split the data based on the User's Threshold
        if "Defend" in strategy:
            # Target: Customers BELOW the threshold
            target_segment = results[results['prob_alive'] < threshold]
            
            # Math: (Total Potential CLV) * (Recovery Rate)
            total_potential_value = target_segment['predicted_clv'].sum()
            revenue_impact = total_potential_value * (impact_rate / 100)
            
            segment_name = "At-Risk Customers"
            action_verb = "Recaptured"
            
        else:
            # Target: Customers ABOVE the threshold
            target_segment = results[results['prob_alive'] >= threshold]
            
            # Math: (Total Potential CLV) * (Upsell Rate)
            # Note: Simplification - assuming we increase their value by X%
            total_potential_value = target_segment['predicted_clv'].sum()
            revenue_impact = total_potential_value * (impact_rate / 100)
            
            segment_name = "Loyal Customers"
            action_verb = "Generated"

        # Counts for Context
        count = len(target_segment)
        avg_val = total_potential_value / count if count > 0 else 0

        # --- 3. RESULTS DASHBOARD ---
        with col2:
            st.subheader("📊 Projected Outcomes")
            
            # The Big Number
            st.metric(
                label=f"💰 New Revenue {action_verb} (12 Months)",
                value=f"${revenue_impact:,.2f}",
                delta=f"{impact_rate}% {impact_label}"
            )
            
            # Contextual Data
            st.info(
                f"""
                **Segment Analysis:**
                * **Targeting:** {count} {segment_name}
                * **Total Segment Value:** ${total_potential_value:,.2f}
                * **Avg Value per Customer:** ${avg_val:,.2f}
                """
            )

            # Visualizing the scale
            if count > 0:
                st.progress(min(impact_rate / 100, 1.0))
            else:
                st.warning("No customers found in this segment with current settings.")

            # --- 4. ACTIONABLE INTELLIGENCE (EXPORT) ---
            st.markdown("### 📤 Take Action")

            # Create a clean dataframe for the marketing team
            if count > 0:
                export_data = target_segment[['monetary_value', 'prob_alive', 'predicted_clv']].copy()
                export_data.columns = ['Total Spend', 'Retention Probability', 'CLV (12mo)']
                export_data['Recommended Action'] = "Email Campaign" if "Defend" in strategy else "VIP Offer"

                # Convert to CSV
                csv = export_data.to_csv().encode('utf-8')

                # Dynamic File Name (e.g., "at_risk_customers.csv" or "loyal_customers.csv")
                file_name_prefix = "at_risk" if "Defend" in strategy else "loyal_vip"
                file_label = "🔴 Download At-Risk List" if "Defend" in strategy else "🟢 Download VIP List"

                st.download_button(
                    label=file_label,
                    data=csv,
                    file_name=f"{file_name_prefix}_customers.csv",
                    mime="text/csv",
                    help="Download this list to import into Mailchimp, Klaviyo, or Excel."
                )
            else:
                st.caption("No customers in this segment to export.")

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
        st.info("📊 Please click '🚀 Run Analysis' above to generate customer insights and unlock the simulator.")

else:
    st.info("👋 Welcome! Please upload a CSV with 'customer_id', 'revenue', and 'date' columns to begin.")