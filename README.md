🚀 Revenue Architect: Predictive Customer Intelligence
Revenue Architect is an end-to-end data engineering and machine learning pipeline designed to transform raw transactional logs into actionable business strategy. It features an automated "Data Janitor" and a probabilistic BG/NBD model to predict customer churn and lifetime value.

🌐 Live Demo on Streamlit

🛠️ The Problem & The Solution
In retail, data is often chaotic—missing customer IDs, inconsistent date formats, and "silent" churn (customers who stop buying without canceling a subscription).

This project solves these challenges by:

Automated Pre-processing: A robust cleaning pipeline that handles returns, date normalization, and data type inconsistencies.

Probabilistic Modeling: Moving beyond static RFM to use the BG/NBD (Beta-Geometric/Negative Binomial Distribution) model for predicting the probability of customer "vitality."

Actionable Intelligence: Generating "At-Risk" reports that can be downloaded and used immediately for marketing campaigns.

🚀 Key Features
Predictive Engine: Calculates the probability that a customer is still active ("Alive") versus churned.

Segment Architect: Categorizes the customer base into Repeat Buyers vs. One-Time Buyers.

Data Janitor: Automatically filters out cancellations (return invoices) and cleans null identifiers.

Interactive Visuals: Dynamic heatmaps showing the relationship between Recency and Frequency.

Campaign Exporter: One-click CSV download for targeting high-value, at-risk customers.

💻 Tech Stack
Language: Python 3.9+

Framework: Streamlit

Modeling: Lifetimes (BG/NBD Model), Scikit-Learn

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Plotly

📊 Dataset
This project uses the UCI Online Retail II dataset, containing transactions for a UK-based non-store online retail between 2009 and 2011.

Dataset Link

⚙️ Installation & Setup
To run this project locally:

Clone the repository:

Bash
git clone https://github.com/rameezahmad9726/revenue_architect.git
cd revenue_architect
Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
streamlit run app.py
🤝 Contact
Rameez Ahmad GitHub | LinkedIn
