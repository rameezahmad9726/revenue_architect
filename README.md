# revenue_architect
Enterprise-grade Customer Value Estimator. An automated data pipeline that transforms raw, messy transactional logs into actionable RFM insights using Random Forest &amp; Streamlit.

# 🚀 Revenue Architect: Intelligent Customer Value Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> **A robust, end-to-end Machine Learning pipeline that automates the "Dirty Data" problem and generates Customer Lifetime Value (CLV) scores.**

---

## 💼 Business Problem
Most revenue prediction models fail in the real world because transactional data is messy—inconsistent dates, missing IDs, and mixed types. Companies spend 80% of their time cleaning data and only 20% analyzing it.

**Revenue Architect** solves this by enforcing a strict "Data Contract." It features a resilient **Data Janitor** module that ingests chaotic CSVs, standardizes them, and feeds them into an **RFM-based Predictive Engine** to identify high-value customers instantly.

---

## 🏗️ System Architecture

The application is decoupled into three production-grade modules:

### 1. 🧹 The Data Janitor (`cleaner.py`)
* **Defensive Programming:** Automatically detects and strips metadata rows, fixes mixed-type columns, and creates safety copies (`df.copy()`) to prevent mutation bugs.
* **Fuzzy Column Matching:** Uses "Survivor-Logic" to find columns like `revenue`, `sales`, or `amount` even if the user names them incorrectly.
* **Date Normalization:** Casts arbitrary timestamp formats into standardized `YYYY-MM-DD` objects.

### 2. 🧠 The Predictive Core (`model.py`)
* **Automated Feature Engineering:** Converts raw log data into **Recency, Frequency, and Monetary (RFM)** features on the fly.
* **Scoring Logic:** Uses a **Random Forest Regressor** to rank customers by their probability of high future spend.
* **Scalability:** Tested on the **Online Retail II dataset (40,000+ rows)** with sub-second inference time.

### 3. 📊 The Interface (`app.py`)
* **Self-Service BI:** A Streamlit-based frontend that allows non-technical stakeholders to upload data and download reports.
* **Actionable Visualizations:** Uses color-graded data tables to highlight "Whale" customers (High Value) vs. "Churn Risk" customers.

---

## 📸 Screenshots

| **Stage 1: The Janitor** | **Stage 2: The Intelligence** |
|:---:|:---:|
| *Automatically cleans and fixes "invalid" data.* | *Identifies VIP customers using RFM scores.* |
| *(Add your screenshot of the Raw vs Clean table here)* | *(Add your screenshot of the Green Prediction Table here)* |

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data Processing:** Pandas, Pandera
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Persistence:** Joblib
* **Visualization:** Matplotlib / Streamlit Native

---

## 🚀 Quick Start (Local)

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/revenue-architect.git](https://github.com/YOUR_USERNAME/revenue-architect.git)
   cd revenue-architect
