import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class RevenuePredictor:
    def __init__(self):
        # Bundling scaler and model for professional deployment
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def engineer_features(self, df):
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        # 1. Smart Column Mapping - The "Survivor" Logic
        # We look for common aliases for revenue and customer_id
        rev_options = [c for c in df.columns if any(x in c for x in ['rev', 'amount', 'sales', 'price'])]
        id_options = [c for c in df.columns if any(x in c for x in ['cust', 'id', 'user'])]
        date_options = [c for c in df.columns if 'date' in c]

        # Use the first match found, or fallback to standard names
        rev_col = rev_options[0] if rev_options else 'revenue'
        id_col = id_options[0] if id_options else 'customer_id'
        date_col = date_options[0] if date_options else 'date'

        # 2. Safety Check - If the column literally isn't there, we don't crash the app
        if rev_col not in df.columns or date_col not in df.columns:
            # We return an empty RFM table rather than crashing
            return pd.DataFrame(columns=['recency', 'frequency', 'monetary'])

        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['date'])

        latest_date = df['date'].max()
        
        rfm = df.groupby(id_col).agg({
            'date': lambda x: (latest_date - x.max()).days,
            id_col: 'count',
            rev_col: 'sum'
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        return rfm

    def train(self, df):
        features = self.engineer_features(df)
        X = features[['recency', 'frequency']]
        y = features['monetary']
        
        # Ensure data is sufficient
        if len(X) < 2:
            return None
            
        self.pipeline.fit(X, y)
        
        # Create directory if missing
        os.makedirs('data/processed', exist_ok=True)
        joblib.dump(self.pipeline, 'data/processed/revenue_model.pkl')

    def predict(self, input_data):
        model_path = 'data/processed/revenue_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model.predict(input_data)
        return [0] * len(input_data)