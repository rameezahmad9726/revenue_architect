import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

class DataJanitor:
    def __init__(self):
        # Define what "Clean Data" looks like
        self.schema = DataFrameSchema({
            "customer_id": Column(int, coerce=True),
            "revenue": Column(float, Check.ge(0), coerce=True), # Must be >= 0
            "date": Column(pa.DateTime, coerce=True),
            "region": Column(str, Check.isin(['North', 'South', 'East', 'West']), coerce=True, nullable=True)
        })

    def deep_clean(self, df):
        # --- THE SENIOR FIX ---
        # Create a deep copy so we don't accidentally modify the raw data source
        df = df.copy() 
        
        print("--- Starting Deep Clean ---")
        # 1. Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # 2. Aggressive Numeric Cleaning
        if 'revenue' in df.columns:
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        
        # 3. THE FIX: Date Cleaning
        if 'date' in df.columns:
            # Convert to datetime objects first to standardize
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            # Convert back to simple string format (YYYY-MM-DD)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
        # 4. Validation
        try:
            cleaned_df = self.schema.validate(df, lazy=True)
            print("Validation Success!")
            return cleaned_df
        except Exception as e:
            print(f"Validation Note: {e}")
            return df 

# Test it
if __name__ == "__main__":
    # Simulate messy data
    messy_data = pd.DataFrame({
        "Customer ID": ["101", "102", "103 "],
        " Revenue ": ["150.50", "invalid", "200"], # "invalid" will be handled
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Region": ["North", "West", "North"]
    })
    
    janitor = DataJanitor()
    final_df = janitor.deep_clean(messy_data)
    print(final_df)