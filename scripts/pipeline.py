import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import sqlite3
import os

# Create data directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# --------------------------------------------------------------------------------
# Step 1: Data Integration & Star Schema (Migration from DB)
# --------------------------------------------------------------------------------
def load_and_transform_data():
    conn = sqlite3.connect('data/raw/erp_retail_data.db')
    
    # Extract Fact & Dimensions (preventing aggregated measure inflation)
    inv_fact = pd.read_sql("SELECT * FROM inventory_fact", conn)
    product_dim = pd.read_sql("SELECT * FROM product_dim", conn)
    store_dim = pd.read_sql("SELECT * FROM store_dim", conn)
    
    # Basic ABC Classification (Parento Principle)
    # Revenue = Price * Quantity
    product_dim['Revenue_Potential'] = product_dim['Base_Price'] * 1000 # placeholder volume
    product_dim['ABC_Class'] = pd.qcut(product_dim['Revenue_Potential'], 3, labels=['C', 'B', 'A'])
    
    conn.close()
    
    # Save transformed artifacts for Tableau
    inv_fact.to_csv('data/processed/inventory_fact.csv', index=False)
    product_dim.to_csv('data/processed/product_dim.csv', index=False)
    store_dim.to_csv('data/processed/store_dim.csv', index=False)
    
    return inv_fact, product_dim

# --------------------------------------------------------------------------------
# Step 2: Custom SPEC Scoring Logic (Stock-keeping-oriented Prediction Error Costs)
# --------------------------------------------------------------------------------
def calculate_spec_grouped(y_true, y_pred, product_ids):
    """
    Calculates cumulative cost of errors per SKU to prevent cross-product pollution.
    Penalizes stockouts (0.75) vs overstocks (0.25).
    """
    eval_df = pd.DataFrame({'PID': product_ids, 'Actual': y_true, 'Pred': y_pred})
    sku_costs = []

    for _, group in eval_df.groupby('PID'):
        act, pre = group['Actual'].values, group['Pred'].values
        n = len(act)
        if n == 0: continue
        
        cum_act, cum_pre = np.cumsum(act), np.cumsum(pre)
        cost = 0.0
        
        for t in range(n):
            # Calculate unfulfilled demand and excess inventory
            unfulfilled = np.maximum(0, cum_act[:t+1] - cum_pre[t])
            excess = np.maximum(0, cum_pre[:t+1] - cum_act[t])
            
            # Weighted penalty: Stockout risk is 3x more costly than trapped capital
            penalties = np.maximum(unfulfilled * 0.75, excess * 0.25)
            cost += np.sum(penalties * (t - np.arange(t+1) + 1))
            
        sku_costs.append(cost / n)
        
    return np.mean(sku_costs)

# --------------------------------------------------------------------------------
# Step 3: Predictive Engine (XGBoost)
# --------------------------------------------------------------------------------
def train_inventory_model(df):
    # Simplified Feature Engineering
    df['Day'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    
    X = df[['Product_ID', 'Store_ID', 'Day', 'Month']]
    y = df['Demand']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    
    return model

if __name__ == "__main__":
    print("Initializing Inventory Capital Optimizer Pipeline...")
    # Load
    inv_fact, product_dim = load_and_transform_data()
    
    # Train (Simplified)
    model = train_inventory_model(inv_fact)
    
    # Evaluate with SPEC
    preds = model.predict(inv_fact[['Product_ID', 'Store_ID', 'Day', 'Month']])
    spec_score = calculate_spec_grouped(inv_fact['Demand'], preds, inv_fact['Product_ID'])
    
    print(f"Model Training Complete. SPEC Score: {spec_score:.4f}")
