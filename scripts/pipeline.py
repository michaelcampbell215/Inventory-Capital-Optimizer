import os
import sqlite3
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "erp_retail_data.db")
CSV_SOURCE = os.path.join(SCRIPT_DIR, "..", "data", "raw", "retail_store_inventory.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
CONFIG = {
    "LEAD_TIMES": {"Furniture": 14, "Electronics": 10, "Toys": 7, "Clothing": 5, "Groceries": 2},
    "SHELF_LIFE": {"Groceries": 7, "Clothing": 45, "Electronics": 90, "Furniture": 180, "Toys": 60},
    "TARGET_MARKDOWN": 0.25,
    "DEFAULT_LEAD_TIME": 3,
    "DEFAULT_SHELF_LIFE": 30,
    "TURNOVER_SCALE": {
        "Furniture": 0.02,
        "Electronics": 0.04,
        "Clothing": 0.08,
        "Toys": 0.06,
        "Groceries": 0.15,
    },
    "DEFAULT_TURNOVER_SCALE": 0.10,
}

# Extract & Transform
def create_db(db_path, csv_path):
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0: return
    print("Initializing Relational ERP Database...")
    df = pd.read_csv(csv_path)
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
    conn = sqlite3.connect(db_path)
    df[["Product_ID", "Category", "Price"]].drop_duplicates("Product_ID").to_sql("Products", conn, if_exists="replace", index=False)
    df[["Store_ID", "Region"]].drop_duplicates("Store_ID").to_sql("Stores", conn, if_exists="replace", index=False)
    
    # 3. Create Dim_Date Table
    df["Date"] = pd.to_datetime(df["Date"])
    dates = pd.DataFrame({"Date": df["Date"].unique()})
    dates["Year"] = dates["Date"].dt.year
    dates["Month"] = dates["Date"].dt.month
    dates["Month_Name"] = dates["Date"].dt.month_name()
    dates["Quarter"] = dates["Date"].dt.quarter
    dates["Day_of_Week"] = dates["Date"].dt.day_name()
    dates["Is_Weekend"] = dates["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Merge Holiday Promotion 
    if "Holiday_Promotion" in df.columns:
        holiday_map = df[["Date", "Holiday_Promotion"]].drop_duplicates("Date")
        dates = dates.merge(holiday_map, on="Date", how="left")
    
    dates.to_sql("Dim_Date", conn, if_exists="replace", index=False)
    
    df.drop(columns=["Category", "Region", "Price", "Holiday_Promotion"], errors="ignore").to_sql("Transactions", conn, if_exists="replace", index=False)
    conn.close()

def extract(db_path):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT t.*, p.Category, p.Price, s.Region, d.Holiday_Promotion 
    FROM Transactions t 
    JOIN Products p ON t.Product_ID = p.Product_ID 
    JOIN Stores s ON t.Store_ID = s.Store_ID
    JOIN Dim_Date d ON t.Date = d.Date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def transform(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Month"] = df["Date"].dt.month

    # Feature Engineering: One-Hot Encoding
    cat_cols = ["Store_ID", "Category", "Region", "Weather_Condition", "Seasonality"]

    # Create dummies
    dummies = pd.get_dummies(df[cat_cols], drop_first=False)

    df = pd.concat([df, dummies], axis=1)

    # Numeric Cleaning
    numeric_cols = ["Inventory_Level", "Price", "Discount", "Units_Sold", "Holiday_Promotion", "Competitor_Pricing"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# Metric Definition 
def calculate_spec_grouped(y_true, y_pred, product_ids):
    eval_df = pd.DataFrame({'PID': product_ids, 'Actual': y_true, 'Pred': y_pred})
    sku_costs = []

    for _, group in eval_df.groupby('PID'):
        act, pre = group['Actual'].values, group['Pred'].values
        n = len(act)
        if n == 0: continue
        cum_act, cum_pre = np.cumsum(act), np.cumsum(pre)
        cost = 0.0
        for t in range(n):
            unfulfilled = np.maximum(0, cum_act[:t+1] - cum_pre[t])
            excess = np.maximum(0, cum_pre[:t+1] - cum_act[t])
            penalties = np.maximum(unfulfilled * 0.75, excess * 0.25)
            cost += np.sum(penalties * (t - np.arange(t+1) + 1))
        sku_costs.append(cost / n)
    return np.mean(sku_costs)

# Model Comparison
def select_model_champion(df):
    train_df, _ = train_test_split(df, test_size=0.2, shuffle=False)
    X_train = train_df.drop(columns=["Units_Sold", "Date"], errors="ignore")
    y_train = train_df["Units_Sold"]
    features = X_train.select_dtypes(include=[np.number]).columns.tolist()
   
    features = [f for f in features if f != 'Product_ID']
        
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    if HAS_LGBM:
        candidates["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)

    print("--- Model Comparison: SPEC Cost Comparison ---")
    scores = {}
    for name, model in candidates.items():
        model.fit(X_train[features], y_train)
        preds = model.predict(X_train[features])
        score = calculate_spec_grouped(y_train, preds, X_train['Product_ID'])
        scores[name] = score
        print(f" -> {name} Avg SPEC Cost: ${score:.2f}")

    winner_name = min(scores, key=scores.get)
    print(f"\n Best Model: {winner_name}")

    champion_model = candidates[winner_name]
    champion_model.fit(X_train[features], y_train)
    return champion_model, features, winner_name

# Prescriptive Prep
def generate_tableau_output(df, model, features, config):
    output = df.copy()
    output['Predicted_Demand'] = model.predict(df[features])
    
    # Category-Level Volatility (Dynamic Safety Stock Foundation)
    output['Error_Sq'] = (output['Units_Sold'] - output['Predicted_Demand'])**2
    output['Category_RMSE'] = output.groupby('Category')['Error_Sq'].transform(lambda x: np.sqrt(x.mean()))
    
    # Lead Time mapping
    output['Lead_Time'] = output['Category'].map(config['LEAD_TIMES']).fillna(config['DEFAULT_LEAD_TIME'])

    # Tiered Overstock Logic (Perishability)
    output['Max_Shelf_Life'] = output['Category'].map(config['SHELF_LIFE']).fillna(config['DEFAULT_SHELF_LIFE'])
    output['Days_of_Supply'] = output['Inventory_Level'] / output['Predicted_Demand'].clip(lower=1)
    output['Is_Overstocked'] = output['Days_of_Supply'] > output['Max_Shelf_Life']

    scale_map = config.get('TURNOVER_SCALE', {})
    default_scale = config.get('DEFAULT_TURNOVER_SCALE', 0.10)
    output['Turnover_Scale'] = output['Category'].map(scale_map).fillna(default_scale)
    raw_turns = np.where(
        output['Inventory_Level'] > 0,
        (output['Predicted_Demand'] * 365) / output['Inventory_Level'],
        0
    )
    output['Portfolio_Turnover'] = raw_turns * output['Turnover_Scale']
    output.drop(columns=['Turnover_Scale'], inplace=True)  # helper col, not needed downstream

    # ABC Classification (Pareto Ranking)
    df['Revenue'] = df['Units_Sold'] * df['Price']
    prod_rev = df.groupby('Product_ID')['Revenue'].sum().sort_values(ascending=False)
    cum_rev_pct = prod_rev.cumsum() / prod_rev.sum()
    abc_map = cum_rev_pct.apply(lambda x: 'A' if x <= 0.20 else ('B' if x <= 0.50 else 'C'))
    output['ABC_Class'] = output['Product_ID'].map(abc_map)

    # Smart Markdown Logic
    existing_discount = output["Discount"] / 100 if "Discount" in output.columns else 0
    target_markdown = config.get("TARGET_MARKDOWN", 0.25)
    
    final_discount_pct = np.maximum(existing_discount, target_markdown)
    
    output["Current_Discount_Pct"] = existing_discount * 100
    
    output["Suggested_Price"] = np.where(
        output["Is_Overstocked"],
        output["Price"] * (1 - final_discount_pct), 
        output["Price"] * (1 - existing_discount)    
    )

    return output

def save_star_schema(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. inventory_fact.csv
    fact_cols = [
        "Date", "Store_ID", "Product_ID", "Inventory_Level",
        "Units_Sold", "Predicted_Demand", "Portfolio_Turnover",
        "Days_of_Supply", "Is_Overstocked", "Current_Discount_Pct", "Competitor_Pricing"
    ]
    fact_df = df[fact_cols].copy()
    fact_df.to_csv(f"{output_dir}/inventory_fact.csv", index=False)
    
    # 2. product_dim.csv
    prod_cols = [
        "Product_ID", "Category", "Price", "ABC_Class", 
        "Category_RMSE", "Lead_Time", "Max_Shelf_Life"
    ]
    prod_df = df[prod_cols].drop_duplicates("Product_ID")
    prod_df.to_csv(f"{output_dir}/product_dim.csv", index=False)
    
    # 3. store_dim.csv
    store_cols = ["Store_ID", "Region"]
    store_df = df[store_cols].drop_duplicates("Store_ID")
    store_df.to_csv(f"{output_dir}/store_dim.csv", index=False)
    
    # 4. date_dim.csv
    df["Date"] = pd.to_datetime(df["Date"])
    date_df = pd.DataFrame({"Date": df["Date"].unique()})
    date_df["Year"] = date_df["Date"].dt.year
    date_df["Month"] = date_df["Date"].dt.month
    date_df["Month_Name"] = date_df["Date"].dt.month_name()
    date_df["Quarter"] = date_df["Date"].dt.quarter
    date_df["Day_of_Week"] = date_df["Date"].dt.day_name()
    date_df["Is_Weekend"] = date_df["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Add Holiday Promotion (using first occurrence per date)
    if "Holiday_Promotion" in df.columns:
        holiday_map = df.groupby("Date")["Holiday_Promotion"].first().reset_index()
        date_df = date_df.merge(holiday_map, on="Date", how="left")
        
    date_df.to_csv(f"{output_dir}/date_dim.csv", index=False)
    
    print(f"--- Star Schema Exported to {output_dir} ---")
    print(f" -> inventory_fact.csv: {len(fact_df):,} rows")
    print(f" -> product_dim.csv   : {len(prod_df):,} rows")
    print(f" -> store_dim.csv     : {len(store_df):,} rows")
    print(f" -> date_dim.csv      : {len(date_df):,} rows")

if __name__ == "__main__":
    create_db(DB_PATH, CSV_SOURCE)
    raw_df = extract(DB_PATH)
    clean_df = transform(raw_df)
    
    # 1. Selection
    
    champion, feats, winner_name = select_model_champion(clean_df)
    
    # 2. Preparation
    tableau_df = generate_tableau_output(clean_df, champion, feats, CONFIG)
    
    # 3. Validation Audit
    print("\n--- Final Integrity Check ---")
    print(f"ABC Distribution:\n{tableau_df['ABC_Class'].value_counts(normalize=True).map(lambda n: f'{n:.1%}')}")
    print(f"Unique Category RMSEs calculated: {tableau_df['Category_RMSE'].nunique()}")
    
    # 4. Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_star_schema(tableau_df, OUTPUT_DIR)
    print(f"\n SUCCESS: Final report generated via {winner_name} for Tableau.")