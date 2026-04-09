"""
Car Sales Volume & Revenue Forecasting using Machine Learning
=============================================================
This project uses Regression Models (Linear Regression & Random Forest)
to forecast the number of car sales (Sales Volume) and Total Revenue 
based on Year, Month, Company, and Dealer Region.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Car Sales.xlsx - car_data.csv",
)

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    df.rename(columns={"Price ($)": "Price", "Dealer_No ": "Dealer_No", "Dealer_No": "Dealer_No"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
    df.dropna(subset=["Price"], inplace=True)
    print(f"[OK] Loaded {len(df):,} records.")
    return df

def run_eda(df: pd.DataFrame):
    print("\n[EDA] Generating Sales Trend Visualizations ...")
    
    # Simple Monthly Sales Trend
    monthly = df.set_index("Date").resample("ME").agg(
        Sales_Count=("Price", "count"),
        Total_Revenue=("Price", "sum"),
    )
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(monthly.index, monthly["Sales_Count"], width=20, color="#42A5F5", alpha=0.7, label="Sales Count")
    ax1.set_ylabel("Number of Sales", color="#42A5F5", fontsize=11)
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["Total_Revenue"], color="#EF5350", linewidth=2, marker="o", label="Revenue")
    ax2.set_ylabel("Total Revenue ($)", color="#EF5350", fontsize=11)
    ax1.set_title("Historical Monthly Sales Count & Revenue Trend", fontsize=14, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_historical_monthly_sales.png"), dpi=150)
    plt.close(fig)

def prepare_sales_forecasting_data(df: pd.DataFrame):
    """Aggregate data to forecast monthly sales volume and revenue by Company and Region."""
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # 1. Aggregate Sales by Year, Month, Company, and Dealer_Region
    agg_df = df.groupby(["Year", "Month", "Company", "Dealer_Region"]).agg(
        Sales_Count=("Price", "count"),
        Total_Revenue=("Price", "sum")
    ).reset_index()
    
    # 2. Encode categorical parameters using LabelEncoder
    cat_cols = ["Company", "Dealer_Region"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        agg_df[col + "_enc"] = le.fit_transform(agg_df[col].astype(str))
        encoders[col] = le
        
    # 3. Define Features (X) and Targets (y)
    feature_cols = ["Year", "Month", "Company_enc", "Dealer_Region_enc"]
    target_cols = ["Sales_Count", "Total_Revenue"]
    
    X = agg_df[feature_cols]
    y = agg_df[target_cols]
    
    print(f"[FEATURES] Aggregated Dataset size: {len(X):,} Monthly Groupings")
    return X, y, feature_cols, encoders

def evaluate_multi_model(name, y_true, y_pred):
    """Evaluate multi-output regression (Sales Count & Revenue)."""
    # y_true and y_pred have 2 columns: [Sales_Count, Total_Revenue]
    mae_count = mean_absolute_error(y_true.iloc[:, 0], y_pred[:, 0])
    r2_count = r2_score(y_true.iloc[:, 0], y_pred[:, 0])
    
    mae_rev = mean_absolute_error(y_true.iloc[:, 1], y_pred[:, 1])
    r2_rev = r2_score(y_true.iloc[:, 1], y_pred[:, 1])
    
    print(f"   {name:25s} | COUNT: R2={r2_count:.4f}, MAE={mae_count:.2f} | REVENUE: R2={r2_rev:.4f}, MAE=${mae_rev:,.2f}")
    return {
        "Model": name, 
        "Count_R2": r2_count, "Count_MAE": mae_count,
        "Rev_R2": r2_rev, "Rev_MAE": mae_rev
    }

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"\n[TRAIN] Training Models to forecast Sales Volume & Revenue ...")
    
    results = []
    
    # Model 1: Linear Regression (Multi-output is natively supported)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    res_lr = evaluate_multi_model("Linear Regression", y_test, lr.predict(X_test))
    results.append(res_lr)
    
    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    res_rf = evaluate_multi_model("Random Forest Regressor", y_test, rf.predict(X_test))
    results.append(res_rf)
    
    results_df = pd.DataFrame(results)
    return rf, results_df

def main():
    print("=" * 70)
    print("  CAR SALES VOLUME FORECASTING MODEL TRAINING")
    print("=" * 70)

    # Load data
    df = load_and_clean(DATA_FILE)
    
    # EDA
    run_eda(df)
    
    # Prepare aggregated features
    X, y, feature_cols, encoders = prepare_sales_forecasting_data(df)
    
    # Train ML models
    rf_model, results_df = train_and_evaluate(X, y)
    
    # Save the Random Forest Model for the predict.py script
    model_path = os.path.join(OUTPUT_DIR, "rf_sales_model.pt")
    joblib.dump({
        "model": rf_model,
        "encoders": encoders,
        "feature_cols": feature_cols
    }, model_path)
    
    print("\n" + "=" * 70)
    print(f"[OK] Training complete. Model & Encoders saved to: {model_path}")
    print("You can now run 'py predict/predict.py' to forecast future sales!")
    print("=" * 70)

if __name__ == "__main__":
    main()
