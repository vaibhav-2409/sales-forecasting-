"""
Car Sales Forecasting using Machine Learning
=============================================
This project uses Regression Models (Linear Regression & Random Forest)
to forecast car sale prices and analyze related sales parameters.

Dataset: Car Sales.xlsx - car_data.csv (~23,907 records)
"""

import os
import warnings
import numpy as np
import pandas as pd
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

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Car Sales.xlsx - car_data.csv",
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA LOADING & CLEANING                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, clean column names, drop irrelevant columns."""
    df = pd.read_csv(path, encoding="utf-8")

    # Standardise column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # Rename for convenience
    rename_map = {"Price ($)": "Price", "Dealer_No ": "Dealer_No", "Dealer_No": "Dealer_No"}
    df.rename(columns=rename_map, inplace=True)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # Clean engine column (remove non-ASCII artefacts)
    if "Engine" in df.columns:
        df["Engine"] = df["Engine"].str.replace("\xa0", " ", regex=False)
        df["Engine"] = df["Engine"].str.replace("Â", "", regex=False)

    # Drop columns not useful for modelling
    drop_cols = ["Car_id", "Customer Name", "Phone", "Dealer_No"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Drop rows with missing target
    df.dropna(subset=["Price"], inplace=True)

    print(f"[OK] Loaded {len(df):,} records  |  Columns: {list(df.columns)}")
    print(f"   Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. EXPLORATORY DATA ANALYSIS (EDA)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def run_eda(df: pd.DataFrame):
    """Generate and save EDA visualisations."""
    print("\n[EDA] Running Exploratory Data Analysis ...")

    # 2-a  Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price"], bins=50, kde=True, ax=axes[0], color="#2196F3")
    axes[0].set_title("Distribution of Car Prices", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    sns.boxplot(x=df["Price"], ax=axes[1], color="#4CAF50")
    axes[1].set_title("Price Box-Plot", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_price_distribution.png"), dpi=150)
    plt.close(fig)

    # 2-b  Annual Income distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Annual Income"], bins=50, kde=True, ax=ax, color="#FF9800")
    ax.set_title("Distribution of Customer Annual Income", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annual Income ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_income_distribution.png"), dpi=150)
    plt.close(fig)

    # 2-c  Monthly sales trend (count & revenue)
    monthly = df.set_index("Date").resample("ME").agg(
        Sales_Count=("Price", "count"),
        Total_Revenue=("Price", "sum"),
    )
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(monthly.index, monthly["Sales_Count"], width=20, color="#42A5F5", alpha=0.7, label="Sales Count")
    ax1.set_ylabel("Number of Sales", color="#42A5F5", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#42A5F5")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["Total_Revenue"], color="#EF5350", linewidth=2, marker="o", markersize=4, label="Revenue")
    ax2.set_ylabel("Total Revenue ($)", color="#EF5350", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#EF5350")
    ax1.set_title("Monthly Sales Count & Revenue Trend", fontsize=14, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_monthly_sales_trend.png"), dpi=150)
    plt.close(fig)

    # 2-d  Top 15 companies by sales count
    fig, ax = plt.subplots(figsize=(12, 6))
    top_companies = df["Company"].value_counts().head(15)
    sns.barplot(x=top_companies.values, y=top_companies.index, ax=ax, palette="Blues_r")
    ax.set_title("Top 15 Car Companies by Sales Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Sales")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_top_companies.png"), dpi=150)
    plt.close(fig)

    # 2-e  Average price by body style
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_price_body = df.groupby("Body Style")["Price"].mean().sort_values(ascending=False)
    sns.barplot(x=avg_price_body.values, y=avg_price_body.index, ax=ax, palette="Oranges_r")
    ax.set_title("Average Price by Body Style", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Price ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_avg_price_body_style.png"), dpi=150)
    plt.close(fig)

    # 2-f  Sales by Gender
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gender_counts = df["Gender"].value_counts()
    axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
                colors=["#42A5F5", "#EF5350"], startangle=140, textprops={"fontsize": 11})
    axes[0].set_title("Sales by Gender", fontsize=13, fontweight="bold")
    sns.boxplot(x="Gender", y="Price", data=df, ax=axes[1], palette=["#42A5F5", "#EF5350"])
    axes[1].set_title("Price Distribution by Gender", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_gender_analysis.png"), dpi=150)
    plt.close(fig)

    # 2-g  Transmission & Engine type
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(x="Transmission", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Sales by Transmission Type", fontsize=13, fontweight="bold")
    sns.countplot(x="Engine", data=df, ax=axes[1], palette="Set3")
    axes[1].set_title("Sales by Engine Type", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_transmission_engine.png"), dpi=150)
    plt.close(fig)

    # 2-h  Sales by Dealer Region
    fig, ax = plt.subplots(figsize=(12, 5))
    region_counts = df["Dealer_Region"].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax, palette="coolwarm")
    ax.set_title("Sales Count by Dealer Region", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Sales")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_dealer_region.png"), dpi=150)
    plt.close(fig)

    # 2-i  Correlation heatmap (numeric columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "09_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # 2-j  Color popularity
    fig, ax = plt.subplots(figsize=(8, 5))
    color_counts = df["Color"].value_counts()
    ax.pie(color_counts.values, labels=color_counts.index, autopct="%1.1f%%",
           colors=["#ECEFF1", "#263238", "#EF5350"], startangle=140,
           textprops={"fontsize": 11})
    ax.set_title("Sales by Car Color", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "10_color_popularity.png"), dpi=150)
    plt.close(fig)

    print("   Saved 10 EDA charts -> output/")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. FEATURE ENGINEERING                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def engineer_features(df: pd.DataFrame):
    """Create model-ready features from the raw dataframe."""
    df = df.copy()

    # Date-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter

    # Label-encode categorical columns
    cat_cols = ["Gender", "Dealer_Name", "Company", "Model", "Engine",
                "Transmission", "Color", "Body Style", "Dealer_Region"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Feature matrix
    feature_cols = [
        "Annual Income", "Year", "Month", "DayOfWeek", "Quarter",
        "Gender_enc", "Company_enc", "Model_enc", "Engine_enc",
        "Transmission_enc", "Color_enc", "Body Style_enc", "Dealer_Region_enc",
    ]
    # Keep only cols that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["Price"]

    print(f"\n[FEATURES] Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y, feature_cols, encoders


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4 & 5. MODEL TRAINING & EVALUATION                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def evaluate_model(name, y_true, y_pred):
    """Compute and print regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"   {name:30s}  MAE={mae:>10,.2f}   RMSE={rmse:>10,.2f}   R²={r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_models(X, y):
    """Train Linear Regression & Random Forest, return metrics & models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[TRAIN] Training models  (Train={len(X_train):,}  |  Test={len(X_test):,})")

    results = []

    # ── Model 1: Linear Regression (MANDATORY) ──────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    res_lr = evaluate_model("Linear Regression", y_test, y_pred_lr)
    results.append(res_lr)

    # ── Model 2: Random Forest Regressor (ADDITIONAL) ────────────────────────
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    res_rf = evaluate_model("Random Forest Regressor", y_test, y_pred_rf)
    results.append(res_rf)

    # ── Actual vs Predicted plots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name, y_pred, color in [
        (axes[0], "Linear Regression", y_pred_lr, "#1E88E5"),
        (axes[1], "Random Forest", y_pred_rf, "#43A047"),
    ]:
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=color)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Price ($)", fontsize=11)
        ax.set_ylabel("Predicted Price ($)", fontsize=11)
        ax.set_title(f"{name}\nActual vs Predicted", fontsize=13, fontweight="bold")
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "11_actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # ── Residual plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name, y_pred, color in [
        (axes[0], "Linear Regression", y_pred_lr, "#1E88E5"),
        (axes[1], "Random Forest", y_pred_rf, "#43A047"),
    ]:
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=color)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Predicted Price ($)")
        ax.set_ylabel("Residual ($)")
        ax.set_title(f"{name} — Residuals", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "12_residual_plots.png"), dpi=150)
    plt.close(fig)

    # ── Model comparison bar chart ───────────────────────────────────────────
    results_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, color in zip(axes, ["MAE", "RMSE", "R2"], ["#1E88E5", "#EF5350", "#43A047"]):
        bars = ax.bar(results_df["Model"], results_df[metric], color=color, width=0.5)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:,.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "13_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return lr, rf, X_test, y_test, results_df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. FEATURE IMPORTANCE (Random Forest)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def plot_feature_importance(rf_model, feature_cols):
    """Bar chart of Random Forest feature importances."""
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=importances[idx], y=np.array(feature_cols)[idx], ax=ax, palette="viridis")
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "14_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("\n[IMPORTANCE] Top-5 features driving price:")
    for i in range(min(5, len(feature_cols))):
        print(f"      {i+1}. {feature_cols[idx[i]]:25s}  ({importances[idx[i]]:.4f})")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. SALES FORECASTING (Monthly Trend)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def forecast_monthly_sales(df: pd.DataFrame, months_ahead: int = 6):
    """Fit a linear trend on monthly aggregated sales and forecast ahead."""
    print(f"\n[FORECAST] Forecasting monthly sales for the next {months_ahead} months ...")

    monthly = df.set_index("Date").resample("ME").agg(
        Sales_Count=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Total_Revenue=("Price", "sum"),
    )
    monthly["Month_Num"] = np.arange(len(monthly))

    # ── Linear trend for Sales Count ─────────────────────────────────────────
    X_month = monthly[["Month_Num"]]
    y_count = monthly["Sales_Count"]
    y_revenue = monthly["Total_Revenue"]

    lr_count = LinearRegression().fit(X_month, y_count)
    lr_revenue = LinearRegression().fit(X_month, y_revenue)

    future_nums = np.arange(len(monthly), len(monthly) + months_ahead).reshape(-1, 1)
    future_dates = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=months_ahead, freq="ME")

    pred_count = lr_count.predict(future_nums)
    pred_revenue = lr_revenue.predict(future_nums)

    # ── Plot forecast ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Sales count
    axes[0].plot(monthly.index, y_count, "o-", color="#1E88E5", label="Historical Sales Count")
    axes[0].plot(future_dates, pred_count, "s--", color="#EF5350", linewidth=2, label="Forecasted Sales Count")
    axes[0].axvline(monthly.index[-1], color="gray", linestyle=":", alpha=0.7)
    axes[0].set_title("Monthly Sales Count — Forecast", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Sales Count")
    axes[0].legend()

    # Revenue
    axes[1].plot(monthly.index, y_revenue, "o-", color="#43A047", label="Historical Revenue")
    axes[1].plot(future_dates, pred_revenue, "s--", color="#FF9800", linewidth=2, label="Forecasted Revenue")
    axes[1].axvline(monthly.index[-1], color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title("Monthly Revenue — Forecast", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "15_sales_forecast.png"), dpi=150)
    plt.close(fig)

    # Print forecast table
    forecast_df = pd.DataFrame({
        "Month": future_dates.strftime("%Y-%m"),
        "Predicted_Sales_Count": np.round(pred_count).astype(int),
        "Predicted_Revenue ($)": np.round(pred_revenue, 2),
    })
    print("\n   Forecasted Monthly Sales:")
    print(forecast_df.to_string(index=False))

    return forecast_df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    print("=" * 70)
    print("  CAR SALES FORECASTING — Machine Learning Project")
    print("=" * 70)

    # 1. Load & clean
    df = load_and_clean(DATA_FILE)

    # 2. EDA
    run_eda(df)

    # 3. Feature engineering
    X, y, feature_cols, encoders = engineer_features(df)

    # 4-5. Train & evaluate models
    lr_model, rf_model, X_test, y_test, results_df = train_models(X, y)

    # 6. Feature importance
    plot_feature_importance(rf_model, feature_cols)

    # 7. Sales forecasting
    forecast_df = forecast_monthly_sales(df, months_ahead=6)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\n[OK] All outputs saved to:", OUTPUT_DIR)
    print("=" * 70)

    # Save results to CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "sales_forecast.csv"), index=False)


if __name__ == "__main__":
    main()
