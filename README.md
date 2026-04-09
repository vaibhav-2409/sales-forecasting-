# Car Sales Forecasting — Predictive Analysis Project

A machine learning project that forecasts **car sale prices** and analyzes sales trends using **Linear Regression** (primary) and **Random Forest Regressor** (additional) models.

## Dataset

| Property | Value |
|----------|-------|
| File | `Car Sales.xlsx - car_data.csv` |
| Records | ~23,907 |
| Features | Date, Gender, Annual Income, Company, Model, Engine, Transmission, Color, Price, Body Style, Dealer Region |

## Project Structure

```
Predictive analysis pjt/
├── Car Sales.xlsx - car_data.csv   # Raw dataset
├── car_sales_forecasting.py        # Main ML pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── output/                         # Generated charts & results
    ├── 01_price_distribution.png
    ├── 02_income_distribution.png
    ├── 03_monthly_sales_trend.png
    ├── 04_top_companies.png
    ├── 05_avg_price_body_style.png
    ├── 06_gender_analysis.png
    ├── 07_transmission_engine.png
    ├── 08_dealer_region.png
    ├── 09_correlation_heatmap.png
    ├── 10_color_popularity.png
    ├── 11_actual_vs_predicted.png
    ├── 12_residual_plots.png
    ├── 13_model_comparison.png
    ├── 14_feature_importance.png
    ├── 15_sales_forecast.png
    ├── model_results.csv
    └── sales_forecast.csv
```

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the project
python car_sales_forecasting.py
```

## Models Used

### 1. Linear Regression (Mandatory)
Standard OLS regression predicting car price from engineered features.

### 2. Random Forest Regressor (Additional)
Ensemble of 200 decision trees (max depth 15) for improved non-linear prediction.

## Key Features

- **Exploratory Data Analysis**: 10 visualization charts covering price distributions, sales trends, company rankings, demographic analysis
- **Feature Engineering**: Date decomposition, categorical encoding for 9 variables
- **Model Evaluation**: MAE, RMSE, R² comparison between models
- **Feature Importance**: Identifies top price-driving factors from the Random Forest model
- **Sales Forecasting**: 6-month ahead linear trend forecast on monthly sales count and revenue

## Technologies

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
