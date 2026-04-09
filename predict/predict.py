"""
Car Sales Volume & Revenue Predictor
====================================
Use this script to load the saved Random Forest model and forecast
Volume (Sales Count) and Total Revenue for a specific Month and Year,
broken down by Company and Dealer Region.
"""

import os
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Path to the exported model inside the output directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "rf_sales_model.pt")

def predict_sales_volume(year, month, company, dealer_region):
    """Forecasts Sales Count and Total Revenue for a given period & category."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 'py car_sales_forecasting.py' first to generate the rf_sales_model.pt file.")
        return None

    # 1. Load the model and encoders
    data = joblib.load(MODEL_PATH)
    rf_model = data["model"]
    encoders = data["encoders"]
    feature_cols = data["feature_cols"]

    # 2. Extract input data
    input_data = {
        "Year": year,
        "Month": month,
        "Company": company,
        "Dealer_Region": dealer_region
    }
    
    # 3. Encode categorical inputs safely
    encoded_input = {}
    for col in feature_cols:
        if col in ["Year", "Month"]:
            encoded_input[col] = input_data[col]
        elif col.endswith("_enc"):
            raw_col_name = col.replace("_enc", "")
            raw_value = str(input_data[raw_col_name])
            encoder = encoders[raw_col_name]
            
            if raw_value in encoder.classes_:
                encoded_input[col] = encoder.transform([raw_value])[0]
            else:
                print(f"Warning: '{raw_value}' is an unknown {raw_col_name}. Defaulting to '{encoder.classes_[0]}'.")
                encoded_input[col] = encoder.transform([encoder.classes_[0]])[0]
                
    # 4. Build the DataFrame
    input_df = pd.DataFrame([encoded_input], columns=feature_cols)
    
    # 5. Make Multi-Output Prediction: [Sales_Count, Total_Revenue]
    prediction = rf_model.predict(input_df)[0]
    sales_count = int(round(prediction[0]))
    total_rev = prediction[1]
    
    # 6. Print result
    print("\n" + "=" * 55)
    print(f"  [SALES FORECAST] {company} in {dealer_region} ({month:02d}/{year})")
    print("=" * 55)
    print(f"  > Forecasted Monthly Revenue:  ${total_rev:,.2f}")
    print(f"  > Forecasted Cars Sold:        {sales_count:,} units")
    print("=" * 55 + "\n")
    
    return sales_count, total_rev


if __name__ == "__main__":
    # =========================================================================
    #  MODIFY THESE VALUES TO FORECAST YOUR OWN SALES METRICS
    # =========================================================================
    
    # Example 1: Forecasting Toyota sales in Austin for July 2024
    predict_sales_volume(
        year=2024,
        month=7,
        company="Toyota",
        dealer_region="Austin"
    )

    # Example 2: Forecasting BMW sales in Scottsdale for December 2024
    predict_sales_volume(
        year=2024,
        month=12,
        company="BMW",
        dealer_region="Scottsdale"
    )
