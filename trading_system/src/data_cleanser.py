import pandas as pd
import os
from .data_crawler import TICKERS, DATA_DIR # Use . for relative import

CLEANED_DATA_DIR = DATA_DIR # Store cleaned data in the same directory for simplicity for now
# Could be a different sub-directory like data/cleaned/ if preferred later

def load_raw_data(metal_name, frequency):
    """
    Loads raw CSV data for a specific metal and frequency.

    Args:
        metal_name (str): The name of the metal (e.g., "GOLD").
        frequency (str): "daily" or "weekly".

    Returns:
        pandas.DataFrame: DataFrame with raw data, or None if file not found.
    """
    filename = f"{metal_name.lower()}_{frequency.lower()}_prices.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Error: Raw data file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"Successfully loaded raw data from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading raw data from {filepath}: {e}")
        return None

def select_and_rename_columns(df, metal_name):
    """
    Selects relevant columns, sets 'Date' as index, and renames columns.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        metal_name (str): Name of the metal (e.g., "GOLD").

    Returns:
        pandas.DataFrame: Processed DataFrame.
    """
    if 'Date' not in df.columns:
        print(f"Error: 'Date' column not found in DataFrame for {metal_name}.")
        # Attempt to find a date-like column if 'Date' is missing (e.g. from yfinance index)
        if df.index.name == 'Date' and isinstance(df.index, pd.DatetimeIndex):
             df = df.reset_index() # Move Date index to column
        else: # Try to find any datetime64 column and rename it
            date_col = None
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
            if date_col:
                print(f"Warning: 'Date' column not found, using '{date_col}' as Date column for {metal_name}.")
                df = df.rename(columns={date_col: 'Date'})
            else:
                print(f"Critical Error: No suitable Date column found for {metal_name}. Cannot proceed with this metal.")
                return pd.DataFrame() # Return empty DataFrame


    # yfinance often returns 'Adj Close'
    # If it's not there, fall back to 'Close'
    price_col_options = ['Adj Close', 'Close']
    volume_col_options = ['Volume']

    selected_price_col = None
    for col in price_col_options:
        if col in df.columns:
            selected_price_col = col
            break
    if selected_price_col is None:
        print(f"Error: Neither 'Adj Close' nor 'Close' found for {metal_name}.")
        return pd.DataFrame()

    selected_volume_col = None
    for col in volume_col_options:
        if col in df.columns:
            selected_volume_col = col
            break
    if selected_volume_col is None:
        print(f"Warning: 'Volume' column not found for {metal_name}. Proceeding without volume data for this metal.")
        # Create a dummy volume column full of NaNs if we want to keep structure
        df['METAL_Volume_Dummy'] = pd.NA
        selected_volume_col = 'METAL_Volume_Dummy'


    df = df[['Date', selected_price_col, selected_volume_col]].copy()
    df.rename(columns={
        selected_price_col: f"{metal_name.upper()}_Adj_Close",
        selected_volume_col: f"{metal_name.upper()}_Volume"
    }, inplace=True)

    df.set_index('Date', inplace=True)
    return df

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame using ffill then bfill.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with missing values handled.
    """
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def ensure_data_types(df):
    """
    Ensures numeric columns are of float type.
    Assumes 'Date' is already the index and is datetime type.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with corrected data types.
    """
    for col in df.columns:
        if "Adj_Close" in col or "Volume" in col:
            try:
                df[col] = df[col].astype(float)
            except ValueError as e:
                print(f"Warning: Could not convert column {col} to float: {e}. Check for non-numeric data.")
                # Optionally, force conversion and set errors to NaN:
                # df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def run_data_cleansing():
    """
    Orchestrates the data cleansing process:
    - Prompts user for frequency.
    - Loads raw data for Gold, Silver, Platinum.
    - Cleans each DataFrame.
    - Merges them into a single DataFrame.
    - Saves the cleaned, merged DataFrame.
    """
    print("\n--- Data Cleansing Process ---")
    while True:
        frequency = input("Enter the frequency of raw data to cleanse (daily/weekly): ").lower()
        if frequency in ["daily", "weekly"]:
            break
        else:
            print("Invalid frequency. Please enter 'daily' or 'weekly'.")

    cleaned_dfs = []
    metal_names = list(TICKERS.keys()) # ["GOLD", "SILVER", "PLATINUM"]

    for metal_name in metal_names:
        print(f"\nProcessing {metal_name}...")
        raw_df = load_raw_data(metal_name, frequency)
        if raw_df is None or raw_df.empty:
            print(f"Skipping {metal_name} due to loading error or empty data.")
            continue

        processed_df = select_and_rename_columns(raw_df, metal_name)
        if processed_df.empty:
            print(f"Skipping {metal_name} due to column selection/renaming error.")
            continue

        # Ensure Date index is DatetimeIndex before filling, if not already
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            try:
                processed_df.index = pd.to_datetime(processed_df.index)
            except Exception as e:
                print(f"Error converting index to DatetimeIndex for {metal_name}: {e}. Skipping this metal.")
                continue

        # Sort by date before filling, crucial for time series
        processed_df.sort_index(inplace=True)

        processed_df = handle_missing_values(processed_df) # Handles NaNs within the series
        processed_df = ensure_data_types(processed_df)

        cleaned_dfs.append(processed_df)

    if not cleaned_dfs:
        print("\nNo data available to merge after individual processing. Exiting cleansing process.")
        return

    print("\nMerging DataFrames for all metals...")
    # Use pd.concat for joining on index
    # Outer join includes all dates from all dataframes
    merged_df = pd.concat(cleaned_dfs, axis=1, join='outer')

    # Sort index again after concat as it might not preserve order if dfs had different date ranges
    merged_df.sort_index(inplace=True)

    print("Handling missing values in merged DataFrame (post-merge fill)...")
    # Apply ffill and bfill again on the merged frame to handle NaNs from outer join
    # This fills gaps where one metal might have data and another doesn't on a particular day
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)

    # Drop rows where all price data is still NaN
    # This typically affects rows at the very beginning if bfill couldn't find data
    price_columns = [col for col in merged_df.columns if "Adj_Close" in col]
    if not price_columns:
        print("Error: No price columns found in the merged DataFrame. Cannot proceed.")
        return

    merged_df.dropna(subset=price_columns, how='all', inplace=True)

    if merged_df.empty:
        print("Merged DataFrame is empty after NaN handling. No data to save.")
        return

    # Save the cleaned and merged data
    output_filename = f"cleaned_combined_{frequency.lower()}_prices.csv"
    output_filepath = os.path.join(CLEANED_DATA_DIR, output_filename)

    try:
        merged_df.to_csv(output_filepath)
        print(f"\nSuccessfully saved cleaned and merged data to: {output_filepath}")
        print(f"Shape of the cleaned data: {merged_df.shape}")
        print("Sample of cleaned data (first 5 rows):")
        print(merged_df.head())
    except Exception as e:
        print(f"Error saving cleaned data to {output_filepath}: {e}")

if __name__ == "__main__":
    # Example of how to run (assuming raw data files exist)
    # 1. First, you'd need to run data_crawler.py or the fetch-data CLI command
    #    to generate the raw CSVs (e.g., gold_daily_prices.csv)
    # 2. Then, you can run this script directly for testing.

    # For direct testing, we might need to ensure data_crawler's constants are accessible
    # or simulate the presence of data files.

    print("Running Data Cleanser module directly for testing...")
    # Create dummy raw data files for testing if they don't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    dummy_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    dummy_data_gold = {
        'Date': dummy_dates,
        'Open': [180, 181, 180.5, 182, 181.5],
        'High': [182, 181.5, 181, 183, 182],
        'Low': [179, 180, 180, 181, 180.5],
        'Close': [181, 180.5, 180.8, 182.5, 181],
        'Adj Close': [181, 180.5, 180.8, 182.5, 181], # yfinance typically provides this
        'Volume': [10000, 12000, 11000, 13000, 10500]
    }
    dummy_df_gold = pd.DataFrame(dummy_data_gold)
    # Introduce a NaN
    dummy_df_gold.loc[2, 'Adj Close'] = pd.NA
    dummy_df_gold.loc[3, 'Volume'] = pd.NA

    dummy_data_silver = {
        'Date': dummy_dates, # Same dates for simplicity in dummy data
        'Adj Close': [22, 22.1, 21.9, pd.NA, 22.3], # NaN here
        'Volume': [5000, 5200, 5100, 5300, 5050]
    }
    dummy_df_silver = pd.DataFrame(dummy_data_silver)

    # Platinum with slightly different dates for testing outer join and ffill/bfill
    dummy_dates_plat = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
    dummy_data_platinum = {
        'Date': dummy_dates_plat,
        'Adj Close': [1000, 1002, 1001, 1005, 1003],
        'Volume': [2000, 2200, 2100, 2300, 2050]
    }
    dummy_df_platinum = pd.DataFrame(dummy_data_platinum)


    # Save dummy files if they don't exist to allow direct execution for testing
    # This is a simplified version; actual data would come from data_crawler
    if not os.path.exists(os.path.join(DATA_DIR, "gold_daily_prices.csv")):
        dummy_df_gold.to_csv(os.path.join(DATA_DIR, "gold_daily_prices.csv"), index=False)
    if not os.path.exists(os.path.join(DATA_DIR, "silver_daily_prices.csv")):
        dummy_df_silver.to_csv(os.path.join(DATA_DIR, "silver_daily_prices.csv"), index=False)
    if not os.path.exists(os.path.join(DATA_DIR, "platinum_daily_prices.csv")):
         dummy_df_platinum.to_csv(os.path.join(DATA_DIR, "platinum_daily_prices.csv"), index=False)

    run_data_cleansing()
