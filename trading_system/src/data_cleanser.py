import pandas as pd
import os
from .data_crawler import TICKERS, DATA_DIR as METAL_DATA_DIR # Use . for relative import
# Import MACRO_INDICATORS_FRED from macro_data_crawler to know what files to look for
from .macro_data_crawler import MACRO_INDICATORS_FRED, DATA_DIR_MACRO

CLEANED_DATA_DIR = METAL_DATA_DIR # Base directory for cleaned outputs
MERGED_DATA_DIR = METAL_DATA_DIR # Directory for final merged output (metal + macro)
# Could be a different sub-directory like data/cleaned/ or data/final/ if preferred later

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
    filepath = os.path.join(METAL_DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"Error: Raw metal price data file not found: {filepath}")
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

def run_data_cleansing(frequency_to_clean):
    """
    Orchestrates the data cleansing process for metal prices.
    - Takes frequency as a parameter.
    - Loads raw data for Gold, Silver, Platinum for that frequency.
    - Cleans each DataFrame.
    - Merges them into a single DataFrame.
    - Saves the cleaned, merged DataFrame.
    Args:
        frequency_to_clean (str): "daily" or "weekly".
    """
    print(f"\n--- Data Cleansing Process for {frequency_to_clean} metal prices ---")

    if frequency_to_clean not in ["daily", "weekly"]:
        print("Invalid frequency provided to run_data_cleansing. Must be 'daily' or 'weekly'.")
        return

    cleaned_dfs = []
    metal_names = list(TICKERS.keys()) # ["GOLD", "SILVER", "PLATINUM"]

    for metal_name in metal_names:
        print(f"\nProcessing {metal_name}...")
        raw_df = load_raw_data(metal_name, frequency_to_clean)
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
    output_filename = f"cleaned_combined_{frequency_to_clean.lower()}_prices.csv"
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
    if not os.path.exists(METAL_DATA_DIR): # Corrected from DATA_DIR to METAL_DATA_DIR
        os.makedirs(METAL_DATA_DIR)

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
    if not os.path.exists(os.path.join(METAL_DATA_DIR, "gold_daily_prices.csv")):
        dummy_df_gold.to_csv(os.path.join(METAL_DATA_DIR, "gold_daily_prices.csv"), index=False)
    if not os.path.exists(os.path.join(METAL_DATA_DIR, "silver_daily_prices.csv")):
        dummy_df_silver.to_csv(os.path.join(METAL_DATA_DIR, "silver_daily_prices.csv"), index=False)
    if not os.path.exists(os.path.join(METAL_DATA_DIR, "platinum_daily_prices.csv")):
         dummy_df_platinum.to_csv(os.path.join(METAL_DATA_DIR, "platinum_daily_prices.csv"), index=False)

    run_data_cleansing(frequency_to_clean="daily") # Example: pass frequency directly
    # After running run_data_cleansing(), you can then test run_merge_with_macro_data()
    # Example: Assuming 'daily' frequency was used for run_data_cleansing()
    # print("\n--- Testing Macro Data Merging ---")
    # create_dummy_macro_files() # You'd need a helper to create dummy macro files for testing
    # run_merge_with_macro_data(metal_data_frequency="daily")


# --- Functions for Macro Data Processing and Merging ---

def load_single_macro_indicator(indicator_details):
    """
    Loads a single processed macro indicator CSV file.
    The filename is constructed based on indicator_details.
    """
    country_code = indicator_details["country_code"].lower()
    # Construct filename part from indicator name, e.g., US_REAL_GDP_YOY_QUARTERLY -> real_gdp_yoy_quarterly
    filename_indicator_part = "_".join(indicator_details["name"].split("_")[1:]).lower()

    filename = f"{country_code}_{filename_indicator_part}_fred.csv"
filename_indicator_part = "_".join(indicator_details["name"].split("_")[1:]).lower()

    filename = f"{country_code}_{filename_indicator_part}_fred.csv"
    # Use os.path.normpath and os.path.abspath to sanitize the file path
    # import os
    filepath = os.path.normpath(os.path.join(DATA_DIR_MACRO, filename))
    if not filepath.startswith(os.path.abspath(DATA_DIR_MACRO)):
        print(f"Error: Invalid file path: {filepath}")
        return None

    if not os.path.exists(filepath):
        print(f"Info: Macro data file not found, will be skipped: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        if df.empty:
            print(f"Warning: Macro data file is empty: {filepath}")
            return None

    if not os.path.exists(filepath):
        print(f"Info: Macro data file not found, will be skipped: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        # The CSVs from macro_data_crawler should have the target_name as column name
        # df.rename(columns={df.columns[0]: indicator_details["name"]}, inplace=True)
        if df.empty:
            print(f"Warning: Macro data file is empty: {filepath}")
            return None
        # Ensure column name matches the indicator's target name if there's any mismatch potential
        if df.columns[0] != indicator_details["name"]:
            print(f"Warning: Column name mismatch in {filepath}. Expected '{indicator_details['name']}', found '{df.columns[0]}'. Renaming.")
            df.rename(columns={df.columns[0]: indicator_details["name"]}, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading macro data from {filepath}: {e}")
        return None

def resample_and_fill_series(series, target_frequency_pd, series_name=""):
    """
    Resamples a series to the target frequency and forward fills.
    target_frequency_pd: 'D' for daily, 'W-FRI' for weekly (ending Friday), etc.
    """
    if series is None or series.empty:
        return None

    # print(f"Original series {series_name} head before resample:\n{series.head(3)}")
    # print(f"Original series {series_name} tail before resample:\n{series.tail(3)}")
    # print(f"Resampling {series_name} to {target_frequency_pd}...")

    # Resample to the target frequency. This creates NaNs for new timepoints.
    resampled_series = series.resample(target_frequency_pd).asfreq()

    # Forward fill the NaNs.
    # For macro data, ffill is generally appropriate (value persists until new one is known)
    filled_series = resampled_series.ffill()

    # print(f"Filled series {series_name} head after resample/ffill:\n{filled_series.head(7)}")
    # print(f"Filled series {series_name} tail after resample/ffill:\n{filled_series.tail(7)}")
    return filled_series


def run_merge_with_macro_data(metal_data_frequency):
    """
    Orchestrates the merging of cleaned metal price data with processed macro data.
    - Loads the `cleaned_combined_{metal_data_frequency}_prices.csv`.
    - Loads all available macro indicators specified in MACRO_INDICATORS_FRED.
    - Processes and resamples each macro indicator.
    - Merges them with the metal prices.
    - Saves the final combined DataFrame.
    """
    # Import statement for secure input validation
    # import re  # Used for input validation

    print(f"
--- Merging Metal Prices with Macroeconomic Data ({metal_data_frequency} frequency) ---")

    # Validate metal_data_frequency input
    valid_frequencies = ['daily', 'weekly']
    if not isinstance(metal_data_frequency, str) or metal_data_frequency.lower() not in valid_frequencies:
        raise ValueError(f"Invalid metal_data_frequency. Must be one of {valid_frequencies}")

    metal_data_frequency = metal_data_frequency.lower()

    # Rest of the function remains unchanged
    # ...
    """
    Orchestrates the merging of cleaned metal price data with processed macro data.
    - Loads the `cleaned_combined_{metal_data_frequency}_prices.csv`.
    - Loads all available macro indicators specified in MACRO_INDICATORS_FRED.
    - Processes and resamples each macro indicator.
    - Merges them with the metal prices.
    - Saves the final combined DataFrame.
    """
    print(f"\n--- Merging Metal Prices with Macroeconomic Data ({metal_data_frequency} frequency) ---")

    # 1. Load cleaned metal price data
    cleaned_metal_filename = f"cleaned_combined_{metal_data_frequency.lower()}_prices.csv"
    cleaned_metal_filepath = os.path.join(CLEANED_DATA_DIR, cleaned_metal_filename)

    if not os.path.exists(cleaned_metal_filepath):
        print(f"Error: Cleaned metal price data file not found: {cleaned_metal_filepath}")
        print("Please run 'clean-data' command first for the desired frequency.")
        return

    try:
        df_metals = pd.read_csv(cleaned_metal_filepath, index_col='Date', parse_dates=True)
        print(f"Loaded cleaned metal prices from: {cleaned_metal_filepath} (Shape: {df_metals.shape})")
    except Exception as e:
        print(f"Error loading cleaned metal prices from {cleaned_metal_filepath}: {e}")
        return

    if df_metals.empty:
        print("Cleaned metal price data is empty. Cannot proceed with merge.")
        return

    # Determine pandas target frequency string for resampling
    if metal_data_frequency == "daily":
        pd_target_freq = "D" # Daily
    elif metal_data_frequency == "weekly":
        # Assuming weekly metal data is aligned to Friday or similar.
        # If it's daily data being targeted to weekly, need a consistent day.
        # If it's already weekly, the index should be fine.
        # For resampling macro data to weekly, let's align to Friday.
        pd_target_freq = "W-FRI"
        # If df_metals is already weekly, we might want to use its exact dates.
        # For now, let's make sure df_metals itself is on this frequency if it's weekly.
        if not pd.infer_freq(df_metals.index) == 'W-FRI': # Or other weekly freq
             print(f"Warning: Metal data frequency is weekly, but not inferred as 'W-FRI'. Resampling metals to W-FRI.")
             df_metals = df_metals.resample('W-FRI').last() # Use last observation in the week for metals
    else:
        print(f"Error: Unsupported metal_data_frequency for merging: {metal_data_frequency}")
        return

    # 2. Load, process, and collect all macro indicators
    all_processed_macro_series = []
    skipped_indicators_count = 0

    for indicator_details in MACRO_INDICATORS_FRED:
        if indicator_details["id"] == "TODO_FIND_CN_RATE": # Skip placeholders defined in crawler
            print(f"Skipping placeholder macro indicator: {indicator_details['name']}")
            skipped_indicators_count +=1
            continue

        print(f"\nProcessing macro indicator: {indicator_details['name']}")
        df_indicator = load_single_macro_indicator(indicator_details)

        if df_indicator is None or df_indicator.empty:
            print(f"No data loaded for {indicator_details['name']}. Skipping this indicator.")
            skipped_indicators_count +=1
            continue

        # Assuming df_indicator has one column with the indicator_details["name"]
        series_indicator = df_indicator[indicator_details["name"]]

        # Transformations (e.g., GDP to YoY) are assumed to be done by macro_data_crawler.py
        # Here, we primarily focus on resampling and filling to match metal data frequency.

        # Ensure the series index is DatetimeIndex
        if not isinstance(series_indicator.index, pd.DatetimeIndex):
            try:
                series_indicator.index = pd.to_datetime(series_indicator.index)
            except Exception as e:
                print(f"Error converting index to DatetimeIndex for {indicator_details['name']}: {e}. Skipping.")
                skipped_indicators_count +=1
                continue

        series_indicator.sort_index(inplace=True) # Ensure chronological order

        # Resample and forward-fill
        # The resampling target frequency should align with the metal data's index.
        # We use pd_target_freq which is 'D' or 'W-FRI'.
        processed_series = resample_and_fill_series(series_indicator, pd_target_freq, series_name=indicator_details['name'])

        if processed_series is not None and not processed_series.empty:
            all_processed_macro_series.append(processed_series)
        else:
            print(f"No data after processing for {indicator_details['name']}. Skipping.")
            skipped_indicators_count +=1

    if not all_processed_macro_series:
        print("\nNo macro data could be processed. Merging will only contain metal prices.")
        # We could save just the (potentially resampled) metal prices, or exit.
        # For now, let's proceed, the final merge will just be df_metals.
    else:
        print(f"\nSuccessfully processed {len(all_processed_macro_series)} macro indicators (skipped {skipped_indicators_count}).")


    # 3. Merge processed macro series with metal price DataFrame
    # Start with df_metals as the base for merging
    df_final_merged = df_metals.copy()

    for macro_series in all_processed_macro_series:
        if macro_series is not None and not macro_series.empty:
            # Merge using an outer join to keep all dates from both, then align.
            # However, since we resample macro_series to the metal_data index's frequency,
            # a left join on df_metals' index should be appropriate.
            # This ensures the final DataFrame uses the date range of the metal prices.
            df_final_merged = df_final_merged.join(macro_series, how='left')
            print(f"Joined {macro_series.name}. Shape after join: {df_final_merged.shape}")


    # After joining, the macro columns might have NaNs at the beginning if metal data starts earlier
    # than the (resampled and filled) macro data.
    # Also, if metal data has gaps, left join will preserve those NaNs for metal prices.
    # We should forward-fill the newly joined macro columns to fill these leading NaNs.
    # This ensures that the earliest available macro data is propagated backward to the start of the metal data period.

    macro_column_names = [s.name for s in all_processed_macro_series if s is not None]
    if macro_column_names:
        print("\nForward-filling macro columns in the final merged DataFrame...")
        df_final_merged[macro_column_names] = df_final_merged[macro_column_names].ffill()
        # Optionally, also back-fill to handle cases where macro data might start later AND end earlier
        # but this is less common if macro data is usually available up to recent dates.
        # df_final_merged[macro_column_names] = df_final_merged[macro_column_names].bfill()


    # 4. Save the final merged DataFrame
    output_filename = f"final_combined_{metal_data_frequency.lower()}_data.csv"
    output_filepath = os.path.join(MERGED_DATA_DIR, output_filename)

    try:
        df_final_merged.to_csv(output_filepath)
        print(f"\nSuccessfully saved final merged data (metals + macro) to: {output_filepath}")
        print(f"Shape of the final merged data: {df_final_merged.shape}")
        print("Sample of final merged data (first 5 rows):")
        print(df_final_merged.head())
        print("Sample of final merged data (last 5 rows):")
        print(df_final_merged.tail())
        # Check for NaNs in the final output
        nan_counts = df_final_merged.isnull().sum()
        if nan_counts.sum() > 0:
            print("\nNaN values found in the final merged DataFrame:")
            print(nan_counts[nan_counts > 0])
        else:
            print("\nNo NaN values found in the final merged DataFrame.")

    except Exception as e:
        print(f"Error saving final merged data to {output_filepath}: {e}")

def create_dummy_macro_files():
    """Helper to create dummy macro files for testing run_merge_with_macro_data"""
    if not os.path.exists(DATA_DIR_MACRO):
        os.makedirs(DATA_DIR_MACRO)

    dummy_macro_data = {
        "US_REAL_GDP_YOY_QUARTERLY": pd.Series(
            [2.5, 2.6, 2.7, 2.8, 2.4],
            index=pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01', '2021-01-01']),
            name="US_REAL_GDP_YOY_QUARTERLY"
        ),
        "US_CPI_YOY_MONTHLY": pd.Series(
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
            index=pd.date_range(start='2020-01-01', periods=12, freq='MS'), # Monthly Start
            name="US_CPI_YOY_MONTHLY"
        ),
        "US_FED_FUNDS_RATE_MONTHLY": pd.Series(
            [0.25, 0.25, 0.15, 0.10, 0.10, 0.10],
            index=pd.date_range(start='2020-01-01', periods=6, freq='MS'),
            name="US_FED_FUNDS_RATE_MONTHLY"
        )
    }

    for indicator_config in MACRO_INDICATORS_FRED:
        indicator_name = indicator_config["name"]
        if indicator_name in dummy_macro_data:
            series = dummy_macro_data[indicator_name]
            country_code = indicator_config["country_code"].lower()
            filename_part = "_".join(indicator_name.split("_")[1:]).lower()
            filepath = os.path.join(DATA_DIR_MACRO, f"{country_code}_{filename_part}_fred.csv")
            series.to_frame().to_csv(filepath)
            print(f"Created dummy macro file: {filepath}")
        # Create empty files for other indicators to test skipping logic
        elif indicator_config["id"] != "TODO_FIND_CN_RATE":
            country_code = indicator_config["country_code"].lower()
            filename_part = "_".join(indicator_name.split("_")[1:]).lower()
            filepath = os.path.join(DATA_DIR_MACRO, f"{country_code}_{filename_part}_fred.csv")
            if not os.path.exists(filepath): # Avoid overwriting if some were created by crawler
                 pd.DataFrame().to_csv(filepath) # Empty dataframe
                 print(f"Created empty dummy macro file (to test skipping): {filepath}")
    print("Dummy macro files creation attempt finished.")
