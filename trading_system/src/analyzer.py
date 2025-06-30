import pandas as pd
import os
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error


from .data_cleanser import CLEANED_DATA_DIR # Using CLEANED_DATA_DIR from data_cleanser

# Constants for column name patterns
ADJ_CLOSE_SUFFIX = "_Adj_Close"
VOLUME_SUFFIX = "_Volume"
TICKERS_LIST = ["GOLD", "SILVER", "PLATINUM"] # Should align with data_crawler.TICKERS keys


def test_stationarity(series, series_name="Unnamed Series", significance_level=0.05, verbose=True):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Args:
        series (pd.Series): The time series data to test.
        series_name (str): Name of the series for printing.
        significance_level (float): The significance level for hypothesis testing.
        verbose (bool): If True, prints detailed results.

    Returns:
        bool: True if the series is stationary, False otherwise.
    """
    if verbose:
        print(f"\n--- ADF Test for Stationarity: {series_name} ---")

    # Handle potential NaNs by dropping them, as ADF test cannot handle them
    series_cleaned = series.dropna()
    if series_cleaned.empty:
        if verbose:
            print("Series is empty after dropping NaNs. Cannot perform ADF test.")
        return False # Or handle as an error

    try:
        result = adfuller(series_cleaned)
        p_value = result[1]
        is_stationary = p_value < significance_level

        if verbose:
            print(f"ADF Statistic: {result[0]:.4f}")
            print(f"P-value: {p_value:.4f}")
            print("Critical Values:")
            for key, value in result[4].items():
                print(f"\t{key}: {value:.4f}")

            if is_stationary:
                print(f"Result: The series '{series_name}' is likely STATIONARY (p-value < {significance_level}).")
            else:
                print(f"Result: The series '{series_name}' is likely NON-STATIONARY (p-value >= {significance_level}).")

        return is_stationary
    except Exception as e:
        if verbose:
            print(f"Error performing ADF test for {series_name}: {e}")
        return False # Consider the series non-stationary or handle error appropriately


def perform_cointegration_test(df, det_order=0, k_ar_diff=1, significance_level=0.05, verbose=True):
    """
    Performs the Johansen cointegration test.

    Args:
        df (pd.DataFrame): DataFrame with the time series (e.g., Adj Close prices).
                           Columns should be the series to test.
        det_order (int): Deterministic trend order. 0 for constant, 1 for constant and trend.
        k_ar_diff (int): Number of lagged differences in the VAR model.
        significance_level (float): Significance level to compare critical values (e.g., 0.05 for 95% confidence).
        verbose (bool): If True, prints detailed results.

    Returns:
        int: The number of cointegrating relationships found.
    """
    if verbose:
        print(f"\n--- Johansen Cointegration Test (det_order={det_order}, k_ar_diff={k_ar_diff}) ---")
        print(f"Series being tested: {', '.join(df.columns)}")

    if df.isnull().values.any():
        print("Warning: DataFrame contains NaN values. Dropping rows with NaNs before cointegration test.")
        df_cleaned = df.dropna()
    else:
        df_cleaned = df

    if len(df_cleaned) < df_cleaned.shape[1] * 2 or len(df_cleaned) < k_ar_diff + 2 : # Basic check for sufficient data
        print("Error: Not enough observations for Johansen test after NaN handling or due to k_ar_diff.")
        return -1 # Indicate error or insufficient data

    try:
        result = coint_johansen(df_cleaned, det_order=det_order, k_ar_diff=k_ar_diff)
    except Exception as e:
        print(f"Error during Johansen cointegration test: {e}")
        return -1


    # Determine critical value column based on significance level
    # Critical values array has shape (n_series, 3) for 90%, 95%, 99%
    crit_value_col_idx = 1 # Default to 95% (0.05 significance)
    if significance_level == 0.10: # 90%
        crit_value_col_idx = 0
    elif significance_level == 0.01: # 99%
        crit_value_col_idx = 2

    num_series = df_cleaned.shape[1]
    num_cointegrating_relations_trace = 0
    num_cointegrating_relations_max_eig = 0

    if verbose:
        print("\nTrace Statistic Test:")
        print("Trace Stat | Crit 90%   | Crit 95%   | Crit 99%   | Null Hypothesis (r<=)")
        print("--------------------------------------------------------------------------")
    for i in range(num_series):
        trace_stat = result.lr1[i]
        crit_90, crit_95, crit_99 = result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2]
        if verbose:
            print(f"{trace_stat:<10.3f} | {crit_90:<10.3f} | {crit_95:<10.3f} | {crit_99:<10.3f} | r <= {i}")
        # For trace test, we reject H0 if Trace Stat > Critical Value
        # We are looking for the first time we FAIL to reject H0. That 'i' is the number of cointegrating relations.
        if trace_stat > result.cvt[i, crit_value_col_idx]:
            num_cointegrating_relations_trace = i + 1
        else: # First time we fail to reject
            if i == num_cointegrating_relations_trace: # handles the r=0 case correctly if it wasn't incremented
                 pass # num_cointegrating_relations_trace remains what it was (potentially 0 if first test fails to reject)
            # No need to break here for trace, standard practice is to show all then conclude.
            # However, for determining the number, the first non-rejection is key.
            # The loop continues but num_cointegrating_relations_trace should hold the highest r for which H0 was rejected.

    if verbose:
        print("\nMax Eigenvalue Statistic Test:")
        print("Max Eigen  | Crit 90%   | Crit 95%   | Crit 99%   | Null Hypothesis (r=) vs (r=r+1)")
        print("------------------------------------------------------------------------------------")
    for i in range(num_series):
        max_eig_stat = result.lr2[i]
        crit_90, crit_95, crit_99 = result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2]
        if verbose:
             print(f"{max_eig_stat:<10.3f} | {crit_90:<10.3f} | {crit_95:<10.3f} | {crit_99:<10.3f} | r = {i} vs r = {i+1}")
        # For max eigenvalue test, we reject H0 if Max Eigen Stat > Critical Value
        # We are looking for the first time we FAIL to reject H0. That 'i' is the number of cointegrating relations.
        if max_eig_stat > result.cvm[i, crit_value_col_idx]:
            num_cointegrating_relations_max_eig = i + 1
        else:
            if i == num_cointegrating_relations_max_eig:
                pass


    # Typically, users might prefer one test or look for agreement.
    # For simplicity, we can report both or choose one (e.g., trace test result).
    # A more robust approach might involve checking if both agree or using specific rules.
    # Here, we'll take the result from the trace statistic as primary for the return value.
    final_num_coint_relations = num_cointegrating_relations_trace

    if verbose:
        print(f"\nConclusion (at {significance_level*100}% significance):")
        print(f"Based on Trace Statistic: {num_cointegrating_relations_trace} cointegrating relationship(s) found.")
        print(f"Based on Max Eigenvalue Statistic: {num_cointegrating_relations_max_eig} cointegrating relationship(s) found.")
        # print(f"Reporting {final_num_coint_relations} cointegrating relationship(s).")

    return final_num_coint_relations


def run_multivariate_analysis():
    """
    Orchestrates the multivariate analysis process:
    - Loads cleaned data.
    - Prompts user for levels or returns.
    - Performs stationarity tests.
    - Performs cointegration test (if levels are chosen).
    - Prints summary and guidance for next modeling steps.
    """
    print("\n--- Multivariate Analysis ---")

    while True:
        frequency = input("Enter data frequency of cleaned file to analyze (daily/weekly): ").lower()
        if frequency in ["daily", "weekly"]:
            break
        else:
            print("Invalid frequency. Please enter 'daily' or 'weekly'.")

    cleaned_filename = f"cleaned_combined_{frequency.lower()}_prices.csv"
    cleaned_filepath = os.path.join(CLEANED_DATA_DIR, cleaned_filename)

    if not os.path.exists(cleaned_filepath):
        print(f"Error: Cleaned data file not found: {cleaned_filepath}")
        print("Please run the 'fetch-data' and 'clean-data' commands first.")
        return

    try:
        df_cleaned = pd.read_csv(cleaned_filepath, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Error loading cleaned data from {cleaned_filepath}: {e}")
        return

    # Select only the Adj Close columns for Gold, Silver, Platinum
    adj_close_cols = [f"{ticker}{ADJ_CLOSE_SUFFIX}" for ticker in TICKERS_LIST]

    # Verify that these columns exist
    missing_cols = [col for col in adj_close_cols if col not in df_cleaned.columns]
    if missing_cols:
        print(f"Error: The following expected price columns are missing in {cleaned_filename}: {', '.join(missing_cols)}")
        return

    df_analysis = df_cleaned[adj_close_cols].copy()
    df_analysis.dropna(inplace=True) # Drop any rows with NaNs in price data before analysis

    if df_analysis.empty:
        print("DataFrame is empty after selecting price columns and dropping NaNs. Cannot proceed.")
        return

    while True:
        analysis_type = input("Analyze price levels or percentage returns? (levels/returns): ").lower()
        if analysis_type in ["levels", "returns"]:
            break
        else:
            print("Invalid choice. Please enter 'levels' or 'returns'.")

    data_to_test = df_analysis
    if analysis_type == "returns":
        data_to_test = df_analysis.pct_change().dropna()
        print("\nCalculating percentage returns for analysis...")
        if data_to_test.empty:
            print("DataFrame of returns is empty (e.g. only one row in original data). Cannot proceed.")
            return

    print(f"\nPerforming stationarity tests on {analysis_type}...")
    stationarity_results = {}
    for col_name in data_to_test.columns:
        is_stationary = test_stationarity(data_to_test[col_name], series_name=col_name)
        stationarity_results[col_name] = is_stationary

    num_coint_relations = -1 # Default to error/not run
    if analysis_type == "levels":
        # Cointegration test is typically done on price levels
        # Check if all level series are non-stationary before suggesting cointegration makes sense
        all_levels_non_stationary = all(not stat for stat in stationarity_results.values())

        if not all_levels_non_stationary:
            print("\nNote: Not all price level series were found to be non-stationary.")
            print("Cointegration analysis is most meaningful when all series are I(1) (non-stationary in levels, stationary in first differences).")
            print("Proceeding with cointegration test on levels, but interpret with caution if some series are already stationary.")

        # Ensure there are enough observations for k_ar_diff.
        # A common default for k_ar_diff (lags in VAR) is 1 or 2 for daily/weekly.
        # Johansen test needs more data points than (num_series * k_ar_diff + num_series + 1)
        # For 3 series, k_ar_diff=1, det_order=0, it implies > 3*1 + 3 + 1 = 7 data points. Let's be safer.
        k_ar_lags = 1 # A common starting point for lags in the VAR for Johansen
        if len(data_to_test) > (data_to_test.shape[1] * k_ar_lags + data_to_test.shape[1] + 10): # Heuristic check
            num_coint_relations = perform_cointegration_test(data_to_test, k_ar_diff=k_ar_lags)
        else:
            print(f"Skipping Johansen cointegration test: Insufficient data points ({len(data_to_test)}) for the number of series and lags.")
            print("Need more historical data to perform a reliable cointegration test.")


    print("\n--- Multivariate Analysis Summary ---")
    print(f"Analysis performed on: {analysis_type.capitalize()}")
    print("Stationarity Results:")
    for series, is_stat in stationarity_results.items():
        print(f"  - {series}: {'Stationary' if is_stat else 'Non-Stationary'}")

    if analysis_type == "levels":
        if num_coint_relations > 0:
            print(f"\nCointegration Test on Price Levels found {num_coint_relations} cointegrating relationship(s).")
            print("Recommendation: A Vector Error Correction Model (VECM) may be appropriate for these series.")
        elif num_coint_relations == 0:
            print("\nCointegration Test on Price Levels found NO cointegrating relationships.")
            print("Recommendation: If series are non-stationary, consider modeling their first differences (returns) using a VAR model.")
            print("If series are already stationary in levels, a VAR model on levels might be appropriate.")
        else: # num_coint_relations == -1 (error or not enough data)
            print("\nCointegration Test on Price Levels could not be reliably performed or was skipped.")
            print("Recommendation: Check data and test parameters. If series are non-stationary, consider VAR on returns.")

    elif analysis_type == "returns":
        all_returns_stationary = all(stationarity_results.values())
        if all_returns_stationary:
            print("\nAll return series are likely stationary.")
            print("Recommendation: A VAR model on these returns may be appropriate for further modeling.")
        else:
            print("\nNot all return series were found to be stationary. Further transformation or investigation may be needed before VAR modeling.")
            print("Consider checking individual ADF test p-values and lag selection.")

    print("\nFurther modeling (VAR/VECM estimation, forecasting, IRF, Granger causality) can be implemented based on these findings.")


if __name__ == "__main__":
    # This block is for testing the analyzer module directly.
    # It assumes that 'cleaned_combined_daily_prices.csv' or 'cleaned_combined_weekly_prices.csv'
    # exists in the CLEANED_DATA_DIR (which is 'data/').
    # You would typically run data_crawler and data_cleanser first.

    print("Running Analyzer module directly for testing multivariate functions...")

    # To make this runnable, let's create a dummy cleaned file if it doesn't exist
    dummy_cleaned_file_path = os.path.join(CLEANED_DATA_DIR, "cleaned_combined_daily_prices.csv")
    if not os.path.exists(dummy_cleaned_file_path):
        print(f"Dummy file {dummy_cleaned_file_path} not found. Creating one for test.")
        if not os.path.exists(CLEANED_DATA_DIR):
            os.makedirs(CLEANED_DATA_DIR)

        dates = pd.date_range(start='2022-01-01', periods=100, freq='B') # Approx 100 business days
        data = {
            'GOLD_Adj_Close': [1800 + i + 10 * (i//20) + (j*0.1 if j % 2 == 0 else -j*0.1) for i,j in enumerate(range(100))], # Trend + noise
            'GOLD_Volume': [100000 + i*100 for i in range(100)],
            'SILVER_Adj_Close': [22 + 0.05 * i + 5 * (i//25) + (j*0.05 if j % 3 == 0 else -j*0.05) for i,j in enumerate(range(100))], # Different trend + noise
            'SILVER_Volume': [500000 + i*50 for i in range(100)],
            'PLATINUM_Adj_Close': [1000 + 0.5 * i + (-1)**i * 2 + (j*0.08 if j % 2 != 0 else -j*0.08) for i,j in enumerate(range(100))], # Another pattern
            'PLATINUM_Volume': [20000 + i*20 for i in range(100)]
        }
        # Introduce some cointegration-like behavior for GOLD and SILVER for test
        data['SILVER_Adj_Close'] = [g * (22/1800) + 2*((-1)**i) for i,g in enumerate(data['GOLD_Adj_Close'])]

        dummy_df = pd.DataFrame(data, index=dates)
        dummy_df.index.name = 'Date'
        # Add some NaNs to test handling
        dummy_df.loc[dummy_df.index[10], 'GOLD_Adj_Close'] = pd.NA
        dummy_df.loc[dummy_df.index[20], 'SILVER_Adj_Close'] = pd.NA

        dummy_df.to_csv(dummy_cleaned_file_path)
        print(f"Dummy file {dummy_cleaned_file_path} created.")

    run_multivariate_analysis()
    # To test deep learning part (if dummy file is suitable or if you fetch real data first):
    # print("\nRunning Analyzer module directly for testing deep learning functions...")
    # run_deep_learning_forecast() # This would require more setup for dummy data or real data
