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


# Import statements
import os
import pandas as pd
from typing import Dict, Any
from auth_service import validate_user_role  # Import the authentication service

def run_multivariate_analysis(frequency: str, analysis_type: str, data_dir: str = CLEANED_DATA_DIR, verbose: bool = True) -> Dict[str, Any]:
    """
    Orchestrates the multivariate analysis process with proper role-based access control.
    
    Args:
        frequency (str): Data frequency to analyze ("daily" or "weekly").
        analysis_type (str): Type of analysis ("levels" or "returns").
        data_dir (str): Directory where cleaned data files are stored. Defaults to CLEANED_DATA_DIR.
        verbose (bool): If True, prints detailed logs to console.

    Returns:
        dict: A dictionary containing analysis results or error message.
    """
    if not validate_user_role('analyst'):
        return {"error": "Unauthorized access. User does not have the required role."}

    # Rest of the function remains unchanged
    # ...
    """
    Orchestrates the multivariate analysis process:
    - Loads cleaned data based on specified frequency.
    - Performs analysis based on specified type (levels or returns).
    - Performs stationarity tests.
    - Performs cointegration test (if levels are chosen).
    - Prints summary and guidance for next modeling steps if verbose is True.

    Args:
        frequency (str): Data frequency to analyze ("daily" or "weekly").
        analysis_type (str): Type of analysis ("levels" or "returns").
        data_dir (str): Directory where cleaned data files are stored. Defaults to CLEANED_DATA_DIR.
        verbose (bool): If True, prints detailed logs to console.

    Returns:
        dict: A dictionary containing analysis results (stationarity, cointegration).
              Returns None if critical errors occur (e.g., file not found).
    """
    if verbose:
        print("\n--- Multivariate Analysis ---")

    if frequency not in ["daily", "weekly"]:
        if verbose:
            print(f"Invalid frequency: {frequency}. Please use 'daily' or 'weekly'.")
        # Consider raising an error or returning a specific failure indicator
        return {"error": f"Invalid frequency: {frequency}. Please use 'daily' or 'weekly'."}

    if analysis_type not in ["levels", "returns"]:
        if verbose:
            print(f"Invalid analysis_type: {analysis_type}. Please use 'levels' or 'returns'.")
        return {"error": f"Invalid analysis_type: {analysis_type}. Please use 'levels' or 'returns'."}

    cleaned_filename = f"cleaned_combined_{frequency.lower()}_prices.csv"
# Import os.path for secure path handling
# Import pathlib for secure path validation
import os.path
from pathlib import Path

def run_multivariate_analysis(frequency: str, analysis_type: str, data_dir: str = CLEANED_DATA_DIR, verbose: bool = True):
    # ... (previous code remains unchanged)

    cleaned_filename = f"cleaned_combined_{frequency.lower()}_prices.csv"
    
    # Use os.path.abspath to get the absolute path and resolve any '..' in the path
    data_dir_abs = os.path.abspath(data_dir)
    
    # Use Path to create a path object and resolve to its absolute path
    cleaned_filepath = Path(data_dir_abs).resolve() / cleaned_filename
    
    # Ensure the resolved path is still within the intended directory
    if not str(cleaned_filepath).startswith(str(Path(CLEANED_DATA_DIR).resolve())):
        print(f"Error: Invalid data directory path: {data_dir}")
        return

    if not cleaned_filepath.exists():
        print(f"Error: Cleaned data file not found: {cleaned_filepath}")
        print("Please run the 'fetch-data' and 'clean-data' commands first.")
        return

    # ... (rest of the code remains unchanged)
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
        if verbose:
            print("DataFrame is empty after selecting price columns and dropping NaNs. Cannot proceed.")
        return {"error": "DataFrame is empty after selecting price columns and dropping NaNs."}

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
    # Placeholder for returning structured results
    return {
        "frequency": frequency,
        "analysis_type": analysis_type,
        "stationarity_results": stationarity_results,
        "cointegration_relations": num_coint_relations if analysis_type == "levels" else "N/A"
    }


# --- Deep Learning Forecasting (LSTM) ---

def create_dataset(X, y, sequence_length=60):
    """
    Creates sequences and corresponding labels for LSTM model.
    """
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def build_lstm_model(sequence_length, num_features, lstm_units=50, dropout_rate=0.2):
    """
    Builds a simple LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(sequence_length, num_features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1)) # Predicting a single value
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_deep_learning_forecast(
    data_file_path: str,
    target_metal_ticker: str, # e.g., "GOLD"
    feature_columns: list[str], # e.g., ["GOLD_Adj_Close", "SILVER_Adj_Close", "US_CPI_YOY"]
    sequence_length: int = 60,
    forecast_horizon: int = 30, # Not directly used in simple LSTM prediction structure, but good for context
    epochs: int = 50,
    batch_size: int = 32,
    train_test_split_ratio: float = 0.8,
    verbose: bool = True
):
    """
    Orchestrates the Deep Learning (LSTM) forecasting process.

    Args:
        data_file_path (str): Path to the CSV data file (e.g., "final_combined_daily_data.csv").
        target_metal_ticker (str): The ticker of the metal to forecast (e.g., "GOLD", "SILVER").
                                   The target column will be assumed as f"{target_metal_ticker}{ADJ_CLOSE_SUFFIX}".
        feature_columns (list[str]): List of column names to use as features.
                                      The target column should also be included in this list if it's to be used as a feature.
        sequence_length (int): Number of past time steps to use for predicting the next time step.
        forecast_horizon (int): Number of future days to forecast (conceptual, actual LSTM predicts next step).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        train_test_split_ratio (float): Ratio for splitting data into training and testing sets.
        verbose (bool): If True, prints detailed logs.

    Returns:
        dict: A dictionary containing forecast results, metrics, or error messages.
    """
    if verbose:
        print("\n--- Deep Learning Forecasting (LSTM) ---")
        print(f"Target Metal: {target_metal_ticker}, Data File: {data_file_path}")
        print(f"Features: {feature_columns}, Sequence Length: {sequence_length}, Horizon: {forecast_horizon}")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")

    target_column = f"{target_metal_ticker}{ADJ_CLOSE_SUFFIX}"
    if target_column not in feature_columns:
        if verbose:
            print(f"Warning: Target column '{target_column}' was not explicitly in feature_columns. Assuming it's the one to predict.")
        # It's typical for the target to also be a feature in time series LSTM.

    # 1. Load Data
    if not os.path.exists(data_file_path):
        if verbose:
            print(f"Error: Data file not found: {data_file_path}")
        return {"error": f"Data file not found: {data_file_path}"}
    try:
        df = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
    except Exception as e:
        if verbose:
            print(f"Error loading data from {data_file_path}: {e}")
        return {"error": f"Error loading data: {e}"}

    # Ensure all specified feature columns and target column exist
    required_cols = list(set(feature_columns + [target_column]))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        if verbose:
            print(f"Error: The following columns are missing in the data file: {', '.join(missing_cols)}")
        return {"error": f"Missing columns: {', '.join(missing_cols)}"}

    df_features = df[feature_columns].copy()
    df_target = df[target_column].copy()

    # Handle NaNs (simple forward fill and backfill for now)
    df_features.fillna(method='ffill', inplace=True)
    df_features.fillna(method='bfill', inplace=True)
    df_target.fillna(method='ffill', inplace=True)
    df_target.fillna(method='bfill', inplace=True)

    if df_features.isnull().values.any() or df_target.isnull().values.any():
        if verbose:
            print("Error: Data still contains NaNs after fill. LSTM cannot proceed.")
        return {"error": "Data contains NaNs after fill. Cannot proceed."}

    if len(df_features) < sequence_length + 10: # Arbitrary minimum length for meaningful train/test
        if verbose:
            print(f"Error: Insufficient data for sequence length {sequence_length}. Need at least {sequence_length + 10} data points.")
        return {"error": "Insufficient data for the given sequence length."}


    # 2. Scale Data
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df_features)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(df_target.values.reshape(-1,1)) # Target is single column

    # 3. Create Sequences
    X, y = create_dataset(scaled_features, scaled_target.flatten(), sequence_length)
    if X.shape[0] == 0:
        if verbose:
            print(f"Error: Not enough data to create sequences with length {sequence_length}. (X shape: {X.shape})")
        return {"error": "Not enough data to create sequences."}


    # 4. Split Data
    split_index = int(X.shape[0] * train_test_split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        if verbose:
            print(f"Error: Not enough data for training or testing after split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return {"error": "Not enough data for training or testing after split."}


    # 5. Build and Train LSTM Model
    if verbose:
        print(f"Building LSTM model... Input shape for LSTM: {X_train.shape[1:]}")
    model = build_lstm_model(sequence_length=X_train.shape[1], num_features=X_train.shape[2])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if verbose:
        print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of training data for validation during training
        callbacks=[early_stopping],
        verbose=1 if verbose else 0
    )

    # 6. Evaluate Model
    if verbose:
        print("Evaluating model...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler_target.inverse_transform(predictions_scaled)
    y_test_inversed = scaler_target.inverse_transform(y_test.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_test_inversed, predictions))
    mae = mean_absolute_error(y_test_inversed, predictions)

    if verbose:
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")

    # 7. Make Future Forecast (simple version: predict next N steps iteratively)
    # This is a simplified way and has limitations (error accumulation)
    # For a proper horizon forecast, the model or data prep might need adjustment.
    last_sequence = scaled_features[-sequence_length:]
    forecast_scaled = []
    current_sequence = last_sequence.reshape(1, sequence_length, scaled_features.shape[1])

    for _ in range(forecast_horizon):
        next_pred_scaled = model.predict(current_sequence)[0,0] # Get single value
        forecast_scaled.append(next_pred_scaled)

        # Create new input for next prediction:
        # This is tricky as we only predicted the target variable.
        # If other features are used, we'd need to predict them too or use assumptions.
        # For simplicity, let's assume a naive forecast for other features (e.g., they stay constant or follow a trend)
        # A more robust way is to only use lagged values of the target as features, or use a multi-output model.
        # Here, we'll just append the predicted target and roll the window. This is a strong simplification
        # if multiple features are used.

        # This simplified forecast loop assumes the target variable is the first feature,
        # or that the model structure is designed for single-feature input derived from multi-feature context.
        # For a multi-feature input model, this forecast loop is naive.
        # A proper multi-step forecast with multi-feature inputs is more complex.

        # Let's assume for this scaffold the features for the next step are just rolled,
        # with the new prediction replacing the target variable's part in the sequence.
        # This part is highly dependent on how features are structured and if the model is univariate or multivariate.
        # Given `feature_columns` can be multiple, this naive loop is problematic.
        # For now, this part will be a placeholder for actual multi-step forecasting logic.

        # Simplification: We'll just use the model to predict one step ahead of the test set for demonstration.
        # A proper multi-step forecast for `forecast_horizon` requires more careful handling.
        pass # Placeholder for proper multi-step forecasting

    # For now, let's return the last prediction as 'next_day_forecast'
    # And the test set predictions.
    last_actual_date = df_target.index[-1]
    forecast_dates_test = df_target.index[split_index + sequence_length:]

    # Generating future dates for the forecast_horizon (conceptual)
    future_forecast_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=df.index.freqstr or 'B')


    # This is a placeholder for what the actual 'forecast_horizon' predictions would be
    # For now, we'll just show the last prediction made on the test set.
    # A real multi-step forecast would involve iterative prediction or a model designed for multi-step output.
    future_predictions_placeholder = ["TODO: Implement multi-step forecast"] * forecast_horizon
    if len(predictions) > 0:
        future_predictions_placeholder = [predictions[-1][0]] * forecast_horizon # Naive: assume last prediction holds


    if verbose:
        print(f"Last actual value ({target_column} on {last_actual_date.strftime('%Y-%m-%d')}): {df_target.iloc[-1]:.2f}")
        if len(predictions) > 0:
             print(f"Example prediction (last from test set, for date {forecast_dates_test[-1].strftime('%Y-%m-%d')}): {predictions[-1][0]:.2f}")
        print(f"Conceptual forecast for next {forecast_horizon} periods: {future_predictions_placeholder[:3]}...")


    return {
        "message": "LSTM forecast process completed.",
        "target_metal": target_metal_ticker,
        "rmse": rmse,
        "mae": mae,
        "test_predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        "test_actual_values": y_test_inversed.tolist() if isinstance(y_test_inversed, np.ndarray) else y_test_inversed,
        "test_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates_test],
        "conceptual_future_forecast": future_predictions_placeholder,
        "conceptual_future_dates": [d.strftime('%Y-%m-%d') for d in future_forecast_dates],
        "model_summary": model.summary(print_fn=lambda x: x) # Get summary as string
    }


if __name__ == "__main__":
    # This block is for testing the analyzer module directly.
    # It assumes that data files exist in the CLEANED_DATA_DIR or MERGED_DATA_DIR (typically 'trading_system/data/').
    # You would typically run data fetching and cleansing steps first.

    print("--- Running Analyzer module directly for testing ---")

    # Ensure CLEANED_DATA_DIR exists
    if not os.path.exists(CLEANED_DATA_DIR):
        os.makedirs(CLEANED_DATA_DIR)
        print(f"Created directory: {CLEANED_DATA_DIR}")

    # --- Test Multivariate Analysis ---
    print("\n--- Testing Multivariate Analysis ---")
    dummy_cleaned_file_path_mv = os.path.join(CLEANED_DATA_DIR, "cleaned_combined_daily_prices.csv")
    if not os.path.exists(dummy_cleaned_file_path_mv):
        print(f"Dummy file for multivariate test ({dummy_cleaned_file_path_mv}) not found. Creating one.")
        dates_mv = pd.date_range(start='2020-01-01', periods=200, freq='B')
        data_mv = {
            'GOLD_Adj_Close': np.random.rand(200) * 100 + 1800 + np.arange(200) * 0.5,
            'SILVER_Adj_Close': np.random.rand(200) * 10 + 20 + np.arange(200) * 0.1,
            'PLATINUM_Adj_Close': np.random.rand(200) * 50 + 1000 + np.arange(200) * 0.2,
        }
        # Simulate some cointegration
        data_mv['SILVER_Adj_Close'] = data_mv['GOLD_Adj_Close'] * (20/1800) + np.random.rand(200)*2
        dummy_df_mv = pd.DataFrame(data_mv, index=dates_mv)
        dummy_df_mv.index.name = 'Date'
        dummy_df_mv.iloc[10:15, 0] = np.nan # Add some NaNs
        dummy_df_mv.to_csv(dummy_cleaned_file_path_mv)
        print(f"Dummy file created: {dummy_cleaned_file_path_mv}")

    print("\nRunning Multivariate Analysis (Levels, Daily):")
    mv_results_levels = run_multivariate_analysis(frequency="daily", analysis_type="levels", data_dir=CLEANED_DATA_DIR, verbose=True)
    # print(f"Multivariate Analysis (Levels) Results: {mv_results_levels}")

    print("\nRunning Multivariate Analysis (Returns, Daily):")
    mv_results_returns = run_multivariate_analysis(frequency="daily", analysis_type="returns", data_dir=CLEANED_DATA_DIR, verbose=True)
    # print(f"Multivariate Analysis (Returns) Results: {mv_results_returns}")


    # --- Test Deep Learning Forecasting ---
    print("\n\n--- Testing Deep Learning Forecasting ---")
    # This uses a different data file, typically one that includes macro data (final_combined_...)
    # For simplicity in this test script, we'll reuse CLEANED_DATA_DIR, but in practice, it might be MERGED_DATA_DIR.
    # Let's assume the DL function can run on a "cleaned" file too if features are just prices.

    # For DL, we often use a file that might be named "final_combined_daily_data.csv"
    # Let's create a dummy version of that if it doesn't exist in CLEANED_DATA_DIR for testing purposes
    dummy_final_data_path_dl = os.path.join(CLEANED_DATA_DIR, "final_combined_daily_data.csv")

    if not os.path.exists(dummy_final_data_path_dl):
        print(f"Dummy file for DL test ({dummy_final_data_path_dl}) not found. Creating one.")
        dates_dl = pd.date_range(start='2020-01-01', periods=300, freq='B') # More data for DL
        data_dl = {
            'GOLD_Adj_Close': np.random.rand(300) * 100 + 1700 + np.arange(300) * 0.3,
            'SILVER_Adj_Close': np.random.rand(300) * 10 + 18 + np.arange(300) * 0.05,
            'PLATINUM_Adj_Close': np.random.rand(300) * 50 + 900 + np.arange(300) * 0.1,
            'US_CPI_YOY': np.random.rand(300) * 2 + 1.5, # Dummy macro feature
            'US_REAL_GDP_YOY': np.random.rand(300) * 1 + 2.0 # Dummy macro feature
        }
        dummy_df_dl = pd.DataFrame(data_dl, index=dates_dl)
        dummy_df_dl.index.name = 'Date'
        dummy_df_dl.iloc[20:25, 0] = np.nan # Add some NaNs
        dummy_df_dl.iloc[30:35, 3] = np.nan
        dummy_df_dl.to_csv(dummy_final_data_path_dl)
        print(f"Dummy file created: {dummy_final_data_path_dl}")

    print("\nRunning Deep Learning Forecast (GOLD, Daily):")
    # Ensure feature_columns exist in the dummy_final_data_path_dl
    dl_feature_cols = ['GOLD_Adj_Close', 'SILVER_Adj_Close', 'US_CPI_YOY']
    dl_results = run_deep_learning_forecast(
        data_file_path=dummy_final_data_path_dl,
        target_metal_ticker="GOLD",
        feature_columns=dl_feature_cols,
        sequence_length=30, # Shorter for faster test
        forecast_horizon=7,
        epochs=2, # Minimal epochs for quick test
        batch_size=16,
        verbose=True
    )
    # print(f"Deep Learning Forecast Results: {dl_results}")
    if dl_results and "error" not in dl_results:
        print(f"DL RMSE: {dl_results.get('rmse')}")
        print(f"DL MAE: {dl_results.get('mae')}")
        # print(f"DL Model Summary: \n{dl_results.get('model_summary')}")
    elif dl_results:
        print(f"DL Error: {dl_results.get('error')}")

    print("\n--- Analyzer module testing complete ---")
