import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager
import pandas as pd
import numpy as np # Added for array operations in plots
import plotly.graph_objects as go

# Adjust path to import from parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import analyzer
from src.data_cleanser import CLEANED_DATA_DIR, MERGED_DATA_DIR # For file lookups
from src.analyzer import TICKERS_LIST, ADJ_CLOSE_SUFFIX # For default selections

# --- Helper to Capture Print Output (useful for functions that print a lot) ---
@contextmanager
def st_capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = output_catcher = StringIO()
    try:
        yield output_catcher
    finally:
        sys.stdout = old_stdout

# --- Page Configuration ---
st.set_page_config(page_title="Financial Analysis", layout="wide")
st.title("🔬 Financial Analysis")

st.markdown("""
This section allows you to perform various financial analyses on your data.
Ensure that you have fetched and prepared the necessary data using the 'Data Management' page.
The `final_combined_<frequency>_data.csv` file is typically used here.
""")

# --- Section 1: Multivariate Analysis (ADF, Johansen) ---
st.header("1. Multivariate Analysis (Stationarity & Cointegration)")
with st.expander("Run Multivariate Analysis", expanded=True):
    st.markdown("""
    Performs stationarity tests (Augmented Dickey-Fuller) and cointegration tests (Johansen)
    on the price series of Gold, Silver, and Platinum from the `cleaned_combined_<frequency>_prices.csv` file.
    This helps in understanding the time series properties for modeling.
    """)

    col_mv1, col_mv2 = st.columns(2)
    with col_mv1:
        mv_analysis_freq = st.selectbox("Data Frequency for Analysis", ["daily", "weekly"], key="mv_analysis_freq")
    with col_mv2:
        mv_analysis_type = st.selectbox("Analyze Price Levels or Returns?", ["levels", "returns"], key="mv_analysis_type")

    if st.button("Run Multivariate Analysis", key="run_mv_analysis_btn"):
        st.info(f"Starting multivariate analysis for {mv_analysis_freq} data on {mv_analysis_type}...")
        with st.spinner("Analyzing... please wait."):
            # Capture print output from the backend function to display as logs
            with st_capture_stdout() as captured_mv_output:
                try:
                    results = analyzer.run_multivariate_analysis(
                        frequency=mv_analysis_freq,
                        analysis_type=mv_analysis_type,
                        data_dir=CLEANED_DATA_DIR, # uses the default from data_cleanser
                        verbose=True # Ensure backend prints details
                    )

                    if results and "error" in results:
                        st.error(f"Analysis Error: {results['error']}")
                    elif results:
                        st.success("Multivariate analysis complete!")
                        st.subheader("Analysis Results:")
                        st.write(f"**Frequency Analyzed:** {results.get('frequency', 'N/A').capitalize()}")
                        st.write(f"**Type of Analysis:** {results.get('analysis_type', 'N/A').capitalize()}")

                        st.markdown("**Stationarity Results (ADF Test):**")
                        stationarity = results.get('stationarity_results', {})
                        if stationarity:
                            for series, is_stationary in stationarity.items():
                                st.write(f"- `{series}`: {'Stationary' if is_stationary else 'Non-Stationary'}")
                        else:
                            st.write("No stationarity results available.")

                        if results.get('analysis_type') == 'levels':
                            st.markdown("**Cointegration Test (Johansen):**")
                            coint_relations = results.get('cointegration_relations', 'N/A')
                            if coint_relations == 'N/A' or coint_relations == -1:
                                st.write("Cointegration test was not performed or encountered an issue (e.g., insufficient data).")
                            else:
                                st.write(f"Number of cointegrating relationships found: **{coint_relations}**")
                                if coint_relations > 0:
                                    st.markdown("> *Recommendation: A Vector Error Correction Model (VECM) may be appropriate.*")
                                else:
                                    st.markdown("> *Recommendation: If series are non-stationary, consider modeling their first differences (returns) using a VAR model. If series are already stationary in levels, a VAR model on levels might be appropriate.*")
                    else:
                        st.warning("Analysis did not return any results. Check logs.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during multivariate analysis: {e}")

            log_output_mv = captured_mv_output.getvalue()
            if log_output_mv:
                with st.expander("Show Full Log Output (Multivariate Analysis)", expanded=False):
                    st.text_area("", log_output_mv, height=300)


# --- Section 2: Deep Learning Forecasting (LSTM) ---
st.header("2. Deep Learning Forecasting (LSTM)")
# --- Section 2: Deep Learning Forecasting (LSTM) ---
st.header("2. Deep Learning Forecasting (LSTM)")
with st.expander("Run LSTM Forecast", expanded=False):
    st.markdown("""
    Train an LSTM (Long Short-Term Memory) neural network for forecasting a target metal's price.
    This uses a `final_combined_<frequency>_data.csv` file, which should include market prices and
    any macroeconomic features you wish to use.
    **Note:** This feature requires TensorFlow and can be computationally intensive.
    """)

    def load_data_files():
        available_dl_data_files = []
        if os.path.exists(MERGED_DATA_DIR):
            available_dl_data_files = sorted([f for f in os.listdir(MERGED_DATA_DIR) if f.startswith("final_combined_") and f.endswith(".csv")], reverse=True)
        
        if not available_dl_data_files:
            st.warning(f"No merged data files (e.g., `final_combined_daily_data.csv`) found in `{MERGED_DATA_DIR}`. Please run 'Data Management' steps to create one for DL forecasting.")
        
        return available_dl_data_files

    def select_data_file(available_files):
        return st.selectbox(
            "Select Merged Data File for LSTM:",
            available_files,
            key="dl_data_file",
            help="Choose the dataset (typically including macro features) for training the LSTM."
        )

    def load_columns(selected_file):
        all_columns = []
        if selected_file:
            try:
                dl_df_path = os.path.join(MERGED_DATA_DIR, selected_file)
                temp_df = pd.read_csv(dl_df_path, nrows=1)
                all_columns = temp_df.columns.tolist()
                if 'Date' in all_columns:
                    all_columns.remove('Date')
            except Exception as e:
                st.error(f"Could not read columns from {selected_file}: {e}")
                all_columns = [f"{t}{ADJ_CLOSE_SUFFIX}" for t in TICKERS_LIST]
        return all_columns

    def select_model_parameters():
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            dl_target_metal = st.selectbox(
                "Target Metal to Forecast:",
                TICKERS_LIST,
                key="dl_target_metal",
                help="Select the metal whose price you want to forecast."
            )
            dl_epochs = st.number_input("Training Epochs:", min_value=1, max_value=200, value=20, step=1, key="dl_epochs")
            dl_batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=32, step=8, key="dl_batch_size")

        with col_dl2:
            dl_sequence_length = st.number_input("Sequence Length (Lookback Window):", min_value=10, max_value=180, value=60, step=5, key="dl_seq_len")
            dl_forecast_horizon = st.number_input("Forecast Horizon (Days):", min_value=1, max_value=90, value=30, step=1, key="dl_horizon")

        return dl_target_metal, dl_epochs, dl_batch_size, dl_sequence_length, dl_forecast_horizon

    def select_features(all_columns, dl_target_metal):
        st.markdown("**Feature Selection for LSTM:**")
        dl_target_col_name = f"{dl_target_metal}{ADJ_CLOSE_SUFFIX}"
        st.info(f"The target column for forecasting will be `{dl_target_col_name}`. It should also be selected as a feature if you want to use its lagged values.")

        default_features = [col for col in all_columns if ADJ_CLOSE_SUFFIX in col]
        if not default_features and all_columns:
            default_features = all_columns[:min(3, len(all_columns))]

        return st.multiselect(
            "Select Feature Columns for LSTM model:",
            options=all_columns,
            default=default_features,
            key="dl_features",
            help="Choose columns to use as input features. Include the target column if using its history."
        )

    def run_lstm_forecast(data_file_full_path, dl_target_metal, dl_feature_columns, dl_sequence_length, dl_forecast_horizon, dl_epochs, dl_batch_size):
        st.info(f"Starting LSTM forecast for {dl_target_metal} using {os.path.basename(data_file_full_path)}...")
        st.info(f"Features: {dl_feature_columns}")
        st.info(f"Params: Seq Len={dl_sequence_length}, Horizon={dl_forecast_horizon}, Epochs={dl_epochs}, Batch={dl_batch_size}")

        with st.spinner("Training LSTM model... this can take a significant amount of time."):
            with st_capture_stdout() as captured_dl_output:
                try:
                    import tensorflow

                    dl_results = analyzer.run_deep_learning_forecast(
                        data_file_path=data_file_full_path,
                        target_metal_ticker=dl_target_metal,
                        feature_columns=dl_feature_columns,
                        sequence_length=dl_sequence_length,
                        forecast_horizon=dl_forecast_horizon,
                        epochs=dl_epochs,
                        batch_size=dl_batch_size,
                        verbose=True
                    )

                    display_lstm_results(dl_results, dl_target_metal)

                except ImportError:
                    st.error("TensorFlow library not found. Please install it to use this feature: `pip install tensorflow`")
                except Exception as e:
                    st.error(f"An unexpected error occurred during LSTM forecasting: {str(e)}")

            log_output_dl = captured_dl_output.getvalue()
            if log_output_dl:
                with st.expander("Show Full Log Output (LSTM Forecast)", expanded=False):
                    st.text_area("", log_output_dl, height=400)

    def display_lstm_results(dl_results, dl_target_metal):
        if dl_results and "error" in dl_results:
            st.error(f"LSTM Forecast Error: {dl_results['error']}")
        elif dl_results:
            st.success("LSTM Forecasting complete!")
            st.subheader(f"Forecast Results for {dl_results.get('target_metal', 'N/A')}")

            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("Test RMSE", f"{dl_results.get('rmse', 0):.4f}")
            col_metric2.metric("Test MAE", f"{dl_results.get('mae', 0):.4f}")

            plot_test_predictions(dl_results)
            display_future_forecast(dl_results, dl_target_metal)

            with st.expander("View Model Summary"):
                model_summary_str = dl_results.get("model_summary", "Not available.")
                st.text(model_summary_str)
        else:
            st.warning("LSTM forecast did not return any results. Check logs.")

    def plot_test_predictions(dl_results):
        test_actual = dl_results.get('test_actual_values')
        test_preds = dl_results.get('test_predictions')
        test_dates_str = dl_results.get('test_dates')

        if test_actual and test_preds and test_dates_str:
            try:
                test_dates = [pd.to_datetime(d) for d in test_dates_str]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=np.array(test_actual).flatten(), mode='lines', name='Actual Test Values'))
                fig.add_trace(go.Scatter(x=test_dates, y=np.array(test_preds).flatten(), mode='lines', name='Predicted Test Values'))
                fig.update_layout(title='LSTM: Actual vs. Predicted Values (Test Set)',
                                  xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_err:
                st.warning(f"Could not plot test predictions: {plot_err}")

    def display_future_forecast(dl_results, dl_target_metal):
        future_forecast = dl_results.get('conceptual_future_forecast')
        future_dates_str = dl_results.get('conceptual_future_dates')
        if future_forecast and future_dates_str:
            try:
                future_dates = [pd.to_datetime(d) for d in future_dates_str]
                st.subheader(f"Conceptual Forecast for next {len(future_forecast)} periods")

                df_future = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
                st.dataframe(df_future.set_index('Date'))

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines+markers', name='Conceptual Future Forecast'))
                fig_future.update_layout(title=f'LSTM: Conceptual Future Forecast for {dl_target_metal}',
                                         xaxis_title='Date', yaxis_title='Forecasted Price')
                st.plotly_chart(fig_future, use_container_width=True)
            except Exception as plot_err:
                st.warning(f"Could not display future forecast data: {plot_err}")
        else:
            st.info("Conceptual future forecast data not available or not generated by the backend.")

    available_files = load_data_files()
    selected_file = select_data_file(available_files)
    all_columns = load_columns(selected_file)
    dl_target_metal, dl_epochs, dl_batch_size, dl_sequence_length, dl_forecast_horizon = select_model_parameters()
    dl_feature_columns = select_features(all_columns, dl_target_metal)

    if st.button("Run Deep Learning Forecast (LSTM)", key="run_dl_btn"):
        if not selected_file:
            st.error("Please select a data file for LSTM forecasting.")
        elif not dl_target_metal:
            st.error("Please select a target metal to forecast.")
        elif not dl_feature_columns:
            st.error("Please select at least one feature column for the LSTM.")
        elif f"{dl_target_metal}{ADJ_CLOSE_SUFFIX}" not in dl_feature_columns:
            st.warning(f"The target column '{dl_target_metal}{ADJ_CLOSE_SUFFIX}' is not in your selected features. It's often beneficial to include it. Proceeding anyway.")
        else:
            data_file_full_path = os.path.join(MERGED_DATA_DIR, selected_file)
            run_lstm_forecast(data_file_full_path, dl_target_metal, dl_feature_columns, dl_sequence_length, dl_forecast_horizon, dl_epochs, dl_batch_size)

st.sidebar.info("🔬 Use statistical and machine learning models to analyze data properties and generate forecasts.")
st.markdown("---")
st.markdown("*Remember that financial forecasts are inherently uncertain. These tools are for analysis and exploration.*")
    st.markdown("""
    Train an LSTM (Long Short-Term Memory) neural network for forecasting a target metal's price.
    This uses a `final_combined_<frequency>_data.csv` file, which should include market prices and
    any macroeconomic features you wish to use.
    **Note:** This feature requires TensorFlow and can be computationally intensive.
    """)

    # Get list of available final_combined_*.csv files from MERGED_DATA_DIR
    available_dl_data_files = []
    if os.path.exists(MERGED_DATA_DIR):
        available_dl_data_files = sorted([f for f in os.listdir(MERGED_DATA_DIR) if f.startswith("final_combined_") and f.endswith(".csv")], reverse=True)

    if not available_dl_data_files:
        st.warning(f"No merged data files (e.g., `final_combined_daily_data.csv`) found in `{MERGED_DATA_DIR}`. Please run 'Data Management' steps to create one for DL forecasting.")
        # st.stop() # Option to halt if no files, or let it proceed and error on button click

    selected_dl_data_file = st.selectbox(
        "Select Merged Data File for LSTM:",
        available_dl_data_files,
        key="dl_data_file",
        help="Choose the dataset (typically including macro features) for training the LSTM."
    )

    # Attempt to load columns from the selected file for feature selection
    all_columns = []
    if selected_dl_data_file:
        try:
            dl_df_path = os.path.join(MERGED_DATA_DIR, selected_dl_data_file)
            temp_df = pd.read_csv(dl_df_path, nrows=1) # Read only header
            all_columns = temp_df.columns.tolist()
            if 'Date' in all_columns: # Assuming 'Date' is index and not a feature
                all_columns.remove('Date')
        except Exception as e:
if 'Date' in all_columns: # Assuming 'Date' is index and not a feature
                all_columns.remove('Date')
        except Exception as e:
            # import html
            st.error(f"Could not read columns from {selected_dl_data_file}: {html.escape(str(e))}") # Sanitize error message
            all_columns = [f"{t}{ADJ_CLOSE_SUFFIX}" for t in TICKERS_LIST] # Fallback
            all_columns = [f"{t}{ADJ_CLOSE_SUFFIX}" for t in TICKERS_LIST] # Fallback


    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        dl_target_metal = st.selectbox(
            "Target Metal to Forecast:",
            TICKERS_LIST, # ["GOLD", "SILVER", "PLATINUM"]
            key="dl_target_metal",
            help="Select the metal whose price you want to forecast."
        )
        dl_target_col_name = f"{dl_target_metal}{ADJ_CLOSE_SUFFIX}"

        dl_epochs = st.number_input("Training Epochs:", min_value=1, max_value=200, value=20, step=1, key="dl_epochs") # Reduced default for speed
        dl_batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=32, step=8, key="dl_batch_size")

    with col_dl2:
        dl_sequence_length = st.number_input("Sequence Length (Lookback Window):", min_value=10, max_value=180, value=60, step=5, key="dl_seq_len")
        dl_forecast_horizon = st.number_input("Forecast Horizon (Days):", min_value=1, max_value=90, value=30, step=1, key="dl_horizon")
        # Train/test split ratio - could be advanced option
        # dl_train_test_split = st.slider("Train/Test Split Ratio:", 0.5, 0.95, 0.8, 0.05, key="dl_split")


    st.markdown("**Feature Selection for LSTM:**")
    st.info(f"The target column for forecasting will be `{dl_target_col_name}`. It should also be selected as a feature if you want to use its lagged values.")

    default_features = [col for col in all_columns if ADJ_CLOSE_SUFFIX in col] # Default to all Adj Close prices
    if not default_features and all_columns: # If no adj_close, pick first few
        default_features = all_columns[:min(3, len(all_columns))]

    dl_feature_columns = st.multiselect(
        "Select Feature Columns for LSTM model:",
        options=all_columns,
        default=default_features,
        key="dl_features",
        help="Choose columns to use as input features. Include the target column if using its history."
    )


    if st.button("Run Deep Learning Forecast (LSTM)", key="run_dl_btn"):
        if not selected_dl_data_file:
            st.error("Please select a data file for LSTM forecasting.")
        elif not dl_target_metal:
            st.error("Please select a target metal to forecast.")
        elif not dl_feature_columns:
            st.error("Please select at least one feature column for the LSTM.")
        elif dl_target_col_name not in dl_feature_columns :
             st.warning(f"The target column '{dl_target_col_name}' is not in your selected features. It's often beneficial to include it. Proceeding anyway.")
        else:
            data_file_full_path = os.path.join(MERGED_DATA_DIR, selected_dl_data_file)
            st.info(f"Starting LSTM forecast for {dl_target_metal} using {selected_dl_data_file}...")
            st.info(f"Features: {dl_feature_columns}")
            st.info(f"Params: Seq Len={dl_sequence_length}, Horizon={dl_forecast_horizon}, Epochs={dl_epochs}, Batch={dl_batch_size}")

            with st.spinner("Training LSTM model... this can take a significant amount of time."):
                # Capture print output from the backend function
                with st_capture_stdout() as captured_dl_output:
                    try:
                        # Ensure tensorflow is available (as it's an optional import in analyzer)
                        import tensorflow

                        dl_results = analyzer.run_deep_learning_forecast(
                            data_file_path=data_file_full_path,
                            target_metal_ticker=dl_target_metal,
                            feature_columns=dl_feature_columns,
                            sequence_length=dl_sequence_length,
                            forecast_horizon=dl_forecast_horizon,
                            epochs=dl_epochs,
                            batch_size=dl_batch_size,
                            # train_test_split_ratio=dl_train_test_split, # If slider added
                            verbose=True # Backend will print details
                        )

                        if dl_results and "error" in dl_results:
                            st.error(f"LSTM Forecast Error: {dl_results['error']}")
                        elif dl_results:
                            st.success("LSTM Forecasting complete!")
                            st.subheader(f"Forecast Results for {dl_results.get('target_metal', 'N/A')}")

                            col_metric1, col_metric2 = st.columns(2)
                            col_metric1.metric("Test RMSE", f"{dl_results.get('rmse', 0):.4f}")
                            col_metric2.metric("Test MAE", f"{dl_results.get('mae', 0):.4f}")

                            # Plot actual vs predicted for test set
                            test_actual = dl_results.get('test_actual_values')
                            test_preds = dl_results.get('test_predictions')
                            test_dates_str = dl_results.get('test_dates')

                            if test_actual and test_preds and test_dates_str:
                                try:
                                    test_dates = [pd.to_datetime(d) for d in test_dates_str]

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=test_dates, y=np.array(test_actual).flatten(), mode='lines', name='Actual Test Values'))
                                    fig.add_trace(go.Scatter(x=test_dates, y=np.array(test_preds).flatten(), mode='lines', name='Predicted Test Values'))
                                    fig.update_layout(title='LSTM: Actual vs. Predicted Values (Test Set)',
                                                      xaxis_title='Date', yaxis_title='Price')
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as plot_err:
                                    st.warning(f"Could not plot test predictions: {plot_err}")


                            # Display conceptual future forecast (if available and meaningful)
                            future_forecast = dl_results.get('conceptual_future_forecast')
                            future_dates_str = dl_results.get('conceptual_future_dates')
                            if future_forecast and future_dates_str:
                                try:
                                    future_dates = [pd.to_datetime(d) for d in future_dates_str]
                                    st.subheader(f"Conceptual Forecast for next {len(future_forecast)} periods")

                                    # Create a DataFrame for easier display
                                    df_future = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})
                                    st.dataframe(df_future.set_index('Date'))

                                    # Plot future forecast
                                    fig_future = go.Figure()
                                    fig_future.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines+markers', name='Conceptual Future Forecast'))
                                    fig_future.update_layout(title=f'LSTM: Conceptual Future Forecast for {dl_target_metal}',
                                                           xaxis_title='Date', yaxis_title='Forecasted Price')
                                    st.plotly_chart(fig_future, use_container_width=True)
                                except Exception as plot_err:
                                     st.warning(f"Could not display future forecast data: {plot_err}")
                            else:
                                st.info("Conceptual future forecast data not available or not generated by the backend.")

                            with st.expander("View Model Summary"):
                                model_summary_str = dl_results.get("model_summary", "Not available.")
                                st.text(model_summary_str)

                        else:
                            st.warning("LSTM forecast did not return any results. Check logs.")

                    except ImportError:
                        st.error("TensorFlow library not found. Please install it to use this feature: `pip install tensorflow`")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during LSTM forecasting: {str(e)}")

                log_output_dl = captured_dl_output.getvalue()
                if log_output_dl:
                    with st.expander("Show Full Log Output (LSTM Forecast)", expanded=False):
                        st.text_area("", log_output_dl, height=400) # Larger height for DL logs

st.sidebar.info("🔬 Use statistical and machine learning models to analyze data properties and generate forecasts.")
st.markdown("---")
st.markdown("*Remember that financial forecasts are inherently uncertain. These tools are for analysis and exploration.*")
