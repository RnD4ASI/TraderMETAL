import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager
import pandas as pd

# Adjust path to import from parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import backtester # Assuming backtester.py is structured to be callable
from src.data_cleanser import MERGED_DATA_DIR # To construct file paths

# --- Helper to Capture Print Output ---
@contextmanager
def st_capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = output_catcher = StringIO()
    try:
        yield output_catcher
    finally:
        sys.stdout = old_stdout

# --- Page Configuration ---
st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("⏱️ Strategy Backtester")

st.markdown("""
Test your trading strategies against historical data.
This tool uses the `final_combined_<frequency>_data.csv` file generated from 'Data Management'.
Currently, only an SMA (Simple Moving Average) Crossover strategy is implemented.
""")

# --- Backtesting Setup ---
st.header("Backtest Setup")

# Get list of available final_combined_*.csv files
available_data_files = []
if os.path.exists(MERGED_DATA_DIR):
    available_data_files = [f for f in os.listdir(MERGED_DATA_DIR) if f.startswith("final_combined_") and f.endswith(".csv")]

if not available_data_files:
    st.warning(f"No merged data files found in `{MERGED_DATA_DIR}`. Please run the 'Data Management' steps first to create a `final_combined_..._data.csv` file.")
    st.stop()

selected_data_file = st.selectbox(
    "Select Merged Data File for Backtest:",
    available_data_files,
    help="Choose the dataset you want to backtest on."
)
data_file_path = os.path.join(MERGED_DATA_DIR, selected_data_file)

# Infer frequency from filename
bt_frequency = "daily" # Default
if "weekly" in selected_data_file:
    bt_frequency = "weekly"
st.info(f"Inferred data frequency: **{bt_frequency.capitalize()}**")


# Load columns from selected file to populate asset choices
try:
    temp_df_cols = pd.read_csv(data_file_path, nrows=0).columns.tolist()
    asset_price_cols = [col for col in temp_df_cols if "Adj_Close" in col] # Filter for price columns
except Exception as e:
    st.error(f"Could not read columns from {selected_data_file}: {e}")
    asset_price_cols = ["GOLD_Adj_Close", "SILVER_Adj_Close", "PLATINUM_Adj_Close"] # Fallback

bt_asset_column = st.selectbox(
    "Select Asset to Trade:",
    asset_price_cols,
    index=0 if asset_price_cols else -1, # Default to first if available
    help="Choose the price column of the asset you want the strategy to trade."
)


st.subheader("SMA Crossover Strategy Parameters")
col_sma1, col_sma2 = st.columns(2)
with col_sma1:
    bt_short_sma = st.number_input("Short SMA Window", min_value=5, max_value=100, value=20, step=1)
with col_sma2:
    bt_long_sma = st.number_input("Long SMA Window", min_value=10, max_value=200, value=50, step=1)

st.subheader("Backtest Parameters")
col_cap, col_cost = st.columns(2)
with col_cap:
    bt_initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=backtester.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f")
with col_cost:
    bt_txn_cost_pct = st.number_input("Transaction Cost (as decimal, e.g., 0.001 for 0.1%)", min_value=0.0, max_value=0.1, value=backtester.DEFAULT_TRANSACTION_COST_PCT, step=0.0005, format="%.4f")


if st.button("Run Backtest", key="run_backtest_btn"):
    if not bt_asset_column:
        st.error("Please select an asset to trade.")
    elif bt_short_sma >= bt_long_sma:
        st.error("Short SMA window must be less than Long SMA window.")
    else:
        st.info(f"Starting backtest for {bt_asset_column} (SMA {bt_short_sma}/{bt_long_sma}) on {selected_data_file}...")
        with st.spinner("Backtesting... this might take a moment."):
            # The backtester module's run_backtesting_workflow is designed to print to console.
            # We capture this output.
            with st_capture_stdout() as captured_output:
                try:
                    results = backtester.run_backtesting_workflow(
                        data_file_path=data_file_path, # Full path
                        frequency=bt_frequency,
                        asset_column=bt_asset_column,
                        short_window=bt_short_sma,
                        long_window=bt_long_sma,
                        initial_capital=bt_initial_capital,
                        transaction_cost_pct=bt_txn_cost_pct
                    )
                    # run_backtesting_workflow currently prints results.
                    # If it returned a dict, we could display it more nicely.
                    # For now, we display the captured print output.

                    log_output = captured_output.getvalue()
                    st.text_area("Backtest Log & Results", log_output, height=400)

                    # Try to extract key metrics if possible (assuming a standard output format from the backend)
                    # This is a bit fragile and ideally, the backend function would return a structured dict.
                    lines = log_output.splitlines()
                    metrics_to_display = {}
                    for line in lines:
                        if ":" in line:
                            key, val = line.split(":", 1)
                            key = key.strip()
                            val = val.strip()
                            if "Return" in key or "Ratio" in key or "Drawdown" in key or "Value" in key:
                                try:
                                    # Attempt to convert percentage/dollar values to float for st.metric
                                    if "%" in val:
                                        metrics_to_display[key] = float(val.replace("%","").strip()) / 100
                                    elif "$" in val:
                                         metrics_to_display[key] = float(val.replace("$","").replace(",","").strip())
                                    else:
                                        metrics_to_display[key] = float(val)
                                except ValueError:
                                    pass # Keep as string if not convertible

                    if metrics_to_display:
                        st.subheader("Key Performance Metrics (Extracted)")
                        cols = st.columns(min(len(metrics_to_display), 3)) # Display up to 3 metrics per row
                        col_idx = 0
                        for k, v in metrics_to_display.items():
                            if isinstance(v, float) and ("Return" in k or "Drawdown" in k) :
                                cols[col_idx % 3].metric(label=k, value=f"{v*100:.2f}%")
                            elif isinstance(v, float) and "Value" in k:
                                cols[col_idx % 3].metric(label=k, value=f"${v:,.2f}")
                            elif isinstance(v, float) and "Ratio" in k:
                                cols[col_idx % 3].metric(label=k, value=f"{v:.2f}")
                            col_idx +=1

                    st.success("Backtest complete!")

                    # TODO: If backtester.run_backtesting_workflow is refactored to return portfolio_history,
                    # we can plot it here using st.line_chart(results["portfolio_history"]['total_value'])

                except Exception as e:
                    st.error(f"Error during backtest: {e}")
                    st.text_area("Error Log (Backtest)", captured_output.getvalue(), height=200)

st.sidebar.info("⏱️ Test trading strategies on historical market data.")
st.markdown("---")
st.markdown("*More strategies and visualization options will be added in future updates.*")
