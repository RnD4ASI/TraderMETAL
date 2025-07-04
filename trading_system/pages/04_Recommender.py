import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager
import pandas as pd

# Adjust path to import from parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import recommender # Assuming recommender.py is structured to be callable
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
st.set_page_config(page_title="Trading Recommender", layout="wide")
st.title("💡 Trading Recommender")

st.markdown("""
Get trading recommendations based on a combination of technical signals
and (simplified) macroeconomic context.
This tool uses the `final_combined_<frequency>_data.csv` file.
""")

# --- Recommendation Setup ---
st.header("Recommendation Setup")

# Get list of available final_combined_*.csv files
available_data_files = []
if os.path.exists(MERGED_DATA_DIR):
    available_data_files = [f for f in os.listdir(MERGED_DATA_DIR) if f.startswith("final_combined_") and f.endswith(".csv")]

if not available_data_files:
    st.warning(f"No merged data files found in `{MERGED_DATA_DIR}`. Please run the 'Data Management' steps first to create a `final_combined_..._data.csv` file.")
    st.stop()

selected_data_file_rec = st.selectbox(
    "Select Merged Data File for Recommendation:",
    available_data_files,
    key="rec_data_file_select",
    help="Choose the dataset to base the recommendation on."
)
data_file_path_rec = os.path.join(MERGED_DATA_DIR, selected_data_file_rec)

# Load columns from selected file to populate asset choices
try:
    temp_df_cols_rec = pd.read_csv(data_file_path_rec, nrows=0).columns.tolist()
    asset_price_cols_rec = [col for col in temp_df_cols_rec if "Adj_Close" in col]
except Exception as e:
# Import html module for HTML escaping
import html

try:
    temp_df_cols_rec = pd.read_csv(data_file_path_rec, nrows=0).columns.tolist()
    asset_price_cols_rec = [col for col in temp_df_cols_rec if "Adj_Close" in col]
except Exception as e:
    st.error(f"Could not read columns from {html.escape(str(selected_data_file_rec))}: {html.escape(str(e))}")
    asset_price_cols_rec = ["GOLD_Adj_Close", "SILVER_Adj_Close", "PLATINUM_Adj_Close"] # Fallback
    asset_price_cols_rec = ["GOLD_Adj_Close", "SILVER_Adj_Close", "PLATINUM_Adj_Close"] # Fallback


rec_asset_column = st.selectbox(
    "Select Asset for Recommendation:",
    asset_price_cols_rec,
    index=0 if asset_price_cols_rec else -1,
    key="rec_asset_select",
    help="Choose the asset for which you want a recommendation."
)

st.subheader("Technical Signal Parameters (SMA Crossover)")
col_sma_rec1, col_sma_rec2 = st.columns(2)
with col_sma_rec1:
    rec_short_sma = st.number_input("Short SMA Window", min_value=5, max_value=100, value=20, step=1, key="rec_short_sma")
with col_sma_rec2:
    rec_long_sma = st.number_input("Long SMA Window", min_value=10, max_value=200, value=50, step=1, key="rec_long_sma")


if st.button("Get Recommendation", key="get_recommendation_btn"):
    if not rec_asset_column:
        st.error("Please select an asset for recommendation.")
    elif rec_short_sma >= rec_long_sma:
        st.error("Short SMA window must be less than Long SMA window.")
    else:
        st.info(f"Generating recommendation for {rec_asset_column} (SMA {rec_short_sma}/{rec_long_sma}) using data from {selected_data_file_rec}...")
        with st.spinner("Analyzing..."):
            # The recommender module's run_recommendation_workflow prints to console.
            with st_capture_stdout() as captured_output:
                try:
                    # Call the backend function
                    recommender.run_recommendation_workflow(
                        data_file_path=data_file_path_rec, # Full path
                        asset_column=rec_asset_column,
                        short_sma=rec_short_sma,
                        long_sma=rec_long_sma
                    )

                    log_output = captured_output.getvalue()

                    # Display raw log for debugging or full info
                    with st.expander("Show Full Log Output", expanded=False):
                        st.text_area("Recommendation Generation Log", log_output, height=250)

                    # Parse the recommendation from the log (this is fragile)
                    # Assumes a specific output format from run_recommendation_workflow
                    recommendation_text = "N/A"
                    reason_text = "Could not parse from log."
                    asset_name_log = rec_asset_column
                    date_log = "N/A"

                    lines = log_output.splitlines()
                    for i, line in enumerate(lines):
                        if "--- Recommendation ---" in line: # Start parsing from here
                            for sub_line in lines[i+1:]:
                                if "Asset:" in sub_line:
                                    asset_name_log = sub_line.split(":",1)[1].strip()
                                if "Date of Data:" in sub_line:
                                    date_log = sub_line.split(":",1)[1].strip()
                                if "Recommendation:" in sub_line:
                                    recommendation_text = sub_line.split(":",1)[1].strip()
                                if "Reason:" in sub_line:
                                    reason_text = sub_line.split(":",1)[1].strip()
                                    break # Found all parts we need
                            break

                    st.subheader(f"Recommendation for: {asset_name_log}")
                    st.caption(f"Based on data up to: {date_log}")

                    if recommendation_text == "BUY":
                        st.success(f"**{recommendation_text}**")
                    elif recommendation_text == "SELL":
st.caption(f"Based on data up to: {date_log}")

                    if recommendation_text == "BUY":
                        st.success("**BUY**")
                    elif recommendation_text == "SELL":
                        st.error("**SELL**")
                    elif recommendation_text == "HOLD":
                        st.warning("**HOLD**")
                    else:
                        st.info(f"**{html.escape(recommendation_text)}**")  # import html

                    st.markdown("**Reasoning:**")
                    st.markdown(f"> {html.escape(reason_text)}")  # import html

                except Exception as e:
                    st.error(f"Error during recommendation generation: {str(e)}")
                    st.text_area("Error Log (Recommendation)", captured_output.getvalue(), height=200)
                    elif recommendation_text == "HOLD":
                        st.warning(f"**{recommendation_text}**")
                    else:
                        st.info(f"**{recommendation_text}**")

                    st.markdown("**Reasoning:**")
                    st.markdown(f"> {reason_text}")

                except Exception as e:
                    st.error(f"Error during recommendation generation: {e}")
                    st.text_area("Error Log (Recommendation)", captured_output.getvalue(), height=200)

st.sidebar.info("💡 Get trading signals based on technicals and macro context.")
st.markdown("---")
st.markdown("*The recommendation logic is currently simple and will be enhanced with more sophisticated models.*")
