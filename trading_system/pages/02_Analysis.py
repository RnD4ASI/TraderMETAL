import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager
import pandas as pd # Required for st.date_input and other potential uses

# Adjust path to import from parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import analyzer # Assuming analyzer.py is structured to be callable

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
    on the price series of Gold, Silver, and Platinum.
    This helps in understanding the time series properties for modeling.
    """)

    analysis_freq = st.selectbox("Data Frequency for Analysis", ["daily", "weekly"], key="analysis_freq_mv")

    # The analyzer.run_multivariate_analysis() currently uses input() for choices.
    # This needs refactoring in analyzer.py similar to data_crawler and data_cleanser.
    # For now, this button will run it, but user will be prompted in terminal if streamlit is run from there.
    # Or, it might error out if input() is not available in Streamlit's execution context.
    # TODO: Refactor analyzer.py to accept parameters for 'levels'/'returns'.

    st.info("Note: The current version of `analyzer.run_multivariate_analysis` might prompt for input (levels/returns) in the terminal if not refactored for direct parameter input.")

    if st.button("Run Multivariate Analysis", key="run_mv_analysis_btn"):
        st.info(f"Starting multivariate analysis for {analysis_freq} data...")
        with st.spinner("Analyzing... please wait."):
            with st_capture_stdout() as captured_output:
                try:
                    # --- This part needs refactoring in analyzer.py ---
                    # For now, we simulate how it might be called if refactored.
                    # Let's assume a refactored function:
                    # analyzer.run_multivariate_analysis(frequency=analysis_freq, analysis_type='levels') # or 'returns'
                    # Since it's not refactored, we call it as is. User might need to interact with terminal.

                    st.error("Multivariate analysis requires refactoring of `analyzer.py` to accept parameters directly. This feature is currently not fully supported via the UI.")
                    # TODO: Refactor analyzer.py to accept parameters for 'levels'/'returns' directly.
                    # For now, this feature is not fully functional via the UI without manual terminal interaction.
                    # Remove this HACK block once analyzer.py is refactored.

                    st.text_area("Log Output (Multivariate Analysis)", captured_output.getvalue(), height=300)
                    st.success("Multivariate analysis complete!")
                except Exception as e:
                    st.error(f"Error during multivariate analysis: {e}")
                    st.text_area("Error Log (Multivariate Analysis)", captured_output.getvalue(), height=200)
                finally:
                    if 'original_input' in locals(): # Ensure it was set
                        __builtins__.input = original_input


# --- Section 2: Deep Learning Forecasting (LSTM) ---
st.header("2. Deep Learning Forecasting (LSTM)")
with st.expander("Run LSTM Forecast (Experimental)", expanded=False):
    st.warning("This feature is experimental and requires TensorFlow. The backend `analyzer.run_deep_learning_forecast` also uses `input()` and needs refactoring for a seamless UI experience.")
    st.info("For now, if you run this, you might need to interact with prompts in the terminal where Streamlit is running.")

    if st.button("Run Deep Learning Forecast (LSTM)", key="run_dl_btn", disabled=True): # Disabled until refactored
        st.info("Starting LSTM forecasting process...")
        # This would ideally call a refactored analyzer.run_deep_learning_forecast with parameters
        # For now, it's a placeholder for what needs to be built.
        st.markdown("Imagine LSTM training output and forecast plots here...")
        # with st.spinner("Training LSTM model... this can take a long time."):
        #     with st_capture_stdout() as captured_output:
        #         try:
        #             analyzer.run_deep_learning_forecast() # This needs major refactoring in analyzer.py
        #             st.text_area("Log Output (LSTM Forecast)", captured_output.getvalue(), height=300)
        #             st.success("LSTM Forecasting complete!")
        #         except Exception as e:
        #             st.error(f"Error during LSTM forecasting: {e}")
        #             st.text_area("Error Log (LSTM Forecast)", captured_output.getvalue(), height=200)

st.sidebar.info("🔬 Analysis tools to understand data properties and make forecasts.")
st.markdown("---")
st.markdown("*Further analysis methods and UI improvements will be added in future updates.*")

# Ensure pandas is imported if not already, for type hinting or direct use.
import pandas as pd
# This is a common import, ensure it's available.
# If it's not used directly in this file after refactoring, it's fine.
# It was used for pd.to_datetime in date_input values in 01_Data_Management.py.
