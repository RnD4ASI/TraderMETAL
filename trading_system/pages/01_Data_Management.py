import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager

# Adjust path to import from parent directory's src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import data_crawler
from src import macro_data_crawler
from src import data_cleanser

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
st.set_page_config(page_title="Data Management", layout="wide")
st.title("📊 Data Management")

st.markdown("""
This page allows you to fetch, cleanse, and merge financial and macroeconomic data.
Follow the steps sequentially for the best results.
""")

# --- Section 1: Fetch Metal Prices ---
st.header("1. Fetch Metal Prices (Gold, Silver, Platinum)")
with st.expander("Fetch Metal Price Data", expanded=False):
    st.markdown("Downloads historical price data for Gold (GLD), Silver (SLV), and Platinum (PPLT) ETFs.")
    col1, col2, col3 = st.columns(3)
    with col1:
        mp_start_date = st.date_input("Start Date (Metals)", value=pd.to_datetime("2018-01-01"), key="mp_start")
    with col2:
        mp_end_date = st.date_input("End Date (Metals)", value=pd.to_datetime("today"), key="mp_end")
    with col3:
        mp_freq = st.selectbox("Frequency (Metals)", ["daily", "weekly"], key="mp_freq")

    if st.button("Fetch Metal Prices", key="fetch_metals_btn"):
        if mp_start_date >= mp_end_date:
            st.error("End date must be after start date for metal prices.")
        else:
            # Mock get_user_input_for_data_fetching for data_crawler
            original_input = data_crawler.get_user_input_for_data_fetching
            def mock_input_metal():
                interval = "1d" if mp_freq == "daily" else "1wk"
                return mp_start_date.strftime("%Y-%m-%d"), mp_end_date.strftime("%Y-%m-%d"), interval, mp_freq.capitalize()

            st.info(f"Fetching {mp_freq} metal prices from {mp_start_date.strftime('%Y-%m-%d')} to {mp_end_date.strftime('%Y-%m-%d')}...")
            with st.spinner("Fetching... please wait."):
                with st_capture_stdout() as captured_output:
                    try:
                        data_crawler.run_data_crawl(
                            start_date_str=mp_start_date.strftime("%Y-%m-%d"),
                            end_date_str=mp_end_date.strftime("%Y-%m-%d"),
                            frequency_selected=mp_freq
                        )
                        st.text_area("Log Output (Metal Prices)", captured_output.getvalue(), height=200)
                        st.success("Metal price data fetching complete!")
                    except Exception as e:
                        st.error(f"Error fetching metal prices: {e}")
                        st.text_area("Error Log (Metal Prices)", captured_output.getvalue(), height=200)

# --- Section 2: Fetch Macroeconomic Data ---
st.header("2. Fetch Macroeconomic Data (FRED)")
with st.expander("Fetch Macroeconomic Data", expanded=False):
    st.markdown("""
    Downloads macroeconomic indicators (GDP, CPI, Interest Rates) for major economies from FRED.
    **Requires a FRED_API_KEY in your `.env` file.**
    """)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        macro_start_date = st.date_input("Start Date (Macro)", value=pd.to_datetime("2018-01-01"), key="macro_start")
    with col_m2:
        macro_end_date = st.date_input("End Date (Macro)", value=pd.to_datetime("today"), key="macro_end")

    if st.button("Fetch Macro Data", key="fetch_macro_btn"):
        if not macro_data_crawler.FRED_API_KEY:
            st.error("FRED_API_KEY not found. Please set it in your .env file in the project's root directory.")
        elif macro_start_date >= macro_end_date:
            st.error("End date must be after start date for macro data.")
        else:
            st.info(f"Fetching macro data from {macro_start_date} to {macro_end_date}...")
            with st.spinner("Fetching macro data... this may take a moment."):
                with st_capture_stdout() as captured_output:
                    try:
                        macro_data_crawler.run_macro_data_crawl(
                            start_date_str=macro_start_date.strftime("%Y-%m-%d"),
                            end_date_str=macro_end_date.strftime("%Y-%m-%d")
                        )
                        st.text_area("Log Output (Macro Data)", captured_output.getvalue(), height=200)
                        st.success("Macroeconomic data fetching complete!")
                    except Exception as e:
                        st.error(f"Error fetching macro data: {e}")
                        st.text_area("Error Log (Macro Data)", captured_output.getvalue(), height=200)

# --- Section 3: Clean Metal Prices ---
st.header("3. Clean Raw Metal Price Data")
with st.expander("Clean Metal Price Data", expanded=False):
    st.markdown("Cleans and combines the raw metal price CSVs into a single file per frequency.")
    clean_freq = st.selectbox("Frequency of raw metal data to clean", ["daily", "weekly"], key="clean_freq_select")

    if st.button("Clean Metal Data", key="clean_metals_btn"):
        st.info(f"Cleaning {clean_freq} metal price data...")
        with st.spinner("Cleaning data..."):
            with st_capture_stdout() as captured_output:
                try:
                    data_cleanser.run_data_cleansing(frequency_to_clean=clean_freq)
                    st.text_area("Log Output (Clean Metal Data)", captured_output.getvalue(), height=200)
                    st.success(f"Metal data cleaning for {clean_freq} frequency complete!")
                except Exception as e:
                    st.error(f"Error cleaning metal data: {e}")
                    st.text_area("Error Log (Clean Metal Data)", captured_output.getvalue(), height=200)

# --- Section 4: Merge Metal and Macro Data ---
st.header("4. Merge Cleaned Metal Prices with Macro Data")
with st.expander("Merge Data", expanded=False):
    st.markdown("""
    Merges the cleaned metal price data (e.g., `cleaned_combined_daily_prices.csv`)
    with the fetched macroeconomic data. The output will be a file like
    `final_combined_daily_data.csv`.
    """)
    merge_freq = st.selectbox("Frequency of cleaned metal data to merge", ["daily", "weekly"], key="merge_freq_select")

    if st.button("Merge Data", key="merge_data_btn"):
        st.info(f"Merging {merge_freq} metal prices with macroeconomic data...")
        with st.spinner("Merging data..."):
            with st_capture_stdout() as captured_output:
                try:
                    data_cleanser.run_merge_with_macro_data(metal_data_frequency=merge_freq)
                    st.text_area("Log Output (Merge Data)", captured_output.getvalue(), height=200)
                    st.success(f"Data merging for {merge_freq} frequency complete!")
                except Exception as e:
                    st.error(f"Error merging data: {e}")
                    st.text_area("Error Log (Merge Data)", captured_output.getvalue(), height=200)

st.sidebar.info("💡 Tip: Run these steps in order. Ensure data is fetched before cleaning, and cleaned/fetched before merging.")

# Need to import pandas for date conversions if not already
import streamlit as st
import sys
import os
from io import StringIO
from contextlib import contextmanager
import pandas as pd
# If it's not used directly in this file after refactoring, it's fine.
# It was used for pd.to_datetime in date_input values.
