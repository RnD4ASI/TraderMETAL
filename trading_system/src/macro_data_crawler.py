"""
Module for fetching macroeconomic data.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# It's good practice to get API keys from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")

DATA_DIR_MACRO = "trading_system/data/macro" # Storing in a sub-directory

# Define the series IDs for the macroeconomic indicators
# Each item is a dict with:
#   - name: User-friendly name for the final column (e.g., US_REAL_GDP_YOY_QUARTERLY)
#   - id: FRED series ID
#   - type: 'gdp', 'cpi', 'rate' (for processing logic in cleanser)
#   - country_code: 'US', 'EA', 'CN', 'UK', 'JP'
#   - fetch_format: 'level' or 'yoy' (how it's fetched or if it needs calculation)
#   - base_freq_months: For 'level' type that needs YoY, the number of periods in a year (e.g., 1 for annual, 4 for quarterly, 12 for monthly)
MACRO_INDICATORS_FRED = [
    # --- United States ---
    {
        "name": "US_REAL_GDP_YOY_QUARTERLY", "id": "GDPC1", "type": "gdp", "country_code": "US",
        "fetch_format": "level", "base_freq_periods": 4 # Quarterly data, 4 periods for YoY from level
    },
    {
        "name": "US_CPI_YOY_MONTHLY", "id": "CPIAUCSL", "type": "cpi", "country_code": "US",
        "fetch_format": "level", "base_freq_periods": 12 # Monthly data, 12 periods for YoY from level
    },
    {"name": "US_FED_FUNDS_RATE_MONTHLY", "id": "FEDFUNDS", "type": "rate", "country_code": "US", "fetch_format": "direct"},

    # --- Euro Area ---
    {
        "name": "EA_REAL_GDP_YOY_QUARTERLY", "id": "CLVMNACSCAB1GQEA19", "type": "gdp", "country_code": "EA",
        "fetch_format": "level", "base_freq_periods": 4
    },
    { # This series is already YoY: "Euro area (19 countries) - HICP - Overall index, Annual rate of change"
        "name": "EA_HICP_YOY_MONTHLY", "id": "CP0000EZ19M086NEST", "type": "cpi", "country_code": "EA",
        "fetch_format": "direct" # Assuming it's already YoY from FRED
    },
    {"name": "EA_ECB_DEPOSIT_FACILITY_RATE_DAILY", "id": "ECBDFR", "type": "rate", "country_code": "EA", "fetch_format": "direct"},

    # --- China --- (FRED data for China can be limited or use specific growth metrics)
    { # "Real Gross Domestic Product, Percent Change from Year Ago, Quarterly, Seasonally Adjusted"
        "name": "CN_REAL_GDP_YOY_QUARTERLY", "id": "RGDPNACNA666NRUG", "type": "gdp", "country_code": "CN",
        "fetch_format": "direct" # This specific series is already YoY growth
    },
    { # "Consumer Price Index: All Items for China, Index 2015=100, Monthly, Not Seasonally Adjusted"
      # This seems to be an index, would need YoY calculation.
        "name": "CN_CPI_YOY_MONTHLY", "id": "CHNCPIALLMINMEI", "type": "cpi", "country_code": "CN",
        "fetch_format": "level", "base_freq_periods": 12
        # Alternative if available: CPALTT01CNM659N (CPI Total All Items, % Change on YoY, Monthly)
    },
    # Interest rate for China from FRED is tricky. Using a proxy or needing alternative source.
    # For now, let's omit or find a suitable one.
    # Example: "Interbank Offered Rate (Overnight)": IR3TTS01CNM156N (might be too volatile)
    # Let's use a placeholder or a more stable policy rate if found.
    # For now, placeholder:
    # {"name": "CN_POLICY_RATE_MONTHLY", "id": "TODO_FIND_CN_RATE", "type": "rate", "country_code": "CN", "fetch_format": "direct"},


    # --- United Kingdom ---
    { # "Gross Domestic Product: chained volume measures: Seasonally adjusted £m"
        "name": "UK_REAL_GDP_YOY_QUARTERLY", "id": "IGSNETNGDPLQGB", "type": "gdp", "country_code": "UK",
        "fetch_format": "level", "base_freq_periods": 4
    },
    { # "Consumer Price Index (UK): All items - Index 2015=100"
        "name": "UK_CPI_YOY_MONTHLY", "id": "GBRCPIALLMINMEI", "type": "cpi", "country_code": "UK",
        "fetch_format": "level", "base_freq_periods": 12
    },
    {"name": "UK_BOE_POLICY_RATE_MONTHLY", "id": "BOEINTD", "type": "rate", "country_code": "UK", "fetch_format": "direct"},

    # --- Japan ---
    { # "Real Gross Domestic Product, Expenditure Approach, Billions of Chained 2015 Yen, Quarterly, Seasonally Adjusted"
        "name": "JP_REAL_GDP_YOY_QUARTERLY", "id": "JPNRGDPEXP", "type": "gdp", "country_code": "JP",
        "fetch_format": "level", "base_freq_periods": 4
    },
    { # "Consumer Price Index: All Items for Japan, Index 2020=100, Monthly, Seasonally Adjusted"
        "name": "JP_CPI_YOY_MONTHLY", "id": "JPNCPIALLMINMEI", "type": "cpi", "country_code": "JP",
        "fetch_format": "level", "base_freq_periods": 12
    },
    # "Interest Rates, Discount Rate for Japan, Percent, Monthly, Not Seasonally Adjusted" - might be outdated
    # Using "Monetary Base, End of Period, Monthly" - No, that's not a policy rate.
    # BOJDPBAL "Balance of Policy-Rate Balances at the Bank of Japan, Average of Month, JPY Trillions" - also not a rate
    # INTDSRJPM193N "Short-Term Policy Rates for Japan, Percent per Annum, Monthly, Not Seasonally Adjusted"
    {"name": "JP_BOJ_POLICY_RATE_MONTHLY", "id": "INTDSRJPM193N", "type": "rate", "country_code": "JP", "fetch_format": "direct"},
]

# Helper function to get the actual series ID for fetching, and original ID if needed for YoY calc
def get_fetch_details(indicator_name_target):
    for indicator in MACRO_INDICATORS_FRED:
        if indicator["name"] == indicator_name_target:
            if indicator["fetch_format"] == "level":
                return indicator["id"], indicator["id"], indicator.get("base_freq_periods")
            else: # direct
                return indicator["id"], None, None # Original ID not needed if direct
    return None, None, None


def get_fred_client():
    """Initializes and returns a Fred client."""
    if not FRED_API_KEY:
        print("Error: FRED_API_KEY not found in environment variables.")
        print("Please set it in a .env file or as an environment variable.")
        print("You can obtain a key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    return Fred(api_key=FRED_API_KEY)

def fetch_fred_series(fred_client, series_id, start_date, end_date, series_name=""):
    """
    Fetches a single series from FRED.

    Args:
        fred_client (Fred): Initialized Fred client.
        series_id (str): The FRED series ID.
        start_date (str/datetime): Start date for the data.
        end_date (str/datetime): End date for the data.
        series_name (str): Descriptive name for logging.

    Returns:
        pd.Series: Series data, or None if fetching fails.
    """
    print(f"Fetching {series_name} ({series_id}) from {start_date} to {end_date}...")
    try:
        data = fred_client.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if data.empty:
            print(f"No data found for {series_name} ({series_id}) in the specified period.")
            return None

        # Clean column name for series if needed (though get_series returns a Series)
        # data.name = series_id # The series already has a name, often the ID
        print(f"Successfully fetched {len(data)} data points for {series_name} ({series_id}).")
        return data.dropna() # Drop NaNs which can occur
    except Exception as e:
        print(f"Error fetching data for {series_name} ({series_id}): {e}")
        return None

def calculate_yoy_change(series, periods_in_year):
    """Calculates Year-over-Year percentage change for a series."""
    if series is None or len(series) < periods_in_year:
        return None
    # (value_current_period / value_same_period_last_year - 1) * 100
    yoy_series = series.pct_change(periods=periods_in_year) * 100
    return yoy_series.dropna()


def save_macro_data_to_csv(df, indicator_name, country_prefix):
    """
    Saves the DataFrame to a CSV file in the data/macro directory.
    """
    if not os.path.exists(DATA_DIR_MACRO):
        os.makedirs(DATA_DIR_MACRO)
        print(f"Created directory: {DATA_DIR_MACRO}")

    filename = f"{country_prefix.lower()}_{indicator_name.lower()}_fred.csv"
os.makedirs(DATA_DIR_MACRO)
        print(f"Created directory: {DATA_DIR_MACRO}")

    filename = secure_filename(f"{country_prefix.lower()}_{indicator_name.lower()}_fred.csv")  # import werkzeug.security
    filepath = os.path.join(DATA_DIR_MACRO, filename)

    try:

    try:
        df.to_csv(filepath)
        print(f"Macro data for {country_prefix} {indicator_name} saved to {filepath}")
    except Exception as e:
        print(f"Error saving macro data for {country_prefix} {indicator_name} to CSV: {e}")


def run_macro_data_crawl(start_date_str=None, end_date_str=None, max_years_window=10):
    """
    Main function to run the macroeconomic data crawling process using FRED.
    Prompts for dates if not provided.
    """
    print("\n--- Macroeconomic Data Fetching (FRED) ---")

    fred = get_fred_client()
    if not fred:
        return

    if start_date_str is None or end_date_str is None:
        while True:
            if start_date_str is None:
                start_date_str_input = input(f"Enter start date for macro data (YYYY-MM-DD, press Enter for {max_years_window} years ago): ")
                if not start_date_str_input:
                    start_date = datetime.now() - timedelta(days=max_years_window * 365.25)
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    print(f"Defaulting start date to: {start_date_str}")
                    break
                else:
                    start_date_str = start_date_str_input
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                break
            except ValueError:
                print("Invalid start date format. Please use YYYY-MM-DD.")
                start_date_str = None # Reset to ask again

        while True:
            if end_date_str is None:
                end_date_str_input = input("Enter end date for macro data (YYYY-MM-DD, press Enter for today): ")
                if not end_date_str_input:
                    end_date = datetime.now()
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    print(f"Defaulting end date to: {end_date_str}")
                    break
                else:
                    end_date_str = end_date_str_input
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                if end_date <= start_date:
                    print("End date must be after start date.")
                    end_date_str = None # Reset to ask again
                elif (end_date - start_date).days > max_years_window * 365.25 * 1.5: # Allow slightly larger for sparse macro
                    print(f"The time window is very large ({((end_date - start_date).days / 365.25):.1f} years). Consider a shorter window.")
                    # Not strictly enforcing max_years_window here as macro data is sparse
                    break
                elif end_date > datetime.now():
                    print("End date cannot be in the future. Setting to today.")
                    end_date = datetime.now()
                    end_date_str = end_date.strftime("%Y-%m-%d")
                break
            except ValueError:
                print("Invalid end date format. Please use YYYY-MM-DD.")
                end_date_str = None # Reset to ask again
    else:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")


    all_data_fetched_successfully = True

    for indicator_details in MACRO_INDICATORS_FRED:
        target_name = indicator_details["name"]
        series_id_to_fetch = indicator_details["id"]
        country_code = indicator_details["country_code"]
        fetch_format = indicator_details["fetch_format"]
        base_freq_periods = indicator_details.get("base_freq_periods")

        # Construct a simpler filename component from the target_name
        # e.g., US_REAL_GDP_YOY_QUARTERLY -> real_gdp_yoy_quarterly
        filename_indicator_component = "_".join(target_name.split("_")[1:]).lower()

        if series_id_to_fetch == "TODO_FIND_CN_RATE": # Skip placeholders
            print(f"Skipping placeholder indicator: {target_name}")
            continue

        raw_series_data = None
        final_series_to_save = None

        if fetch_format == "level" and base_freq_periods:
            # Fetch a bit more historical data for YoY calculation
            # Calculate how many years back needed based on periods (e.g., 12 monthly periods = 1 year)
            years_back_for_yoy = (base_freq_periods / 12.0) if indicator_details["type"] == "cpi" else (base_freq_periods / 4.0) # crude

            # Ensure we have enough data for the first YoY calculation point
            # e.g. for monthly CPI (12 periods), need 12 previous points. For quarterly GDP (4 periods), need 4 previous.
            # The actual number of days to go back depends on the true frequency of the base_freq_periods.
            # If base_freq_periods = 12 (monthly data), go back 1 year + buffer.
            # If base_freq_periods = 4 (quarterly data), go back 1 year + buffer.
            # If base_freq_periods = 1 (annual data), go back 1 year + buffer.
            # So, effectively, go back roughly 1 year + buffer for all these cases.
            buffer_days = 90 # Extra buffer
            yoy_start_date = start_date - timedelta(days=365 + buffer_days)


            raw_series_data = fetch_fred_series(fred, series_id_to_fetch,
                                                yoy_start_date.strftime("%Y-%m-%d"),
                                                end_date_str,
                                                series_name=f"{target_name} (raw for YoY calc)")
            if raw_series_data is not None:
                yoy_calculated_series = calculate_yoy_change(raw_series_data, base_freq_periods)
                if yoy_calculated_series is not None:
                    # Filter back to the original requested date range
                    final_series_to_save = yoy_calculated_series[start_date_str:end_date_str]
                else:
                    print(f"Could not calculate YoY for {target_name} from {series_id_to_fetch}.")
                    all_data_fetched_successfully = False
            else:
                all_data_fetched_successfully = False
        elif fetch_format == "direct":
            # Direct fetch for series that are already in desired format (e.g., already YoY or a rate)
            final_series_to_save = fetch_fred_series(fred, series_id_to_fetch,
                                               start_date_str, end_date_str,
                                               series_name=target_name)
            if final_series_to_save is None:
                all_data_fetched_successfully = False
        else:
            print(f"Unknown fetch_format '{fetch_format}' for {target_name}. Skipping.")
            all_data_fetched_successfully = False
            continue


        if final_series_to_save is not None and not final_series_to_save.empty:
            df_to_save = final_series_to_save.to_frame(name=target_name) # Use the target name as column name
            save_macro_data_to_csv(df_to_save, filename_indicator_component, country_code)
        elif final_series_to_save is not None and final_series_to_save.empty:
            print(f"No data returned for {target_name} ({series_id_to_fetch}) in the range after processing. Not saving file.")
        elif final_series_to_save is None and fetch_format != "level": # Error already printed by fetch_fred_series
             pass
        elif final_series_to_save is None and fetch_format == "level": # Error from YoY calc or initial fetch
             pass


    print("\nMacroeconomic data fetching process completed.")
    if not all_data_fetched_successfully:
        print("Some series may have failed or had issues during fetching/processing. Please check logs above.")


if __name__ == "__main__":
    # For direct testing:
    # 1. Make sure you have a .env file in the root of the project (where AGENTS.md is)
    #    with your FRED_API_KEY, e.g.:
    #    FRED_API_KEY=your_actual_fred_api_key
    # 2. Run this script from the `trading_system` directory:
    #    python src/macro_data_crawler.py

    # Create .env file for user if it doesn't exist (for testing convenience)
    if not os.path.exists("../../.env") and not FRED_API_KEY: # Check root .env
         if not os.path.exists(".env") and not FRED_API_KEY: # Check local .env
            try:
                with open(".env", "w") as f:
                    f.write("FRED_API_KEY=YOUR_KEY_HERE\n")
                print("Created a dummy .env file. Please replace YOUR_KEY_HERE with your actual FRED API key.")
            except IOError:
                print("Could not create a .env file. Please create one manually with your FRED_API_KEY.")

    run_macro_data_crawl(start_date_str="2018-01-01", end_date_str="2023-12-31")
    # Example: run_macro_data_crawl() # to get interactive prompts
