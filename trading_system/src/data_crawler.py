import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Define constants
DATA_DIR = "data"
TICKERS = {
    "GOLD": "GLD",
    "SILVER": "SLV",
    "PLATINUM": "PPLT"
}
MAX_YEARS_WINDOW = 10

# get_user_input_for_data_fetching() is removed. Inputs will be passed to run_data_crawl.

def fetch_historical_data(metal_name, ticker, start_date, end_date, interval):
    """
    Fetches historical price data for a given ticker using yfinance.

    Args:
        metal_name (str): Name of the metal (e.g., "GOLD").
        ticker (str): The stock ticker symbol (e.g., "GLD").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval ("1d" for daily, "1wk" for weekly).

    Returns:
        pandas.DataFrame: DataFrame with historical data, or None if fetching fails.
    """
    print(f"\nFetching {interval} data for {metal_name} ({ticker}) from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            print(f"No data found for {ticker} in the specified period.")
            return None
        print(f"Successfully fetched {len(data)} data points for {metal_name}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def save_data_to_csv(df, metal_name, frequency_name):
    """
    Saves the DataFrame to a CSV file in the data directory.

    Args:
        df (pandas.DataFrame): Data to save.
        metal_name (str): Name of the metal.
        frequency_name (str): "Daily" or "Weekly".
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filename = f"{metal_name.lower()}_{frequency_name.lower()}_prices.csv"
    filepath = os.path.join(DATA_DIR, filename)

    try:
        df.to_csv(filepath)
        print(f"Data for {metal_name} saved to {filepath}")
    except Exception as e:
        print(f"Error saving data for {metal_name} to CSV: {e}")


def run_data_crawl(start_date_str, end_date_str, frequency_selected):
    """
    Main function to run the data crawling process.
    Args:
        start_date_str (str): Start date in "YYYY-MM-DD" format.
        end_date_str (str): End date in "YYYY-MM-DD" format.
        frequency_selected (str): "daily" or "weekly".
    """
    print(f"\n--- Data Crawling Initiated for {frequency_selected} data ---")
    print(f"Period: {start_date_str} to {end_date_str}")

    try:
        start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    if end_date_dt <= start_date_dt:
        print("End date must be after start date.")
        return
    if (end_date_dt - start_date_dt).days > MAX_YEARS_WINDOW * 365.25:
        print(f"The maximum time window allowed is {MAX_YEARS_WINDOW} years.")
        return
    if end_date_dt > datetime.now():
        print("End date cannot be in the future. Corrected to today.")
        end_date_dt = datetime.now()
        end_date_str = end_date_dt.strftime("%Y-%m-%d")

    if frequency_selected not in ["daily", "weekly"]:
        print("Invalid frequency. Please choose 'daily' or 'weekly'.")
        return

    interval = "1d" if frequency_selected == "daily" else "1wk"
    freq_name_capitalized = frequency_selected.capitalize()
    all_data_fetched = True

    for metal, ticker_symbol in TICKERS.items():
        data_df = fetch_historical_data(metal, ticker_symbol, start_date_str, end_date_str, interval)
        if data_df is not None and not data_df.empty:
            save_data_to_csv(data_df, metal, freq_name_capitalized)
        else:
            print(f"Could not fetch or save data for {metal}.")
            all_data_fetched = False

    if all_data_fetched:
        print("\nAll requested metal price data has been fetched and saved successfully.")
    else:
        print("\nSome metal price data could not be fetched or saved. Please check the messages above.")

if __name__ == "__main__":
    # This part is for testing the module directly
    print("Running Data Crawler directly for testing...")
    # Example: Fetch 1 year of daily data
    start_example = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_example = datetime.now().strftime("%Y-%m-%d")
    run_data_crawl(start_example, end_example, "daily")

    # Example: Fetch 2 years of weekly data
    # start_example_wk = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%d")
    # end_example_wk = datetime.now().strftime("%Y-%m-%d")
    # run_data_crawl(start_example_wk, end_example_wk, "weekly")
