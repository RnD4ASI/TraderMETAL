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

def get_user_input_for_data_fetching():
    """
    Prompts the user for start date, end date, and data frequency.
    Validates the input.

    Returns:
        tuple: (start_date_str, end_date_str, interval_str) or None if input is invalid.
    """
    print("\n--- Data Fetching Configuration ---")

    while True:
        start_date_str = input("Enter start date (YYYY-MM-DD): ")
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            break
        except ValueError:
            print("Invalid start date format. Please use YYYY-MM-DD.")

    while True:
        end_date_str = input("Enter end date (YYYY-MM-DD): ")
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            if end_date <= start_date:
                print("End date must be after start date.")
            elif (end_date - start_date).days > MAX_YEARS_WINDOW * 365.25: # Approximate days in 10 years
                print(f"The maximum time window allowed is {MAX_YEARS_WINDOW} years.")
                print(f"Requested window: {(end_date - start_date).days / 365.25:.2f} years.")
            elif end_date > datetime.now():
                print("End date cannot be in the future. Setting to today.")
                end_date = datetime.now()
                end_date_str = end_date.strftime("%Y-%m-%d")
                break
            else:
                break
        except ValueError:
            print("Invalid end date format. Please use YYYY-MM-DD.")

    while True:
        frequency = input("Enter data frequency (daily/weekly): ").lower()
        if frequency in ["daily", "weekly"]:
            interval = "1d" if frequency == "daily" else "1wk"
            break
        else:
            print("Invalid frequency. Please enter 'daily' or 'weekly'.")

    return start_date_str, end_date_str, interval, frequency.capitalize()


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


def run_data_crawl():
    """
    Main function to run the data crawling process.
    """
    inputs = get_user_input_for_data_fetching()
    if not inputs:
        return

    start_date, end_date, interval, freq_name = inputs
    all_data_fetched = True

    for metal, ticker_symbol in TICKERS.items():
        data_df = fetch_historical_data(metal, ticker_symbol, start_date, end_date, interval)
        if data_df is not None and not data_df.empty:
            save_data_to_csv(data_df, metal, freq_name)
        else:
            print(f"Could not fetch or save data for {metal}.")
            all_data_fetched = False

    if all_data_fetched:
        print("\nAll requested data has been fetched and saved successfully.")
    else:
        print("\nSome data could not be fetched or saved. Please check the messages above.")

if __name__ == "__main__":
    # This part is for testing the module directly
    # In the main application, `run_data_crawl` would be imported and called.
    print("Running Data Crawler directly for testing...")
    run_data_crawl()
