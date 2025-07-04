import click
from src import data_crawler
from src import data_cleanser
from src import analyzer
from src import macro_data_crawler
from src import backtester
from src import recommender

@click.group()
def cli():
    """
    Trading Analysis and Recommendation System
    """
    pass

@cli.command()
def fetch_data():
    """
    Fetches historical price data for Gold, Silver, and Platinum.
    You will be prompted to enter the start date, end date, and data frequency (daily/weekly).
    The maximum time window is 10 years.
    """
    # click.echo("Starting data fetching process...") # Message now in data_crawler
    try:
        data_crawler.run_data_crawl()
    except Exception as e:
        click.echo(f"An error occurred during the data fetching workflow: {e}", err=True)

@cli.command()
def clean_data():
    """
    Cleans the fetched raw data files for Gold, Silver, and Platinum.
    This involves handling missing values, aligning dates, and merging
    the data into a single CSV file.
    You will be prompted to specify the frequency (daily/weekly) of the
    raw data you wish to cleanse.
    """
    # click.echo("Starting data cleansing process...") # Message now in data_cleanser
    try:
        data_cleanser.run_data_cleansing()
    except Exception as e:
        click.echo(f"An error occurred during the data cleansing workflow: {e}", err=True)

@cli.command()
def analyze_mv():
    """
    Performs multivariate analysis on the cleaned data.
    This includes stationarity tests (ADF) on price levels or returns,
    and a Johansen cointegration test on price levels to identify
    long-run relationships between Gold, Silver, and Platinum.
    """
    # click.echo("Starting multivariate analysis...") # Message now in analyzer
    try:
        analyzer.run_multivariate_analysis()
    except Exception as e:
        click.echo(f"An error occurred during the multivariate analysis workflow: {e}", err=True)

@cli.command()
def analyze_dl():
    """
    Performs forecasting using a Deep Learning (LSTM) model.
    You will be prompted for various parameters like target metal,
    features, sequence length, forecast horizon, and training settings.
    This command trains an LSTM model on historical data and provides
    a forecast. Note: Requires TensorFlow to be installed.
    """
    # click.echo("Starting Deep Learning forecasting process...") # Message now in analyzer
    try:
        # Ensure TensorFlow is available before running, or let analyzer handle it
        try:
            import tensorflow
        except ImportError:
            click.echo("TensorFlow not found. Please install TensorFlow to use this feature: pip install tensorflow", err=True)
            return
        analyzer.run_deep_learning_forecast()
    except Exception as e:
        click.echo(f"An error occurred during the Deep Learning analysis workflow: {e}", err=True)

@cli.command()
@click.option('--start-date', default=None, help="Start date for macro data (YYYY-MM-DD). Optional.")
@click.option('--end-date', default=None, help="End date for macro data (YYYY-MM-DD). Optional.")
def fetch_macro_data(start_date, end_date):
    """
    Fetches macroeconomic data (e.g., GDP, CPI, Interest Rates) using FRED.
    If start/end dates are not provided, you will be prompted or defaults will be used.
    Requires FRED_API_KEY environment variable.
    """
    try:
        macro_data_crawler.run_macro_data_crawl(start_date_str=start_date, end_date_str=end_date)
    except Exception as e:
        click.echo(f"An error occurred during the macroeconomic data fetching workflow: {e}", err=True)

@cli.command()
@click.option('--frequency', type=click.Choice(['daily', 'weekly'], case_sensitive=False), required=True, help="Frequency of the metal price data to merge with.")
def merge_macro_data(frequency):
    """
    Merges cleaned metal price data with fetched macroeconomic data.
    The metal price data for the specified frequency must already exist
    (i.e., 'clean-data' command should have been run).
    Macroeconomic data should also have been fetched using 'fetch-macro-data'.
    The output is saved as 'final_combined_{frequency}_data.csv'.
    """
    try:
        data_cleanser.run_merge_with_macro_data(metal_data_frequency=frequency)
    except Exception as e:
        click.echo(f"An error occurred during the data merging workflow: {e}", err=True)

@cli.command()
@click.option('--data-file', required=True, help="Path to the (merged) data CSV file (e.g., final_combined_daily_data.csv).")
@click.option('--frequency', type=click.Choice(['daily', 'weekly'], case_sensitive=False), required=True, help="Frequency of the data used ('daily' or 'weekly').")
@click.option('--asset', required=True, help="Asset column name to trade (e.g., GOLD_Adj_Close).")
@click.option('--short-sma', type=int, default=20, help="Short window for SMA Crossover strategy.")
@click.option('--long-sma', type=int, default=50, help="Long window for SMA Crossover strategy.")
@click.option('--initial-capital', type=float, default=backtester.DEFAULT_INITIAL_CAPITAL, help="Initial capital for backtest.")
@click.option('--txn-cost-pct', type=float, default=backtester.DEFAULT_TRANSACTION_COST_PCT, help="Transaction cost percentage per trade.")
def backtest(data_file, frequency, asset, short_sma, long_sma, initial_capital, txn_cost_pct):
    """
    Runs a backtest for an SMA Crossover strategy on the specified asset.
    Requires the merged data file from 'merge-macro-data'.
    """
    data_file_path = f"trading_system/data/{data_file}" # Assuming files are in data dir
    try:
        backtester.run_backtesting_workflow(
            data_file_path=data_file_path,
            frequency=frequency,
            asset_column=asset,
            short_window=short_sma,
            long_window=long_sma,
            initial_capital=initial_capital,
            transaction_cost_pct=txn_cost_pct
        )
    except Exception as e:
        click.echo(f"An error occurred during the backtesting workflow: {e}", err=True)

@cli.command()
@click.option('--data-file', required=True, help="Path to the (merged) data CSV file (e.g., final_combined_daily_data.csv).")
@click.option('--asset', required=True, help="Asset column name for recommendation (e.g., GOLD_Adj_Close).")
@click.option('--short-sma', type=int, default=20, help="Short window for SMA technical signal.")
@click.option('--long-sma', type=int, default=50, help="Long window for SMA technical signal.")
def recommend(data_file, asset, short_sma, long_sma):
    """
    Generates a trading recommendation for the specified asset.
    Uses SMA Crossover for technical signal and a placeholder for macro context.
    Requires the merged data file from 'merge-macro-data'.
    """
    data_file_path = os.path.join(data_cleanser.MERGED_DATA_DIR, data_file)
    try:
        recommender.run_recommendation_workflow(
            data_file_path=data_file_path,
            asset_column=asset,
            short_sma=short_sma,
            long_sma=long_sma
        )
    except Exception as e:
        click.echo(f"An error occurred during the recommendation workflow: {e}", err=True)


# Future commands for univariate analysis, backtesting etc. can be added here:
# @cli.command()
# def analyze_uni():
#     """Runs univariate analysis and forecasting."""
#     click.echo("Starting univariate analysis and forecasting...")
#     # Call analyzer.run_univariate_analysis() here
#     click.echo("Univariate analysis and forecasting completed.")

# @cli.command()
# def recommend():
#     """Generates trading recommendations."""
#     click.echo("Generating recommendations...")
#     # Call recommender.run_recommendation() here
#     click.echo("Recommendations generated.")

if __name__ == "__main__":
    cli()
