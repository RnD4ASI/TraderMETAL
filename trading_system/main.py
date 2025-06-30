import click
from src import data_crawler
from src import data_cleanser
from src import analyzer # Import the analyzer module

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
