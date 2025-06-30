import click
from src import data_crawler
from src import data_cleanser # Import the new module

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
    click.echo("Starting data fetching process...")
    try:
        data_crawler.run_data_crawl()
        # Message is now part of data_crawler's run_data_crawl
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
    click.echo("Starting data cleansing process...")
    try:
        data_cleanser.run_data_cleansing()
        # Message is now part of data_cleanser's run_data_cleansing
    except Exception as e:
        click.echo(f"An error occurred during the data cleansing workflow: {e}", err=True)

# Future commands can be added here:
# @cli.command()
# def analyze():
#     """Runs analysis and forecasting."""
#     click.echo("Starting analysis and forecasting...")
#     # Call analyzer.run_analysis() here
#     click.echo("Analysis and forecasting completed.")

# @cli.command()
# def recommend():
#     """Generates trading recommendations."""
#     click.echo("Generating recommendations...")
#     # Call recommender.run_recommendation() here
#     click.echo("Recommendations generated.")

if __name__ == "__main__":
    cli()
