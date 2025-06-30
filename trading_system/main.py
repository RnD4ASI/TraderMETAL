import click
from src import data_crawler

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
        click.echo("\nData fetching process completed.")
        click.echo(f"Please check the '{data_crawler.DATA_DIR}/' directory for the output CSV files.")
    except Exception as e:
        click.echo(f"An error occurred during data fetching: {e}", err=True)

# Future commands can be added here:
# @cli.command()
# def clean_data():
#     """Cleans the fetched data."""
#     click.echo("Starting data cleaning process...")
#     # Call data_cleanser.run_cleaning() here
#     click.echo("Data cleaning process completed.")

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
