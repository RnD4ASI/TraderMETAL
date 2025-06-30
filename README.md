# Trading Analysis and Recommendation System

This project aims to provide a trading analysis and recommendation system for Gold, Silver, and Platinum.

## Features (Planned)

*   Data Crawling: Extracts historical prices for Gold, Silver, and Platinum.
*   Data Cleansing: Cleans and prepares financial time series data.
*   Analysis and Forecasting: Performs various analyses and generates price forecasts.
*   Backtesting: Tests trading strategies against historical data.
*   Recommendation: Provides BUY/HOLD/SELL recommendations with confidence levels and expected returns.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd trading-analysis-system
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you encounter issues with `pip install` due to environment path problems, ensure your shell's current working directory is correctly recognized or try specifying absolute paths if necessary.)*

## Usage

The application is run from the command line from within the `trading_system` directory.

**1. Fetch Raw Data:**

   Use the `fetch-data` command to download historical price data for Gold, Silver, and Platinum.
   You will be prompted to enter a start date, end date (max 10-year window), and frequency (daily/weekly).

   ```bash
   cd trading_system
   python main.py fetch-data
   ```
   Raw data files (e.g., `gold_daily_prices.csv`) will be saved in the `trading_system/data/` directory.

**2. Clean and Combine Data:**

   After fetching the raw data, use the `clean-data` command to process it. This step handles missing values, aligns data by date across the different metals, selects relevant columns, and saves a combined CSV file.
   You will be prompted to specify the frequency (daily/weekly) of the raw data you wish to cleanse (this should match the frequency used for `fetch-data`).

   ```bash
   cd trading_system
   python main.py clean-data
   ```
   The cleansed and combined data (e.g., `cleaned_combined_daily_prices.csv`) will be saved in the `trading_system/data/` directory.

**(Further steps for analysis, backtesting, and recommendations will be added as they are implemented.)**

## Analysis Methodology

(Link to or content of `ANALYSIS_METHODOLOGY.md` to be added)
