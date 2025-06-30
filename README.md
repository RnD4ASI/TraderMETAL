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

**3. Perform Multivariate Analysis (Stationarity and Cointegration):**

   Use the `analyze-mv` command to perform initial multivariate analysis on the cleaned data. This involves:
   *   Testing each metal's price series (or returns) for stationarity using the Augmented Dickey-Fuller (ADF) test.
   *   Performing a Johansen cointegration test on the price levels to identify potential long-run equilibrium relationships between Gold, Silver, and Platinum.
   You will be prompted to choose the data frequency (daily/weekly) and whether to analyze price levels or percentage returns.

   ```bash
   cd trading_system
   python main.py analyze-mv
   ```
   The results of these tests, including interpretations, will be printed to the console. This helps in understanding the time series properties and guides the selection of appropriate models (like VAR or VECM) for more advanced forecasting.

**4. Perform Deep Learning Forecasting (LSTM):**

   Use the `analyze-dl` command to train an LSTM (Long Short-Term Memory) neural network for forecasting. This model can capture complex non-linear patterns in the time series.
   *   You will be prompted to select the target metal, feature columns (which can include prices/volumes of all three metals), sequence length (lookback window), forecast horizon, and training parameters (epochs, batch size).
   *   The command will preprocess the data, build and train the LSTM model, evaluate it on a test set, and provide a forecast for the target metal.
   *   **Note:** This feature requires `tensorflow` to be installed (`pip install tensorflow`). Training can be computationally intensive and time-consuming.

   ```bash
   cd trading_system
   python main.py analyze-dl
   ```
   Model training progress, evaluation metrics (RMSE, MAE), and the final forecast will be printed to the console.

**(Further steps for detailed VAR/VECM modeling, other univariate analyses, backtesting, and recommendations will be added as they are implemented.)**

## Analysis Methodology

For detailed information on the analytical methods used, including data preprocessing, technical indicators, multivariate analysis (ADF, Johansen tests), and deep learning forecasting (LSTM), please refer to the [ANALYSIS_METHODOLOGY.md](ANALYSIS_METHODOLOGY.md) file.
