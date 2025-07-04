import pandas as pd
import numpy as np
from .data_cleanser import MERGED_DATA_DIR # To locate merged data files
import os

# --- Constants ---
RISK_FREE_RATE = 0.0 # For Sharpe Ratio calculation, can be adjusted
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_TRANSACTION_COST_PCT = 0.001 # 0.1% per trade

# --- Helper Functions for Metrics ---
def calculate_total_return(portfolio_values):
    """Calculates the total return from a series of portfolio values."""
    if portfolio_values.empty:
        return 0.0
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

def calculate_annualized_return(total_return, num_days, data_frequency):
    """Calculates annualized return."""
    if data_frequency == "daily":
        trading_days_per_year = 252
    elif data_frequency == "weekly":
        trading_days_per_year = 52
    else: # Should not happen if input is validated
        trading_days_per_year = 252
        print(f"Warning: Unknown data frequency '{data_frequency}' for annualized return. Assuming daily (252 days/year).")

    if num_days == 0: return 0.0
    years = num_days / trading_days_per_year
    if years == 0: return total_return # Avoid division by zero if less than a year of data for annualization
    annualized_ret = (1 + total_return) ** (1 / years) - 1
    return annualized_ret

def calculate_annualized_volatility(portfolio_returns, data_frequency):
    """Calculates annualized volatility from a series of portfolio returns."""
    if portfolio_returns.empty or len(portfolio_returns) < 2:
        return 0.0

    if data_frequency == "daily":
        trading_days_per_year = 252
    elif data_frequency == "weekly":
        trading_days_per_year = 52
    else:
        trading_days_per_year = 252
        print(f"Warning: Unknown data frequency '{data_frequency}' for annualized volatility. Assuming daily (252 days/year).")

    volatility = portfolio_returns.std() * np.sqrt(trading_days_per_year)
    return volatility

def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate=RISK_FREE_RATE):
    """Calculates the Sharpe Ratio."""
    if annualized_volatility == 0:
        return np.nan # Avoid division by zero; Sharpe is undefined
    return (annualized_return - risk_free_rate) / annualized_volatility

def calculate_max_drawdown(portfolio_values):
    """Calculates the Maximum Drawdown."""
    if portfolio_values.empty:
        return 0.0
    # Calculate the cumulative maximum value up to each point
    cumulative_max = portfolio_values.cummax()
    # Calculate drawdowns from the cumulative maximum
# Calculate the cumulative maximum value up to each point
    cumulative_max = portfolio_values.cummax()
    # Calculate drawdowns from the cumulative maximum
    drawdowns = np.where(cumulative_max != 0, (portfolio_values - cumulative_max) / cumulative_max, 0)
    # Get the minimum (largest negative) drawdown
    max_dd = np.min(drawdowns)
    return max_dd if not np.isnan(max_dd) else 0.0


# --- Simple SMA Crossover Strategy ---
    # Get the minimum (largest negative) drawdown
    max_dd = drawdowns.min()
    return max_dd if not pd.isna(max_dd) else 0.0


# --- Simple SMA Crossover Strategy ---
class SMACrossoverStrategy:
    def __init__(self, asset_column, short_window, long_window, data_frequency="daily"):
        self.asset_column = asset_column
        self.short_window = short_window
        self.long_window = long_window
        self.data_frequency = data_frequency # 'daily' or 'weekly'
        self.signals = None
        self.data = None

        if short_window >= long_window:
            raise ValueError("Short window must be less than long window for SMA Crossover.")

    def generate_signals(self, data_df):
        """Generates trading signals based on SMA crossover."""
        self.data = data_df.copy()
        if self.asset_column not in self.data.columns:
            raise ValueError(f"Asset column '{self.asset_column}' not found in data.")

        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['price'] = self.data[self.asset_column]
        self.signals['short_mavg'] = self.data[self.asset_column].rolling(window=self.short_window, min_periods=1).mean()
        self.signals['long_mavg'] = self.data[self.asset_column].rolling(window=self.long_window, min_periods=1).mean()

        # Generate signal: 1 for buy, -1 for sell, 0 for hold
        # Initial signal is 0
        self.signals['signal'] = 0.0

        # When short_mavg crosses above long_mavg, generate buy signal (1)
        self.signals.loc[self.signals['short_mavg'] > self.signals['long_mavg'], 'signal'] = 1.0

        # When short_mavg crosses below long_mavg, generate sell signal (-1)
        # This condition should only apply if it was previously a buy or neutral.
        # For simplicity, we'll allow direct switch from buy to sell based on crossover.
        self.signals.loc[self.signals['short_mavg'] < self.signals['long_mavg'], 'signal'] = -1.0

        # Create actual trading orders: difference from previous signal
        # This means a trade happens when the signal changes.
        # E.g., from hold (0) to buy (1) -> buy order
        # E.g., from buy (1) to sell (-1) -> sell order (close long, open short, or just sell asset)
        # For simplicity, let's assume we are either long or flat (no short selling for now).
        # Signal: 1 = Go Long (if flat), Stay Long (if long)
        # Signal: -1 = Go Flat (if long)

        # Refined signal logic for positions:
        # If short > long: target position is 1 (long)
        # If short < long: target position is 0 (flat)
        self.signals['target_position'] = 0.0
        self.signals.loc[self.signals['short_mavg'] > self.signals['long_mavg'], 'target_position'] = 1.0
        self.signals.loc[self.signals['short_mavg'] < self.signals['long_mavg'], 'target_position'] = 0.0 # Go flat

        # Actual trades are when target_position changes from previous day's actual position
        # This will be handled by the backtesting loop.

        # Drop NaNs created by rolling means, signals start when both SMAs are available
        self.signals = self.signals.dropna(subset=['short_mavg', 'long_mavg'])

        print(f"Signals generated for {self.asset_column} using SMA({self.short_window}, {self.long_window}).")
        # print("Sample signals (tail):")
        # print(self.signals.tail())
        return self.signals


# --- Backtesting Engine ---
def run_backtest(data_df, strategy_instance, initial_capital, transaction_cost_pct, data_frequency):
    """
    Runs the backtest for a given strategy.

    Args:
        data_df (pd.DataFrame): DataFrame with price data (and any other indicators needed by strategy).
        strategy_instance (object): An instance of a strategy class with a generate_signals method.
        initial_capital (float): The starting capital for the backtest.
        transaction_cost_pct (float): Transaction cost as a percentage of trade value.
        data_frequency (str): 'daily' or 'weekly' for annualized calculations.

    Returns:
        dict: A dictionary containing performance metrics and portfolio details.
    """
    print("\n--- Running Backtest ---")
    signals_df = strategy_instance.generate_signals(data_df)
    if signals_df.empty:
        print("No signals generated by the strategy. Cannot run backtest.")
        return None

    # Align signals_df index with data_df if they became different (e.g. due to dropna in signals)
    # The backtest loop will iterate over signals_df.index

    portfolio = pd.DataFrame(index=signals_df.index)
    portfolio['holdings'] = 0.0  # Value of asset held
    portfolio['cash'] = float(initial_capital)
    portfolio['total_value'] = float(initial_capital)
    portfolio['asset_position'] = 0.0 # Number of units of the asset held
    portfolio['trades'] = 0 # Number of trades executed

    current_position_units = 0.0 # Units of asset

    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Transaction Cost (per trade): {transaction_cost_pct*100:.3f}%")

    for i in range(len(signals_df)):
        date = signals_df.index[i]
        price = signals_df['price'].iloc[i]
        target_position_signal = signals_df['target_position'].iloc[i] # 1 for long, 0 for flat

        # Carry forward previous day's values if no trade
        if i > 0:
            portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
            portfolio.loc[date, 'asset_position'] = portfolio['asset_position'].iloc[i-1]
            # Holdings value will be updated based on current price
        else: # First day
            portfolio.loc[date, 'cash'] = initial_capital
            portfolio.loc[date, 'asset_position'] = 0.0


        # --- Trade Execution Logic ---
        # Decision based on target_position_signal and current_position_units

        # If target is LONG (1.0) and currently FLAT (0 units)
        if target_position_signal == 1.0 and current_position_units == 0:
            # Buy: spend all available cash
            cash_to_invest = portfolio['cash'].iloc[i-1] if i > 0 else initial_capital
            cost_of_transaction = cash_to_invest * transaction_cost_pct
            cash_after_cost = cash_to_invest - cost_of_transaction
            units_to_buy = cash_after_cost / price

            current_position_units = units_to_buy
            portfolio.loc[date, 'cash'] = 0.0 # All cash used
            portfolio.loc[date, 'asset_position'] = current_position_units
            num_trades_executed += 1

        # If target is FLAT (0.0) and currently LONG (>0 units)
        elif target_position_signal == 0.0 and current_position_units > 0:
            # Sell: liquidate all holdings
            proceeds_from_sale = current_position_units * price
            cost_of_transaction = proceeds_from_sale * transaction_cost_pct
            cash_after_cost = proceeds_from_sale - cost_of_transaction

            portfolio.loc[date, 'cash'] = (portfolio['cash'].iloc[i-1] if i > 0 else 0) + cash_after_cost
            current_position_units = 0.0
            portfolio.loc[date, 'asset_position'] = current_position_units
            portfolio.loc[date, 'trades'] += 1
            print(f"{date.strftime('%Y-%m-%d')}: SELL signal (go flat). Price: ${price:,.2f}. Selling all units.")

        # Update holdings value based on current price and position
        portfolio.loc[date, 'holdings'] = current_position_units * price
        portfolio.loc[date, 'total_value'] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]

    # --- Performance Calculation ---
    portfolio['returns'] = portfolio['total_value'].pct_change().fillna(0)

    total_ret = calculate_total_return(portfolio['total_value'])
    num_days_in_backtest = len(portfolio) # Number of periods in the backtest

    annualized_ret = calculate_annualized_return(total_ret, num_days_in_backtest, data_frequency)
    annualized_vol = calculate_annualized_volatility(portfolio['returns'], data_frequency)
    sharpe = calculate_sharpe_ratio(annualized_ret, annualized_vol, RISK_FREE_RATE)
    max_dd = calculate_max_drawdown(portfolio['total_value'])

    num_trades = portfolio['trades'].sum()

    print("\n--- Backtest Results ---")
    print(f"Data Frequency: {data_frequency.capitalize()}")
    print(f"Backtest Period: {portfolio.index.min().strftime('%Y-%m-%d')} to {portfolio.index.max().strftime('%Y-%m-%d')}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Annualized Return: {annualized_ret*100:.2f}%")
    print(f"Annualized Volatility: {annualized_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"Total Trades: {num_trades}")
    print(f"Final Portfolio Value: ${portfolio['total_value'].iloc[-1]:,.2f}")

    results = {
        "total_return": total_ret,
        "annualized_return": annualized_ret,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "final_portfolio_value": portfolio['total_value'].iloc[-1],
        "portfolio_history": portfolio,
        "signals_df": signals_df
    }
    return results


def run_backtesting_workflow(data_file_path, frequency, asset_column, short_window, long_window,
                             initial_capital=DEFAULT_INITIAL_CAPITAL,
                             transaction_cost_pct=DEFAULT_TRANSACTION_COST_PCT):
    """
    Main workflow function for running a backtest from CLI parameters.
    """
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found: {data_file_path}")
        return

    try:
        data_df = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
        print(f"Loaded data for backtesting from: {data_file_path} (Shape: {data_df.shape})")
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}")
        return

    if asset_column not in data_df.columns:
        print(f"Error: Asset column '{asset_column}' not found in the data file.")
        print(f"Available columns: {', '.join(data_df.columns)}")
        return

    # Instantiate strategy
    try:
        strategy = SMACrossoverStrategy(
            asset_column=asset_column,
            short_window=short_window,
            long_window=long_window,
            data_frequency=frequency
        )
    except ValueError as e:
        print(f"Error initializing strategy: {e}")
        return

    # Run backtest
    backtest_results = run_backtest(
        data_df=data_df,
        strategy_instance=strategy,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        data_frequency=frequency
    )

    if backtest_results:
        # Further actions like plotting or saving detailed results can be added here
        # For example, save portfolio history to a CSV
        # portfolio_hist_path = os.path.join(MERGED_DATA_DIR, f"portfolio_history_{asset_column}_sma_{short_window}_{long_window}.csv")
        # backtest_results["portfolio_history"].to_csv(portfolio_hist_path)
        # print(f"Portfolio history saved to: {portfolio_hist_path}")
        pass
    else:
        print("Backtest failed to produce results.")


if __name__ == "__main__":
    # --- Example Usage for Direct Testing ---
    print("Running Backtester module directly for testing...")

    # 1. Ensure 'final_combined_daily_data.csv' exists.
    #    You'd typically run:
    #    python main.py fetch-data (and provide inputs for daily, e.g., 2015-2023)
    #    python main.py fetch-macro-data (e.g., 2015-2023)
    #    python main.py clean-data (and select daily)
    #    python main.py merge-macro-data --frequency daily

    # For testing, let's create a simplified dummy 'final_combined_daily_data.csv'
    dummy_data_path = os.path.join(MERGED_DATA_DIR, "final_combined_daily_data.csv")

    if not os.path.exists(dummy_data_path):
        print(f"Dummy data file {dummy_data_path} not found. Creating one for test.")
        if not os.path.exists(MERGED_DATA_DIR):
            os.makedirs(MERGED_DATA_DIR)

        dates = pd.date_range(start='2020-01-01', periods=500, freq='B') # Approx 2 years of business days
        price_data = 100 + np.cumsum(np.random.randn(500) * 0.5) # Random walk prices
        gold_prices = pd.Series(price_data, index=dates, name="GOLD_Adj_Close")

        # Simulate some other columns that might be in final_combined
        silver_prices = pd.Series(50 + np.cumsum(np.random.randn(500) * 0.3), index=dates, name="SILVER_Adj_Close")
        us_gdp = pd.Series(np.random.randn(500) * 0.1 + 2, index=dates, name="US_REAL_GDP_YOY_QUARTERLY").ffill()

        dummy_df = pd.concat([gold_prices, silver_prices, us_gdp], axis=1)
        dummy_df.index.name = 'Date'
        dummy_df.to_csv(dummy_data_path)
        print(f"Dummy file {dummy_data_path} created with GOLD_Adj_Close and other columns.")

    # Parameters for the test
    test_data_file = dummy_data_path
    test_frequency = "daily"
    test_asset = "GOLD_Adj_Close" # Must exist in the dummy file
    test_short_sma = 20
    test_long_sma = 50
    test_initial_capital = 100000
    test_txn_cost = 0.001

    print(f"\n--- Test Backrun with SMA Crossover ({test_short_sma}/{test_long_sma}) on {test_asset} ---")
    run_backtesting_workflow(
        data_file_path=test_data_file,
        frequency=test_frequency,
        asset_column=test_asset,
        short_window=test_short_sma,
        long_window=test_long_sma,
        initial_capital=test_initial_capital,
        transaction_cost_pct=test_txn_cost
    )
