import pandas as pd
import os
from .data_cleanser import MERGED_DATA_DIR # To locate merged data files
# Potentially import strategies or signal generators from backtester or analyzer
from .backtester import SMACrossoverStrategy # Example: reuse SMA strategy for signals

# --- Recommendation Logic ---

class SimpleRecommender:
    def __init__(self, data_df, asset_column, short_window=20, long_window=50):
        self.data_df = data_df
        self.asset_column = asset_column
        self.short_window = short_window
        self.long_window = long_window

        if asset_column not in data_df.columns:
            raise ValueError(f"Asset column '{asset_column}' not found in data.")

    def get_technical_signal(self):
        """
        Generates a technical signal based on SMA Crossover.
        Returns: 'BUY', 'SELL', or 'HOLD' for the latest data point.
        """
        # Use the SMACrossoverStrategy's signal generation part
        # We only care about the most recent signal for recommendation
        try:
            sma_strategy = SMACrossoverStrategy(
                asset_column=self.asset_column,
                short_window=self.short_window,
                long_window=self.long_window
            )
            signals_df = sma_strategy.generate_signals(self.data_df)
            if signals_df.empty:
                return "HOLD", "SMA signals could not be generated."

            latest_target_position = signals_df['target_position'].iloc[-1]

            if latest_target_position == 1.0:
                return "BUY", f"SMA({self.short_window}/{self.long_window}) indicates bullish crossover."
            elif latest_target_position == 0.0: # Assuming 0 means "flat" or "sell to go flat"
                 # Check previous state to differentiate HOLD vs SELL
                if len(signals_df) > 1 and signals_df['target_position'].iloc[-2] == 1.0:
                    return "SELL", f"SMA({self.short_window}/{self.long_window}) indicates bearish crossover (exit long)."
                return "HOLD", f"SMA({self.short_window}/{self.long_window}) indicates neutral/flat position."
            else: # Should not happen with current SMA logic if target_position is 0 or 1
                return "HOLD", f"SMA({self.short_window}/{self.long_window}) state is neutral."

        except Exception as e:
            print(f"Error generating SMA signal for recommender: {e}")
            return "HOLD", "Error in technical signal generation."

    def get_macro_context_signal(self):
        """
        Placeholder for generating a signal based on macro context.
        For now, returns a neutral signal.
        """
        # Example (very naive): If US_REAL_GDP_YOY_QUARTERLY is available and > 2.0, favorable.
        gdp_col = "US_REAL_GDP_YOY_QUARTERLY" # Example
        if gdp_col in self.data_df.columns and not self.data_df[gdp_col].empty:
# Example (very naive): If US_REAL_GDP_YOY_QUARTERLY is available and > 2.0, favorable.
        gdp_col = "US_REAL_GDP_YOY_QUARTERLY" # Example
        if gdp_col in self.data_df.columns and not self.data_df[gdp_col].empty:
            try:
                latest_gdp = self.data_df[gdp_col].dropna().iloc[-1]
                if not pd.isna(latest_gdp) and isinstance(latest_gdp, (int, float)):
                    if latest_gdp > 2.5: # Arbitrary threshold
                        return "POSITIVE", f"Recent GDP ({latest_gdp:.2f}%) is strong."
                    elif latest_gdp < 1.0: # Arbitrary threshold
                        return "NEGATIVE", f"Recent GDP ({latest_gdp:.2f}%) is weak."
                    else:
                        return "NEUTRAL", f"Recent GDP ({latest_gdp:.2f}%) is moderate."
                else:
                    return "NEUTRAL", "Invalid GDP value encountered."
            except Exception as e:
                print(f"Error processing GDP data: {e}")
                return "NEUTRAL", "Error in macro context analysis."
        return "NEUTRAL", "Macro context not fully analyzed or data unavailable."

    def get_overall_recommendation(self):
            if latest_gdp > 2.5: # Arbitrary threshold
                return "POSITIVE", f"Recent GDP ({latest_gdp:.2f}%) is strong."
            elif latest_gdp < 1.0: # Arbitrary threshold
                return "NEGATIVE", f"Recent GDP ({latest_gdp:.2f}%) is weak."
            else:
                return "NEUTRAL", f"Recent GDP ({latest_gdp:.2f}%) is moderate."
        return "NEUTRAL", "Macro context not fully analyzed or data unavailable."

    def get_overall_recommendation(self):
        """
        Combines signals to produce a final recommendation.
        """
        tech_signal, tech_reason = self.get_technical_signal()
        macro_signal, macro_reason = self.get_macro_context_signal() # Placeholder

        print(f"\n--- Recommendation Inputs ---")
        print(f"Technical Signal ({self.asset_column}): {tech_signal} ({tech_reason})")
        print(f"Macro Context Signal: {macro_signal} ({macro_reason})")

        # Simple combination logic (can be much more sophisticated)
        if tech_signal == "BUY":
            if macro_signal == "NEGATIVE":
                return "HOLD", "Buy signal from technicals, but macro context is negative. Caution advised."
            return "BUY", f"{tech_reason} Macro context: {macro_reason}."
        elif tech_signal == "SELL": # Technical SELL means exit long
            if macro_signal == "POSITIVE":
                 return "HOLD", "Sell signal (exit long) from technicals, but macro context is positive. Consider holding."
            return "SELL", f"{tech_reason} Macro context: {macro_reason}."
        else: # tech_signal is HOLD
            if macro_signal == "POSITIVE":
                return "HOLD", f"Technicals neutral, but macro context positive. Monitor for buy opportunities. ({tech_reason})"
            elif macro_signal == "NEGATIVE":
                return "HOLD", f"Technicals neutral, macro context negative. Caution. ({tech_reason})"
            return "HOLD", f"Both technicals and macro context appear neutral. ({tech_reason})"


def run_recommendation_workflow(data_file_path, asset_column, short_sma=20, long_sma=50):
    """
    Main workflow to generate and print a recommendation.
    """
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found: {data_file_path}")
        return

    try:
        data_df = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
        print(f"Loaded data for recommendation from: {data_file_path} (Shape: {data_df.shape})")
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}")
        return

    if data_df.empty:
        print("Data file is empty. Cannot generate recommendation.")
        return

    if asset_column not in data_df.columns:
        print(f"Error: Asset column '{asset_column}' not found in the data file.")
        print(f"Available columns: {', '.join(data_df.columns)}")
        return

    try:
        recommender = SimpleRecommender(data_df, asset_column, short_window=short_sma, long_window=long_sma)
        recommendation, reason = recommender.get_overall_recommendation()

        print("\n--- Recommendation ---")
        print(f"Asset: {asset_column}")
        print(f"Date of Data: {data_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Recommendation: {recommendation.upper()}")
        print(f"Reason: {reason}")

    except ValueError as e:
        print(f"Error initializing recommender: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during recommendation generation: {e}")


if __name__ == "__main__":
    print("Running Recommender module directly for testing...")

    # Assumes 'final_combined_daily_data.csv' exists from previous steps or dummy creation
    dummy_data_path = os.path.join(MERGED_DATA_DIR, "final_combined_daily_data.csv")

    if not os.path.exists(dummy_data_path):
        print(f"Dummy data file {dummy_data_path} not found. Creating one for test (simplified).")
        if not os.path.exists(MERGED_DATA_DIR):
            os.makedirs(MERGED_DATA_DIR)

        dates = pd.date_range(start='2022-01-01', periods=100, freq='B')
        # Create data where short SMA just crossed above long SMA for GOLD
        price_data_gold = np.concatenate([
            np.linspace(100, 90, 40), # Initial downtrend
            np.linspace(90, 110, 60)  # Uptrend causing crossover
        ])
        gold_prices = pd.Series(price_data_gold, index=dates, name="GOLD_Adj_Close")

        # Simulate some other columns
        silver_prices = pd.Series(50 + np.cumsum(np.random.randn(100) * 0.3), index=dates, name="SILVER_Adj_Close")
        us_gdp = pd.Series(np.linspace(1.0, 2.5, 100), index=dates, name="US_REAL_GDP_YOY_QUARTERLY").ffill() # Positive GDP trend

        dummy_df = pd.concat([gold_prices, silver_prices, us_gdp], axis=1)
        dummy_df.index.name = 'Date'
        dummy_df.to_csv(dummy_data_path)
        print(f"Dummy file {dummy_data_path} created.")

    test_asset = "GOLD_Adj_Close"
    print(f"\n--- Test Recommendation for {test_asset} ---")
    run_recommendation_workflow(
        data_file_path=dummy_data_path,
        asset_column=test_asset,
        short_sma=10, # Shorter windows for smaller dummy dataset
        long_sma=20
    )

    # Test with different data to simulate SELL/HOLD
    if os.path.exists(dummy_data_path): # Create new dummy data for sell
        dates_sell = pd.date_range(start='2022-01-01', periods=100, freq='B')
        price_data_gold_sell = np.concatenate([
            np.linspace(100, 110, 40), # Initial uptrend
            np.linspace(110, 90, 60)  # Downtrend causing crossover
        ])
        gold_prices_sell = pd.Series(price_data_gold_sell, index=dates_sell, name="GOLD_Adj_Close")
        us_gdp_weak = pd.Series(np.linspace(2.0, 0.5, 100), index=dates_sell, name="US_REAL_GDP_YOY_QUARTERLY").ffill() # Weak GDP trend
        dummy_df_sell = pd.concat([gold_prices_sell, us_gdp_weak], axis=1)
        dummy_df_sell.index.name = 'Date'
        dummy_df_sell.to_csv(dummy_data_path, mode='w') # Overwrite
        print(f"Overwrote dummy file {dummy_data_path} for SELL signal test.")

        print(f"\n--- Test Recommendation for {test_asset} (expecting SELL/HOLD) ---")
        run_recommendation_workflow(
            data_file_path=dummy_data_path,
            asset_column=test_asset,
            short_sma=10,
            long_sma=20
        )
