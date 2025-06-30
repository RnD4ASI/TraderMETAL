# Analysis Methodology

This document outlines the analytical methods used in the Trading Analysis and Recommendation System.

## 1. Data Preprocessing

*   **Data Source**: Historical price data for Gold, Silver, and Platinum (e.g., from ETFs like GLD, SLV, PPLT or futures contracts) will be fetched using the `yfinance` library or a similar financial data API.
*   **Frequency**: Users can select daily or weekly data.
*   **Data Cleansing**:
    *   **Missing Values**: Forward-fill (`ffill`) will be primarily used to handle missing data points, assuming the last known price is the best estimate for a non-trading day. Other methods like interpolation might be considered if `ffill` is insufficient.
    *   **Alignment**: Data for all metals will be aligned by date to ensure comparability. This involves merging the datasets and handling dates where one metal might have data while another doesn't (typically by using an inner join or by forward-filling after an outer join).
    *   **Outliers**: Initial implementation will focus on robust data sources. Extreme outlier detection (e.g., using standard deviation thresholds or IQR) might be added later if data quality issues are observed.

## 2. Technical Indicators and Analysis Techniques

The following technical indicators and analysis techniques will be implemented. The parameters for these indicators (e.g., window lengths) will be configurable or set to commonly used defaults.

### 2.1. Trend Indicators

*   **Simple Moving Average (SMA)**:
    *   **Calculation**: The average price over a specified number of periods.
    *   **Usage**: Helps identify the direction of the trend. Crossovers of different SMAs (e.g., 50-day vs. 200-day) can signal trend changes.
    *   **Default Periods**: 20-day, 50-day, 200-day.
*   **Exponential Moving Average (EMA)**:
    *   **Calculation**: Similar to SMA but gives more weight to recent prices, making it more responsive to new information.
    *   **Usage**: Similar to SMA for trend identification and crossover signals, but reacts faster to price changes.
    *   **Default Periods**: 12-day, 26-day (often used in MACD), 50-day.

### 2.2. Volatility Indicators

*   **Bollinger Bands**:
    *   **Calculation**: Consists of a middle band (SMA, typically 20-day) and upper and lower bands that are typically two standard deviations above and below the middle band.
    *   **Usage**: Helps assess volatility and identify overbought/oversold conditions. Prices are considered overbought when they touch the upper band and oversold when they touch the lower band. "Walking the bands" can indicate strong trends.
    *   **Default Parameters**: 20-day SMA, 2 standard deviations.
*   **Standard Deviation (Volatility)**:
    *   **Calculation**: The statistical measure of price dispersion. Calculated over a rolling window.
    *   **Usage**: Higher standard deviation indicates higher volatility, and vice-versa. Can be used to set stop-loss levels or to gauge market risk.
    *   **Default Period**: 20-day.

### 2.3. Momentum Indicators (To be considered for future enhancement)

*   **Relative Strength Index (RSI)**:
    *   **Calculation**: Measures the speed and change of price movements. Oscillates between 0 and 100.
    *   **Usage**: Traditionally, RSI is considered overbought when above 70 and oversold when below 30. Divergences between RSI and price can signal potential trend reversals.
*   **Moving Average Convergence Divergence (MACD)**:
    *   **Calculation**: Calculated by subtracting the 26-period EMA from the 12-period EMA. A 9-period EMA of the MACD (the "signal line") is then plotted on top of the MACD line.
    *   **Usage**: MACD crossovers, signal line crossovers, and divergences can be used as buy/sell signals.

### 2.4. Correlation Analysis

*   **Pearson Correlation Coefficient**:
    *   **Calculation**: Measures the linear correlation between the prices of Gold, Silver, and Platinum. Calculated over a rolling window or for the entire period.
    *   **Usage**: Helps understand how the metals move in relation to each other. A high positive correlation means they tend to move in the same direction, while a negative correlation means they move in opposite directions.
    *   **Default Window**: 60-day or 90-day rolling correlation.

## 3. Forecasting Models (Initial Implementation)

*   **Autoregressive Integrated Moving Average (ARIMA)**:
    *   **Description**: A statistical model used for analyzing and forecasting time series data. It combines autoregression (AR), differencing (I for Integrated), and moving average (MA) components.
    *   **Parameter Selection**: (p, d, q) parameters will be determined using techniques like ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots, or by using automated parameter selection algorithms (e.g., `auto_arima` if a suitable library is used).
    *   **Usage**: To generate short-term price forecasts (e.g., 30, 60, 90 days).
*   **Exponential Smoothing (e.g., Holt-Winters)**:
    *   **Description**: Another widely used time series forecasting method that assigns exponentially decreasing weights to past observations. Holt-Winters can also model trend and seasonality.
    *   **Usage**: Similar to ARIMA, for generating short-term price forecasts.

## 4. Backtesting Methodology

*   **Strategy Definition**: Users can define simple strategies based on signals from the implemented technical indicators (e.g., "Buy when 50-day SMA crosses above 200-day SMA," "Sell when price crosses below lower Bollinger Band").
*   **Execution**: Trades are simulated based on historical data. Assumptions about transaction costs and slippage will be minimal in the initial version but can be added later.
*   **Performance Metrics**:
    *   **Total Return**: The overall percentage gain or loss over the backtesting period.
    *   **Annualized Return**: The geometric average amount of money earned by an investment each year over a given time period.
    *   **Sharpe Ratio**: Measures risk-adjusted return (average return earned in excess of the risk-free rate per unit of volatility or total risk).
    *   **Maximum Drawdown**: The largest peak-to-trough decline during a specific period, indicating downside risk.
    *   **Win Rate**: Percentage of profitable trades.

## 5. Recommendation Logic

*   **Signal Aggregation**: Recommendations (BUY, HOLD, SELL) will be generated by aggregating signals from:
    *   Trend indicators (e.g., price above/below key SMAs/EMAs, SMA crossovers).
    *   Volatility indicators (e.g., price relative to Bollinger Bands).
    *   Forecasting models (e.g., predicted price movement).
*   **Weighting (Future Enhancement)**: Initially, a simple voting or rule-based system will be used. Future enhancements could involve weighting different signals based on their historical performance or user preference.
*   **Confidence Level**: The confidence level might be derived from:
    *   The number of indicators confirming a particular signal.
    *   The strength of the signal (e.g., how far the price is from a moving average, or the probability output from a forecasting model).
*   **Expected Return**: Based on the forecasts from ARIMA/Exponential Smoothing models for 30, 60, and 90-day horizons. The reliability of these long-range point forecasts will be clearly communicated as being indicative and subject to significant uncertainty.

This document will be updated as new analytical methods are added or existing ones are refined.
