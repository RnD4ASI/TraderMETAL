# Analysis Methodology

This document outlines the analytical methods used in the Trading Analysis and Recommendation System.

## 1. Data Preprocessing

*   **Data Source**: Historical price data for Gold, Silver, and Platinum (e.g., from ETFs like GLD, SLV, PPLT or futures contracts) will be fetched using the `yfinance` library or a similar financial data API. The raw data for each metal is stored in separate CSV files.
*   **Frequency**: Users can select daily or weekly data for fetching and subsequent cleansing.
*   **Data Cleansing Process**: The goal of the cleansing process is to produce a single, unified DataFrame containing the relevant price and volume data for all three metals, aligned by date, and ready for analysis.

    1.  **Load Individual Metal Data**:
        *   Raw CSV files for Gold, Silver, and Platinum (based on the user-selected frequency from the data crawling step) are loaded into separate pandas DataFrames.

    2.  **Column Selection and Renaming**:
        *   For each DataFrame, the 'Date' column is parsed and set as the index.
        *   Relevant columns are selected: 'Adj Close' (Adjusted Close price) and 'Volume'.
        *   To avoid ambiguity after merging, these columns are renamed to be metal-specific. For example, for Gold (GLD), 'Adj Close' becomes `GOLD_Adj Close` and 'Volume' becomes `GOLD_Volume`. This pattern is applied to Silver (e.g., `SILVER_Adj Close`) and Platinum (e.g., `PLATINUM_Adj Close`).

    3.  **Handle Missing Values (Individual Series)**:
        *   Missing values (NaNs) within each individual metal's time series are handled. The primary strategy is:
            *   Forward-fill (`ffill`): Propagates the last valid observation forward.
            *   Backward-fill (`bfill`): Fills remaining NaNs (typically at the beginning of the series if `ffill` couldn't fill them) with the next valid observation.
        *   This step ensures that each metal's series is as complete as possible before merging.

    4.  **Data Type Conversion**:
        *   The Date index is ensured to be of datetime type (typically handled by `pd.read_csv` with `parse_dates` and `index_col`).
        *   All price ('Adj Close') and 'Volume' columns are converted to `float` data type to ensure they are suitable for numerical calculations.

    5.  **Merge DataFrames**:
        *   The cleansed DataFrames for Gold, Silver, and Platinum are merged into a single DataFrame.
        *   The merge is performed on the Date index using an `outer` join. This ensures that all dates present in any of the individual datasets are included in the final DataFrame.
        *   After the outer join, it's possible that some metals might have NaN values on dates where other metals traded. These are handled by applying another round of `ffill` and then `bfill` on the merged DataFrame. This helps to align data on days where, for instance, one market might have been open while another was closed for a short period.

    6.  **Final NaN Handling**:
        *   After merging and subsequent `ffill`/`bfill`, any rows where all adjusted close price columns (`GOLD_Adj Close`, `SILVER_Adj Close`, `PLATINUM_Adj Close`) are *still* NaN are dropped. This typically removes dates at the very beginning or end of the overall period if no metal had data.

    7.  **Save Cleansed Data**:
        *   The final, combined, and cleansed DataFrame is saved to a new CSV file in the `data/` directory (e.g., `cleaned_combined_daily_prices.csv` or `cleaned_combined_weekly_prices.csv`).

    *   **Outliers**: Initial implementation will focus on robust data sources and the described NaN handling. Explicit outlier detection and treatment (e.g., winsorization, clipping based on standard deviations or IQR) is not part of the initial cleansing step but can be considered as a future enhancement if data quality issues from the source prove problematic for analysis.

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

## 3. Multivariate Time Series Analysis

This section details advanced techniques used to model and understand the interdependencies between the prices of Gold, Silver, and Platinum. The initial implementation will focus on stationarity testing and cointegration analysis, which are foundational for selecting appropriate multivariate models like VAR or VECM.

### 3.1. Stationarity Testing

*   **Concept**: A time series is stationary if its statistical properties (mean, variance, autocorrelation) are constant over time. Most time series models, including those used in multivariate analysis, assume stationarity. Price series of financial assets are often non-stationary (exhibiting trends or unit roots).
*   **Methodology - Augmented Dickey-Fuller (ADF) Test**:
    *   The ADF test is used to test for a unit root in a time series sample.
    *   **Null Hypothesis (H0)**: The time series has a unit root (it is non-stationary).
    *   **Alternative Hypothesis (H1)**: The time series does not have a unit root (it is stationary).
*   **Interpretation**:
    *   The test produces a test statistic, a p-value, and critical values for different confidence levels.
    *   If the p-value is less than a chosen significance level (e.g., 0.05), or if the test statistic is more negative than the critical values, the null hypothesis is rejected, and the series is considered stationary.
    *   If price series are found to be non-stationary, their first differences (returns) are often tested, as returns are typically stationary.

### 3.2. Cointegration Analysis

*   **Concept**: If two or more non-stationary time series are cointegrated, it means there is a long-run, statistically significant equilibrium relationship between them, even if they diverge in the short term. A linear combination of these series is stationary.
*   **Methodology - Johansen Test**:
    *   The Johansen test is a procedure for testing cointegration of several I(1) (integrated of order 1, i.e., non-stationary but their first difference is stationary) time series.
    *   It can determine the number of cointegrating relationships (cointegrating rank) within a group of time series.
    *   **Deterministic Trend Assumption**: The test can be run with different assumptions about the deterministic trend in the data (e.g., no trend, constant, constant and linear trend). The initial implementation will primarily use a constant (`det_order = 0`).
    *   **Lag Order**: The test requires specifying the number of lagged differences (`k_ar_diff`) in the underlying VAR model. This can be determined using information criteria (AIC, BIC) on a preliminary VAR.
*   **Interpretation**:
    *   The Johansen test produces two main statistics:
        *   **Trace Statistic**: Tests the null hypothesis that the number of cointegrating vectors is less than or equal to `r` against an alternative that it is greater than `r`.
        *   **Maximum Eigenvalue Statistic**: Tests the null hypothesis that the number of cointegrating vectors is `r` against an alternative of `r+1`.
    *   For each statistic, it's compared against critical values at given significance levels (e.g., 90%, 95%, 99%).
    *   The number of cointegrating relationships is determined by sequentially testing from `r=0` up to `r=n-1` (where `n` is the number of time series). The first non-rejection of the null hypothesis for each statistic indicates the rank. Often, both statistics are considered.

### 3.3. Vector Autoregression (VAR) Models

*   **Concept**: VAR models are used to capture the linear interdependencies among multiple time series. Each variable in a VAR has an equation explaining its evolution based on its own lagged values and the lagged values of the other variables in the model.
*   **Applicability**:
    *   Suitable for modeling multiple stationary time series.
    *   If original price series are non-stationary and *not* cointegrated, VAR models can be applied to their first differences (returns).
    *   If price series are non-stationary but cointegrated, a VECM is more appropriate.
*   **Key Aspects (Future Enhancements)**:
    *   **Lag Order Selection**: Determining the optimal number of lags using information criteria (AIC, BIC, HQIC).
    *   **Model Estimation**: Estimating the coefficients of the VAR model.
    *   **Stability Check**: Ensuring the estimated VAR model is stable (roots of the characteristic polynomial lie outside the unit circle).
    *   **Granger Causality Tests**: To determine if past values of one series can predict current values of another.
    *   **Impulse Response Functions (IRFs)**: To trace the effect of a shock (impulse) in one variable on itself and other variables in the system over time.
    *   **Forecasting**: Generating joint forecasts for all variables in the system.

### 3.4. Vector Error Correction Models (VECM)

*   **Concept**: A VECM is a specialized VAR model designed for use with non-stationary time series that are found to be cointegrated. It incorporates the cointegrating relationships (long-run equilibrium) into the model, allowing for analysis of both short-term dynamics and long-term adjustments.
*   **Applicability**: Used when time series (e.g., price levels) are non-stationary but cointegrated. The cointegration rank determined by the Johansen test is a key input (`k_ar_diff` in `statsmodels` VECM is the number of lags in differences, cointegration rank `r` is also specified).
*   **Key Aspects (Future Enhancements)**:
    *   The VECM includes "error correction" terms that represent deviations from the long-run equilibrium. The coefficients of these terms indicate the speed of adjustment back to equilibrium.
    *   Similar analyses to VAR (IRFs, Granger causality, forecasting) can be performed within the VECM framework, but their interpretation is richer due to the presence of the long-run relationship.

The initial implementation in `analyzer.py` will focus on performing stationarity tests (ADF) and cointegration tests (Johansen) to guide the choice of appropriate multivariate models for future development.

## 4. Forecasting Models

### 4.1 Univariate Forecasting Models (ARIMA, Exponential Smoothing)

*   **Autoregressive Integrated Moving Average (ARIMA)**:
    *   **Description**: A statistical model used for analyzing and forecasting time series data. It combines autoregression (AR), differencing (I for Integrated), and moving average (MA) components.
    *   **Parameter Selection**: (p, d, q) parameters will be determined using techniques like ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots, or by using automated parameter selection algorithms (e.g., `auto_arima` if a suitable library is used).
    *   **Usage**: To generate short-term price forecasts (e.g., 30, 60, 90 days) for individual metal series.
*   **Exponential Smoothing (e.g., Holt-Winters)**:
    *   **Description**: Another widely used time series forecasting method that assigns exponentially decreasing weights to past observations. Holt-Winters can also model trend and seasonality.
    *   **Usage**: Similar to ARIMA, for generating short-term price forecasts for individual metal series.

### 4.2 Deep Learning Models for Forecasting (LSTM)

*   **Concept - Long Short-Term Memory (LSTM) Networks**:
    *   LSTMs are a type of Recurrent Neural Network (RNN) particularly well-suited for learning from sequential data like time series.
    *   They are designed to overcome the vanishing gradient problem that can affect traditional RNNs, allowing them to learn long-range dependencies.
    *   LSTMs use a system of "gates" (input, forget, output) within their memory cells to control the flow of information, deciding what to store, what to discard, and what to output.
*   **Proposed Approach - Multivariate LSTM Forecasting**:
    *   The initial implementation will focus on forecasting the adjusted close price of a single target metal (e.g., Gold).
    *   Input features will include the historical adjusted close prices of all three metals (Gold, Silver, Platinum) and potentially their trading volumes. This allows the model to learn from inter-metal relationships.
    *   Using price returns (percentage changes) instead of absolute price levels as input features will be considered, as returns are often more stationary and can lead to better model performance.
*   **Data Preprocessing for LSTMs**:
    *   **Feature Selection**: Choose relevant columns from the cleaned, combined dataset (e.g., `GOLD_Adj_Close`, `SILVER_Adj_Close`, `PLATINUM_Adj_Close`, `GOLD_Volume`, `SILVER_Volume`, `PLATINUM_Volume`).
    *   **Scaling/Normalization**: All selected input features will be scaled, typically to a range of [0, 1] using `MinMaxScaler` from `scikit-learn`. The scaler is fitted *only* on the training dataset to prevent data leakage, and then used to transform the validation and test sets, as well as any new data for forecasting. The scaler for the target variable is stored separately to inverse-transform predictions back to their original scale.
    *   **Sequence Creation**: Time series data is converted into sequences of a fixed length (`sequence_length` or lookback window). For example, using the last 60 days of data (features) to predict the next day's price (target). This creates `(X, y)` pairs where `X` is a 3D array (samples, timesteps, features) and `y` is the target value(s).
    *   **Train/Validation/Test Split**: Data is split chronologically to respect the temporal order. A common split might be 70% for training, 10-15% for validation (tuning hyperparameters, early stopping), and 15-20% for final model evaluation (test set). There must be no overlap between these sets.
*   **Model Architecture (Sample)**:
    *   **Input Layer**: Defines the shape of the input sequences (`sequence_length`, `number_of_features`).
    *   **LSTM Layer(s)**: One or more LSTM layers (e.g., `tf.keras.layers.LSTM`) with a specified number of units (neurons).
    *   **Dropout Layer(s)**: `tf.keras.layers.Dropout` can be added after LSTM layers to reduce overfitting by randomly setting a fraction of input units to 0 during training.
    *   **Dense Output Layer**: A fully connected `tf.keras.layers.Dense` layer with a number of units corresponding to the forecast horizon (e.g., 1 unit for predicting a single next step). A linear activation function is typically used for regression tasks like price forecasting.
*   **Training Process**:
    *   **Compilation**: The model is compiled with a loss function (e.g., Mean Squared Error - `MSE` for regression), an optimizer (e.g., `Adam`), and evaluation metrics (e.g., Root Mean Squared Error - `RMSE`, Mean Absolute Error - `MAE`).
    *   **Fitting**: The model is trained using the `fit()` method on the training data (`X_train`, `y_train`), with the validation data (`X_val`, `y_val`) used to monitor performance.
    *   **Early Stopping**: An `EarlyStopping` callback (`tf.keras.callbacks.EarlyStopping`) will be used to monitor the validation loss and stop training if it doesn't improve for a specified number of epochs (`patience`), preventing overfitting and saving training time. The weights of the best performing epoch on the validation set are typically restored.
*   **Evaluation**:
    *   The trained model's performance is assessed on the unseen test set (`X_test`, `y_test`).
    *   Predictions are made, inverse-scaled to the original price/return domain, and then compared against the actual inverse-scaled target values using metrics like RMSE and MAE.
*   **Forecasting**:
    *   To predict future values, the most recent `sequence_length` of (scaled) data is fed into the trained model.
    *   For multi-step forecasting (predicting more than one step ahead), an iterative approach might be used where the prediction for one step is fed back as an input for predicting the next step, or the model is trained to predict multiple steps directly. The initial implementation will likely focus on single-step or a short fixed-horizon forecast.

## 5. Backtesting Methodology

*   **Strategy Definition**: Users can define simple strategies based on signals from the implemented technical indicators (e.g., "Buy when 50-day SMA crosses above 200-day SMA," "Sell when price crosses below lower Bollinger Band").
*   **Execution**: Trades are simulated based on historical data. Assumptions about transaction costs and slippage will be minimal in the initial version but can be added later.
*   **Performance Metrics**:
    *   **Total Return**: The overall percentage gain or loss over the backtesting period.
    *   **Annualized Return**: The geometric average amount of money earned by an investment each year over a given time period.
    *   **Sharpe Ratio**: Measures risk-adjusted return (average return earned in excess of the risk-free rate per unit of volatility or total risk).
    *   **Maximum Drawdown**: The largest peak-to-trough decline during a specific period, indicating downside risk.
    *   **Win Rate**: Percentage of profitable trades.

## 6. Recommendation Logic

*   **Signal Aggregation**: Recommendations (BUY, HOLD, SELL) will be generated by aggregating signals from:
    *   Trend indicators (e.g., price above/below key SMAs/EMAs, SMA crossovers).
    *   Volatility indicators (e.g., price relative to Bollinger Bands).
    *   Forecasting models (e.g., predicted price movement from univariate or multivariate models).
*   **Weighting (Future Enhancement)**: Initially, a simple voting or rule-based system will be used. Future enhancements could involve weighting different signals based on their historical performance or user preference.
*   **Confidence Level**: The confidence level might be derived from:
    *   The number of indicators confirming a particular signal.
    *   The strength of the signal (e.g., how far the price is from a moving average, or the probability output from a forecasting model).
    *   Agreement between univariate and multivariate model outlooks.
*   **Expected Return**: Based on the forecasts from implemented models for 30, 60, and 90-day horizons. The reliability of these forecasts will be clearly communicated as being indicative and subject to significant uncertainty.

This document will be updated as new analytical methods are added or existing ones are refined.
