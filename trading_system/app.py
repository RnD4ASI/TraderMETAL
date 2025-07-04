import streamlit as st

st.set_page_config(
    page_title="Trading Analysis System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Trading Analysis and Recommendation System")

st.sidebar.success("Select a feature above.")

st.markdown(
    """
    Welcome to the Trading Analysis and Recommendation System!

    This application provides tools for fetching financial data, performing analysis,
    backtesting trading strategies, and generating recommendations.

    **Please use the sidebar to navigate to the different features of the application.**

    ### Available Features:
    - **Data Management**: Fetch raw market and macroeconomic data, cleanse it, and merge it for analysis.
    - **Analysis**: Perform multivariate time series analysis and (soon) deep learning forecasts.
    - **Backtester**: Test trading strategies on historical data.
    - **Recommender**: Get trading recommendations based on technical and (soon) fundamental factors.

    **Note on API Keys:**
    - Fetching macroeconomic data requires a FRED API Key. Please ensure you have a `.env` file
      in the root directory of this project with your `FRED_API_KEY`.
      Example `.env` file content:
      ```
      FRED_API_KEY=your_actual_fred_api_key
      ```
    ---
    *This system is for educational and demonstration purposes only. Trading involves risk.*
    """
)

# Instructions on how to run the Streamlit app:
# 1. Make sure all dependencies are installed: pip install -r requirements.txt
# 2. Ensure your FRED_API_KEY is in a .env file in the trading_system directory or project root.
# 3. Open your terminal in the `trading_system` directory.
# 4. Run the command: streamlit run app.py
#
# To create a multi-page app, we will create a directory called `pages`
# in the same directory as `app.py`, and add .py files to it.
# Streamlit automatically picks them up.
