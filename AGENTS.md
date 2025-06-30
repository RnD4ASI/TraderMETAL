## Agent Guidelines for Trading Analysis System

This document provides guidelines for AI agents working on this project.

### Project Goal
Develop a trading analysis and recommendation system for Gold, Silver, and Platinum, featuring data crawling, cleansing, analysis, forecasting, backtesting, and recommendation capabilities. The system should be professionally designed, scalable, and usable by non-programmers.

### Key Development Principles
1.  **Modularity**: Keep components (data crawling, cleansing, analysis, etc.) in separate, well-defined modules.
2.  **Clarity**: Write clear, well-documented code. Explain complex logic and algorithms.
3.  **Configuration**: Use configuration files or environment variables for sensitive information (like API keys) and tunable parameters.
4.  **Error Handling**: Implement robust error handling and provide informative error messages to the user.
5.  **Testing**: Write unit tests for key functionalities. Aim for good test coverage.
6.  **User-Friendliness**: The final application should have a clear command-line interface (CLI) with straightforward instructions.
7.  **Documentation**:
    *   Maintain `README.md` with setup and usage instructions.
    *   Document the analysis methodologies in `ANALYSIS_METHODOLOGY.md`.
    *   Ensure code comments are thorough.
8.  **Scalability**: Design with future enhancements in mind. For example, make it easy to add new analytical methods or support additional assets.
9.  **Data Handling**:
    *   Fetched data should be stored in a structured way (e.g., CSV files in the `data/` directory).
    *   Ensure data integrity and consistency.
10. **Dependencies**: Keep `requirements.txt` up-to-date.

### Specific Instructions for Modules

*   **`data_crawler.py`**:
    *   Must allow users to specify a time window (max 10 years) and frequency (daily/weekly).
    *   Use reliable data sources. `yfinance` is a good starting point for public ETFs/futures.
    *   Handle potential API errors gracefully.
*   **`data_cleanser.py`**:
    *   Implement common time-series cleansing techniques (handling NaNs, aligning dates).
    *   Ensure data is suitable for analysis.
*   **`analyzer.py`**:
    *   Document all analytical methods used in `ANALYSIS_METHODOLOGY.md`.
    *   Be creative in analyzing relationships between metals.
    *   For statistical tests (e.g., ADF, Johansen), ensure the output is user-friendly, clearly explaining the null hypothesis, test statistics, p-values (if applicable), critical values, and the conclusion drawn from the test. The goal is to make complex statistical output understandable to a non-expert user.
*   **`backtester.py`**:
    *   Provide clear metrics for backtest performance.
    *   Make backtesting an optional feature for the user.
*   **`recommender.py`**:
    *   Clearly explain how recommendations are derived.
    *   Provide confidence levels and expected returns (30, 60, 90 days) if feasible and statistically sound.

### Code Style
*   Follow PEP 8 for Python code.
*   Use meaningful variable and function names.

### Commits and Pull Requests
*   Write clear and concise commit messages.
*   If applicable, ensure PRs are reviewed and pass any CI checks.

By following these guidelines, we can build a robust and useful trading analysis system.
