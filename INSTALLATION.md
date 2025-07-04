# Installation and Setup Guide

This guide provides step-by-step instructions to install, configure, and run the Trading Analysis and Recommendation System.

## 1. Prerequisites

*   **Python**: Version 3.8 or higher. You can download Python from [python.org](https://www.python.org/downloads/).
*   **Git**: Required for cloning the repository. You can download Git from [git-scm.com](https://git-scm.com/downloads).
*   **pip**: Python's package installer, usually comes with Python. Ensure it's up to date (`python -m pip install --upgrade pip`).

## 2. Clone the Repository

Open your terminal or command prompt and run the following command to clone the project repository:

```bash
git clone <repository_url>
cd <repository_directory_name> # e.g., cd trading-analysis-system
```
Replace `<repository_url>` with the actual URL of the Git repository and `<repository_directory_name>` with the name of the folder created by the clone command.

## 3. Create a Virtual Environment (Recommended)

Using a virtual environment helps manage project dependencies and avoids conflicts with other Python projects.

Navigate into the project's root directory (e.g., `trading-analysis-system`) and run:

```bash
# For macOS and Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

## 4. Install Dependencies

With the virtual environment active, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This command will download and install all necessary libraries, including Streamlit, Pandas, TensorFlow (for Deep Learning features), etc.

## 5. Set Up FRED API Key (Required for Macroeconomic Data)

The system uses the Federal Reserve Economic Data (FRED) service to fetch macroeconomic indicators. You need a FRED API key for this functionality.

*   **Get a FRED API Key**:
    1.  Go to the [FRED API Keys page](https://fred.stlouisfed.org/docs/api/api_key.html).
    2.  Request an API key if you don't have one. It's free.
*   **Configure the API Key**:
    1.  In the root directory of the project (e.g., `trading-analysis-system`), create a file named `.env`.
    2.  Add your FRED API key to this file in the following format:

        ```
        FRED_API_KEY=your_actual_fred_api_key
        ```
        Replace `your_actual_fred_api_key` with the key you obtained from FRED.

    *Note: The `.env` file should be in the project's root directory, which is the parent of the `trading_system` directory.*

## 6. Running the Application

The application has a primary web-based interface powered by Streamlit.

### Using the Streamlit Web Interface (Recommended)

1.  Ensure your virtual environment is active.
2.  Navigate to the `trading_system` directory within the project:
    ```bash
    cd trading_system
    ```
3.  Run the Streamlit application using the following command:
    ```bash
    streamlit run app.py
    ```
4.  Streamlit will typically open the application in your default web browser automatically. If not, it will display a local URL (e.g., `http://localhost:8501`) that you can open in your browser.
5.  Use the sidebar in the web application to navigate through different features: Data Management, Analysis, Backtester, and Recommender.

### Using the Command-Line Interface (CLI - Alternative)

The system also offers a command-line interface for accessing backend functionalities directly. This might be useful for scripting or advanced users.

1.  Ensure your virtual environment is active.
2.  Navigate to the `trading_system` directory:
    ```bash
    cd trading_system
    ```
3.  You can see available commands by running:
    ```bash
    python main.py --help
    ```
4.  To run a specific command, for example, to fetch metal price data:
    ```bash
    python main.py fetch-data
    ```
    You will be prompted for necessary inputs like dates and frequency. Other commands include `clean-data`, `analyze-mv`, `fetch-macro-data`, etc.

## 7. Troubleshooting

*   **TensorFlow Issues**: If you encounter problems related to TensorFlow (especially on systems with specific CPU/GPU configurations), refer to the official TensorFlow installation guide for more detailed instructions specific to your OS and hardware.
*   **`ModuleNotFoundError`**: Ensure your virtual environment is active and that you ran `pip install -r requirements.txt` from the project's root directory. If running CLI commands, ensure you are in the `trading_system` directory.
*   **FRED API Key Error**: If you see errors related to the FRED API key, double-check that the `.env` file is correctly named, located in the project root, and contains the valid key in the specified format (`FRED_API_KEY=...`).

You should now be all set to use the Trading Analysis and Recommendation System!
