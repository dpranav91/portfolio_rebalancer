# Portfolio Rebalancer

This project is a Streamlit application for rebalancing investment portfolios. It allows users to input their existing balances, target allocations, and total investable amount to calculate the necessary adjustments to achieve the desired portfolio balance.

## Features

- Input existing balances for different asset classes.
- Set target allocation percentages for each asset class.
- Calculate the additional investment needed to achieve the target allocations.
- Display the rebalanced portfolio and recommendations for additional investments.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/portfolio_rebalancer.git
    cd portfolio_rebalancer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

3. Use the sidebar to input your total investable amount, existing balances, and target allocation percentages.

4. View the rebalanced portfolio and recommendations for additional investments in the main panel.

