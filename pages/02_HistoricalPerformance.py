import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Page Layout
st.set_page_config(layout="wide", page_title="Stock Performance Dashboard", page_icon="📈")


# Default configurations
DEFAULT_INDICES = {
    "NSE Large Cap": "^NSEI",  # NIFTY 50 Index (Large-cap index for NSE)
    "NSE Mid Cap": "^NSEMDCP50",  # NIFTY Midcap 50 Index
    "BSE SMALLCAP INDEX": "BSE-SMLCAP.BO",  # NIFTY Smallcap 100 Index
    "NASDAQ": "^IXIC",  # NASDAQ Composite Index
    "S&P 500": "^GSPC",  # S&P 500 Index
    "Dow Jones": "^DJI",  # Dow Jones Industrial Average
    "Nvidia": "NVDA",  # Nvidia Corporation
    "Teradata": "TDC",  # Teradata Corporation
    "Gold ETF": "GC=F",  # Gold Futures (COMEX)
    "Bitcoin": "BTC-USD",  # Bitcoin
    "Ethereum": "ETH-USD",  # Ethereum
}
DEFAULT_PERIODS = ["1d", "2d", "5d", "1mo", "3mo", "6mo", "1y", "3y", "5y"]

# App title
st.title("Stock Performance Dashboard")

# Sidebar for user inputs
def get_user_inputs():
    """Get user-selected indices and periods from the sidebar."""
    st.sidebar.header("Configuration")
    
    selected_indices = st.sidebar.multiselect(
        "Select Indices", list(DEFAULT_INDICES.keys()), default=list(DEFAULT_INDICES.keys())
    )
    selected_periods = st.sidebar.multiselect(
        "Select Periods", DEFAULT_PERIODS, default=DEFAULT_PERIODS
    )
    return selected_indices, selected_periods

# Function to fetch historical data
@st.cache_data
def fetch_historical_data(ticker_symbol):
    """Fetch historical data for the given ticker symbol."""
    ticker = yf.Ticker(ticker_symbol)
    return ticker.history(period="max", interval="1d")

# Function to calculate metrics
def calculate_metrics(data, start_date, end_date):
    """Calculate absolute return and XIRR for the given data."""
    try:
        start_price = data.loc[start_date]["Close"]
        end_price = data.loc[end_date]["Close"]
        
        # Absolute return calculation
        absolute_return = (end_price - start_price) / start_price * 100
        
        # CAGR calculation
        days_diff = (end_date - start_date).days / 365  # Convert days to years
        if days_diff < 1:
            cagr = None
        else:
            cagr = ((end_price / start_price) ** (1 / days_diff) - 1) * 100
            
        return f'{round(absolute_return, 2)} %', f'{round(cagr, 2)}%' if cagr else ""
    except Exception:
        return "", ""

# Helper function to map periods to timedelta
def map_period_to_timedelta(period):
    """Map period strings to timedelta objects."""
    if period.endswith("d"):
        return timedelta(days=int(period[:-1]))
    elif period.endswith("mo"):
        return timedelta(days=int(period[:-2]) * 30)
    elif period.endswith("y"):
        return timedelta(days=int(period[:-1]) * 365)
    else:
        raise ValueError(f"Unsupported period format: {period}")

def process_all_indices(selected_indices, selected_periods):
    """Process data for all selected indices and combine into a single table."""
    combined_data = {}

    for index_name in selected_indices:
        ticker_symbol = DEFAULT_INDICES[index_name]
        historical_data = fetch_historical_data(ticker_symbol)
        historical_data.index = pd.to_datetime(historical_data.index)

        # Debug: Check if historical data is fetched
        # st.write(f"Data for {index_name} ({ticker_symbol}):")
        # st.write(historical_data.tail())

        index_data = {}
        for period in selected_periods:
            try:
                end_date = datetime.now()
                start_date = end_date - map_period_to_timedelta(period)

                # Ensure both start_date and historical_data.index are timezone-naive
                start_date = start_date.replace(tzinfo=None)
                end_date = end_date.replace(tzinfo=None)
                historical_data.index = historical_data.index.tz_localize(None)

                # st.write(f"Start date: {start_date}, End date: {end_date}")
                # st.write(f"Historical data index range: {historical_data.index.min()} to {historical_data.index.max()}")

                # Filter data safely
                filtered_data = historical_data[(historical_data.index >= start_date) & (historical_data.index <= end_date)]

                # Debug: Check filtered data
                # st.write(f"Filtered data for {index_name} ({period}):")
                # st.write(filtered_data)

                if not filtered_data.empty:
                    abs_return, cagr = calculate_metrics(filtered_data, filtered_data.index[0], filtered_data.index[-1])
                    cagr_text =  f"[ CAGR: {cagr}]" if cagr else ""
                    index_data[period] = f"{abs_return}\n{cagr_text}" if abs_return else ""
                else:
                    index_data[period] = "N/A"
            except Exception as e:
                index_data[period] = "Error"
        
        combined_data[index_name] = index_data

    # Create DataFrame for display
    # columns = pd.MultiIndex.from_product([selected_periods, ["Absolute Return (%)", "XIRR (%)"]])
    # st.write(combined_data)
    combined_df = pd.DataFrame.from_dict(combined_data, orient='index')#, columns=columns)
    
    # Display the table in Streamlit
    st.table(combined_df)


# Main function to run the app
def main():
    """Main function to run the Streamlit app."""
    # Get user inputs from sidebar
    selected_indices, selected_periods = get_user_inputs()

    # Add a button to trigger table generation
    # if st.button("Generate Table"):
    # Process all selected indices when button is clicked
    process_all_indices(selected_indices, selected_periods)

main()
