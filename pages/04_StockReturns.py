import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np  # Usually implicitly used by pandas, but good to have if needed
import os
import random
import json
from pathlib import Path

###
# set env var to use mock data
# os.environ['MOCK_YF_DATA'] = 'True'
###

# --- Cache Configuration ---
CACHE_DIR = Path("cache/stock_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TODAY = datetime.now().strftime("%d%m%y")

def get_cache_file(ticker):
    """Get the cache file path for a given ticker."""
    return CACHE_DIR / f"{ticker}_{TODAY}.json"

def save_to_cache(ticker, data):
    """Save stock data to cache file."""
    cache_file = get_cache_file(ticker)
    try:
        # Convert data to serializable format
        cache_data = {
            'history': data[0].to_json() if data[0] is not None else None,
            'info': data[1],
            'error': data[2]
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        st.warning(f"Failed to cache data for {ticker}: {str(e)}")

def load_from_cache(ticker):
    """Load stock data from cache file if it exists."""
    cache_file = get_cache_file(ticker)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            # Convert data back to original format
            history = pd.read_json(cache_data['history']) if cache_data['history'] else None
            return history, cache_data['info'], cache_data['error']
        except Exception as e:
            st.warning(f"Failed to load cached data for {ticker}: {str(e)}")
            return None
    return None

# --- Mock Data Generator ---
def generate_mock_data(ticker_symbol, days=730):  # 2 years of data
    """Generate mock stock data for testing purposes."""
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate base price with some randomness
    base_price = random.uniform(50, 500)
    
    # Generate mock data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': []
    }
    
    current_price = base_price
    for _ in range(len(dates)):
        # Generate daily price movement
        daily_change = random.uniform(-0.02, 0.02)  # ±2% daily change
        current_price *= (1 + daily_change)
        
        # Generate OHLC prices
        open_price = current_price * random.uniform(0.99, 1.01)
        high_price = max(open_price, current_price) * random.uniform(1.0, 1.02)
        low_price = min(open_price, current_price) * random.uniform(0.98, 1.0)
        close_price = current_price
        
        data['Open'].append(open_price)
        data['High'].append(high_price)
        data['Low'].append(low_price)
        data['Close'].append(close_price)
        data['Volume'].append(int(random.uniform(1e6, 1e7)))
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Generate mock info
    info = {
        'currentPrice': close_price,
        'regularMarketPrice': close_price,
        'marketCap': random.uniform(1e9, 1e12),  # Random market cap between 1B and 1T
        'regularMarketVolume': int(random.uniform(1e6, 1e7)),
        'trailingPE': random.uniform(10, 30),
        'dividendYield': random.uniform(0, 0.05),
        'beta': random.uniform(0.8, 1.5),
        'exchange': 'NYSE',
        'currency': 'USD',
        'sector': random.choice(['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial']),
        'industry': random.choice(['Software', 'Hardware', 'Biotech', 'Banking', 'Retail']),
        'website': f'https://www.{ticker_symbol.lower()}.com',
        'country': 'United States'
    }
    
    return df, info

# --- Helper Functions ---
def get_stock_data(ticker_symbol):
    """Fetches stock data using yfinance or mock data if MOCK_YF_DATA is set."""
    try:

        if os.getenv('MOCK_YF_DATA'):
            history, info = generate_mock_data(ticker_symbol)
            data = (history, info, None)
            return data

        # Try to load from cache first
        cached_data = load_from_cache(ticker_symbol)
        if cached_data is not None:
            return cached_data
                
        ticker = yf.Ticker(ticker_symbol)
        # Get enough history for 1-year chart + 52-week range + prior day for daily return
        history = ticker.history(period="2y")
        info = ticker.info
        if history.empty:
            data = (None, None, f"No historical data found for {ticker_symbol}. It might be delisted or invalid.")
            save_to_cache(ticker_symbol, data)
            return data
        # Check if essential info is missing (often happens for indices, crypto, etc.)
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            # Use last close from history if market price isn't in info (fallback)
            if not history.empty:
                info["currentPrice"] = history["Close"].iloc[-1]
            else:  # If both info and history are problematic
                data = (None, None, f"Could not retrieve basic info for {ticker_symbol}.")
                save_to_cache(ticker_symbol, data)
                return data
        data = (history, info, None)  # Return history, info, and no error message
        save_to_cache(ticker_symbol, data)
        return data
    except Exception as e:
        st.error(f"An error occurred fetching data for {ticker_symbol}: {e}")
        data = (None, None, str(e))
        return data


def calculate_returns(history_df):
    """Calculates various period returns from historical data."""
    returns = {}
    if len(history_df) < 2:
        return returns  # Not enough data
    today = history_df.index[-1]
    close_prices = history_df["Close"]
    current_price = close_prices.iloc[-1]
    # --- Daily Return ---
    if len(close_prices) >= 2:
        prev_close = close_prices.iloc[-2]
        returns["Daily"] = ((current_price / prev_close) - 1) * 100
    else:
        returns["Daily"] = np.nan
    # --- Time-based Returns ---
    periods = {
        "MTD": today.replace(day=1),
        "3M": today - pd.DateOffset(months=3),
        "6M": today - pd.DateOffset(months=6),
        "YTD": datetime(today.year, 1, 1),
        "1Y": today - pd.DateOffset(years=1),
    }
    # Adjust start dates if they fall on a weekend/holiday (go to previous business day)
    # Use asof for robust lookup even if exact date isn't in index
    for name, start_date in periods.items():
        try:
            # Find the closest available trading day price *on or before* the start date
            start_price = close_prices.asof(start_date)
            if pd.notna(start_price) and start_price > 0:
                returns[name] = ((current_price / start_price) - 1) * 100
            else:
                returns[name] = np.nan  # Cannot calculate if no valid start price found
        except Exception:  # Catch potential errors during lookup
            returns[name] = np.nan
    return returns


def format_market_cap(mc):
    """Formats market cap into readable string (Millions, Billions, Trillions)."""
    if mc is None or not isinstance(mc, (int, float)):
        return "N/A"
    if mc >= 1e12:
        return f"${mc / 1e12:.2f} T"
    elif mc >= 1e9:
        return f"${mc / 1e9:.2f} B"
    elif mc >= 1e6:
        return f"${mc / 1e6:.2f} M"
    else:
        return f"${mc:,.0f}"


# --- Streamlit App ---
st.set_page_config(layout="wide")  # Use wide layout for more space
st.title(":chart_with_upwards_trend: Stock Analysis Dashboard")
# --- Input Area ---
st.sidebar.header("Input")

# Input method selection
input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Manual Entry", "Kite Holdings Upload"]
)

# Store Kite data in session state to access it later
if "kite_data" not in st.session_state:
    st.session_state.kite_data = None

if input_method == "Manual Entry":
    ticker_input = st.sidebar.text_area(
        "Enter Stock Tickers (comma-separated)",
        "AAPL, MSFT, GOOGL, NVDA, ^GSPC",  # Default examples
        height=100,
    )
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()] if ticker_input else []
    st.session_state.kite_data = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload Kite Holdings File", type=['csv'])
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            required_columns = ['Instrument']
            if not all(col in df.columns for col in required_columns):
                st.error("The uploaded file doesn't have the required column: 'Instrument'")
                tickers = []
                st.session_state.kite_data = None
            else:
                # Extract tickers from the Instrument column
                tickers = df['Instrument'].tolist()
                # Store the full dataframe in session state
                st.session_state.kite_data = df
                
                # Display the parsed data
                st.sidebar.success(f"Found {len(tickers)} stocks in the uploaded file")
                with st.sidebar.expander("View Uploaded Data"):
                    st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            tickers = []
            st.session_state.kite_data = None
    else:
        tickers = []
        st.session_state.kite_data = None

analyze_button = st.sidebar.button("Analyze Stocks")

if analyze_button:
    if not tickers:
        st.warning("Please enter ticker symbols or upload a Kite Holdings file.")
    else:
        if os.getenv('MOCK_YF_DATA'):
            st.warning("Using mock data for stock data")

        # Process all stocks first to get sorting data
        stock_data = []
        for ticker_symbol in tickers:
            history, info, error_msg = get_stock_data(ticker_symbol)
            if error_msg or history is None or info is None:
                continue

            returns = calculate_returns(history)
            daily_return = returns.get("Daily", np.nan)
            yearly_return = returns.get("1Y", np.nan)
            current_price = info.get("currentPrice")

            # Calculate position in 52-week range
            history_1y = history[
                history.index > (history.index[-1] - pd.DateOffset(years=1))
            ]
            fifty_two_week_low = history_1y["Low"].min() if not history_1y.empty else np.nan
            fifty_two_week_high = history_1y["High"].max() if not history_1y.empty else np.nan
            
            # Safe calculation of position with null checks
            try:
                if (pd.notna(current_price) and 
                    pd.notna(fifty_two_week_low) and 
                    pd.notna(fifty_two_week_high) and 
                    fifty_two_week_high > fifty_two_week_low):
                    position = (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low) * 100
                else:
                    position = np.nan
            except:
                logging.error(f"Error calculating position for {ticker_symbol}")
                position = np.nan

            # Format range position for display
            range_position_str = f"{position:.0f}%" if pd.notna(position) else "N/A"
            
            # Safe formatting of values
            ticker_str = str(ticker_symbol) if ticker_symbol is not None else "Unknown"
            price_str = f"{current_price:.2f}" if pd.notna(current_price) else "N/A"
            daily_str = f"{daily_return:+.2f}%" if pd.notna(daily_return) else "N/A"
            yearly_str = f"{yearly_return:+.2f}%" if pd.notna(yearly_return) else "N/A"
            
            key_info = (
                f"{ticker_str} | "
                f"Price: {price_str} | "
                f"1D: {daily_str} | "
                f"1Y: {yearly_str} | "
                f"52W: {range_position_str}"
            )
            
            with st.expander(key_info, expanded=False):
                # Show Kite Holdings data if available
                if st.session_state.kite_data is not None and ticker_symbol in st.session_state.kite_data['Instrument'].values:
                    st.subheader("Kite Holdings Data")
                    kite_row = st.session_state.kite_data[st.session_state.kite_data['Instrument'] == ticker_symbol].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    
                    # Format monetary values with commas
                    def format_money(value):
                        try:
                            return f"₹{float(value):,.2f}"
                        except:
                            return value
                    
                    # Column 1: Basic Info
                    with col1:
                        st.markdown("**Holdings Info**")
                        st.write(f"Quantity: {int(kite_row['Qty.'])}")
                        st.write(f"Avg. Cost: {format_money(kite_row['Avg. cost'])}")
                        st.write(f"LTP: {format_money(kite_row['LTP'])}")
                    
                    # Column 2: Investment Details
                    with col2:
                        st.markdown("**Investment Details**")
                        st.write(f"Invested: {format_money(kite_row['Invested'])}")
                        st.write(f"Current Value: {format_money(kite_row['Cur. val'])}")
                    
                    # Column 3: Performance
                    with col3:
                        st.markdown("**Performance**")
                        pnl = float(kite_row['P&L'])
                        pnl_color = "green" if pnl >= 0 else "red"
                        net_chg = float(kite_row['Net chg.'])
                        net_color = "green" if net_chg >= 0 else "red"
                        day_chg = float(kite_row['Day chg.'])
                        day_color = "green" if day_chg >= 0 else "red"
                        
                        st.markdown(f"P&L: <span style='color: {pnl_color}'>{format_money(pnl)}</span>", unsafe_allow_html=True)
                        st.markdown(f"Net Change: <span style='color: {net_color}'>{net_chg:+.2f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"Day Change: <span style='color: {day_color}'>{day_chg:+.2f}%</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")  # Separator between Kite data and YF data
                
                with st.spinner(f"Fetching and processing data for {ticker_symbol}..."):
                    if error_msg:
                        st.error(f"Failed to process {ticker_symbol}: {error_msg}")
                        continue  # Skip to the next ticker
                    if history is None or info is None:
                        st.error(f"Could not retrieve sufficient data for {ticker_symbol}.")
                        continue
                # --- Calculations ---
                current_price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or history["Close"].iloc[-1]
                )  # Robust current price check
                returns = calculate_returns(history)
                # 52-Week Range - use history High/Low over approx last 252 trading days
                history_1y = history[
                    history.index > (history.index[-1] - pd.DateOffset(years=1))
                ]
                fifty_two_week_low = (
                    history_1y["Low"].min() if not history_1y.empty else np.nan
                )
                fifty_two_week_high = (
                    history_1y["High"].max() if not history_1y.empty else np.nan
                )
                # --- Display Data ---
                col1, col2, col3 = st.columns([1.5, 1, 1])  # Adjust column ratios as needed
                with col1:
                    st.subheader("Key Info")
                    daily_delta = returns.get("Daily", np.nan)
                    delta_str = f"{daily_delta:.2f}%" if pd.notna(daily_delta) else None
                    st.metric(
                        label=f"Current Price ({info.get('currency', 'USD')})",
                        value=f"{current_price:.2f}",
                        delta=delta_str,
                    )
                    # Display 52-week range textually
                    if pd.notna(fifty_two_week_low) and pd.notna(fifty_two_week_high):
                        st.write(
                            f"**52-Week Range:** ${fifty_two_week_low:.2f} - ${fifty_two_week_high:.2f}"
                        )
                        # Visual Indicator for 52-Week Range (Progress Bar)
                        if (
                            fifty_two_week_high > fifty_two_week_low
                        ):  # Avoid division by zero
                            position = (current_price - fifty_two_week_low) / (
                                fifty_two_week_high - fifty_two_week_low
                            )
                            position = max(0, min(1, position))  # Clamp between 0 and 1
                            st.progress(position)
                            st.caption(
                                f"Current price is at {position*100:.1f}% of the 52-week range."
                            )
                        else:
                            st.caption(
                                "Cannot visualize range position (low >= high or data missing)."
                            )
                    else:
                        st.write("52-Week Range: N/A")
                    # Other Details
                    st.write(f"**Market Cap:** {format_market_cap(info.get('marketCap'))}")
                    st.write(
                        f"**Volume:** {info.get('regularMarketVolume', info.get('volume', 'N/A')):,.0f}"
                    )  # Try different volume keys
                    st.write(
                        f"**P/E Ratio (TTM):** {info.get('trailingPE', 'N/A'):.2f}"
                        if isinstance(info.get("trailingPE"), float)
                        else "N/A"
                    )
                    div_yield = info.get("dividendYield")
                    st.write(
                        f"**Dividend Yield:** {div_yield*100:.2f}%"
                        if isinstance(div_yield, float)
                        else "N/A"
                    )
                    st.write(
                        f"**Beta:** {info.get('beta', 'N/A'):.2f}"
                        if isinstance(info.get("beta"), float)
                        else "N/A"
                    )
                    st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
                with col2:
                    st.subheader("Performance (%)")
                    perf_labels = ["Daily", "MTD", "3M", "6M", "YTD", "1Y"]
                    for label in perf_labels:
                        value = returns.get(label, np.nan)
                        if pd.notna(value):
                            color = "green" if value >= 0 else "red"
                            st.markdown(f"**{label}:** <span style='color: {color}'>{value:+.2f}%</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{label}:** N/A")
                with col3:
                    st.subheader("About")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                # --- Plotting ---
                st.subheader("1-Year Price Chart")
                if not history_1y.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=history_1y.index,
                            y=history_1y["Close"],
                            mode="lines",
                            name="Close Price",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title=f"Price ({info.get('currency', 'USD')})",
                        margin=dict(l=20, r=20, t=30, b=20),  # Compact margins
                        xaxis_rangeslider_visible=False,  # Hide the default range slider
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker_symbol}")
                else:
                    st.warning(f"Not enough data to plot 1-year chart for {ticker_symbol}.")
elif analyze_button and not ticker_input:
    st.sidebar.warning("Please enter ticker symbols in the text area.")
else:
    st.info("Enter stock tickers in the sidebar and click 'Analyze Stocks'.")
