import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np  # Usually implicitly used by pandas, but good to have if needed
import os
import random

###
# set env var to use mock data
os.environ['MOCK_YF_DATA'] = 'True'
###

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
            return history, info, None  # Return None for error_msg when using mock data
        
        ticker = yf.Ticker(ticker_symbol)
        # Get enough history for 1-year chart + 52-week range + prior day for daily return
        history = ticker.history(period="2y")
        info = ticker.info
        if history.empty:
            return (
                None,
                None,
                f"No historical data found for {ticker_symbol}. It might be delisted or invalid.",
            )
        # Check if essential info is missing (often happens for indices, crypto, etc.)
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            # Use last close from history if market price isn't in info (fallback)
            if not history.empty:
                info["currentPrice"] = history["Close"].iloc[-1]
            else:  # If both info and history are problematic
                return None, None, f"Could not retrieve basic info for {ticker_symbol}."
        return history, info, None  # Return history, info, and no error message
    except Exception as e:
        st.error(f"An error occurred fetching data for {ticker_symbol}: {e}")
        return None, None, str(e)


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
ticker_input = st.sidebar.text_area(
    "Enter Stock Tickers (comma-separated)",
    "AAPL, MSFT, GOOGL, NVDA, ^GSPC",  # Default examples
    height=100,
)
analyze_button = st.sidebar.button("Analyze Stocks")
if analyze_button and ticker_input:
    if os.getenv('MOCK_YF_DATA'):
        st.warning("Using mock data for stock data")
    tickers = [
        ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()
    ]
    if not tickers:
        st.warning("Please enter at least one valid ticker symbol.")
    else:
        st.success(f"Analyzing: {', '.join(tickers)}")
        for ticker_symbol in tickers:
            history, info, error_msg = get_stock_data(ticker_symbol)
            # show key info (stock name, current price, daily return, yearly return)
            returns = calculate_returns(history)
            daily_return = returns.get("Daily", np.nan)
            yearly_return = returns.get("1Y", np.nan)
            current_price = info.get("currentPrice")
            
            key_info = f"{ticker_symbol} | Current Price: ${current_price:.2f} | Daily Return: {daily_return:+.2f}% | Yearly Return: {yearly_return:+.2f}%"
            with st.expander(key_info, expanded=False):
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data to plot 1-year chart for {ticker_symbol}.")
elif analyze_button and not ticker_input:
    st.sidebar.warning("Please enter ticker symbols in the text area.")
else:
    st.info("Enter stock tickers in the sidebar and click 'Analyze Stocks'.")
