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
from io import StringIO

###
# set env var to use mock data
# os.environ['MOCK_YF_DATA'] = 'True'
###

# --- Cache Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache/stock_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TODAY = datetime.now().strftime("%d%m%y")

STOCK_MAPPING_FILE = PROJECT_ROOT / "stock_ticker_mapping.json"


def get_stock_mapping():
    """Get the stock mapping from the stock_ticker_mapping.json file."""
    with open(STOCK_MAPPING_FILE, 'r') as f:
        return json.load(f)

try:
    STOCK_MAPPING = get_stock_mapping()
except Exception as e:
    st.error(f"Failed to load stock mapping: {str(e)}")
    STOCK_MAPPING = {}

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
            # Convert data back to original format using StringIO
            history = pd.read_json(StringIO(cache_data['history'])) if cache_data['history'] else None
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

        ticker_symbol_yf = STOCK_MAPPING.get(ticker_symbol, ticker_symbol)
        ticker = yf.Ticker(ticker_symbol_yf)
        # Get enough history for 1-year chart + 52-week range + prior day for daily return
        history = ticker.history(period="2y")
        info = ticker.info
        if history.empty:
            data = (None, None, f"No historical data found for {ticker_symbol}. It might be delisted or invalid.")
            return data
        # Check if essential info is missing (often happens for indices, crypto, etc.)
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            # Use last close from history if market price isn't in info (fallback)
            if not history.empty:
                info["currentPrice"] = history["Close"].iloc[-1]
            else:  # If both info and history are problematic
                data = (None, None, f"Could not retrieve basic info for {ticker_symbol}.")
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
    
    # Daily Return
    if len(close_prices) >= 2:
        prev_close = close_prices.iloc[-2]
        returns["Daily"] = ((current_price / prev_close) - 1) * 100
    else:
        returns["Daily"] = np.nan
    
    # Time-based Returns
    periods = {
        "MTD": today.replace(day=1),
        "3M": today - pd.DateOffset(months=3),
        "6M": today - pd.DateOffset(months=6),
        "YTD": datetime(today.year, 1, 1),
        "1Y": today - pd.DateOffset(years=1),
    }
    
    for name, start_date in periods.items():
        try:
            start_price = close_prices.asof(start_date)
            if pd.notna(start_price) and start_price > 0:
                returns[name] = ((current_price / start_price) - 1) * 100
            else:
                returns[name] = np.nan
        except Exception:
            returns[name] = np.nan
    
    return returns

def prepare_stock_data(ticker_symbol):
    """Prepare all stock data including calculations and formatting."""
    history, info, error_msg = get_stock_data(ticker_symbol)
    if error_msg or history is None or info is None:
        return None
    
    # Calculate returns
    returns = calculate_returns(history)
    daily_return = returns.get("Daily", np.nan)
    yearly_return = returns.get("1Y", np.nan)
    
    # Get current price
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or history["Close"].iloc[-1]
    
    # Calculate 52-week position
    position, history_1y = calculate_52w_position(history, info)
    
    return {
        'symbol': ticker_symbol,
        'history': history,
        'info': info,
        'returns': returns,
        'current_price': current_price,
        'position': position,
        'daily_return': daily_return,
        'yearly_return': yearly_return,
        'history_1y': history_1y,
        '52w_low': history_1y['Low'].min() if history_1y is not None else None,
        '52w_high': history_1y['High'].max() if history_1y is not None else None,
    }

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


def calculate_52w_position(history, info):
    """Calculate the position in 52-week range."""
    try:
        history_1y = history[history.index > (history.index[-1] - pd.DateOffset(years=1))]
        
        if history_1y.empty:
            logging.warning("No 1-year history data")
            return np.nan, None
        
        fifty_two_week_low = history_1y["Low"].min()
        fifty_two_week_high = history_1y["High"].max()
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or history["Close"].iloc[-1]

        if (pd.notna(current_price) and 
            pd.notna(fifty_two_week_low) and 
            pd.notna(fifty_two_week_high) and 
            fifty_two_week_high > fifty_two_week_low):
            
            position = (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low) * 100
            position = max(0, min(100, position))  # Clamp between 0 and 100
            
            logging.info(f"Position calculation:")
            logging.info(f"Current: {current_price}, Low: {fifty_two_week_low}, High: {fifty_two_week_high}")
            logging.info(f"Position: {position}%")
            return position, history_1y
        else:
            if not pd.notna(current_price):
                logging.warning("Invalid current price")
            if not pd.notna(fifty_two_week_low) or not pd.notna(fifty_two_week_high):
                logging.warning("Invalid 52-week range")
            if fifty_two_week_high <= fifty_two_week_low:
                logging.warning("High <= Low in 52-week range")
            return np.nan, history_1y
    except Exception as e:
        logging.error(f"Error calculating position: {str(e)}")
        return np.nan, None

def format_stock_data(stock):
    """Format stock data for display and download."""
    position = stock['position']
    current_price = stock['current_price']
    daily_return = stock['daily_return']
    yearly_return = stock['yearly_return']
    returns = stock['returns']
    
    return {
        'symbol': stock['symbol'],
        'current_price': f"{current_price:.2f}" if pd.notna(current_price) else "N/A",
        'daily_return': f"{daily_return:+.2f}%" if pd.notna(daily_return) else "N/A",
        'yearly_return': f"{yearly_return:+.2f}%" if pd.notna(yearly_return) else "N/A",
        '52w_position': f"{position:.1f}%" if pd.notna(position) else "N/A",
        '52w_low': f"{stock['52w_low']:.2f}" if stock['52w_low'] is not None else "N/A",
        '52w_high': f"{stock['52w_high']:.2f}" if stock['52w_high'] is not None else "N/A",
        'returns': {k: f"{v:+.2f}%" if pd.notna(v) else "N/A" for k, v in returns.items()},
        # 'info': stock['info']
    }

def serialize_stock_data(stock):
    """Convert stock data into a JSON-serializable format."""
    # Extract numeric values from formatted strings
    def extract_numeric(value):
        if value == "N/A":
            return None
        try:
            # Remove % and convert to float
            return float(value.replace('%', '').replace('+', ''))
        except (ValueError, AttributeError):
            return None

    return {
        'symbol': stock['symbol'],
        'current_price': extract_numeric(stock['current_price']),
        'daily_return': extract_numeric(stock['daily_return']),
        'yearly_return': extract_numeric(stock['yearly_return']),
        '52w_position': extract_numeric(stock['52w_position']),
        '52w_low': extract_numeric(stock['52w_low']),
        '52w_high': extract_numeric(stock['52w_high']),
        'returns': {k: extract_numeric(v) for k, v in stock['returns'].items()},
        # 'info': {
        #     k: (float(v) if isinstance(v, (int, float)) else str(v))
        #     for k, v in stock['info'].items()
        #     if isinstance(v, (int, float, str))
        # }
    }

def create_download_button(download_data):
    """Create download button in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Dashboard Data")
    
    if download_data:
        # Convert stock data to serializable format
        serialized_data = [serialize_stock_data(stock) for stock in download_data]
        json_str = json.dumps(serialized_data, indent=2)
        
        st.sidebar.download_button(
            label="Download Data as JSON",
            data=json_str,
            file_name=f"stock_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
        
        with st.sidebar.expander("Preview Download Data"):
            st.json(serialized_data)
    else:
        st.sidebar.info("Analyze stocks to enable download")

def display_stock_info(stock):
    """Display stock information in expander."""
    ticker_symbol = stock['symbol']
    current_price = stock['current_price']
    daily_return = stock['daily_return']
    yearly_return = stock['yearly_return']
    position = stock['position']
    
    # Format values for display
    range_position_str = f"{position:.0f}%" if pd.notna(position) else "N/A"
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
    
    # Create anchor for this section
    anchor = f"stock_{ticker_symbol}"
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    
    with st.expander(key_info, expanded=True):
        # Add back to summary link
        st.markdown(
            f'<div style="text-align: right; margin-bottom: 1rem;">'
            f'<a href="#summary_table" style="text-decoration: none;">⬆️ Back to Summary</a>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Show Kite Holdings data if available
        if st.session_state.kite_data is not None and ticker_symbol in st.session_state.kite_data['Instrument'].values:
            display_kite_data(ticker_symbol)
        
        display_stock_details(stock)

def format_money(value):
    """Format monetary values with currency symbol and thousands separator."""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"₹{value:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def display_kite_data(ticker_symbol):
    """Display Kite Holdings data."""
    st.subheader("Kite Holdings Data")
    kite_row = st.session_state.kite_data[st.session_state.kite_data['Instrument'] == ticker_symbol].iloc[0]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Holdings Info**")
        st.write(f"Quantity: {int(kite_row['Qty.'])}")
        st.write(f"Avg. Cost: {format_money(kite_row['Avg. cost'])}")
        st.write(f"LTP: {format_money(kite_row['LTP'])}")
    
    with col2:
        st.markdown("**Investment Details**")
        st.write(f"Invested: {format_money(kite_row['Invested'])}")
        st.write(f"Current Value: {format_money(kite_row['Cur. val'])}")
    
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
    
    st.markdown("---")

def display_stock_details(stock):
    """Display detailed stock information."""
    info = stock['info']
    returns = stock['returns']
    history_1y = stock['history_1y']
    
    col1, col2, col3 = st.columns([1.5, 1, 1])
    
    with col1:
        st.subheader("Key Info")
        daily_delta = returns.get("Daily", np.nan)
        delta_str = f"{daily_delta:.2f}%" if pd.notna(daily_delta) else None
        
        # Safe formatting of current price
        current_price = stock['current_price']
        price_str = f"{current_price:.2f}" if pd.notna(current_price) else "N/A"
        
        st.metric(
            label=f"Current Price ({info.get('currency', 'USD')})",
            value=price_str,
            delta=delta_str,
        )
        
        # Display 52-week range
        if history_1y is not None:
            fifty_two_week_low = history_1y["Low"].min()
            fifty_two_week_high = history_1y["High"].max()
            if pd.notna(fifty_two_week_low) and pd.notna(fifty_two_week_high):
                st.write(f"**52-Week Range:** ${fifty_two_week_low:.2f} - ${fifty_two_week_high:.2f}")
                
                # Add progress bar for 52-week range position
                if fifty_two_week_high > fifty_two_week_low:
                    if pd.notna(current_price):
                        position = (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
                        position = max(0, min(1, position))  # Clamp between 0 and 1
                        st.progress(position)
                        st.caption(f"Current price is at {position*100:.1f}% of the 52-week range.")
                    else:
                        st.caption("Current price not available for range position.")
                else:
                    st.caption("Cannot visualize range position (low >= high).")
            else:
                st.write("52-Week Range: N/A")
        
        # Other details
        st.write(f"**Market Cap:** {format_market_cap(info.get('marketCap'))}")
        volume = info.get('regularMarketVolume', info.get('volume'))
        volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "N/A"
        st.write(f"**Volume:** {volume_str}")
        
        pe_ratio = info.get('trailingPE')
        st.write(f"**P/E Ratio (TTM):** {pe_ratio:.2f}" if isinstance(pe_ratio, float) else "N/A")
        
        div_yield = info.get('dividendYield')
        st.write(f"**Dividend Yield:** {div_yield*100:.2f}%" if isinstance(div_yield, float) else "N/A")
        
        beta = info.get('beta')
        st.write(f"**Beta:** {beta:.2f}" if isinstance(beta, float) else "N/A")
        
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
    
    # Plotting
    if history_1y is not None and not history_1y.empty:
        st.subheader("1-Year Price Chart")
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
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{stock['symbol']}")

def create_stock_summary_table(stock_data):
    """Create a summary table with links to detailed sections."""
    if not stock_data:
        return
    
    st.markdown("### Stock Summary")
    
    # Create anchor for the summary table
    st.markdown('<div id="summary_table"></div>', unsafe_allow_html=True)
    
    # Create table data
    table_data = []
    for stock in stock_data:
        ticker = stock['symbol']
        current_price = stock['current_price']
        daily_return = stock['daily_return']
        position = stock['position']
        
        # Format values
        price_str = f"{current_price:.2f}" if pd.notna(current_price) else "N/A"
        daily_str = f"{daily_return:+.2f}%" if pd.notna(daily_return) else "N/A"
        position_str = f"{position:.1f}%" if pd.notna(position) else "N/A"
        
        # Create link to details
        details_link = f'<a href="#stock_{ticker}" style="text-decoration: none;">🔍 Details</a>'
        
        table_data.append({
            'Symbol': ticker,
            'Price': price_str,
            'Daily Return': daily_str,
            '52W Position': position_str,
            'Details': details_link
        })
    
    # Convert to DataFrame and display
    df = pd.DataFrame(table_data)
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("---")

# --- Main Application ---
def main():
    st.set_page_config(layout="wide")
    st.title(":chart_with_upwards_trend: Stock Analysis Dashboard")
    
    # Initialize session state
    if "kite_data" not in st.session_state:
        st.session_state.kite_data = None
    
    # Sidebar input
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Manual Entry", "Kite Holdings Upload"]
    )
    
    tickers = []
    if input_method == "Manual Entry":
        ticker_input = st.sidebar.text_area(
            "Enter Stock Tickers (comma-separated)",
            "AAPL, MSFT, GOOGL, NVDA, ^GSPC",
            height=100,
        )
        tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
        st.session_state.kite_data = None
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Kite Holdings File", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Instrument' not in df.columns:
                st.error("The uploaded file doesn't have the required column: 'Instrument'")
                tickers = []
            else:
                tickers = df['Instrument'].tolist()
                st.session_state.kite_data = df
                st.sidebar.success(f"Found {len(tickers)} stocks in the uploaded file")
                with st.sidebar.expander("View Uploaded Data"):
                    st.dataframe(df)
    
    analyze_button = st.sidebar.button("Analyze Stocks")
    
    if analyze_button:
        if not tickers:
            st.warning("Please enter ticker symbols or upload a Kite Holdings file.")
        
        if os.getenv('MOCK_YF_DATA'):
            st.warning("Using mock data for stock data")
        
        # add progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process all stocks
        stock_data = []
        for i, ticker_symbol in enumerate(tickers):
            stock = prepare_stock_data(ticker_symbol)
            if stock is not None:
                stock_data.append(stock)
            progress_bar.progress((i + 1) / len(tickers))
            progress_text.text(f"Processing {ticker_symbol} ({i + 1}/{len(tickers)})")
        
        # clear progress bar
        progress_bar.empty()
        progress_text.empty()
        
        # Sort stocks by position
        stock_data.sort(key=lambda x: float('-inf') if pd.isna(x['position']) else x['position'], reverse=True)
        
        # Create summary table
        create_stock_summary_table(stock_data)
        
        # Display detailed information for each stock
        st.markdown("### Stock Details")
        for stock in stock_data:
            display_stock_info(stock)
        
        # Prepare download data
        download_data = [format_stock_data(stock) for stock in stock_data]
        create_download_button(download_data)

if __name__ == "__main__":
    main()
