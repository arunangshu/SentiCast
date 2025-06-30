import streamlit as st
import time
import pandas as pd
from datetime import datetime
import sys
import os
import traceback

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project components
from app.utils.data_fetcher import fetch_crypto_price, fetch_historical_data, SUPPORTED_CRYPTOS, APIError, get_current_usd_to_inr_rate
from app.components.sidebar import render_sidebar
from app.components.charts import price_chart, price_change_gauge, market_stats_metrics, rsi_chart
from app.components.data_analysis import render_data_analysis_tab
from app.components.prediction import render_prediction_tab
from app.components.sentiment_analysis import render_sentiment_analysis_tab

# Set page config
st.set_page_config(
    page_title="SentiCast - Crypto Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
    <style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Only apply Poppins to these specific elements */
    .stTextInput label,
    .stNumberInput label, 
    .stSelectbox label,
    div.stTextInput input,
    div.stNumberInput input,
    textarea,
    div.stMarkdown p,
    div.stMarkdown li,
    div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3, 
    div.stMarkdown h4, div.stMarkdown h5, div.stMarkdown h6,
    div.stText,
    div[role="alert"],
    div.stTitle,
    div.stHeader,
    div.stSubheader,
    .stButton button p,
    div[data-testid="stForm"],
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] span,
    div[data-testid="stMetricDelta"] span,
    table {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Style for headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Style adjustments */
    .main .block-container {padding-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    div[data-testid="stMetricDelta"] {font-size: 0.8rem;}
    
    /* Error message styling */
    .error-message {
        background-color: #ffcdd2;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-family: 'Poppins', sans-serif !important;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

def render_error_message(error_msg):
    """
    Render an error message with proper formatting
    
    Args:
        error_msg (str): The error message to display
    """
    st.markdown(f"""
    <div class="error-message">
    <strong>API Error:</strong><br/>
    {error_msg}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    
    # Initialize session state for auto-refresh control if not exists
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    # Render sidebar and get selected options
    selected_crypto, historical_days, refresh_interval = render_sidebar()
    
    # Get trading symbol from selected cryptocurrency name
    trading_symbol = SUPPORTED_CRYPTOS[selected_crypto]
    
    # Display info message about data source
    st.info("""
    Data sources: Binance API for crypto prices, Exchange Rate API for USD to INR conversion,
    Meta Content Library API for Facebook posts, and Reddit search for social media sentiment analysis.
    Prices are converted from USD to INR using real-time exchange rates.
    """)
    
    # Auto-refresh control buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.session_state.auto_refresh_enabled:
            elapsed = time.time() - st.session_state.last_refresh_time
            remaining = max(0, refresh_interval - elapsed)
            st.caption(f"Auto-refresh every {refresh_interval} seconds. Next refresh in {int(remaining)} seconds.")
        else:
            st.caption(f"Auto-refresh is disabled. Data is static.")
    
    with col2:
        if st.button("Refresh Now"):
            st.session_state.last_refresh_time = time.time()
            st.rerun()
            
    with col3:
        if st.session_state.auto_refresh_enabled:
            if st.button("Stop Auto-Refresh"):
                st.session_state.auto_refresh_enabled = False
        else:
            if st.button("Enable Auto-Refresh"):
                st.session_state.auto_refresh_enabled = True
                st.session_state.last_refresh_time = time.time()
                
    # Check if it's time to auto-refresh
    if st.session_state.auto_refresh_enabled:
        current_time = time.time()
        if current_time - st.session_state.last_refresh_time >= refresh_interval:
            st.session_state.last_refresh_time = current_time
            st.rerun()
    
    # Try to get current USD to INR exchange rate
    try:
        usd_to_inr_rate = get_current_usd_to_inr_rate()
        st.sidebar.metric(
            "USD to INR Rate", 
            f"₹{usd_to_inr_rate:.2f}",
            delta=None
        )
    except APIError as e:
        st.sidebar.warning(f"Could not fetch exchange rate: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Made by Arunangshu Karmakar")
    
    try:
        # Fetch current cryptocurrency data
        crypto_data = fetch_crypto_price(trading_symbol)
        
        # Display cryptocurrency header with image and name
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(crypto_data['image'], width=80)
        with col2:
            st.title(f"{crypto_data['name']} ({crypto_data['symbol']})")
            st.caption(f"Last updated: {datetime.fromisoformat(crypto_data['last_updated'].replace('Z', '+00:00') if 'Z' in crypto_data['last_updated'] else crypto_data['last_updated']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Display key metrics
        st.subheader("Market Overview")
        market_stats_metrics(crypto_data)
        
    except APIError as e:
        st.header(f"{selected_crypto} Market Data")
        render_error_message(str(e))
    
    # Historical data section
    try:
        # Fetch historical data
        historical_data = fetch_historical_data(trading_symbol, days=int(historical_days))
        
        # Create tabs for different charts
        price_tab, rsi_tab, analysis_tab, prediction_tab, sentiment_tab = st.tabs(["Price Chart", "RSI Indicator", "Data Analysis", "Prediction", "Social Media Sentiment"])
        
        # Display price chart in the first tab
        with price_tab:
            st.subheader("Price History")
            if not historical_data.empty:
                fig = price_chart(historical_data, selected_crypto)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display RSI chart in the second tab
        with rsi_tab:
            st.subheader("Relative Strength Index (RSI)")
            st.caption("RSI above 70 indicates overbought conditions, below 30 indicates oversold conditions")
            if not historical_data.empty and len(historical_data) >= 14:
                rsi_fig = rsi_chart(historical_data, selected_crypto)
                if rsi_fig:
                    st.plotly_chart(rsi_fig, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate RSI")
            else:
                st.warning("Not enough historical data to calculate RSI")
                
        # Display Data Analysis tab
        with analysis_tab:
            # Use fixed key prefixes to maintain state between refreshes
            render_data_analysis_tab(historical_data, key_prefix="analysis_fixed")
            
        # Display Prediction tab
        with prediction_tab:
            # Use fixed key prefixes to maintain state between refreshes
            render_prediction_tab(historical_data, key_prefix="prediction_fixed")
            
        # Display Sentiment Analysis tab
        with sentiment_tab:
            # Use fixed key prefixes to maintain state between refreshes
            render_sentiment_analysis_tab(selected_crypto, key_prefix="sentiment_fixed")
        
    except APIError as e:
        st.subheader("Historical Data")
        render_error_message(str(e))
    
    # Price change gauge section
    try:
        if 'price_change_percentage_24h' in crypto_data:
            st.subheader("24h Price Change")
            gauge_fig = price_change_gauge(crypto_data['price_change_percentage_24h'])
            st.plotly_chart(gauge_fig, use_container_width=True)
    except (APIError, KeyError, NameError) as e:
        # Skip this section if crypto_data isn't available
        pass
    
    # Display additional information
    st.subheader("About")
    st.info(f"""
    **SentiCast** provides real-time cryptocurrency data and analysis.
    The platform incorporates social media sentiment data from Facebook and Reddit
    to provide comprehensive market insights.
    
    All cryptocurrency prices are shown in Indian Rupees (INR), converted from USD using current exchange rates.
    """)

if __name__ == "__main__":
    main() 