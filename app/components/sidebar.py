import streamlit as st
from app.utils.data_fetcher import SUPPORTED_CRYPTOS
from datetime import timedelta

def render_sidebar():
    """
    Render the sidebar with controls and information
    
    Returns:
        tuple: Selected cryptocurrency, historical data period in days, update interval
    """
    st.sidebar.title("SentiCast")
    st.sidebar.image("https://img.icons8.com/fluency/96/cryptocurrency.png", width=80)
    st.sidebar.markdown("### Real-time Cryptocurrency Analytics")
    
    # Cryptocurrency selection
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        list(SUPPORTED_CRYPTOS.keys()),
        index=0
    )
    
    # Historical data period selection
    st.sidebar.subheader("Historical Data Period")
    
    # Set max values based on 5 years maximum
    max_hours = 5 * 365 * 24  # 5 years in hours
    max_days = 5 * 365        # 5 years in days
    max_months = 5 * 12       # 5 years in months
    max_years = 5             # 5 years
    
    # Create the selectbox for period unit first
    period_unit = st.sidebar.selectbox(
        "Unit",
        ["hours", "days", "months", "years"],
        index=1,  # Default to days
        key="sidebar_period_unit"
    )
    
    # Determine max value based on selected unit
    if period_unit == "hours":
        max_value = max_hours
        default_value = min(24, max_value)  # Default to 24 hours
    elif period_unit == "days":
        max_value = max_days
        default_value = min(7, max_value)   # Default to 7 days
    elif period_unit == "months":
        max_value = max_months
        default_value = min(1, max_value)   # Default to 1 month
    else:  # years
        max_value = max_years
        default_value = min(1, max_value)   # Default to 1 year
    
    period_value = st.sidebar.number_input(
        "Period",
        min_value=1,
        max_value=max_value,
        value=default_value,
        step=1,
        key="sidebar_period_value"
    )
    
    # Calculate historical days based on selected period and unit
    if period_unit == "hours":
        historical_days = period_value / 24  # Convert hours to days
    elif period_unit == "days":
        historical_days = period_value
    elif period_unit == "months":
        historical_days = period_value * 30  # Approximate months as 30 days
    else:  # years
        historical_days = period_value * 365  # Approximate years as 365 days
    
    # Display the equivalent in days for clarity
    if period_unit != "days":
        st.sidebar.caption(f"Equivalent to approximately {historical_days:.1f} days")
    
    # Refresh interval selection
    refresh_interval = st.sidebar.select_slider(
        "Refresh Interval",
        options=[10, 30, 60, 300],
        value=60,
        format_func=lambda x: f"{x} seconds"
    )
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **SentiCast** - Cryptocurrency Analysis Platform
    
    Features:
    - Real-time price tracking in INR
    - Interactive historical charts
    - Price prediction using time series models
    - Reddit posts sentiment analysis
    - News articles from NewsAPI
    - Advanced data analysis tools
    """)
    
    return selected_crypto, historical_days, refresh_interval 