import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def price_chart(historical_data, crypto_name):
    """
    Create an interactive price chart using Plotly
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical price data
        crypto_name (str): Name of the cryptocurrency
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create candlestick chart if we have OHLC data
    if len(historical_data) >= 2:
        fig = go.Figure()
        
        # Add line chart
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['price'],
            mode='lines',
            name=f'{crypto_name} Price',
            line=dict(color='#2962ff', width=2)
        ))
        
        # Calculate moving average if we have enough data points
        if len(historical_data) >= 7:
            historical_data['MA7'] = historical_data['price'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['MA7'],
                mode='lines',
                name='7-period MA',
                line=dict(color='#ff6d00', width=2, dash='dash')
            ))
        
        # Add RSI if we have enough data points
        if len(historical_data) >= 14:
            historical_data = add_rsi(historical_data, window=14)
        
        # Layout configuration
        fig.update_layout(
            title=f'{crypto_name} Price Chart (INR)',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    else:
        # If no historical data is available
        st.warning(f"Not enough historical data available for {crypto_name}")
        return None

def add_rsi(data, window=14):
    """
    Add Relative Strength Index to dataframe
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window (int): RSI window period
        
    Returns:
        pd.DataFrame: DataFrame with RSI added
    """
    delta = data['price'].diff()
    
    # Make two series: one for gains and one for losses
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Calculate rolling averages
    avg_gain = up.rolling(window=window, min_periods=1).mean()
    avg_loss = down.rolling(window=window, min_periods=1).mean()
    
    # Calculate RS based on average gains and losses
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def price_change_gauge(price_change_24h):
    """
    Create a gauge chart for 24h price change percentage
    
    Args:
        price_change_24h (float): 24h price change percentage
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set color based on value
    if price_change_24h >= 0:
        color = "#00c853"  # Green for positive
    else:
        color = "#d50000"  # Red for negative
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price_change_24h,
        title={'text': "24h Change (%)"},
        gauge={
            'axis': {'range': [-20, 20]},
            'bar': {'color': color},
            'steps': [
                {'range': [-20, -10], 'color': "rgba(213, 0, 0, 0.3)"},
                {'range': [-10, 0], 'color': "rgba(213, 0, 0, 0.15)"},
                {'range': [0, 10], 'color': "rgba(0, 200, 83, 0.15)"},
                {'range': [10, 20], 'color': "rgba(0, 200, 83, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': price_change_24h
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    
    return fig

def rsi_chart(historical_data, crypto_name):
    """
    Create an RSI chart
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical data including RSI
        crypto_name (str): Name of the cryptocurrency
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if 'RSI' in historical_data.columns and len(historical_data) > 14:
        # Filter out NaN values
        df = historical_data.dropna(subset=['RSI'])
        
        # Create RSI chart
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#673ab7', width=2)
        ))
        
        # Add overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=df['timestamp'].iloc[0],
            y0=70,
            x1=df['timestamp'].iloc[-1],
            y1=70,
            line=dict(color='red', dash='dash'),
            name='Overbought'
        )
        
        fig.add_shape(
            type='line',
            x0=df['timestamp'].iloc[0],
            y0=30,
            x1=df['timestamp'].iloc[-1],
            y1=30,
            line=dict(color='green', dash='dash'),
            name='Oversold'
        )
        
        # Add midline
        fig.add_shape(
            type='line',
            x0=df['timestamp'].iloc[0],
            y0=50,
            x1=df['timestamp'].iloc[-1],
            y1=50,
            line=dict(color='gray', dash='dot'),
            name='Midline'
        )
        
        # Layout configuration
        fig.update_layout(
            title=f'{crypto_name} RSI (14)',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    else:
        return None

def volume_bar_chart(historical_data, crypto_name):
    """
    Create a bar chart for trading volume
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical data
        crypto_name (str): Name of the cryptocurrency
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if 'volume' in historical_data.columns and len(historical_data) > 0:
        # Create bar chart for volume
        fig = px.bar(
            historical_data,
            x='timestamp',
            y='volume',
            title=f'{crypto_name} Trading Volume',
            labels={'timestamp': 'Date', 'volume': 'Volume (INR)'},
            color_discrete_sequence=['#673ab7']
        )
        
        # Layout configuration
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    else:
        return None

def market_stats_metrics(crypto_data):
    """
    Display key market statistics as metrics
    
    Args:
        crypto_data (dict): Dictionary containing cryptocurrency market data
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price (INR)", 
            f"₹{crypto_data['current_price_inr']:,.2f}", 
            f"{crypto_data['price_change_percentage_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            "24h High (INR)", 
            f"₹{crypto_data['high_24h_inr']:,.2f}"
        )
    
    with col3:
        st.metric(
            "24h Low (INR)", 
            f"₹{crypto_data['low_24h_inr']:,.2f}"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Market Cap (INR)", 
            f"₹{crypto_data['market_cap_inr']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Trading Volume (INR)", 
            f"₹{crypto_data['total_volume_inr']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Market Cap Rank", 
            f"#{crypto_data['market_cap_rank']}"
        ) 