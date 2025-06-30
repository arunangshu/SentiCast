import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def centered_moving_average(data, window_size):
    """
    Calculate centered moving average
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        window_size (int): Moving average window size (must be odd)
        
    Returns:
        pd.Series: Series with centered moving average values
    """
    # Ensure data exists
    if data.empty or 'price' not in data.columns:
        return pd.Series()
    
    # Handle NaN values in the input
    price_data = data['price'].copy()
    if price_data.isnull().any():
        # Fill small gaps with linear interpolation
        price_data = price_data.interpolate(method='linear', limit=3)
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate centered moving average
    ma = price_data.rolling(window=window_size, center=True).mean()
    
    # For debugging
    nan_count = ma.isnull().sum()
    total_count = len(ma)
    if nan_count > 0:
        print(f"Moving average has {nan_count}/{total_count} NaN values ({nan_count/total_count:.1%})")
    
    return ma

def downsample_data(data, factor):
    """
    Downsample data by taking every nth value
    
    Args:
        data (pd.Series): Series to downsample
        factor (int): Downsampling factor
        
    Returns:
        pd.Series: Downsampled series
    """
    if data is None or data.empty:
        return pd.Series()
    
    # Ensure factor is at least 1
    factor = max(1, factor)
    
    # Handle NaN values
    if data.isnull().any():
        # Option 1: Skip NaN values when downsampling
        valid_indices = data.dropna().index
        if len(valid_indices) > 0:
            # Take every nth valid index
            downsampled_indices = valid_indices[::factor]
            return data.loc[downsampled_indices]
    
    # Standard downsampling if no NaN issues
    return data.iloc[::factor]

def check_stationarity(data, significance=0.05):
    """
    Perform ADF test to check stationarity
    
    Args:
        data (pd.Series): Series to test
        significance (float): Significance level
        
    Returns:
        tuple: (is_stationary, adf_result, p_value)
    """
    # Drop NA values
    data = data.dropna()
    
    # Skip test if too few data points
    if len(data) < 20:
        return False, None, 1.0
        
    # Perform ADF test
    adf_test = adfuller(data, autolag='AIC')
    p_value = adf_test[1]
    
    # Determine if stationary
    is_stationary = p_value < significance
    
    return is_stationary, adf_test, p_value

def plot_adf_results(adf_result, p_value, significance=0.05):
    """
    Create a visual representation of ADF test results
    
    Args:
        adf_result: ADF test result
        p_value (float): p-value from ADF test
        significance (float): Significance level
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with ADF results
    """
    if adf_result is None:
        return go.Figure()
        
    # Create figure
    fig = go.Figure()
    
    # Add test statistic
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=p_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ADF Test p-value"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue" if p_value < significance else "red"},
            'steps': [
                {'range': [0, significance], 'color': "rgba(0, 200, 83, 0.3)"},
                {'range': [significance, 1], 'color': "rgba(213, 0, 0, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': p_value
            }
        },
        delta={'reference': significance, 'decreasing': {'color': 'green'}, 'increasing': {'color': 'red'}}
    ))
    
    # Create a more detailed results table as annotation
    critical_values = adf_result[4]
    test_statistic = adf_result[0]
    
    annotation_text = (
        f"<b>Test Statistic:</b> {test_statistic:.4f}<br>"
        f"<b>p-value:</b> {p_value:.4f}<br><br>"
        f"<b>Critical Values:</b><br>"
        f"1%: {critical_values['1%']:.4f}<br>"
        f"5%: {critical_values['5%']:.4f}<br>"
        f"10%: {critical_values['10%']:.4f}<br><br>"
        f"<b>Result:</b> {'Stationary (reject H0)' if p_value < significance else 'Non-stationary (fail to reject H0)'}"
    )
    
    fig.add_annotation(
        x=0.5,
        y=0.4,
        text=annotation_text,
        showarrow=False,
        align="center"
    )
    
    # Layout configuration
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def difference_series(data, order=1):
    """
    Apply differencing to make a series stationary
    
    Args:
        data (pd.Series): Series to difference
        order (int): Differencing order
        
    Returns:
        pd.Series: Differenced series
    """
    return data.diff(order).dropna()

def plot_time_series(original_series, processed_series, title="Time Series"):
    """
    Plot original and processed time series
    
    Args:
        original_series (pd.Series): Original time series
        processed_series (pd.Series): Processed time series
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    fig = go.Figure()
    
    if not original_series.empty:
        fig.add_trace(go.Scatter(
            x=original_series.index,
            y=original_series,
            mode='lines',
            name='Original',
            line=dict(color='#2962ff', width=1.5),
            opacity=0.3
        ))
    
    if not processed_series.empty:
        fig.add_trace(go.Scatter(
            x=processed_series.index,
            y=processed_series,
            mode='lines',
            name='Processed',
            line=dict(color='#ff6d00', width=2)
        ))
    
    # Layout configuration
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
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

def compute_acf_pacf(data, max_lags=40):
    """
    Compute ACF and PACF values
    
    Args:
        data (pd.Series): Stationary time series
        max_lags (int): Maximum number of lags
        
    Returns:
        tuple: (acf_values, pacf_values, confidence_intervals)
    """
    # Ensure data is finite and has no NaN values
    data = data.dropna()
    
    # Display debug info
    st.text(f"ACF/PACF calculation: {len(data)} valid data points")
    
    # Safety check for very small datasets
    if len(data) <= 10:
        st.error("Too few data points for ACF/PACF calculation")
        # Return minimal placeholder values
        return np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), 0.95
    
    # Adjust max_lags if data is too small (use at most 1/3 of data length)
    max_lags = min(max_lags, int(len(data) / 3))
    
    # Ensure max_lags is at least 5
    max_lags = max(5, max_lags)
    
    st.text(f"Using {max_lags} lags for ACF/PACF calculation")
    
    # Calculate ACF safely with FFT method (more robust)
    try:
        acf_values = acf(data, nlags=max_lags, fft=True)
    except Exception as e:
        # Fallback for ACF calculation
        st.warning(f"Error in ACF calculation: {str(e)}")
        acf_values = np.zeros(max_lags+1)
        acf_values[0] = 1.0
    
    # Try multiple methods for PACF calculation
    try:
        # Try with YW method first (more stable than OLS)
        pacf_values = pacf(data, nlags=max_lags, method='yw')
    except Exception as e1:
        try:
            # Fall back to LD method
            st.warning(f"YW method failed: {str(e1)}, trying LD method")
            pacf_values = pacf(data, nlags=max_lags, method='ld')
        except Exception as e2:
            try:
                # Last resort: use OLS method with smaller lag
                st.warning(f"LD method failed: {str(e2)}, trying OLS with reduced lags")
                pacf_values = pacf(data, nlags=min(max_lags, 10), method='ols')
            except Exception as e3:
                # If all else fails, create placeholder
                st.error(f"All PACF methods failed: {str(e3)}")
                pacf_values = np.zeros(max_lags+1)
                pacf_values[0] = 1.0
    
    # Calculate 95% confidence intervals
    conf_level = 1.96 / np.sqrt(len(data))
    
    return acf_values, pacf_values, conf_level

def plot_acf_pacf(acf_values, pacf_values, conf_level):
    """
    Create interactive ACF and PACF plots
    
    Args:
        acf_values (np.array): ACF values
        pacf_values (np.array): PACF values
        conf_level (float): Confidence level bounds
        
    Returns:
        tuple: (acf_fig, pacf_fig) - Plotly figures for ACF and PACF
    """
    # Create ACF plot
    acf_fig = go.Figure()
    
    # Add bars
    acf_fig.add_trace(go.Bar(
        x=list(range(len(acf_values))),
        y=acf_values,
        name='ACF',
        marker_color='blue'
    ))
    
    # Add confidence intervals
    acf_fig.add_shape(
        type='line',
        x0=0,
        y0=conf_level,
        x1=len(acf_values) - 1,
        y1=conf_level,
        line=dict(color='red', dash='dash'),
    )
    
    acf_fig.add_shape(
        type='line',
        x0=0,
        y0=-conf_level,
        x1=len(acf_values) - 1,
        y1=-conf_level,
        line=dict(color='red', dash='dash'),
    )
    
    acf_fig.update_layout(
        title='Autocorrelation Function (ACF)',
        xaxis_title='Lag',
        yaxis_title='Correlation',
        template='plotly_white'
    )
    
    # Create PACF plot
    pacf_fig = go.Figure()
    
    # Add bars
    pacf_fig.add_trace(go.Bar(
        x=list(range(len(pacf_values))),
        y=pacf_values,
        name='PACF',
        marker_color='green'
    ))
    
    # Add confidence intervals
    pacf_fig.add_shape(
        type='line',
        x0=0,
        y0=conf_level,
        x1=len(pacf_values) - 1,
        y1=conf_level,
        line=dict(color='red', dash='dash'),
    )
    
    pacf_fig.add_shape(
        type='line',
        x0=0,
        y0=-conf_level,
        x1=len(pacf_values) - 1,
        y1=-conf_level,
        line=dict(color='red', dash='dash'),
    )
    
    pacf_fig.update_layout(
        title='Partial Autocorrelation Function (PACF)',
        xaxis_title='Lag',
        yaxis_title='Correlation',
        template='plotly_white'
    )
    
    return acf_fig, pacf_fig

def suggest_arima_parameters(acf_values, pacf_values, conf_level, seasonal=False):
    """
    Suggest ARIMA or SARIMA parameters based on ACF/PACF analysis
    
    Args:
        acf_values (np.array): ACF values
        pacf_values (np.array): PACF values
        conf_level (float): Confidence level bounds
        seasonal (bool): Whether to suggest seasonal parameters
        
    Returns:
        dict: Suggested parameters
    """
    # Count significant lags
    sig_acf = [i for i, val in enumerate(acf_values) if abs(val) > conf_level and i > 0]
    sig_pacf = [i for i, val in enumerate(pacf_values) if abs(val) > conf_level and i > 0]
    
    # Limit to first 10 significant values
    sig_acf = sig_acf[:10] if len(sig_acf) > 10 else sig_acf
    sig_pacf = sig_pacf[:10] if len(sig_pacf) > 10 else sig_pacf
    
    # Determine p, d, q values
    # p: Order of the AR term (from PACF)
    # q: Order of the MA term (from ACF)
    # d: Differencing order (assumed to be already applied)
    
    # AR process: PACF cuts off, ACF tails off
    if len(sig_pacf) <= 3 and len(sig_acf) > 3:
        p = max(sig_pacf) if sig_pacf else 0
        q = 0
        model_type = "AR"
    # MA process: ACF cuts off, PACF tails off
    elif len(sig_acf) <= 3 and len(sig_pacf) > 3:
        p = 0
        q = max(sig_acf) if sig_acf else 0
        model_type = "MA"
    # ARMA process: both ACF and PACF tail off
    elif len(sig_acf) > 3 and len(sig_pacf) > 3:
        p = 1
        q = 1
        model_type = "ARMA"
    # White noise or other
    else:
        p = 0
        q = 0
        model_type = "None or other"
    
    # For seasonal models
    P, D, Q, s = 0, 0, 0, 0
    
    if seasonal:
        # Look for patterns at seasonal lags
        seasonal_lags = [7, 12, 24, 365]  # Common seasonal periods
        
        for lag in seasonal_lags:
            if lag < len(acf_values):
                # Check for significant autocorrelation at seasonal lag
                if abs(acf_values[lag]) > conf_level:
                    s = lag
                    P = 1
                    Q = 1
                    D = 1
                    break
    
    # Create suggestions
    suggestions = {
        'model_type': model_type,
        'non_seasonal': {'p': p, 'd': 1, 'q': q},  # Assuming d=1 as we've already differenced
        'seasonal': {'P': P, 'D': D, 'Q': Q, 's': s}
    }
    
    return suggestions

def render_data_analysis_tab(historical_data, key_prefix="analysis"):
    """
    Render the Data Analysis tab content
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical data
        key_prefix (str): Prefix to make unique keys for widgets
    """
    st.subheader("Time Series Analysis")
    
    if historical_data.empty or len(historical_data) < 20:
        st.warning("Not enough data for time series analysis. Need at least 20 data points.")
        return
    
    # Create a copy of the data to avoid modifying the original
    data = historical_data.copy()
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Set timestamp as index for time series analysis
    data = data.set_index('timestamp')
    
    # Store and retrieve settings in session state for persistence between refreshes
    state_key = f"{key_prefix}_state"
    if state_key not in st.session_state:
        # Initialize state first time
        st.session_state[state_key] = {
            'ma_window': 5,
            'downsample_factor': 2,
            'seasonal': False
        }
    
    # Select moving average window size and downsampling factor
    col1, col2 = st.columns(2)
    
    with col1:
        ma_window = st.number_input(
            "Moving Average Window Size (odd number)",
            min_value=1,
            max_value=51,
            value=st.session_state[state_key]['ma_window'],
            step=2,
            help="Window size for centered moving average (must be odd)",
            key=f"{key_prefix}_ma_window_input"
        )
        # Update session state
        st.session_state[state_key]['ma_window'] = ma_window
    
    with col2:
        downsample_factor = st.number_input(
            "Downsampling Factor",
            min_value=1,
            max_value=10,
            value=st.session_state[state_key]['downsample_factor'],
            step=1,
            help="Take every nth value after moving average",
            key=f"{key_prefix}_downsample_factor_input"
        )
        # Update session state
        st.session_state[state_key]['downsample_factor'] = downsample_factor
    
    # Calculate moving average
    data['ma'] = centered_moving_average(data, ma_window)
    
    # Downsample the moving average
    ma_downsampled = downsample_data(data['ma'], downsample_factor)
    
    # Show data points info with more details
    original_count = len(data)
    ma_count = len(data['ma'].dropna())
    downsampled_count = len(ma_downsampled.dropna())
    
    st.info(f"""
    Data points summary:
    - Original data: {original_count} points
    - After MA ({ma_window}): {ma_count} points ({ma_count/original_count:.1%})
    - After downsampling (1:{downsample_factor}): {downsampled_count} points ({downsampled_count/original_count:.1%})
    """)
    
    # Check if we have enough data after processing
    if len(ma_downsampled.dropna()) < 20:
        st.warning(f"Not enough data points after processing. Try reducing the MA window size or downsampling factor.")
    
    # Plot the original and processed time series
    st.plotly_chart(
        plot_time_series(
            data['price'], 
            ma_downsampled, 
            title=f"Price with {ma_window}-period Moving Average (1:{downsample_factor} downsampled)"
        ),
        use_container_width=True
    )
    
    # Stationarity Testing
    st.subheader("Stationarity Analysis")
    
    # Initialize variables
    current_series = ma_downsampled.copy()
    is_stationary = False
    differencing_order = 0
    max_differencing = 3  # Maximum differencing order
    
    # Store the original series for later use if it's already stationary
    original_series = current_series.copy()
    
    # Create containers for each differencing level
    diff_containers = []
    for i in range(max_differencing + 1):
        diff_containers.append(st.container())
    
    # Check original series stationarity
    with diff_containers[0]:
        st.subheader("Original Series")
        is_stationary, adf_result, p_value = check_stationarity(current_series)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(plot_adf_results(adf_result, p_value), use_container_width=True)
        
        with col2:
            status = "✅ Stationary" if is_stationary else "❌ Non-stationary"
            st.info(f"Status: {status}")
            st.metric("p-value", f"{p_value:.4f}")
            if not is_stationary:
                st.warning("Series is not stationary. Applying first differencing...")
    
    # Apply differencing if needed
    while not is_stationary and differencing_order < max_differencing:
        differencing_order += 1
        current_series = difference_series(current_series)
        
        with diff_containers[differencing_order]:
            st.subheader(f"After Differencing (Order {differencing_order})")
            is_stationary, adf_result, p_value = check_stationarity(current_series)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(plot_time_series(
                    pd.Series(), current_series, 
                    title=f"Differenced Series (Order {differencing_order})"
                ), use_container_width=True)
            
            with col2:
                st.plotly_chart(plot_adf_results(adf_result, p_value), use_container_width=True)
                
                status = "✅ Stationary" if is_stationary else "❌ Non-stationary"
                st.info(f"Status: {status}")
                st.metric("p-value", f"{p_value:.4f}")
                
                if not is_stationary and differencing_order < max_differencing:
                    st.warning(f"Series is still not stationary. Applying {differencing_order + 1}nd differencing...")
                elif not is_stationary:
                    st.error("Maximum differencing order reached but series is still not stationary.")
    
    # If we found a stationary series, perform ACF/PACF analysis
    if is_stationary:
        st.subheader("ACF and PACF Analysis")
        
        # Use the appropriate series for ACF/PACF analysis
        analysis_series = original_series if differencing_order == 0 else current_series
        
        # Add debugging information
        st.info(f"Series length: {len(analysis_series)}, NaN values: {analysis_series.isnull().sum()}")
        
        # Check for problematic data before computing ACF/PACF
        if analysis_series.isnull().sum() > len(analysis_series) * 0.5:  # More than 50% NaN
            st.warning("Too many missing values for ACF/PACF analysis.")
        elif len(analysis_series.dropna()) < 10:  # Less than 10 non-NaN values
            st.warning("Not enough valid data points for ACF/PACF analysis.")
        elif analysis_series.dropna().std() < 1e-10:  # Near-zero variance
            st.warning("Series has no variance after differencing. Cannot perform ACF/PACF analysis.")
        else:
            try:
                # Use only non-NaN values for analysis
                clean_series = analysis_series.dropna()
                
                # Compute ACF and PACF values with error handling
                acf_values, pacf_values, conf_level = compute_acf_pacf(clean_series)
                
                # Plot ACF and PACF
                acf_fig, pacf_fig = plot_acf_pacf(acf_values, pacf_values, conf_level)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(acf_fig, use_container_width=True)
                
                with col2:
                    st.plotly_chart(pacf_fig, use_container_width=True)
                    
                # Suggest ARIMA/SARIMA parameters
                st.subheader("Model Parameter Suggestions")
                
                # Check for seasonality option
                seasonal = st.checkbox("Check for seasonality (SARIMA)", value=st.session_state[state_key]['seasonal'], key=f"{key_prefix}_seasonal_checkbox")
                st.session_state[state_key]['seasonal'] = seasonal
                
                suggestions = suggest_arima_parameters(acf_values, pacf_values, conf_level, seasonal)
                
                # Display suggestions
                st.info(f"Suggested model type: {suggestions['model_type']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("ARIMA parameters:")
                    params = suggestions['non_seasonal']
                    # Use d=0 if we're using the original series (already stationary)
                    d_value = 0 if differencing_order == 0 else differencing_order
                    st.code(f"ARIMA(p={params['p']}, d={d_value}, q={params['q']})")
                    
                if seasonal:
                    with col2:
                        st.write("SARIMA parameters:")
                        seasonal_params = suggestions['seasonal']
                        if seasonal_params['s'] > 0:
                            st.code(
                                f"SARIMA(p={params['p']}, d={d_value}, q={params['q']}, "
                                f"P={seasonal_params['P']}, D={seasonal_params['D']}, "
                                f"Q={seasonal_params['Q']}, s={seasonal_params['s']})"
                            )
                        else:
                            st.info("No significant seasonality detected")
            except Exception as e:
                st.error(f"Error performing ACF/PACF analysis: {str(e)}")
                st.info("Try increasing the moving average window or downsampling factor to smooth the data.")
    else:
        st.error("Could not achieve stationarity with the maximum allowed differencing order.")
