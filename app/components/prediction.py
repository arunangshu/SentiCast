import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from app.components.data_analysis import centered_moving_average, downsample_data
import contextlib
import os
import sys
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics for the model predictions
    
    Args:
        actual (pd.Series): Actual values
        predicted (pd.Series): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Only compare where we have both actual and predicted values
    common_idx = actual.index.intersection(predicted.index)
    
    if len(common_idx) == 0:
        # No overlapping data points
        return {
            "rmse": None,
            "mae": None,
            "mape": None,
            "r2": None,
            "overlap_count": 0
        }
    
    # Extract the overlapping data points
    actual_overlap = actual[common_idx]
    pred_overlap = predicted[common_idx]
    
    # Drop NaN values
    valid_mask = ~(np.isnan(actual_overlap) | np.isnan(pred_overlap))
    actual_overlap = actual_overlap[valid_mask]
    pred_overlap = pred_overlap[valid_mask]
    
    if len(actual_overlap) == 0:
        # All values were NaN
        return {
            "rmse": None,
            "mae": None,
            "mape": None,
            "r2": None,
            "overlap_count": 0
        }
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_overlap, pred_overlap))
    mae = mean_absolute_error(actual_overlap, pred_overlap)
    
    # Calculate MAPE
    if (actual_overlap == 0).any():
        # Avoid division by zero in MAPE calculation
        mape = np.mean(np.abs((actual_overlap - pred_overlap) / (actual_overlap + 1e-10))) * 100
    else:
        mape = np.mean(np.abs((actual_overlap - pred_overlap) / actual_overlap)) * 100
    
    # Calculate R²
    r2 = r2_score(actual_overlap, pred_overlap)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "overlap_count": len(common_idx)
    }

def plot_predictions(processed_series, train_data, predictions, train_end_date, title="Model Predictions"):
    """
    Plot processed series with training data and predictions
    
    Args:
        processed_series (pd.Series): Processed time series (MA + downsampled)
        train_data (pd.Series): Training portion of the processed series
        predictions (pd.Series): Predicted values
        train_end_date: Date where training data ends
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Plot full processed series with low opacity
    fig.add_trace(go.Scatter(
        x=processed_series.index,
        y=processed_series,
        mode='lines',
        name='Full Data',
        line=dict(color='#2962ff', width=1.5),
        opacity=0.3
    ))
    
    # Plot training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data,
        mode='lines',
        name='Training Data',
        line=dict(color='#ff6d00', width=2)
    ))
    
    # Plot predictions
    if predictions is not None and not predictions.empty:
        # Split predictions into in-sample and out-of-sample
        # Convert train_end_date to datetime64 for comparison with index
        if isinstance(train_end_date, (str, datetime.date)):
            train_end_dt = pd.to_datetime(train_end_date)
        else:
            train_end_dt = train_end_date
            
        in_sample_pred = predictions[predictions.index <= train_end_dt]
        out_sample_pred = predictions[predictions.index > train_end_dt]
        
        # Plot in-sample predictions (training period)
        if not in_sample_pred.empty:
            fig.add_trace(go.Scatter(
                x=in_sample_pred.index,
                y=in_sample_pred,
                mode='lines',
                name='In-sample Predictions',
                line=dict(color='#00c853', width=2, dash='dot')
            ))
        
        # Plot out-of-sample predictions (future)
        if not out_sample_pred.empty:
            fig.add_trace(go.Scatter(
                x=out_sample_pred.index,
                y=out_sample_pred,
                mode='lines',
                name='Future Predictions',
                line=dict(color='#00c853', width=2)
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

def train_arima_model(train_data, params):
    """
    Train an ARIMA model
    
    Args:
        train_data (pd.Series): Training data
        params (dict): Model parameters
        
    Returns:
        model: Trained ARIMA model
    """
    try:
        # Extract parameters
        p = params['p']
        d = params['d']
        q = params['q']
        
        # Create and fit the model
        model = ARIMA(
            train_data,
            order=(p, d, q)
        )
        
        return model.fit()
    
    except Exception as e:
        st.error(f"Error training ARIMA model: {str(e)}")
        return None

def train_sarima_model(train_data, params):
    """
    Train a SARIMA model
    
    Args:
        train_data (pd.Series): Training data
        params (dict): Model parameters
        
    Returns:
        model: Trained SARIMA model
    """
    try:
        # Extract parameters
        p = params['p']
        d = params['d']
        q = params['q']
        P = params['P']
        D = params['D']
        Q = params['Q']
        s = params['s']
        
        # Create and fit the model
        model = SARIMAX(
            train_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s)
        )
        
        return model.fit(disp=False)
    
    except Exception as e:
        st.error(f"Error training SARIMA model: {str(e)}")
        return None

def get_freq_from_series(series):
    """
    Calculate the exact frequency of a time series
    
    Args:
        series (pd.Series): Time series with datetime index
        
    Returns:
        str: Frequency string for predictions
    """
    # Check if index is datetime
    if not pd.api.types.is_datetime64_dtype(series.index):
        return None
    
    # Calculate time differences
    if len(series) < 2:
        return None
    
    # Calculate median time difference for robustness
    time_diffs = series.index.to_series().diff().dropna()
    if len(time_diffs) == 0:
        return None
    
    # Get median difference in seconds for stability
    median_diff_seconds = time_diffs.median().total_seconds()
    
    # Return the exact frequency in seconds
    return f"{int(median_diff_seconds)}S"

def make_predictions(model, start_date, end_date, freq=None, in_sample=False):
    """
    Make predictions using a trained model
    
    Args:
        model: Trained ARIMA or SARIMA model
        start_date: Start date for predictions
        end_date: End date for predictions
        freq (str): Frequency string for date range
        in_sample (bool): Whether to include in-sample predictions
        
    Returns:
        pd.Series: Predicted values
    """
    try:
        # Create prediction index
        if freq is None:
            # Default to daily frequency if not specified
            freq = 'D'
        
        # Make predictions
        if in_sample:
            # Include in-sample predictions
            prediction = model.predict(start=0, end=end_date)
        else:
            # Only out-of-sample predictions
            prediction = model.predict(start=start_date, end=end_date)
        
        return prediction
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def render_prediction_tab(historical_data, key_prefix="prediction"):
    """
    Render the Prediction tab content
    
    Args:
        historical_data (pd.DataFrame): DataFrame with historical data
        key_prefix (str): Prefix to make unique keys for widgets
    """
    st.subheader("Time Series Forecasting")
    
    # Initialize predictions variable to avoid UnboundLocalError
    predictions = None
    
    if historical_data.empty or len(historical_data) < 20:
        st.warning("Not enough data for time series forecasting. Need at least 20 data points.")
        return
    
    # Create a copy of the data to avoid modifying the original
    data = historical_data.copy()
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Set timestamp as index for time series analysis
    data = data.set_index('timestamp')
    
    # Select moving average window size and downsampling factor
    col1, col2 = st.columns(2)
    
    with col1:
        ma_window = st.number_input(
            "Moving Average Window Size (odd number)",
            min_value=1,
            max_value=51,
            value=5,
            step=2,
            help="Window size for centered moving average (must be odd)",
            key=f"{key_prefix}_ma_window_input"
        )
    
    with col2:
        downsample_factor = st.number_input(
            "Downsampling Factor",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Take every nth value after moving average",
            key=f"{key_prefix}_downsample_factor_input"
        )
    
    # Calculate moving average
    data['ma'] = centered_moving_average(data, ma_window)
    
    # Downsample the moving average
    ma_downsampled = downsample_data(data['ma'], downsample_factor)
    
    # Store data in session state for persistence between reruns
    state_key = f"{key_prefix}_data"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            'original_data': data['price'],
            'ma_downsampled': ma_downsampled,
            'predictions': None,
            'model_trained': False,
            'last_ma_window': ma_window,
            'last_downsample_factor': downsample_factor
        }
    else:
        # Update data in session state if parameters have changed
        if (st.session_state[state_key].get('last_ma_window') != ma_window or 
            st.session_state[state_key].get('last_downsample_factor') != downsample_factor):
            st.session_state[state_key]['original_data'] = data['price']
            st.session_state[state_key]['ma_downsampled'] = ma_downsampled
            st.session_state[state_key]['last_ma_window'] = ma_window
            st.session_state[state_key]['last_downsample_factor'] = downsample_factor
    
    # Show data points info
    st.info(f"Total data points: {len(data)}, Moving Average: {len(data['ma'].dropna())}, Downsampled MA: {len(ma_downsampled)}")
    
    # Determine data frequency for predictions
    freq = get_freq_from_series(ma_downsampled)
    if freq:
        seconds = int(freq[:-1])  # Remove 'S' at the end
        if seconds < 60:
            display_freq = f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            display_freq = f"{minutes} minutes"
        elif seconds < 86400:
            hours = seconds // 3600
            display_freq = f"{hours} hours"
        else:
            days = seconds // 86400
            display_freq = f"{days} days"
        st.info(f"Detected data frequency: {display_freq} between points")
    
    # Model Selection
    st.subheader("Model Configuration")
    
    # Select model type
    model_type = st.selectbox(
        "Select Model Type",
        ["ARIMA", "SARIMA", "LSTM"],
        key=f"{key_prefix}_model_type_select"
    )
    
    # Model parameters based on model type
    if model_type == "ARIMA":
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_arima_p_param")
        with col2:
            d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1, key=f"{key_prefix}_arima_d_param")
        with col3:
            q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_arima_q_param")
        
        # Store parameters
        params = {'p': p, 'd': d, 'q': q}
        
    elif model_type == "SARIMA":
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_sarima_p_param")
        with col2:
            d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1, key=f"{key_prefix}_sarima_d_param")
        with col3:
            q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_sarima_q_param")
            
        st.markdown("Seasonal Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            P = st.number_input("P (Seasonal AR)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_sarima_P_param")
        with col2:
            D = st.number_input("D (Seasonal Diff)", min_value=0, max_value=2, value=1, key=f"{key_prefix}_sarima_D_param")
        with col3:
            Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=10, value=1, key=f"{key_prefix}_sarima_Q_param")
        with col4:
            s = st.number_input("s (Seasonality)", min_value=1, max_value=365, value=7, key=f"{key_prefix}_sarima_s_param")
        
        # Store parameters
        params = {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s}
        
    else:  # LSTM
        st.markdown("LSTM Network Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            units = st.number_input(
                "Units (Neurons)", 
                min_value=16, 
                max_value=256, 
                value=64, 
                step=16,
                help="Number of neurons in the LSTM layer",
                key=f"{key_prefix}_lstm_units_param"
            )
            
            dropout = st.slider(
                "Dropout Rate", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2, 
                step=0.1,
                help="Dropout rate to prevent overfitting",
                key=f"{key_prefix}_lstm_dropout_param"
            )
            
        with col2:
            seq_length = st.number_input(
                "Sequence Length (Lookback)", 
                min_value=3, 
                max_value=30, 
                value=10,
                help="Number of past time steps to use for prediction",
                key=f"{key_prefix}_lstm_seq_length_param"
            )
            
            batch_size = st.select_slider(
                "Batch Size", 
                options=[8, 16, 32, 64, 128],
                value=32,
                help="Number of samples per gradient update",
                key=f"{key_prefix}_lstm_batch_size_param"
            )
        
        epochs = st.slider(
            "Training Epochs", 
            min_value=10, 
            max_value=200, 
            value=50, 
            step=10,
            help="Maximum number of training epochs (early stopping may reduce this)",
            key=f"{key_prefix}_lstm_epochs_param"
        )
        
        # Check if we have enough data for the sequence length
        if len(ma_downsampled) <= seq_length:
            st.warning(f"Not enough data points for sequence length of {seq_length}. Reduce sequence length or increase data.")
        
        # Store parameters
        params = {
            'units': units,
            'dropout': dropout,
            'seq_length': seq_length,
            'batch_size': batch_size,
            'epochs': epochs
        }
    
    # Training and prediction settings
    st.subheader("Training and Prediction Settings")
    
    # Get min and max dates from data
    min_date = ma_downsampled.index.min().date()
    max_date = ma_downsampled.index.max().date()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training period selection
        st.markdown("Training Period")
        train_start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_train_start_date"
        )
        
        train_end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_train_end_date"
        )
        
        # Validate dates
        if train_start_date > train_end_date:
            st.error("Start date must be before end date")
        
    with col2:
        # Prediction horizon
        st.markdown("Prediction Period")
        pred_start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            key=f"{key_prefix}_pred_start_date"
        )
        
        # Calculate a default forecast horizon (20% of training data)
        default_forecast_days = max(1, int((max_date - min_date).days * 0.2))
        future_date = max_date + datetime.timedelta(days=default_forecast_days)
        
        pred_end_date = st.date_input(
            "End Date",
            value=future_date,
            min_value=min_date,
            key=f"{key_prefix}_pred_end_date"
        )
        
        # Validate dates
        if pred_start_date > pred_end_date:
            st.error("Start date must be before end date")
    
    # Filter data for training
    try:
        # Convert dates to datetime for filtering
        train_start = pd.Timestamp(train_start_date)
        train_end = pd.Timestamp(train_end_date)
        pred_start = pd.Timestamp(pred_start_date)
        pred_end = pd.Timestamp(pred_end_date)
        
        # Filter data
        mask = (ma_downsampled.index >= train_start) & (ma_downsampled.index <= train_end)
        train_data = ma_downsampled[mask]
        
        if len(train_data) < 10:
            st.warning("Training data has less than 10 points. Model may not perform well.")
    
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return
    
    # Train model button
    if st.button("Train Model and Generate Predictions", key=f"{key_prefix}_train_model_btn"):
        with st.spinner("Training model and generating predictions..."):
            # Initialize predictions variable
            predictions = None
            
            # Train model based on type
            if model_type == "ARIMA":
                model_result = train_arima_model(train_data, params)
                
                if model_result is not None:
                    # Generate predictions
                    predictions = make_predictions(
                        model_result, 
                        pred_start, 
                        pred_end,
                        freq=freq,  # Pass the exact frequency
                        in_sample=(pred_start <= train_end)
                    )
                    
            elif model_type == "SARIMA":
                model_result = train_sarima_model(train_data, params)
                
                if model_result is not None:
                    # Generate predictions
                    predictions = make_predictions(
                        model_result, 
                        pred_start, 
                        pred_end,
                        freq=freq,  # Pass the exact frequency
                        in_sample=(pred_start <= train_end)
                    )
                    
            else:  # LSTM
                # Check if we have enough data
                if len(train_data) <= params['seq_length']:
                    st.error(f"Not enough training data for sequence length of {params['seq_length']}. Reduce sequence length or increase training data.")
                    model_result = None
                    predictions = None
                else:
                    # Train LSTM model
                    st.subheader("LSTM Model Training")
                    
                    # Display model architecture
                    model_architecture = f"""
                    **Model Architecture:**
                    - Input shape: ({params['seq_length']}, 1)
                    - LSTM layer 1: {params['units']} units with dropout {params['dropout']}
                    - LSTM layer 2: {params['units']//2} units with dropout {params['dropout']}
                    - Dense output layer: 1 unit
                    """
                    st.info(model_architecture)
                    
                    # Visualize model architecture with a diagram
                    architecture_html = visualize_lstm_architecture(
                        params['units'], 
                        params['dropout'], 
                        params['seq_length']
                    )
                    st.markdown(architecture_html, unsafe_allow_html=True)
                    
                    # Display training information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training samples", f"{len(train_data) - params['seq_length']}")
                    with col2:
                        st.metric("Max Epochs", f"{params['epochs']}")
                    
                    # Train model with progress visualization
                    model_result = train_lstm_model(train_data, params)
                    
                    if model_result is not None:
                        # Show model summary
                        st.success("✅ Model training complete!")
                        
                        # Generate predictions
                        with st.spinner("Generating predictions..."):
                            predictions = make_lstm_predictions(
                                model_result,
                                pred_start,
                                pred_end,
                                freq=freq
                            )
                
            # Process predictions regardless of model type
            # Initialize predictions variable at the top level to avoid UnboundLocalError
            if 'predictions' not in locals():
                predictions = None
                
            if predictions is not None:
                # Store predictions and mark training as successful
                st.session_state[state_key]['predictions'] = predictions
                st.session_state[state_key]['model_trained'] = True
                
                # Calculate metrics for overlapping part
                metrics = calculate_metrics(ma_downsampled, predictions)
                
                # Display metrics if there's overlap
                if metrics["overlap_count"] > 0:
                    st.subheader("Model Evaluation")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    
                    with metric_col2:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                    
                    with metric_col3:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    
                    with metric_col4:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                    
                    st.caption(f"Based on {metrics['overlap_count']} overlapping data points")
                
                st.success("Model training and predictions completed successfully!")
                # Store metrics in session state
                st.session_state[state_key]['metrics'] = metrics
            else:
                st.error("Failed to generate predictions")
    
    # Check if model has already been trained and has predictions
    if st.session_state[state_key].get('model_trained') and st.session_state[state_key].get('predictions') is not None:
        # Get predictions from session state
        predictions = st.session_state[state_key]['predictions']
        
        # Plot the original series, training data, and predictions
        st.subheader("Model Predictions Visualization")
        st.plotly_chart(
            plot_predictions(ma_downsampled, train_data, predictions, train_end_date),
            use_container_width=True
        )
        
        # Add a caption for the plot
        st.caption("""
        Note:
        - Dotted green line represents in-sample predictions (within training period)
        - Solid green line represents future predictions
        - Original data is shown in faded blue
        - Training data is shown in orange
        """)
        
        # Display metrics if there's overlap
        if 'metrics' in st.session_state[state_key]:
            metrics = st.session_state[state_key]['metrics']
            if metrics["overlap_count"] > 0:
                st.subheader("Model Evaluation")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                
                with metric_col2:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                
                with metric_col3:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
                
                with metric_col4:
                    st.metric("R²", f"{metrics['r2']:.4f}")
                
                st.caption(f"Based on {metrics['overlap_count']} overlapping data points")
    else:
        # No model trained yet
        st.info("To see predictions, train a model using the form above.")

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM model
    
    Args:
        data (np.array): Input data array
        seq_length (int): Sequence length (lookback window)
        
    Returns:
        tuple: (X, y) sequences and targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm_model(train_data, params):
    """
    Train an LSTM model
    
    Args:
        train_data (pd.Series): Training data
        params (dict): Model parameters
        
    Returns:
        dict: Trained LSTM model and related components
    """
    try:
        # Extract parameters
        units = params['units']
        dropout = params['dropout']
        epochs = params['epochs']
        batch_size = params['batch_size']
        seq_length = params['seq_length']
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled, seq_length)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units // 2))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model with early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        # Create progress display elements
        progress_bar = st.progress(0)
        loss_metric = st.empty()
        epoch_text = st.empty()
        
        # List to store training history for plotting
        loss_history = []
        
        # Create a custom callback to update progress
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                epoch_text.text(f"Epoch {epoch+1}/{epochs}")
            
            def on_epoch_end(self, epoch, logs=None):
                # Update progress bar
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                
                # Update loss metric
                loss_value = logs.get('loss')
                if loss_value is not None:
                    loss_metric.metric("Training Loss", f"{loss_value:.6f}")
                    loss_history.append(loss_value)
        
        # Train model without suppressing output
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop, TrainingProgressCallback()]
        )
        
        # Clear the progress elements
        progress_bar.empty()
        loss_metric.empty()
        epoch_text.empty()
        
        # Plot training history
        if loss_history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(loss_history) + 1)),
                y=loss_history,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#00c853', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="LSTM Training Loss History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display total epochs trained
            st.caption(f"Model trained for {len(loss_history)} epochs (early stopping may have reduced from max {epochs} epochs)")
        
        # Return model and components needed for prediction
        return {
            'model': model,
            'scaler': scaler,
            'seq_length': seq_length,
            'train_data': train_data,
            'history': loss_history
        }
    
    except Exception as e:
        st.error(f"Error training LSTM model: {str(e)}")
        return None

def make_lstm_predictions(model_result, start_date, end_date, freq=None):
    """
    Make predictions using a trained LSTM model
    
    Args:
        model_result (dict): Dictionary with trained model and components
        start_date: Start date for predictions
        end_date: End date for predictions
        freq (str): Frequency string for date range
        
    Returns:
        pd.Series: Predicted values
    """
    try:
        # Extract model components
        model = model_result['model']
        scaler = model_result['scaler']
        seq_length = model_result['seq_length']
        train_data = model_result['train_data']
        
        # Create date range for predictions
        if freq is None:
            freq = 'D'  # Default daily frequency
        
        # Create prediction dates
        pred_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Get the last sequence from training data
        last_sequence = train_data.values[-seq_length:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # Initialize predictions list
        predictions = []
        curr_sequence = last_sequence_scaled.copy()
        
        # Generate predictions one by one
        for _ in range(len(pred_dates)):
            # Reshape for LSTM input [1, sequence_length, 1]
            X_pred = curr_sequence.reshape(1, seq_length, 1)
            
            # Predict next value (suppress tensorflow warnings)
            with suppress_stdout():
                next_pred_scaled = model.predict(X_pred, verbose=0)
            
            # Inverse transform to get actual value
            next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence by removing first element and adding the prediction
            curr_sequence = np.append(curr_sequence[1:], next_pred_scaled)[..., np.newaxis]
        
        # Create pandas Series with dates as index
        predictions_series = pd.Series(predictions, index=pred_dates)
        
        return predictions_series
    
    except Exception as e:
        st.error(f"Error making LSTM predictions: {str(e)}")
        return None

# Custom context manager to suppress stdout
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def visualize_lstm_architecture(units, dropout, seq_length):
    """
    Create a visual representation of the LSTM model architecture
    
    Args:
        units (int): Number of units in first LSTM layer
        dropout (float): Dropout rate
        seq_length (int): Input sequence length
        
    Returns:
        str: HTML for displaying the architecture diagram
    """
    # Calculate node positions and sizes
    input_layer = (0, 0.5, f"Input\n({seq_length}, 1)")
    lstm1_layer = (1, 0.5, f"LSTM\n({units})")
    dropout1_layer = (1.5, 0.5, f"Dropout\n({dropout})")
    lstm2_layer = (2.5, 0.5, f"LSTM\n({units//2})")
    dropout2_layer = (3, 0.5, f"Dropout\n({dropout})")
    output_layer = (4, 0.5, "Dense\n(1)")
    
    layers = [input_layer, lstm1_layer, dropout1_layer, lstm2_layer, dropout2_layer, output_layer]
    connections = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 3)), plt.gca()
    
    # Draw nodes
    for i, (x, y, label) in enumerate(layers):
        color = '#1565C0' if i == 0 else '#4CAF50' if i == len(layers)-1 else '#FFC107'
        ax.add_patch(plt.Circle((x, y), 0.2, color=color, alpha=0.7))
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_x, start_y, _ = layers[start_idx]
        end_x, end_y, _ = layers[end_idx]
        ax.arrow(start_x + 0.2, start_y, end_x - start_x - 0.4, 0,
                 head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # Set chart properties
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('LSTM Network Architecture', fontsize=12, pad=20)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Return HTML for displaying the image
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%">'
