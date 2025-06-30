# SentiCast - Cryptocurrency Analytics and Forecasting Platform

A robust Streamlit application for real-time cryptocurrency analytics, interactive visualizations, and advanced time-series forecasting of cryptocurrency prices.

## Features

### Real-time Market Data
- Real-time cryptocurrency price tracking in INR with dynamic USD to INR exchange rate conversion
- Interactive historical price charts with customizable time periods
- Key market indicators and statistics (price, market cap, volume, 24h change)
- Support for multiple cryptocurrencies including Bitcoin, Ethereum, Ripple, Dogecoin, and Tether
- Customizable refresh intervals and history periods
- Relative Strength Index (RSI) indicator visualization

### Advanced Data Analysis
- Interactive time-series preprocessing:
  - Configurable moving average window selection (odd values only)
  - Adjustable downsampling factor to reduce noise
- Statistical time series diagnostics:
  - Augmented Dickey-Fuller (ADF) stationarity testing with visual results
  - Automatic differencing until stationarity is achieved
  - Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis with confidence intervals
  - Intelligent ARIMA/SARIMA parameter suggestions based on ACF/PACF patterns
  - Seasonality detection for time series data

### Time Series Forecasting
- Multiple forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - LSTM (Long Short-Term Memory) neural networks
- Fully configurable model hyperparameters:
  - For ARIMA: p, d, q values
  - For SARIMA: p, d, q, P, D, Q, and seasonal period
  - For LSTM: units, dropout rate, sequence length, batch size, epochs
- Configurable training/testing period selection
- Custom prediction horizon settings
- Comprehensive model evaluation metrics:
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Coefficient of determination (R²)
- Visualization of both in-sample (training period) and out-of-sample (future) predictions

## Tech Stack

- **Frontend & App Framework**: Streamlit
- **Data Processing & Analysis**:
  - Pandas - Data manipulation and analysis
  - NumPy - Numerical computing
  - Statsmodels - Time series analysis and statistical models
- **Forecasting Models**:
  - ARIMA/SARIMA from Statsmodels
  - LSTM neural networks using TensorFlow/Keras
- **Data Visualization**:
  - Plotly - Interactive charts
  - Matplotlib - Static visualizations
- **API Integration**:
  - Binance API - Cryptocurrency data
  - Exchange Rate API - USD to INR conversion
- **Performance Metrics**:
  - Scikit-learn - Model evaluation metrics

## Project Structure

```
SentiCast/
├── app/
│   ├── __init__.py
│   ├── main.py             # Main Streamlit application
│   ├── components/         # Reusable UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py      # Sidebar components and controls
│   │   ├── charts.py       # Chart visualization components
│   │   ├── data_analysis.py # Time-series analysis components
│   │   └── prediction.py   # Time-series forecasting components
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── data_fetcher.py # Functions to fetch cryptocurrency data
├── run_app.py              # App launcher
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SentiCast.git
   cd SentiCast
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run_app.py
   ```

## Detailed Usage Guide

### Main Dashboard

1. **Select a cryptocurrency** from the sidebar dropdown menu
2. **Adjust the historical data period** using the slider (default: 30 days)
3. **Set your preferred refresh interval** to control how often data updates
4. Explore the interactive market overview metrics:
   - Current price
   - 24h price change
   - Market cap
   - 24h trading volume
5. Use the "Refresh Now" button to manually update data or toggle auto-refresh functionality

### Price Chart Tab

1. View the interactive price chart for your selected cryptocurrency
2. Hover over the chart to see detailed price information at specific dates
3. Use the built-in Plotly controls to zoom, pan, or save the chart as an image

### RSI Indicator Tab

1. Analyze the Relative Strength Index (RSI) to identify overbought/oversold conditions
2. Values above 70 indicate potential overbought conditions
3. Values below 30 indicate potential oversold conditions
4. Use this information for technical analysis and entry/exit points

### Data Analysis Tab

1. **Configure preprocessing options**:
   - Set moving average window size (must be odd number)
   - Adjust downsampling factor to reduce noise
   
2. **Analyze time series characteristics**:
   - View the original vs. processed time series data
   - Examine stationarity with ADF test results
   - Check p-value against significance level
   - View critical values for hypothesis testing
   
3. **Apply automatic differencing**:
   - Use the "Apply Differencing" button to make series stationary
   - Select differencing order manually or let the system determine it
   - View the differenced series and updated stationarity test
   
4. **Analyze ACF and PACF**:
   - Examine autocorrelation patterns with confidence intervals
   - Identify significant lags in partial autocorrelation
   - Understand AR and MA components for model selection
   
5. **Get model parameter suggestions**:
   - Receive intelligent ARIMA parameter recommendations based on ACF/PACF
   - Get SARIMA parameter suggestions if seasonality is detected
   - Use these suggestions in the Prediction tab

### Prediction Tab

1. **Configure preprocessing**:
   - Set moving average window and downsampling factor
   - These settings apply to data before modeling
   
2. **Select model type**:
   - Choose between ARIMA, SARIMA, and LSTM models
   - For SARIMA, additional seasonal parameters are required
   - For LSTM, neural network architecture parameters are available
   
3. **Configure model parameters**:
   - Set p, d, q values for ARIMA components
   - For SARIMA, also set P, D, Q, and seasonal period (s)
   - For LSTM, configure neural network architecture:
     - Number of units (neurons)
     - Dropout rate for regularization
     - Sequence length (lookback window)
     - Batch size and number of epochs
   - You can use the suggestions from the Data Analysis tab for ARIMA/SARIMA
   
4. **Set training period**:
   - Use the date range slider to select training data
   - Balance between more training data and more recent data
   
5. **Set prediction horizon**:
   - Choose how many days into the future to predict
   - Longer horizons typically have greater uncertainty
   
6. **Train model and generate predictions**:
   - Click "Train Model" to fit the model and make predictions
   - For LSTM models, view real-time training progress and loss metrics
   - View both in-sample (training period) and out-of-sample (future) predictions
   - For LSTM, visualize the neural network architecture
   
7. **Evaluate model performance**:
   - Examine RMSE, MAE, MAPE, and R² metrics
   - Lower RMSE, MAE, MAPE, and higher R² indicate better performance
   - Compare different parameter configurations to find optimal models

## Data Sources

The application leverages the following data sources:
- **Binance API**: Real-time cryptocurrency data
- **Exchange Rate API**: Current USD to INR conversion rates

## Future Enhancements

- Twitter sentiment analysis integration for market sentiment indicators
- Portfolio management and tracking tools
- Support for additional cryptocurrencies
- Additional deep learning forecasting models (GRU, Transformer)
- Volatility forecasting
- Technical indicator overlays on price charts
- Backtesting framework for trading strategies
- Export capabilities for charts and predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.