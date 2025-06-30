# SentiCast - Cryptocurrency Analytics and Forecasting Platform

A Streamlit application for real-time cryptocurrency analytics, visualization, and price prediction using time-series forecasting models.

## Features

- Real-time cryptocurrency price tracking in INR with dynamic USD to INR exchange rate
- Interactive historical price charts with customizable time periods
- Key market indicators and statistics
- Support for Bitcoin, Ethereum, Ripple, Dogecoin, and Tether
- Customizable refresh intervals and history periods
- Advanced data analysis with time series diagnostics (stationarity tests, ACF/PACF)
- Automated ARIMA/SARIMA parameter suggestions based on time series properties
- Time-series forecasting with ARIMA and SARIMA models
- Configurable model hyperparameters
- Prediction performance metrics (RMSE, MAE, MAPE, R²)

## Project Structure

```
SentiCast/
├── app/
│   ├── __init__.py
│   ├── main.py             # Main Streamlit application
│   ├── components/         # Reusable UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py      # Sidebar components
│   │   ├── charts.py       # Chart components
│   │   ├── data_analysis.py # Time-series analysis components
│   │   └── model_trainer.py # Time-series model training interface
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── data_fetcher.py # Functions to fetch crypto data
│       └── time_series_models.py # Time-series forecasting models
├── run.py                  # Helper script to set Python path
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
   
   Or alternatively:
   ```
   python run.py
   ```

## Usage

### Cryptocurrency Data Visualization

1. Select a cryptocurrency from the sidebar dropdown
2. Adjust the historical data period using the slider
3. Set your preferred refresh interval
4. Explore the interactive charts and market data

### Time Series Data Analysis

1. Navigate to the "Data Analysis" tab
2. Configure moving average window size (must be odd) and downsampling factor
3. Analyze stationarity with ADF tests and automatic differencing
4. Examine the ACF and PACF plots with confidence intervals
5. Get intelligent suggestions for ARIMA/SARIMA model parameters based on ACF/PACF patterns
6. Use seasonality detection for SARIMA parameter recommendations

### Time-Series Forecasting

1. Navigate to the "Price Prediction" section in the sidebar
2. Select your desired model (ARIMA or SARIMA)
3. Configure model hyperparameters:
   - For ARIMA: p, d, q values
   - For SARIMA: p, d, q, P, D, Q, and seasonal period
4. Select training data range
5. Choose prediction horizon (in days)
6. Click "Train Model" to generate forecasts
7. View prediction results and performance metrics

## Data Sources

The application utilizes the Binance API to fetch real-time cryptocurrency data and a foreign exchange API to get the latest USD to INR conversion rates.

## Future Enhancements

- Twitter sentiment analysis integration
- Portfolio management tools
- Additional cryptocurrencies support
- Deep learning forecasting models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 