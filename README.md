# SentiCast - Cryptocurrency Analytics and Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3F4F75?logo=plotly)](https://plotly.com/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.13%2B-blue)](https://www.statsmodels.org/)

[![Binance API](https://img.shields.io/badge/Binance%20API-Market%20Data-F0B90B)](https://binance-docs.github.io/apidocs/)
[![News API](https://img.shields.io/badge/News%20API-Articles-CF0000)](https://newsapi.org/)
[![Reddit](https://img.shields.io/badge/Reddit-Social%20Data-FF4500?logo=reddit)](https://www.reddit.com/)

[![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-success)](https://github.com/arunangshu/SentiCast/)
[![ARIMA](https://img.shields.io/badge/ARIMA-Modeling-informational)](https://github.com/arunangshu/SentiCast/)
[![LSTM](https://img.shields.io/badge/LSTM-Neural%20Networks-blueviolet)](https://github.com/arunangshu/SentiCast/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/arunangshu/SentiCast/graphs/commit-activity)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/arunangshu/SentiCast/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://senticast.streamlit.app/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/Documentation-Yes-brightgreen.svg)](https://github.com/arunangshu/SentiCast/)

A robust Streamlit application for real-time cryptocurrency analytics, interactive visualizations, and advanced time-series forecasting of cryptocurrency prices.

## Features

### Real-time Market Data
- Real-time cryptocurrency price tracking in INR with dynamic USD to INR exchange rate conversion
- Interactive historical price charts with customizable time periods
- Key market indicators and statistics (price, market cap, volume, 24h change)
- Support for multiple cryptocurrencies including Bitcoin, Ethereum, Ripple, Dogecoin, and Tether
- Customizable refresh intervals and history periods
- Relative Strength Index (RSI) indicator visualization

### Social Media & News Analysis
- Comprehensive cryptocurrency sentiment analysis from multiple sources:
  - Reddit posts using direct Reddit search functionality
  - News articles from NewsAPI with full article descriptions
- Filtering options for both data sources:
  - Reddit: Sort by upvotes or recency, minimum upvotes filter, keyword search
  - News: Sort by recency or alphabetically, keyword search
- Modern UI with Poppins font and clean card-based design
- Post engagement metrics (upvotes for Reddit)
- Clickable post titles that link to original content
- Full article descriptions and post content without truncation
- News article thumbnails with source attribution

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
  - News API - Latest cryptocurrency news
  - Reddit Search API - Social media posts
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
│   │   ├── sentiment_analysis.py # Social media sentiment analysis
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

### Social Media & News Tab

1. **Choose data sources**:
   - Toggle between Reddit posts and News articles
   - Each source has dedicated filtering options

2. **Reddit filtering options**:
   - Sort by most upvotes or most recent posts
   - Set minimum upvotes threshold
   - Filter by specific keywords
   - Click post titles to view original content on Reddit

3. **News filtering options**:
   - Sort by most recent or alphabetically by title
   - Filter by specific keywords
   - View article thumbnails and full descriptions
   - Click headlines to read complete articles

4. **Analyze market sentiment**:
   - Track discussions and news coverage about cryptocurrencies
   - Identify trending topics and public perception
   - Use social sentiment as a complementary market indicator

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
- **Reddit Search**: Posts and discussions from cryptocurrency subreddits
- **News API**: Latest news articles related to cryptocurrencies

## Streamlit Cloud Deployment

When deploying the application on Streamlit Cloud, please note the following:

1. **Reddit Data**: Due to CORS restrictions, the direct Reddit search functionality will automatically fall back to using sample data from the `search.json` file. This is a known limitation of making cross-origin requests from Streamlit Cloud.

2. **News API**: The News API has rate limits that may be quickly reached when deployed on Streamlit Cloud. The application will automatically fall back to sample data from `news.json` if rate limits are exceeded.

3. **Sample Data**: To ensure the application works smoothly on Streamlit Cloud:
   - Make sure `search.json` and `news.json` are included in your repository
   - These files are explicitly excluded from `.gitignore` to ensure they're available

4. **Environment Variables**: For full functionality, set the following environment variables in Streamlit Cloud:
   - `NEWS_API_KEY`: Your News API key for higher rate limits
   - `STREAMLIT_ENV`: Set to 'production' for deployment

5. **Local Development**: When running locally, the application will attempt to fetch real-time data from both Reddit and News API before falling back to sample data.

## Future Enhancements

- ~~Twitter sentiment analysis integration for market sentiment indicators~~ (Implemented Facebook and Reddit instead)
- Portfolio management and tracking tools
- Support for additional cryptocurrencies
- Additional deep learning forecasting models (GRU, Transformer)
- Volatility forecasting
- Technical indicator overlays on price charts
- Backtesting framework for trading strategies
- Export capabilities for charts and predictions
- Sentiment scoring and analysis using NLP models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 