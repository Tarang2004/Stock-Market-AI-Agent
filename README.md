# Stock Market AI Agent

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical data. It includes both a command-line interface and a web-based dashboard using Streamlit.

## Features

- Download stock data from Yahoo Finance
- Preprocess and normalize data
- Train LSTM model for price prediction
- Interactive web interface with Streamlit
- Visualize predictions with interactive charts
- Customizable model parameters

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd stock-market-ai-agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

To run the basic stock predictor:

```bash
python stock_predictor.py
```

This will train a model on AAPL stock data and show a plot of predictions.

### Web Interface

To launch the Streamlit web interface:

```bash
streamlit run app.py
```

The web interface allows you to:
- Enter any stock symbol
- Select date range for training data
- Adjust model parameters
- View interactive predictions
- See model performance metrics

## Model Parameters

- **Sequence Length**: Number of past days to consider for prediction (default: 60)
- **Training Epochs**: Number of training iterations (default: 20)
- **Batch Size**: Number of samples per training batch (default: 32)

## Notes

- The model uses only closing prices for predictions
- Data is normalized before training
- 80% of data is used for training, 20% for testing
- Predictions are denormalized before display

## Requirements

See `requirements.txt` for full list of dependencies. Main requirements:
- Python 3.8+
- TensorFlow
- yfinance
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly 