import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol, start_date, end_date, sequence_length=60):
        self.symbol = symbol.upper()
        self.start_date = self.validate_date(start_date)
        self.end_date = self.validate_date(end_date)
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.raw_data = None
        
        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError(f"Start date ({self.start_date}) must be before end date ({self.end_date})")
            
        # Check if date range is reasonable
        date_diff = (self.end_date - self.start_date).days
        if date_diff < sequence_length:
            raise ValueError(f"Date range ({date_diff} days) is too short. Need at least {sequence_length} days.")
        
    def validate_date(self, date_input):
        """Validate and convert date input to datetime object"""
        if isinstance(date_input, str):
            try:
                date_obj = datetime.strptime(date_input, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {date_input}. Use YYYY-MM-DD format.")
        elif isinstance(date_input, datetime):
            date_obj = date_input
        elif hasattr(date_input, 'date'):  # Handle datetime.date objects
            date_obj = datetime.combine(date_input, datetime.min.time())
        else:
            raise ValueError(f"Invalid date type: {type(date_input)}. Use string (YYYY-MM-DD) or datetime object.")
        
        # Ensure date is not in the future
        today = datetime.now().date()
        if date_obj.date() > today:
            raise ValueError(f"Date {date_obj.date()} is in the future. Please select a date on or before {today}.")
        
        return date_obj
    
    def get_stock_data(self):
        """Get stock data from Yahoo Finance"""
        print(f"\nGetting stock data for {self.symbol}")
        print(f"Start date: {self.start_date} (type: {type(self.start_date)})")
        print(f"End date: {self.end_date} (type: {type(self.end_date)})")
        
        # Convert dates to string format if they aren't already
        if not isinstance(self.start_date, str):
            self.start_date = self.start_date.strftime("%Y-%m-%d")
        if not isinstance(self.end_date, str):
            self.end_date = self.end_date.strftime("%Y-%m-%d")
            
        print(f"Using dates: {self.start_date} to {self.end_date}")
        
        try:
            # First try yf.download
            print("\nTrying yf.download...")
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
            print(f"Data shape: {data.shape}")
            if not data.empty:
                print("First few rows:")
                print(data.head())
                return data
                
            # If that fails, try Ticker.history
            print("\nTrying Ticker.history...")
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            print(f"Data shape: {data.shape}")
            if not data.empty:
                print("First few rows:")
                print(data.head())
                return data
                
            # If both fail, try period-based approach
            print("\nTrying period-based approach...")
            days = (datetime.strptime(self.end_date, "%Y-%m-%d") - datetime.strptime(self.start_date, "%Y-%m-%d")).days
            data = ticker.history(period=f"{days}d")
            print(f"Data shape: {data.shape}")
            if not data.empty:
                print("First few rows:")
                print(data.head())
                return data
                
            raise ValueError(f"No data returned for {self.symbol}")
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise ValueError(f"Failed to download data for {self.symbol}: {str(e)}")
    
    def prepare_data(self, data):
        """Prepare data for LSTM model with enhanced preprocessing"""
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} days of data, but got {len(data)} days")
        
        print(f"Preparing data with sequence length: {self.sequence_length}")
        
        # Extract close price
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Normalize the data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            # Input sequence (past sequence_length days)
            X.append(scaled_data[i:(i + self.sequence_length)])
            # Target (next day's price)
            y.append(scaled_data[i + self.sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        print(f"Created {len(X)} sequences for training")
        
        return X, y
    
    def build_model(self):
        """Build and compile enhanced LSTM model"""
        model = Sequential([
            # First LSTM layer with dropout
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            
            # Second LSTM layer with dropout
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(50),
            Dropout(0.2),
            
            # Dense output layer
            Dense(1)
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer='adam', 
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        print("Model built successfully")
        return model
    
    def train_model(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1):
        """Train the model with enhanced callbacks"""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty")
        
        print(f"Training model with {len(X_train)} samples for {epochs} epochs")
        
        # Set up callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=0
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Don't shuffle time series data
        )
        
        print("Model training completed")
        return history
    
    def predict(self, X):
        """Make predictions and return denormalized values"""
        if len(X) == 0:
            raise ValueError("No data provided for prediction")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Denormalize predictions
        denormalized_predictions = self.scaler.inverse_transform(predictions)
        
        return denormalized_predictions
    
    def predict_future(self, days=5):
        """Predict future prices for specified number of days"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if self.raw_data is None:
            raise ValueError("No historical data available. Call get_stock_data() first.")
        
        # Get the last sequence_length days of data
        last_data = self.raw_data['Close'].values[-self.sequence_length:].reshape(-1, 1)
        last_scaled = self.scaler.transform(last_data)
        
        predictions = []
        current_sequence = last_scaled.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next day
            next_pred = self.model.predict(X, verbose=0)
            predictions.append(float(next_pred[0, 0]))
            
            # Update sequence by removing first element and adding prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # Denormalize predictions
        predictions = np.array(predictions).reshape(-1, 1)
        denormalized_predictions = self.scaler.inverse_transform(predictions)
        
        return denormalized_predictions.flatten()
    
    def plot_predictions(self, actual, predicted, title="Stock Price Prediction", save_path=None):
        """Plot actual vs predicted prices with enhanced visualization"""
        if len(actual) == 0 or len(predicted) == 0:
            raise ValueError("No data to plot")
        
        plt.figure(figsize=(15, 8))
        
        # Plot actual vs predicted
        plt.plot(actual, label='Actual Price', color='blue', linewidth=2)
        plt.plot(predicted, label='Predicted Price', color='red', linewidth=2, alpha=0.8)
        
        # Calculate and display metrics
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        
        plt.title(f'{title}\nMSE: ${mse:.2f}, MAE: ${mae:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some styling
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()
    
    def get_data_info(self):
        """Get information about the loaded data"""
        if self.raw_data is None:
            return "No data loaded"
        
        info = {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_days': len(self.raw_data),
            'date_range': f"{self.raw_data.index[0].strftime('%Y-%m-%d')} to {self.raw_data.index[-1].strftime('%Y-%m-%d')}",
            'price_range': f"${self.raw_data['Close'].min():.2f} - ${self.raw_data['Close'].max():.2f}",
            'sequence_length': self.sequence_length
        }
        return info

def main():
    """Enhanced example usage with better error handling and date selection"""
    
    # Example with user-specified dates
    symbol = "AAPL"
    
    # You can specify custom dates here
    start_date = "2021-01-01"  # Start date
    end_date = "2024-01-01"    # End date
    
    # Or use relative dates
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    try:
        print(f"Starting stock prediction for {symbol}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Initialize predictor
        predictor = StockPredictor(symbol, start_date, end_date, sequence_length=60)
        
        # Get data info
        predictor.get_stock_data()
        data_info = predictor.get_data_info()
        print("\nData Info:")
        for key, value in data_info.items():
            print(f"  {key}: {value}")
        
        # Prepare data
        data = predictor.get_stock_data()
        X, y = predictor.prepare_data(data)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Build and train model
        predictor.build_model()
        history = predictor.train_model(X_train, y_train, epochs=25, batch_size=32)
        
        # Make predictions
        test_predictions = predictor.predict(X_test)
        actual_prices = predictor.scaler.inverse_transform(y_test)
        
        # Plot results
        predictor.plot_predictions(
            actual_prices.flatten(),
            test_predictions.flatten(),
            f"{symbol} Stock Price Prediction ({start_date} to {end_date})"
        )
        
        # Predict future prices
        print("\nPredicting next 5 days:")
        future_predictions = predictor.predict_future(5)
        for i, pred in enumerate(future_predictions, 1):
            print(f"Day +{i}: ${pred:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()