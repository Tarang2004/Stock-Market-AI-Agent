import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from stock_predictor import StockPredictor
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

class FuturePricePredictor:
    def __init__(self, trained_predictor):
        """
        Initialize with a trained StockPredictor instance
        
        Args:
            trained_predictor: Trained StockPredictor instance
        """
        if trained_predictor.model is None:
            raise ValueError("The predictor must have a trained model")
        
        self.predictor = trained_predictor
        self.symbol = trained_predictor.symbol
        self.sequence_length = trained_predictor.sequence_length
        self.scaler = trained_predictor.scaler
        self.model = trained_predictor.model
        
        # Get last known price and data
        if hasattr(trained_predictor, 'raw_data') and trained_predictor.raw_data is not None:
            self.last_known_prices = trained_predictor.raw_data['Close'].values
            self.last_date = trained_predictor.raw_data.index[-1]
        else:
            raise ValueError("No historical data available in the trained predictor")
    
    def predict_future_advanced(self, days=30, confidence_level=0.95, monte_carlo_runs=100):
        """
        Advanced future price prediction with confidence intervals
        
        Args:
            days: Number of days to predict
            confidence_level: Confidence level for intervals (0.95 = 95%)
            monte_carlo_runs: Number of Monte Carlo simulations for confidence intervals
            
        Returns:
            dict: Contains predictions, confidence intervals, and metadata
        """
        # Get the last sequence for prediction
        last_sequence = self.last_known_prices[-self.sequence_length:].reshape(-1, 1)
        last_scaled = self.scaler.transform(last_sequence)
        
        # Store predictions from multiple runs
        all_predictions = []
        
        for run in range(monte_carlo_runs):
            predictions = []
            current_sequence = last_scaled.copy()
            
            for day in range(days):
                # Reshape for model input
                X = current_sequence.reshape(1, self.sequence_length, 1)
                
                # Make prediction
                next_pred = self.model.predict(X, verbose=0)[0, 0]
                
                # Add small random noise for Monte Carlo simulation
                if run > 0:  # Don't add noise to the first run (base prediction)
                    noise_factor = 0.02  # 2% noise
                    noise = np.random.normal(0, noise_factor)
                    next_pred += noise
                
                predictions.append(float(next_pred))
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
            
            # Denormalize predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            denormalized_predictions = self.scaler.inverse_transform(predictions_array).flatten()
            all_predictions.append(denormalized_predictions)
        
        # Convert to numpy array for easier manipulation
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Calculate trend analysis
        trend_analysis = self._analyze_trend(mean_predictions)
        
        # Calculate volatility
        volatility = self._calculate_volatility(all_predictions)
        
        return {
            'dates': future_dates,
            'predictions': mean_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'standard_deviation': std_predictions,
            'trend_analysis': trend_analysis,
            'volatility': volatility,
            'last_known_price': float(self.last_known_prices[-1]),
            'last_date': self.last_date,
            'symbol': self.symbol
        }
    
    def _analyze_trend(self, predictions):
        """Analyze the trend in predictions"""
        if len(predictions) < 2:
            return {'direction': 'stable', 'strength': 0, 'description': 'Insufficient data'}
        
        # Calculate overall trend
        start_price = predictions[0]
        end_price = predictions[-1]
        total_change = ((end_price - start_price) / start_price) * 100
        
        # Calculate trend strength (consistency)
        price_changes = np.diff(predictions)
        positive_changes = np.sum(price_changes > 0)
        negative_changes = np.sum(price_changes < 0)
        total_changes = len(price_changes)
        
        if total_changes == 0:
            trend_consistency = 0
        else:
            trend_consistency = abs(positive_changes - negative_changes) / total_changes
        
        # Determine trend direction
        if abs(total_change) < 1:
            direction = 'stable'
        elif total_change > 0:
            direction = 'bullish'
        else:
            direction = 'bearish'
        
        # Determine trend strength
        if trend_consistency > 0.7:
            strength = 'strong'
        elif trend_consistency > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        return {
            'direction': direction,
            'strength': strength,
            'total_change_percent': total_change,
            'consistency': trend_consistency,
            'description': f"{strength.capitalize()} {direction} trend with {total_change:.2f}% expected change"
        }
    
    def _calculate_volatility(self, all_predictions):
        """Calculate volatility metrics"""
        # Calculate daily returns for each simulation
        all_returns = []
        for predictions in all_predictions:
            returns = np.diff(predictions) / predictions[:-1]
            all_returns.extend(returns)
        
        volatility = np.std(all_returns) * np.sqrt(252)  # Annualized volatility
        
        return {
            'daily_volatility': np.std(all_returns),
            'annualized_volatility': volatility,
            'description': 'High' if volatility > 0.3 else 'Medium' if volatility > 0.15 else 'Low'
        }
    
    def plot_future_predictions(self, future_data, show_historical_days=30):
        """
        Create comprehensive visualization of future predictions
        
        Args:
            future_data: Output from predict_future_advanced()
            show_historical_days: Number of historical days to show
        """
        # Prepare historical data
        historical_prices = self.last_known_prices[-show_historical_days:]
        historical_dates = pd.date_range(
            end=self.last_date,
            periods=len(historical_prices),
            freq='D'
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=[
                f'{self.symbol} - Future Price Predictions',
                'Prediction Confidence Band Width'
            ],
            vertical_spacing=0.1
        )
        
        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=historical_prices,
                name='Historical Prices',
                line=dict(color='blue', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Plot future predictions
        fig.add_trace(
            go.Scatter(
                x=future_data['dates'],
                y=future_data['predictions'],
                name='Future Predictions',
                line=dict(color='red', width=2),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        # Plot confidence intervals
        fig.add_trace(
            go.Scatter(
                x=future_data['dates'],
                y=future_data['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_data['dates'],
                y=future_data['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{int(future_data["confidence_level"]*100)}% Confidence Interval',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ),
            row=1, col=1
        )
        
        # Plot confidence band width
        band_width = future_data['upper_bound'] - future_data['lower_bound']
        fig.add_trace(
            go.Scatter(
                x=future_data['dates'],
                y=band_width,
                name='Uncertainty',
                line=dict(color='orange', width=1),
                mode='lines'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Stock Price Predictions - {len(future_data["predictions"])} Days Ahead',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=800,
            hovermode='x unified',
            showlegend=True
        )
        
        # Add trend analysis annotation
        trend = future_data['trend_analysis']
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Trend: {trend['description']}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def generate_investment_insights(self, future_data, investment_amount=10000):
        """
        Generate investment insights based on predictions
        
        Args:
            future_data: Output from predict_future_advanced()
            investment_amount: Hypothetical investment amount
            
        Returns:
            dict: Investment insights and recommendations
        """
        current_price = future_data['last_known_price']
        predictions = future_data['predictions']
        trend = future_data['trend_analysis']
        volatility = future_data['volatility']
        
        # Calculate potential returns
        final_price = predictions[-1]
        expected_return = ((final_price - current_price) / current_price) * 100
        
        # Calculate risk metrics
        max_loss = ((min(future_data['lower_bound']) - current_price) / current_price) * 100
        max_gain = ((max(future_data['upper_bound']) - current_price) / current_price) * 100
        
        # Risk assessment
        if volatility['annualized_volatility'] > 0.4:
            risk_level = 'High'
        elif volatility['annualized_volatility'] > 0.2:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Investment recommendation
        if expected_return > 10 and risk_level != 'High':
            recommendation = 'BUY'
        elif expected_return < -5 or risk_level == 'High':
            recommendation = 'SELL/AVOID'
        else:
            recommendation = 'HOLD/NEUTRAL'
        
        # Calculate hypothetical investment outcomes
        shares = investment_amount / current_price
        expected_value = shares * final_price
        expected_profit = expected_value - investment_amount
        
        return {
            'current_price': current_price,
            'predicted_price': final_price,
            'expected_return_percent': expected_return,
            'max_potential_loss_percent': max_loss,
            'max_potential_gain_percent': max_gain,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'trend_direction': trend['direction'],
            'trend_strength': trend['strength'],
            'volatility_level': volatility['description'],
            'investment_simulation': {
                'investment_amount': investment_amount,
                'shares_purchased': shares,
                'expected_final_value': expected_value,
                'expected_profit': expected_profit,
                'expected_profit_percent': (expected_profit / investment_amount) * 100
            },
            'key_insights': self._generate_key_insights(future_data, expected_return, risk_level)
        }
    
    def _generate_key_insights(self, future_data, expected_return, risk_level):
        """Generate key insights for investment decision"""
        insights = []
        
        trend = future_data['trend_analysis']
        volatility = future_data['volatility']
        
        # Trend insights
        if trend['direction'] == 'bullish' and trend['strength'] == 'strong':
            insights.append("Strong upward trend predicted with high consistency")
        elif trend['direction'] == 'bearish' and trend['strength'] == 'strong':
            insights.append("Strong downward trend predicted - consider selling")
        elif trend['strength'] == 'weak':
            insights.append("Trend shows weak consistency - market uncertainty")
        
        # Return insights
        if expected_return > 15:
            insights.append("High potential returns but verify with fundamental analysis")
        elif expected_return < -10:
            insights.append("Significant potential losses predicted")
        
        # Risk insights
        if risk_level == 'High':
            insights.append("High volatility expected - suitable only for risk-tolerant investors")
        elif risk_level == 'Low':
            insights.append("Low volatility suggests stable price movement")
        
        # General insights
        confidence_band_avg = np.mean(future_data['upper_bound'] - future_data['lower_bound'])
        if confidence_band_avg / future_data['last_known_price'] > 0.2:
            insights.append("Wide confidence intervals suggest high prediction uncertainty")
        
        return insights

# Initialize session state for page if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# Handle page navigation
if st.session_state.current_page == 'future_predictions':
    st.experimental_set_query_params()
    st.rerun()

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("Stock Price Predictor")
st.write("Predict stock prices using LSTM neural networks")

# Sidebar inputs
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()

# Date range selection with better defaults and validation
st.sidebar.header("Date Range Selection")
st.sidebar.write("Select the date range for historical data:")

# Set default dates
default_end_date = datetime.now().date()
default_start_date = default_end_date - timedelta(days=365*2)  # 2 years of data

# Date inputs
start_date = st.date_input(
    "Start Date",
    value=default_start_date,
    min_value=datetime(2000, 1, 1).date(),
    max_value=default_end_date,
    help="Select a start date for training data"
)

end_date = st.date_input(
    "End Date",
    value=default_end_date,
    min_value=start_date,
    max_value=default_end_date,
    help="Select an end date for training data"
)

# Convert dates to string format
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Validate date range
if (end_date - start_date).days < 100:
    st.warning("⚠️ Selected date range is too short. Please select at least 100 days of data.")
    st.stop()

if (end_date - start_date).days > 365*5:
    st.warning("⚠️ Selected date range is very long. This may take a while to process.")

# Model parameters
st.sidebar.header("Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60, 
                                  help="Number of past days to consider for each prediction")
epochs = st.sidebar.slider("Training Epochs", 10, 100, 20,
                          help="Number of training iterations (more = longer training time)")
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32,
                              help="Number of samples processed together")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    test_split = st.slider("Test Split Ratio", 0.1, 0.3, 0.2, 0.05,
                          help="Percentage of data used for testing")
    show_training_history = st.checkbox("Show Training History", True)
    confidence_level = st.slider("Prediction Confidence Level", 0.8, 0.99, 0.95, 0.01,
                                help="Confidence level for prediction intervals")
    monte_carlo_runs = st.slider("Monte Carlo Simulations", 50, 200, 100, 10,
                                help="Number of simulations for prediction intervals")

if st.sidebar.button("Train Model", type="primary"):
    try:
        # Input validation
        if start_date >= end_date:
            st.error("❌ Start date must be before end date")
            st.stop()
            
        if not symbol or len(symbol) < 1:
            st.error("❌ Please enter a valid stock symbol")
            st.stop()
            
        if (end_date - start_date).days < sequence_length:
            st.error(f"❌ Date range ({end_date - start_date} days) must be longer than sequence length ({sequence_length} days)")
            st.stop()
        
        # Debug: Show what will be passed to StockPredictor
        st.info(f"**Debug Info:**\nSymbol: {symbol}\nStart Date: {start_date_str}\nEnd Date: {end_date_str}")
        
        # Create columns for status display
        status_col, info_col = st.columns([2, 1])
        
        with status_col:
            with st.spinner("🔄 Initializing and downloading data..."):
                # Initialize predictor with string dates
                predictor = StockPredictor(
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    sequence_length=sequence_length
                )
                
                # Get and prepare data
                data = predictor.get_stock_data()
                st.success(f"✅ Downloaded {len(data)} days of data for {symbol}")
                # Debug: Show first few rows of data
                st.write("**First few rows of downloaded data:**")
                st.write(pd.DataFrame(data).head())
        
        with info_col:
            st.info(f"**Symbol:** {symbol}\n**Start:** {start_date_str}\n**End:** {end_date_str}")
        
        with st.spinner("🔄 Preparing data and building model..."):
            X, y = predictor.prepare_data(data)
            
            # Split data with custom ratio
            train_size = int(len(X) * (1 - test_split))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
            
            # Build model
            predictor.build_model()
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 Training model..."):
            # Custom callback to update progress
            class StreamlitCallback:
                def __init__(self, epochs, progress_bar, status_text):
                    self.epochs = epochs
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.epochs
                    self.progress_bar.progress(progress)
                    loss = logs.get('loss', 0) if logs else 0
                    self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.6f}")
            
            # Train model
            history = predictor.train_model(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size,
                verbose=0  # Suppress console output
            )
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
        
        st.success("✅ Model training completed!")
        
        # Make predictions
        with st.spinner("📊 Generating predictions..."):
            test_predictions = predictor.predict(X_test)
            actual_prices = predictor.scaler.inverse_transform(y_test)
            
            # Also get training predictions for full comparison
            train_predictions = predictor.predict(X_train)
            train_actual = predictor.scaler.inverse_transform(y_train)
        
        # Create comprehensive plot
        fig = go.Figure()
        
        # Create date index for x-axis
        total_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        available_dates = total_dates[:len(data)]
        
        # Training data
        train_dates = available_dates[sequence_length:sequence_length+len(train_actual)]
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_actual.flatten(),
            name="Training Actual",
            line=dict(color="blue", width=1),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=train_predictions.flatten(),
            name="Training Predicted",
            line=dict(color="lightblue", width=1),
            opacity=0.7
        ))
        
        # Test data
        test_dates = available_dates[sequence_length+len(train_actual):sequence_length+len(train_actual)+len(actual_prices)]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=actual_prices.flatten(),
            name="Test Actual",
            line=dict(color="green", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predictions.flatten(),
            name="Test Predicted",
            line=dict(color="red", width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price Prediction ({start_date_str} to {end_date_str})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        mse = ((actual_prices - test_predictions) ** 2).mean()
        mae = np.abs(actual_prices - test_predictions).mean()
        mape = np.mean(np.abs((actual_prices - test_predictions) / actual_prices)) * 100
        
        with col1:
            st.metric("Mean Squared Error", f"${mse:.2f}")
        with col2:
            st.metric("Mean Absolute Error", f"${mae:.2f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            final_loss = history.history['loss'][-1]
            st.metric("Final Training Loss", f"{final_loss:.6f}")
        
        # Add user-friendly summary
        st.markdown("---")
        st.subheader("📊 Model Performance Summary")
        
        # Calculate trend
        last_actual = float(actual_prices[-1])
        last_predicted = float(test_predictions[-1])
        price_change = ((last_predicted - last_actual) / last_actual) * 100
        
        # Store necessary data in session state for future predictions
        st.session_state.predictor = predictor
        st.session_state.end_date = end_date
        st.session_state.last_actual = last_actual
        st.session_state.mape = mape
        st.session_state.data = data  # Store the raw data
        st.session_state.model_trained = True  # Flag to indicate model is trained
        
        # Create a summary message
        summary = f"""
        ### Current Analysis for {symbol}
        
        **Latest Price**: ${last_actual:.2f}
        
        **Model Confidence**: {100 - float(mape):.1f}% accurate
        
        **Price Trend**: {'📈 Upward' if price_change > 0 else '📉 Downward'} trend predicted
        
        **Expected Change**: {abs(price_change):.1f}% {'increase' if price_change > 0 else 'decrease'}
        
        **Model Reliability**: {'High' if float(mape) < 5 else 'Medium' if float(mape) < 10 else 'Low'}
        """
        
        st.markdown(summary)
        
        # Add 24-hour prediction section
        st.markdown("---")
        st.subheader("🕒 24-Hour Price Trend Prediction")
        
        try:
            # Get the last sequence for prediction
            last_sequence = data['Close'].values[-sequence_length:].reshape(-1, 1)
            last_scaled = predictor.scaler.transform(last_sequence)
            
            # Make prediction for next day
            X = last_scaled.reshape(1, sequence_length, 1)
            next_day_pred = predictor.model.predict(X, verbose=0)[0, 0]
            next_day_price = float(predictor.scaler.inverse_transform([[next_day_pred]])[0][0])
            
            # Calculate price change and direction
            current_price = float(data['Close'].iloc[-1])
            price_change = next_day_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Calculate prediction confidence based on recent model accuracy
            recent_actual = actual_prices[-5:].flatten()  # Last 5 days of actual prices
            recent_pred = test_predictions[-5:].flatten()  # Last 5 days of predictions
            recent_mape = np.mean(np.abs((recent_actual - recent_pred) / recent_actual)) * 100
            confidence = 100 - recent_mape
            
            # Create columns for prediction display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Price",
                    f"${next_day_price:.2f}",
                    f"{price_change_percent:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1f}%",
                    f"Based on last 5 days"
                )
            
            with col3:
                trend_icon = "📈" if price_change > 0 else "📉"
                st.metric(
                    "Trend Direction",
                    f"{trend_icon} {'Upward' if price_change > 0 else 'Downward'}",
                    f"Next 24 hours"
                )
            
            # Add detailed prediction analysis
            st.markdown("### 📊 Prediction Analysis")
            
            # Calculate volatility
            recent_volatility = float(np.std(data['Close'].pct_change().dropna().tail(20)) * 100)
            
            # Generate prediction insights
            insights = []
            if abs(price_change_percent) > recent_volatility:
                insights.append(f"Strong price movement expected ({abs(price_change_percent):.1f}% vs {recent_volatility:.1f}% average volatility)")
            else:
                insights.append(f"Moderate price movement expected ({abs(price_change_percent):.1f}% vs {recent_volatility:.1f}% average volatility)")
            
            if confidence > 80:
                insights.append("High confidence prediction based on recent model accuracy")
            elif confidence > 60:
                insights.append("Moderate confidence prediction - consider market conditions")
            else:
                insights.append("Low confidence prediction - exercise caution")
            
            # Display insights
            for insight in insights:
                st.write(f"- {insight}")
            
            # Add disclaimer
            st.markdown("""
            ⚠️ **Disclaimer**: This 24-hour prediction is based on historical patterns and technical analysis. 
            Market conditions can change rapidly, and this prediction should not be used as the sole basis for trading decisions.
            """)
            
        except Exception as e:
            st.error(f"Error generating 24-hour prediction: {str(e)}")
            st.error("Please ensure the model is properly trained and data is available.")
        
        # Display training history
        if show_training_history:
            st.subheader("📈 Training History")
            history_df = pd.DataFrame(history.history)
            st.line_chart(history_df)
        
        # Data summary
        with st.expander("📊 Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Training Data:**")
                st.write(f"- Samples: {len(X_train)}")
                st.write(f"- Date range: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}")
                
            with col2:
                st.write("**Test Data:**")
                st.write(f"- Samples: {len(X_test)}")
                st.write(f"- Date range: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
        
    except ValueError as ve:
        st.error(f"❌ Data Error: {str(ve)}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

# Add enhanced information section
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 About the Model
- **LSTM Neural Networks**: Uses Long Short-Term Memory networks designed for time series prediction
- **Historical Data**: Downloads real stock data from Yahoo Finance
- **Sequence Learning**: Learns patterns from past price movements
- **Date Range**: Select your desired historical period for training

### 💡 Tips for Better Results
- **Longer periods**: Use 2-3 years of data for better training
- **Recent data**: Include recent market conditions
- **Stable stocks**: Well-established stocks often have more predictable patterns
- **Market hours**: Stock markets are closed on weekends and holidays

### ⚙️ Parameter Guide
- **Sequence Length**: Days of history to consider (30-120 recommended)
- **Epochs**: Training iterations (20-50 for quick results)
- **Batch Size**: Training efficiency (32 is usually good)
- **Test Split**: Portion for validation (20% recommended)

### 📊 Popular Stock Symbols
- **Tech**: AAPL, GOOGL, MSFT, TSLA
- **Finance**: JPM, BAC, GS, WFC  
- **Market**: SPY (S&P 500), QQQ (NASDAQ)
""")

# Add disclaimer
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer**: This tool is for educational purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research before making investment decisions.
""")