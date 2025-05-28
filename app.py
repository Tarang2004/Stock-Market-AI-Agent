import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from stock_predictor import StockPredictor
import plotly.graph_objects as go
import numpy as np

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
        
        # Add future predictions
        st.markdown("---")
        st.subheader("🔮 Future Price Predictions")
        
        # Add a slider for number of days to predict
        future_days = st.slider("Number of days to predict", 1, 30, 5)
        
        if st.button("Predict Future Prices"):
            try:
                with st.spinner("Generating future predictions..."):
                    # Get future predictions
                    future_predictions = predictor.predict_future(days=future_days)
                    
                    # Create a DataFrame for future predictions
                    future_dates = pd.date_range(
                        start=end_date + timedelta(days=1),
                        periods=future_days
                    )
                    
                    # Create DataFrame with predictions
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    })
                    
                    # Calculate daily changes
                    future_df['Daily Change'] = future_df['Predicted Price'].pct_change() * 100
                    future_df['Daily Change'] = future_df['Daily Change'].fillna(0)
                    
                    # Display the predictions
                    st.write("### Predicted Prices")
                    st.dataframe(future_df.style.format({
                        'Predicted Price': '${:.2f}',
                        'Daily Change': '{:+.2f}%'
                    }))
                    
                    # Plot future predictions
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=available_dates,
                        y=data['Close'],
                        name="Historical Prices",
                        line=dict(color="blue")
                    ))
                    
                    # Add future predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        name="Future Predictions",
                        line=dict(color="red", dash="dash")
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Prediction ({start_date_str} to {future_dates[-1].strftime('%Y-%m-%d')})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("### 📝 Interpretation")
                    interpretation = f"""
                    Based on the model's analysis:
                    
                    - **Short-term Trend**: {'Upward' if float(future_predictions[-1]) > last_actual else 'Downward'}
                    - **Expected Range**: ${float(min(future_predictions)):.2f} - ${float(max(future_predictions)):.2f}
                    - **Volatility**: {'High' if float(np.std(future_predictions)) > float(np.std(data['Close'][-30:])) else 'Low'}
                    - **Confidence Level**: {'High' if float(mape) < 5 else 'Medium' if float(mape) < 10 else 'Low'}
                    
                    ⚠️ **Disclaimer**: These predictions are based on historical patterns and should not be used as the sole basis for investment decisions.
                    """
                    st.markdown(interpretation)
                    
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.error("Please make sure the model is properly trained before generating predictions.")
        
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