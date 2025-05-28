"""
Test script to diagnose yfinance issues
Run this to check if yfinance is working properly
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys

def test_yfinance_connection():
    """Test different methods of fetching data from Yahoo Finance"""
    
    print("=== Testing yfinance connectivity ===\n")
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
    
    # Test date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    print(f"Test date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"yfinance version: {yf.__version__}")
    print()
    
    for symbol in test_symbols:
        print(f"Testing {symbol}...")
        
        # Method 1: yf.download
        try:
            data1 = yf.download(
                symbol, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            if not data1.empty:
                print(f"  ✅ yf.download: Success - {len(data1)} days")
            else:
                print(f"  ❌ yf.download: No data returned")
        except Exception as e:
            print(f"  ❌ yf.download: Error - {str(e)}")
        
        # Method 2: Ticker.history
        try:
            ticker = yf.Ticker(symbol)
            data2 = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            if not data2.empty:
                print(f"  ✅ Ticker.history: Success - {len(data2)} days")
            else:
                print(f"  ❌ Ticker.history: No data returned")
        except Exception as e:
            print(f"  ❌ Ticker.history: Error - {str(e)}")
        
        # Method 3: Period-based
        try:
            data3 = ticker.history(period="1mo")
            if not data3.empty:
                print(f"  ✅ Period-based: Success - {len(data3)} days")
                print(f"    Latest price: ${data3['Close'].iloc[-1]:.2f}")
                print(f"    Date range: {data3.index[0].strftime('%Y-%m-%d')} to {data3.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"  ❌ Period-based: No data returned")
        except Exception as e:
            print(f"  ❌ Period-based: Error - {str(e)}")
        
        print()

def test_specific_symbol_and_date():
    """Test with specific symbol and date range that might be causing issues"""
    
    print("=== Testing specific scenarios ===\n")
    
    symbol = "AAPL"
    
    # Test different date ranges
    test_cases = [
        # Recent data
        {
            'name': 'Recent 30 days',
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        },
        # 1 year data
        {
            'name': '1 year data',
            'start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        },
        # Fixed historical range
        {
            'name': 'Fixed historical (2023)',
            'start': '2023-01-01',
            'end': '2023-12-31'
        },
        # Very recent range
        {
            'name': 'Last 5 days',
            'start': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d')
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing {test_case['name']}: {test_case['start']} to {test_case['end']}")
        
        try:
            data = yf.download(
                symbol,
                start=test_case['start'],
                end=test_case['end'],
                progress=False
            )
            
            if not data.empty:
                print(f"  ✅ Success: {len(data)} days")
                print(f"  📊 Columns: {list(data.columns)}")
                if len(data) > 0:
                    print(f"  💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            else:
                print(f"  ❌ No data returned")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
        
        print()

def check_internet_and_yahoo():
    """Check internet connectivity and Yahoo Finance access"""
    
    print("=== Checking connectivity ===\n")
    
    try:
        import requests
        
        # Test general internet
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connection: OK")
        else:
            print("❌ Internet connection: Issues")
            
        # Test Yahoo Finance
        response = requests.get("https://finance.yahoo.com", timeout=10)
        if response.status_code == 200:
            print("✅ Yahoo Finance access: OK")
        else:
            print(f"❌ Yahoo Finance access: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connectivity test failed: {str(e)}")
        print("Try: pip install requests")
    
    print()

def test_stock_download(symbol="AAPL", days=30):
    """
    Test stock data download using different methods
    """
    print(f"\nTesting stock data download for {symbol}")
    
    # Calculate date range (using historical data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    
    # Method 1: Using yf.download
    print("Method 1: yf.download")
    try:
        data1 = yf.download(symbol, start=start_date, end=end_date)
        print(f"Data shape: {data1.shape}")
        print("First few rows:")
        print(data1.head())
        print("\n")
    except Exception as e:
        print(f"Error with yf.download: {str(e)}")
    
    # Method 2: Using Ticker.history
    print("Method 2: Ticker.history")
    try:
        ticker = yf.Ticker(symbol)
        data2 = ticker.history(start=start_date, end=end_date)
        print(f"Data shape: {data2.shape}")
        print("First few rows:")
        print(data2.head())
        print("\n")
    except Exception as e:
        print(f"Error with Ticker.history: {str(e)}")
    
    # Method 3: Using period-based approach
    print("Method 3: Period-based")
    try:
        data3 = ticker.history(period=f"{days}d")
        print(f"Data shape: {data3.shape}")
        print("First few rows:")
        print(data3.head())
        print("\n")
    except Exception as e:
        print(f"Error with period-based approach: {str(e)}")
    
    print("Test Summary:")
    if 'data1' in locals() and not data1.empty:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify the stock symbol is correct")
        print("3. Try updating yfinance: pip install --upgrade yfinance")
        print("4. Check if Yahoo Finance is accessible in your region")

def test_app_parameters():
    """Test with the exact parameters being used in the app"""
    print("\n=== Testing with App Parameters ===")
    
    # Test parameters
    symbol = "AAPL"
    start_date = "2023-05-29"
    end_date = "2024-05-14"  # Using current date instead of future date
    
    print(f"\nTesting with parameters:")
    print(f"Symbol: {symbol}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    
    try:
        # Try yf.download
        print("\nTrying yf.download...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        print(f"Data shape: {data.shape}")
        print("First few rows:")
        print(data.head())
        
        # Try Ticker.history
        print("\nTrying Ticker.history...")
        ticker = yf.Ticker(symbol)
        data2 = ticker.history(start=start_date, end=end_date)
        print(f"Data shape: {data2.shape}")
        print("First few rows:")
        print(data2.head())
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    """Run all diagnostic tests"""
    
    print("🔍 yfinance Diagnostic Tool")
    print("=" * 50)
    
    # Check versions
    print(f"Python version: {sys.version}")
    try:
        print(f"yfinance version: {yf.__version__}")
        print(f"pandas version: {pd.__version__}")
    except:
        print("Could not get package versions")
    
    print("\n")
    
    # Run tests
    check_internet_and_yahoo()
    test_yfinance_connection()
    test_specific_symbol_and_date()
    
    print("=== Recommendations ===")
    print("If tests failed, try:")
    print("1. Update yfinance: pip install --upgrade yfinance")
    print("2. Check your internet connection")
    print("3. Try again later (Yahoo may be blocking requests)")
    print("4. Use a VPN if you're in a restricted region")
    print("5. Consider using alternative data sources")

    print("\nStarting yfinance test...")
    print("=" * 50)
    
    # Test with different symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        test_stock_download(symbol)
    
    # Test with invalid symbol
    print("\nTesting invalid symbol...")
    test_stock_download("INVALID_SYMBOL")

    # Test app parameters
    test_app_parameters()

if __name__ == "__main__":
    main()