import pandas as pd
import yfinance as yf

# ADDED ^VIX and ^TNX to the default tickers list
def load_financial_data(tickers=["SPY", "TLT", "GLD", "^VIX", "^TNX"], start_date="2010-01-01", end_date="2026-06-01"):
    """
    Downloads historical Close prices and computes daily percentage returns.
    Includes VIX and 10-Year Treasury Yield for macro context.
    """
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    
    data = yf.download(tickers, start=start_date, end=end_date)
    close_prices = data['Close'].dropna()
    
    # Calculate daily percentage returns/changes
    returns = close_prices.pct_change().dropna()
    
    print(f"Data successfully loaded. Shape: {returns.shape}")
    return close_prices, returns

if __name__ == "__main__":
    prices, rets = load_financial_data()
    print("\nSample Data (Notice VIX and TNX columns):")
    print(rets.head())