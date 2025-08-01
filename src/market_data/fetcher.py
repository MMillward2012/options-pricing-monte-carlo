# src/market_data/fetcher.py

import yfinance as yf
from datetime import datetime

def get_risk_free_rate():
    """Fetches the current 13-week Treasury Bill yield as the risk-free rate."""
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.history(period='1d')['Close'].iloc[-1]
        return rate / 100  # Convert from percentage to decimal
    except Exception:
        print("Could not fetch live risk-free rate. Defaulting to 5.0%.")
        return 0.05

def get_option_chain_and_market_data(ticker_symbol='AAPL', expiry_idx=0):
    """Fetches option chain and relevant market data for a given ticker."""
    ticker = yf.Ticker(ticker_symbol)
    
    expiry_dates = ticker.options
    if not expiry_dates or len(expiry_dates) <= expiry_idx:
        raise ValueError(f"No expiry dates found for {ticker_symbol} at index {expiry_idx}.")
    
    expiry_date = expiry_dates[expiry_idx]
    
    option_chain = ticker.option_chain(expiry_date)
    calls = option_chain.calls
    puts = option_chain.puts # Also fetch puts for arbitrage
    
    T = (datetime.strptime(expiry_date, '%Y-%m-%d') - datetime.now()).days / 365.0
    S0 = ticker.history(period='1d')['Close'].iloc[-1]
    
    return calls, puts, S0, T, expiry_date