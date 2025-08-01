# src/strategies/arbitrage.py

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# NEW: Import our new function
from src.market_data.fetcher import get_risk_free_rate

class ArbitrageScanner:
    """Scans for put-call parity violations in real time."""
    
    def __init__(self, tickers=['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']):
        self.tickers = tickers
        # NEW: Fetch the live risk-free rate on initialization
        self.risk_free_rate = get_risk_free_rate()
        
    def scan_put_call_parity_violations(self, violation_threshold_pct=1.0):
        """
        Put-Call Parity: C - P = S - K*e^(-rT)
        When this breaks, there's a potential arbitrage opportunity.
        """
        print(f"Scanning for Put-Call Parity violations using r={self.risk_free_rate:.2%}...")
        violations = []
        
        for ticker in self.tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                S0 = ticker_obj.history(period='1d')['Close'].iloc[-1]
                expiry = ticker_obj.options[0]
                chain = ticker_obj.option_chain(expiry)
                
                # Merge calls and puts on the same strike
                merged = pd.merge(
                    chain.calls, chain.puts, on='strike', suffixes=('_call', '_put')
                )
                
                # Filter for liquid options
                liquid = merged[(merged['volume_call'] > 10) & (merged['volume_put'] > 10)].copy()
                
                if liquid.empty: continue
                    
                # Calculate parity
                T = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365.0
                liquid['theoretical_diff'] = S0 - liquid['strike'] * np.exp(-self.risk_free_rate * T)
                liquid['actual_diff'] = liquid['lastPrice_call'] - liquid['lastPrice_put']
                liquid['violation_amt'] = (liquid['actual_diff'] - liquid['theoretical_diff']).abs()
                liquid['violation_pct'] = (liquid['violation_amt'] / liquid['lastPrice_call']) * 100

                # Find significant violations
                found = liquid[liquid['violation_pct'] > violation_threshold_pct].copy()
                if not found.empty:
                    found['ticker'] = ticker
                    violations.append(found)
                        
            except Exception:
                continue
        
        if not violations:
            return pd.DataFrame()
            
        return pd.concat(violations, ignore_index=True)