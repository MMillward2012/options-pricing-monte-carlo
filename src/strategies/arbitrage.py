# src/strategies/arbitrage.py

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.market_data.fetcher import get_risk_free_rate

class ArbitrageScanner:
    """Scans for executable put-call parity violations considering bid-ask spreads."""
    
    def __init__(self, tickers=['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']):
        self.tickers = tickers
        self.risk_free_rate = get_risk_free_rate()
        
    def scan_put_call_parity_violations(self):
        """
        Finds arbitrage opportunities where the theoretical profit is greater
        than the cost of crossing the bid-ask spread.
        """
        print(f"Scanning for EXECutable Put-Call Parity violations using r={self.risk_free_rate:.2%}...")
        violations = []
        
        for ticker in self.tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                S0 = ticker_obj.history(period='1d')['Close'].iloc[-1]
                expiry = ticker_obj.options[0]
                chain = ticker_obj.option_chain(expiry)
                
                merged = pd.merge(
                    chain.calls, chain.puts, on='strike', suffixes=('_call', ' '
                                                                             '_put')
                )
                
                # Filter for options that have a valid bid and ask
                liquid = merged[
                    (merged['volume_call'] > 10) & (merged['volume_put'] > 10) &
                    (merged['bid_call'] > 0) & (merged['ask_put'] > 0)
                ].copy()
                
                if liquid.empty: continue
                    
                T = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days / 365.0
                if T <= 0: continue

                # THEORETICAL VALUE
                liquid['theoretical_value'] = S0 - liquid['strike'] * np.exp(-self.risk_free_rate * T)

                # STRATEGY 1: Synthetic Long (Buy Call, Sell Put)
                # We buy the call at its ask, and sell the put at its bid.
                liquid['strategy1_cost'] = liquid['ask_call'] - liquid['bid_put']
                liquid['strategy1_profit'] = liquid['theoretical_value'] - liquid['strategy1_cost']

                # STRATEGY 2: Synthetic Short (Sell Call, Buy Put)
                # We sell the call at its bid, and buy the put at its ask.
                liquid['strategy2_cost'] = liquid['ask_put'] - liquid['bid_call']
                liquid['strategy2_profit'] = liquid['strategy2_cost'] - liquid['theoretical_value']

                # Identify profitable arbitrage opportunities
                strat1_opps = liquid[liquid['strategy1_profit'] > 0].copy()
                strat1_opps['strategy'] = 'Buy Synthetic Long'
                strat1_opps['executable_profit'] = strat1_opps['strategy1_profit']

                strat2_opps = liquid[liquid['strategy2_profit'] > 0].copy()
                strat2_opps['strategy'] = 'Buy Synthetic Short'
                strat2_opps['executable_profit'] = strat2_opps['strategy2_profit']

                found = pd.concat([strat1_opps, strat2_opps])
                
                if not found.empty:
                    found['ticker'] = ticker
                    violations.append(found)
                        
            except Exception:
                continue
        
        if not violations:
            return pd.DataFrame()
            
        return pd.concat(violations, ignore_index=True)