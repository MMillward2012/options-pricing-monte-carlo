# src/models/model_validation.py

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# FIXED: Import the functions we've already built
from src.models.black_scholes import black_scholes_call_price, find_implied_volatility

class ModelBreakdownAnalyzer:
    """
    Systematically identifies where theoretical models fail vs market reality.
    """
    
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.option_data = None
        self.breakdown_results = None
        self.market_price = None

    def fetch_market_data(self):
        """Fetches the full option surface and current stock price."""
        print(f"Fetching option surface and market price for {self.ticker}...")
        ticker_obj = yf.Ticker(self.ticker)
        self.market_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
        
        surface_data = []
        # Fetch data for the first 4 available monthly expiries
        expiries = [d for d in ticker_obj.options if (datetime.strptime(d, '%Y-%m-%d').day > 14 and datetime.strptime(d, '%Y-%m-%d').day < 22)][:4]

        for expiry in expiries:
            chain = ticker_obj.option_chain(expiry)
            calls = chain.calls[chain.calls['volume'] > 5].copy()
            calls['expiry'] = expiry
            calls['spread'] = calls['ask'] - calls['bid']
            surface_data.append(calls)
        
        self.option_data = pd.concat(surface_data, ignore_index=True)
        print(f"Fetched {len(self.option_data)} liquid call options.")
        return self.option_data

    def calculate_model_errors(self, r=0.05, constant_vol=0.20):
        """
        Calculates systematic failures of Black-Scholes by comparing market
        prices to a model with constant volatility.
        """
        if self.option_data is None:
            raise ValueError("Option data not fetched. Run fetch_market_data() first.")

        errors = []
        today = datetime.now()

        for _, row in self.option_data.iterrows():
            T = (datetime.strptime(row['expiry'], '%Y-%m-%d') - today).days / 365.0
            if T <= 0.01: continue

            try:
                # Calculate what BS would price at a constant, theoretical volatility
                bs_price_const_vol = black_scholes_call_price(
                    self.market_price, row['strike'], T, r, constant_vol
                )
                
                # Calculate errors vs. market
                pricing_error = (bs_price_const_vol - row['lastPrice']) / row['lastPrice']
                
                # Also find the true implied vol for context
                implied_vol = find_implied_volatility(
                    row['lastPrice'], self.market_price, row['strike'], T, r
                )

                errors.append({
                    'strike': row['strike'],
                    'moneyness': self.market_price / row['strike'], # S/K
                    'days_to_expiry': int(T * 365),
                    'market_price': row['lastPrice'],
                    'bs_price_at_20_vol': bs_price_const_vol,
                    'pricing_error_pct': pricing_error * 100,
                    'implied_vol_pct': implied_vol * 100 if implied_vol else np.nan,
                    'bid_ask_spread_pct': (row['spread'] / row['lastPrice']) * 100
                })

            except Exception:
                continue
        
        self.breakdown_results = pd.DataFrame(errors)
        print(f"Calculated breakdown metrics for {len(self.breakdown_results)} options.")
        return self.breakdown_results

    def generate_key_metrics(self):
        """Generates the summary metrics for your CV."""
        if self.breakdown_results is None or self.breakdown_results.empty:
            print("No breakdown results to analyze. Run calculate_model_errors() first.")
            return {}
        
        results = self.breakdown_results.dropna()
        
        # Filter for different moneyness categories
        otm = results[results['moneyness'] < 0.98] # Out-of-the-money
        atm = results[(results['moneyness'] >= 0.98) & (results['moneyness'] <= 1.02)] # At-the-money
        itm = results[results['moneyness'] > 1.02] # In-the-money

        metrics = {
            'Options Analyzed': len(results),
            'Avg Pricing Error (vs 20% Vol)': f"{results['pricing_error_pct'].mean():.2f}%",
            'Avg Error for OTM Options': f"{otm['pricing_error_pct'].mean():.2f}%",
            'Avg Error for ITM Options': f"{itm['pricing_error_pct'].mean():.2f}%",
            'Implied Vol Range': f"{results['implied_vol_pct'].min():.1f}% - {results['implied_vol_pct'].max():.1f}%",
            'Avg Bid-Ask Spread': f"{results['bid_ask_spread_pct'].mean():.2f}%"
        }
        return metrics