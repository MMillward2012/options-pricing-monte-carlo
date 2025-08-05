# src/risk/hedging.py

import numpy as np
import pandas as pd
import time

# Import our custom modules
from src.models.black_scholes import black_scholes_call_price, black_scholes_delta
from src.models.monte_carlo import simulate_gbm_paths

class DeltaHedgingEngine:
    """
    Production-grade delta hedging simulation engine.
    """
    
    def __init__(self, transaction_cost=0.001):
        self.transaction_cost = transaction_cost
        
    def simulate_hedging_pnl(self, S0, K, T, r, sigma, n_simulations, steps=252):
        """
        Main hedging simulation method.
        """
        return simulate_delta_hedging(S0, K, T, r, sigma, n_simulations, steps)
        
    def analyze_hedging_performance(self, S0, K, T, r, sigma, n_simulations=10000):
        """
        Comprehensive hedging performance analysis.
        """
        pnl_results = self.simulate_hedging_pnl(S0, K, T, r, sigma, n_simulations)
        
        stats = {
            'mean_pnl': np.mean(pnl_results),
            'std_pnl': np.std(pnl_results),
            'var_95': np.percentile(pnl_results, 5),
            'var_99': np.percentile(pnl_results, 1),
            'max_loss': np.min(pnl_results),
            'max_gain': np.max(pnl_results)
        }
        
        return stats, pnl_results

def simulate_delta_hedging(S0, K, T, r, sigma, n_simulations, steps=252):
    """
    Simulates a delta hedging strategy over multiple Monte Carlo paths.
    Returns the distribution of final Profit and Loss (P&L).
    """
    dt = T / steps
    
    # Step 1: Generate all possible stock price paths
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_simulations)
    
    # Initialize cash account to track P&L
    cash_account = np.zeros(n_simulations)
    
    # Step 2: Set up the initial hedge at time t=0
    # We are short one call, so we receive the premium
    initial_option_price = black_scholes_call_price(S0, K, T, r, sigma)
    cash_account += initial_option_price
    
    # Hedge by buying delta shares
    delta = black_scholes_delta(S0, K, T, r, sigma)
    stock_holdings = delta
    cash_account -= stock_holdings * S0

    # Step 3: Simulate the hedging process through time for each path
    for t in range(1, steps):
        # Earn interest on cash
        cash_account *= np.exp(r * dt)
        
        # Calculate new delta for the next period's hedge
        time_remaining = T - (t * dt)
        new_delta = black_scholes_delta(paths[t], K, time_remaining, r, sigma)

        # Rebalance the hedge: buy/sell shares
        cash_account -= (new_delta - stock_holdings) * paths[t]
        stock_holdings = new_delta

    # Step 4: Final settlement at expiry
    # Close out stock position
    cash_account += stock_holdings * paths[-1]
    
    # Settle the option payoff (we are short, so we pay out)
    final_option_payoff = np.maximum(paths[-1] - K, 0)
    final_pnl = cash_account - final_option_payoff
    
    return final_pnl