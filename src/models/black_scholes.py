# src/models/black_scholes.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call_price(S0, K, T, r, sigma):
    """Calculates the analytical Black-Scholes price for a European call."""
    if sigma <= 0 or T <= 0:
        return np.maximum(0, S0 - K * np.exp(-r * T))
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def black_scholes_delta(S0, K, T, r, sigma):
    """Calculates the analytical Black-Scholes delta for a European call."""
    if sigma <= 0 or T <= 0:
        return 1.0 if S0 > K else 0.0
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def find_implied_volatility(target_price, S0, K, T, r):
    """Inverts the Black-Scholes formula to find implied volatility."""
    objective_func = lambda sigma: black_scholes_call_price(S0, K, T, r, sigma) - target_price
    
    try:
        # Search within a reasonable volatility bracket
        return brentq(objective_func, 0.01, 5.0) 
    except (ValueError, RuntimeError):
        return np.nan