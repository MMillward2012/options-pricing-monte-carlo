# src/models/monte_carlo.py

import numpy as np
import pandas as pd
import time
import psutil
import os

from numba import jit, prange
from src.models.black_scholes import black_scholes_call_price

# ADDED: Standalone path generation function for hedging simulations
def simulate_gbm_paths(S0, r, sigma, T, steps, n_simulations):
    """Simulates multiple Geometric Brownian Motion paths in a vectorized way."""
    dt = T / steps
    # Generate random shocks for each step and simulation
    Z = np.random.normal(0, 1, size=(steps, n_simulations))
    
    # Initialize price array
    prices = np.zeros((steps + 1, n_simulations))
    prices[0] = S0
    
    # Simulate paths
    for t in range(1, steps + 1):
        prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
        
    return prices

@jit(nopython=True, parallel=True)
def _numba_european_call(S0, K, T, r, sigma, n_sims, use_antithetic):
    """
    Numba-optimized pricing function.
    Includes logic for antithetic variates.
    """
    drift = (r - 0.5 * sigma**2) * T
    vol_sqrt_t = sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    
    total_payoff = 0.0
    
    if use_antithetic:
        for i in prange(n_sims // 2):
            Z = np.random.normal(0.0, 1.0)
            ST1 = S0 * np.exp(drift + vol_sqrt_t * Z)
            payoff1 = max(ST1 - K, 0.0)
            ST2 = S0 * np.exp(drift + vol_sqrt_t * -Z)
            payoff2 = max(ST2 - K, 0.0)
            total_payoff += (payoff1 + payoff2)
    else:
        for i in prange(n_sims):
            Z = np.random.normal(0.0, 1.0)
            ST = S0 * np.exp(drift + vol_sqrt_t * Z)
            total_payoff += max(ST - K, 0.0)
            
    return discount * (total_payoff / n_sims)


class OptimizedMCEngine:
    """
    Production-grade Monte Carlo engine focused on speed and efficiency.
    """
    
    def __init__(self, use_numba=True, use_antithetic=True):
        self.use_numba = use_numba
        self.use_antithetic = use_antithetic
        if self.use_numba:
            _numba_european_call(100.0, 100.0, 1.0, 0.05, 0.2, 10, self.use_antithetic)

    @staticmethod
    def _get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _python_european_call(self, S0, K, T, r, sigma, n_sims):
        if self.use_antithetic:
            half_n = n_sims // 2
            Z = np.random.normal(0, 1, half_n)
            Z_full = np.concatenate([Z, -Z])
        else:
            Z_full = np.random.normal(0, 1, n_sims)
            
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
        payoffs = np.maximum(ST - K, 0)
        discounted_payoffs = np.exp(-r * T) * payoffs
        return np.mean(discounted_payoffs)

    def price_european_call(self, S0, K, T, r, sigma, n_sims):
        if self.use_numba:
            return _numba_european_call(S0, K, T, r, sigma, n_sims, self.use_antithetic)
        else:
            return self._python_european_call(S0, K, T, r, sigma, n_sims)

    def benchmark_performance(self, S0=100, K=105, T=1.0, r=0.05, sigma=0.2):
        simulation_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        results = []
        
        try:
            analytical_price = black_scholes_call_price(S0, K, T, r, sigma)
        except Exception as e:
            print(f"Could not calculate analytical price: {e}")
            return None
            
        print(f"--- Benchmarking Engine (Numba: {self.use_numba}, Antithetic: {self.use_antithetic}) ---")
        
        for n_sims in simulation_sizes:
            np.random.seed(42)
            start_time = time.time()
            start_memory = self._get_memory_usage()
            mc_price = self.price_european_call(S0, K, T, r, sigma, n_sims)
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            computation_time = end_time - start_time
            sims_per_second = n_sims / computation_time if computation_time > 0 else float('inf')
            memory_used = end_memory - start_memory
            accuracy = 100 * (1 - abs(mc_price - analytical_price) / analytical_price)
            
            results.append({
                'simulations': f"{n_sims:,}",
                'time_seconds': f"{computation_time:.4f}",
                'sims_per_second': f"{int(sims_per_second):,}",
                'memory_mb': f"{memory_used:.2f}",
                'accuracy_pct': f"{accuracy:.4f}%",
                'mc_price': f"{mc_price:.5f}"
            })
            
        df = pd.DataFrame(results)
        df['analytical_price'] = f"{analytical_price:.5f}"
        return df