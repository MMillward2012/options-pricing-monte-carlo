# Options Pricing Monte Carlo Engine

A high-performance quantitative finance toolkit for options pricing, risk management, and market analysis using Monte Carlo simulation with institutional-grade accuracy and speed.

## 🎯 Overview

This project implements a comprehensive options pricing and analysis system achieving **99.99% accuracy** against Black-Scholes analytical solutions while delivering **138M+ simulations per second** through optimized Monte Carlo engines. The system integrates real-time market data, advanced risk management, and arbitrage detection capabilities.

## ✨ Key Features

### High-Performance Computing
- **138M+ simulations/second** with Numba-optimized Monte Carlo engine
- **99.99% accuracy** against Black-Scholes analytical pricing
- Antithetic variance reduction and control variates optimization
- Memory-efficient algorithms for large-scale simulations

### Comprehensive Options Analysis
- Complete Black-Scholes and Monte Carlo pricing implementations
- Full Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Put-call parity validation and arbitrage detection
- Volatility surface analysis and implied volatility calculations

### Risk Management Tools
- Delta hedging simulation with transaction cost modeling
- Historical backtesting framework with regime analysis
- Stress testing across market crash scenarios
- Liquidity modeling and market impact analysis

### Real-Time Market Integration
- Live options data fetching via Yahoo Finance API
- Automated data cleaning and validation
- CSV/TXT export with timestamped results
- Multi-asset portfolio analysis capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NumPy, pandas, matplotlib, seaborn
- numba, yfinance, scipy

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MMillward2012/options-pricing-monte-carlo.git
cd options-pricing-monte-carlo
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the main analysis:**
```bash
jupyter notebook notebooks/main.ipynb
```

### Basic Usage

```python
from src.models.monte_carlo import OptimizedMCEngine
from src.models.black_scholes import BlackScholesModel

# Initialize high-performance Monte Carlo engine
engine = OptimizedMCEngine(use_numba=True, use_antithetic=True)

# Price European call option
result = engine.price_with_greeks(
    S0=100,     # Current stock price
    K=105,      # Strike price
    T=1.0,      # Time to expiration (years)
    vol=0.2,    # Volatility
    n_sims=100000
)

print(f"Option Price: ${result['price']:.4f}")
print(f"Delta: {result['delta']:.4f}")
print(f"Accuracy vs Black-Scholes: {result['accuracy']:.2%}")
```

## 📊 Performance Benchmarks

| Engine Type | Simulations/Second | Accuracy vs Analytical | Memory Usage |
|-------------|-------------------|------------------------|--------------|
| Standard Monte Carlo | 2.5M | 99.5% | High |
| Optimized (Numba) | 138M+ | 99.99% | Medium |
| With Variance Reduction | 164M+ | 99.99% | Medium |

## 🏗️ Project Structure

```
options-pricing-monte-carlo/
├── src/
│   ├── models/              # Core pricing models
│   │   ├── monte_carlo.py   # Optimized Monte Carlo engines
│   │   ├── black_scholes.py # Analytical pricing models
│   │   └── model_validation.py # Accuracy testing
│   ├── risk/                # Risk management tools
│   │   ├── hedging.py       # Delta hedging simulation
│   │   ├── backtesting.py   # Historical analysis
│   │   └── greeks.py        # Greeks calculation
│   ├── strategies/          # Trading strategies
│   │   ├── arbitrage.py     # Arbitrage detection
│   │   └── market_impact.py # Market impact modeling
│   ├── market_data/         # Data fetching and processing
│   │   ├── fetcher.py       # Live data retrieval
│   │   └── cleaner.py       # Data validation
│   └── utils/               # Utility functions
├── notebooks/
│   └── main.ipynb          # Comprehensive analysis notebook
├── results/                # Output files and charts
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## 📈 Advanced Features

### Arbitrage Detection
```python
from src.strategies.arbitrage import ArbitrageScanner

scanner = ArbitrageScanner(['AAPL', 'MSFT', 'NVDA'])
opportunities = scanner.find_put_call_parity_violations()
profit_potential = scanner.calculate_arbitrage_value(opportunities)
```

### Delta Hedging Simulation
```python
from src.risk.hedging import DeltaHedgingEngine

hedging_engine = DeltaHedgingEngine(
    transaction_cost_bp=5,  # 5 basis points
    rebalance_threshold=0.1
)
pnl_results = hedging_engine.simulate_hedging(
    option_params, market_data, n_days=252
)
```

### Market Regime Analysis
```python
from src.risk.regime_analysis import MarketRegimeAnalyzer

analyzer = MarketRegimeAnalyzer()
regime_summary = analyzer.analyze_market_conditions(['AAPL', 'SPY'])
```

## 🧪 Testing

Run the test suite to validate all components:

```bash
python -m pytest tests/ -v
```

## 📝 Documentation

- **[Options Basics](docs/Options_basic.md)** - Introduction to options theory
- **[Model Limitations](docs/model_limitations.md)** - Understanding model constraints
- **[Stochastic Intuition](docs/stochastic_intuition.md)** - Monte Carlo methodology

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [QuantLib](https://www.quantlib.org/) - Comprehensive quantitative finance library
- [PyQL](https://github.com/enthought/pyql) - Python bindings for QuantLib
- [zipline](https://github.com/quantopian/zipline) - Algorithmic trading library

## 📧 Contact

Matthew Millward - [GitHub](https://github.com/MMillward2012)

Project Link: [https://github.com/MMillward2012/options-pricing-monte-carlo](https://github.com/MMillward2012/options-pricing-monte-carlo)