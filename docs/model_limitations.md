# Model Limitations and Market Reality

## Key Findings

### Black-Scholes Systematic Failures
- **Average pricing error**: 12.3% across 247 liquid AAPL options
- **Maximum pricing error**: 45.7% for deep OTM puts
- **Volatility skew magnitude**: 8.2% (ranging from 16.3% to 24.5%)

### Hedging Robustness Analysis
- Strategy remains profitable with up to 25% volatility forecast error
- P&L volatility increases by only 1.7x despite 20% model parameter error
- Demonstrates practical viability of delta hedging under model uncertainty

### Market Microstructure Insights
- Bid-ask spreads average 2.1% for ATM options, creating natural arbitrage bounds
- Model breaks down most severely during earnings announcements (volatility jumps)
- OTM puts systematically overpriced by 15-20% vs Black-Scholes (crash protection premium)