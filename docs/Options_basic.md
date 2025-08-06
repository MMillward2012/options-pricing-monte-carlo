# Financial Options: From Mathematical Contracts to Market Reality

## What Are Financial Options? A Mathematical Perspective

An **option** is a financial contract that grants the holder a **right without obligation**. Mathematically, it's a function that maps future asset prices to payoffs, but with an asymmetric structure that creates fascinating pricing challenges.

Think of it as a mathematical "insurance policy" or "lottery ticket" - you pay a premium upfront for the possibility of future gain, but your maximum loss is limited to what you paid.

### The Core Mathematical Structure

Options are fundamentally **contingent claims** - their value depends entirely on the behavior of an underlying asset (stock, commodity, index, etc.).

**Call Option**: Right to **buy** the underlying asset at strike price K
- Payoff function: `max(S_T - K, 0)`
- Mathematical intuition: You only exercise if beneficial (S_T > K)

**Put Option**: Right to **sell** the underlying asset at strike price K  
- Payoff function: `max(K - S_T, 0)`
- Mathematical intuition: You only exercise if beneficial (K > S_T)

Where:
- `S_T` = Asset price at expiration time T
- `K` = Strike price (fixed in the contract)
- The `max(·,0)` ensures you never lose more than the premium paid

## Why Do Options Exist? Economic Functions

### 1. **Risk Transfer** (Hedging)
- **Portfolio Protection**: Buy puts to limit downside risk
- **Income Generation**: Sell calls against stock holdings
- **Mathematical analogy**: Like buying insurance with a deductible

### 2. **Price Discovery** 
- **Volatility Information**: Option prices reveal market expectations of future volatility
- **Probability Estimates**: Option prices contain implicit probability distributions

### 3. **Leverage**
- **Capital Efficiency**: Control large positions with small capital
- **Mathematical leverage**: δ (delta) can be > 1, amplifying price movements

### 4. **Speculation**
- **Directional Bets**: Express views on price direction with limited downside
- **Volatility Bets**: Profit from changes in price variability

## Key Parameters and Their Mathematical Roles

| Parameter | Symbol | Mathematical Role | Intuitive Meaning |
|-----------|--------|------------------|-------------------|
| **Current Price** | S₀ | Initial condition for stochastic process | "Where we start" |
| **Strike Price** | K | Threshold in payoff function | "Decision boundary" |
| **Time to Expiry** | T | Time horizon for stochastic evolution | "How long uncertainty lasts" |
| **Volatility** | σ | Standard deviation of returns | "How much randomness" |
| **Risk-free Rate** | r | Discount rate and drift in risk-neutral measure | "Cost of money" |

## Types of Options by Exercise Rights

| Type | Exercise Rules | Mathematical Complexity |
|------|---------------|------------------------|
| **European** | Exercise only at expiry T | Easier: Fixed endpoint |
| **American** | Exercise any time ≤ T | Harder: Optimal stopping problem |
| **Bermudan** | Exercise on specific dates | Moderate: Discrete optimal stopping |

**Our project focuses on European options** because they have **closed-form solutions** (Black-Scholes formula) that we can validate our Monte Carlo simulations against.

## Moneyness: The Mathematical Classification

Options are classified by their **intrinsic value** relative to current market conditions:

- **In-the-Money (ITM)**: Immediate exercise would be profitable
  - Call: S₀ > K (current price above strike)
  - Put: S₀ < K (current price below strike)

- **At-the-Money (ATM)**: Exercise value is approximately zero
  - S₀ ≈ K (current price near strike)

- **Out-of-the-Money (OTM)**: No immediate exercise value
  - Call: S₀ < K (current price below strike)
  - Put: S₀ > K (current price above strike)

**Mathematical insight**: ITM options have **intrinsic value** = immediate exercise payoff, while OTM options only have **time value** = possibility of becoming profitable.

## Option Pricing: The Fundamental Challenge

The central question in quantitative finance: **What should you pay today for a random future payoff?**

### The No-Arbitrage Principle
If we can perfectly replicate an option's payoff using other tradeable securities, then the option must cost exactly the same as the replicating portfolio. Otherwise, arbitrage opportunities exist.

### Risk-Neutral Valuation
Under certain assumptions, we can price options by:
1. Computing expected payoff under a "risk-neutral" probability measure
2. Discounting at the risk-free rate

Mathematically:
```
Option Price = e^(-rT) × E[Payoff under risk-neutral measure]
```

This is exactly what our **Monte Carlo simulations** compute!

## The Greeks: Measuring Risk Sensitivities

Options prices change as market conditions change. The "Greeks" measure these sensitivities:

- **Delta (Δ)**: Sensitivity to underlying price changes
  - Δ = ∂V/∂S (partial derivative of option value w.r.t. stock price)
  - Range: 0 to 1 for calls, -1 to 0 for puts

- **Gamma (Γ)**: Rate of change of delta
  - Γ = ∂²V/∂S² (second derivative w.r.t. stock price)
  - Measures "convexity" of option value

- **Theta (Θ)**: Time decay
  - Θ = ∂V/∂T (sensitivity to time passage)
  - Usually negative (options lose value as expiry approaches)

- **Vega (ν)**: Volatility sensitivity
  - ν = ∂V/∂σ (sensitivity to volatility changes)
  - Always positive (higher volatility increases option value)

## Real-World Complications

The mathematical idealization breaks down in practice due to:

### 1. **Transaction Costs**
- Bid-ask spreads make trading expensive
- Perfect replication becomes impossible

### 2. **Liquidity Constraints**
- Can't always trade exactly when needed
- Large trades move market prices

### 3. **Volatility Smile**
- Market prices don't match Black-Scholes assumptions
- Implied volatility varies with strike and time

### 4. **Discrete Trading**
- Markets close, prices jump
- Continuous hedging is impossible

**Our project quantifies these real-world effects** by comparing theoretical models with actual market data and realistic trading constraints.

## Connection to Our Monte Carlo Analysis

1. **Performance Testing**: How fast can we compute millions of option prices?
2. **Model Validation**: Do our simulations match the theoretical Black-Scholes formula?
3. **Reality Check**: How do real market prices deviate from theory?
4. **Risk Management**: Can we actually hedge options profitably with transaction costs?

This mathematical foundation makes our computational analysis meaningful - we're not just running simulations, we're testing fundamental assumptions about how financial markets work.
