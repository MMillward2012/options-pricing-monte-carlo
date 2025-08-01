# Stochastic Intuition Cheat Sheet

This document summarises key concepts in stochastic processes and option pricing, useful for understanding Monte Carlo simulations and interviews.

---

## 1. Brownian Motion (Wiener Process)

- Continuous-time stochastic process modeling random fluctuations.
- Starts at 0.
- Has independent, normally distributed increments.
- Variance of increments grows linearly with time.
- Represents the "random noise" in stock price movements.

---

## 2. Geometric Brownian Motion (GBM)

- Model for stock prices where the log of price follows Brownian motion with drift.
- Stochastic Differential Equation (SDE):

  $$
  dS_t = \mu S_t\, dt + \sigma S_t\, dW_t
  $$

  where:
  - $S_t$ = Stock price at time $t$
  - $\mu$ = Expected return (drift)
  - $\sigma$ = Volatility (random fluctuation scale)
  - $dW_t$ = Increment of Brownian motion

- Captures both deterministic growth and randomness.

---

## 3. Risk-Neutral Measure

- A probability measure where all assets grow at the risk-free rate $r$.
- Used to price derivatives without knowing investors' risk preferences.
- Under risk-neutral measure, expected asset return = risk-free rate.
- Option pricing formula under risk-neutral measure:

  $$
  \text{Option Price} = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]
  $$

  where:
  - $T$ = time to maturity
  - $\mathbb{Q}$ = risk-neutral probability measure

---

## 4. Monte Carlo Simulation for Option Pricing

- Simulate many possible future stock price paths under the risk-neutral measure.
- Calculate the payoff of the option at maturity for each path.
- Average the payoffs and discount to present value.
- Approximate the fair option price.

---

## Sample Interview Questions

**Q:** What is Brownian motion?  
**A:** A continuous stochastic process with independent, normally distributed increments, modeling random movements like stock prices.

**Q:** Why use the risk-neutral measure?  
**A:** It allows pricing options by discounting expected payoffs assuming all assets grow at the risk-free rate, avoiding dependence on risk preferences.

**Q:** How does Monte Carlo simulation price options?  
**A:** By simulating many risk-neutral stock price paths, calculating payoffs at maturity, averaging them, and discounting back to present value.

