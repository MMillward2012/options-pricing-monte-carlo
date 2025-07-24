# Options Trading Basics — Quick Reference

## Key Terms

- **Option**: A financial contract giving the holder the right (but not obligation) to buy or sell an asset at a specified price before or at expiry.

- **Call Option**: Right to **buy** the underlying asset at the strike price.

- **Put Option**: Right to **sell** the underlying asset at the strike price.

- **Strike Price (K)**: The fixed price at which the asset can be bought (call) or sold (put).

- **Expiry Date (T)**: The date at which the option contract expires.

- **Premium**: The price you pay to buy an option contract upfront.

- **Exercise**: Using your right to buy (call) or sell (put) the asset at the strike price.

---

## Types of Options

| Type     | Exercise Rights                                      |
|----------|-----------------------------------------------------|
| European | Exercise only **at expiry**                         |
| American | Exercise **any time** before or at expiry           |
| Bermudan | Exercise on **specific dates** before or at expiry  |

---

## Payoff Formulas at Expiry

- **Call option payoff**:
$$
\max(S_T - K, 0)
$$
- **Put option payoff**:
$$
\max(K - S_T, 0)
$$
where $ S_T $ is the asset price at expiry.

---

## Profit and Loss

$$
\text{Profit} = \text{Payoff} - \text{Premium}
$$

- You pay the premium **upfront**.
- Payoff is realized at expiry (or when exercised).

---

## Additional Concepts

- **Underlying asset**: The stock, commodity, index, etc. the option is based on.

- **In the Money (ITM)**: When exercising the option would be profitable (e.g., $ S_T > K $ for calls).

- **Out of the Money (OTM)**: When exercising would result in no profit (e.g., $ S_T \leq K $ for calls).

- **At the Money (ATM)**: When the asset price is approximately equal to the strike price.

- **Protective Put**: Owning the underlying asset and buying a put to limit downside risk.

- **Trading options**: You can buy and sell options contracts without exercising, to lock in profits or hedge.
