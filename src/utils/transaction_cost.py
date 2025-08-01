# src/utils/transaction_cost.py

"""
Transaction Cost Model - The Reality Check
This is what separates theoretical profits from actual profits.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TransactionCosts:
    """Comprehensive transaction cost structure"""
    bid_ask_spread_bps: float = 20.0  # 20 bps typical for liquid options
    commission_per_contract: float = 0.50  # Per option contract
    market_impact_bps: float = 5.0  # Additional cost for large trades
    financing_rate_bps: float = 200.0  # Cost of capital (2% annually)
    exchange_fees_bps: float = 0.5  # Exchange/regulatory fees

@dataclass
class MarketConstraints:
    """Real-world trading constraints"""
    max_position_size: float = 1000000  # $1M max position
    max_daily_volume_pct: float = 0.10  # Can't trade >10% of daily volume
    min_liquidity_threshold: int = 100  # Minimum daily volume
    max_spread_threshold_bps: float = 50.0  # Don't trade if spread >50bps

class TransactionCostCalculator:
    """
    Calculate realistic transaction costs for options strategies.
    This is the difference between academic backtests and real trading.
    """
    
    def __init__(self, costs: TransactionCosts = None, constraints: MarketConstraints = None):
        self.costs = costs or TransactionCosts()
        self.constraints = constraints or MarketConstraints()
        
    def calculate_option_transaction_cost(self, 
                                        option_price: float,
                                        position_size: int,
                                        bid_ask_spread: float,
                                        daily_volume: int) -> Dict[str, float]:
        """
        Calculate total transaction cost for an option trade.
        Returns breakdown of all cost components.
        """
        
        notional = option_price * position_size * 100  # Options are x100 multiplier
        
        # 1. Bid-ask spread cost (most important)
        spread_cost = bid_ask_spread * position_size * 100
        
        # 2. Market impact (increases with position size relative to volume)
        volume_ratio = position_size / max(daily_volume, 1)
        impact_multiplier = min(volume_ratio * 2, 1.0)  # Caps at 100% impact
        market_impact = notional * (self.costs.market_impact_bps / 10000) * impact_multiplier
        
        # 3. Fixed costs
        commission = self.costs.commission_per_contract * position_size
        exchange_fees = notional * (self.costs.exchange_fees_bps / 10000)
        
        # 4. Financing cost (for positions held overnight)
        financing_daily = notional * (self.costs.financing_rate_bps / 10000) / 252
        
        total_cost = spread_cost + market_impact + commission + exchange_fees
        
        return {
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'commission': commission,
            'exchange_fees': exchange_fees,
            'financing_daily': financing_daily,
            'total_entry_cost': total_cost,
            'cost_bps': (total_cost / notional) * 10000 if notional > 0 else 0,
            'notional': notional
        }
    
    def calculate_arbitrage_profit_after_costs(self, 
                                             arbitrage_opportunity: Dict,
                                             position_size: int = 10) -> Dict[str, float]:
        """
        The money shot: How much profit is left after realistic costs?
        This gives you the metrics Jane Street wants to see.
        """
        
        # Gross arbitrage profit
        gross_profit = arbitrage_opportunity['violation_amt'] * position_size * 100
        
        # Calculate costs for both legs of the arbitrage
        call_data = arbitrage_opportunity.get('call_data', {})
        put_data = arbitrage_opportunity.get('put_data', {})
        
        # Call option transaction costs
        call_costs = self.calculate_option_transaction_cost(
            option_price=call_data.get('lastPrice', 5.0),
            position_size=position_size,
            bid_ask_spread=call_data.get('spread', 0.10),
            daily_volume=call_data.get('volume', 100)
        )
        
        # Put option transaction costs  
        put_costs = self.calculate_option_transaction_cost(
            option_price=put_data.get('lastPrice', 3.0),
            position_size=position_size,
            bid_ask_spread=put_data.get('spread', 0.10),
            daily_volume=put_data.get('volume', 100)
        )
        
        total_transaction_costs = call_costs['total_entry_cost'] + put_costs['total_entry_cost']
        net_profit = gross_profit - total_transaction_costs
        
        return {
            'gross_profit': gross_profit,
            'call_transaction_costs': call_costs['total_entry_cost'],
            'put_transaction_costs': put_costs['total_entry_cost'],
            'total_transaction_costs': total_transaction_costs,
            'net_profit': net_profit,
            'profit_margin_pct': (net_profit / gross_profit) * 100 if gross_profit > 0 else -100,
            'is_profitable': net_profit > 0,
            'breakeven_position_size': self._calculate_breakeven_size(arbitrage_opportunity),
            'cost_breakdown': {
                'call_costs': call_costs,
                'put_costs': put_costs
            }
        }
    
    def calculate_hedging_costs(self, 
                              hedge_trades: List[Dict],
                              holding_period_days: int = 1) -> Dict[str, float]:
        """
        Calculate the cost of dynamic hedging strategy.
        Critical for evaluating if hedging actually adds value.
        """
        
        total_costs = 0
        trade_count = len(hedge_trades)
        total_notional = 0
        
        cost_breakdown = {
            'rebalancing_costs': 0,
            'financing_costs': 0,
            'slippage_costs': 0
        }
        
        for trade in hedge_trades:
            trade_size = abs(trade.get('delta_change', 0))
            stock_price = trade.get('stock_price', 100)
            
            # Rebalancing costs (bid-ask spread on stock)
            stock_spread_bps = 2.0  # Typical for liquid stocks
            rebalancing_cost = trade_size * stock_price * (stock_spread_bps / 10000)
            
            # Financing costs
            notional = trade_size * stock_price
            financing_cost = notional * (self.costs.financing_rate_bps / 10000) * (holding_period_days / 252)
            
            cost_breakdown['rebalancing_costs'] += rebalancing_cost
            cost_breakdown['financing_costs'] += financing_cost
            total_notional += notional
            
        total_costs = sum(cost_breakdown.values())
        
        return {
            'total_hedging_costs': total_costs,
            'cost_per_trade': total_costs / max(trade_count, 1),
            'cost_bps_of_notional': (total_costs / max(total_notional, 1)) * 10000,
            'trade_count': trade_count,
            'cost_breakdown': cost_breakdown,
            'average_trade_size': total_notional / max(trade_count, 1)
        }
    
    def _calculate_breakeven_size(self, arbitrage_opportunity: Dict) -> int:
        """Calculate minimum position size needed to overcome transaction costs"""
        violation_per_contract = arbitrage_opportunity['violation_amt'] * 100
        
        # Estimate minimum transaction costs
        min_costs = (self.costs.commission_per_contract * 2 +  # Both call and put
                    self.costs.bid_ask_spread_bps / 10000 * 500)  # Assume $5 avg premium
        
        if violation_per_contract <= 0:
            return float('inf')
            
        return max(1, int(np.ceil(min_costs / violation_per_contract)))
    
    def assess_strategy_scalability(self, 
                                  strategy_returns: np.ndarray,
                                  avg_position_size: float,
                                  daily_volume_constraint: float) -> Dict[str, float]:
        """
        Determine how much capital can be deployed before returns degrade.
        Critical for assessing real-world viability.
        """
        
        # Calculate returns at different scale levels
        scale_factors = np.array([1, 2, 5, 10, 20, 50])
        scaled_returns = []
        
        for factor in scale_factors:
            # Market impact increases quadratically with size
            impact_penalty = (factor - 1) * 0.002  # 20bps penalty per 2x scale
            
            # Volume constraint penalty
            volume_penalty = max(0, (factor * avg_position_size / daily_volume_constraint - 1) * 0.01)
            
            adjusted_returns = strategy_returns - impact_penalty - volume_penalty
            scaled_returns.append(np.mean(adjusted_returns))
        
        # Find optimal scale (where marginal returns = marginal costs)
        optimal_scale_idx = np.argmax(np.array(scaled_returns) * scale_factors)
        optimal_scale = scale_factors[optimal_scale_idx]
        
        return {
            'optimal_scale_factor': optimal_scale,
            'max_profitable_aum': optimal_scale * avg_position_size,
            'returns_at_scale': dict(zip(scale_factors, scaled_returns)),
            'capacity_constraint': 'volume' if volume_penalty > impact_penalty else 'market_impact'
        }

def generate_cost_impact_report(strategy_results: Dict, 
                               position_sizes: List[int] = [1, 5, 10, 25, 50]) -> pd.DataFrame:
    """
    Generate a comprehensive report showing how transaction costs impact strategy profitability.
    This creates the compelling metrics for your CV.
    """
    
    calculator = TransactionCostCalculator()
    report_data = []
    
    for size in position_sizes:
        # Calculate costs for this position size
        cost_analysis = calculator.calculate_arbitrage_profit_after_costs(
            strategy_results, position_size=size
        )
        
        report_data.append({
            'position_size': size,
            'gross_profit': cost_analysis['gross_profit'],
            'net_profit': cost_analysis['net_profit'],
            'transaction_costs': cost_analysis['total_transaction_costs'],
            'profit_margin_pct': cost_analysis['profit_margin_pct'],
            'is_profitable': cost_analysis['is_profitable'],
            'cost_ratio': cost_analysis['total_transaction_costs'] / cost_analysis['gross_profit'] if cost_analysis['gross_profit'] > 0 else 1.0
        })
    
    return pd.DataFrame(report_data)


# Example usage and key metrics calculation
def calculate_strategy_alpha_after_costs(gross_returns: np.ndarray, 
                                       trading_frequency: int = 252) -> Dict[str, float]:
    """
    Calculate the holy grail: actual alpha after all costs.
    This is what goes on your CV.
    """
    
    calculator = TransactionCostCalculator()
    
    # Estimate trading costs based on frequency
    annual_trading_cost_bps = (calculator.costs.bid_ask_spread_bps + 
                              calculator.costs.market_impact_bps) * trading_frequency / 252
    
    # Adjust returns for costs
    cost_adjusted_returns = gross_returns - (annual_trading_cost_bps / 10000)
    
    # Calculate key metrics
    alpha_gross = np.mean(gross_returns) * 252
    alpha_net = np.mean(cost_adjusted_returns) * 252
    alpha_erosion = alpha_gross - alpha_net
    
    sharpe_gross = np.mean(gross_returns) / np.std(gross_returns) * np.sqrt(252)
    sharpe_net = np.mean(cost_adjusted_returns) / np.std(cost_adjusted_returns) * np.sqrt(252) if np.std(cost_adjusted_returns) > 0 else 0
    
    return {
        'alpha_gross_pct': alpha_gross * 100,
        'alpha_net_pct': alpha_net * 100,
        'alpha_erosion_pct': alpha_erosion * 100,
        'sharpe_gross': sharpe_gross,
        'sharpe_net': sharpe_net,
        'cost_impact_on_sharpe': sharpe_gross - sharpe_net,
        'breakeven_alpha_pct': (annual_trading_cost_bps / 100),
        'is_viable': alpha_net > 0
    }