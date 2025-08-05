# src/utils/liquidity_modeling.py
# Advanced Liquidity and Slippage Model for Options Trading

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LiquidityMetrics:
    """Container for liquidity metrics"""
    bid_ask_spread_pct: float
    daily_volume: float
    open_interest: float
    market_impact_coeff: float
    liquidity_score: float  # 0-100 scale
    execution_difficulty: str  # Easy, Medium, Hard, Impossible
    max_position_size: float

@dataclass
class SlippageModel:
    """Container for slippage calculations"""
    linear_component: float  # Linear market impact
    quadratic_component: float  # Quadratic market impact
    temporary_impact: float  # Temporary price impact
    permanent_impact: float  # Permanent price impact
    bid_ask_cost: float  # Half-spread cost

class AdvancedLiquidityModel:
    """
    Institutional-grade liquidity and slippage modeling for options
    Based on academic research and market microstructure theory
    """
    
    def __init__(self):
        self.liquidity_cache = {}
        self.volume_profiles = {}
        self.impact_models = {}
        
    def analyze_option_liquidity(self, 
                                ticker: str, 
                                strike: float, 
                                expiry: str,
                                option_type: str = 'call') -> LiquidityMetrics:
        """
        Comprehensive liquidity analysis for specific option contract
        """
        
        # Fetch option chain data
        option_data = self._fetch_option_data(ticker, strike, expiry, option_type)
        
        if option_data is None:
            return self._create_default_liquidity_metrics()
        
        # Calculate bid-ask spread
        bid_ask_spread_pct = self._calculate_bid_ask_spread(option_data)
        
        # Estimate daily volume and open interest
        volume_metrics = self._estimate_volume_metrics(option_data, ticker)
        
        # Calculate market impact coefficient
        impact_coeff = self._calculate_market_impact_coefficient(
            volume_metrics, bid_ask_spread_pct, ticker
        )
        
        # Calculate liquidity score (0-100)
        liquidity_score = self._calculate_liquidity_score(
            bid_ask_spread_pct, volume_metrics, impact_coeff
        )
        
        # Determine execution difficulty
        execution_difficulty = self._classify_execution_difficulty(liquidity_score)
        
        # Calculate maximum position size
        max_position_size = self._calculate_max_position_size(
            volume_metrics, liquidity_score, ticker
        )
        
        return LiquidityMetrics(
            bid_ask_spread_pct=bid_ask_spread_pct,
            daily_volume=volume_metrics['daily_volume'],
            open_interest=volume_metrics['open_interest'],
            market_impact_coeff=impact_coeff,
            liquidity_score=liquidity_score,
            execution_difficulty=execution_difficulty,
            max_position_size=max_position_size
        )
    
    def calculate_slippage(self, 
                          ticker: str,
                          strike: float,
                          expiry: str,
                          position_size: float,
                          urgency: str = 'normal',
                          option_type: str = 'call') -> SlippageModel:
        """
        Calculate comprehensive slippage for option trade
        """
        
        # Get liquidity metrics
        liquidity = self.analyze_option_liquidity(ticker, strike, expiry, option_type)
        
        # Calculate position size as percentage of daily volume
        volume_ratio = abs(position_size) / max(liquidity.daily_volume, 1)
        
        # Linear market impact (based on Almgren-Chriss model)
        linear_impact = self._calculate_linear_impact(
            volume_ratio, liquidity.market_impact_coeff, urgency
        )
        
        # Quadratic market impact (for large trades)
        quadratic_impact = self._calculate_quadratic_impact(
            volume_ratio, liquidity.liquidity_score
        )
        
        # Temporary vs permanent impact split
        temporary_impact, permanent_impact = self._split_market_impact(
            linear_impact + quadratic_impact, urgency, liquidity.liquidity_score
        )
        
        # Bid-ask spread cost
        bid_ask_cost = liquidity.bid_ask_spread_pct / 2  # Half-spread for crossing
        
        return SlippageModel(
            linear_component=linear_impact,
            quadratic_component=quadratic_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            bid_ask_cost=bid_ask_cost
        )
    
    def simulate_realistic_execution(self, 
                                   trades: List[Dict],
                                   ticker: str = 'SPY') -> Dict:
        """
        Simulate realistic trade execution with liquidity constraints
        """
        
        execution_results = []
        total_slippage = 0
        total_impact = 0
        executed_notional = 0
        failed_trades = 0
        
        for trade in trades:
            try:
                # Extract trade parameters
                strike = trade['strike']
                expiry = trade['expiry']
                position_size = trade['position_size']
                option_type = trade.get('option_type', 'call')
                urgency = trade.get('urgency', 'normal')
                
                # Get liquidity metrics
                liquidity = self.analyze_option_liquidity(ticker, strike, expiry, option_type)
                
                # Check if trade is executable
                if position_size > liquidity.max_position_size:
                    execution_results.append({
                        'trade': trade,
                        'status': 'REJECTED',
                        'reason': 'Position size exceeds liquidity limits',
                        'max_executable': liquidity.max_position_size
                    })
                    failed_trades += 1
                    continue
                
                # Calculate slippage
                slippage = self.calculate_slippage(ticker, strike, expiry, position_size, urgency, option_type)
                
                # Calculate total execution cost
                total_execution_cost = (
                    slippage.linear_component + 
                    slippage.quadratic_component + 
                    slippage.bid_ask_cost
                )
                
                # Simulate execution with some randomness
                actual_slippage = total_execution_cost * (1 + np.random.normal(0, 0.1))
                actual_slippage = max(actual_slippage, slippage.bid_ask_cost)  # Minimum cost is bid-ask
                
                execution_results.append({
                    'trade': trade,
                    'status': 'EXECUTED',
                    'slippage_pct': actual_slippage,
                    'slippage_cost': position_size * actual_slippage,
                    'liquidity_score': liquidity.liquidity_score,
                    'execution_difficulty': liquidity.execution_difficulty
                })
                
                total_slippage += actual_slippage
                total_impact += position_size * actual_slippage
                executed_notional += abs(position_size)
                
            except Exception as e:
                execution_results.append({
                    'trade': trade,
                    'status': 'ERROR',
                    'reason': str(e)
                })
                failed_trades += 1
        
        # Calculate summary statistics
        avg_slippage = total_slippage / len(trades) if trades else 0
        execution_rate = (len(trades) - failed_trades) / len(trades) if trades else 0
        
        return {
            'execution_results': execution_results,
            'summary': {
                'total_trades': len(trades),
                'executed_trades': len(trades) - failed_trades,
                'failed_trades': failed_trades,
                'execution_rate': execution_rate,
                'avg_slippage_pct': avg_slippage,
                'total_impact_cost': total_impact,
                'executed_notional': executed_notional
            }
        }
    
    def _fetch_option_data(self, ticker: str, strike: float, expiry: str, option_type: str) -> Optional[Dict]:
        """Fetch option data from market"""
        try:
            stock = yf.Ticker(ticker)
            options = stock.option_chain(expiry)
            
            if option_type.lower() == 'call':
                chain = options.calls
            else:
                chain = options.puts
            
            # Find closest strike
            closest_strike_idx = (chain['strike'] - strike).abs().idxmin()
            option_data = chain.loc[closest_strike_idx].to_dict()
            
            return option_data
            
        except Exception as e:
            print(f"Warning: Could not fetch option data for {ticker} {strike} {expiry}: {e}")
            return None
    
    def _calculate_bid_ask_spread(self, option_data: Dict) -> float:
        """Calculate bid-ask spread percentage"""
        if option_data is None:
            return 0.10  # Default 10% spread
        
        bid = option_data.get('bid', 0)
        ask = option_data.get('ask', 0)
        
        if bid <= 0 or ask <= 0:
            return 0.15  # Wide spread for illiquid options
        
        mid_price = (bid + ask) / 2
        spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0.20
        
        return min(spread_pct, 0.50)  # Cap at 50%
    
    def _estimate_volume_metrics(self, option_data: Dict, ticker: str) -> Dict:
        """Estimate volume and open interest metrics"""
        
        if option_data is None:
            return {'daily_volume': 100, 'open_interest': 500}
        
        volume = option_data.get('volume', 0)
        open_interest = option_data.get('openInterest', 0)
        
        # If no volume data, estimate based on underlying
        if volume == 0:
            # Estimate based on underlying volume and moneyness
            try:
                stock = yf.Ticker(ticker)
                stock_volume = stock.info.get('averageVolume', 1000000)
                estimated_volume = max(stock_volume * 0.01, 50)  # 1% of stock volume
            except:
                estimated_volume = 100
            volume = estimated_volume
        
        if open_interest == 0:
            open_interest = max(volume * 5, 100)  # Estimate OI as 5x volume
        
        return {
            'daily_volume': float(volume),
            'open_interest': float(open_interest)
        }
    
    def _calculate_market_impact_coefficient(self, volume_metrics: Dict, spread: float, ticker: str) -> float:
        """Calculate market impact coefficient based on Almgren-Chriss model"""
        
        volume = volume_metrics['daily_volume']
        
        # Base impact coefficient (higher for less liquid)
        base_impact = 0.1 / np.sqrt(volume)  # Square root law
        
        # Adjust for spread (wider spread = more impact)
        spread_adjustment = 1 + spread * 2
        
        # Adjust for asset class (options generally less liquid)
        options_multiplier = 2.0
        
        impact_coeff = base_impact * spread_adjustment * options_multiplier
        
        return min(impact_coeff, 1.0)  # Cap at 100%
    
    def _calculate_liquidity_score(self, spread: float, volume_metrics: Dict, impact_coeff: float) -> float:
        """Calculate liquidity score (0-100)"""
        
        # Volume component (0-40 points)
        volume_score = min(40, np.log10(volume_metrics['daily_volume']) * 8)
        
        # Spread component (0-30 points)
        spread_score = max(0, 30 - spread * 300)
        
        # Open interest component (0-20 points)
        oi_score = min(20, np.log10(volume_metrics['open_interest']) * 4)
        
        # Impact component (0-10 points, inverse relationship)
        impact_score = max(0, 10 - impact_coeff * 10)
        
        total_score = volume_score + spread_score + oi_score + impact_score
        return min(total_score, 100)
    
    def _classify_execution_difficulty(self, liquidity_score: float) -> str:
        """Classify execution difficulty based on liquidity score"""
        if liquidity_score >= 80:
            return "Easy"
        elif liquidity_score >= 60:
            return "Medium"
        elif liquidity_score >= 40:
            return "Hard"
        else:
            return "Impossible"
    
    def _calculate_max_position_size(self, volume_metrics: Dict, liquidity_score: float, ticker: str) -> float:
        """Calculate maximum recommendable position size"""
        
        daily_volume = volume_metrics['daily_volume']
        
        # Base rule: Don't exceed certain percentage of daily volume
        if liquidity_score >= 80:
            max_volume_pct = 0.10  # 10% of daily volume
        elif liquidity_score >= 60:
            max_volume_pct = 0.05  # 5% of daily volume
        elif liquidity_score >= 40:
            max_volume_pct = 0.02  # 2% of daily volume
        else:
            max_volume_pct = 0.01  # 1% of daily volume
        
        max_size = daily_volume * max_volume_pct
        
        # Add absolute caps based on underlying
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', 1e9)
            
            if market_cap > 100e9:  # Large cap
                absolute_cap = 1000000  # $1M
            elif market_cap > 10e9:   # Mid cap
                absolute_cap = 500000   # $500k
            else:                     # Small cap
                absolute_cap = 100000   # $100k
                
        except:
            absolute_cap = 500000  # Default $500k cap
        
        return min(max_size, absolute_cap)
    
    def _calculate_linear_impact(self, volume_ratio: float, impact_coeff: float, urgency: str) -> float:
        """Calculate linear market impact component"""
        
        # Base linear impact
        linear_impact = impact_coeff * volume_ratio
        
        # Urgency multiplier
        urgency_multipliers = {
            'low': 0.5,      # Patient execution
            'normal': 1.0,   # Standard execution
            'high': 2.0,     # Urgent execution
            'immediate': 5.0  # Market orders
        }
        
        multiplier = urgency_multipliers.get(urgency, 1.0)
        
        return linear_impact * multiplier
    
    def _calculate_quadratic_impact(self, volume_ratio: float, liquidity_score: float) -> float:
        """Calculate quadratic market impact for large trades"""
        
        # Quadratic impact kicks in for trades > 5% of daily volume
        if volume_ratio <= 0.05:
            return 0
        
        # Higher quadratic impact for less liquid options
        liquidity_factor = max(0.1, (100 - liquidity_score) / 100)
        
        quadratic_component = liquidity_factor * (volume_ratio ** 2) * 0.5
        
        return quadratic_component
    
    def _split_market_impact(self, total_impact: float, urgency: str, liquidity_score: float) -> Tuple[float, float]:
        """Split market impact into temporary and permanent components"""
        
        # Higher urgency = more temporary impact
        # Lower liquidity = more permanent impact
        
        if urgency == 'immediate':
            temp_ratio = 0.8  # 80% temporary
        elif urgency == 'high':
            temp_ratio = 0.7  # 70% temporary
        elif urgency == 'normal':
            temp_ratio = 0.6  # 60% temporary
        else:  # low urgency
            temp_ratio = 0.4  # 40% temporary
        
        # Adjust for liquidity (less liquid = more permanent impact)
        liquidity_adjustment = liquidity_score / 100
        temp_ratio *= liquidity_adjustment
        
        temporary_impact = total_impact * temp_ratio
        permanent_impact = total_impact * (1 - temp_ratio)
        
        return temporary_impact, permanent_impact
    
    def _create_default_liquidity_metrics(self) -> LiquidityMetrics:
        """Create default liquidity metrics when data is unavailable"""
        return LiquidityMetrics(
            bid_ask_spread_pct=0.15,
            daily_volume=100,
            open_interest=500,
            market_impact_coeff=0.05,
            liquidity_score=30,
            execution_difficulty="Hard",
            max_position_size=50000
        )

class VolumeProfileAnalyzer:
    """
    Analyze intraday volume profiles for optimal execution timing
    """
    
    def __init__(self):
        self.volume_patterns = {}
    
    def analyze_optimal_execution_times(self, ticker: str) -> Dict:
        """
        Analyze historical volume patterns to find optimal execution windows
        """
        
        # In a real implementation, this would analyze intraday data
        # For now, we'll provide theoretical insights
        
        optimal_times = {
            'market_open': {
                'time_window': '09:30-10:00',
                'volume_percentile': 95,
                'liquidity_score': 85,
                'recommended': False,
                'reason': 'High volatility and wide spreads'
            },
            'morning_session': {
                'time_window': '10:00-11:30',
                'volume_percentile': 70,
                'liquidity_score': 90,
                'recommended': True,
                'reason': 'Good liquidity with stabilized spreads'
            },
            'lunch_period': {
                'time_window': '11:30-13:30',
                'volume_percentile': 40,
                'liquidity_score': 60,
                'recommended': False,
                'reason': 'Lower volume and wider spreads'
            },
            'afternoon_session': {
                'time_window': '13:30-15:00',
                'volume_percentile': 65,
                'liquidity_score': 80,
                'recommended': True,
                'reason': 'Decent liquidity and reasonable spreads'
            },
            'market_close': {
                'time_window': '15:00-16:00',
                'volume_percentile': 90,
                'liquidity_score': 75,
                'recommended': False,
                'reason': 'High volume but increased volatility'
            }
        }
        
        return optimal_times

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize the liquidity model
    liquidity_model = AdvancedLiquidityModel()
    
    # Example: Analyze SPY option liquidity
    print("🔍 LIQUIDITY ANALYSIS EXAMPLE")
    print("="*50)
    
    try:
        # Analyze a specific option
        liquidity_metrics = liquidity_model.analyze_option_liquidity(
            ticker='SPY',
            strike=450,
            expiry='2024-12-20',
            option_type='call'
        )
        
        print(f"\n📊 SPY 450 Call (Dec 2024) Liquidity Metrics:")
        print(f"   Bid-Ask Spread: {liquidity_metrics.bid_ask_spread_pct:.2%}")
        print(f"   Daily Volume: {liquidity_metrics.daily_volume:,.0f}")
        print(f"   Open Interest: {liquidity_metrics.open_interest:,.0f}")
        print(f"   Liquidity Score: {liquidity_metrics.liquidity_score:.0f}/100")
        print(f"   Execution Difficulty: {liquidity_metrics.execution_difficulty}")
        print(f"   Max Position Size: ${liquidity_metrics.max_position_size:,.0f}")
        
        # Calculate slippage for different position sizes
        print(f"\n💧 SLIPPAGE ANALYSIS:")
        position_sizes = [10000, 50000, 100000, 250000]
        
        for pos_size in position_sizes:
            slippage = liquidity_model.calculate_slippage(
                ticker='SPY',
                strike=450,
                expiry='2024-12-20',
                position_size=pos_size
            )
            
            total_slippage = (slippage.linear_component + 
                            slippage.quadratic_component + 
                            slippage.bid_ask_cost)
            
            print(f"   ${pos_size:,} position: {total_slippage:.2%} total slippage")
            print(f"     Linear: {slippage.linear_component:.2%}")
            print(f"     Quadratic: {slippage.quadratic_component:.2%}")
            print(f"     Bid-Ask: {slippage.bid_ask_cost:.2%}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("This is expected if market data is not available")
    
    # Example: Simulate realistic execution
    print(f"\n🎯 EXECUTION SIMULATION EXAMPLE")
    print("="*50)
    
    sample_trades = [
        {'strike': 450, 'expiry': '2024-12-20', 'position_size': 25000, 'option_type': 'call'},
        {'strike': 460, 'expiry': '2024-12-20', 'position_size': 50000, 'option_type': 'call'},
        {'strike': 440, 'expiry': '2024-12-20', 'position_size': 100000, 'option_type': 'put'},
    ]
    
    execution_results = liquidity_model.simulate_realistic_execution(sample_trades, 'SPY')
    
    print(f"\n📈 EXECUTION RESULTS:")
    summary = execution_results['summary']
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Executed: {summary['executed_trades']}")
    print(f"   Failed: {summary['failed_trades']}")
    print(f"   Execution Rate: {summary['execution_rate']:.1%}")
    print(f"   Avg Slippage: {summary['avg_slippage_pct']:.2%}")
    print(f"   Total Impact Cost: ${summary['total_impact_cost']:,.0f}")
    
    # Volume profile analysis
    volume_analyzer = VolumeProfileAnalyzer()
    optimal_times = volume_analyzer.analyze_optimal_execution_times('SPY')
    
    print(f"\n⏰ OPTIMAL EXECUTION TIMING:")
    for period, metrics in optimal_times.items():
        status = "✅ RECOMMENDED" if metrics['recommended'] else "❌ AVOID"
        print(f"   {period.replace('_', ' ').title()} ({metrics['time_window']}): {status}")
        print(f"     Reason: {metrics['reason']}")
    
    print(f"\n" + "="*50)
    print("🎉 LIQUIDITY MODEL DEMONSTRATION COMPLETE")
    print("="*50)