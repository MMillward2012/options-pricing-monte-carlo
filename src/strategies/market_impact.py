import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketImpactAnalyzer:
    """
    Analyze real market constraints and impact costs.
    """
    
    def __init__(self, tickers: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']):
        self.tickers = tickers
        self.market_data = {}
        self.liquidity_metrics = {}
        
    # UPDATED with more robust fetching logic
    def fetch_comprehensive_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get detailed market data including volume, spreads, and intraday patterns.
        """
        print("Fetching comprehensive market data...")
        for ticker in self.tickers:
            try:
                print(f"--> Processing ticker: {ticker}")
                ticker_obj = yf.Ticker(ticker)
                hist_data = ticker_obj.history(period='60d', interval='1d', timeout=10)
                
                option_data = []
                try:
                    expiry_dates = ticker_obj.options[:3] 
                    
                    for expiry in expiry_dates:
                        try:
                            chain = ticker_obj.option_chain(expiry)
                            if chain.calls.empty and chain.puts.empty:
                                print(f"    - No option data for {ticker} on expiry {expiry}.")
                                continue
                            
                            calls, puts = chain.calls, chain.puts
                            calls['option_type'], calls['expiry'] = 'call', expiry
                            puts['option_type'], puts['expiry'] = 'put', expiry
                            
                            option_data.append(calls)
                            option_data.append(puts)
                        except Exception as e:
                            print(f"    - Could not fetch option chain for {ticker} expiry {expiry}: {e}")
                            continue
                except Exception as e:
                    print(f"    - No options available for {ticker}: {e}")
                
                if option_data:
                    options_df = pd.concat(option_data, ignore_index=True)
                else:
                    options_df = pd.DataFrame()
                
                self.market_data[ticker] = {
                    'stock_data': hist_data,
                    'options_data': options_df
                }
                
            except Exception as e:
                print(f"--> FAILED to fetch all data for {ticker}: {e}")
                continue
        
        return self.market_data
     
    def analyze_liquidity_constraints(self) -> Dict[str, Dict]:
        """
        Calculate realistic position size limits based on market liquidity.
        """
        print("\nAnalyzing liquidity constraints...")
        
        # First fetch data if not already done
        if not self.market_data:
            self.fetch_comprehensive_market_data()
            
        constraints = {}
        
        for ticker, data in self.market_data.items():
            print(f"--> Analyzing: {ticker}")
            stock_data = data['stock_data']
            options_data = data['options_data']
            
            if stock_data.empty:
                print(f"    - FAIL: Stock data is empty for {ticker}. Skipping.")
                continue
            
            # Stock liquidity analysis
            avg_daily_volume = stock_data['Volume'].mean()
            current_price = stock_data['Close'].iloc[-1]
            avg_daily_dollar_volume = (stock_data['Volume'] * stock_data['Close']).mean()
            
            # Maximum position size (1% of daily volume rule)
            max_stock_dollar_position = avg_daily_dollar_volume * 0.01
            
            # Options liquidity analysis
            if not options_data.empty:
                liquid_options = options_data[
                    (options_data['volume'] > 10) & 
                    (options_data['bid'] > 0) & 
                    (options_data['ask'] > 0)
                ].copy()
                
                if not liquid_options.empty:
                    # Calculate average spread
                    avg_spread_pct = ((liquid_options['ask'] - liquid_options['bid']) / 
                                    (liquid_options['lastPrice'] + 1e-6)).mean() * 100
                    liquid_strikes_count = len(liquid_options)
                    total_option_volume = liquid_options['volume'].sum()
                else:
                    avg_spread_pct = 5.0  # Default high spread if no liquid options
                    liquid_strikes_count = 0
                    total_option_volume = 0
            else:
                avg_spread_pct = 5.0  # Default high spread if no options data
                liquid_strikes_count = 0
                total_option_volume = 0
            
            strategy_capacity = avg_daily_dollar_volume * 0.01

            constraints[ticker] = {
                'avg_daily_dollar_volume': avg_daily_dollar_volume,
                'max_stock_dollar_position': max_stock_dollar_position,
                'estimated_strategy_capacity': strategy_capacity,
                'liquid_options_contracts': total_option_volume,
                'liquid_strikes_count': liquid_strikes_count,
                'avg_option_spread_pct': avg_spread_pct,  # Fixed key name
                'avg_bid_ask_spread_pct': avg_spread_pct,  # Keep both for compatibility
                'current_price': current_price,
                'avg_daily_volume': avg_daily_volume
            }
            print(f"    - SUCCESS: Metrics calculated for {ticker}.")
        
        self.liquidity_metrics = constraints
        return constraints
    
    def calculate_market_impact_curve(self, ticker: str, trade_sizes: List[float]) -> pd.DataFrame:
        """
        Model how market impact increases with trade size.
        Critical for position sizing and strategy scalability.
        """
        
        if ticker not in self.liquidity_metrics:
            return pd.DataFrame()
        
        metrics = self.liquidity_metrics[ticker]
        max_position = metrics['max_stock_dollar_position']
        
        impact_data = []
        
        for trade_size in trade_sizes:
            # Market impact model: sqrt(trade_size / daily_volume)
            size_ratio = trade_size / max_position if max_position > 0 else 0
            
            # Base impact (spreads)
            base_impact_bps = metrics['avg_option_spread_pct'] * 100  # Convert to bps
            
            # Volume impact (increases with square root of size)
            volume_impact_bps = 50 * np.sqrt(size_ratio) if size_ratio > 0 else 0
            
            # Timing impact (larger trades take longer, create more impact)
            timing_impact_bps = 20 * size_ratio if size_ratio > 0.5 else 0
            
            total_impact_bps = base_impact_bps + volume_impact_bps + timing_impact_bps
            
            # Calculate cost in dollars
            cost_dollars = trade_size * (total_impact_bps / 10000)
            
            impact_data.append({
                'trade_size_dollars': trade_size,
                'size_ratio_vs_daily_volume': size_ratio,
                'base_impact_bps': base_impact_bps,
                'volume_impact_bps': volume_impact_bps,
                'timing_impact_bps': timing_impact_bps,
                'total_impact_bps': total_impact_bps,
                'cost_dollars': cost_dollars,
                'cost_percentage': (cost_dollars / trade_size) * 100 if trade_size > 0 else 0
            })
        
        return pd.DataFrame(impact_data)
    
    def analyze_cross_asset_opportunities(self) -> Dict[str, float]:
        """
        Find arbitrage opportunities across related assets.
        This is the kind of alpha generation Jane Street looks for.
        """
        
        opportunities = {}
        
        # ETF-underlying arbitrage (SPY vs components)
        if 'SPY' in self.market_data and 'AAPL' in self.market_data:
            spy_data = self.market_data['SPY']['stock_data']
            aapl_data = self.market_data['AAPL']['stock_data']
            
            if not spy_data.empty and not aapl_data.empty:
                # Calculate correlation breakdown
                spy_returns = spy_data['Close'].pct_change().dropna()
                aapl_returns = aapl_data['Close'].pct_change().dropna()
                
                # Align dates
                common_dates = spy_returns.index.intersection(aapl_returns.index)
                spy_aligned = spy_returns.loc[common_dates]
                aapl_aligned = aapl_returns.loc[common_dates]
                
                if len(common_dates) > 20:
                    # Calculate rolling correlation
                    correlation = spy_aligned.rolling(20).corr(aapl_aligned).dropna()
                    
                    if len(correlation) > 0:
                        # Find correlation breakdown periods
                        mean_corr = correlation.mean()
                        corr_std = correlation.std()
                        breakdown_threshold = mean_corr - 2 * corr_std
                        
                        breakdown_days = (correlation < breakdown_threshold).sum()
                        breakdown_pct = breakdown_days / len(correlation) * 100
                        
                        opportunities['spy_aapl_correlation_breakdown'] = {
                            'mean_correlation': mean_corr,
                            'breakdown_days_pct': breakdown_pct,
                            'avg_breakdown_alpha_bps': abs(correlation - mean_corr).mean() * 10000,
                            'max_breakdown_alpha_bps': abs(correlation - mean_corr).max() * 10000
                        }
        
        return opportunities
    
    def calculate_optimal_position_sizing(self, 
                                        expected_returns: np.ndarray,
                                        volatility: float,
                                        max_position_size: float,
                                        risk_free_rate: float = 0.05) -> Dict[str, float]:
        """
        Calculate optimal position sizes using Kelly Criterion with realistic constraints.
        This shows sophisticated risk management thinking.
        """
        
        if len(expected_returns) == 0 or volatility <= 0:
            return {'optimal_fraction': 0, 'optimal_position_size': 0}
        
        # Kelly Criterion: f = (μ - r) / σ²
        excess_return = np.mean(expected_returns) - risk_free_rate / 252
        kelly_fraction = excess_return / (volatility ** 2) if volatility > 0 else 0
        
        # Apply constraints
        # 1. Never risk more than 25% of capital on single bet
        kelly_fraction = min(kelly_fraction, 0.25)
        
        # 2. Never go negative (no leverage beyond available capital)
        kelly_fraction = max(kelly_fraction, 0)
        
        # 3. Apply liquidity constraints
        optimal_position_dollars = kelly_fraction * max_position_size
        
        # 4. Calculate risk metrics at optimal size
        position_vol = optimal_position_dollars * volatility
        daily_var_95 = position_vol * 1.645  # 95% VaR
        
        return {
            'kelly_fraction': kelly_fraction,
            'optimal_position_size': optimal_position_dollars,
            'daily_var_95': daily_var_95,
            'expected_daily_pnl': optimal_position_dollars * excess_return,
            'risk_reward_ratio': excess_return / volatility if volatility > 0 else 0,
            'max_drawdown_estimate': position_vol * 2.33  # 99% worst case
        }
    
    def generate_capacity_analysis(self) -> Dict[str, Dict]:
        """
        Determine strategy capacity - how much money can be deployed profitably.
        This is the metric that determines if a strategy is worth implementing.
        """
        
        capacity_analysis = {}
        
        for ticker, metrics in self.liquidity_metrics.items():
            
            # Base capacity from liquidity constraints
            base_capacity = metrics['max_stock_dollar_position']
            
            # Reduce capacity based on spread costs
            spread_penalty = metrics['avg_option_spread_pct'] / 100
            spread_adjusted_capacity = base_capacity * (1 - spread_penalty)
            
            # Further reduce for market impact
            impact_adjusted_capacity = spread_adjusted_capacity * 0.7  # 30% haircut for impact
            
            # Calculate revenue potential
            # Assume 10bps daily alpha opportunity
            daily_alpha_bps = 10
            annual_revenue_potential = impact_adjusted_capacity * (daily_alpha_bps / 10000) * 252
            
            # Calculate required infrastructure costs
            # (This is very rough but gives order of magnitude)
            infrastructure_cost = 500000  # $500K annually for systems, data, etc.
            
            capacity_analysis[ticker] = {
                'base_liquidity_capacity': base_capacity,
                'spread_adjusted_capacity': spread_adjusted_capacity,
                'final_deployable_capacity': impact_adjusted_capacity,
                'annual_revenue_potential': annual_revenue_potential,
                'infrastructure_costs': infrastructure_cost,
                'net_revenue_potential': annual_revenue_potential - infrastructure_cost,
                'is_economically_viable': annual_revenue_potential > infrastructure_cost * 2,  # 2x cost coverage
                'capacity_utilization_pct': (impact_adjusted_capacity / base_capacity) * 100 if base_capacity > 0 else 0,
                'break_even_alpha_bps': (infrastructure_cost / impact_adjusted_capacity) * 10000 / 252 if impact_adjusted_capacity > 0 else float('inf')
            }
        
        return capacity_analysis
    
    def _calculate_liquidity_score(self, dollar_volume: float, spread_pct: float, option_count: int) -> float:
        """Calculate a composite liquidity score (0-100)"""
        
        # Normalize components
        volume_score = min(dollar_volume / 10000000, 1.0) * 40  # $10M daily = max volume score
        spread_score = max(0, (5 - spread_pct) / 5) * 40  # <1% spread = max spread score  
        depth_score = min(option_count / 50, 1.0) * 20  # 50+ liquid strikes = max depth score
        
        return volume_score + spread_score + depth_score
    
    def generate_executive_summary(self) -> Dict[str, any]:  # Changed return type annotation
        """
        Generate the key metrics for your CV/interviews.
        This is what hiring managers want to see.
        """
        
        if not self.liquidity_metrics:
            return {
                'error': 'No liquidity metrics available. Run analyze_liquidity_constraints() first.',
                'total_addressable_market': '$0',
                'average_bid_ask_spread_pct': 'N/A',
                'liquid_opportunities_count': 0,
                'economically_viable_strategies': '0/0',
                'annual_revenue_potential': '$0',
                'average_capacity_utilization_pct': '0.0%',
                'key_constraint': 'no_data',
                'scalability_rating': 'None'
            }
        
        # Aggregate metrics across all tickers
        total_deployable_capital = sum(
            metrics['max_stock_dollar_position'] for metrics in self.liquidity_metrics.values()
        )
        
        avg_spread = np.mean([
            metrics['avg_option_spread_pct'] for metrics in self.liquidity_metrics.values()
        ])
        
        total_liquid_strikes = sum(
            metrics['liquid_strikes_count'] for metrics in self.liquidity_metrics.values()
        )
        
        # Generate capacity analysis
        capacity_data = self.generate_capacity_analysis()
        viable_strategies = sum(1 for analysis in capacity_data.values() if analysis['is_economically_viable'])
        
        total_revenue_potential = sum(
            analysis['annual_revenue_potential'] for analysis in capacity_data.values()
        )
        
        return {
            'total_addressable_market': f"${total_deployable_capital:,.0f}",
            'average_bid_ask_spread_pct': f"{avg_spread:.2f}%", 
            'liquid_opportunities_count': total_liquid_strikes,
            'economically_viable_strategies': f"{viable_strategies}/{len(capacity_data)}",
            'annual_revenue_potential': f"${total_revenue_potential:,.0f}",
            'average_capacity_utilization_pct': f"{np.mean([a['capacity_utilization_pct'] for a in capacity_data.values()]) if capacity_data else 0:.1f}%",
            'key_constraint': 'liquidity' if avg_spread < 2.0 else 'transaction_costs',
            'scalability_rating': 'High' if total_revenue_potential > 2000000 else 'Medium' if total_revenue_potential > 500000 else 'Low'
        }


def run_comprehensive_market_analysis(tickers: List[str] = None) -> Dict[str, any]:
    """
    Run the full market impact analysis pipeline.
    This generates all the metrics you need for your CV.
    """
    
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    print("🔍 Analyzing market microstructure and liquidity constraints...")
    
    analyzer = MarketImpactAnalyzer(tickers)
    
    # Step 1: Fetch comprehensive market data
    print("   📊 Fetching market data...")
    market_data = analyzer.fetch_comprehensive_market_data()
    
    # Step 2: Analyze liquidity constraints
    print("   💧 Analyzing liquidity constraints...")
    liquidity_metrics = analyzer.analyze_liquidity_constraints()
    
    # Step 3: Find cross-asset opportunities
    print("   🔄 Scanning cross-asset opportunities...")
    cross_asset_opps = analyzer.analyze_cross_asset_opportunities()
    
    # Step 4: Generate executive summary
    print("   📋 Generating executive summary...")
    executive_summary = analyzer.generate_executive_summary()
    
    print("✅ Market analysis complete!")
    
    return {
        'liquidity_metrics': liquidity_metrics,
        'cross_asset_opportunities': cross_asset_opps,
        'executive_summary': executive_summary,
        'analyzer': analyzer  # Return analyzer for further analysis
    }