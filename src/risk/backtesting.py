# src/risk/backtesting.py
# Enhanced Historical Backtesting Engine with P&L Tracking

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResults:
    """Container for comprehensive backtest results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    information_ratio: float
    calmar_ratio: float
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series
    drawdown_series: pd.Series
    crisis_performance: Dict[str, Dict]
    trade_statistics: Dict[str, Any]
    risk_metrics: Dict[str, float]

class HistoricalBacktester:
    """
    Institutional-grade backtesting engine with crisis period analysis
    """
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.crisis_periods = {
            'COVID_CRASH': ('2020-02-19', '2020-04-07'),
            'INFLATION_SURGE': ('2021-11-01', '2022-10-31'), 
            'BANKING_CRISIS': ('2023-03-01', '2023-05-31'),
            'TECH_SELLOFF': ('2022-01-01', '2022-12-31')
        }
        
    def run_strategy_backtest(self, 
                            strategy_func: callable,
                            ticker: str = 'SPY',
                            start_date: str = '2020-01-01',
                            end_date: str = '2024-12-31',
                            benchmark_ticker: str = 'SPY') -> BacktestResults:
        """
        Run comprehensive historical backtest with actual P&L tracking
        """
        print(f"🚀 Starting Historical Backtest")
        print(f"📊 Strategy: {strategy_func.__name__}")
        print(f"🎯 Ticker: {ticker} | Benchmark: {benchmark_ticker}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.0f}")
        
        # Fetch historical data
        historical_data = self._fetch_comprehensive_data(ticker, start_date, end_date)
        benchmark_data = self._fetch_benchmark_data(benchmark_ticker, start_date, end_date)
        
        # Run daily strategy simulation
        daily_results = self._simulate_daily_strategy(
            strategy_func, historical_data, ticker
        )
        
        # Calculate comprehensive performance metrics
        backtest_results = self._calculate_performance_metrics(
            daily_results, benchmark_data, start_date, end_date
        )
        
        # Analyze crisis period performance
        crisis_performance = self._analyze_crisis_performance(
            daily_results, ticker
        )
        
        # Generate detailed results
        results = BacktestResults(
            total_return=backtest_results['total_return'],
            annualized_return=backtest_results['annualized_return'],
            volatility=backtest_results['volatility'],
            sharpe_ratio=backtest_results['sharpe_ratio'],
            max_drawdown=backtest_results['max_drawdown'],
            information_ratio=backtest_results['information_ratio'],
            calmar_ratio=backtest_results['calmar_ratio'],
            daily_pnl=daily_results['daily_pnl'],
            cumulative_pnl=daily_results['cumulative_pnl'],
            drawdown_series=daily_results['drawdown_series'],
            crisis_performance=crisis_performance,
            trade_statistics=daily_results['trade_stats'],
            risk_metrics=backtest_results['risk_metrics']
        )
        
        # Display results
        self._display_backtest_summary(results)
        
        return results
    
    def _fetch_comprehensive_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch comprehensive historical data including options"""
        print(f"📥 Fetching historical data for {ticker}...")
        
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)
            
            # Calculate additional technical indicators
            hist_data['Returns'] = hist_data['Close'].pct_change()
            hist_data['Volatility_20D'] = hist_data['Returns'].rolling(20).std() * np.sqrt(252)
            hist_data['VIX_Proxy'] = hist_data['Volatility_20D'] * 100
            
            # Fetch VIX data if available
            try:
                vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
                hist_data['VIX'] = vix_data.reindex(hist_data.index, method='ffill')
            except:
                hist_data['VIX'] = hist_data['VIX_Proxy']
            
            # Calculate implied volatility proxies
            hist_data['IV_Rank'] = hist_data['VIX'].rolling(252).rank(pct=True)
            hist_data['Vol_Regime'] = pd.cut(hist_data['IV_Rank'], 
                                           bins=[0, 0.25, 0.75, 1.0], 
                                           labels=['Low_Vol', 'Medium_Vol', 'High_Vol'])
            
            print(f"✅ Successfully fetched {len(hist_data)} days of data")
            return hist_data.dropna()
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def _fetch_benchmark_data(self, benchmark_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch benchmark data for comparison"""
        benchmark = yf.Ticker(benchmark_ticker)
        benchmark_data = benchmark.history(start=start_date, end=end_date)
        benchmark_data['Returns'] = benchmark_data['Close'].pct_change()
        return benchmark_data.dropna()
    
    def _simulate_daily_strategy(self, strategy_func: callable, data: pd.DataFrame, ticker: str) -> Dict:
        """Simulate strategy execution day by day"""
        print("🔄 Running daily strategy simulation...")
        
        daily_pnl = []
        daily_positions = []
        trade_count = 0
        current_capital = self.initial_capital
        
        for i, (date, row) in enumerate(data.iterrows()):
            if i < 30:  # Need some history for calculations
                daily_pnl.append(0)
                continue
            
            # Get historical window for strategy
            hist_window = data.iloc[max(0, i-30):i+1]
            
            # Execute strategy
            try:
                strategy_result = strategy_func(hist_window, row, current_capital)
                
                # Parse strategy result
                position_pnl = strategy_result.get('pnl', 0)
                position_size = strategy_result.get('position_size', 0)
                trade_signal = strategy_result.get('trade_signal', 'HOLD')
                
                daily_pnl.append(position_pnl)
                daily_positions.append(position_size)
                current_capital += position_pnl
                
                if trade_signal != 'HOLD':
                    trade_count += 1
                    
            except Exception as e:
                daily_pnl.append(0)
                daily_positions.append(0)
        
        # Convert to series
        daily_pnl_series = pd.Series(daily_pnl, index=data.index)
        cumulative_pnl = daily_pnl_series.cumsum()
        
        # Calculate drawdown
        rolling_max = cumulative_pnl.expanding().max()
        drawdown_series = (cumulative_pnl - rolling_max)
        
        trade_stats = {
            'total_trades': trade_count,
            'avg_daily_pnl': daily_pnl_series.mean(),
            'pnl_std': daily_pnl_series.std(),
            'positive_days': (daily_pnl_series > 0).sum(),
            'negative_days': (daily_pnl_series < 0).sum(),
            'win_rate': (daily_pnl_series > 0).mean()
        }
        
        return {
            'daily_pnl': daily_pnl_series,
            'cumulative_pnl': cumulative_pnl,
            'drawdown_series': drawdown_series,
            'trade_stats': trade_stats,
            'final_capital': current_capital
        }
    
    def _calculate_performance_metrics(self, daily_results: Dict, benchmark_data: pd.DataFrame, 
                                     start_date: str, end_date: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        daily_pnl = daily_results['daily_pnl']
        cumulative_pnl = daily_results['cumulative_pnl']
        
        # Basic performance metrics
        total_return = cumulative_pnl.iloc[-1] / self.initial_capital
        days = len(daily_pnl)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # Risk metrics
        daily_returns = daily_pnl / self.initial_capital
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        max_drawdown = daily_results['drawdown_series'].min() / self.initial_capital
        
        # Information ratio (vs benchmark)
        benchmark_returns = benchmark_data['Returns'].reindex(daily_pnl.index, method='ffill')
        excess_returns = daily_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Advanced risk metrics
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'risk_metrics': {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        }
    
    def _analyze_crisis_performance(self, daily_results: Dict, ticker: str) -> Dict:
        """Analyze performance during crisis periods"""
        crisis_performance = {}
        daily_pnl = daily_results['daily_pnl']
        
        for crisis_name, (start, end) in self.crisis_periods.items():
            try:
                crisis_pnl = daily_pnl[start:end]
                if len(crisis_pnl) > 0:
                    crisis_performance[crisis_name] = {
                        'total_pnl': crisis_pnl.sum(),
                        'daily_avg_pnl': crisis_pnl.mean(),
                        'volatility': crisis_pnl.std(),
                        'max_daily_loss': crisis_pnl.min(),
                        'max_daily_gain': crisis_pnl.max(),
                        'positive_days': (crisis_pnl > 0).sum(),
                        'total_days': len(crisis_pnl)
                    }
            except:
                crisis_performance[crisis_name] = {'error': 'Data not available'}
        
        return crisis_performance
    
    def _display_backtest_summary(self, results: BacktestResults):
        """Display comprehensive backtest summary"""
        print(f"\n" + "="*60)
        print("📊 HISTORICAL BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n💰 PERFORMANCE METRICS:")
        print(f"   Total Return: {results.total_return:.2%}")
        print(f"   Annualized Return: {results.annualized_return:.2%}")
        print(f"   Volatility: {results.volatility:.2%}")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Information Ratio: {results.information_ratio:.2f}")
        print(f"   Calmar Ratio: {results.calmar_ratio:.2f}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"   Max Drawdown: {results.max_drawdown:.2%}")
        print(f"   VaR (95%): {results.risk_metrics['var_95']:.2%}")
        print(f"   CVaR (95%): {results.risk_metrics['cvar_95']:.2%}")
        print(f"   Skewness: {results.risk_metrics['skewness']:.2f}")
        print(f"   Kurtosis: {results.risk_metrics['kurtosis']:.2f}")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades: {results.trade_statistics['total_trades']:,}")
        print(f"   Win Rate: {results.trade_statistics['win_rate']:.2%}")
        print(f"   Avg Daily P&L: ${results.trade_statistics['avg_daily_pnl']:,.2f}")
        print(f"   P&L Std Dev: ${results.trade_statistics['pnl_std']:,.2f}")
        
        print(f"\n🚨 CRISIS PERIOD PERFORMANCE:")
        for crisis, performance in results.crisis_performance.items():
            if 'error' not in performance:
                print(f"   {crisis}:")
                print(f"     Total P&L: ${performance['total_pnl']:,.0f}")
                print(f"     Avg Daily P&L: ${performance['daily_avg_pnl']:,.0f}")
                print(f"     Max Daily Loss: ${performance['max_daily_loss']:,.0f}")

def sample_delta_neutral_strategy(hist_window: pd.DataFrame, current_row: pd.Series, capital: float) -> Dict:
    """
    Sample delta-neutral options strategy for backtesting
    """
    
    # Simple volatility-based strategy
    current_vol = current_row['Volatility_20D']
    avg_vol = hist_window['Volatility_20D'].mean()
    vol_zscore = (current_vol - avg_vol) / hist_window['Volatility_20D'].std()
    
    # Position sizing based on volatility regime
    if abs(vol_zscore) > 1.5:  # High vol regime - sell volatility
        position_size = min(capital * 0.02, 50000)  # 2% risk, max $50k
        expected_pnl = position_size * 0.001 * np.sign(-vol_zscore)  # Simple P&L model
        trade_signal = 'SELL_VOL' if vol_zscore > 0 else 'BUY_VOL'
    else:
        position_size = 0
        expected_pnl = 0
        trade_signal = 'HOLD'
    
    # Add some noise to simulate real trading
    actual_pnl = expected_pnl + np.random.normal(0, abs(expected_pnl) * 0.1)
    
    return {
        'pnl': actual_pnl,
        'position_size': position_size,
        'trade_signal': trade_signal,
        'vol_zscore': vol_zscore
    }

# Example usage
if __name__ == "__main__":
    # Initialize backtester
    backtester = HistoricalBacktester(initial_capital=1_000_000)
    
    # Run backtest on sample strategy
    results = backtester.run_strategy_backtest(
        strategy_func=sample_delta_neutral_strategy,
        ticker='SPY',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    results.cumulative_pnl.plot(title='Cumulative P&L')
    plt.ylabel('P&L ($)')
    
    plt.subplot(2, 2, 2)
    results.drawdown_series.plot(title='Drawdown', color='red')
    plt.ylabel('Drawdown ($)')
    
    plt.subplot(2, 2, 3)
    results.daily_pnl.rolling(30).mean().plot(title='30-Day Rolling Avg P&L')
    plt.ylabel('Daily P&L ($)')
    
    plt.subplot(2, 2, 4)
    results.daily_pnl.hist(bins=50, alpha=0.7, title='Daily P&L Distribution')
    plt.xlabel('Daily P&L ($)')
    
    plt.tight_layout()
    plt.show()