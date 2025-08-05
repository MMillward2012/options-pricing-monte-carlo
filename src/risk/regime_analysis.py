import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeAnalyzer:
    """
    Identify market regimes and their impact on options pricing.
    This is sophisticated risk management that Jane Street values highly.
    """
    
    def __init__(self, ticker='SPY', lookback_days=504):  # ~2 years
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.regimes = {}
        self.regime_transitions = {}
        
    def identify_volatility_regimes(self, n_regimes=3):
        """
        Use Gaussian Mixture Models to identify distinct volatility regimes.
        Low vol (trending), Medium vol (normal), High vol (crisis).
        """
        print(f"🎯 Identifying volatility regimes for {self.ticker}...")
        
        # Get market data
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period=f'{self.lookback_days}d', interval='1d')
        
        if hist.empty:
            return None
            
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate multiple volatility measures
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        vol_21d = returns.rolling(21).std() * np.sqrt(252)
        vol_63d = returns.rolling(63).std() * np.sqrt(252)
        
        # Additional regime indicators
        rolling_skew = returns.rolling(21).skew()
        rolling_kurt = returns.rolling(21).apply(lambda x: x.kurtosis())
        price_momentum = hist['Close'].pct_change(21)  # 21-day return
        
        # Create feature matrix for regime identification
        features = pd.DataFrame({
            'vol_5d': vol_5d,
            'vol_21d': vol_21d, 
            'vol_63d': vol_63d,
            'skewness': rolling_skew,
            'kurtosis': rolling_kurt,
            'momentum': price_momentum,
            'returns': returns
        }).dropna()
        
        if len(features) < 100:
            return None
            
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[['vol_21d', 'skewness', 'kurtosis']])
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
        regime_labels = gmm.fit_predict(features_scaled)
        
        # Add regime labels to features
        features['regime'] = regime_labels
        features['date'] = features.index
        
        # Characterize each regime
        regime_characteristics = {}
        regime_names = ['Low Vol', 'Medium Vol', 'High Vol']
        
        for i in range(n_regimes):
            regime_data = features[features['regime'] == i]
            
            regime_characteristics[i] = {
                'name': regime_names[i] if i < len(regime_names) else f'Regime {i}',
                'avg_volatility': regime_data['vol_21d'].mean(),
                'avg_returns': regime_data['returns'].mean() * 252,  # Annualized
                'avg_skewness': regime_data['skewness'].mean(),
                'avg_kurtosis': regime_data['kurtosis'].mean(),
                'frequency': len(regime_data) / len(features),
                'max_drawdown': self._calculate_max_drawdown(regime_data['returns']),
                'sharpe_ratio': (regime_data['returns'].mean() * 252) / (regime_data['returns'].std() * np.sqrt(252)) if regime_data['returns'].std() > 0 else 0
            }
        
        # Sort regimes by volatility for consistent naming
        sorted_regimes = sorted(regime_characteristics.items(), key=lambda x: x[1]['avg_volatility'])
        regime_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_regimes)}
        
        # Remap regime labels
        features['regime'] = features['regime'].map(regime_mapping)
        
        # Update characteristics with sorted order
        self.regime_characteristics = {new_id: char for new_id, (_, char) in enumerate(sorted_regimes)}
        self.regime_data = features
        
        return self.regime_characteristics
    
    def analyze_regime_transitions(self):
        """
        Analyze how markets transition between regimes.
        Critical for risk management and position sizing.
        """
        if not hasattr(self, 'regime_data'):
            return None
            
        regime_series = self.regime_data['regime']
        transitions = []
        
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] != regime_series.iloc[i-1]:
                transitions.append({
                    'date': regime_series.index[i],
                    'from_regime': regime_series.iloc[i-1],
                    'to_regime': regime_series.iloc[i]
                })
        
        if not transitions:
            return None
            
        transitions_df = pd.DataFrame(transitions)
        
        # Calculate transition probabilities
        transition_matrix = pd.crosstab(
            regime_series.shift(1), 
            regime_series, 
            normalize='index'
        ).fillna(0)
        
        # Average regime duration
        regime_durations = {}
        for regime in regime_series.unique():
            regime_periods = []
            current_length = 0
            
            for r in regime_series:
                if r == regime:
                    current_length += 1
                else:
                    if current_length > 0:
                        regime_periods.append(current_length)
                    current_length = 0
            
            if current_length > 0:
                regime_periods.append(current_length)
                
            regime_durations[regime] = np.mean(regime_periods) if regime_periods else 0
        
        return {
            'transition_matrix': transition_matrix,
            'transitions': transitions_df,
            'avg_regime_duration_days': regime_durations,
            'total_transitions': len(transitions)
        }
    
    def calculate_regime_dependent_greeks(self, S0=100, K=105, T=0.25, r=0.05):
        """
        Calculate how option Greeks change across different market regimes.
        This shows sophisticated understanding of regime-dependent risk.
        """
        from src.models.black_scholes import black_scholes_call_price
        from scipy.stats import norm
        
        if not hasattr(self, 'regime_characteristics'):
            return None
            
        regime_greeks = {}
        
        for regime_id, characteristics in self.regime_characteristics.items():
            # Use regime-specific volatility
            sigma = characteristics['avg_volatility']
            
            # Calculate Black-Scholes Greeks for this regime
            d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Greeks calculations
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
            theta = -(S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                     r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
            vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Option price
            option_price = black_scholes_call_price(S0, K, T, r, sigma)
            
            regime_greeks[regime_id] = {
                'regime_name': characteristics['name'],
                'volatility_used': sigma,
                'option_price': option_price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'gamma_risk_1pct_move': 0.5 * gamma * (S0 * 0.01)**2,  # P&L from 1% move
                'theta_decay_daily': theta,
                'vega_risk_1vol_point': vega * 0.01  # P&L from 1 vol point move
            }
        
        return regime_greeks
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown for a return series"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def stress_test_across_regimes(self, portfolio_delta=1000, portfolio_gamma=50, portfolio_vega=200):
        """
        Stress test a portfolio across different market regimes.
        This is exactly what Jane Street risk managers do.
        """
        if not hasattr(self, 'regime_characteristics'):
            return None
            
        stress_results = {}
        
        # Define stress scenarios for each regime
        stress_scenarios = {
            0: {'price_shocks': [-0.15, -0.05, 0.05, 0.15], 'vol_shocks': [-0.05, 0.05, 0.10]},  # Low vol
            1: {'price_shocks': [-0.25, -0.10, 0.10, 0.25], 'vol_shocks': [-0.08, 0.08, 0.15]},  # Medium vol
            2: {'price_shocks': [-0.40, -0.20, 0.20, 0.40], 'vol_shocks': [-0.15, 0.15, 0.25]}   # High vol
        }
        
        for regime_id, characteristics in self.regime_characteristics.items():
            regime_stress = {}
            scenarios = stress_scenarios.get(regime_id, stress_scenarios[1])  # Default to medium vol
            
            worst_case_pnl = 0
            best_case_pnl = 0
            
            for price_shock in scenarios['price_shocks']:
                for vol_shock in scenarios['vol_shocks']:
                    # Calculate P&L components
                    delta_pnl = portfolio_delta * price_shock * 100  # Assuming $100 stock price
                    gamma_pnl = 0.5 * portfolio_gamma * (price_shock * 100)**2
                    vega_pnl = portfolio_vega * vol_shock * 100  # Convert to vol points
                    
                    total_pnl = delta_pnl + gamma_pnl + vega_pnl
                    
                    worst_case_pnl = min(worst_case_pnl, total_pnl)
                    best_case_pnl = max(best_case_pnl, total_pnl)
            
            regime_stress = {
                'regime_name': characteristics['name'],
                'regime_frequency': characteristics['frequency'],
                'worst_case_pnl': worst_case_pnl,
                'best_case_pnl': best_case_pnl,
                'expected_vol': characteristics['avg_volatility'],
                'risk_adjusted_capacity': abs(worst_case_pnl) * characteristics['frequency']  # Risk-weighted exposure
            }
            
            stress_results[regime_id] = regime_stress
        
        # Calculate portfolio-wide metrics
        total_risk_weighted_exposure = sum(result['risk_adjusted_capacity'] for result in stress_results.values())
        
        return {
            'regime_stress_results': stress_results,
            'total_risk_weighted_exposure': total_risk_weighted_exposure,
            'diversification_benefit': len(stress_results) > 1,
            'max_single_regime_loss': min(result['worst_case_pnl'] for result in stress_results.values())
        }
    
    def generate_regime_executive_summary(self):
        """
        Generate executive summary of regime analysis.
        Perfect for presenting to Jane Street interviewers.
        """
        # Run all analyses
        regime_chars = self.identify_volatility_regimes()
        transitions = self.analyze_regime_transitions()
        greeks = self.calculate_regime_dependent_greeks()
        stress = self.stress_test_across_regimes()
        
        if not regime_chars:
            return {'error': 'Insufficient data for regime analysis'}
            
        # Calculate key insights
        vol_range = max(char['avg_volatility'] for char in regime_chars.values()) - \
                   min(char['avg_volatility'] for char in regime_chars.values())
        
        most_common_regime = max(regime_chars.items(), key=lambda x: x[1]['frequency'])
        highest_risk_regime = max(regime_chars.items(), key=lambda x: x[1]['avg_volatility'])
        
        return {
            'analysis_ticker': self.ticker,
            'regimes_identified': len(regime_chars),
            'volatility_range': f"{vol_range:.1%}",
            'most_common_regime': most_common_regime[1]['name'],
            'most_common_regime_frequency': f"{most_common_regime[1]['frequency']:.1%}",
            'highest_risk_regime': highest_risk_regime[1]['name'],
            'highest_risk_vol': f"{highest_risk_regime[1]['avg_volatility']:.1%}",
            'regime_transitions_per_year': transitions['total_transitions'] / 2 * 252 if transitions else 0,
            'avg_regime_persistence_days': np.mean(list(transitions['avg_regime_duration_days'].values())) if transitions else 0,
            'max_stress_loss': stress['max_single_regime_loss'] if stress else 0,
            'risk_diversification_available': stress['diversification_benefit'] if stress else False,
            'model_complexity_required': 'High' if vol_range > 0.3 else 'Medium' if vol_range > 0.15 else 'Low'
        }

def run_regime_analysis(ticker='SPY'):
    """
    Run comprehensive regime analysis.
    This demonstrates institutional-level risk management understanding.
    """
    print("🎭 MARKET REGIME ANALYSIS")
    print("="*40)
    
    analyzer = MarketRegimeAnalyzer(ticker)
    summary = analyzer.generate_regime_executive_summary()
    
    return summary, analyzer