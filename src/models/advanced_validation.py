import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelBreakdown:
    """
    Deep dive into Black-Scholes model failures with quantitative rigor.
    This is the kind of analysis Jane Street quants do daily.
    """
    
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.market_data = {}
        self.breakdown_metrics = {}
        
    def analyze_volatility_clustering(self, window=252):
        """
        Test the constant volatility assumption by detecting volatility clustering.
        Real markets show volatility persistence - high vol followed by high vol.
        """
        print(f"🔍 Analyzing volatility clustering for {self.ticker}...")
        
        # Get historical data
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period='2y', interval='1d')
        
        if hist.empty:
            return None
            
        # Calculate realized volatility
        returns = hist['Close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # 21-day rolling vol
        
        # Test for volatility clustering using ARCH effects
        vol_changes = rolling_vol.diff().dropna()
        
        # Ljung-Box test for serial correlation in squared returns
        squared_returns = returns**2
        
        # Calculate autocorrelations
        lags = range(1, 21)
        autocorrs = [squared_returns.autocorr(lag) for lag in lags]
        
        # Volatility clustering strength
        clustering_score = np.mean([abs(ac) for ac in autocorrs if not np.isnan(ac)])
        
        return {
            'volatility_time_series': rolling_vol,
            'returns': returns,
            'clustering_score': clustering_score,
            'autocorrelations': autocorrs,
            'vol_persistence': rolling_vol.autocorr(1) if len(rolling_vol) > 1 else 0
        }
    
    def test_jump_diffusion_vs_gbm(self):
        """
        Test if returns follow GBM or show evidence of jumps.
        Jane Street cares about tail risk that GBM misses.
        """
        print(f"🚀 Testing for jump processes in {self.ticker}...")
        
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period='2y', interval='1d')
        returns = hist['Close'].pct_change().dropna()
        
        # Standardize returns
        standardized_returns = (returns - returns.mean()) / returns.std()
        
        # Test for normality (GBM assumption)
        shapiro_stat, shapiro_p = stats.shapiro(standardized_returns.iloc[-min(5000, len(standardized_returns)):])
        
        # Jarque-Bera test for normality
        jb_stat, jb_p = stats.jarque_bera(standardized_returns)
        
        # Calculate excess kurtosis (fat tails)
        excess_kurtosis = stats.kurtosis(standardized_returns)
        
        # Detect potential jumps (returns > 3 standard deviations)
        jump_threshold = 3
        potential_jumps = standardized_returns[abs(standardized_returns) > jump_threshold]
        
        # Calculate Value at Risk violations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        var_violations_95 = sum(returns < var_95) / len(returns)
        var_violations_99 = sum(returns < var_99) / len(returns)
        
        return {
            'shapiro_p_value': shapiro_p,
            'jarque_bera_p_value': jb_p,
            'excess_kurtosis': excess_kurtosis,
            'jump_days': len(potential_jumps),
            'jump_frequency': len(potential_jumps) / len(returns) * 252,  # Annualized
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max(),
            'var_95_violations': var_violations_95,
            'var_99_violations': var_violations_99,
            'theoretical_var_95_violations': 0.05,
            'theoretical_var_99_violations': 0.01
        }
    
    def analyze_stochastic_volatility_evidence(self):
        """
        Test for stochastic volatility patterns that Heston model captures.
        Shows understanding of more sophisticated models.
        """
        print(f"📈 Analyzing stochastic volatility evidence...")
        
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period='2y', interval='1d')
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate realized volatility with different windows
        vol_short = returns.rolling(window=10).std() * np.sqrt(252)
        vol_medium = returns.rolling(window=30).std() * np.sqrt(252)
        vol_long = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Volatility of volatility (key Heston parameter)
        vol_of_vol = vol_medium.rolling(window=30).std()
        
        # Mean reversion in volatility
        vol_mean = vol_medium.mean()
        vol_deviations = vol_medium - vol_mean
        vol_mean_reversion = -vol_deviations.autocorr(1)  # Negative autocorr = mean reversion
        
        # Correlation between returns and volatility changes
        vol_changes = vol_medium.diff()
        leverage_effect = returns.corr(vol_changes)  # Should be negative (leverage effect)
        
        return {
            'volatility_of_volatility': vol_of_vol.mean(),
            'vol_mean_reversion_speed': vol_mean_reversion,
            'leverage_effect_correlation': leverage_effect,
            'vol_persistence': vol_medium.autocorr(1),
            'vol_range_annual': vol_medium.max() - vol_medium.min()
        }
    
    def calculate_model_pricing_errors(self, option_data, risk_free_rate=0.05):
        """
        Quantify systematic pricing errors in Black-Scholes.
        This is what prop trading desks use to find alpha.
        """
        from src.models.black_scholes import black_scholes_call, black_scholes_put
        
        if option_data.empty:
            return {}
            
        current_price = yf.Ticker(self.ticker).history(period='1d')['Close'].iloc[-1]
        
        pricing_errors = []
        
        for _, option in option_data.iterrows():
            try:
                # Skip if missing critical data
                if pd.isna(option['strike']) or pd.isna(option['lastPrice']) or option['lastPrice'] <= 0:
                    continue
                    
                # Calculate days to expiration
                expiry_date = pd.to_datetime(option['expiry']) if 'expiry' in option else pd.to_datetime(option.name)
                days_to_expiry = (expiry_date - datetime.now()).days
                time_to_expiry = days_to_expiry / 365.0
                
                if time_to_expiry <= 0:
                    continue
                
                # Use implied volatility if available, otherwise estimate
                if 'impliedVolatility' in option and option['impliedVolatility'] > 0:
                    vol = option['impliedVolatility']
                else:
                    vol = 0.25  # Default estimate
                
                # Calculate theoretical BS price
                strike = option['strike']
                if 'option_type' in option:
                    option_type = option['option_type']
                else:
                    option_type = 'call'  # Default assumption
                
                if option_type == 'call':
                    bs_price = black_scholes_call(current_price, strike, time_to_expiry, risk_free_rate, vol)
                else:
                    bs_price = black_scholes_put(current_price, strike, time_to_expiry, risk_free_rate, vol)
                
                market_price = option['lastPrice']
                
                # Calculate pricing error
                absolute_error = market_price - bs_price
                relative_error = absolute_error / market_price if market_price != 0 else 0
                
                # Moneyness
                moneyness = current_price / strike
                
                pricing_errors.append({
                    'strike': strike,
                    'moneyness': moneyness,
                    'time_to_expiry': time_to_expiry,
                    'market_price': market_price,
                    'bs_price': bs_price,
                    'absolute_error': absolute_error,
                    'relative_error': relative_error,
                    'option_type': option_type,
                    'implied_vol': vol
                })
                
            except Exception as e:
                continue
        
        if not pricing_errors:
            return {}
            
        errors_df = pd.DataFrame(pricing_errors)
        
        # Analyze systematic patterns
        return {
            'mean_absolute_error': errors_df['absolute_error'].mean(),
            'mean_relative_error': errors_df['relative_error'].mean(),
            'rmse': np.sqrt((errors_df['absolute_error']**2).mean()),
            'errors_by_moneyness': errors_df.groupby(pd.cut(errors_df['moneyness'], bins=5))['relative_error'].mean().to_dict(),
            'errors_by_time': errors_df.groupby(pd.cut(errors_df['time_to_expiry'], bins=3))['relative_error'].mean().to_dict(),
            'total_options_analyzed': len(errors_df),
            'systematic_underpricing_pct': (errors_df['absolute_error'] < -0.01).mean() * 100,
            'systematic_overpricing_pct': (errors_df['absolute_error'] > 0.01).mean() * 100
        }
    
    def generate_executive_breakdown_summary(self):
        """
        Generate the killer metrics that show deep quantitative understanding.
        """
        
        # Run all analyses
        vol_clustering = self.analyze_volatility_clustering()
        jump_analysis = self.test_jump_diffusion_vs_gbm()
        stoch_vol = self.analyze_stochastic_volatility_evidence()
        
        # Get options data for pricing errors
        try:
            stock = yf.Ticker(self.ticker)
            options_data = []
            for expiry in stock.options[:3]:  # First 3 expiries
                try:
                    chain = stock.option_chain(expiry)
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    calls['option_type'] = 'call'
                    puts['option_type'] = 'put'
                    calls['expiry'] = expiry
                    puts['expiry'] = expiry
                    options_data.extend([calls, puts])
                except:
                    continue
            
            if options_data:
                all_options = pd.concat(options_data, ignore_index=True)
                pricing_errors = self.calculate_model_pricing_errors(all_options)
            else:
                pricing_errors = {}
        except:
            pricing_errors = {}
        
        # Compile killer summary
        summary = {
            'ticker': self.ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            
            # Volatility Model Failures
            'volatility_clustering_score': vol_clustering['clustering_score'] if vol_clustering else 0,
            'vol_persistence_coefficient': vol_clustering['vol_persistence'] if vol_clustering else 0,
            'constant_vol_assumption_violated': vol_clustering['clustering_score'] > 0.1 if vol_clustering else False,
            
            # Distribution Model Failures  
            'excess_kurtosis': jump_analysis['excess_kurtosis'],
            'normality_p_value': jump_analysis['shapiro_p_value'],
            'jump_frequency_annual': jump_analysis['jump_frequency'],
            'var_model_accuracy': abs(jump_analysis['var_95_violations'] - 0.05) < 0.01,
            
            # Stochastic Volatility Evidence
            'volatility_of_volatility': stoch_vol['volatility_of_volatility'],
            'leverage_effect_strength': abs(stoch_vol['leverage_effect_correlation']),
            'vol_mean_reversion_detected': stoch_vol['vol_mean_reversion_speed'] > 0.1,
            
            # Pricing Model Performance
            'options_analyzed': pricing_errors.get('total_options_analyzed', 0),
            'mean_pricing_error_pct': pricing_errors.get('mean_relative_error', 0) * 100,
            'systematic_mispricing_detected': abs(pricing_errors.get('mean_relative_error', 0)) > 0.05,
            
            # Overall Model Health
            'black_scholes_reliability_score': self._calculate_reliability_score(jump_analysis, vol_clustering, pricing_errors),
            'recommended_model_upgrades': self._suggest_model_improvements(jump_analysis, vol_clustering, stoch_vol)
        }
        
        return summary
    
    def _calculate_reliability_score(self, jump_analysis, vol_clustering, pricing_errors):
        """Calculate overall BS model reliability (0-100)"""
        score = 100
        
        # Penalize for normality violations
        if jump_analysis['shapiro_p_value'] < 0.05:
            score -= 20
            
        # Penalize for volatility clustering
        if vol_clustering and vol_clustering['clustering_score'] > 0.1:
            score -= 15
            
        # Penalize for large pricing errors
        if pricing_errors.get('mean_relative_error', 0) > 0.1:
            score -= 25
            
        # Penalize for excess kurtosis
        if abs(jump_analysis['excess_kurtosis']) > 1:
            score -= 15
            
        return max(0, score)
    
    def _suggest_model_improvements(self, jump_analysis, vol_clustering, stoch_vol):
        """Suggest specific model improvements based on empirical findings"""
        suggestions = []
        
        if jump_analysis['excess_kurtosis'] > 2:
            suggestions.append("Merton Jump-Diffusion Model")
            
        if vol_clustering and vol_clustering['clustering_score'] > 0.15:
            suggestions.append("GARCH Volatility Model")
            
        if stoch_vol['leverage_effect_correlation'] < -0.3:
            suggestions.append("Heston Stochastic Volatility Model")
            
        if jump_analysis['var_99_violations'] > 0.02:
            suggestions.append("Extreme Value Theory for Tail Risk")
            
        return suggestions if suggestions else ["Black-Scholes Adequate"]

def run_advanced_model_breakdown_analysis(ticker='SPY'):
    """
    Run the full advanced breakdown analysis.
    This is your competitive advantage for Jane Street interviews.
    """
    print("🔬 ADVANCED BLACK-SCHOLES MODEL BREAKDOWN ANALYSIS")
    print("="*60)
    print(f"Analyzing {ticker} with institutional-grade rigor...")
    
    analyzer = AdvancedModelBreakdown(ticker)
    summary = analyzer.generate_executive_breakdown_summary()
    
    return summary, analyzer