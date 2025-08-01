import pandas as pd

class GreeksSurfaceAnalyzer:
    """
    Multi-dimensional risk analysis.
    This shows you understand portfolio-level risk management.
    """
    
    def generate_greeks_surface(self, S_range, T_range, vol_range):
        """
        Calculate Greeks across price/time/vol dimensions
        Shows you can handle multi-factor risk
        """
        results = []
        
        for S in S_range:
            for T in T_range:
                for vol in vol_range:
                    if T > 0:
                        greeks = self.calculate_all_greeks(S, K, T, r, vol)
                        results.append({
                            'spot': S,
                            'time': T,
                            'vol': vol,
                            **greeks
                        })
        
        return pd.DataFrame(results)
    
    def identify_risk_concentrations(self, portfolio_positions):
        """
        Find where your portfolio is most vulnerable
        This is actual risk management, not just academic exercise
        """
        total_delta = sum(pos['quantity'] * pos['delta'] for pos in portfolio_positions)
        total_gamma = sum(pos['quantity'] * pos['gamma'] for pos in portfolio_positions)
        total_vega = sum(pos['quantity'] * pos['vega'] for pos in portfolio_positions)
        
        # Risk concentration analysis
        risk_metrics = {
            'net_delta': total_delta,
            'gamma_risk': total_gamma,
            'vol_risk': total_vega,
            'largest_single_position': max(pos['notional'] for pos in portfolio_positions),
            'concentration_ratio': max(pos['notional'] for pos in portfolio_positions) / sum(pos['notional'] for pos in portfolio_positions)
        }
        
        return risk_metrics