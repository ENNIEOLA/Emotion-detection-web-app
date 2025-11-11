import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import time

class PortfolioAnalyzer:
    """
    Advanced Portfolio Analytics System
    Integrates multiple financial APIs for comprehensive portfolio analysis
    """
    
    def __init__(self):
        # Free API endpoints 
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.alpha_vantage_key = "demo"  # Replace with your free key from alphavantage.co
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        
    def fetch_crypto_data(self, coin_ids: List[str]) -> pd.DataFrame:
        """Fetch cryptocurrency data from CoinGecko API"""
        try:
            url = f"{self.coingecko_base}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': ','.join(coin_ids),
                'order': 'market_cap_desc',
                'sparkline': False,
                'price_change_percentage': '1h,24h,7d,30d'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
           
            available_cols = ['id', 'symbol', 'current_price', 'market_cap', 'total_volume']
            optional_cols = ['price_change_percentage_24h', 'price_change_percentage_7d_in_currency', 
                           'price_change_percentage_30d_in_currency']
            
            cols_to_return = available_cols.copy()
            for col in optional_cols:
                if col in df.columns:
                    cols_to_return.append(col)
            
            return df[cols_to_return]
        except Exception as e:
            print(f"Error fetching crypto data: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data(self, symbol: str) -> Dict:
        """Fetch stock data from Alpha Vantage API"""
        try:
            url = self.alpha_vantage_base
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%').rstrip('%'),
                    'volume': int(quote.get('06. volume', 0))
                }
            return {}
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return {}
    
    def calculate_portfolio_metrics(self, holdings: List[Dict]) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        total_value = sum(h['value'] for h in holdings)
        total_cost = sum(h['cost_basis'] for h in holdings)
        
        total_return = total_value - total_cost
        return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0
        
        for h in holdings:
            h['weight'] = (h['value'] / total_value * 100) if total_value > 0 else 0
        
        # Calculate diversification metrics
        weights = np.array([h['weight'] / 100 for h in holdings])
        herfindahl_index = np.sum(weights ** 2)
        diversification_ratio = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_return': total_return,
            'return_percentage': return_pct,
            'num_positions': len(holdings),
            'herfindahl_index': herfindahl_index,
            'diversification_ratio': diversification_ratio,
            'largest_position': max(holdings, key=lambda x: x['weight'])['asset'] if holdings else None,
            'largest_position_weight': max(h['weight'] for h in holdings) if holdings else 0
        }
    
    def calculate_risk_metrics(self, returns: List[float]) -> Dict:
        """Calculate portfolio risk metrics"""
        returns_array = np.array(returns)
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe Ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        mean_return = np.mean(returns_array) * 252
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns_array, 5)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'mean_return': mean_return
        }
    
    def generate_rebalancing_recommendations(self, holdings: List[Dict], 
                                            target_weights: Dict[str, float]) -> List[Dict]:
        """Generate portfolio rebalancing recommendations"""
        recommendations = []
        total_value = sum(h['value'] for h in holdings)
        
        for holding in holdings:
            current_weight = holding['weight']
            target_weight = target_weights.get(holding['asset'], 0)
            difference = target_weight - current_weight
            
            if abs(difference) > 2:  # Only recommend if difference > 2%
                target_value = total_value * (target_weight / 100)
                action_value = target_value - holding['value']
                
                recommendations.append({
                    'asset': holding['asset'],
                    'action': 'BUY' if action_value > 0 else 'SELL',
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'difference': difference,
                    'amount': abs(action_value)
                })
        
        return sorted(recommendations, key=lambda x: abs(x['difference']), reverse=True)
    
    def analyze_correlation(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between assets"""
        returns = price_data.pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix
    
    def generate_report(self, portfolio_metrics: Dict, risk_metrics: Dict, 
                       recommendations: List[Dict]) -> str:
        """Generate comprehensive portfolio analysis report"""
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        report.append("\nPORTFOLIO SUMMARY")
        report.append("-" * 60)
        report.append(f"Total Value:        ${portfolio_metrics['total_value']:,.2f}")
        report.append(f"Total Cost Basis:   ${portfolio_metrics['total_cost']:,.2f}")
        report.append(f"Total Return:       ${portfolio_metrics['total_return']:,.2f} ({portfolio_metrics['return_percentage']:.2f}%)")
        report.append(f"Number of Positions: {portfolio_metrics['num_positions']}")
        
        report.append("\nRISK METRICS")
        report.append("-" * 60)
        report.append(f"Annual Volatility:   {risk_metrics['volatility']:.2%}")
        report.append(f"Sharpe Ratio:        {risk_metrics['sharpe_ratio']:.3f}")
        report.append(f"Max Drawdown:        {risk_metrics['max_drawdown']:.2%}")
        report.append(f"Value at Risk (95%): {risk_metrics['var_95']:.2%}")
        report.append(f"Expected Return:     {risk_metrics['mean_return']:.2%}")
        
        report.append("\nDIVERSIFICATION")
        report.append("-" * 60)
        report.append(f"Diversification Ratio: {portfolio_metrics['diversification_ratio']:.2f}")
        report.append(f"Largest Position:      {portfolio_metrics['largest_position']} ({portfolio_metrics['largest_position_weight']:.1f}%)")
        
        if recommendations:
            report.append("\nREBALANCING RECOMMENDATIONS")
            report.append("-" * 60)
            for rec in recommendations[:5]:  # Top 5 recommendations
                report.append(f"{rec['action']:4} {rec['asset']:10} ${rec['amount']:>10,.2f} "
                            f"({rec['current_weight']:.1f}% → {rec['target_weight']:.1f}%)")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """Main execution function - Demo portfolio analysis"""
    # Fix Windows encoding issues
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("Initializing Portfolio Analytics System...\n")
    
    analyzer = PortfolioAnalyzer()
    
    # Sample portfolio holdings
    sample_portfolio = [
        {'asset': 'bitcoin', 'quantity': 0.5, 'cost_basis': 15000, 'value': 0},
        {'asset': 'ethereum', 'quantity': 2.0, 'cost_basis': 4000, 'value': 0},
        {'asset': 'cardano', 'quantity': 1000, 'cost_basis': 500, 'value': 0},
    ]
    
    # Fetch live crypto data
    print("Fetching live market data...\n")
    crypto_ids = [h['asset'] for h in sample_portfolio]
    crypto_data = analyzer.fetch_crypto_data(crypto_ids)
    
    if not crypto_data.empty:
        # Update portfolio values with live prices
        for holding in sample_portfolio:
            price_data = crypto_data[crypto_data['id'] == holding['asset']]
            if not price_data.empty:
                current_price = price_data.iloc[0]['current_price']
                holding['value'] = holding['quantity'] * current_price
                holding['current_price'] = current_price
        
        print("Current Holdings:")
        print("-" * 60)
        for h in sample_portfolio:
            profit = h['value'] - h['cost_basis']
            profit_pct = (profit / h['cost_basis'] * 100) if h['cost_basis'] > 0 else 0
            print(f"{h['asset'].upper():10} | Qty: {h['quantity']:>8.4f} | "
                  f"Value: ${h['value']:>10,.2f} | P&L: ${profit:>8,.2f} ({profit_pct:>6.2f}%)")
        
        # Calculate portfolio metrics
        portfolio_metrics = analyzer.calculate_portfolio_metrics(sample_portfolio)
        
        # Generate sample returns for risk analysis
        np.random.seed(42)
        sample_returns = np.random.normal(0.001, 0.02, 100)  # 100 days of returns
        risk_metrics = analyzer.calculate_risk_metrics(sample_returns)
        
        # Define target allocation
        target_allocation = {
            'bitcoin': 50,
            'ethereum': 35,
            'cardano': 15
        }
        
        recommendations = analyzer.generate_rebalancing_recommendations(
            sample_portfolio, target_allocation
        )
        
        # Generate and print report
        report = analyzer.generate_report(portfolio_metrics, risk_metrics, recommendations)
        print("\n" + report)
        
        # Display live market data
        print("\nLIVE MARKET DATA")
        print("-" * 60)
        display_cols = ['symbol', 'current_price']
        if 'price_change_percentage_24h' in crypto_data.columns:
            display_cols.append('price_change_percentage_24h')
        if 'price_change_percentage_7d_in_currency' in crypto_data.columns:
            display_cols.append('price_change_percentage_7d_in_currency')
        print(crypto_data[display_cols].to_string(index=False))
        
    else:
        print("Could not fetch market data. Please check your internet connection.")
    
    print("\nAnalysis complete!")
    print("\nTO CUSTOMIZE:")
    print("   1. Get free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key")
    print("   2. Replace 'demo' with your key in line 16")
    print("   3. Modify sample_portfolio with your actual holdings")
    print("   4. Add more assets and adjust target_allocation")


if __name__ == "__main__":
    main()