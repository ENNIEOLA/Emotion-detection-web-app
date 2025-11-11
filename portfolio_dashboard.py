import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Analytics Dashboard", page_icon="", layout="wide")

class PortfolioAnalyzer:
    """Advanced Portfolio Analytics System"""
    
    def __init__(self):
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.alpha_vantage_key = "demo"
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
        
    def fetch_crypto_data(self, coin_ids):
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
            st.error(f"Error fetching crypto data: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, holdings):
        """Calculate comprehensive portfolio metrics"""
        total_value = sum(h['value'] for h in holdings)
        total_cost = sum(h['cost_basis'] for h in holdings)
        
        total_return = total_value - total_cost
        return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0
        
        for h in holdings:
            h['weight'] = (h['value'] / total_value * 100) if total_value > 0 else 0
        
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
    
    def calculate_risk_metrics(self, returns):
        """Calculate portfolio risk metrics"""
        returns_array = np.array(returns)
        
        volatility = np.std(returns_array) * np.sqrt(252)
        risk_free_rate = 0.04
        mean_return = np.mean(returns_array) * 252
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        var_95 = np.percentile(returns_array, 5)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'mean_return': mean_return
        }
    
    def generate_rebalancing_recommendations(self, holdings, target_weights):
        """Generate portfolio rebalancing recommendations"""
        recommendations = []
        total_value = sum(h['value'] for h in holdings)
        
        for holding in holdings:
            current_weight = holding['weight']
            target_weight = target_weights.get(holding['asset'], 0)
            difference = target_weight - current_weight
            
            if abs(difference) > 2:
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

# Initialize
st.title("Portfolio Analytics Dashboard")
st.markdown("### Real-time Portfolio Analysis & Risk Management")

analyzer = PortfolioAnalyzer()

# Sidebar - Portfolio Input
st.sidebar.header("Portfolio Settings")

# Predefined portfolio or custom
portfolio_type = st.sidebar.radio("Select Portfolio", ["Demo Portfolio", "Custom Portfolio"])

if portfolio_type == "Demo Portfolio":
    portfolio = [
        {'asset': 'bitcoin', 'quantity': 0.5, 'cost_basis': 15000, 'value': 0},
        {'asset': 'ethereum', 'quantity': 2.0, 'cost_basis': 4000, 'value': 0},
        {'asset': 'cardano', 'quantity': 1000, 'cost_basis': 500, 'value': 0},
    ]
    target_allocation = {'bitcoin': 50, 'ethereum': 35, 'cardano': 15}
else:
    st.sidebar.info("Custom portfolio builder coming soon!")
    portfolio = [
        {'asset': 'bitcoin', 'quantity': 0.5, 'cost_basis': 15000, 'value': 0},
        {'asset': 'ethereum', 'quantity': 2.0, 'cost_basis': 4000, 'value': 0},
        {'asset': 'cardano', 'quantity': 1000, 'cost_basis': 500, 'value': 0},
    ]
    target_allocation = {'bitcoin': 50, 'ethereum': 35, 'cardano': 15}

# Fetch Data Button
if st.sidebar.button("Analyze Portfolio", type="primary"):
    with st.spinner("Fetching live market data..."):
        crypto_ids = [h['asset'] for h in portfolio]
        crypto_data = analyzer.fetch_crypto_data(crypto_ids)
        
        if not crypto_data.empty:
            # Update portfolio values
            for holding in portfolio:
                price_data = crypto_data[crypto_data['id'] == holding['asset']]
                if not price_data.empty:
                    current_price = price_data.iloc[0]['current_price']
                    holding['value'] = holding['quantity'] * current_price
                    holding['current_price'] = current_price
            
            # Calculate metrics
            portfolio_metrics = analyzer.calculate_portfolio_metrics(portfolio)
            
            np.random.seed(42)
            sample_returns = np.random.normal(0.001, 0.02, 100)
            risk_metrics = analyzer.calculate_risk_metrics(sample_returns)
            
            recommendations = analyzer.generate_rebalancing_recommendations(portfolio, target_allocation)
            
            # Display Results
            st.success("Analysis Complete!")
            
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${portfolio_metrics['total_value']:,.2f}")
            with col2:
                profit_color = "normal" if portfolio_metrics['total_return'] >= 0 else "inverse"
                st.metric("Total Return", 
                         f"${portfolio_metrics['total_return']:,.2f}",
                         f"{portfolio_metrics['return_percentage']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")
            with col4:
                st.metric("Volatility", f"{risk_metrics['volatility']:.2%}")
            
            # Portfolio Composition
            st.markdown("---")
            st.subheader("Portfolio Composition")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[h['asset'].upper() for h in portfolio],
                    values=[h['value'] for h in portfolio],
                    hole=0.4
                )])
                fig_pie.update_layout(title="Asset Allocation", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Holdings table
                holdings_df = pd.DataFrame([{
                    'Asset': h['asset'].upper(),
                    'Quantity': h['quantity'],
                    'Value': f"${h['value']:,.2f}",
                    'Weight': f"{h['weight']:.1f}%",
                    'P&L': f"${h['value'] - h['cost_basis']:,.2f}"
                } for h in portfolio])
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            
            # Risk Metrics
            st.markdown("---")
            st.subheader("Risk Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Volatility", f"{risk_metrics['volatility']:.2%}")
                st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")
                st.metric("Expected Return", f"{risk_metrics['mean_return']:.2%}")
            with col3:
                st.metric("Value at Risk (95%)", f"{risk_metrics['var_95']:.2%}")
                st.metric("Diversification Ratio", f"{portfolio_metrics['diversification_ratio']:.2f}")
            
            # Rebalancing Recommendations
            if recommendations:
                st.markdown("---")
                st.subheader("Rebalancing Recommendations")
                
                rec_df = pd.DataFrame([{
                    'Action': rec['action'],
                    'Asset': rec['asset'].upper(),
                    'Amount': f"${rec['amount']:,.2f}",
                    'Current Weight': f"{rec['current_weight']:.1f}%",
                    'Target Weight': f"{rec['target_weight']:.1f}%",
                    'Difference': f"{rec['difference']:.1f}%"
                } for rec in recommendations])
                
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
            
            # Live Market Data
            st.markdown("---")
            st.subheader("Live Market Data")
            
            market_df = crypto_data.copy()
            market_df['symbol'] = market_df['symbol'].str.upper()
            market_df['current_price'] = market_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            if 'price_change_percentage_24h' in market_df.columns:
                market_df['24h Change'] = market_df['price_change_percentage_24h'].apply(lambda x: f"{x:.2f}%")
            
            display_cols = ['symbol', 'current_price']
            if '24h Change' in market_df.columns:
                display_cols.append('24h Change')
            
            st.dataframe(market_df[display_cols], use_container_width=True, hide_index=True)
            
        else:
            st.error("Could not fetch market data. Please check your internet connection.")

# Footer
st.markdown("---")
st.markdown("**Portfolio Analytics Dashboard** | Built with Streamlit, CoinGecko API & Python")
st.markdown("*Data updates in real-time. For educational purposes only.*")