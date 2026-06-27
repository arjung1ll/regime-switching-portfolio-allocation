import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_loader import load_financial_data
from hmm_model import train_hmm_model
from allocation import simulate_portfolio

# Set a professional plotting style
sns.set_theme(style="whitegrid")

def calculate_cumulative_performance(portfolio_data):
    """
    Converts daily returns into cumulative growth (equity curves).
    Assumes a starting value of $1.00.
    """
    print("Calculating cumulative performance metrics...")
    portfolio_data['Strategy_Equity'] = (1 + portfolio_data['Strategy_Return']).cumprod()
    portfolio_data['Benchmark_Equity'] = (1 + portfolio_data['SPY']).cumprod()
    return portfolio_data

# ---> NEW METRICS FUNCTION ADDED HERE <---
def print_risk_metrics(portfolio_data):
    """
    Calculates and prints the Annualized Sharpe Ratio and Maximum Drawdown.
    """
    print("\n=== Risk & Performance Metrics ===")
    
    # 1. Calculate Strategy Metrics
    strat_returns = portfolio_data['Strategy_Return']
    strat_annual_return = strat_returns.mean() * 252
    strat_annual_vol = strat_returns.std() * np.sqrt(252)
    # Assuming risk-free rate is 0% for simplicity
    strat_sharpe = strat_annual_return / strat_annual_vol 
    
    # Calculate Strategy Max Drawdown
    strat_cum_max = portfolio_data['Strategy_Equity'].cummax()
    strat_drawdown = (portfolio_data['Strategy_Equity'] - strat_cum_max) / strat_cum_max
    strat_max_dd = strat_drawdown.min()
    
    # 2. Calculate Benchmark (SPY) Metrics
    bench_returns = portfolio_data['SPY']
    bench_annual_return = bench_returns.mean() * 252
    bench_annual_vol = bench_returns.std() * np.sqrt(252)
    bench_sharpe = bench_annual_return / bench_annual_vol
    
    # Calculate Benchmark Max Drawdown
    bench_cum_max = portfolio_data['Benchmark_Equity'].cummax()
    bench_drawdown = (portfolio_data['Benchmark_Equity'] - bench_cum_max) / bench_cum_max
    bench_max_dd = bench_drawdown.min()
    
    # 3. Print Results
    print(f"HMM Strategy -> Sharpe Ratio: {strat_sharpe:.2f} | Max Drawdown: {strat_max_dd*100:.2f}%")
    print(f"Benchmark    -> Sharpe Ratio: {bench_sharpe:.2f} | Max Drawdown: {bench_max_dd*100:.2f}%\n")

def plot_results(portfolio_data):
    """
    Plots the equity curves and shades the background based on the HMM regime.
    """
    print("Generating performance chart...")
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_data.index, portfolio_data['Strategy_Equity'], label='HMM Dynamic Strategy', color='blue', linewidth=1.5)
    plt.plot(portfolio_data.index, portfolio_data['Benchmark_Equity'], label='Benchmark (100% SPY)', color='gray', linestyle='--', alpha=0.7)
    
    max_val = portfolio_data[['Strategy_Equity', 'Benchmark_Equity']].max().max()
    plt.fill_between(portfolio_data.index, 0, max_val + 1, where=(portfolio_data['Regime'] == 1), color='red', alpha=0.15, label='Detected Volatile Regime')
    
    plt.title('Multivariate Regime-Switching Asset Allocation', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return ($1 Growth)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.ylim(0.5, max_val + 0.5) 
    
    plt.tight_layout()
    plt.savefig('hmm_performance_chart.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("=== INITIALIZING QUANTITATIVE PIPELINE ===")
    
    prices, rets = load_financial_data()
    model, states = train_hmm_model(rets)
    portfolio = simulate_portfolio(rets, states)
    portfolio_eval = calculate_cumulative_performance(portfolio)
    
    # ---> CALLING THE NEW FUNCTION HERE <---
    print_risk_metrics(portfolio_eval)
    
    plot_results(portfolio_eval)
    print("=== PIPELINE COMPLETE ===")