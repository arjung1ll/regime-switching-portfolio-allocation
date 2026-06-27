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
    
    # (1 + daily_return) cumulative product gives the compounding growth
    portfolio_data['Strategy_Equity'] = (1 + portfolio_data['Strategy_Return']).cumprod()
    
    # We use a 100% SPY buy-and-hold as our benchmark to beat
    portfolio_data['Benchmark_Equity'] = (1 + portfolio_data['SPY']).cumprod()
    
    return portfolio_data

def plot_results(portfolio_data):
    """
    Plots the equity curves and shades the background based on the HMM regime.
    """
    print("Generating performance chart...")
    
    plt.figure(figsize=(14, 7))
    
    # Plot the Strategy and the Benchmark
    plt.plot(portfolio_data.index, portfolio_data['Strategy_Equity'], 
             label='HMM Dynamic Strategy', color='blue', linewidth=1.5)
    
    plt.plot(portfolio_data.index, portfolio_data['Benchmark_Equity'], 
             label='Benchmark (100% SPY)', color='gray', linestyle='--', alpha=0.7)
    
    # Shade the background red when the model is in Regime 1 (Volatile/Bear)
    # Note: We use max() to ensure the shading covers the whole y-axis height
    max_val = portfolio_data[['Strategy_Equity', 'Benchmark_Equity']].max().max()
    
    plt.fill_between(portfolio_data.index, 0, max_val + 1, 
                     where=(portfolio_data['Regime'] == 1), 
                     color='red', alpha=0.15, label='Detected Volatile Regime')
    
    # Formatting the chart for a professional tear-sheet look
    plt.title('Multivariate Regime-Switching Asset Allocation', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return ($1 Growth)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.ylim(0.5, max_val + 0.5) 
    
    # Save the plot to the folder so it can be uploaded to GitHub
    plt.tight_layout()
    plt.savefig('hmm_performance_chart.png', dpi=300)
    
    # Display the chart locally
    plt.show()

if __name__ == "__main__":
    print("=== INITIALIZING QUANTITATIVE PIPELINE ===")
    
    # 1. Extract
    prices, rets = load_financial_data()
    
    # 2. Model
    model, states = train_hmm_model(rets)
    
    # 3. Simulate
    portfolio = simulate_portfolio(rets, states)
    
    # 4. Evaluate
    portfolio_eval = calculate_cumulative_performance(portfolio)
    
    # 5. Visualize
    plot_results(portfolio_eval)
    
    print("=== PIPELINE COMPLETE ===")