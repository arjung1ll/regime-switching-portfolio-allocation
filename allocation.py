import pandas as pd

def simulate_portfolio(returns_data, regimes):
    """
    Simulates a regime-switching portfolio.
    Regime 0 (Assumed Bull/Calm): 80% Equities, 20% Bonds
    Regime 1 (Assumed Bear/Volatile): 30% Equities, 50% Bonds, 20% Gold
    """
    print("Simulating Portfolio Allocations...")
    
    # Create a copy of the returns dataframe
    strategy_data = returns_data.copy()
    
    # Add our predicted regimes to the data. 
    # IMPORTANT: We shift the regime forward by 1 day! 
    # We only know today's regime at the end of today, so we apply it to tomorrow's trading.
    strategy_data['Regime'] = regimes
    strategy_data['Regime'] = strategy_data['Regime'].shift(1)
    
    # Drop the first day since it has no yesterday to shift from
    strategy_data = strategy_data.dropna()
    
    # Initialize an empty column for our strategy's daily returns
    strategy_data['Strategy_Return'] = 0.0
    
    # Loop through and apply weights based on the regime
    for index, row in strategy_data.iterrows():
        if row['Regime'] == 0:
            # 80% SPY, 20% TLT, 0% GLD
            daily_ret = (0.80 * row['SPY']) + (0.20 * row['TLT']) + (0.00 * row['GLD'])
        else:
            # 30% SPY, 50% TLT, 20% GLD
            daily_ret = (0.30 * row['SPY']) + (0.50 * row['TLT']) + (0.20 * row['GLD'])
            
        strategy_data.at[index, 'Strategy_Return'] = daily_ret
        
    print("Simulation Complete.")
    return strategy_data

if __name__ == "__main__":
    from data_loader import load_financial_data
    from hmm_model import train_hmm_model
    
    prices, rets = load_financial_data()
    model, states = train_hmm_model(rets)
    portfolio = simulate_portfolio(rets, states)
    
    print("\nFirst 5 days of Strategy Returns:")
    print(portfolio[['SPY', 'Regime', 'Strategy_Return']].head())