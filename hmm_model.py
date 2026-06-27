import numpy as np
import pandas as pd
from hmmlearn import hmm
from data_loader import load_financial_data

def train_hmm_model(returns_data, n_regimes=2):
    """
    Trains a Multivariate Gaussian HMM using SPY, VIX, and 10-Year Yields.
    """
    print(f"Training {n_regimes}-State Multivariate Hidden Markov Model...")
    
    # MULTIVARIATE UPGRADE: We select our three features instead of just one.
    # We multiply by 100 to scale the decimals and prevent convergence warnings.
    features = returns_data[['SPY', '^VIX', '^TNX']] * 100
    
    # Convert the pandas dataframe into a 2D numpy matrix 
    # (It now has 3 columns instead of 1)
    feature_matrix = features.values
    
    model = hmm.GaussianHMM(
        n_components=n_regimes, 
        covariance_type="full", 
        n_iter=1000, 
        random_state=42
    )
    
    # Fit the model using all three macro signals at once
    model.fit(feature_matrix)
    hidden_states = model.predict(feature_matrix)
    
    print("Multivariate HMM Training Complete.")
    return model, hidden_states

if __name__ == "__main__":
    prices, rets = load_financial_data()
    model, states = train_hmm_model(rets)
    
    test_df = pd.DataFrame({
        'SPY_Return': rets['SPY'],
        'VIX_Change': rets['^VIX'],
        'Yield_Change': rets['^TNX'],
        'Regime': states
    })
    
    print("\nFirst 10 days of Multivariate Regime Classifications:")
    print(test_df.head(10))