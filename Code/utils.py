"""
Utility functions and classes for power system contract negotiation.
"""

class Expando:
    """
    A small utility class which can have attributes set dynamically.
    Used for flexible storage of variables, constraints, and results.
    """
    pass

def build_dataframe(data, input_name):
    """
    Convert dictionary data to a formatted pandas DataFrame.
    
    Args:
        data (dict): Dictionary with (time, scenario) tuples as keys
        input_name (str): Name of the input variable
    
    Returns:
        pd.DataFrame: Formatted DataFrame with scenarios as columns
    """
    import pandas as pd
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=[str(input_name)])
    
    # Reset index and split into Time and Scenario
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Time', 'Scenario'])
    df.reset_index(inplace=True)
    
    # Rename columns
    df.columns = ['Time', 'Scenario', str(input_name)]
    
    # Pivot DataFrame
    df_pivot = df.pivot(index='Time', columns='Scenario', values=str(input_name))
    
    # Rename columns to scen_1, scen_2, etc.
    df_pivot.columns = [f"scen_{i+1}" for i in range(len(df_pivot.columns))]
    
    # Reset index
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.drop(columns=['Time'])
    
    return df_pivot

def calculate_cvar(earnings, alpha):
    """
    Calculate the Conditional Value at Risk (CVaR) for given earnings.
    
    Args:
        earnings (numpy.array): Array of earnings values
        alpha (float): Confidence level for CVaR calculation
    
    Returns:
        float: Calculated CVaR value
    """
    import numpy as np
    
    earnings = np.array(earnings)
    earnings_sorted = np.sort(earnings)
    var_threshold_lower = np.percentile(earnings_sorted, (1-alpha)*100)
    tail_losses = earnings_sorted[earnings_sorted <= var_threshold_lower]
    cvar_loss = np.mean(tail_losses)
    
    return cvar_loss

import numpy as np
from scipy import stats

def simulate_price_scenarios(price_data,KG,num_scenarios,num_time_periods,distribution='normal'):
    """
    Simulate price scenarios using different distributions based on input data characteristics.
    
    Args:
        price_data (np.ndarray): Original price data array
        num_scenarios (int): Number of scenarios to simulate
        distribution (str): Type of distribution ('normal', 'lognormal', or 'empirical')
    
    Returns:
        np.ndarray: Simulated price scenarios
    """
    # Get data shape and parameters
    mean_price_true = np.mean(price_data)
    std_price_true = np.std(price_data)

    if KG !=0:
        mean_price = mean_price_true + KG*mean_price_true
    else:
        mean_price = mean_price_true
        
    
    if (distribution == 'normal'):
        # Normal distribution simulation
        simulated_prices = np.random.normal(mean_price, std_price_true, 
                                          size=(num_time_periods, num_scenarios))
    
    elif (distribution == 'lognormal'):
        # Log-normal distribution parameters
        mu = np.log(mean_price**2 / np.sqrt(std_price_true**2 + mean_price**2))
        sigma = np.sqrt(np.log(1 + (std_price_true**2 / mean_price**2)))
        simulated_prices = np.random.lognormal(mu, sigma, 
                                             size=(num_time_periods, num_scenarios))
    
    elif (distribution == 'empirical'):
        # Empirical distribution using kernel density estimation
        kernel = stats.gaussian_kde(price_data.flatten())
        simulated_prices = kernel.resample(size=(num_time_periods, num_scenarios))
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")
    
    return simulated_prices

def analyze_price_distribution(price_data, simulated_prices_dict):
    """
    Analyze and compare original and simulated price distributions.
    
    Args:
        price_data (np.ndarray): Original price data array
        simulated_prices_dict (dict): Dictionary of simulated prices for each distribution
    
    Returns:
        dict: Statistical metrics for each distribution
    """
    results = {}
    
    # Original data statistics
    results['original'] = {
        'mean': np.mean(price_data),
        'std': np.std(price_data),
        'skewness': stats.skew(price_data.flatten()),
        'kurtosis': stats.kurtosis(price_data.flatten()),
        'percentiles': np.percentile(price_data, [5, 25, 50, 75, 95])
    }
    
    # Simulated data statistics
    for dist_name, sim_prices in simulated_prices_dict.items():
        results[dist_name] = {
            'mean': np.mean(sim_prices),
            'std': np.std(sim_prices),
            'skewness': stats.skew(sim_prices.flatten()),
            'kurtosis': stats.kurtosis(sim_prices.flatten()),
            'percentiles': np.percentile(sim_prices, [5, 25, 50, 75, 95])
        }
    
    return results