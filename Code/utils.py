"""
Utility functions and classes for power system contract negotiation.
"""
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew, kurtosis, norm
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import matplotlib.pyplot as plt
import os


class Expando:
    """A small utility class which can have attributes set dynamically.

    This class provides a flexible container that allows dynamic attribute
    setting, useful for storing variables, constraints, and results.
    """
    pass

def build_dataframe(data: Dict[tuple, Any], input_name: str) -> pd.DataFrame:
    """Convert dictionary data to a formatted pandas DataFrame.
    
    Parameters
    ----------
    data : dict
        Dictionary with (time, scenario) tuples as keys
    input_name : str
        Name of the input variable
    
    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with scenarios as columns
    """
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

def calculate_cvar(earnings: np.ndarray, alpha: float) -> float:
    """Calculate the Conditional Value at Risk (CVaR) for given earnings.
    
    Parameters
    ----------
    earnings : ndarray
        Array of earnings values
    alpha : float
        Confidence level for CVaR calculation, between 0 and 1
    
    Returns
    -------
    float
        Calculated CVaR value
    """
    earnings = np.array(earnings)
    earnings_sorted = np.sort(earnings)
    var_threshold_lower = np.percentile(earnings_sorted, (1-alpha)*100)
    tail_losses = earnings_sorted[earnings_sorted <= var_threshold_lower]
    cvar_loss = np.mean(tail_losses)
    
    return cvar_loss

def simulate_price_scenarios(
    price_data: np.ndarray,
    KG: float,
    num_scenarios: int,
    num_time_periods: int,
    distribution: str = 'normal'
) -> np.ndarray:
    """Simulate price scenarios using different distributions.
    
    Parameters
    ----------
    price_data : ndarray
        Original price data array
    KG : float
        Price bias parameter
    num_scenarios : int
        Number of scenarios to simulate
    num_time_periods : int
        Number of time periods to simulate
    distribution : {'normal', 'lognormal', 'empirical'}, optional
        Type of distribution to use, by default 'normal'
    
    Returns
    -------
    ndarray
        Simulated price scenarios with shape (num_time_periods, num_scenarios)
    
    Raises
    ------
    ValueError
        If distribution type is not supported
    """
    mean_price_true = np.mean(price_data)
    std_price_true = np.std(price_data)

    mean_price = mean_price_true + KG*mean_price_true if KG != 0 else mean_price_true
        
    if distribution == 'normal':
        simulated_prices = norm.rvs(loc=mean_price, scale=std_price_true, 
                                  size=(num_time_periods, num_scenarios))
    
    elif distribution == 'lognormal':
        mu = np.log(mean_price**2 / np.sqrt(std_price_true**2 + mean_price**2))
        sigma = np.sqrt(np.log(1 + (std_price_true**2 / mean_price**2)))
        simulated_prices = np.random.lognormal(mu, sigma, 
                                             size=(num_time_periods, num_scenarios))
    
    elif distribution == 'empirical':
        kernel = gaussian_kde(price_data.flatten())
        simulated_prices = kernel.resample(size=(num_time_periods, num_scenarios))
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution}. "
                        f"Supported types are: 'normal', 'lognormal', 'empirical'")
    
    return simulated_prices

def analyze_price_distribution(
    price_data: np.ndarray,
    simulated_prices_dict: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """Analyze and compare original and simulated price distributions.
    
    Parameters
    ----------
    price_data : ndarray
        Original price data array
    simulated_prices_dict : dict
        Dictionary of simulated prices for each distribution type
    
    Returns
    -------
    dict
        Statistical metrics for each distribution including mean, standard deviation,
        skewness, kurtosis, and percentiles
    """
    results = {}
    
    # Original data statistics
    results['original'] = {
        'mean': np.mean(price_data),
        'std': np.std(price_data),
        'skewness': skew(price_data.flatten()),
        'kurtosis': kurtosis(price_data.flatten()),
        'percentiles': np.percentile(price_data, [5, 25, 50, 75, 95])
    }
    
    # Simulated data statistics
    for dist_name, sim_prices in simulated_prices_dict.items():
        results[dist_name] = {
            'mean': np.mean(sim_prices),
            'std': np.std(sim_prices),
            'skewness': skew(sim_prices.flatten()),
            'kurtosis': kurtosis(sim_prices.flatten()),
            'percentiles': np.percentile(sim_prices, [5, 25, 50, 75, 95])
        }
    
    return results

def optimize_nash_product(
    price_data: np.ndarray,
    A_G6: float,
    A_L2: float,
    net_earnings_no_contract_G6: np.ndarray,
    net_earnings_no_contract_L2: np.ndarray,
    Zeta_G6: float,
    Zeta_L2: float,
    strikeprice_min: float,
    strikeprice_max: float,
    contract_amount_min: float,
    contract_amount_max: float,
    alpha: float = 0.95,
    plot: bool = False,
    plot_dir: str = None,
    filename: str = None
) -> tuple[float, float, float]:
    """
    Optimize the Nash product (Ug-Zg)*(Ul-Zl) using SciPy's optimization.
    
    Parameters
    ----------
    // ...existing parameter docs...
    plot : bool, optional
        If True, creates a 2D contour plot of the Nash product, by default False
    plot_dir : str, optional
        Directory to save plots to, by default None
    filename : str, optional
        If provided, saves the plot to this filename, by default None
    
    Returns
    -------
    tuple[float, float, float]
        Optimal strike price, optimal contract amount, and optimal Nash product value
    """
    def calculate_utilities(S: float, M: float) -> tuple[float, float, float]:
        """Calculate utilities and Nash product for given strike price and contract amount."""
        # Calculate revenues for G6 with contract
        SMG6 = (S - price_data) * M
        Scen_revenue_G6 = np.sum(SMG6, axis=0) + net_earnings_no_contract_G6
        
        # Calculate revenues for L2 with contract
        SML2 = (price_data - S) * M
        Scen_revenue_L2 = np.sum(SML2, axis=0) + net_earnings_no_contract_L2
        
        # Calculate CVaR for both parties
        CVaRG6 = calculate_cvar(Scen_revenue_G6, alpha)
        CVaRL2 = calculate_cvar(Scen_revenue_L2, alpha)
        
        # Calculate utilities
        UG6 = (1 - A_G6) * np.mean(Scen_revenue_G6) + A_G6 * CVaRG6
        UL2 = (1 - A_L2) * np.mean(Scen_revenue_L2) + A_L2 * CVaRL2
        
        # Calculate Nash product
        nash_prod = (UG6 - Zeta_G6) * (UL2 - Zeta_L2)
        
        return UG6, UL2, nash_prod

    def objective(x):
        S, M = x
        _, _, nash_prod = calculate_utilities(S, M)
        return -nash_prod  # Return negative for minimization
    
    # Define bounds for variables [S, M]
    bounds = [(strikeprice_min, strikeprice_max), (contract_amount_min, contract_amount_max)]
    
    # Initial guess: midpoint of bounds
    x0 = [(strikeprice_min + strikeprice_max)/2, (contract_amount_min + contract_amount_max)/2]
    

    #Set up correct bounds! 

    
    # Optimize using SLSQP which handles bounds well
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='Newton-CG',
        options={'ftol': 1e-8, 'maxiter': 1000,'disp': True}  # Add more debug info},
    )
    
    if result.success:
        S_opt, M_opt = result.x
        nash_value = -result.fun  # Convert back to positive Nash product
        
        if plot:
            # Create a grid of strike prices and contract amounts
            strike_prices = np.linspace(strikeprice_min, strikeprice_max, 50)
            contract_amounts = np.linspace(contract_amount_min, contract_amount_max, 50)
            S_grid, M_grid = np.meshgrid(strike_prices, contract_amounts)
            nash_values = np.zeros_like(S_grid)
            
            # Calculate Nash product for each point in the grid
            for i in range(len(contract_amounts)):
                for j in range(len(strike_prices)):
                    _, _, nash_prod = calculate_utilities(strike_prices[j], contract_amounts[i])
                    nash_values[i, j] = nash_prod
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            contour = plt.contour(S_grid, M_grid, nash_values, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Nash Product Value')
            
            # Plot optimal point
            plt.plot(S_opt, M_opt, 'ro', markersize=10, label='Optimal Point')
            
            plt.xlabel('Strike Price')
            plt.ylabel('Contract Amount')
            plt.title('Nash Product Optimization Landscape')
            plt.legend()
            
            if filename and plot_dir:
                filepath = os.path.join(plot_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        return S_opt, M_opt, nash_value
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")