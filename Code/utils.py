"""
Utility functions and classes for power system contract negotiation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, skew, kurtosis, norm
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint


import matplotlib.pyplot as plt
import os


class Expando: 
    """A small utility class which can have attributes set dynamically.

    This class provides a flexible container that allows dynamic attribute
    setting, useful for storing variables, constraints, and results.
    """
    pass

class PriceProductionProvider(ABC):
    """Return the arrays the bargaining model needs, no matter the backend."""

    @abstractmethod
    def price_matrix(self, node: str) -> pd.DataFrame: ...
    @abstractmethod
    def production_matrix(self) -> pd.DataFrame: ...
    @property
    @abstractmethod
    def probability(self) -> float: ...


class OPFProvider(PriceProductionProvider):
    def __init__(self, opf_results,prob):
        self._r = opf_results            # OptimalPowerFlow.results
        self._rPROB = prob

    def price_matrix(self, node: str) -> np.ndarray:
        return build_dataframe(self._r.price[node], 'price').values

    def production_matrix(self) -> np.ndarray:
        return build_dataframe(
            self._r.generator_production, 'gen_prod'
        ).values

    @property
    def probability(self) -> float:
        return self._rPROB

class ForecastProvider(PriceProductionProvider):
    def __init__(self, price_df: pd.DataFrame, prod_df: pd.DataFrame, CR_df: pd.DataFrame, load_df: pd.DataFrame, prob: float):
        self._price = price_df          # shape (T, S)
        self._prod  = prod_df           # shape (T, S)
        self._CR    = CR_df                # shape (T, S)
        self._load   = load_df
        self._prob  = prob              # usually 1/S

    def price_matrix(self) -> np.ndarray:
        return self._price
    def production_matrix(self) -> np.ndarray:
        return self._prod
    def capture_rate_matrix(self) -> np.ndarray:
        return self._CR
    def load_matrix(self) ->  pd.DataFrame:
        return self._load

    @property
    def probability(self) -> float:
        return self._prob

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

def calculate_cvar_left(earnings: np.ndarray, alpha: float) -> float:
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
    left_tail = earnings_sorted[earnings_sorted <= var_threshold_lower]
    cvar = np.mean(left_tail)
    return cvar

def calculate_cvar_right(earnings: np.ndarray, alpha: float) -> float:
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
    var_threshold_lower = np.percentile(earnings_sorted, alpha*100)
    right_tail = earnings_sorted[earnings_sorted >= var_threshold_lower]
    cvar = np.mean(right_tail)
    return cvar

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
    old_obj_func:bool,
    hours_in_year: np.ndarray,
    price_G: np.ndarray,
    price_L: np.ndarray,
    A_G: float,
    A_L: float,
    net_earnings_no_contract_G: np.ndarray,
    net_earnings_no_contract_L: np.ndarray,
    Zeta_G: float,
    Zeta_L: float,
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

        
        # Calculate revenues for G with contract
        rev_contract = M  * (S  - price_G)
        rev_contract_total = rev_contract.sum(axis=0)
        Expected = net_earnings_no_contract_G 
        earnings = Expected + rev_contract_total
        CVaR_G = calculate_cvar_left(earnings, alpha)

        UG = (1-A_G)*earnings.mean() + A_G * CVaR_G
        
        rev_contract = M  * ( price_L - S )
        rev_contract_total = rev_contract.sum(axis=0)
        Expected = net_earnings_no_contract_L
        earnings = Expected + rev_contract_total
        CVaR_L = calculate_cvar_left(earnings, alpha)
        
       
        UL = (1-A_L)*earnings.mean() + A_L * CVaR_L
        
        # Calculate CVaR for both parties
        
        # Calculate utilities
        """
        if old_obj_func == True:
            if A_G == 0 and A_L != 0:
                UG = np.mean(earnings_G)
                UL = np.mean(earnings_L) + A_L * CVaRL
            elif A_G != 0 and A_L == 0:
                UG = np.mean(earnings_G) + A_G * CVaRG
                UL = np.mean(earnings_L)
            elif A_G == 0 and A_L == 0:
                UG = np.mean(earnings_G)
                UL = np.mean(earnings_L)
            else:
                UG = np.mean(earnings_G) + A_G * CVaRG
                UL = np.mean(earnings_L) + A_L * CVaRL
        else:
            if  A_G == 0 and A_L != 0:
                UG = np.mean(earnings_G)
                UL = (1 - A_L) * np.mean(earnings_G) + A_L * CVaRL
            elif  A_G != 0 and A_L == 0:
                UG = (1 - A_G) * np.mean(earnings_L) + A_G * CVaRG
                UL = np.mean(earnings_G)
            elif A_G == 0 and A_L == 0:
                UG = np.mean(earnings_G)
                UL = np.mean(earnings_L)
            else:
                UG = (1 - A_G) * np.mean(earnings_G) + A_G * CVaRG
                UL = (1 - A_L) * np.mean(earnings_L) + A_L * CVaRL
        """
        # Calculate Nash product ( Negative for minimization)
        nash_prod = ((UG - Zeta_G) * (UL - Zeta_L))
        
        return UG,UL, nash_prod

    def objective(x):
        S, M = x
        UG, UL, nash_prod = calculate_utilities(S, M)
        return -nash_prod  # Return negative for minimization

    # Constraint: UG - Zeta_G >= 0
    def constraint_UG_minus_ZetaG(x):
        S, M = x
        UG, _, _ = calculate_utilities(S, M)
        return (UG - Zeta_G)

    # else round to 0.001
    # Constraint: UL - Zeta_L >= 0
    def constraint_UL_minus_ZetaL(x):
        S, M = x
        _, UL, _ = calculate_utilities(S, M)
        return (UL - Zeta_L)

    nonlinear_constraint_UG = NonlinearConstraint(constraint_UG_minus_ZetaG, 0, np.inf)
    nonlinear_constraint_UL = NonlinearConstraint(constraint_UL_minus_ZetaL, 0, np.inf)

    # Define bounds for variables [S, M]
    bounds = [(strikeprice_min* hours_in_year*1e-3, strikeprice_max* hours_in_year*1e-3 ) , (contract_amount_min, contract_amount_max  * hours_in_year* 1e-3)]
    x0 = [((strikeprice_min + strikeprice_max) * hours_in_year*1e-3 )/2, (contract_amount_min + contract_amount_max * hours_in_year* 1e-3)/2]

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=[nonlinear_constraint_UG, nonlinear_constraint_UL],
        method='trust-constr',
        hess=None,
        options={ 'maxiter': 5000,'disp': True},
        tol=1e-9
    )
    
    if result.success:
        S_opt, M_opt = result.x
        nash_value = -result.fun  # Convert back to positive Nash product
        
        if plot == True:
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