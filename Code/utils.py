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

class ForecastProvider(PriceProductionProvider):
    def __init__(self, price_df: pd.DataFrame, prod_df: pd.DataFrame, CR_df: pd.DataFrame, load_df: pd.DataFrame, load_CR : pd.DataFrame, prob: np.ndarray):
        self._price = price_df          # shape (T, S)
        self._prod  = prod_df           # shape (T, S)
        self._CR    = CR_df             # shape (T, S)
        self._load  = load_df           # shape (T, S)
        self._load_CR = load_CR         # shape (T, S)
        self._prob  = prob              # S 

    def price_matrix(self) -> np.ndarray:
        return self._price
    def production_matrix(self) -> np.ndarray:
        return self._prod
    def capture_rate_matrix(self) -> np.ndarray:
        return self._CR
    def load_matrix(self) ->  pd.DataFrame:
        return self._load
    def load_capture_rate_matrix(self) -> np.ndarray:
        return self._load_CR

    @property
    def probability(self) -> np.ndarray:
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


def weighted_expected_value(arr, scenario_probs):
    """
    Calculate the probability-weighted expected value for a T x S array.
    arr: array-like, shape (T, S)
    scenario_probs: array-like, shape (S,)
    Returns: float (expected value)
    """
    arr = np.array(arr)
    scenario_probs = np.array(scenario_probs)
    scenario_probs = scenario_probs / scenario_probs.sum()  # Normalize
    
    if arr.ndim == 1:
        # arr is shape (S,)
        expected_value = np.sum(arr * scenario_probs)
    else:
        # arr is shape (T, S)
        scenario_totals = arr.sum(axis=0)  # Sum over time for each scenario
        expected_value = np.sum(scenario_totals * scenario_probs) / arr.shape[0]  # Average over time
    return expected_value

def get_worst_case_mask(z, probabilities, alpha=0.95):
    """
    Identify scenarios in the worst (1-alpha)% tail of z.
    
    Parameters:
    -----------
    z : array-like
        Values to determine worst cases (e.g., z = x * y)
    probabilities : array-like
        Probability of each scenario
    alpha : float
        Confidence level (e.g., 0.95 means worst 5%)
    
    Returns:
    --------
    mask : boolean array
        True for scenarios in the worst tail
    """
    z = np.asarray(z, dtype=float).ravel()
    probs = np.asarray(probabilities, dtype=float).ravel()
    probs = probs / probs.sum()  # Normalize
    
    tail_mass = 1.0 - alpha  # Probability mass in the tail
    
    # Sort by z values (worst to best)
    order = np.argsort(z)
    probs_sorted = probs[order]
    
    # Cumulative probability
    cdf = np.cumsum(probs_sorted)
    
    # Find where cumulative probability reaches tail_mass
    boundary_idx = np.searchsorted(cdf, tail_mass, side='right')
    
    # Create mask for tail scenarios
    mask = np.zeros(len(z), dtype=bool)
    mask[order[:boundary_idx]] = True
    
    return mask


def conditional_cvar(x, z, probabilities, alpha=0.95):
    """
    Calculate CVaR of x conditional on z being in its worst tail.
    
    Parameters:
    -----------
    x : array-like
        Variable to calculate CVaR for
    z : array-like
        Variable that determines worst-case scenarios (e.g., z = x * y)
    probabilities : array-like
        Probability of each scenario
    alpha : float
        Confidence level
    
    Returns:
    --------
    float : Conditional CVaR of x given z is in worst tail
    """
    x = np.asarray(x, dtype=float).ravel()
    probs = np.asarray(probabilities, dtype=float).ravel()
    probs = probs / probs.sum()
    
    # Get worst-case scenarios based on z
    tail_mask = get_worst_case_mask(z, probs, alpha)
    
    # Extract x values and probabilities for tail scenarios
    x_tail = x[tail_mask]
    probs_tail = probs[tail_mask]
    
    # Calculate probability-weighted CVaR of x in the tail
    cvar_x = np.sum(x_tail * probs_tail) / np.sum(probs_tail)
    
    return cvar_x



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


def calculate_cvar_left_simple(earnings: np.ndarray, alpha: float) -> float:
    """
    Left-tail CVaR using simple percentile method (assuming uniform probabilities).
    Returns the conditional mean of the worst (1-alpha) tail.
    alpha is the confidence level (e.g. 0.95 -> worst 5% left tail).
    """
    earnings = np.array(earnings)
    earnings_sorted = np.sort(earnings)  # worst -> best
    tail_mass = 1.0 - alpha
    var_threshold = np.percentile(earnings_sorted, tail_mass * 100)
    left_tail = earnings_sorted[earnings_sorted <= var_threshold]
    cvar = np.mean(left_tail)
    return cvar





def calculate_cvar_left(earnings: np.ndarray, probabilities: pd.DataFrame, alpha: float) -> float:
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

    earnings      = np.asarray(earnings,      dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    #probabilities = probabilities / probabilities.sum()

    # Sort from worst to best outcome
    sorted_idx    = np.argsort(earnings)
    earnings_sorted = earnings[sorted_idx]
    probs_sorted    = probabilities[sorted_idx]

    # Accumulate probabilities until we reach the left-tail mass (1-α)
    tail_prob     = 1.0 - alpha
    cumulative    = np.cumsum(probs_sorted)

    # First index that pushes cumulative mass past the tail
    var_idx       = np.searchsorted(cumulative, tail_prob, side="left")

    # Probability already collected strictly before var_idx
    prob_before   = cumulative[var_idx-1] if var_idx > 0 else 0.0
    needed_from_i = tail_prob - prob_before          # fraction (0 ≤ … ≤ p_i)

    # Weighted sum of earnings in the tail (include fractional boundary weight)
    weighted_sum  = np.dot(earnings_sorted[:var_idx], probs_sorted[:var_idx])
    weighted_sum += earnings_sorted[var_idx] * needed_from_i

    # Divide by exact tail mass to get the conditional expectation
    cvar = weighted_sum / tail_prob
   
   
    
    """ 
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(earnings, bins=30)
    
    # Add vertical line for VaR (threshold)
    plt.axvline(x=earnings_sorted[var_idx], color='red', linestyle='--', 
                label=f'VaR ({(1-alpha)*100:.0f}% percentile): {cvar:.2f}')

    plt.axvline(x=cvar, color='darkred', linestyle='-', 
                label=f'CVaR (Left Tail Mean): {cvar:.2f}')
    plt.show()
    """
    

    return cvar

def _left_tail_mask(arr, probabilities, alpha):

    """
    Create boolean mask for left tail based on probability mass
    
    Args:
        arr: array of values (e.g., earnings)
        probabilities: array of probabilities for each value
        alpha: tail probability (e.g., 0.05 for worst 5%)
    
    Returns:
        boolean mask indicating which scenarios are in the left tail
    """
    tail_prob = 1.0 - alpha
    values        = np.asarray(arr,        dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()

    order         = np.argsort(values)             # worst → best
    probs_sorted  = probabilities[order]
    cdf           = np.cumsum(probs_sorted)

    boundary_idx  = np.searchsorted(cdf, tail_prob, side="left")

    mask = np.zeros_like(values, dtype=bool)
    mask[order[:boundary_idx + 1]] = True

    return order, boundary_idx   # boolean mask indicating left tail scenarios

def _left_tail_weighted_sum(probabilities, values,
                             order, boundary_idx, tail_prob):
    """
    Σ_i  p_i * w_i  over the left tail, taking only the fraction
    of the boundary scenario that is actually in the tail.
    
    Parameters:
    -----------
    probabilities : array-like
        Original probabilities (not masked yet)
    values : array-like
        Original values (not masked yet)
    mask : boolean array
        Mask identifying full scenarios in tail (before boundary)
    boundary_idx : int
        Index in the sorted order where boundary occurs
    order : array
        Sorted indices (worst to best)
    cdf : array
        Cumulative distribution in sorted order
    tail_prob : float
        Target tail probability mass (e.g., 0.05 for 95% confidence)
    
    Returns:
    --------
    float : Probability-weighted sum over the tail
    """
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()

    sorted_values = values[order]
    sorted_probs  = probabilities[order]
    
    # Weighted sum of earnings in the tail (include fractional boundary weight)
    weighted_sum  = np.dot(sorted_values[:boundary_idx], sorted_probs[:boundary_idx])
    needed_from_i = ((1-tail_prob) - sum(sorted_probs[:boundary_idx]))
    weighted_sum += sorted_values[boundary_idx] * needed_from_i
    cvar = weighted_sum / (sum(sorted_probs[:boundary_idx]) + needed_from_i)

    return cvar

    """

    # Calculate fractional weight needed from boundary scenario
    prob_before = cdf[boundary_idx - 1] if boundary_idx > 0 else 0.0
    need = max(0.0, tail_prob - prob_before)  # Fractional weight for boundary
    
    # Sum full scenarios before boundary
    total = np.dot(probs_sorted[:boundary_idx], values_sorted[:boundary_idx])
    
    # Add fractional contribution from boundary scenario
    if boundary_idx < len(values_sorted) and need > 0:
        total += values_sorted[boundary_idx] * need
    
    return total
    """

def _right_tail_mask(arr, alpha):
    var_threshold_lower = np.percentile(arr, (1-alpha)*100)
    right_tail = arr >= var_threshold_lower

    return right_tail               # boolean mask

def _calculate_S_star_PAP_G(x,gamma,A,alpha, production,price,capture_rate,PROB):
    """
    Calculate the optimal strike price S* for the Producer-side in PAP.
    """
    S =x[0]
    pi_G = ((1-gamma) * production * price * capture_rate + 
                    gamma * production * S).sum(axis=0)

    ord_G, bidx_G  = _left_tail_mask(pi_G,PROB, alpha)

    d_piG = (production * (S - price * capture_rate)).sum(axis=0)

    expected_G = (PROB * d_piG).sum()
    tail_G = _left_tail_weighted_sum(PROB,d_piG,ord_G, bidx_G, alpha)
    # Calculate S_star
    S_star = (1-A) * expected_G + A * tail_G

    # Risk adjustment

    return S_star

def _calculate_S_star_PAP_L(x,gamma,A,alpha, production,price,capture_rate,load_CR, load_scenarios, PROB):
    """
    Calculate the optimal strike price S* for the Producer-side in PAP.
    """
    S = x[0]
    pi_L = (-price * load_CR * load_scenarios).sum(axis=0) + (gamma * production * (price * capture_rate - S)).sum(axis=0)

    ord_L, bidx_L  = _left_tail_mask(pi_L,PROB, alpha)
    
    #Derivative function 
    d_piL = (production*capture_rate*price-production*S).sum(axis=0)
    expected_L = (PROB * d_piL).sum()

    tail_L = _left_tail_weighted_sum(PROB,d_piL,ord_L, bidx_L, alpha)
    # Calculate S_star
    S_star = (1 - A) * expected_L + A * tail_L

    return S_star

def _calculate_S_star_BL_G(x, M, A, alpha, production, price, capture_rate, PROB, discount_rate=None, n_time=None):
    """
    Calculate the optimal strike price S* for the Producer-side in PAP, with optional discounting.
    """
    S = x[0]
    # Assume production, price, capture_rate are shape (T, S)
    if discount_rate is not None and n_time is not None:
        # Create discount factors for each time period
        discount_factors = 1 / (1 + discount_rate) ** np.arange(n_time)
        discount_factors = discount_factors[:, None]  # shape (T, 1) for broadcasting
    else:
        discount_factors = 1  # No discounting

    # Discounted (S - price)*M term for each time period
    discounted_S_minus_price_M = (S - price) * M * discount_factors

    # Discount other terms as needed
    discounted_production_price_capture = production * price * capture_rate * discount_factors

    # Calculate pi_G with discounting
    pi_G = (discounted_production_price_capture + discounted_S_minus_price_M).sum(axis=0)

    ord_G, bidx_G = _left_tail_mask(pi_G, PROB, alpha)

    rev_G = (-M * price * discount_factors).sum(axis=0)
    expected_G = (PROB * rev_G).sum()
    test = weighted_expected_value(rev_G, PROB)

    tail_G = _left_tail_weighted_sum(PROB, pi_G, ord_G, bidx_G, alpha)
    # Calculate S_star
    S_star = (1 - A) * expected_G + A * tail_G

    return S_star

def _calculate_S_star_BL_L(
    x, M, A, alpha, production, price, capture_rate, load_CR, load_scenarios, PROB,
    discount_rate=None, n_time=None
):
    """
    Calculate the optimal strike price S* for the Load-side in Baseload contract, with optional discounting.
    """
    S = x[0]
    # Assume all arrays are shape (T, S)
    if discount_rate is not None and n_time is not None:
        discount_factors = 1 / (1 + discount_rate) ** np.arange(n_time)
        discount_factors = discount_factors[:, None]  # shape (T, 1) for broadcasting
    else:
        discount_factors = 1  # No discounting

    # Discounted terms
    discounted_load_term = -load_scenarios * load_CR * price * discount_factors
    discounted_MS_term = ((price - S) * M) * discount_factors

    # Calculate pi_L with discounting
    pi_L = discounted_load_term.sum(axis=0) + discounted_MS_term.sum(axis=0)

    ord_L, bidx_L = _left_tail_mask(pi_L, PROB, alpha)

    rev_L = (M * price * discount_factors).sum(axis=0)
    expected_L = (PROB * rev_L).sum()
    tail_L = _left_tail_weighted_sum(PROB, pi_L, ord_L, bidx_L, alpha)
    # Calculate S_star
    S_star = (1 - A) * expected_L + A * tail_L

    return S_star


def optimize_nash_product(
    price_G: np.ndarray,
    price_L: np.ndarray,
    A_G: float,
    A_L: float,
    Beta_G: float,
    Beta_L: float,
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
        # Calculate Nash product ( Negative for minimization)
        nash_prod = (((UG - Zeta_G)**Beta_G) * ((UL - Zeta_L)**Beta_L))
        
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
    bounds = [(strikeprice_min, strikeprice_max) , (contract_amount_min, contract_amount_max  )]
    x0 = [((strikeprice_min + strikeprice_max)  )/2, (contract_amount_min + contract_amount_max )/2]

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