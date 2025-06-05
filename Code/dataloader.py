"""
Data management module for loading data for monte carlo analysis - Not applicable with OPF model.
OLd datam management is kept for reference and used for OPF data.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from dataclasses import dataclass                # ← add
from typing import Dict, Optional                # ← add
from Forecasting import MonteCarloConfig

@dataclass
class LHLoadConfig:
    time_horizon: int                    # e.g. 24 * DAYS
    daily_means: np.ndarray               # shape (num_days,)
    daily_std: float                      # σ of daily mean distribution
    hourly_weights: np.ndarray            # length-24 vector, mean≈1
    fixed_profiles: Dict[str, np.ndarray] | None = None   # {'L1': …}
    autocorr: Optional[np.ndarray] = None  # Add this if you use autocorr elsewhere

from typing import Dict
import numpy as np
from scipy.stats import qmc, norm


def generate_scenarios(mc_cfg: MonteCarloConfig,
                       lh_cfg: LHLoadConfig):
    """
    Latin-Hypercube-based load scenario generator with auto-numbered L-nodes.

    Parameters
    ----------
    mc_cfg : MonteCarloConfig
    lh_cfg : LHLoadConfig
    all_nodes : bool
        if True, generate stochastic load for every L-node not in fixed_profiles
        if False, only "L" is stochastic
    num_total_loads : int
        total number of L-nodes in the system (e.g., 3 → L1, L, L3)

    Returns
    -------
    Dict[str, np.ndarray]
        mapping from load name (e.g., "L") to hourly array (T, S)
    """
    rng = np.random.default_rng(mc_cfg.random_seed)
    num_L_scen = mc_cfg.n_simulations

    num_hours = 24
    num_days = lh_cfg.time_horizon // num_hours
    daily_mean = lh_cfg.daily_means[:num_days]
    load_std = lh_cfg.daily_std
    hw = lh_cfg.hourly_weights / lh_cfg.hourly_weights.mean()

    # Autocorrelation
    default_acf = np.array([
        1.00000, 0.68366, 0.22233, -0.09257, -0.16865, -0.04008,
        0.18943, 0.36306, 0.28063, 0.12285, 0.00094, -0.05240,
        -0.05279, -0.03001, -0.00707, 0.00596, 0.00903, 0.00644,
        0.00251, -0.00028, -0.00137, -0.00125, -0.00065, -0.00011,
        0.00017, 0.00022, 0.00015, 0.00005, -0.00001, -0.00004
    ])
    acf = lh_cfg.autocorr if lh_cfg.autocorr is not None else default_acf
    acf = np.pad(acf, (0, max(0, num_days - len(acf))), mode='constant')

    # Covariance matrix
    cov_matrix = np.zeros((num_days, num_days))
    for i in range(num_days):
        for j in range(num_days):
            lag = abs(i - j)
            cov_matrix[i, j] = acf[lag] * load_std**2

    # Latin-Hypercube Sampling
    lhs_sampler = qmc.LatinHypercube(d=num_days, seed=rng)
    U = lhs_sampler.random(n=num_L_scen)
    Z = norm.ppf(U)
    Chol = np.linalg.cholesky(cov_matrix)
    daily_avg_loads = daily_mean + np.dot(Z, Chol.T)
    daily_avg_loads = np.maximum(daily_avg_loads, 0)

    # Expand daily → hourly
    hourly_loads = daily_avg_loads[:, :, None] * hw[None, None, :]
    hourly_loads = hourly_loads.reshape(num_L_scen, num_days * num_hours).T  # (T, S)

    # Start dictionary
    load_dict: Dict[str, np.ndarray] = {}

    # Fixed loads
    if lh_cfg.fixed_profiles:
        for name, profile in lh_cfg.fixed_profiles.items():
            if profile.shape[0] != num_hours:
                raise ValueError(f"Fixed profile for {name} must have 24 hours")
            repeated = np.tile(profile, num_days)[:, None]        # (T, 1)
            load_dict[name] = np.repeat(repeated, num_L_scen, axis=1)  # (T, S)

    return load_dict

class InputData:
    """
    Class to store and manage input data for power system simulations.
    """
    def __init__(
        self, 
        LOADS: list, 
        TIME: list,
        NUM_SCENARIOS: int,
        SCENARIOS: list,
        PROB: float,
        generator_contract_capacity: int,
        load_scenarios: np.ndarray,

        retail_price: float,
        strikeprice_min: float,
        strikeprice_max: float,
        contract_amount_min: int,
        contract_amount_max: int,
        A_L: float,
        A_G: float,  
        K_L: float, 
        K_G: float, 
        alpha: float
    ):
        # Basic system parameters
        self.LOADS = LOADS
        self.TIME = TIME
        self.NUM_SCENARIOS = NUM_SCENARIOS
        self.SCENARIOS_L = SCENARIOS
        self.PROB = PROB
        
        # Generator parameters
   
        self.generator_contract_capacity = generator_contract_capacity
        
        # Load parameters
        self.load_scenarios = load_scenarios

        
        # Contract parameters
        self.retail_price = retail_price
        self.strikeprice_min = strikeprice_min
        self.strikeprice_max = strikeprice_max
        self.contract_amount_min = contract_amount_min
        self.contract_amount_max = contract_amount_max
        
        # Risk and price parameters
        self.A_L = A_L
        self.A_G = A_G
        self.K_L = K_L
        self.K_G = K_G
        self.alpha = alpha

def load_data(time_horizon: int, NUM_SCENARIOS: int, A_G: float, A_L: float):
    """
    Load system data and create initial parameters.
    
    Args:
        hours (int): Number of hours per day
        days (int): Number of days to simulate
        scen (int): Number of scenarios
        beta_G (float): Generator risk aversion parameter
        beta_L (float): Load risk aversion parameter
    
    Returns:
        tuple: System parameters and data
    """
    LOADS = ['L2']
    # Time parameters
    TIME = range(0, time_horizon) # yearas
    SCENARIOS = range(NUM_SCENARIOS)
    PROB = 1/len(SCENARIOS)

    # Load data
    daily_load_mean = np.array([
        337.01, 319.10, 285.94, 268.12, 318.61, 329.53, 335.84, 336.94, 
        316.81, 270.06, 250.76, 297.36, 310.81, 322.45, 338.52, 360.43, 
        341.99, 312.55, 351.49, 349.64, 363.59, 367.08, 336.56, 300.43, 
        285.71, 329.89, 335.36, 336.34, 337.69, 336.93
    ])
    load_mean = np.mean(daily_load_mean)

    load_std = np.sqrt(834.5748)

    load_scenarios = np.random.normal(load_mean, load_std, size=(time_horizon, NUM_SCENARIOS)) * 8760 / 1000 # Scale to yearly load convert to GWh
    
    # Generate load scenarios
    L1_3=np.array([
        [350.00, 322.93, 305.04, 296.02, 287.16, 291.59, 296.02, 314.07,
         300.00, 276.80, 261.47, 253.73, 246.13, 249.93, 253.73, 269.20,
         250.00, 230.66, 217.89, 211.44, 205.11, 208.28, 211.44, 224.33],
        [408.25, 448.62, 430.73, 426.14, 421.71, 412.69, 390.37, 363.46,
         349.93, 384.53, 369.20, 365.26, 361.47, 353.73, 344.36, 311.53,
         291.61, 320.44, 307.67, 304.39, 301.22, 294.78, 278.83, 259.61]
    ])

    hourly_weight_factors = np.array([
        0.8, 0.7, 0.6, 0.5, 0.6, 0.7,  # Early morning
        1.2, 1.5, 1.8, 1.7, 1.6, 1.4,  # Morning peak
        1.3, 1.2, 1.1, 1.0, 1.1, 1.2,  # Afternoon
        1.4, 1.6, 1.8, 1.7, 1.5, 1.3   # Evening peak
    ])
    hourly_weight_factors /= hourly_weight_factors.mean() # This is complete arbitrary valyes taken

    # ---------- Monte-Carlo *meta* config (shared by all stochastic sources)
    """
    mc_cfg = MonteCarloConfig(n_simulations=NUM_SCENARIOS, random_seed=42)

    lh_cfg = LHLoadConfig(
        time_horizon = time_horizon,
        daily_means   =daily_load_mean,     # shape (30,)
        daily_std     = load_std,
        hourly_weights= hourly_weight_factors,     # shape (24,)
        fixed_profiles={
            "L1": L1_3[0],           # shape (24,)
            "L3": L1_3[1]
        }
    )

    load_scenarios   = generate_scenarios(mc_cfg, lh_cfg)
    """
    
    generator_contract_capacity = 100 # MW

    # Contract parameters
    retail_price = 25 # EUR/MWh
    strikeprice_min = 50 # EUR/MWh
    strikeprice_max = 110 # EUR/MWh
    contract_amount_min = 0
    contract_amount_max = generator_contract_capacity * 2
    
    # Risk parameters
    A_L = A_L
    A_G = A_G
    K_L = 0
    K_G = 0
    alpha = 0.95

    return ( 
        LOADS, TIME, SCENARIOS, PROB,
        generator_contract_capacity,
        load_scenarios, retail_price,
        strikeprice_min, strikeprice_min, strikeprice_max,
        contract_amount_min, contract_amount_max,
        A_L, A_G, K_L, K_G, alpha
    )