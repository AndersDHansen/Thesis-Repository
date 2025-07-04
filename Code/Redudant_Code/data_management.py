"""
Data management module for power system contract negotiation.
Handles data loading, scenario generation, and input data structures.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from dataclasses import dataclass                # ← add
from typing import Dict, Optional                # ← add
from Code.Redudant_Code.Forecasting import MonteCarloConfig

@dataclass
class LHLoadConfig:
    horizon_hours: int                    # e.g. 24 * DAYS
    daily_means: np.ndarray               # shape (num_days,)
    daily_std: float                      # σ of daily mean distribution
    hourly_weights: np.ndarray            # length-24 vector, mean≈1
    fixed_profiles: Dict[str, np.ndarray] | None = None   # {'L1': …}
    autocorr: Optional[np.ndarray] = None  # Add this if you use autocorr elsewhere

from typing import Dict
import numpy as np
from scipy.stats import qmc, norm


def generate_scenarios(mc_cfg: MonteCarloConfig,
                       lh_cfg: LHLoadConfig,
                       all_nodes: bool = False,
                       num_total_loads: int = 3) -> Dict[str, np.ndarray]:
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
    if lh_cfg.horizon_hours % num_hours != 0:
        raise ValueError("horizon_hours must be a multiple of 24")
    num_days = lh_cfg.horizon_hours // num_hours
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

    # Auto-number stochastic loads
    for i in range(1, num_total_loads + 1):         # L1, L, ..., LN
        lname = f"L{i}"
        if lname in load_dict:
            continue                                # skip fixed ones
        if not all_nodes and lname != "L":
            continue                                # only allow L if all_nodes=False
        load_dict[lname] = hourly_loads.copy()      # same profile to all dynamic loads

    return load_dict

class InputData:
    """
    Class to store and manage input data for power system simulations.
    """
    def __init__(
        self, 
        GENERATORS: list, 
        LOADS: list, 
        NODES: list,
        TIME: list,
        SCENARIOS_L: list,
        PROB: float,
        generator_cost_fixed: dict, 
        generator_cost_a: dict,
        generator_cost_b: dict,   
        generator_capacity: dict, 
        generator_contract_capacity: int,
        load_capacity: dict,
        mapping_buses: dict,
        mapping_generators: dict,
        mapping_loads: dict,
        branch_capacity: pd.DataFrame,
        bus_susceptance: pd.DataFrame,
        slack_bus: str,
        system_data: dict,
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
        self.GENERATORS = GENERATORS
        self.LOADS = LOADS
        self.NODES = NODES
        self.TIME = TIME
        self.SCENARIOS_L = SCENARIOS_L
        self.PROB = PROB
        
        # Generator parameters
        self.generator_cost_fixed = generator_cost_fixed
        self.generator_cost_a = generator_cost_a
        self.generator_cost_b = generator_cost_b
        self.generator_capacity = generator_capacity
        self.generator_contract_capacity = generator_contract_capacity
        
        # Load parameters
        self.load_capacity = load_capacity
        self.num_L_scen = len(SCENARIOS_L)
        
        # Network topology
        self.mapping_buses = mapping_buses
        self.mapping_generators = mapping_generators
        self.mapping_loads = mapping_loads
        self.branch_capacity = branch_capacity
        self.bus_susceptance = bus_susceptance
        self.slack_bus = slack_bus
        
        # System data
        self.system_data = system_data
        
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

def load_data(time_horizon: int, scen: int, A_G: float, A_L: float):
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
    
    # System base parameters
    So = 100  # Base Apparent Power
    Vo = 10.  # Base Voltage
    pi_b = 0.05  # Voltage penalty
    System_data = {'So': So, 'Vo': Vo, 'pi_b': pi_b}

    # System components
    GENERATORS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
    LOADS = ['L1', 'L2', 'L3']
    NODES = ['N1', 'N2', 'N3', 'N4', 'N5']
    
    # Time parameters
    TIME = range(0, time_horizon) # yearas
    SCENARIOS_L = range(scen)
    PROB = 1/len(SCENARIOS_L)

    # Network topology
    mapping_buses = pd.DataFrame({
        'N1': [0, 1, 1, 1, 1],
        'N2': [1, 0, 1, 1, 1],
        'N3': [1, 1, 0, 1, 1],
        'N4': [1, 1, 1, 0, 1],
        'N5': [1, 1, 1, 1, 0]
    }, index=NODES)

    # Branch data
    Line_reactance = np.array([0.0281, 0.0304, 0.0064, 0.0108, 0.0297, 0.0297])
    Line_from = np.array([1, 1, 1, 2, 3, 4])
    Line_to = np.array([2, 4, 5, 3, 4, 5])
    Linecap = np.array([250, 150, 400, 350, 240, 240])

    # Initialize matrices
    branch_capacity = np.zeros((len(NODES), len(NODES)))
    bus_reactance = np.zeros((len(NODES), len(NODES)))

    # Fill matrices
    for i in range(len(Line_from)):
        node1 = Line_from[i] - 1
        node2 = Line_to[i] - 1
        cap = Linecap[i]
        reactance = Line_reactance[i]
        
        # Branch capacity
        branch_capacity[node1, node2] = cap
        branch_capacity[node2, node1] = cap
        
        # Bus reactance
        susceptance = 1 / reactance
        bus_reactance[node1, node1] += susceptance
        bus_reactance[node2, node2] += susceptance
        bus_reactance[node1, node2] -= susceptance
        bus_reactance[node2, node1] -= susceptance

    branch_capacity_df = pd.DataFrame(branch_capacity, index=NODES, columns=NODES)
    bus_reactance_df = pd.DataFrame(bus_reactance, index=NODES, columns=NODES)

    # Generator data
    fixed_Cost = {'G1': 1600, 'G2': 1200, 'G3': 8500, 'G4': 1000, 'G5': 5400, 'G6': 0}
    generator_cost_a = {'G1': 14, 'G2': 15, 'G3': 25, 'G4': 30, 'G5': 10, 'G6': 10}
    generator_cost_b = {'G1': 0.005, 'G2': 0.006, 'G3': 0.01, 'G4': 0.012, 'G5': 0.007, 'G6': 0.005}
    generator_capacity = {'G1': 110, 'G2': 110, 'G3': 520, 'G4': 200, 'G5': 600, 'G6': 300}
    generator_contract_capacity = 100 # 300 if OPF 
    
    mapping_generators = pd.DataFrame({
        'N1': [1, 1, 0, 0, 0, 0],
        'N2': [0, 0, 0, 0, 0, 0],
        'N3': [0, 0, 1, 0, 0, 1],
        'N4': [0, 0, 0, 1, 0, 0],
        'N5': [0, 0, 0, 0, 1, 0]
    }, index=GENERATORS)

    # Load data
    daily_load_mean = np.array([
        337.01, 319.10, 285.94, 268.12, 318.61, 329.53, 335.84, 336.94, 
        316.81, 270.06, 250.76, 297.36, 310.81, 322.45, 338.52, 360.43, 
        341.99, 312.55, 351.49, 349.64, 363.59, 367.08, 336.56, 300.43, 
        285.71, 329.89, 335.36, 336.34, 337.69, 336.93
    ])

    load_std = np.sqrt(834.5748)
    
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
    mc_cfg = MonteCarloConfig(n_simulations=scen, random_seed=42)

    lh_cfg = LHLoadConfig(
        horizon_hours = hours * days,
        daily_means   =daily_load_mean,     # shape (30,)
        daily_std     = load_std,
        hourly_weights= hourly_weight_factors,     # shape (24,)
        fixed_profiles={
            "L1": L1_3[0],           # shape (24,)
            "L3": L1_3[1]
        }
    )

    load_capacity = generate_scenarios(mc_cfg, lh_cfg, all_nodes=True, num_total_loads=3)
    

    
    mapping_loads = pd.DataFrame({
        'N1': [0, 0, 0],
        'N2': [1, 0, 0],
        'N3': [0, 1, 0],
        'N4': [0, 0, 1],
        'N5': [0, 0, 0]
    }, index=LOADS)

    # Contract parameters
    retail_price = 25
    strikeprice_min = 15
    strikeprice_max = 25
    contract_amount_min = 0
    contract_amount_max = generator_contract_capacity * 2
    
    # Risk parameters
    A_L = A_L
    A_G = A_G
    K_L = 0
    K_G = 0
    alpha = 0.95

    return (
        GENERATORS, LOADS, NODES, TIME, SCENARIOS_L, PROB,
        fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity,generator_contract_capacity,
        load_capacity, mapping_buses, mapping_generators, mapping_loads,
        branch_capacity_df, bus_reactance_df, System_data,retail_price,
        strikeprice_min, strikeprice_min, strikeprice_max,
        contract_amount_min, contract_amount_max,
        A_L, A_G, K_L, K_G, alpha
    )