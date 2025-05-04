"""
Data management module for power system contract negotiation.
Handles data loading, scenario generation, and input data structures.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc


def generate_scenarios(num_L_scen, num_days, num_hours, L, daily_load_mean, load_std, all_nodes=False):
    """
    Generate load scenarios for the power system.
    
    Args:
        num_L_scen (int): Number of load scenarios
        num_days (int): Number of days to simulate
        num_hours (int): Number of hours per day
        L (int): Number of loads
        daily_load_mean (array): Mean daily load values
        load_std (float): Load standard deviation
        all_nodes (bool): Whether to generate scenarios for all nodes
    
    Returns:
        dict: Dictionary containing load scenarios
    """
    # Autocorrelation function
    autocorrelation = np.array([
        1.00000, 0.68366, 0.22233, -0.09257, -0.16865, -0.04008,
        0.18943, 0.36306, 0.28063, 0.12285, 0.00094, -0.05240,
        -0.05279, -0.03001, -0.00707, 0.00596, 0.00903, 0.00644,
        0.00251, -0.00028, -0.00137, -0.00125, -0.00065, -0.00011,
        0.00017, 0.00022, 0.00015, 0.00005, -0.00001, -0.00004
    ])
    autocorrelation = np.pad(autocorrelation, (0, num_days - len(autocorrelation[:num_days])), mode='constant')

    L1_3 = np.array([
        [350.00, 322.93, 305.04, 296.02, 287.16, 291.59, 296.02, 314.07,
         300.00, 276.80, 261.47, 253.73, 246.13, 249.93, 253.73, 269.20,
         250.00, 230.66, 217.89, 211.44, 205.11, 208.28, 211.44, 224.33],
        [408.25, 448.62, 430.73, 426.14, 421.71, 412.69, 390.37, 363.46,
         349.93, 384.53, 369.20, 365.26, 361.47, 353.73, 344.36, 311.53,
         291.61, 320.44, 307.67, 304.39, 301.22, 294.78, 278.83, 259.61]
    ])

    # Construct covariance matrix
    cov_matrix = np.zeros((num_days, num_days))
    for i in range(num_days):
        for j in range(num_days):
            lag = abs(i - j)
            if lag < len(autocorrelation):
                cov_matrix[i, j] = autocorrelation[lag] * load_std**2

    # Latin Hypercube Sampling
    lhs_sampler = qmc.LatinHypercube(d=num_days)
    samples = lhs_sampler.random(n=num_L_scen)
    
    # Transform samples using Cholesky decomposition
    Chol = np.linalg.cholesky(cov_matrix)
    daily_avg_loads = daily_load_mean[:num_days] + np.dot(samples, Chol.T)
    daily_avg_loads = np.maximum(daily_avg_loads, 0)

    # Hourly weight factors
    hourly_weight_factors = np.array([
        0.8, 0.7, 0.6, 0.5, 0.6, 0.7,  # Early morning
        1.2, 1.5, 1.8, 1.7, 1.6, 1.4,  # Morning peak
        1.3, 1.2, 1.1, 1.0, 1.1, 1.2,  # Afternoon
        1.4, 1.6, 1.8, 1.7, 1.5, 1.3   # Evening peak
    ])
    hourly_weight_factors /= hourly_weight_factors.mean()

    # Generate hourly loads
    load_dict = {}
    hourly_loads = np.zeros((num_L_scen, num_days, num_hours))
    
    for n in range(L):
        if not all_nodes and n == 1:
            load_dict['L'+str(n)] = np.tile(L1_3[0][:num_hours], (num_L_scen, num_days)).T
            
            for sample in range(num_L_scen):
                for day in range(num_days):
                    hourly_loads[sample, day, :] = daily_avg_loads[sample, day] * hourly_weight_factors[:num_hours]
            
            load_dict['L'+str(n+1)] = hourly_loads.reshape(num_hours*num_days, num_L_scen)
            load_dict['L'+str(n+2)] = np.tile(L1_3[0][:num_hours], (num_L_scen, num_days)).T
        
        elif all_nodes:
            for sample in range(num_L_scen):
                for day in range(num_days):
                    hourly_loads[sample, day, :] = daily_avg_loads[sample, day] * hourly_weight_factors[:num_hours]
    
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
        A_L2: float,
        A_G6: float,  
        K_L2: float, 
        K_G6: float, 
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
        self.A_L2 = A_L2
        self.A_G6 = A_G6
        self.K_L2 = K_L2
        self.K_G6 = K_G6
        self.alpha = alpha

def load_data(hours: int, days: int, scen: int, A_G6: float, A_L2: float):
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
    TIME = range(0, hours * days)
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
    load_capacity = generate_scenarios(len(SCENARIOS_L), days, hours, len(LOADS), daily_load_mean, load_std)
    
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
    contract_amount_max = 600
    
    # Risk parameters
    A_L2 = A_L2
    A_G6 = A_G6
    K_L2 = 0
    K_G6 = 0
    alpha = 0.95

    return (
        GENERATORS, LOADS, NODES, TIME, SCENARIOS_L, PROB,
        fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity,
        load_capacity, mapping_buses, mapping_generators, mapping_loads,
        branch_capacity_df, bus_reactance_df, System_data,
        strikeprice_min, retail_price, strikeprice_min, strikeprice_max,
        contract_amount_min, contract_amount_max,
        A_L2, A_G6, K_L2, K_G6, alpha
    )