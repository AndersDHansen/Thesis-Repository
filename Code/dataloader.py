"""
Data management module for loading data for power system simulations.
"""
import numpy as np
import pandas as pd

from typing import Dict
import numpy as np
from scipy.stats import qmc, norm




class InputData:
    """
    Class to store and manage input data for power system simulations.
    """
    def __init__(
        self, 
        TIME: list,
        NUM_SCENARIOS: int,
        SCENARIOS: list,
        PROB: float,
        generator_contract_capacity: int,

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
        self.TIME = TIME
        self.NUM_SCENARIOS = NUM_SCENARIOS
        self.SCENARIOS_L = SCENARIOS
        self.PROB = PROB
    
        # Generator parameters
        self.generator_contract_capacity = generator_contract_capacity
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

def load_data(time_horizon: int, num_scenarios: int, A_G: float, A_L: float):
    """
    Load system data and create initial parameters.
    
    Args:
        time_horizon (int): Number of time periods to simulate
        num_scenarios (int): Number of scenarios
        A_G (float): Generator risk aversion parameter
        A_L (float): Load risk aversion parameter
    
    Returns:
        InputData: System data object
    """
    # Time parameters
    TIME = range(0, time_horizon) 
    SCENARIOS = range(num_scenarios)
    PROB = 1/len(SCENARIOS)    # Load scenarios from CSV file
    generator_contract_capacity = 100  # MW

    # Contract parameters
    retail_price = 25  # EUR/MWh
    strikeprice_min = 40  # EUR/MWh
    strikeprice_max = 170  # EUR/MWh
    contract_amount_min = 0
    contract_amount_max = generator_contract_capacity * 2

    # Risk parameters
    K_L = 0  # Price bias
    K_G = 0
    alpha = 0.95
  
    input_data = InputData(
        TIME=TIME,
        NUM_SCENARIOS=num_scenarios,
        SCENARIOS=SCENARIOS,
        PROB=PROB,
        generator_contract_capacity=generator_contract_capacity,
        retail_price=retail_price,
        strikeprice_min=strikeprice_min,
        strikeprice_max=strikeprice_max,
        contract_amount_min=contract_amount_min,
        contract_amount_max=contract_amount_max,
        A_L=A_L,
        A_G=A_G,
        K_L=K_L,
        K_G=K_G,
        alpha=alpha
    )

    return input_data