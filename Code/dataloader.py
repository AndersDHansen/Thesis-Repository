
import numpy as np

from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from utils import weighted_expected_value
class InputData:
    """
    Class to store and manage input data for power system simulations.
    """
    def __init__(
        self, 
        TIME: list,
        n_time: int,
        NUM_SCENARIOS: int,
        SCENARIOS: list,
        generator_contract_capacity: int,

        retail_price: float,
        strikeprice_min: float,
        strikeprice_max: float,
        contract_amount_min: int,
        contract_amount_max: int,
        gamma_max: float,
        A_L: float,
        A_G: float,
        tau_L: float,
        tau_G: float,  
        K_L_price: float, 
        K_G_price: float,
        K_L_prod: float,
        K_G_prod: float,
        alpha: float,
        Barter: bool,
        contract_type: str,
    ):
        # Basic system parameters
        self.TIME = TIME
        self.n_time = n_time
        self.NUM_SCENARIOS = NUM_SCENARIOS
        self.SCENARIOS_L = SCENARIOS
    
        # Generator parameters
        self.generator_contract_capacity = generator_contract_capacity
        # Contract parameters
        # Converting values to be ing Mio EUR/GWH per year
        self.retail_price = retail_price 
        self.strikeprice_min = strikeprice_min
        self.strikeprice_max = strikeprice_max
        self.contract_amount_min = contract_amount_min
        self.contract_amount_max = contract_amount_max
        self.gamma_max = gamma_max
        
        # Risk and price parameters
        self.A_L = A_L
        self.A_G = A_G
        self.tau_L = tau_L
        self.tau_G = tau_G
        self.K_L_price = K_L_price
        self.K_G_price = K_G_price
        self.K_L_prod = K_L_prod
        self.K_G_prod = K_G_prod
        self.alpha = alpha

        #Contract type and relaxation for modelling 
        self.Barter = Barter
        self.contract_type = contract_type

    def load_data_from_provider(self, provider):
        """Load and initialize data from provider."""
        # Load data matrices

        self.hours_in_year = 8760
        self.price_true = provider.price_matrix()
        self.production = provider.production_matrix()
        self.capture_rate = provider.capture_rate_matrix()
        self.load_scenarios = provider.load_matrix()
        self.load_CR = provider.load_capture_rate_matrix()
        self.PROB = provider._prob
        
        self.price_true = self.price_true.iloc[:self.n_time]
        self.production = self.production.iloc[:self.n_time]
        self.capture_rate = self.capture_rate.iloc[:self.n_time]
        self.load_scenarios = self.load_scenarios.iloc[:self.n_time]
        self.load_CR = self.load_CR.iloc[:self.n_time]

        # set columns 
        price_columns = self.price_true.columns
        self.production.columns = price_columns
        self.capture_rate.columns = price_columns   
        self.load_scenarios.columns = price_columns
        self.load_CR.columns = price_columns
        
        Capture_price_L = self.price_true * self.load_CR
        Capture_price_L_avg = weighted_expected_value(Capture_price_L, self.PROB)
        self.strikeprice_max = Capture_price_L_avg * 1.2
        
        print(f"Strike price bounds: {self.strikeprice_min:.6f} to {self.strikeprice_max:.6f}")
         
        return self
    



        return 

def load_data(opt_time_horizon: int, num_scenarios: int, A_G: float, A_L: float, 
              tau_L: float, tau_G: float, Barter: bool = True, 
              contract_type: str = "baseload") -> InputData:
    """Load system data and create initial parameters."""
    
    # Time parameters - these will be overwritten by load_data_from_provider
    # but we need them for initialization
    TIME = list(range(0, opt_time_horizon))  # Convert to list for consistency
    SCENARIOS = list(range(num_scenarios))  # Convert to list for consistency
    n_time = opt_time_horizon

    # Initialize PROB as array, not scalar
    
    generator_contract_capacity = 30  # MW
    
    # Contract parameters
    retail_price = 0 * 1e-3  # EUR/MWh
    strikeprice_min = 40 * 1e-3  # EUR/MWh
    strikeprice_max = 120 * 1e-3  # EUR/MWh (will be overwritten)
    contract_amount_min = 0
    contract_amount_max = generator_contract_capacity * 8760 * 1e-3  # GWh/year
    gamma_max = 1
    
    # Risk parameters
    K_L = 0
    K_G = 0
    alpha = 0.95
    
    input_data = InputData(
        TIME=TIME,
        n_time=n_time,
        NUM_SCENARIOS=num_scenarios,
        SCENARIOS=SCENARIOS,
        generator_contract_capacity=generator_contract_capacity,
        retail_price=retail_price, 
        strikeprice_min=strikeprice_min,
        strikeprice_max=strikeprice_max,
        contract_amount_min=contract_amount_min,
        contract_amount_max=contract_amount_max,
        gamma_max=gamma_max,
        A_L=A_L,
        A_G=A_G,
        tau_L=tau_L,
        tau_G=tau_G,
        K_L_price=K_L,
        K_G_price=K_G,
        K_L_prod=K_L,
        K_G_prod=K_G,
        alpha=alpha,
        Barter=Barter,
        contract_type=contract_type
    )
    
    return input_data