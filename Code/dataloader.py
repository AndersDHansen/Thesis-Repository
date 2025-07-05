
import numpy as np

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
        gamma_max: float,
        A_L: float,
        A_G: float,
        Beta_L: float,
        Beta_G: float,  
        K_L: float, 
        K_G: float, 
        alpha: float,
        Barter: bool,
        contract_type: str,
    ):
        # Basic system parameters
        self.TIME = TIME
        self.NUM_SCENARIOS = NUM_SCENARIOS
        self.SCENARIOS_L = SCENARIOS
        self.PROB = PROB
    
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
        self.Beta_L = Beta_L
        self.Beta_G = Beta_G
        self.K_L = K_L
        self.K_G = K_G
        self.alpha = alpha

        #Contract type and relaxation for modelling 
        self.Barter = Barter
        self.contract_type = contract_type

    def load_data_from_provider(self, provider):
        """Load and initialize data from provider."""
        # Load data matrices
        self.hours_in_year = 8760
        self.price_true = provider.price_matrix()
        self.price_true = self.price_true 
        self.production = provider.production_matrix()
        self.capture_rate = provider.capture_rate_matrix()
        self.load_scenarios = provider.load_matrix()
        self.load_CR = provider.load_capture_rate_matrix()
        
        # Set scenario indices based on data dimensions
        if hasattr(self, 'price_true') and self.price_true is not None:
            self.SCENARIOS_L = list(range(self.price_true.shape[1]))
            self.TIME = list(range(self.price_true.shape[0]))
            # Equal probability for all scenarios
            self.PROB = np.ones(self.price_true.shape[1]) / self.price_true.shape[1]
        
        return self

def load_data(time_horizon: int, num_scenarios: int, A_G: float, A_L: float, Beta_L: float, Beta_G: float, Barter: bool = True, contract_type: str = "baseload") -> InputData:
    """
    Load system data and create initial parameters.
    
    Args:
        time_horizon (int): Number of time periods to simulate
        num_scenarios (int): Number of scenarios
        Beta_L (float): Load negotiation power parameter
        Beta_G (float): Generator negotiation power parameter
        A_G (float): Generator risk aversion parameter
        A_L (float): Load risk aversion parameter
    
    Returns:
        InputData: System data object
    """
    # Time parameters
    TIME = range(0, time_horizon) 
    SCENARIOS = range(num_scenarios)
    PROB = 1/len(SCENARIOS)    # Load scenarios from CSV file
    generator_contract_capacity = 30  # MW
    hours_in_year = 8760
    # Contract parameters
    retail_price = 0*1e-3 # EUR/MWh
    strikeprice_min = 20 *1e-3  # EUR/MWh # use lcoe for onshore widn (Average LCOE of Onshore Lazard , can try higher like 86, find upper gap)
    strikeprice_max = 70 *1e-3 # EUR/MWh # what would be a good maximum value?
    contract_amount_min = 0
    contract_amount_max = generator_contract_capacity  * hours_in_year *1e-3 # GWH/year
    gamma_max = 1  # Maximum contract (relevant for PAP contracts)

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
        # Converting values to be ing Mio EUR/GWH per year

        retail_price=retail_price, 
        strikeprice_min=strikeprice_min,
        strikeprice_max=strikeprice_max,
        contract_amount_min=contract_amount_min,
        contract_amount_max=contract_amount_max,
        gamma_max=gamma_max,
        A_L=A_L,
        A_G=A_G,
        Beta_L=Beta_L,
        Beta_G=Beta_G,
        K_L=K_L,
        K_G=K_G,
        alpha=alpha,
        Barter=Barter,
        contract_type=contract_type

    )

    return input_data