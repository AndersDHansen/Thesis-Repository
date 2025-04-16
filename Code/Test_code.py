
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import pandas as pd 
import scipy.stats as stats
from scipy.stats import qmc
from tqdm import tqdm
import seaborn as sns
from matplotlib.patches import Polygon # Import added at the top of the file


np.random.seed(42)
#%%
def generate_scenarions(num_L_scen,num_days,num_hours,L,daily_load_mean,load_std,all_nodes = False):

    # Make function value - Later #######
    # Autocorrelation function (example values)
    # This represents the autocorrelation at lags 0, 1, 2, ..., (num_days - 1)
    autocorrelation = np.array([
    1.00000, 0.68366, 0.22233, -0.09257, -0.16865, -0.04008,
    0.18943, 0.36306, 0.28063, 0.12285, 0.00094, -0.05240,
    -0.05279, -0.03001, -0.00707, 0.00596, 0.00903, 0.00644,
    0.00251, -0.00028, -0.00137, -0.00125, -0.00065, -0.00011,
    0.00017, 0.00022, 0.00015, 0.00005, -0.00001, -0.00004
    ])  # Example autocorrelation values
    autocorrelation = np.pad(autocorrelation, (0, num_days - len(autocorrelation[:num_days])), mode='constant')  # Pad with zeros if needed

    L1_3 = np.array([       [350.00, 322.93, 305.04, 296.02, 287.16, 291.59, 296.02, 314.07,
                            300.00, 276.80, 261.47, 253.73, 246.13, 249.93, 253.73, 269.20,
                            250.00, 230.66, 217.89, 211.44, 205.11, 208.28, 211.44, 224.33],

                            [408.25, 448.62, 430.73, 426.14, 421.71, 412.69, 390.37, 363.46,
                            349.93, 384.53, 369.20, 365.26, 361.47, 353.73, 344.36, 311.53,
                            291.61, 320.44, 307.67, 304.39, 301.22, 294.78, 278.83, 259.61] ])

    # Construct the covariance matrix using the autocorrelation function
    cov_matrix = np.zeros((num_days, num_days))
    for i in range(num_days):
        for j in range(num_days):
            lag = abs(i - j)  # Time lag between days i and j
            if lag < len(autocorrelation):
                cov_matrix[i, j] = autocorrelation[lag] * load_std**2  # Scale by variance

    # Latin Hypercube Sampling (LHS)
    lhs_sampler = qmc.LatinHypercube(d=num_days)
    samples = lhs_sampler.random(n=num_L_scen)

    # Transform samples to the desired distribution using Cholesky decomposition
    Chol = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition of the covariance matrix
    daily_avg_loads = daily_load_mean[:num_days] + np.dot(samples, Chol.T)  # Transform samples to the desired distribution

    # Truncate to ensure non-negative loads
    daily_avg_loads = np.maximum(daily_avg_loads, 0)

    # Generate hourly loads using intraday weight factors
    # Example: Define intraday weight factors (e.g., higher during peak hours)
    hourly_weight_factors = np.array([
        0.8, 0.7, 0.6, 0.5, 0.6, 0.7,  # Early morning (12 AM - 6 AM)
        1.2, 1.5, 1.8, 1.7, 1.6, 1.4,  # Morning peak (6 AM - 12 PM)
        1.3, 1.2, 1.1, 1.0, 1.1, 1.2,  # Afternoon (12 PM - 6 PM)
        1.4, 1.6, 1.8, 1.7, 1.5, 1.3   # Evening peak (6 PM - 12 AM)
    ])

    hourly_weight_factors /= hourly_weight_factors.mean()  # Normalize so that the mean is 1

    # Generate hourly loads
    load_dict = {}
    hourly_loads = np.zeros((num_L_scen, num_days, num_hours))
    for n in range(L):
        if all_nodes == False and n == 1:
          
            # fix later
            load_dict['L'+str(n)] = np.tile(L1_3[0][:num_hours],(num_L_scen,num_days)).T

            for sample in range(num_L_scen):
                for day in range(num_days):
                    # Scale the daily average load by the hourly weight factors
                    hourly_loads[sample, day, :] = daily_avg_loads[sample, day] * hourly_weight_factors[:num_hours]
            load_dict['L'+str(n+1)]  = hourly_loads.reshape(num_hours*num_days,num_L_scen, )
            load_dict['L'+str(n+2)] = np.tile(L1_3[0][:num_hours],(num_L_scen,num_days)).T

        elif all_nodes == True:
             for sample in range(num_L_scen):
                for day in range(num_days):
                    # Scale the daily average load by the hourly weight factors
                    hourly_loads[sample, day, :] = daily_avg_loads[sample, day] * hourly_weight_factors[:num_hours]

    #   [Scenario,day,hour]
       
        
    return load_dict

def load_data(hours,days,scen,beta_G, beta_L):
    # Load Data
    # Base Apparent Power
    So = 100
    #Base Voltage
    Vo = 10.
    #Voltage penalty
    pi_b = 0.05

    System_data = {'So':So,'Vo':Vo,'pi_b':pi_b}

    #System 
    GENERATORS = ['G1','G2','G3','G4','G5','G6'] #range of generators (G1...G6)  
    LOADS = ['L1','L2','L3'] 
    NODES = ['N1','N2','N3','N4','N5']
    N = len(NODES)
    L = len(LOADS)
    G = len(GENERATORS)
    #Hours
    T = hours #24
    DAYS = days#30
    TIME = range(0,T*DAYS)
    SCENARIOS_L =range(scen) # Number of scenarios for load 500? # something with scenario is cucking optimization
    PROB = 1/len(SCENARIOS_L)
    mapping_buses = pd.DataFrame({'N1':[0,1,1,1,1],'N2':[1,0,1,1,1],'N3':[1,1,0,1,1],'N4':[1,1,1,0,1],'N5':[1,1,1,1,0]},index = NODES)

    # Branch Data 
    Line_reactane = np.array([0.0281,0.0304,0.0064,0.0108,0.0297,0.0297])

    Line_from = np.array([1,1,1,2,3,4])
    Line_to = np.array([2,4,5,3,4,5])
    Linecap = np.array([250,150,400,350,240,240])#+100

    # Initialize matrices
    branch_capacity = np.zeros((N, N))
    bus_reactance = np.zeros((N, N))

    # Fill branch capacity matrix
    for i in range(len(Line_from)):
        node1 = Line_from[i] - 1  # Convert to zero-indexed
        node2 = Line_to[i] - 1
        cap = Linecap[i]
        branch_capacity[node1, node2] = cap
        branch_capacity[node2, node1] = cap  # Symmetric matrix

    # Fill bus reactance matrix
    for i in range(len(Line_from)):
        node1 = Line_from[i] - 1
        node2 = Line_to[i] - 1
        reactance = Line_reactane[i]
        susceptance = 1 / reactance
        bus_reactance[node1, node1] += susceptance
        bus_reactance[node2, node2] += susceptance
        bus_reactance[node1, node2] -= susceptance
        bus_reactance[node2, node1] -= susceptance

    # Convert to pandas DataFrame
    branch_capacity_df = pd.DataFrame(branch_capacity, index=NODES, columns=NODES)
    bus_reactance_df = pd.DataFrame(bus_reactance, index=NODES, columns=NODES)

    # Generator Data 

    #Generator 6 is the generator that contracts L2 for a financial contract 
    fixed_Cost = {'G1':1600,'G2':1200,'G3':8500,'G4':1000,'G5':5400, 'G6':0}
    generator_cost_a = {'G1':14,'G2':15,'G3':25,'G4':30,'G5':10,'G6':10}
    generator_cost_b = {'G1':0.005,'G2':0.006,'G3':0.01,'G4':0.012,'G5':0.007,'G6':0.005}
    generator_capacity = {'G1':110,'G2':110,'G3':520,'G4':200,'G5':600,'G6':300}
    mapping_generators = pd.DataFrame({'N1':[1,1,0,0,0,0],'N2':[0,0,0,0,0,0],'N3':[0,0,1,0,0,1],'N4':[0,0,0,1,0,0],'N5':[0,0,0,0,1,0]},index = GENERATORS)

    # Load Data 
    daily_load_mean = np.array([
    337.01, 319.10, 285.94, 268.12, 318.61,
    329.53, 335.84, 336.94, 316.81, 270.06,
    250.76, 297.36, 310.81, 322.45, 338.52,
    360.43, 341.99, 312.55, 351.49, 349.64,
    363.59, 367.08, 336.56, 300.43, 285.71,
    329.89, 335.36, 336.34, 337.69, 336.93
    ])

    load_std = np.sqrt(834.5748)  # Standard deviation of the daily load data 
    
    # Generate load scenarios
    load_capacity= generate_scenarions(len(SCENARIOS_L),DAYS,T,L,daily_load_mean,load_std)
    # Create a DataFrame from the Load_data array
    mapping_loads = pd.DataFrame({'N1':[0,0,0],'N2':[1,0,0],'N3':[0,1,0],'N4':[0,0,1],'N5':[0,0,0]},index = LOADS)

   

    # Contract Negotiation Data  #

    # Potentially make into dictionaries instead of lists

    # Consumer Price (for L2)
    retail_price = 25 # $/MWh (f in paper)

    # Strike Price (S)
    strikeprice_min = 15
    strikeprice_max = 25
    # Contract Amount (M)
    contract_amount_min = 0
    contract_amount_max = 600
    # Risk Aversion (A)
    A_L2 = beta_L# Load (Buyer)
    A_G6 = beta_G # Generator (Seller)

    # Price Bias (K)
    K_L2 = 0
    K_G6 = 0

    # Alpha (Confidence Level)
    alpha = 0.95


    return (GENERATORS, LOADS, NODES,TIME,SCENARIOS_L,PROB, fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity, load_capacity, 
            mapping_buses, mapping_generators, mapping_loads, branch_capacity_df, bus_reactance_df, System_data,strikeprice_min,
            retail_price,strikeprice_min,strikeprice_max,contract_amount_min,contract_amount_max,A_L2,A_G6,K_L2,K_G6,alpha)
class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class InputData:
    def __init__(
        self, 
        GENERATORS: list, 
        LOADS: list, 
        NODES: list,
        TIME: list,
        SCENARIOS_L: list,
        PROB: float,
        generator_cost_fixed: dict[str, int], 
        generator_cost_a: dict[str, int],
        generator_cost_b: dict[str, int],   
        generator_capacity: dict[str, int], 
        load_capacity: dict[str, int],
        mapping_buses: dict[str, list],
        mapping_generators: dict[str, list],
        mapping_loads: dict[str, list],
        branch_capacity: pd.DataFrame,
        bus_susceptance: pd.DataFrame,
        slack_bus: str,
        system_data: dict[str, int],
        retail_price: float,
        strikeprice_min: float,
        strikeprice_max: float,
        contract_amount_min: float,
        contract_amount_max: float,
        A_L2: float,
        A_G6: float,
        K_L2: float,
        K_G6: float,
        alpha: int
    ):
        # List of generators 
        self.GENERATORS = GENERATORS
        # List of loads
        self.LOADS = LOADS
        # List of nodes
        self.NODES = NODES
        # List of time periods ( A month: 30 days)
        self.TIME = TIME
        # List of scenarios for load
        self.SCENARIOS_L = SCENARIOS_L
        #Probablity of scenario (equprobable - maybe change later)
        self.PROB = PROB
         # Generators fixed costs (cF^G_i)
        self.generator_cost_fixed= generator_cost_fixed 
        # Generators costs (ca^G_i)
        self.generator_cost_a = generator_cost_a
         # Generators costs (cb^G_i)
        self.generator_cost_b = generator_cost_b  
        # Generators capacity (P^G_i)
        self.generator_capacity = generator_capacity 
        # Loads capacity (P^D_i)
        self.load_capacity = load_capacity 
        #Number of load scenarios 
        self.num_L_scen = len(SCENARIOS_L)
        # matrix of connections between nodes ij (=1 if connected, = 0 if not connected)
        self.mapping_buses = mapping_buses
        # matrix of location of generators at nodes (=1 if generator is at node i, = 0 if not at node i)
        self.mapping_generators = mapping_generators
        # matrix of location of loads at nodes (=1 if load is at node i, = 0 if not at node i)
        self.mapping_loads = mapping_loads
        # matrix of susceptance of branch between nodes ij
        self.branch_capacity = branch_capacity
        # matrix B of bus susceptance
        self.bus_susceptance = bus_susceptance
        # slack bus in network
        self.slack_bus = slack_bus
        # System data
        self.system_data = system_data
        #Retail Price for consumders 
        self.retail_price = retail_price
        # Strike Price (S) Boundaries
        self.strikeprice_min = strikeprice_min
        self.strikeprice_max = strikeprice_max
        # Contract Amount (M) Boundaries
        self.contract_amount_min = contract_amount_min
        self.contract_amount_max = contract_amount_max
        # Risk Aversion (A) Amounts to be tested
        self.A_L2 = A_L2
        self.A_G6 = A_G6
        # Price Bias (K) Amounts to be tested 
        self.K_L2 = K_L2
        self.K_G6 = K_G6

        # Confidence Level (Alpha)
        self.alpha = alpha

class OptimalPowerFlow():

    #SAVE RESULTS TO CSV AND READ INSTEAD OF SOLVING each time at run 

    def __init__(self, input_data: InputData): # initialize class
        self.data = input_data          # define data attributes
        self.variables = Expando()      # define variable attributes
        self.constraints = Expando()    # define constraints attributes
        self.results = Expando()        # define results attributes
        self._build_model()             # build gurobi model
    

    def _build_variables(self):
        # build generator production variables
        self.variables.generator_production = {
            (g,t,s): self.model.addVar(
                lb=0,
                name='Electricity production of generator {0} at t {1} in scenario{2}'.format(g,t,s)
            ) for g in self.data.GENERATORS 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }
        # build voltage angle variables
        self.variables.voltage_angle = {
            (n,t,s): self.model.addVar(
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
                name='Voltage angle at node {0} at time {1} in scenario{2}'.format(n,t,s)
            ) for n in self.data.NODES 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
             
        }
    def _build_constraints(self):
        #Balance equation at each node
        
        self.constraints.balance_constraint = {
            (n,t,s): self.model.addLConstr(
                gp.quicksum(
                    self.variables.generator_production[g,t,s] * self.data.mapping_generators[n][g] for g in self.data.GENERATORS 
                ) - gp.quicksum(
                    self.data.load_capacity[l][t,s] * self.data.mapping_loads[n][l] for l in self.data.LOADS
                ),
                gp.GRB.EQUAL,
                gp.quicksum(
                    self.data.bus_susceptance[n][m] * self.variables.voltage_angle[m,t,s] for m in self.data.NODES 
                ),
                name='Balance equation at node {0} at time {1} in scenario in scenario {2}'.format(n,t,s)
            ) for n in self.data.NODES 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L

        }

        #Max flow between nodes
        self.constraints.max_flow_constraint = {
            (n, m,t,s): self.model.addLConstr(
                self.data.bus_susceptance[n][m] * (self.variables.voltage_angle[n,t,s] - self.variables.voltage_angle[m,t,s]),
                gp.GRB.LESS_EQUAL,
                self.data.branch_capacity[n][m],
                name='Constraint on max flow between nodes {0} and {1} at time {2} in scenario {3}'.format(n, m,t,s)
            ) for n in self.data.NODES for m in [node for node in self.data.NODES if node not in [n]]
              for s in self.data.SCENARIOS_L for t in self.data.TIME
        }

        #Max production of generators
        self.constraints.capacity_constraints = {
            (g,t,s): self.model.addLConstr(
                self.variables.generator_production[g,t,s],
                gp.GRB.LESS_EQUAL,
                self.data.generator_capacity[g],
                name='Constraint on max production of generator {0} at time {1} in scenario {2}'.format(g,t,s)
            ) for g in self.data.GENERATORS[0:6]
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
              
        }

        
        #Set production for generator 6 at all times : 300 MW

        self.constraints.base_load_Gen6 = {
            (t,s): self.model.addLConstr(
            self.variables.generator_production['G6', t,s],
            gp.GRB.EQUAL,
            self.data.generator_capacity['G6'],
            name='Constraint on baseload generator G6 at time {0} in scenario {1}'.format(t,s)
        ) for t in self.data.TIME
          for s in self.data.SCENARIOS_L

        }   

        #Slack bus voltage angle
        self.constraints.slack_bus_constraint = {(t,s): self.model.addLConstr(
            self.variables.voltage_angle[self.data.slack_bus,t,s],
            gp.GRB.EQUAL,
            0,
            name='Slack bus voltage angle'
        ) for t in self.data.TIME
          for s in self.data.SCENARIOS_L
        }
    def _build_objective(self):
        #Objective function
        objective =self.data.PROB * gp.quicksum(
            self.data.generator_cost_a[g] * self.variables.generator_production[g,t,s] +
            self.data.generator_cost_b[g] * (self.variables.generator_production[g,t,s] *self.variables.generator_production[g,t,s]) 
            for g in self.data.GENERATORS for t in self.data.TIME for s in self.data.SCENARIOS_L
        ) 
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
    def _build_model(self):
        self.model = gp.Model(name='Economic dispatch')
        self.model.Params.OutputFlag =  0
        self.model.Params.TimeLimit = 400
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
         # save objective value
        self.results.objective_value = self.model.ObjVal
        # save generator dispatch values
        self.results.generator_production = {g:{(t,s): self.variables.generator_production[g,t,s].x  for t in self.data.TIME for s in self.data.SCENARIOS_L
        }for g in self.data.GENERATORS}
    

        # save price (i.e., dual variable of balance constraint)
        self.results.price ={n: { (t,s): (1/self.data.PROB)*self.constraints.balance_constraint[n,t,s].Pi for t in self.data.TIME for s in self.data.SCENARIOS_L }for n in self.data.NODES }
   
        #Save Load Revenue - Just assuming they all have the same retail price (L2 is the one in question though)

        self.results.generator_revenue = {g: {(t, s): sum((self.results.price[n][t, s] - self.data.generator_cost_a[g]) * self.results.generator_production[g][ t, s]- self.data.generator_cost_b[g] * self.results.generator_production[g][ t, s] ** 2
            for n in self.data.NODES
            if self.data.mapping_generators.loc[g, n] == 1  # Only include nodes where the generator is located
            )
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
            }for g in self.data.GENERATORS}
     


    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")
        
    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        
        print("Optimal energy production cost:")
        print(self.results.objective_value)
        print("Optimal generator dispatches:")
        print(self.results.generator_production)      
        print("Prices at optimality:")
        print(self.results.price)

class ContractNegotiation():
    def __init__(self,input_data:InputData,opf_results: OptimalPowerFlow):
        self.data = input_data  

        #For special Case of negative log 
        self.data.original_strike_min = self.data.strikeprice_min
        self.data.original_strike_max = self.data.strikeprice_max

        self.opf_results = opf_results
        self.results = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self._build_statistics()
        self._build_model()
        
    def _build_dataframe(self,data,input_name):
        # data is a dictionary with keys as tuples (time, scenario) and values as the corresponding values
        # input_name is the name of the input variable (e.g., 'price', 'generator_production', etc.)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index', columns=[str(input_name)])

        # Reset the index and split it into 'Time' and 'Scenario'
        df.index = pd.MultiIndex.from_tuples(df.index, names=['Time', 'Scenario'])
        df.reset_index(inplace=True)

        # Rename columns
        df.columns = ['Time', 'Scenario', str(input_name)]

        # Pivot the DataFrame to have scenarios as columns and time as rows
        df_pivot = df.pivot(index='Time', columns='Scenario', values=str(input_name))

        # Rename the columns to 'scen_1', 'scen_2', etc.
        df_pivot.columns = [f"scen_{i+1}" for i in range(len(df_pivot.columns))]

        # Reset the index to have rows as 0, 1, 2, ...
        df_pivot.reset_index(inplace=True)
        df_pivot = df_pivot.drop(columns=['Time'])


        return df_pivot
    
    def _CVaR (self,earnings):
        #CVaR calculation
        earnings = np.array(earnings)
        earnings_sorted =  np.sort(earnings)
        var_threshold_lower = np.percentile(earnings_sorted,(1-self.data.alpha)*100)
        #print(f"Value at Risk (VaR) at {self.data.alpha*100}% confidence level is {var_threshold_lower}")
        tail_losses = earnings_sorted[earnings_sorted <= var_threshold_lower]
        CVaR_loss = np.mean(tail_losses)
        #print(f"Conditional Value at Risk (CVaR) at {self.data.alpha*100}% confidence level is {CVaR_loss}")

        """
        var_threshold_upper = np.percentile(earnings,(self.data.alpha)*100)
        print(f"Value at Risk (VaR) at {self.data.alpha*100}% confidence level is {var_threshold_upper}")
        tail_win = earnings[var_threshold_upper <= earnings]
        CVaR_win = np.mean(tail_win)
        print(f"Conditional Value at Risk (CVaR) at {self.data.alpha*100}% confidence level is {CVaR_win}")
       
        plt.figure(figsize=(10,6))
        plt.hist(earnings, bins=20, alpha=0.5, label="No Contract", color="blue")
        plt.axvline(CVaR_loss, color='green', linestyle='--', label=f'VaR (Left, {100 - self.data.alpha *100:.0f}%)')
        plt.legend()
        plt.show()
         """
       
        return CVaR_loss
    
    def _build_statistics(self):

       # Calculate the average price at each node
        self.data.price_df = self._build_dataframe(self.opf_results.price['N3'],'price') # Node N3 price in dataframe form 
        self.data.price_true = self.data.price_df.values # Node N3 price in dataframe form 

        self.data.price_G6 = (1+self.data.K_G6)*self.data.price_df.values # Node 3 price in numpy array for calculations with added bias if needed '0' is standard
        self.data.price_L2 = (1+self.data.K_L2)*self.data.price_df.values # Node 3 price in numpy array for calculations with added bias if needed '0' is standard

        self.data.generator_production_df = self._build_dataframe(self.opf_results.generator_production['G6'],'generator_production') # Generator G6
        self.data.generator_production = self.data.generator_production_df.values # Generator G6 in numpy array for calculations

        self.data.price_mean_true = self.data.price_true.mean() # Mean price at node N3
        self.data.price_std_true = self.data.price_true.std(ddof=1) # Std price at node N3

        # Make distributions for the price at node N3
        
        self.data.generator_production_df = self._build_dataframe(self.opf_results.generator_production['G6'],'generator_production') # Generator G6
        self.data.generator_production = self.data.generator_production_df.values # Generator G6 in numpy array for calculations

        self.data.price_aggregated = self.data.price_true.sum() # Aggregated price at node N3 (average over scenarios) 

        # Make distributions for the price at node N3
        self.data.pdf_true = stats.norm.pdf(self.data.price_true, self.data.price_mean_true, self.data.price_std_true) # True distribution 
        
        # If assuming they have different observations of prices ( i.e., different K (bias) values)
        pdf_G6 = stats.norm.pdf(self.data.price_G6, self.data.price_G6.mean(), self.data.price_G6.std()) # G6 observed distribution 
        pdf_test = stats.norm.pdf(self.data.price_true, self.data.price_mean_true+0.01, self.data.price_std_true) # L2 observed distribution 
        
        #Make this a call from utility function 
        self.data.net_earnings_no_contract_G6_df = self._build_dataframe(self.opf_results.generator_revenue['G6'],'generator_revenue')
        self.data.net_earnings_no_contract_G6 = self.data.net_earnings_no_contract_G6_df.sum().values # Generator G6 earnings in numpy array for calculations ( Total Sum per Scenario)

        self.data.net_earnings_no_contract_L2_df = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price_df)
        self.data.net_earnings_no_contract_L2 = self.data.net_earnings_no_contract_L2_df.sum().values # Load L2 earnings in numpy array for calculations
        #CVaR calculation of No contract 
        self.data.CVaR_no_contract_G6 = self._CVaR(self.data.net_earnings_no_contract_G6)

        self.data.CVaR_no_contract_L2 = self._CVaR(self.data.net_earnings_no_contract_L2)

        # Threat Point G6
        self.data.Zeta_G6=(1-self.data.A_G6)*self.data.net_earnings_no_contract_G6.mean() + self.data.A_G6*self.data.CVaR_no_contract_G6

        #Threat Point for L2
        self.data.Zeta_L2=(1-self.data.A_G6)*self.data.net_earnings_no_contract_L2.mean()  + self.data.A_L2*self.data.CVaR_no_contract_L2


        #CvaR prices 
        self.data.CVaR_price = self._CVaR(self.data.price_true) # CVaR price at node N3
    

        #SR # Not correct Missing adding E*K

        self.data.SR = np.min([(self.data.price_mean_true + self.data.A_G6*self.data.CVaR_price + (1+self.data.A_G6*self.data.K_G6))/(len(self.data.TIME)*(1+self.data.A_G6)),
                              (self.data.price_mean_true - self.data.A_G6*self.data.CVaR_price + (1+self.data.A_G6*self.data.K_G6))/(len(self.data.TIME)*(1+self.data.A_G6)),
                              (self.data.price_mean_true + (1+self.data.A_L2)*self.data.K_L2 + self.data.A_L2*self.data.CVaR_price)/(len(self.data.TIME)*(1+self.data.A_L2))])

        #SU
        self.data.SU = np.max([(self.data.price_mean_true + self.data.A_G6*self.data.CVaR_price + (1+self.data.A_G6*self.data.K_G6))/(len(self.data.TIME)*(1+self.data.A_G6)),
                              (self.data.price_mean_true - self.data.A_G6*self.data.CVaR_price + (1+self.data.A_G6*self.data.K_G6))/(len(self.data.TIME)*(1+self.data.A_G6)),
                              (self.data.price_mean_true + (1+self.data.A_L2)*self.data.K_L2+ self.data.A_L2*self.data.CVaR_price)/(len(self.data.TIME)*(1+self.data.A_L2))])

        print(self.data.SR)
        print(self.data.SU)
    
    def _build_model(self):
        self.model = gp.Model(name='Nash Bargaining Model')
        self.model.Params.NonConvex = 2
        #self.model.Params.OutputFlag =  0

        self.model.Params.TimeLimit = 30
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        
        self.model.update()
    
    def _build_variables(self):
     

        # Auxiliary variables for logaritmic terms

        self.variables.arg_G6 = self.model.addVar(lb=1e-18, name="UG6_minus_ZetaG6")
        self.variables.arg_L2 = self.model.addVar(lb=1e-18, name="UL2_minus_ZetaL2")

        # Define logarithmic terms
        self.variables.log_arg_G6 = self.model.addVar(name="log_arg_G6")
        self.variables.log_arg_L2 = self.model.addVar(name="log_arg_L2")

           # build strike price variables
        self.variables.S = self.model.addVar(
            lb=0,
            ub=50,
            name='Strike_Price'
        )

        """
        # build contract amount variables
        self.variables.M = self.model.addVar(
            lb=self.data.contract_amount_min,
            ub=self.data.contract_amount_max,
            name='Contract Amount'
        )
        """
        self.variables.zeta_G6 = self.model.addVar(
            name='Zeta_Auxillary_G6',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
        self.variables.zeta_L2 = self.model.addVar(
            name='Zeta_Auxillary_L2',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
      
        self.variables.eta_G6 = {s: self.model.addVar(
            name='Auxillary_Variable_G6_{0}'.format(s),
            lb=0,ub=gp.GRB.INFINITY)
            for s in self.data.SCENARIOS_L
        }
        self.variables.eta_L2 = {s: self.model.addVar(
            name='Auxillary_Variable_L2_{0}'.format(s),
            lb=0,ub=gp.GRB.INFINITY)
            for s in self.data.SCENARIOS_L
        }

        self.model.update() 
       
    def _build_constraints(self):
        # Single Strike Price Constraint
        self.constraints.strike_price_constraint =  self.model.addLConstr(
        self.variables.S<=50,
        name='Strike_Price_Constraint_Max'
    )
        
         #Strike Price min  constraints
        self.constraints.strike_price_constraint_min = self.model.addLConstr(
            0 <= self.variables.S,
            name='Strike_Price_Constraint_Min'
        )

        #Contract amounts 
        """
        #Contract amount constraints
        self.constraints.contract_amount_constraint = self.model.addLConstr(
            self.variables.M,
            gp.GRB.LESS_EQUAL,
            self.data.contract_amount_max,
            name='Contract Amount Constraint Max'
        )
         #Contract amount constraints
        self.constraints.contract_amount_constraint_min = self.model.addLConstr(
            self.variables.M,
            gp.GRB.GREATER_EQUAL,
            self.data.contract_amount_min,
            name='Contract Amount Constraint Min'
        )
        """

        #Logarithmic constraints
        self.model.addGenConstrLog(self.variables.arg_G6, self.variables.log_arg_G6, "log_G6")
        self.model.addGenConstrLog(self.variables.arg_L2, self.variables.log_arg_L2, "log_L2")
        
        #Risk Constraints       
        self.constraints.eta_G6_constraint = {
            s: self.model.addLConstr(
            self.variables.eta_G6[s] >=self.variables.zeta_G6 - gp.quicksum((self.data.price_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s])
                                                                            +( self.variables.S- self.data.price_G6[t,s])*self.data.generator_production[t,s] for t in self.data.TIME),
            name='Eta_Aversion_Constraint_G6_in_scenario_{0}'.format(s)
        )
        for s in self.data.SCENARIOS_L
        }
        self.constraints.eta_L2_constraint = {
            s: self.model.addLConstr(
            self.variables.eta_L2[s] >=self.variables.zeta_L2 - gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price_L2[t,s])
                                                                            +(self.data.price_L2[t,s] - self.variables.S)*self.data.generator_production[t,s] for t in self.data.TIME),
            name='Eta_Aversion_Constraint_L2_in_scenario_{0}'.format(s)
        )
        for s in self.data.SCENARIOS_L
        }

        self.model.update()

    def _build_objective(self):
        
        M = self.data.generator_capacity['G6'] # Baseload amount
        #Utility function for G6 
        EuG6 =  gp.quicksum((self.data.price_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG6 =  gp.quicksum(( self.variables.S- self.data.price_G6[t,s])*M for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #CVaR for G6 

        CVaRG6 = self.variables.zeta_G6 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_G6[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UG6 = (1-self.data.A_G6) * self.data.PROB*(EuG6+SMG6) + self.data.A_G6*CVaRG6 # This bad boi is the issue


        #Utility function for L2
        EuL2 = gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price_L2[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML2 =  gp.quicksum((self.data.price_L2[t,s] - self.variables.S)*M for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #CvaR for L2
        CVaRL2 = self.variables.zeta_L2 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_L2[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UL2 = (1-self.data.A_L2)*self.data.PROB*(EuL2+ SML2) + self.data.A_L2*CVaRL2

         # Link auxiliary variables to expressions
        self.model.addLConstr(self.variables.arg_G6 == UG6 - self.data.Zeta_G6, "arg_G6_constr")
        self.model.addLConstr(self.variables.arg_L2 == UL2 - self.data.Zeta_L2, "arg_L2_constr")

        self.model.update()

        # Normal Objective function 
        #objective = (UG6 - self.data.Zeta_G6) * (UL2-self.data.Zeta_L2)
        #self.model.setObjective(objective,sense = gp.GRB.MAXIMIZE)

        # Set logarithmic objective
        self.model.setObjective(
            self.variables.log_arg_G6 + self.variables.log_arg_L2,
        GRB.MAXIMIZE) 
        self.model.update()

    def _save_results(self):

        #Plot income distributions for generator and load with contract and without contract

        # save objective value
        self.results.objective_value = self.model.ObjVal
        # save strike price
        self.results.strike_price = self.variables.S.x

        print(self.results.strike_price)

        #Temporary save for contract amount
        self.results.contract_amount = 300 # not actual 
        # save utility for G6
        EuG6 = self.data.net_earnings_no_contract_G6
        
        SMG6 = {(t, s): ( self.results.strike_price-self.data.price_true[t,s]) * self.results.contract_amount
        for t in self.data.TIME for s in self.data.SCENARIOS_L}

        SMG6_df = self._build_dataframe(SMG6,'G6_revenue')
        SMG6_scen = SMG6_df.sum(axis=0)  # Sum across all time periods for each scenario
        # Calcuate CVar for G6
        self.results.CVaRG6 = self._CVaR(EuG6+SMG6_scen.values)
        #Save Utility 
        self.results.utility_G6 = (1-self.data.A_G6)*(EuG6+SMG6_scen.values).mean() + self.data.A_G6*self.results.CVaRG6
      
        # Repeat the same process for L2 revenue
        EuL2 = self.data.net_earnings_no_contract_L2
        SML2 = {(t, s): (self.data.price_true[t, s] - self.results.strike_price) * self.results.contract_amount
                for t in self.data.TIME for s in self.data.SCENARIOS_L}

        SML2_df = self._build_dataframe(SML2,'L2_revenue')
        SML2_scen = SML2_df.sum(axis=0)

        self.results.CVaRL2 = self._CVaR(EuL2+SML2_scen.values)

        utility_L2 = (1-self.data.A_L2)*(EuL2+SML2_scen.values).mean() + self.data.A_L2*self.results.CVaRL2
        #Save Utility
        self.results.utility_L2 = utility_L2

        # --- Calculate accumulated revenue for each scenario ---
        # Accumulated revenue for G6
        accumulated_revenue_G6 = SMG6_df.sum(axis=0)  # Sum across all time periods for each scenario
        self.results.accumulated_revenue_G6 =EuG6+ accumulated_revenue_G6

        # Accumulated revenue for L2
        accumulated_revenue_L2 = SML2_df.sum(axis=0)  # Sum across all time periods for each scenario
        self.results.accumulated_revenue_L2 =EuL2+ accumulated_revenue_L2
    def display_results(self):
        print("-------------------   RESULTS GUROBI  -------------------")
        print(f"Optimal Objective Value (Log): {self.results.objective_value}")
        print(f"Optimal Objective Value : {np.exp(self.results.objective_value)}")

        print(f"Optimal Strike Price: {self.results.strike_price}")
        print(f"Optimal Contract Amount: {self.results.contract_amount}")
        print(f"Optimal Utility for G6: {self.results.utility_G6}")
        print(f"Optimal Utility for L2: {self.results.utility_L2}")
        print(f"Treat Point G6: {self.data.Zeta_G6}")
        print(f"Treat Point L2: {self.data.Zeta_L2}")

    def manual_optimization(self,plot=False,filename = None):
        strike_prices = np.linspace(18,20, 1000)  # Range of strike prices to evaluate
        contract_amounts = np.linspace(self.data.contract_amount_min, self.data.contract_amount_max, 100)
        M = 300

        # Store utilities for each strike price
        utilities_G6 = []
        utilities_L2 = []
        combined_utilities = []  # To store the combined utility for the Nash product
        combined_utilities_log = []

        #Initialize special case values (In case of negative log values)
        self.data.special_case_min_strike = 1000
        self.data.special_case_max_strike = 0

        # Iterate over strike prices
        for strike in tqdm(strike_prices, desc='loading...'):
            # Create dictionaries to store revenues for G6 and L2
            SMG6 = {(t, s): (strike - self.data.price_G6[t, s]) * M for t in self.data.TIME for s in self.data.SCENARIOS_L}
            SML2 = {(t, s): (self.data.price_L2[t, s] - strike) * M for t in self.data.TIME for s in self.data.SCENARIOS_L}

            # Create DataFrames for G6 and L2 revenues
            SMG6_df = self._build_dataframe(SMG6, 'G6_revenue')
            SML2_df = self._build_dataframe(SML2, 'L2_revenue')

            # Calculate total revenues for G6 and L2 across all scenarios
            Scen_revenue_G6 = SMG6_df.sum() + self.data.net_earnings_no_contract_G6
            Scen_revenue_L2 = SML2_df.sum() + self.data.net_earnings_no_contract_L2

            # Calculate CVaR for the worst-case scenarios
            CVaRG6 = self._CVaR(Scen_revenue_G6.values)
            CVaRL2 = self._CVaR(Scen_revenue_L2.values)

            # Calculate utility for G6
            UG6 = (1 - self.data.A_G6) * Scen_revenue_G6.mean() + self.data.A_G6 * CVaRG6
            utilities_G6.append(UG6)

            # Calculate utility for L2
            UL2 = (1 - self.data.A_L2) * Scen_revenue_L2.mean() + self.data.A_L2 * CVaRL2
            utilities_L2.append(UL2)

            # Calculate the Nash product (combined utility)
            combined_utility = (UG6 - self.data.Zeta_G6 ) * (UL2 - self.data.Zeta_L2) 
            # Calculate the Nash product (combined utility)
            combined_utility_log = np.log(np.maximum(UG6 - self.data.Zeta_G6 , 1e-10)) + np.log(np.maximum(UL2 - self.data.Zeta_L2 , 1e-10)) # Avoid log(0) by using a small value
            combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
            """
            if combined_utility_log <0:
                log_abs_diff_G6 = np.log(np.maximum(np.abs(UG6 - self.data.Zeta_G6), 1e-10))
                log_abs_diff_L2 = np.log(np.maximum(np.abs(UL2 - self.data.Zeta_L2), 1e-10))
                # Sum the logarithms
                combined_utility_log = (log_abs_diff_G6 + log_abs_diff_L2)

                if combined_utility_log -combined_utility <1e-6 and strike < self.data.special_case_min_strike:
                    self.data.special_case_min_strike = strike
                if combined_utility_log - combined_utility <1e-6 and strike > self.data.special_case_max_strike:
                    self.data.special_case_max_strike = strike
            """
            combined_utilities.append(combined_utility)
            combined_utilities_log.append(combined_utility_log)

        # Find the strike price that maximizes the Nash product
        max_combined_utility_index = np.argmax(combined_utilities)
        max_combined_utility_index_log = np.argmax(combined_utilities_log)
        optimal_strike_price = strike_prices[max_combined_utility_index]
        optimal_strike_price_log = strike_prices[max_combined_utility_index_log]


        # Include threat points for comparison
        print("\n-------------------   RESULTS Iterative  -------------------")
        print(f"Threat Point G6: {self.data.Zeta_G6}")
        print(f"Threat Point L2: {self.data.Zeta_L2}")
        print(f"Optimal Strike Price: {optimal_strike_price}")
        print(f"Optimal Strike Price log: {optimal_strike_price_log}")

        print(f"Maximum Utility G6: {utilities_G6[max_combined_utility_index]}")
        print(f"Maximum Utility L2: {utilities_L2[max_combined_utility_index]}")

        print(f"Maximum Combined Utility (Nash Product): {combined_utilities[max_combined_utility_index]}")
        #print(f"Maximum Combined Utility Log (Nash Product): {np.exp(combined_utilities_log[max_combined_utility_index_log])}")
        print(f"Maximum Combined Utility Log (Nash Product): {combined_utilities_log[max_combined_utility_index_log]}")

        if plot == True:
            print("Plotting results...")
            # Plot the utilities and Nash product for different strike prices
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))

            # First subplot: Utilities for G6 and L2
            axs[0].plot(strike_prices, utilities_G6, label="Utility G6", color="blue")
            axs[0].plot(strike_prices, utilities_L2, label="Utility L2", color="green")
            axs[0].axvline(optimal_strike_price, color="red", linestyle="--", label=f"Optimal Strike Price: {optimal_strike_price:.5f}")
            axs[0].axvline(optimal_strike_price, color="red", linestyle="--", label=f"Optimal Log Strike Price: {optimal_strike_price_log:.5f}")
            axs[0].set_xlabel("Strike Price")
            axs[0].set_ylabel("Utility")
            axs[0].set_title(f"Utilities for Different Strike Prices")
            axs[0].legend()
            axs[0].grid()

            # Second subplot: Nash Product and Log Nash Product
            axs[1].plot(strike_prices, combined_utilities, label="Nash Product", color="green", linestyle="--")
            axs[1].plot(strike_prices, np.exp(combined_utilities_log), label="Nash Product Log", color="purple", linestyle="--")
            axs[1].axvline(optimal_strike_price, color="red", linestyle="--", label=f"Optimal Strike Price: {optimal_strike_price:.5f}")
            axs[1].axvline(optimal_strike_price_log, color="orange", linestyle="--", label=f"Optimal Log Strike Price: {optimal_strike_price_log:.5f}")
            axs[1].set_xlabel("Strike Price")
            axs[1].set_ylabel("Nash Product")
            axs[1].set_title("Nash Product and Log Nash Product for Different Strike Prices")
            axs[1].legend()
            axs[1].grid()


            # Second subplot: Nash Product and Log Nash Product
            axs[2].plot(utilities_G6, utilities_L2,  color="orange", linestyle="-")
            axs[2].set_xlabel("Utility G6")
            axs[2].set_ylabel("Utility L2")
            axs[2].set_title("Utility G6 vs Utility L2")
            axs[2].grid()
            fig.suptitle(f"Utilities and Nash Product for Different Strike Prices with with Risk AG{self.data.A_G6} and Risk AL{self.data.A_L2}")

            # Adjust layout and show the plot
            plt.tight_layout()
            if filename:
                plt.savefig(filename)
                print(f"Histogram plot saved to {filename}")
                plt.close(fig) # Close the figure after saving
            else:
                plt.show()

        return optimal_strike_price

    def plot_barter_set(self, case: str = 'fig2', filename: str = None):
        """
        Calculates and plots the Utility Possibility Set (UPS) boundaries
        and the resulting barter set based on the specified case (fig2, fig3, fig4).
        Uses data stored within the ContractNegotiation instance (self.data, self.opf_results).

        Args:
            case (str): Specifies which figure/condition to simulate ('fig2', 'fig3', 'fig4').
            filename (str, optional): If provided, saves the plot to this file. Defaults to None.
        """
        print(f"\n--- Generating Barter Set Plot (Method) for Case: {case} ---")


        pi0_G = self.data.net_earnings_no_contract_G6
        pi0_L = self.data.net_earnings_no_contract_L2

        # Reshape LMPs
      

        lambda_sum_scen = np.sum(self.data.price_true, axis=0)

        # Parameters from self.data
        AG = self.data.A_G6
        AL = self.data.A_L2
        MR = self.data.contract_amount_min
        MU = self.data.contract_amount_max
        SR = self.data.strikeprice_min
        SU = self.data.strikeprice_max
        T_hours = len(self.data.TIME)

        # Threat points from self.data (calculated in _build_statistics)
        Zeta_G = self.data.Zeta_G6
        Zeta_L = self.data.Zeta_L2

        if np.isnan(Zeta_G) or np.isnan(Zeta_L):
             print("Error: Threat points (Zeta_G, Zeta_L) are NaN. Cannot plot barter set.")
             return

        # --- Calculate UPS Boundary Curves (V1 and V2) ---
        # Based on Lemma 3: Vary M for fixed S=SR and S=SU
        num_m_points_lemma = 100
        M_values_lemma = np.linspace(MR, MU, num_m_points_lemma)
        UG_V1_lemma, UL_V1_lemma = [], []
        UG_V2_lemma, UL_V2_lemma = [], []

        for M in M_values_lemma:
             # V1: S = SR
            piG_SR_scen = pi0_G + (SR * T_hours - lambda_sum_scen) * M
            piL_SR_scen = pi0_L + (lambda_sum_scen - SR * T_hours) * M
            mean_piG_SR = np.mean(piG_SR_scen)
            mean_piL_SR = np.mean(piL_SR_scen)
            cvar_piG_SR = self._CVaR(piG_SR_scen)
            cvar_piL_SR = self._CVaR(piL_SR_scen)
            UG_V1_lemma.append((1 - AG) * mean_piG_SR + AG * cvar_piG_SR)
            UL_V1_lemma.append((1 - AL) * mean_piL_SR + AL * cvar_piL_SR)

            # V2: S = SU
            piG_SU_scen = pi0_G + (SU * T_hours - lambda_sum_scen) * M
            piL_SU_scen = pi0_L + (lambda_sum_scen - SU * T_hours) * M
            mean_piG_SU = np.mean(piG_SU_scen)
            mean_piL_SU = np.mean(piL_SU_scen)
            cvar_piG_SU = self._CVaR(piG_SU_scen)
            cvar_piL_SU = self._CVaR(piL_SU_scen)
            UG_V2_lemma.append((1 - AG) * mean_piG_SU + AG * cvar_piG_SU)
            UL_V2_lemma.append((1 - AL) * mean_piL_SU + AL * cvar_piL_SU)

        UG_V1_lemma, UL_V1_lemma = np.array(UG_V1_lemma), np.array(UL_V1_lemma)
        UG_V2_lemma, UL_V2_lemma = np.array(UG_V2_lemma), np.array(UL_V2_lemma)
        # Sort V1/V2 points for plotting
        sort_idx_v1 = np.argsort(UG_V1_lemma)
        sort_idx_v2 = np.argsort(UG_V2_lemma)
        UG_V1_plot, UL_V1_plot = UG_V1_lemma[sort_idx_v1], UL_V1_lemma[sort_idx_v1]
        UG_V2_plot, UL_V2_plot = UG_V2_lemma[sort_idx_v2], UL_V2_lemma[sort_idx_v2]

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.plot(UG_V1_plot, UL_V1_plot, 'b-', label='$V_1 (S=S_R)$')
        ax.plot(UG_V2_plot, UL_V2_plot, 'g-', label='$V_2 (S=S_U)$')
        ax.plot(Zeta_G, Zeta_L, 'ro', markersize=8, label='Threat Point $(\zeta_G, \zeta_L)$')

        # --- Determine and plot the barter set based on the case ---
        mask_V1 = (UG_V1_plot >= Zeta_G) & (UL_V1_plot >= Zeta_L)
        mask_V2 = (UG_V2_plot >= Zeta_G) & (UL_V2_plot >= Zeta_L)
        UG_V1_barter = UG_V1_plot[mask_V1]
        UL_V1_barter = UL_V1_plot[mask_V1]
        UG_V2_barter = UG_V2_plot[mask_V2]
        UL_V2_barter = UL_V2_plot[mask_V2]

        # Helper needs to be nested or passed self if it uses instance data/methods
        def find_intersection(ug_curve, ul_curve, zeta_val, axis='G'):
            # Same implementation as before
            if axis == 'G':
                indices = np.where(np.diff(np.sign(ug_curve - zeta_val)))[0]
            else:
                indices = np.where(np.diff(np.sign(ul_curve - zeta_val)))[0]

            if len(indices) > 0:
                idx = indices[0]
                if idx + 1 < len(ug_curve):
                    if axis == 'G':
                        u1, u2 = ug_curve[idx], ug_curve[idx + 1]
                        v1, v2 = ul_curve[idx], ul_curve[idx + 1]
                        if (u2 - u1) == 0: return zeta_val, v1
                        interp_ul = v1 + (v2 - v1) * (zeta_val - u1) / (u2 - u1)
                        return zeta_val, interp_ul
                    else:
                        u1, u2 = ul_curve[idx], ul_curve[idx + 1]
                        v1, v2 = ug_curve[idx], ug_curve[idx + 1]
                        if (u2 - u1) == 0: return v1, zeta_val
                        interp_ug = v1 + (v2 - v1) * (zeta_val - u1) / (u2 - u1)
                        return interp_ug, zeta_val
            if len(ug_curve)>0:
                 if axis == 'G':
                     valid_idx = np.where(ug_curve >= zeta_val)[0]
                     if len(valid_idx)>0: return zeta_val, ul_curve[valid_idx[0]]
                 else:
                     valid_idx = np.where(ul_curve >= zeta_val)[0]
                     if len(valid_idx)>0: return ug_curve[valid_idx[0]], zeta_val
            return None

        # Plot barter set logic (same as before, using nested find_intersection)
        if case == 'fig2':
            title = 'Barter Set (Fig 2: Conditions 29 & 30 Hold)'
            point_I = find_intersection(UG_V1_barter, UL_V1_barter, Zeta_G, axis='G')
            point_J = find_intersection(UG_V1_barter, UL_V1_barter, Zeta_L, axis='L')
            if point_I and point_J and point_I[1] >= Zeta_L and point_J[0] >= Zeta_G:
                vertices_corrected = [ (Zeta_G, Zeta_L), (Zeta_G, point_I[1]), (point_J[0], Zeta_L) ]
                polygon = Polygon(vertices_corrected, closed=True, color='lightcoral', alpha=0.5, label='Barter Set B')
                ax.add_patch(polygon)
                print("Barter set (Triangle) plotted for Fig 2.")
            else:
                 print("Could not determine valid vertices for Fig 2 barter set.")
                 ax.annotate('Barter Set B\n(Region $U \\geq \zeta$)', xy=(Zeta_G, Zeta_L), xytext=(Zeta_G + abs(Zeta_G*0.01), Zeta_L + abs(Zeta_L*0.01)), arrowprops=dict(facecolor='black', arrowstyle='->'))

        elif case == 'fig3':
            title = 'Barter Set (Fig 3: Condition 30 Fails)'
            ax.annotate('Barter Set B = {Threat Point}', xy=(Zeta_G, Zeta_L), xytext=(Zeta_G + abs(Zeta_G*0.01), Zeta_L + abs(Zeta_L*0.01) ), arrowprops=dict(facecolor='black', arrowstyle='->'))
            print("Barter set (Threat Point) indicated for Fig 3.")

        elif case == 'fig4':
            title = 'Barter Set (Fig 4: Condition 29 Fails)'
            point_I = find_intersection(UG_V2_barter, UL_V2_barter, Zeta_G, axis='G')
            point_J = find_intersection(UG_V2_barter, UL_V2_barter, Zeta_L, axis='L')
            if point_I and point_J and point_I[1] >= Zeta_L and point_J[0] >= Zeta_G:
                vertices_corrected = [ (Zeta_G, Zeta_L), (Zeta_G, point_I[1]), (point_J[0], Zeta_L) ]
                polygon = Polygon(vertices_corrected, closed=True, color='lightcoral', alpha=0.5, label='Barter Set B')
                ax.add_patch(polygon)
                print("Barter set (Triangle) plotted for Fig 4.")
            else:
                 print("Could not determine valid vertices for Fig 4 barter set.")
                 ax.annotate('Barter Set B\n(Region $U \\geq \zeta$)', xy=(Zeta_G, Zeta_L), xytext=(Zeta_G + abs(Zeta_G*0.01), Zeta_L + abs(Zeta_L*0.01)), arrowprops=dict(facecolor='black', arrowstyle='->'))
        else:
            title = 'Utility Possibility Set'
            print(f"Unknown case: {case}. Plotting UPS boundaries only.")

        # Final plot adjustments (same as before)
        ax.set_xlabel('GenCo Utility ($U_G$)')
        ax.set_ylabel('LSE Utility ($U_L$)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        all_ug = np.concatenate((UG_V1_plot, UG_V2_plot, [Zeta_G]))
        all_ul = np.concatenate((UL_V1_plot, UL_V2_plot, [Zeta_L]))
        min_ug, max_ug = np.nanmin(all_ug), np.nanmax(all_ug)
        min_ul, max_ul = np.nanmin(all_ul), np.nanmax(all_ul)
        if not (np.isnan(min_ug) or np.isnan(max_ug) or np.isnan(min_ul) or np.isnan(max_ul)):
            ug_range = max_ug - min_ug if max_ug > min_ug else 1.0
            ul_range = max_ul - min_ul if max_ul > min_ul else 1.0
            ax.set_xlim(min_ug - ug_range * 0.1, max_ug + ug_range * 0.1)
            ax.set_ylim(min_ul - ul_range * 0.1, max_ul + ul_range * 0.1)
        else:
             pass #print("Warning: Could not determine plot limits due to NaN values.") # Reduced verbosity

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Barter set plot saved to {filename}")
            plt.close(fig)
        else:
            plt.show()

    def check_barter_conditions(self):
        """
        Checks if conditions (29) and (30) from the paper hold.

        These conditions determine the expected shape of the barter set.
        Condition (29) holding implies the upper boundary is potentially non-trivial.
        Condition (30) holding implies the lower boundary is potentially non-trivial.

        Returns:
            tuple: A tuple (bool | None, bool | None) indicating
                   (condition_29_holds, condition_30_holds).
                   Returns None for a condition if it cannot be calculated due to NaN values.
        """
        print("\n--- Checking Barter Set Conditions (29 & 30) ---")

        # --- 1. Retrieve Necessary Parameters and Data ---
        try:
            AG = self.data.A_G6
            AL = self.data.A_L2
            T_val = len(self.data.TIME)
            KG_abs = self.data.K_G6 # Absolute bias shift
            KL_abs = self.data.K_L2 # Absolute bias shift
            EP_lambda_sum_true = self.data.price_true.sum(axis=0).mean() # Mean of the sum of LMPs across all scenarios
            # CVaR_neg_lam_true corresponds to CVaRαP(-λΣ) in the paper's notation
            CVaR_neg_lam_true = self._CVaR(self.data.price_true.sum(axis=1)) # CVaR of the sum of LMPs across all scenarios
            MR = self.data.contract_amount_min
            MU = self.data.contract_amount_max
            SR = self.data.strikeprice_min # Use the lower bound strike price SR

            # No-contract earnings and perceived lambda_sum for LSE
            pi0_L = self.opf_results.net_earnings_no_contract_L2
            lambda_sum_L2_scen = self.data.price_true.sum(axis=0) # Sum across all time periods for each scenario

        except AttributeError as e:
            print(f"Error accessing necessary data: {e}. Cannot check conditions.")
            return (None, None)

        # --- 2. Check for Invalid Inputs or Potential Division by Zero ---
        required_values = [AG, AL, T_val, KG_abs, KL_abs, EP_lambda_sum_true, CVaR_neg_lam_true, MR, MU, SR]
        if any(np.isnan(v) for v in required_values) or T_val == 0:
            print("Warning: NaN values found in required parameters or T_val is zero. Cannot check conditions.")
            return (None, None)
        if AL == 0:
            print("Warning: AL is zero. Cannot check conditions due to division by zero.")
            return (None, None)
        if (1 + AG) == 0:
             print("Warning: (1 + AG) is zero. Cannot check conditions due to division by zero.")
             return (None, None)


        # --- 3. Calculate the Right-Hand Side (RHS) Threshold ---
        # RHS = [ (AG - AL) / (AL * (1 + AG)) ] * EP(λΣ)
        try:
            rhs_threshold = (AG - AL) / (AL * (1 + AG)) * EP_lambda_sum_true
        except ZeroDivisionError:
             print("Warning: Division by zero calculating RHS threshold. Cannot check conditions.")
             return (None, None)


        # --- 4. Numerically Estimate dCVaRL(-πL)/dM using Finite Difference ---

        delta_M = 1e-6 # Small change in M for finite difference

        # Helper function to calculate CVaR[-pi_L(M, S)] for LSE
        def calculate_cvar_neg_piL(M, S):
            if np.isnan(M) or np.isnan(S): return np.nan
            # Use LSE's perceived lambda_sum
            neg_piL_scen = - (pi0_L + (lambda_sum_L2_scen - S * T_val) * M)
            return self._CVaR(neg_piL_scen) # Use the class's CVaR method

        # Estimate derivative at M = MU, S = SR (for condition 29)
        cvar_at_mu = calculate_cvar_neg_piL(MU, SR)
        cvar_at_mu_plus_delta = calculate_cvar_neg_piL(MU + delta_M, SR)
        if np.isnan(cvar_at_mu) or np.isnan(cvar_at_mu_plus_delta) or delta_M == 0:
             deriv_at_mu = np.nan
             print("Warning: Could not calculate derivative at MU due to NaN CVaR.")
        else:
            deriv_at_mu = (cvar_at_mu_plus_delta - cvar_at_mu) / delta_M

        # Estimate derivative at M = MR, S = SR (for condition 30)
        cvar_at_mr = calculate_cvar_neg_piL(MR, SR)
        # Need point slightly *above* MR for forward difference if MR=0
        mr_eval_point = MR if MR > 0 else delta_M/10 # Use tiny M if MR is 0
        cvar_at_mr_plus_delta = calculate_cvar_neg_piL(mr_eval_point + delta_M, SR)
        # Adjust derivative calculation if MR was 0
        actual_delta_M_mr = (mr_eval_point + delta_M) - mr_eval_point

        if np.isnan(cvar_at_mr) or np.isnan(cvar_at_mr_plus_delta) or actual_delta_M_mr == 0:
             deriv_at_mr = np.nan
             print("Warning: Could not calculate derivative at MR due to NaN CVaR.")
        else:
             # Need CVaR at the actual MR point for the difference calculation
             cvar_at_actual_mr = calculate_cvar_neg_piL(MR, SR) # Recalc just in case mr_eval_point != MR
             if np.isnan(cvar_at_actual_mr):
                 deriv_at_mr = np.nan
                 print("Warning: Could not calculate derivative at MR due to NaN CVaR at MR.")
             else:
                 deriv_at_mr = (cvar_at_mr_plus_delta - cvar_at_actual_mr) / actual_delta_M_mr


        # --- 5. Calculate the Common Offset Term from LHS ---
        # Offset = [ AG(1+AL) / (AL(1+AG)) ]*CVaR(-λΣ) + (1/AL)*KL - [(1+AL)/AL]*KG + T*SR
        # Note: Using CVaR_neg_lam_true which is CVaRαP(-λΣ)
        try:
            term_cvar = (AG * (1 + AL)) / (AL * (1 + AG)) * CVaR_neg_lam_true
            term_bias = (1 / AL) * KL_abs - ((1 + AL) / AL) * KG_abs
            term_strike = T_val * SR
            common_offset_term = term_cvar + term_bias + term_strike
        except (ZeroDivisionError, TypeError, ValueError) as e:
             print(f"Warning: Could not calculate common offset term: {e}. Cannot check conditions.")
             return (None, None)


        # --- 6. Check Conditions ---
        cond29_holds = None
        if not np.isnan(deriv_at_mu) and not np.isnan(common_offset_term) and not np.isnan(rhs_threshold):
            # Condition 29: dCVaR/dM(MU,SR) > RHS + Offset_term_from_LHS (rearranged from paper)
            lhs_val_29 = deriv_at_mu
            rhs_val_29 = rhs_threshold + common_offset_term # Rearranged: d/dM > Thr + Off
            cond29_holds = lhs_val_29 > rhs_val_29
            print(f"Condition (29): dCVaR/dM|MU = {lhs_val_29:.4f} > Threshold+Offset = {rhs_val_29:.4f} ? -> {cond29_holds}")
        else:
            print("Condition (29): Cannot evaluate due to NaN values.")


        cond30_holds = None
        if not np.isnan(deriv_at_mr) and not np.isnan(common_offset_term) and not np.isnan(rhs_threshold):
            # Condition 30: dCVaR/dM(MR,SR) < RHS - Offset_term_from_LHS (rearranged from paper)
             # Careful rearrangement: paper is dCVaR/dM + Offset < Threshold
            lhs_val_30 = deriv_at_mr
            # The offset term structure in (30) seems added in the paper's eq.
            # Let's check the image again.
            # Eq 30: dCVaR/dM + [AG(1+AL)/...]CVaR(-λΣ) + [1/AL]KL - [...]KG + TS < (AG-AL)/[...]E(λΣ)
            # So: lhs_cond30 = deriv_at_mr + common_offset_term
            lhs_val_30 = deriv_at_mr + common_offset_term
            rhs_val_30 = rhs_threshold # The threshold itself
            cond30_holds = lhs_val_30 < rhs_val_30
            print(f"Condition (30): dCVaR/dM|MR + Offset = {lhs_val_30:.4f} < Threshold = {rhs_val_30:.4f} ? -> {cond30_holds}")
        else:
            print("Condition (30): Cannot evaluate due to NaN values.")


        return (cond29_holds, cond30_holds)
    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            #Reset strike price to original values
            self.data.strikeprice_min = self.data.original_strike_min
            self.data.strikeprice_max = self.data.original_strike_max
            self._save_results()
        else:
            #self.model.write("model.lp")
            #self.model.write("model.rlp")
            #self.model.write("model.mps")      
            """
            print(f"optimization of {self.model.ModelName} was not successful - Retrying with optimal bounds found through manual Optimization")
            self.manual_optimization(plot=True,filename = None)
            print(f"Retrying optimization of {self.model.ModelName} with new minimum {self.data.special_case_min_strike} and maximum {self.data.special_case_max_strike} strike prices")
            #Update values to special case values - Bounds should be feasible now 
            self.data.strikeprice_min = self.data.special_case_min_strike
            self.data.strikeprice_max = self.data.special_case_max_strike
            self._build_model()
            print("yaaaaaaaaaaaaaaaaaaaaaaaa",1)
            self.model.optimize()
            self._save_results()       
            """     
              
class Plotting_Class():
    """
    Handles plotting of results from the power system contract negotiation simulation.
    """

    def __init__(self,risk_sensitivity_df, earnings_risk_sensitivity_df,bias_sensitivity_df ,styles=None):
        """
        Initialize the visualizer. Optionally set default styles.
        Args:
            styles (dict, optional): Dictionary defining default plotting styles.
                                      Defaults to None.
        """
        self.risk_sensitivity_df = risk_sensitivity_df
        self.earnings_risk_sensitivity_df = earnings_risk_sensitivity_df
        self.bias_sensitivity_df = bias_sensitivity_df
        self.styles = styles if styles else {}
        # Optional styles for sns
        # sns.set_theme(style="whitegrid")
    
    def _plot_sensitivity_results(self,filename=None):
        """
        Plots the sensitivity analysis results using heatmaps.

        Args:
            results_df (pd.DataFrame): DataFrame from run_sensitivity_analysis.
        """
        if self.risk_sensitivity_df.empty:
            print("No results to plot.")
            return
            
        # Metrics to plot
        metrics = ['StrikePrice', 'Utility_G6', 'Utility_L2', 'NashProduct']
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(2, num_metrics // 2, figsize=(6 * num_metrics, 5))  # Adjust figsize as needed

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        if num_metrics == 1:  # Handle case with only one metric
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]  # Access the correct subplot
            try:
                # Pivot table for heatmap: Index=A_L2, Columns=A_G6, Values=Metric
                pivot_table = self.risk_sensitivity_df.pivot(index='A_L2', columns='A_G6', values=metric)

                # Use scientific notation only for NashProduct
                fmt = ".2f" if metric == "StrikePrice" else ".1e"

                sns.heatmap(pivot_table, ax=ax, annot=True, fmt=fmt, cmap="viridis", cbar=True)
                ax.set_title(f'{metric} vs. Risk Aversion')
                ax.set_xlabel('Risk Aversion Generator (A_G6)')
                ax.set_ylabel('Risk Aversion Load (A_L2)')
                # Ensure axis labels match the data ranges
                ax.invert_yaxis()  # Often conventional for heatmaps derived from matrices
            except Exception as e:
                print(f"Could not plot heatmap for {metric}: {e}")
                ax.set_title(f'{metric} (Plotting Error)')

        # Adjust layout and save/show the plot
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Histogram plot saved to {filename}")
            plt.close(fig)  # Close the figure after saving
        else:
            plt.show()

        print("Plots generated successfully.")

    def _plot_earnings_histograms(self, fixed_A_G6, A_L2_to_plot,filename=None):
        """
        Plots histograms of G6 and L2 net earnings for a fixed A_G6
        and varying A_L2 values.

        Args:
            results_df (pd.DataFrame): DataFrame containing earnings distributions.
            fixed_A_G6 (float): The value of A_G6 to hold constant.
            A_L2_to_plot (list): A list of A_L2 values to plot histograms for.
            filename (str, optional): If provided, saves the plot to this file. Defaults to None (shows plot).
        """
        filtered_results = pd.concat([df[(df['A_G6'] == fixed_A_G6) & (df['A_L2'].isin(A_L2_to_plot)) & 
       (~df['Revenue_G6'].isna()) & (~df['Revenue_L2'].isna())]
        for df in self.earnings_risk_sensitivity_df if isinstance(df, pd.DataFrame) and not df.empty and 'A_G6' in df.columns and 'A_L2' in df.columns], ignore_index=True)
        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- G6 Revenue Histogram ---
        ax_g6 = axes[0]
        for a_l2 in A_L2_to_plot:
            g6_revenue = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_G6'].values
            ax_g6.hist(g6_revenue / 1e5, bins=20, alpha=0.6, label=f'A_L={a_l2}', density=False)  # Scale to millions

        ax_g6.set_title(f'GenCo (G6) Accumulated Revenue Distribution (A_G6 = {fixed_A_G6})')
        ax_g6.set_xlabel('GenCo Revenue ($ x 10^5$)')
        ax_g6.set_ylabel('Frequency')
        ax_g6.legend()
        ax_g6.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # --- L2 Revenue Histogram ---
        ax_l2 = axes[1]
        for a_l2 in A_L2_to_plot:
            l2_revenue = filtered_results[filtered_results['A_L2'] == a_l2]['Revenue_L2'].values
            ax_l2.hist(l2_revenue / 1e5, bins=20, alpha=0.6, label=f'A_L={a_l2}', density=False)  # Scale to millions

        ax_l2.set_title(f'LSE (L2) Accumulated Revenue Distribution (A_G6 = {fixed_A_G6})')
        ax_l2.set_xlabel('LSE Revenue ($ x 10^5$)')
        ax_l2.set_ylabel('Frequency')
        ax_l2.legend()
        ax_l2.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Histogram plot saved to {filename}")
            plt.close(fig)  # Close the figure after saving
        else:
            plt.show()
    def _plot_bias_sensitivity(self,filename=None):
        """
        Plots the sensitivity analysis results using heatmaps.

        Args:
            results_df (pd.DataFrame): DataFrame from run_sensitivity_analysis.
        """
       
        if self.bias_sensitivity_df.empty:
            print("No results to plot.")
            return
            
        # Metrics to plot
        metrics = ['StrikePrice']
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5)) # Adjust figsize as needed
        
        if num_metrics == 1: # Handle case with only one metric
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            try:
            # Pivot table for heatmap: Index=A_L2, Columns=A_G6, Values=Metric
                pivot_table = self.bias_sensitivity_df.pivot(index='KG_Factor', columns='KL_Factor', values=metric)
                
                
                sns.heatmap(pivot_table, ax=ax, annot=True, cmap="viridis",fmt='.3f' ,cbar=True )
                ax.set_title(f'{metric} vs. Price Bias')
                ax.set_xlabel('Price Bias Generator (K_G6)')
                ax.set_ylabel('Price Bias Load (K_L2)')
                # Ensure axis labels match the data ranges
                ax.invert_yaxis()  # Often conventional for heatmaps derived from matrices
            except Exception as e:
                print(f"Could not plot heatmap for {metric}: {e}")
                ax.set_title(f'{metric} (Plotting Error)')

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Histogram plot saved to {filename}")
            plt.close(fig) # Close the figure after saving
        else:
            plt.show()

# ----- NEW Function to run the sensitivity analysis -----
def run_risk_sensitivity_analysis(input_data_base: InputData, opf_results:Expando, A_G6_values:int, A_L2_values:int):
    """
    Runs the ContractNegotiation for different combinations of A_G6 and A_L2.

    Args:
        input_data_template: An instance of InputData with base parameters.
        opf_results: Results from a single OptimalPowerFlow run.
        A_G6_values: List or array of A_G6 values to test.
        A_L2_values: List or array of A_L2 values to test.

    Returns:
        pandas.DataFrame: DataFrame containing results for each parameter combination.
    """
    results_list = []
    earnings_list_scenarios = []
    current_input_data = input_data_template # Start with the template
    for a_g6 in tqdm(A_G6_values, desc="Iterating A_G6"):
        for a_l2 in tqdm(A_L2_values, desc="Iterating A_L2", leave=False):

            
            current_input_data.A_G6 = a_g6
            current_input_data.A_L2 = a_l2
            
            # --- Run Contract Negotiation ---
            try:
                # Instantiate ContractNegotiation with updated input data
                contract_model = ContractNegotiation(current_input_data, opf_results)
                contract_model.run() # Run the Gurobi optimization
                
                # --- Store Results ---
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': contract_model.results.strike_price,
                    'Utility_G6': contract_model.results.utility_G6,
                    'Utility_L2': contract_model.results.utility_L2,
                    'NashProductLog': contract_model.results.objective_value, # Gurobi obj value
                    'NashProduct': np.exp(contract_model.results.objective_value), # Actual Nash product
                })
                earnings_list_scenarios.append(pd.DataFrame({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': contract_model.results.accumulated_revenue_G6,
                    'Revenue_L2': contract_model.results.accumulated_revenue_L2
                }))
            except RuntimeError as e:
                print(f"Optimization failed for A_G6={a_g6}, A_L2={a_l2}: {e}")
                # Store NaN or some indicator for failed runs
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': np.nan,
                    'Utility_G6': np.nan,
                    'Utility_L2': np.nan,
                    'NashProductLog': np.nan,
                    'NashProduct': np.nan,
                })
                earnings_list_scenarios.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': np.nan,
                    'Revenue_L2': np.nan
                })
            except Exception as e: # Catch other potential errors during setup/run
                print(f"An error occurred for A_G6={a_g6}, A_L2={a_l2}: {e}")
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': np.nan,
                    'Utility_G6': np.nan,
                    'Utility_L2': np.nan,
                    'NashProductLog': np.nan,
                    'NashProduct': np.nan,
                })
                earnings_list_scenarios.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': np.nan,
                    'Revenue_L2': np.nan
                })


    return pd.DataFrame(results_list),earnings_list_scenarios

def _run_bias_sensitivity_bias(input_data_base: InputData, opf_results: Expando):
    """
    Performs sensitivity analysis on price bias factors (KG, KL).

    Args:
        input_data_base: The base InputData object with core parameters.
        opf_results: The results object from the OptimalPowerFlow run.

    Returns:
        pandas.DataFrame: A DataFrame containing the sensitivity results
                          (KG_Factor, KL_Factor, Optimal_S, Optimal_M).
    """
    print("\n--- Starting Price Bias Sensitivity Analysis ---")

    # --- Calculate True Expected Price Sum (for scaling bias) ---
    # Need access to _build_dataframe, can make it a static method or call it carefully
 
    price_df_n3 = ContractNegotiation._build_dataframe(None, opf_results.price['N3'], 'price')
   
    price_n3_true = price_df_n3.values


    # Ensure calculation works even with few scenarios/times
    lambda_sum_true = price_n3_true.sum(axis=0) # Sum over time for each scenario
    if lambda_sum_true.size == 0:
         EP_lambda_sum_true = 0 # Handle case with no scenarios
    else:
        EP_lambda_sum_true = lambda_sum_true.mean()

    if EP_lambda_sum_true == 0:
        print("Warning: True Expected Sum of Prices (EP(λΣ)) is zero. Biases will be zero.")
    else:
        print(f"True Expected Sum of Prices (EP(λΣ)): {EP_lambda_sum_true:.4f}")

    # --- Sensitivity Analysis Loop ---
    # Define the bias factors relative to the expected price sum
    # Use the factors from the input_data_base if stored, otherwise define here
   
    K_G6_factors = [-0.01, 0, 0.01]
    K_L2_factors = [-0.01, 0, 0.01]

    results_sensitivity = []
    for kg_factor in K_G6_factors:
        for kl_factor in K_L2_factors:
            try:
                # Avoid division by zero if EP_lambda_sum_true is zero
                print(f"\nRunning Negotiation for KG factor={kg_factor} (Bias={(1+kg_factor)*EP_lambda_sum_true:.2f}), KL factor={kl_factor} (Bias={(1+kl_factor)*EP_lambda_sum_true:.2f})")
                input_data_base.K_G6 = kg_factor # Replacing the K_G6 factor in the input data
                input_data_base.K_L2 = kl_factor #Replacing the K_L2 factor in the input data
                #try:
                # Instantiate ContractNegotiation with current bias values
                # Pass the base OPF results and the current biases
                contract_model = ContractNegotiation(
                    input_data_base, # Pass the base data object
                    opf_results,     # Pass the results from the single OPF run
                )

                # Run the manual optimization to find S
                contract_model.run() # Set to True if you want to see the plot

                # Store results the amount to be changed ------ 300 !!!!!!!!!!!!!!!!!
                results_sensitivity.append({
                    'KG_Factor': kg_factor,
                    'KL_Factor': kl_factor,
                    'KG_Bias': (1+kg_factor)*EP_lambda_sum_true,
                    'KL_Bias': (1+kl_factor)*EP_lambda_sum_true,
                    'StrikePrice': contract_model.results.strike_price, # Store the optimal strike price S
                    'Optimal_M': 300 # Store the fixed M used
                })
                if np.isnan(contract_model.results.strike_price):
                        print(f" -> Negotiation failed or no feasible solution found.")
                else:
                    print(f" -> Optimal S: {contract_model.results.strike_price:.4f}, Optimal M: {300}")
            
            except Exception as e:
                print(f" -> Error during negotiation for KG factor={kg_factor}, KL factor={kl_factor}: {e}")
                # Store error or NaN
                results_sensitivity.append({
                    'KG_Factor': kg_factor,
                    'KL_Factor': kl_factor,
                    'KG_Bias': (1+kg_factor)*EP_lambda_sum_true,
                    'KL_Bias': (1+kl_factor)*EP_lambda_sum_true,
                    'Optimal_S': np.nan,
                    'Optimal_M': np.nan
                })
         

    print("\n--- Price Bias Sensitivity Analysis Complete ---")
    return pd.DataFrame(results_sensitivity)


# ----- Main Execution Block -----
if __name__ =='__main__':

    # --- Define Parameters and Ranges ---
    HOURS = 24
    DAYS = 5
    SCENARIOS = 50 # Keep small for faster testing, increase later
    
    # Define the ranges for risk aversion parameters
    A_G6_values = np.round(np.linspace(0.1, 0.9, 3),2)  # e.g., 5 values from 0.1 to 0.9
    A_L2_values = np.round(np.linspace(0.1, 0.9, 3),2)  # e.g., 5 values from 0.1 to 0.9
    beta_L = 0.5 # Placeholder, will be updated in the loop
    beta_G = 0.5 # Placeholder, will be updated in the loop

    # --- Load Data and Run OPF (Once) ---
    # Use placeholder beta values for the initial load_data call, 
    # they will be overwritten in the loop by A_G6_values/A_L2_values.
    print("Loading data and running initial OPF...")
    (GENERATORS, LOADS, NODES,TIME,SCENARIOS_L,PROB, fixed_Cost, generator_cost_a, 
     generator_cost_b, generator_capacity, load_capacity, mapping_buses, 
     mapping_generators, mapping_loads,branch_capacity_df, bus_reactance_df, 
     System_data,strikeprice_min, retail_price,_, # Renamed strikeprice_min again
     strikeprice_max,contract_amount_min,contract_amount_max,
     _, _, # Placeholder for A_L2, A_G6 from load_data
     K_L2,K_G6,alpha) = load_data(hours=HOURS, days=DAYS, scen=SCENARIOS, 
                                  beta_L=beta_L, beta_G=beta_G) # Use placeholder betas

    # Create the InputData object template
    input_data_template = InputData(
        GENERATORS = GENERATORS, 
        LOADS = LOADS,
        NODES = NODES,
        TIME = TIME,
        SCENARIOS_L = SCENARIOS_L,
        PROB = PROB,
        generator_cost_fixed= fixed_Cost,
        generator_cost_a = generator_cost_a,
        generator_cost_b = generator_cost_b,
        generator_capacity = generator_capacity,
        load_capacity = load_capacity,
        mapping_buses = mapping_buses ,
        mapping_generators = mapping_generators,
        mapping_loads = mapping_loads,
        branch_capacity = branch_capacity_df ,
        bus_susceptance = bus_reactance_df ,
        slack_bus = 'N1',
        retail_price= retail_price,
        system_data = System_data,
        strikeprice_min = strikeprice_min,
        strikeprice_max = strikeprice_max,
        contract_amount_min = contract_amount_min,
        contract_amount_max = contract_amount_max,
        A_L2 = beta_L, # Placeholder, will be updated in loop
        A_G6 = beta_G, # Placeholder, will be updated in loop
        K_L2 = K_L2,
        K_G6 = K_G6,
        alpha = alpha
    )

    # Run Optimal Power Flow
    opf_model = OptimalPowerFlow(input_data_template)
    opf_model.run()
    print("OPF run complete.")
    
    contract_model = ContractNegotiation(input_data_template, opf_model.results)
    contract_model.run() # Run the Gurobi optimization
    contract_model.display_results()
    contract_model.manual_optimization(plot=True)

    # Create the instance - _build_statistics() is called here, calculating Zeta etc.
    contract_plotter = ContractNegotiation(input_data_template, opf_model.results)

    # Check the conditions
    """ 
      cond29, cond30 = contract_model.check_barter_conditions()

    # You can now use the boolean values cond29 and cond30
    if cond29 is not None and cond30 is not None:
        if cond29 and cond30:
            print("--> Expected barter set shape: Triangle (Fig 2)")
        elif not cond30: # Condition 30 failure is key for Fig 3
            print("--> Expected barter set shape: Threat Point (Fig 3)")
        elif not cond29 and cond30: # Condition 29 failure is key for Fig 4
            print("--> Expected barter set shape: Triangle (Fig 4)")
        else: # Both fail?
            print("--> Expected barter set shape: Undetermined (Both conditions fail?)")
    else:
        print("--> Could not determine expected barter set shape.")
    """

    # Call the method on the instance
   # contract_plotter.plot_barter_set(case='fig2', filename='barter_set_fig2_method.png')
   # contract_plotter.plot_barter_set(case='fig3', filename='barter_set_fig3_method.png')
   # contract_plotter.plot_barter_set(case='fig4', filename='barter_set_fig4_method.png')



    # --- Run Sensitivity Analysis ---
    print("Starting sensitivity analysis for A_G6 and A_L2...")
    risk_sensitivity_results_df,sensitivity_earnings_df = run_risk_sensitivity_analysis(
        input_data_template, 
        opf_model.results, 
        A_G6_values, 
        A_L2_values
    )
    print("\nResults Summary:")
    print(risk_sensitivity_results_df)
    print("\nEarnings Summary:")
    print(sensitivity_earnings_df)

    
    bias_sensitivity_results_df = _run_bias_sensitivity_bias(input_data_template,  opf_model.results)

    # --- Plotting ---
    Plot_obj= Plotting_Class(risk_sensitivity_results_df,sensitivity_earnings_df,bias_sensitivity_results_df,styles=None)
    print("\nPlotting results...")
    # Optional name for saving plots filename = "sensitivity_analysis.png"
    # Plot Risk sensitivity results
    Plot_obj._plot_sensitivity_results()

    # Earnings Histograms
    print("\nPlotting earnings histograms...")
    Plot_obj._plot_earnings_histograms(
        fixed_A_G6 = A_G6_values[2], # The A_G6 value we kept constant
        A_L2_to_plot = A_L2_values.tolist() # The A_L2 values we varied
    )
    # Plot bias sensitivity results
    Plot_obj._plot_bias_sensitivity()
  