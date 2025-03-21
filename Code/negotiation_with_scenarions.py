
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import pandas as pd 
import scipy.stats as stats
from scipy.stats import qmc

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(42)

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


def load_data():
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
    T = 3 #24
    DAYS = 1 #30
    TIME = range(0,T*DAYS)
    SCENARIOS_L =range(5) # Number of scenarios for load 500? # something with scenario is cucking optimization
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
    A_L2 = [0.5,1,2] # Load (Buyer)
    A_G6 = [0.5,1,2] # Generator (Seller)

    # Price Bias (K)
    K_L2 = [-0.01,0,0.01]
    K_G6 = [-0.01,0,0.01]

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
        PROB: int,
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
        retail_price: int,
        strikeprice_min: int,
        strikeprice_max: int,
        contract_amount_min: int,
        contract_amount_max: int,
        A_L2: list,
        A_G6: list,
        K_L2: list,
        K_G6: list,
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
            self.data.generator_cost_b[g] * (self.variables.generator_production[g,t,s] ** 2) 
            for g in self.data.GENERATORS for t in self.data.TIME for s in self.data.SCENARIOS_L
        ) 
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
    def _build_model(self):
        self.model = gp.Model(name='Economic dispatch')
        self.model.Params.TimeLimit = 300
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
         # save objective value
        self.results.objective_value = self.model.ObjVal
        # save generator dispatch values
        self.results.generator_production = {
            (g,t,s): self.variables.generator_production[g,t,s].x for g in self.data.GENERATORS for t in self.data.TIME for s in self.data.SCENARIOS_L
        }
    
        # save voltage angles
        #self.results.voltage_angle = {
        #    (n,t,s): round(self.variables.voltage_angle[n,t,s].x, 2) for n in self.data.NODES for t in self.data.TIME for s in self.data.SCENARIOS_L
        #}
        # save optimal flows from node n to node m
        self.results.flows = {
            (n,m,t,s): round(
                self.data.bus_susceptance[n][m]  * (self.variables.voltage_angle[n,t,s].x-self.variables.voltage_angle[m,t,s].x), 2
            ) for n in NODES for m in [node for node in NODES if node not in [n]] for t in self.data.TIME for s in self.data.SCENARIOS_L
        }
        # save price (i.e., dual variable of balance constraint)
        self.results.price = {
            (n,t,s): (1/self.data.PROB)*self.constraints.balance_constraint[n,t,s].Pi for n in self.data.NODES for t in self.data.TIME for s in self.data.SCENARIOS_L
        }
        # save generator capacity sensitivities (i.e., duals of max production constraints)
        #self.results.capacity_sensitivities = {
        #    (g,t,s): self.constraints.capacity_constraints[g,t,s].Pi for g in self.data.GENERATORS for t in self.data.TIME for s in self.data.SCENARIOS_L
        #}
        # save flow capacity sensitivities (i.e., duals of max flow constraints)
        #self.results.max_flow_sensitivities = {k: v.Pi for k,v in self.constraints.max_flow_constraint.items()}


        #Calculate Revenues here loop the shit instead
        # Save Generator Revenue
        #self.results.generator_revenue = {g:{(t,s):
        #    ((self.results.price[n,t,s]-self.data.generator_cost_a[g])*self.results.generator_production[g,t,s] - self.data.generator_cost_b[g] *self.results.generator_production[g,t,s]**2 )* self.data.mapping_generators[n][g] 
        #    for t in self.data.TIME for s in self.data.SCENARIOS_L for n in self.data.NODES
        #}  for g in self.data.GENERATORS}
        #Save Load Revenue - Just assuming they all have the same retail price (L2 is the one in question though)

        self.results.generator_revenue = {g: {(t, s): sum((self.results.price[n, t, s] - self.data.generator_cost_a[g]) * self.results.generator_production[g, t, s]- self.data.generator_cost_b[g] * self.results.generator_production[g, t, s] ** 2
            for n in self.data.NODES
            if self.data.mapping_generators.loc[g, n] == 1  # Only include nodes where the generator is located
            )
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
            }
            for g in self.data.GENERATORS}
        """
        self.results.generator_loss = {g: {(t, s): sum((self.data.generator_cost_a[g] - self.results.price[n, t, s] ) * self.results.generator_production[g, t, s] + self.data.generator_cost_b[g] * self.results.generator_production[g, t, s] ** 2
            for n in self.data.NODES
            if self.data.mapping_generators.loc[g, n] == 1  # Only include nodes where the generator is located
            )
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
            }
            for g in self.data.GENERATORS}
        """


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
        #print("Optimal voltage angels:")
        #print(self.results.voltage_angle)
       
        #print("Optimal flows:")
        #for k, v in self.results.flows.items():
        #    if v != 0:
        #        print((k, v))        
        print("Prices at optimality:")
        print(self.results.price)
        #print("Capacity sensitivities")
        #print(self.results.capacity_sensitivities)
        #rint("Optimal duals of max flow constraints:")
        #for k, v in self.results.max_flow_sensitivities.items():
        #    if v != 0:
        #        print((k, v))
        

class ContractNegotiation():
    def __init__(self,input_data:InputData,opf_results: OptimalPowerFlow,risk=False):
        self.data = input_data  
        self.data.risk= risk
        self.opf_results = opf_results
        self.results = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self._build_statistics()
        self._build_model()
        
    def _build_dataframe(self,data):
        # Convert the dictionary to a Series, which will have a MultiIndex
        df = pd.Series(data)
        # Unstack by the first level of the index (the 'N' and t)
        df = df.unstack(level=0)
        df = df.to_dict()
        return df
    
    def _build_dictionary(self,df):

        dictionary = df.to_dict()
        return dictionary
    
    def _CVaR (self,earnings):
        #CVaR calculation
        #Make this a func to calculate or not. making it a func
        earnings = np.array(earnings)
        
        var_threshold_lower = np.percentile(earnings,(1-self.data.alpha)*100)
        print(f"Value at Risk (VaR) at {self.data.alpha*100}% confidence level is {var_threshold_lower}")
        tail_losses = earnings[earnings >= var_threshold_lower]
        CVaR_loss = np.mean(tail_losses)
        print(f"Conditional Value at Risk (CVaR) at {self.data.alpha*100}% confidence level is {CVaR_loss}")

        """
        var_threshold_upper = np.percentile(earnings,(self.data.alpha)*100)
        print(f"Value at Risk (VaR) at {self.data.alpha*100}% confidence level is {var_threshold_upper}")
        tail_win = earnings[var_threshold_upper <= earnings]
        CVaR_win = np.mean(tail_win)
        print(f"Conditional Value at Risk (CVaR) at {self.data.alpha*100}% confidence level is {CVaR_win}")

        plt.hist(earnings, bins=20, alpha=0.5, label="No Contract", color="blue")
        plt.axvline(CVaR_loss, color='green', linestyle='--', label=f'VaR (Left, {100 - self.data.alpha *100:.0f}%)')
        plt.axvline(CVaR_win,color = 'red', linestyle='--', label=f'VaR (Right, {self.data.alpha *100:.0f}%)')
        plt.legend()
        plt.show()
        """
        return CVaR_loss
    
    def _build_statistics(self):
        # Calculate the average price at each node
        self.data.price = self._build_dataframe(self.opf_results.price)
        self.data.generator_production = self._build_dataframe(self.opf_results.generator_production)
        self.data.N3_mean = np.mean(list(self.data.price['N3'].values()))
        self.data.N3_std =np.std(list(self.data.price['N3'].values()))

        # Make distributions for the price at node N3
        self.data.pdf_true = stats.norm.pdf(list(self.data.price['N3'].values()), self.data.N3_mean, self.data.N3_std) # True distribution 
        
        # If assuming they have different observations of prices ( i.e., different K (bias) values)
        #pdf_G6 = stats.norm.pdf(self.data.price['N3'].values, self.data.N3_mean+K_G6, self.data.N3_std) # G6 observed distribution 
        #pdf_L2 = stats.norm.pdf(self.data.price['N3'].values, self.data.N3_mean+K_L2, self.data.N3_std) # L2 observed distribution 
        
        #Make this a call from utility function 
        self.data.net_earnings_no_contract_G6 = np.array(list(self.opf_results.generator_revenue['G6'].values()))
        self.data.net_earnings_no_contract_L2 = self.data.load_capacity['L2'].flatten()*(np.array(list(self.data.price['N3'].values()))-self.data.retail_price)
        #CVaR calculation of No contract 
        self.data.CVaR_no_contract_G6 = self._CVaR(self.data.net_earnings_no_contract_G6)
        self.data.CVaR_no_contract_G6_loss = self._CVaR(-self.data.net_earnings_no_contract_G6)
        self.data.CVaR_no_contract_L2 = self._CVaR(self.data.net_earnings_no_contract_L2)
        self.data.CVaR_no_contract_L2_loss = self._CVaR(-self.data.net_earnings_no_contract_L2)

        # Threat Point G6
        self.data.Zeta_G6=np.mean(self.data.net_earnings_no_contract_G6)  - self.data.A_G6[1]*self.data.CVaR_no_contract_G6

        #Threat Point for L2
        self.data.Zeta_L2=np.mean(self.data.net_earnings_no_contract_L2)  - self.data.A_L2[1]*self.data.CVaR_no_contract_L2


        #CvaR prices 
        self.data.CVaR_price = self._CVaR(np.array(list(self.data.price['N3'].values())))
        self.data.CVaR_neg_price = self._CVaR(-np.array(list(self.data.price['N3'].values())))

        print(self.data.CVaR_price)
        print(self.data.CVaR_neg_price)

        #SR # Not correct Missing adding E*K

        self.data.SR = np.min([(self.data.N3_mean + self.data.A_G6[1]*self.data.CVaR_price + (1+self.data.A_G6[1]*self.data.K_G6[1]))/(len(self.data.TIME)*(1+self.data.A_G6[1])),
                              (self.data.N3_mean - self.data.A_G6[1]*self.data.CVaR_price + (1+self.data.A_G6[1]*self.data.K_G6[1]))/(len(self.data.TIME)*(1+self.data.A_G6[1])),
                              (self.data.N3_mean + (1+self.data.A_L2[1])*self.data.K_L2[1] - self.data.A_L2[1]*self.data.CVaR_neg_price)/(len(self.data.TIME)*(1+self.data.A_L2[1]))])

        #SU
        self.data.SU = np.max([(self.data.N3_mean + self.data.A_G6[1]*self.data.CVaR_price + (1+self.data.A_G6[1]*self.data.K_G6[1]*self.data.))/(len(self.data.TIME)*(1+self.data.A_G6[1])),
                              (self.data.N3_mean - self.data.A_G6[1]*self.data.CVaR_price + (1+self.data.A_G6[1]*self.data.K_G6[1]))/(len(self.data.TIME)*(1+self.data.A_G6[1])),
                              (self.data.N3_mean + (1+self.data.A_L2[1])*self.data.K_L2[1] - self.data.A_L2[1]*self.data.CVaR_neg_price)/(len(self.data.TIME)*(1+self.data.A_L2[1]))])

        print(self.data.SR)
        print(self.data.SU)
        print()
    def _build_model(self):
        self.model = gp.Model(name='Nash Bargaining Model')
        #self.model.Params.TimeLimit = 300
        self._build_variables()
        self._build_constraints()
        if self.data.risk == True:
            self._build_objective_risk()
        else:
            self._build_objective_norisk()
        self.model.update()
    
    def _build_variables(self):
        # build strike price variables

        self.variables.S = {s: self.model.addVar(
          lb=self.data.strikeprice_min,
            ub=self.data.strikeprice_max,
            name='Strike Price'
            )
            for s in self.data.SCENARIOS_L
        }
        #self.variables.S = self.model.addVar(
        #    lb=self.data.strikeprice_min,
        #    ub=self.data.strikeprice_max,
        #    name='Strike Price'
        #)
    
        # build contract amount variables
        self.variables.M = self.model.addVar(
            lb=self.data.contract_amount_min,
            ub=self.data.contract_amount_max,
            name='Contract Amount'
        )
        self.variables.zeta_G6 = self.model.addVar(
            name='Threat Point G6',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
        self.variables.zeta_L2 = self.model.addVar(
            name='Threat Point G6',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
      
        self.variables.eta_G6 = {(t,s): self.model.addVar(
            name='Auxillary Variable G6',
            lb=0,ub=gp.GRB.INFINITY)
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
        }
        self.variables.eta_L2 = {(t,s): self.model.addVar(
            name='Auxillary Variable L2',
            lb=0,ub=gp.GRB.INFINITY)
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
        }
       
    def _build_constraints(self):

        #Strike Price max constraints
        self.constraints.strike_price_constraint = {s : self.model.addLConstr(
            self.variables.S[s],
            gp.GRB.LESS_EQUAL,
            self.data.strikeprice_max,
            name='Strike Price Constraint Max'
        )
        for s in self.data.SCENARIOS_L
        }
         #Strike Price min  constraints
        self.constraints.strike_price_constraint_min = {s: self.model.addLConstr(
            self.variables.S[s],
            gp.GRB.GREATER_EQUAL,
            self.data.strikeprice_min,
            name='Strike Price Constraint Min'
        )
        for s in self.data.SCENARIOS_L
        }
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
        

        #Risk Constraints   
    
        #Risk Aversion for G6
        self.constraints.eta_G6_constraint = {
            (t,s): self.model.addLConstr(
            self.variables.eta_G6[t,s], gp.GRB.GREATER_EQUAL, self.variables.zeta_G6  -  (-1)*((self.data.price['N3'][t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production['G6'][t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production['G6'][t,s]**2)
                                                 + (self.variables.S[s] - self.data.price['N3'][t,s])*self.data.generator_production['G6'][t,s]),
            #add scenarios here in next iteration 
            name='Risk Aversion Constraint G6'
        )
        for t in self.data.TIME
        for s in self.data.SCENARIOS_L
        }
       

        #Risk Aversion for L2
        self.constraints.eta_L2_constraint = {(t,s): self.model.addLConstr(
            self.variables.eta_L2[t,s], gp.GRB.GREATER_EQUAL, self.variables.zeta_L2 - (-1)* (self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price['N3'][t,s])+ (self.data.price['N3'][t,s] - self.variables.S[s])*self.data.generator_production['G6'][t,s]),
            #add scenarios here in next iteration 
            name='Risk Aversion Constraint G6'
        )
            for t in self.data.TIME
            for s in self.data.SCENARIOS_L
        
        }
        
        
    def _build_objective_risk(self):
        
        #Utility function for G6 
        EuG6 =  gp.quicksum((self.data.price['N3'][t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production['G6'][t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production['G6'][t,s]**2) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG6 =  gp.quicksum((self.variables.S[s] - self.data.price['N3'][t,s])*self.data.generator_production['G6'][t,s] for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #CVaR for G6 

        CVaRG6 = gp.quicksum(self.variables.zeta_G6 - (1/(1-self.data.alpha))*(self.data.PROB*self.variables.eta_G6[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UG6 = self.data.PROB*(EuG6 + SMG6) - self.data.A_G6[1]*CVaRG6


        #Utility function for L2
        EuL2 = gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price['N3'][t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML2 =  gp.quicksum((self.data.price['N3'][t,s] - self.variables.S[s])*self.data.generator_capacity['G6'] for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #CvaR for L2
        CVaRL2 = self.variables.zeta_G6 - (1/(1-self.data.alpha))*gp.quicksum(self.data.PROB*self.variables.eta_L2[t,s] for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UL2 = self.data.PROB*(EuL2 + SML2) - self.data.A_L2[1]*CVaRL2
        
      
        #Objective function
        objective = (UG6 -self.data.Zeta_G6) *(UL2-self.data.Zeta_L2)
        self.model.setObjective(objective,sense = gp.GRB.MAXIMIZE)
        self.model.update()
    
    def _build_objective_norisk(self):
        
         #Utility function for G6 
        EuG6 =  gp.quicksum((self.data.price['N3'][t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production['G6'][t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production['G6'][t,s]**2) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG6 =  gp.quicksum((self.variables.S[s] - self.data.price['N3'][t,s])*self.data.generator_production['G6'][t,s] for t in self.data.TIME for s in self.data.SCENARIOS_L)

     
        #Expectation in utility function
        UG6 = self.data.PROB*(EuG6 + SMG6) 

        # Threat Point G6
        Zeta_G6=np.mean(self.data.net_earnings_no_contract_G6)  

        #Utility function for L2
        EuL2 = gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price['N3'][t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML2 =  gp.quicksum((self.data.price['N3'][t,s] - self.variables.S[s])*self.data.generator_capacity['G6'] for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #Expectation in utility function
        UL2 = self.data.PROB*(EuL2 + SML2) 
        
        #Threat Point for L2
        Zeta_L2=np.mean(self.data.net_earnings_no_contract_L2)  

        #Objective function
        objective = (UG6 -Zeta_G6) *(UL2-Zeta_L2)
        self.model.setObjective(objective,sense = gp.GRB.MAXIMIZE)
        self.model.update()

    def _save_results(self):

        #Plot income distributions for generator and load with contract and without contract

        # save objective value
        self.results.objective_value = self.model.ObjVal
        # save strike price

        #self.results.strike_price = self.variables.S.x

        self.results.strike_price = {s: self.variables.S[s].x for s in self.data.SCENARIOS_L}
        self.results.strike_price_mean = np.mean(np.array(list(self.results.strike_price.values())))
        print(self.results.strike_price)
        print(self.results.strike_price_mean )
        # save contract amount
        #self.results.contract_amount = self.model.M.x

        #Temporary save for contract amount
        self.results.contract_amount = 300 # not actual 
        # save utility for G6
        EuG6 =  np.array(self.data.net_earnings_no_contract_G6)
        print(EuG6)
        SMG6 =  {(t,s):(self.results.strike_price[s] - self.data.price['N3'][t,s])*self.results.contract_amount
                 for t in self.data.TIME for s in self.data.SCENARIOS_L}
        SMG6_arr = np.array(list(SMG6.values()))
     
        CVaRG6 = self._CVaR((-EuG6  -SMG6_arr))
        
        UG6 = np.mean((EuG6 + SMG6_arr)) - self.data.A_G6[1]*CVaRG6
        self.results.utility_G6 = UG6

        # save utility for L2
        EuL2 = np.array(self.data.net_earnings_no_contract_L2)
     
        SML2 =  {(t,s): (self.data.price['N3'][t,s] - self.results.strike_price[s])*self.results.contract_amount 
                 for t in self.data.TIME for s in self.data.SCENARIOS_L}
        SML2_arr = np.array(list(SML2.values())).flatten()
        CVaRL2 = self._CVaR((-EuL2 - SML2_arr))
        UL2 = np.mean((EuL2 + SML2_arr) - self.data.A_L2[1]*CVaRL2)
        self.results.utility_L2 = UL2
        

    def display_results(self):
        print("-------------------   RESULTS  -------------------")
        print("Optimal Objective Value:")
        print(self.results.objective_value)
        print("Optimal Strike Price:")
        print(self.results.strike_price)
        print("Optimal Contract Amount:")
        print(self.results.contract_amount)
        print("Optimal Utility for G6:")
        print(self.results.utility_G6)
        print("Optimal Utility for L2:")
        print(self.results.utility_L2)

    def _build_bargaining_Set(self):
        #Extra constraints 
        return
    
    def Distribution_negotiation(self):
        

                # Parameters (example values)
        #A_G = 1.0  # GenCo's risk-aversion factor
        #A_L = 1.0  # LSE's risk-aversion factor
        #zeta_1 = 100  # Threat point utility for GenCo
        #zeta_2 = 150  # Threat point utility for LSE
        #mu_lambda = 500  # Mean of lambda (sum of LMPs over contract period)
        #sigma_lambda = 50  # Standard deviation of lambda
        T = 24  # Contract period (hours)
        #alpha = 0.95  # Confidence level for CVaR
        #M_lower = 0  # Lower bound for contract amount
        #M_upper = 600  # Upper bound for contract amount
        #S_lower = 15  # Lower bound for strike price
        #S_upper = 25  # Upper bound for strike price

        # Expected net earnings from the day-ahead market (example values)
        #E_G_pi_G_lambda = 1000  # GenCo's expected net earnings
        #E_L_pi_L_0 = 1200  # LSE's expected net earnings

        # CVaR calculation for a normal distribution
        def calculate_cvar(mean, std, alpha):
            """Calculate CVaR for a normal distribution."""
            phi_alpha = norm.pdf(norm.ppf(alpha))  # PDF of standard normal at alpha quantile
            return mean + std * (phi_alpha / (1 - alpha))

        # Objective function
        def objective_function(x):
            """Objective function to maximize (u_G - zeta_1) * (u_L - zeta_2)."""
            M, S = x  # Unpack decision variables

            # Expected earnings
            E_G = E_G_pi_G_lambda + (T * S - self.data.N3_mean) * M
            E_L = E_L_pi_L_0 + (self.data.N3_mean - T * S) * M

            # CVaR calculations
            CVaR_G = calculate_cvar(E_G_pi_G_lambda, self.data.N3_std, alpha) + (T * S - self.data.N3_mean) * M
            CVaR_L = calculate_cvar(E_L_pi_L_0, self.data.N3_std, alpha) + (self.data.N3_mean - T * S) * M

            # Utility functions
            u_G = E_G - self.data.A_G6[1] * CVaR_G
            u_L = E_L - self.data.A_L2[1] * CVaR_L

            # Objective: maximize (u_G - zeta_1) * (u_L - zeta_2)
            return -(u_G - self.data.Zeta_G6) * (u_L - self.data.Zeta_L2)  # Negative for minimization

        # Bounds for M and S
        bounds = [(self.data.contract_amount_min, self.data.contract_amount_max), (self.data.strikeprice_min, self.data.strikeprice_max)]

        # Initial guess for M and S
        initial_guess = [300, 20]  # Example initial guess

        # Constraints (if any)
        # For example, you could add constraints on M and S here if needed
        constraints = []  # No additional constraints in this example

        # Optimize using scipy.optimize.minimize
        result = minimize(
            objective_function,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'  # Sequential Least Squares Programming
        )

        # Output the results
        if result.success:
            optimal_M, optimal_S = result.x
            max_objective_value = -result.fun  # Negate to get the original objective value
            print(f"Optimal Contract Amount (M): {optimal_M}")
            print(f"Optimal Strike Price (S): {optimal_S}")
            print(f"Maximum Objective Value: {max_objective_value}")
        else:
            print("Optimization failed:", result.message)





    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
            self.model.write('model.ilp')
        else:
            self._save_results()
    
            self.model.write('model.ilp')
            raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")
    
if __name__ =='__main__':

    (GENERATORS, LOADS, NODES,TIME,SCENARIOS_L,PROB, fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity, load_capacity, 
            mapping_buses, mapping_generators, mapping_loads,branch_capacity_df, bus_reactance_df, System_data,strikeprice_min,
            retail_price,strikeprice_min,strikeprice_max,contract_amount_min,contract_amount_max,A_L2,A_G6,K_L2,K_G6,alpha) = load_data()
    
    input_data = InputData(
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
        A_L2 = A_L2,
        A_G6 = A_G6,
        K_L2 = K_L2,
        K_G6 = K_G6,
        alpha = alpha
       
    )

    opf_model = OptimalPowerFlow(input_data)
    opf_model.run()
    
    #Final version should be renamed back to 'contract_model', for debugging keeping it as cm
    #With Risk
    risk = True
    #cm = ContractNegotiation(input_data,opf_model.results,risk)
    #cm.run()
    #cm.display_results()

    # No Risk
    risk = True
    cm = ContractNegotiation(input_data,opf_model.results,risk)
    cm.run()
    cm.display_results()
