
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import pandas as pd 
import scipy.stats as stats


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
    #Hours
    T = 24
    TIME = range(0,T)
    mapping_buses = pd.DataFrame({'N1':[0,1,1,1,1],'N2':[1,0,1,1,1],'N3':[1,1,0,1,1],'N4':[1,1,1,0,1],'N5':[1,1,1,1,0]},index = NODES)

    # Branch Data 
    Line_reactane = np.array([0.0281,0.0304,0.0064,0.0108,0.0297,0.0297])

    Line_from = np.array([1,1,1,2,3,4])
    Line_to = np.array([2,4,5,3,4,5])
    Linecap = np.array([250,150,400,350,240,240])

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
    Load_data = np.array([  [350.00, 322.93, 305.04, 296.02, 287.16, 291.59, 296.02, 314.07,
                            300.00, 276.80, 261.47, 253.73, 246.13, 249.93, 253.73, 269.20,
                            250.00, 230.66, 217.89, 211.44, 205.11, 208.28, 211.44, 224.33]
                            ,
                            [337.86, 394.80, 403.82, 408.25, 403.82, 394.80, 390.37, 390.37,
                            307.60, 338.40, 346.13, 349.93, 346.13, 338.40, 334.60, 334.60,
                            256.33, 282.00, 288.44, 291.61, 288.44, 282.00, 278.83, 278.83]
                            ,
                            [408.25, 448.62, 430.73, 426.14, 421.71, 412.69, 390.37, 363.46,
                            349.93, 384.53, 369.20, 365.26, 361.47, 353.73, 344.36, 311.53,
                            291.61, 320.44, 307.67, 304.39, 301.22, 294.78, 278.83, 259.61]])

    # Create a DataFrame from the Load_data array
    load_capacity = pd.DataFrame(Load_data, index=["L1", "L2", "L3"], columns=range(0, 24)).T
    mapping_loads = pd.DataFrame({'N1':[0,0,0],'N2':[1,0,0],'N3':[0,1,0],'N4':[0,0,1],'N5':[0,0,0]},index = LOADS)

    #Make representative of generated load scenarions , for now just 1 
    num_L_scen = 1  

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


    return (GENERATORS, LOADS, NODES,TIME, fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity, load_capacity, 
            mapping_buses, mapping_generators, mapping_loads,num_L_scen, branch_capacity_df, bus_reactance_df, System_data,strikeprice_min,
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
        generator_cost_fixed: dict[str, int], 
        generator_cost_a: dict[str, int],
        generator_cost_b: dict[str, int],   
        generator_capacity: dict[str, int], 
        load_capacity: dict[str, int],
        mapping_buses: dict[str, list],
        mapping_generators: dict[str, list],
        mapping_loads: dict[str, list],
        num_L_scen: int,
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
        # List of time periods
        self.TIME = TIME
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
        self.num_L_scen = num_L_scen
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

    def __init__(self, input_data: InputData): # initialize class
        self.data = input_data          # define data attributes
        self.variables = Expando()      # define variable attributes
        self.constraints = Expando()    # define constraints attributes
        self.results = Expando()        # define results attributes
        self._build_model()             # build gurobi model
    

    def _build_variables(self):
        # build generator production variables
        self.variables.generator_production = {
            (g,t): self.model.addVar(
                lb=0,
                name='Electricity production of generator {0} at t {1}'.format(g,t)
            ) for g in self.data.GENERATORS 
              for t in self.data.TIME
        }
        # build voltage angle variables
        self.variables.voltage_angle = {
            (n,t): self.model.addVar(
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
                name='Voltage angle at node {0} at time {1}'.format(n,t)
            ) for n in self.data.NODES 
              for t in self.data.TIME
        }
    def _build_constraints(self):
        #Balance equation at each node
        """"""
        self.constraints.balance_constraint = {
            (n,t): self.model.addLConstr(
                gp.quicksum(
                    self.variables.generator_production[g,t] * self.data.mapping_generators[n][g] for g in self.data.GENERATORS 
                ) - gp.quicksum(
                    self.data.load_capacity[d][t] * self.data.mapping_loads[n][d] for d in self.data.LOADS
                ),
                gp.GRB.EQUAL,
                gp.quicksum(
                    self.data.bus_susceptance[n][m] * self.variables.voltage_angle[m,t] for m in self.data.NODES 
                ),
                name='Balance equation at node {0} at time {1}'.format(n,t)
            ) for n in self.data.NODES 
              for t in self.data.TIME
        }

        #Max flow between nodes
        self.constraints.max_flow_constraint = {
            (n, m,t): self.model.addLConstr(
                self.data.bus_susceptance[n][m] * (self.variables.voltage_angle[n,t] - self.variables.voltage_angle[m,t]),
                gp.GRB.LESS_EQUAL,
                self.data.branch_capacity[n][m],
                name='Constraint on max flow between nodes {0} and {1} at time {2}'.format(n, m,t)
            ) for n in self.data.NODES for m in [node for node in self.data.NODES if node not in [n]]
              for t in self.data.TIME
        }

        #Max production of generators
        self.constraints.capacity_constraints = {
            (g,t): self.model.addLConstr(
                self.variables.generator_production[g,t],
                gp.GRB.LESS_EQUAL,
                self.data.generator_capacity[g],
                name='Constraint on max production of generator {0} at time {1}'.format(g,t)
            ) for g in self.data.GENERATORS[0:6]
              for t in self.data.TIME
        }

        
        #Set production for generator 6 at all times : 300 MW

        self.constraints.base_load_Gen6 = {
            (t): self.model.addConstr(
            self.variables.generator_production['G6', t],
            gp.GRB.EQUAL,
            self.data.generator_capacity['G6'],
            name='Constraint on baseload generator G6 at time {0}'.format(t)
        ) for t in self.data.TIME
        }

   

        #Slack bus voltage angle
        self.constraints.slack_bus_constraint = {t: self.model.addLConstr(
            self.variables.voltage_angle[self.data.slack_bus,t],
            gp.GRB.EQUAL,
            0,
            name='Slack bus voltage angle'
        ) for t in self.data.TIME
        }
    def _build_objective(self):
        #Objective function
        objective = gp.quicksum(
            self.data.generator_cost_a[g] * self.variables.generator_production[g,t] +
            self.data.generator_cost_b[g] * (self.variables.generator_production[g,t] ** 2) 
            for g in self.data.GENERATORS for t in self.data.TIME
        ) 
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
    def _build_model(self):
        self.model = gp.Model(name='Economic dispatch')
        self.model.Params.TimeLimit = 100
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
         # save objective value
        self.results.objective_value = self.model.ObjVal
        # save generator dispatch values
        self.results.generator_production = {
            (g,t): self.variables.generator_production[g,t].x for g in self.data.GENERATORS for t in self.data.TIME
        }
        # save voltage angles
        self.results.voltage_angle = {
            (n,t): round(self.variables.voltage_angle[n,t].x, 2) for n in self.data.NODES for t in self.data.TIME
        }
        # save optimal flows from node n to node m
        self.results.flows = {
            (n,m,t): round(
                self.data.bus_susceptance[n][m]  * (self.variables.voltage_angle[n,t].x-self.variables.voltage_angle[m,t].x), 2
            ) for n in NODES for m in [node for node in NODES if node not in [n]] for t in self.data.TIME
        }
        # save price (i.e., dual variable of balance constraint)
        self.results.price = {
            (n,t): self.constraints.balance_constraint[n,t].Pi for n in self.data.NODES for t in self.data.TIME
        }
        # save generator capacity sensitivities (i.e., duals of max production constraints)
        self.results.capacity_sensitivities = {
            (g,t): self.constraints.capacity_constraints[g,t].Pi for g in self.data.GENERATORS for t in self.data.TIME
        }
        # save flow capacity sensitivities (i.e., duals of max flow constraints)
        self.results.max_flow_sensitivities = {k: v.Pi for k,v in self.constraints.max_flow_constraint.items()}

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
        print("Optimal voltage angels:")
        print(self.results.voltage_angle)
        print("Optimal flows:")
        for k, v in self.results.flows.items():
            if v != 0:
                print((k, v))        
        print("Prices at optimality:")
        print(self.results.price)
        print("Capacity sensitivities")
        print(self.results.capacity_sensitivities)
        print("Optimal duals of max flow constraints:")
        for k, v in self.results.max_flow_sensitivities.items():
            if v != 0:
                print((k, v))


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
        return df
    
    def _CVaR (self,earnings):
        #CVaR calculation
        #Make this a func to calculate or not. making it a func
        var_threshold = np.percentile(earnings,(1-self.data.alpha)*100)
        print(f"Value at Risk (VaR) at {self.data.alpha*100}% confidence level is {var_threshold}")

        tail_losses = earnings[earnings < var_threshold]
        CVaR = tail_losses.mean()
        print(f"Conditional Value at Risk (CVaR) at {self.data.alpha*100}% confidence level is {CVaR}")

        return CVaR
    
    def _build_statistics(self):
        # Calculate the average price at each node
        self.data.price = self._build_dataframe(self.opf_results.price)
        self.data.generator_production = self._build_dataframe(self.opf_results.generator_production)
        self.data.N3_mean = self.data.price['N3'].mean()
        self.data.N3_std = self.data.price['N3'].std()

        # Make distributions for the price at node N3
        self.data.pdf_true = stats.norm.pdf(self.data.price['N3'].values, self.data.N3_mean, self.data.N3_std) # True distribution 
        
        # If assuming they have different observations of prices ( i.e., different K (bias) values)
        #pdf_G6 = stats.norm.pdf(self.data.price['N3'].values, self.data.N3_mean+K_G6, self.data.N3_std) # G6 observed distribution 
        #pdf_L2 = stats.norm.pdf(self.data.price['N3'].values, self.data.N3_mean+K_L2, self.data.N3_std) # L2 observed distribution 
        
        #Make this a call from utility function 
        self.data.net_earnings_no_contract_G6 = (self.data.price['N3']-self.data.generator_cost_a['G6'])*self.data.generator_production['G6']-self.data.generator_cost_b['G6']*(self.data.generator_production['G6']**2)
        self.data.net_earnings_no_contract_L2 = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price['N3'])
        #CVaR calculation of No contract 
        self.data.CVaR_no_contract_G6 = self._CVaR(self.data.net_earnings_no_contract_G6)
        self.data.CVaR_no_contract_L2 = self._CVaR(self.data.net_earnings_no_contract_L2)

    def _build_model(self):
        self.model = gp.Model(name='Nash Bargaining Model')
        self.model.Params.TimeLimit = 100
        self._build_variables()
        self._build_constraints()
        if self.data.risk == True:
            self._build_objective_risk()
        else:
            self._build_objective_norisk()
        self.model.update()
    
    def _build_variables(self):
        # build strike price variables
        self.variables.S = self.model.addVar(
            lb=self.data.strikeprice_min,
            ub=self.data.strikeprice_max,
            name='Strike Price'
        )
    
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

        self.variables.eta_G6 = self.model.addVar(
            name='Auxillary Variable G6',
            lb=0,ub=gp.GRB.INFINITY)
        self.variables.eta_L2 = self.model.addVar(
            name='Auxillary Variable L2',
            lb=0,ub=gp.GRB.INFINITY)
       
    def _build_constraints(self):

        #Strike Price max constraints
        self.constraints.strike_price_constraint = self.model.addLConstr(
            self.variables.S,
            gp.GRB.LESS_EQUAL,
            self.data.strikeprice_max,
            name='Strike Price Constraint Max'
        )
         #Strike Price min  constraints
        self.constraints.strike_price_constraint_min = self.model.addLConstr(
            self.variables.S,
            gp.GRB.GREATER_EQUAL,
            self.data.strikeprice_min,
            name='Strike Price Constraint Min'
        )
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

        #Risk Constraints 

        #Risk Aversion for G6
        self.constraints.eta_G6_constraint = self.model.addLConstr(
            self.variables.eta_G6 >= gp.quicksum( (self.data.price['N3']-self.data.generator_cost_a['G6'])*self.data.generator_production['G6']-self.data.generator_cost_b['G6']*(self.data.generator_production['G6']**2)+(self.variables.S - self.data.price['N3'])*self.data.generator_capacity['G6']),
            #add scenarios here in next iteration 
            name='Risk Aversion Constraint G6'
        )
        #Risk Aversion for L2
        self.constraints.eta_L2_constraint = self.model.addLConstr(
            self.variables.eta_G6 >= gp.quicksum(self.data.load_capacity['L2']*(self.data.retail_price-self.data.price['N3'])+ (self.data.price['N3'] - self.variables.S)*self.data.generator_capacity['G6']),
            #add scenarios here in next iteration 
            name='Risk Aversion Constraint G6'
        )
        
        return
    def _build_objective_risk(self):
        
        #Utility function for G6 
        EuG6 =  (self.data.price['N3']-self.data.generator_cost_a['G6'])*self.data.generator_production['G6']-self.data.generator_cost_b['G6']*(self.data.generator_production['G6']**2)
        SMG6 =  (self.variables.S - self.data.price['N3'])*self.data.generator_capacity['G6']
        #CVaR for G6 

        CVaRG6 = gp.quicksum(self.data.A_G6[1]*(self.variables.zeta_G6 - (1/(1-self.data.alpha))*self.data.load_capacity['L2'][t]*self.variables.eta_G6) for t in TIME )
        #Expectation in utility function
        UG6 = (1/self.data.num_L_scen)*gp.quicksum(EuG6 + SMG6) - CVaRG6

        # Threat Point G6
        Zeta_G6=self.data.net_earnings_no_contract_G6  - self.data.A_G6[1]*self.data.CVaR_no_contract_G6

        #Utility function for L2
        EuL2 = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price['N3'])
        SML2 =  (self.data.price['N3'] - self.variables.S)*self.data.generator_capacity['G6']
        #CvaR for L2
        CVaRL2 = self.variables.zeta_G6 - (1/(1-self.data.alpha))*gp.quicksum(self.data.load_capacity['L2'][t]*self.variables.eta_L2 for t in TIME)
        #Expectation in utility function
        UL2 = 1/self.data.num_L_scen*gp.quicksum(EuL2 + SML2) - self.data.A_L2[1]*CVaRL2
        
        #Threat Point for L2
        Zeta_L2=self.data.net_earnings_no_contract_L2  - self.data.A_L2[1]*self.data.CVaR_no_contract_L2

        #Objective function
        objective = (UG6 -Zeta_G6) *(UL2-Zeta_L2)
        self.model.setObjective(gp.quicksum(objective))
        self.model.update()
    
    def _build_objective_norisk(self):
        
        #Utility function for G6 
        EuG6 =  (self.data.price['N3']-self.data.generator_cost_a['G6'])*self.data.generator_production['G6']-self.data.generator_cost_b['G6']*(self.data.generator_production['G6']**2)
        SMG6 =  (self.variables.S - self.data.price['N3'])*self.data.generator_capacity['G6']
      

        #Expectation in utility function
        UG6 = (1/self.data.num_L_scen)*gp.quicksum(EuG6 + SMG6) 

        # Threat Point G6
        Zeta_G6=self.data.net_earnings_no_contract_G6 

        #Utility function for L2
        EuL2 = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price['N3'])
        SML2 =  (self.data.price['N3'] - self.variables.S)*self.data.generator_capacity['G6']
      
        #Expectation in utility function
        UL2 = 1/self.data.num_L_scen*gp.quicksum(EuL2 + SML2) 
        
        #Threat Point for L2
        Zeta_L2=self.data.net_earnings_no_contract_L2  

        #Objective function
        objective = (UG6 -Zeta_G6) *(UL2-Zeta_L2)
        self.model.setObjective(gp.quicksum(objective))
        self.model.update()

    def _save_results(self):

        #Plot income distributions for generator and load with contract and without contract

        # save objective value
        self.results.objective_value = self.model.ObjVal
        # save strike price
        self.results.strike_price = self.variables.S.x
        # save contract amount
        #self.results.contract_amount = self.model.M.x

        #Temporary save for contract amount
        self.results.contract_amount = 300 # not actual 
        # save utility for G6
        print("normal incoem")
        EuG6 =  (self.data.price['N3']-self.data.generator_cost_a['G6'])*self.data.generator_production['G6']-self.data.generator_cost_b['G6']*(self.data.generator_production['G6']**2)
        print(EuG6)
        SMG6 =  (self.results.strike_price - self.data.price['N3'])*self.results.contract_amount
        print("contract")
        print(SMG6)
        CVaRG6 = self._CVaR((-EuG6-SMG6))
        print("CVAR")
        print(CVaRG6)
        UG6 = 1/len(self.data.TIME)*(EuG6 + SMG6) - self.data.A_G6[1]*CVaRG6
        self.results.utility_G6 = UG6

        # save utility for L2
        EuL2 = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price['N3'])
        SML2 =  (self.data.price['N3'] - self.results.strike_price)*self.results.contract_amount
        CVaRL2 = self._CVaR((-EuL2-SML2))
        UL2 = 1/len(self.data.TIME)*(EuL2 + SML2) - self.data.A_L2[1]*CVaRL2
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
        return
    
    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")
    
if __name__ =='__main__':

    (GENERATORS, LOADS, NODES,TIME, fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity, load_capacity, 
            mapping_buses, mapping_generators, mapping_loads, num_L_scen,branch_capacity_df, bus_reactance_df, System_data,strikeprice_min,
            retail_price,strikeprice_min,strikeprice_max,contract_amount_min,contract_amount_max,A_L2,A_G6,K_L2,K_G6,alpha) = load_data()
    
    input_data = InputData(
        GENERATORS = GENERATORS, 
        LOADS = LOADS,
        NODES = NODES,
        TIME = TIME,
        generator_cost_fixed= fixed_Cost,
        generator_cost_a = generator_cost_a,
        generator_cost_b = generator_cost_b,
        generator_capacity = generator_capacity,
        load_capacity = load_capacity,
        num_L_scen = num_L_scen,
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
    cm = ContractNegotiation(input_data,opf_model.results,risk)
    cm.run()
    cm.display_results()

    # No Risk
    risk = False
    cm = ContractNegotiation(input_data,opf_model.results,risk)
    cm.run()
    cm.display_results()

    print("done")
    print("2")