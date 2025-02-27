import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

#Baseload contract with CFD scheme

# Load data
# Load profile with variability (MW per hour for a 24-hour period)
base_load_profile = np.array([300, 280, 270, 260, 250, 240, 230, 250, 300, 350, 400, 420, 
                              430, 420, 410, 400, 390, 380, 370, 360, 350, 340, 320, 310])
load_variability = np.random.normal(0, 30, (1000, 24))  # Variability in load
load_scenarios = base_load_profile + load_variability
load_scenarios = np.clip(load_scenarios, 200, 500)  # Ensuring reasonable limits

# No contract utility levels
u_genco_no_contract = 0  # Assume GenCo has 0 net earnings without contract
u_lse_no_contract = 0    # Assume LSE has 0 net earnings without contract

# Risk aversion factors
risk_aversion_genco = 1.0
risk_aversion_lse = 1.5

# Retail Price for load 
retail_price = 25 #$/MWh


def expected_earnings_genco(Q, S):
    return lambda_hour * PG - Cost + (S-lambda_hour)*M  # Example: selling electricity at strike price minus cost

def expected_earnings_lse(Q, S):
    return Q * (50 - S)  # Example: buying electricity at strike price lower than retail price

#Utility function

#

baseload = gp.Model("Nash Bargaining")

# Decision variables
S_price = baseload.addVar(lb=0, vtype=GRB.CONTINUOUS, name="q")
M_amount = baseload.addVar(lb=0, vtype=GRB.CONTINUOUS, name="p")
Cost  = 0

# PG = baseload (set amount)
PG = 300

# Objective function

Objective_Generator = gp.quicksum(lambda_hour[t] * PG - Cost + (S_price-lambda_hour)*M_amount for t in range(24))
Objective_LSE = gp.quicksum(M_amount[t] * (50 - S_price) for t in range(24))


baseload.setObjective(,GRB.MAXIMIZE)

# Constraints 
#Strike Price constraints 
baseload.addConstr(S_price >= 15)
baseload.addConstr(S_price <= 25)
#Contracted M amount constraints
baseload.addConstr(M_amount >= 0)
baseload.addConstr(M_amount >= 600)


class OptimalPowerFlow():

    def __init__(self, input_data: InputData): # initialize class
        self.data = input_data          # define data attributes
        self.variables = Expando()      # define variable attributes
        self.constraints = Expando()    # define constraints attributes
        self.results = Expando()    
        self.resti = Expando    # define results attributes
        self._build_model()             # build gurobi model