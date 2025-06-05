"""
Contract negotiation implementation using Nash bargaining solution.
"""
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import GRB
from utils import Expando, build_dataframe, calculate_cvar_left, simulate_price_scenarios,analyze_price_distribution, calculate_cvar_right
from Barter_Set import Barter_Set
from tqdm import tqdm
import scipy.stats as stats
from matplotlib.patches import Polygon # Import added at the top of the file
import seaborn as sns
import os
from numpy.random import laplace


class ContractNegotiation:
    """
    Implements contract negotiation between generator and load using Nash bargaining.
    """
    def __init__(self, input_data, provider,old_obj_func=False):
        self.data = input_data
        self.provider = provider
        self.old_obj_func = old_obj_func
        self.results = Expando()
        self.scipy_results = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        self._build_statistics()
        self._build_model()

        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def _build_statistics(self):
        """Calculate statistical measures needed for negotiation."""
        # Load data from Forecast or OPF 

        price_mat  = self.provider.price_matrix() # if forecast is used ('N3') is redudant, but causes no error 
        prod_mat   = self.provider.production_matrix() # # if forecast is used ('G') is redudant, but causes no error 
        CR_mat  = self.provider.capture_rate_matrix()
        # Days in per month
        #self.data.days_in_month = prod_mat.index.to_series().groupby(prod_mat.index.to_period('M')).first().dt.days_in_month
        #self.data.hours_in_day = 24# hours in a day 
        #self.data.hours_in_month = self.data.days_in_month * self.data.hours_in_day # hours in a month
        self.data.hours_in_year = 8760 # hours in a year    
        self.data.PROB = self.provider.probability
        self.data.price_true = price_mat
        self.data.capture_rate = CR_mat.values
        self.data.retail_price = 0

        self.data.retail_aggregated = self.data.retail_price *  self.data.hours_in_year *1000  # Retail price aggregated over time periods $/GWh
        
        # Just temporary upscaling load capacity to match production to some extent
        self.data.load_scenarios = self.data.load_scenarios *0.75 # Upscale load capacity for Load
        
        #self.data.K_G = -0.6
        #self.data.K_L = -0.7

        # Calculate biased prices
        self.data.price_G = (self.data.K_G * self.data.price_true.mean()) + self.data.price_true
        self.data.price_L = (self.data.K_L * self.data.price_true.mean()) + self.data.price_true

       
        # Generator production
        self.data.generator_production_df = pd.DataFrame(prod_mat)
        self.data.generator_production = prod_mat
        self.data.price_mean_true = self.data.price_true.mean() # Mean price at node N3
        self.data.price_std_true = self.data.price_true.std(ddof=1) # Std price at node N3

        # for adding fatter tail to tdistirbution
        #scale = 0.15 * self.data.price_true.mean()  # or tweak as needed
        #noise = laplace(loc=0.0, scale=scale, size=self.data.price_true.shape)
        #self.data.price_true += noise

        #Make this a call from utility function 
        self.data.net_earnings_no_contract_G_df=  self.data.capture_rate * self.data.generator_production_df*self.data.price_true
        # True net earnings for G given correct price distribution
        self.data.net_earnings_no_contract_true_G = self.data.net_earnings_no_contract_G_df.sum(axis=0) # Generator G earnings in numpy array for calculations ( Total Sum per Scenario)
        # Biased net earnings for G given biased price distribution K
        self.data.net_earnings_no_contract_priceG_G_df = self.data.capture_rate * self.data.generator_production_df*self.data.price_G
        self.data.net_earnings_no_contract_priceG_G =  self.data.net_earnings_no_contract_priceG_G_df.sum()
        # Load L earnings in numpy array for calculations
        self.data.net_earnings_no_contract_true_L = (self.data.load_scenarios*( - self.data.price_true)).sum(axis=0)
        # Load L earnings in numpy array for calculations
        self.data.net_earnings_no_contract_priceL_L = (self.data.load_scenarios*(-self.data.price_L)).sum(axis=0)
        
        #CVaR calculation of No contract (TRue )
        self.data.CVaR_no_contract_priceG_G = calculate_cvar_left(self.data.net_earnings_no_contract_priceG_G,self.data.alpha) # CVaR for G
        self.data.CVaR_no_contract_priceL_L = calculate_cvar_left(self.data.net_earnings_no_contract_priceL_L,self.data.alpha)

        # New Utility Function 
        self.data.Zeta_G=(1-self.data.A_G)*self.data.net_earnings_no_contract_priceG_G.mean() + self.data.A_G*self.data.CVaR_no_contract_priceG_G
        self.data.Zeta_L=(1-self.data.A_L)*self.data.net_earnings_no_contract_priceL_L.mean() + self.data.A_L*self.data.CVaR_no_contract_priceL_L

        # Threat Points for generator and Load 
      
        num_scenarios = self.data.price_true.shape[1]  # Match your existing number of scenarios
        time_periods = self.data.price_true.shape[0] # Match your existing time periods
        time_periods_hours = time_periods * self.data.hours_in_year  # Total hours in the time period
            
        self.data.lambda_sum_true_per_scenario = self.data.price_true.sum(axis=0) # Sum across time periods for each scenario
        self.data.lambda_sum_G_per_scenario = self.data.price_G.sum(axis=0) # Sum across time periods for each scenario
        self.data.lambda_sum_L_per_scenario = self.data.price_L.sum(axis=0) # Sum across time periods for each scenario
        # Calculate E^P(λ∑) - Expected value of the sum over T (using TRUE distribution)
        self.data.expected_lambda_sum_true = self.data.lambda_sum_true_per_scenario.mean()
        
        # Capture Prices and Capture Rate 

        self.data.Capture_price = self.data.price_true * self.data.capture_rate  # Capture price for G (per month)
        self.data.Capture_price_h = self.data.Capture_price / self.data.hours_in_year  # Capture price per hour for G
        self.data.Capture_rate = self.data.Capture_price / self.data.price_true.mean(axis=0)  # Capture rate for G (should it be the total mean price or the mean price per scenario?)

        # Average Capture Price and Rate 
        self.data.Capture_price_avg = self.data.Capture_price_h.mean()
        Capture_rate_avg = self.data.Capture_rate.mean()

        # Calculate CVaR^P(λ∑) - CVaR of the sum over T (using TRUE distribution)
        # Assumes calculate_cvar returns the expected value of the variable *given* it's in the alpha-tail
        self.data.left_Cvar_capture_lambda_sum_true = calculate_cvar_left(self.data.Capture_rate*self.data.lambda_sum_true_per_scenario, self.data.alpha)
        self.data.left_cvar_lambda_sum_true = calculate_cvar_left(self.data.lambda_sum_true_per_scenario, self.data.alpha)
        self.data.right_cvar_lambda_sum_true = calculate_cvar_right(self.data.lambda_sum_true_per_scenario, self.data.alpha)

        # Calculate CVaR^P(-λ∑) - CVaR of the negative sum over T (using TRUE distribution)
        # This corresponds to the risk of low LMPs
        self.data.left_Cvar_neg_lambda_sum_true = calculate_cvar_left(-self.data.lambda_sum_true_per_scenario, self.data.alpha) 
        self.data.left_Cvar_neg_capture_lambda_sum_true = calculate_cvar_left(-self.data.Capture_rate*self.data.lambda_sum_true_per_scenario, self.data.alpha) 
        self.data.right_cvar_neg_lambda_sum_true = calculate_cvar_left(-self.data.lambda_sum_true_per_scenario, self.data.alpha)

        # Determine the *absolute* bias constants K_G and K_L for the equations
        # Based on Theorem 1, K_G/K_L are absolute shifts.:
        self.data.K_G_lambda_Sigma = self.data.K_G * self.data.expected_lambda_sum_true
        self.data.K_L_lambda_Sigma = self.data.K_L * self.data.expected_lambda_sum_true

       
        # Using the current variables directly as the absolute K values:
  
        # Calculate the denominators T * (1 + A)
        denom_G = time_periods * (1 + self.data.A_G)
        denom_L = time_periods * (1 + self.data.A_L)

        # Avoid division by zero if A_G or A_L happens to be -1 (unlikely but safe)
        if abs(denom_G) < 1e-9: denom_G = 1e-9 # Use a small number instead of zero
        if abs(denom_L) < 1e-9: denom_L = 1e-9 # Use a small number instead of zero

        ############################################## Old defintion ##########################################################################

        # Calculate Term 1 (GenCo perspective, risk of high LMPs)
        # E^P(λ∑) + A_G*CVaR^P(λ∑) + (1 + A_G)*K_G
        term1_G_org =((self.data.expected_lambda_sum_true + self.data.A_G * self.data.right_cvar_lambda_sum_true + (1 + self.data.A_G) *  self.data.K_G_lambda_Sigma)) / denom_G
        testterm1_G_org = (self.data.expected_lambda_sum_true + self.data.A_G * self.data.left_cvar_lambda_sum_true + (1 + self.data.A_G) *  self.data.K_G_lambda_Sigma) / denom_G

        # Calculate Term 2 (GenCo perspective, risk of low LMPs -> using CVaR of negative sum)
        # E^P(λ∑) - A_G*CVaR^P(-λ∑) + (1 + A_G)*K_G
        term2_G_org = ((self.data.expected_lambda_sum_true - self.data.A_G * self.data.right_cvar_neg_lambda_sum_true + (1 + self.data.A_G) *  self.data.K_G_lambda_Sigma)) / denom_G
        testterm2_G_org = (self.data.expected_lambda_sum_true - self.data.A_G * self.data.left_Cvar_neg_lambda_sum_true + (1 + self.data.A_G) *  self.data.K_G_lambda_Sigma) / denom_G

        # Calculate Term 3 (LSE perspective, risk of low LMPs, for SR*)
        # E^P(λ∑) + (1 + A_L)*K_L - A_L*CVaR^P(-λ∑)
        term3_L_SR_org = (self.data.expected_lambda_sum_true + (1 + self.data.A_L) *self.data.K_L_lambda_Sigma - self.data.A_L * self.data.right_cvar_neg_lambda_sum_true) / denom_L
        testterm4_L_SU_org = (self.data.expected_lambda_sum_true + (1 + self.data.A_L) *self.data.K_L_lambda_Sigma - self.data.A_L * self.data.left_Cvar_neg_lambda_sum_true ) / denom_L

        # Calculate Term 4 (LSE perspective, risk of high LMPs, for SU*)
        # E^P(λ∑) + (1 + A_L)*K_L + A_L*CVaR^P(λ∑)
        term4_L_SU_org = (self.data.expected_lambda_sum_true + (1 + self.data.A_L) * self.data.K_L_lambda_Sigma + self.data.A_L * self.data.right_cvar_lambda_sum_true) / denom_L
        testterm3_L_SR_org = (self.data.expected_lambda_sum_true + (1 + self.data.A_L) * self.data.K_L_lambda_Sigma + self.data.A_L * self.data.left_cvar_lambda_sum_true) / denom_L

        # Calculate SR* using Equation (27) - Minimum of the relevant terms
        self.data.SR_star_old = np.min([term1_G_org, term2_G_org, term3_L_SR_org]) * 1e3 # Convert from $/GWh to $/MWh
        testsr = np.min([testterm1_G_org, testterm2_G_org, testterm3_L_SR_org]) *1e3 # Convert from $/GWh to $/MWh
        testsu = np.max([testterm1_G_org, testterm2_G_org, testterm4_L_SU_org] ) * 1e3 # Convert from $/GWh to $/MWh

        # Calculate SU* using Equation (28) - Maximum of the relevant terms
        self.data.SU_star_old = np.max([term1_G_org, term2_G_org, term4_L_SU_org]) * 1e3


        # Print calculated critical bounds for verification during runtime
        print(f"Calculated SR* using original (Eq 27): {self.data.SR_star_old:.5f}")
        print(f"Calculated SU* using original (Eq 28): {self.data.SU_star_old:.5f}")

        print(f"Calculated SR* using test left (Eq 27): {testsr:.5f}")
        print(f"Calculated SU* using test left (Eq 28): {testsu:.5f}")

        # New Objective function defition 
        
        
        self.data.term1_G_new =(( (1-self.data.A_G) * self.data.expected_lambda_sum_true +self.data.K_G_lambda_Sigma ) - self.data.A_G * self.data.left_Cvar_neg_capture_lambda_sum_true)/ time_periods    # SR* numerator for Gen

        # high-price risk (feeds S_U*)
        self.data.term2_G_new = (((1-self.data.A_G) * self.data.expected_lambda_sum_true + self.data.K_G_lambda_Sigma ) + self.data.A_G * self.data.left_Cvar_capture_lambda_sum_true) / time_periods     # SU* numerator for Gen
  
        # Load-serving entity ––––––––––––––––––––––––––––––––––––––––––––––
        # low-price risk  (feeds S_R*)
        self.data.term3_L_SR_new = (self.data.expected_lambda_sum_true
                        + self.data.A_L * self.data.left_cvar_lambda_sum_true
                        + self.data.K_L_lambda_Sigma - self.data.A_L * self.data.expected_lambda_sum_true  ) / time_periods   # SU* numerator for LSE
        # high-price risk (feeds S_U*)
        self.data.term4_L_SU_new = ( self.data.expected_lambda_sum_true
                        - self.data.A_L * self.data.left_Cvar_neg_lambda_sum_true
                        + self.data.K_L_lambda_Sigma  - self.data.A_L * self.data.expected_lambda_sum_true )  / time_periods   # SR* numerator for LSE




         # Calculate SR* using Equation (27) - Minimum of the relevant terms
        self.data.SR_star_new = np.min([self.data.term1_G_new, self.data.term2_G_new, self.data.term3_L_SR_new])   # Convert from $/GWh to $/MWh

        # Calculate SU* using Equation (28) - Maximum of the relevant terms
        self.data.SU_star_new = np.max([self.data.term1_G_new, self.data.term2_G_new, self.data.term4_L_SU_new])  

        
        # Print calculated critical bounds for verification during runtime
        print(f"Calculated SR* using New (Eq 27) (Yearly Price [Mio EUR/GWh]): {self.data.SR_star_new:.5f}")
        print(f"Calculated SU* using new (Eq 28) (Yearly Price [Mio EUR/GWh]: {self.data.SU_star_new:.5f}")

        print(f"Calculated SR* using New (Eq 27) (Hourly Price [EUR/MWh]): {self.data.SR_star_new/self.data.hours_in_year*1e3:.5f}")
        print(f"Calculated SU* using new (Eq 28) (Hourly Price [EUR/MWh]: {self.data.SU_star_new/self.data.hours_in_year*1e3:.5f}")
        print()


    def _build_variables(self):
        """Build optimization variables for contract negotiation."""
        # Auxiliary variables for logaritmic terms
        EPS = 1e-4               # pick something natural for your scale
        self.variables.arg_G = self.model.addVar(lb=EPS, name="UG_minus_ZetaG")
        self.variables.arg_L = self.model.addVar(lb=EPS, name="UL_minus_ZetaL")

        # Define logarithmic terms
        self.variables.log_arg_G = self.model.addVar(lb=EPS,name="log_arg_G")
        self.variables.log_arg_L = self.model.addVar(lb=EPS,name="log_arg_L")

           # build strike price variables
        self.variables.S = self.model.addVar(
            lb=self.data.strikeprice_min * self.data.hours_in_year * 1e-3,  # Convert to $/MWh
            ub=self.data.strikeprice_max * self.data.hours_in_year * 1e-3,
            name='Strike_Price'
        )

        # build contract amount variables
        self.variables.M = self.model.addVar(
            lb=self.data.contract_amount_min,
            ub=self.data.contract_amount_max * self.data.hours_in_year * 1e-3,
            name='Contract Amount'
        )
   
        self.variables.zeta_G = self.model.addVar(
            name='Zeta_Auxillary_G',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
        self.variables.zeta_L = self.model.addVar(
            name='Zeta_Auxillary_L',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
      
        self.variables.eta_G = {s: self.model.addVar(
            name='Auxillary_Variable_G_{0}'.format(s),
            lb=0,ub=gp.GRB.INFINITY)
            for s in self.data.SCENARIOS_L
        }
        self.variables.eta_L = {s: self.model.addVar(
            name='Auxillary_Variable_L_{0}'.format(s),
            lb=0,ub=gp.GRB.INFINITY)
            for s in self.data.SCENARIOS_L
        }

        self.model.update() 

    def _build_constraints(self):
        """Build constraints for contract negotiation."""
         # Single Strike Price Constraint
        self.constraints.strike_price_constraint_max =  self.model.addLConstr(
        self.variables.S<=self.data.strikeprice_max * 1e-3 * self.data.hours_in_year,
        name='Strike_Price_Constraint_Max'
        )
        
        
        self.constraints.strike_price_constraint_min =  self.model.addLConstr(
        self.variables.S>=self.data.strikeprice_min *  1e-3 * self.data.hours_in_year ,
        name='Strike_Price_Constraint_Max'
        )

        #Contract amounts 
        
        #Contract amount constraints
        self.constraints.contract_amount_constraint_max = self.model.addLConstr(
           self.variables.M <= self.data.contract_amount_max * 1e-3 * self.data.hours_in_year,  # Convert to GWh/year
            name='Contract Amount Constraint Max'
        )
         #Contract amount constraints
        self.constraints.contract_amount_constraint_min = self.model.addLConstr(
            self.data.contract_amount_min <= self.variables.M ,  # Convert to GWh/year
            name='Contract Amount Constraint Min'
        )

        #Logarithmic constraints
        self.model.addGenConstrLog(self.variables.arg_G, self.variables.log_arg_G, "log_G")
        self.model.addGenConstrLog(self.variables.arg_L, self.variables.log_arg_L, "log_L")
        
        self.constraints.eta_G_constraint = {
            s: self.model.addConstr(
            #With costs
            #self.variables.eta_G[s] >=self.variables.zeta_G - gp.quicksum((self.data.price_G[t,s]-self.data.generator_cost_a['G'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G']*(self.data.generator_production[t,s]*self.data.generator_production[t,s])
            #                                                                +( self.variables.S- self.data.price_G[t,s])*self.variables.M for t in self.data.TIME),
            self.variables.eta_G[s] >=(self.variables.zeta_G - gp.quicksum(self.data.capture_rate[t,s]*self.data.price_G.iat[t,s]*self.data.generator_production.iat[t,s]
                                                                           +( (self.variables.S)- self.data.price_G.iat[t,s])*(self.variables.M) for t in self.data.TIME)),
            name='Eta_Aversion_Constraint_G_in_scenario_{0}'.format(s)
        )
        for s in self.data.SCENARIOS_L
        }
        self.constraints.eta_L_constraint = {
            s: self.model.addConstr(
            self.variables.eta_L[s] >=(self.variables.zeta_L - gp.quicksum(self.data.load_scenarios[t,s]*(self.data.retail_aggregated-self.data.price_L.values[t,s])
                                                                            +(self.data.price_L.values[t,s] - (self.variables.S))*(self.variables.M) for t in self.data.TIME)),
            name='Eta_Aversion_Constraint_L_in_scenario_{0}'.format(s)
        )
        for s in self.data.SCENARIOS_L
        }
        

        self.model.update()

    def _build_objective(self):
        """Build the objective function for contract negotiation."""
        #Expected Earnings function for G 
        #EuG =  gp.quicksum((self.data.price_G[t,s]-self.data.generator_cost_a['G'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G']*(self.data.generator_production[t,s]*self.data.generator_production[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        EuG =  gp.quicksum(self.data.capture_rate[t,s]*self.data.price_G.iat[t,s]*self.data.generator_production.iat[t,s] for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG =  gp.quicksum((self.variables.S- self.data.price_G.iat[t,s])*(self.variables.M ) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #CVaR for G 
        CVaRG = self.variables.zeta_G - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_G[s])  for s in self.data.SCENARIOS_L)


        #Expected Earnings for L
        EuL = gp.quicksum(self.data.load_scenarios[t,s]*(self.data.retail_aggregated-self.data.price_L.iat[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML =  gp.quicksum((self.data.price_L.iat[t,s] - (self.variables.S))*(self.variables.M)for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #CvaR for L
        CVaRL = self.variables.zeta_L - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_L[s])  for s in self.data.SCENARIOS_L)

        #Build Utiltiy Functions 
        if self.old_obj_func == True:
            UG = self.data.PROB*(EuG+SMG) +  self.data.A_G*CVaRG
            UL = self.data.PROB*(EuL+ SML) + self.data.A_L*CVaRL
        
            #UG = (1/(1+self.data.A_G)) * (self.data.PROB*(EuG+SMG) + self.data.A_G*CVaRG) 
            #UL = (1/(1+self.data.A_L)) * (self.data.PROB*(EuL+ SML) + self.data.A_L*CVaRL)
        else:
            UG = (1-self.data.A_G)*self.data.PROB*(EuG+SMG) + self.data.A_G*CVaRG
            UL = (1-self.data.A_L)*self.data.PROB*(EuL+SML) + self.data.A_L*CVaRL
           


         # Link auxiliary variables to expressions
        self.model.addConstr(self.variables.arg_G == (UG - self.data.Zeta_G), "arg_G_constr")
        self.model.addConstr(self.variables.arg_L == (UL - self.data.Zeta_L), "arg_L_constr")

        self.model.update()

        # Normal Objective function 
        #objective = (UG - self.data.Zeta_G) * (UL-self.data.Zeta_L)
        #self.model.setObjective(objective,sense = gp.GRB.MAXIMIZE)
        
        # Set logarithmic objective
        self.model.setObjective(
        (self.variables.log_arg_G + self.variables.log_arg_L),
        GRB.MAXIMIZE) 
        self.model.update()

    def _build_model(self):
        """Initialize and build the complete optimization model."""
        self.model = gp.Model(name='Nash Bargaining Model')
        self.model.Params.NonConvex = 2
        self.model.Params.FeasibilityTol = 1e-6
        self.model.Params.OutputFlag = 1
        self.model.Params.TimeLimit = 30
        self.model.Params.ObjScale   = 1e-6
        #self.model.Params.NumericFocus = 1
        
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
        """Save optimization results."""
        # Save objective value, strike price, and contract amount
        self.results.objective_value = self.model.ObjVal
        self.results.strike_price = self.variables.S.x
        self.results.strike_price_hour = self.results.strike_price / self.data.hours_in_year * 1000  # Convert to $/MWh
        self.results.contract_amount = self.variables.M.x  # Convert to GWh/year
        self.results.contract_amount_hour = self.results.contract_amount / self.data.hours_in_year  * 1000  # Convert to GWh/hour
        self.results.capture_price = self.data.Capture_price_avg

        # Save their 'actual' values based on the true distribution 
        # Calculate revenues with contract
        EuG = self.data.net_earnings_no_contract_priceG_G

        SMG = ((self.results.strike_price - self.data.price_G) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

        # Calculate CVaR for results G 
        self.results.CVaRG = calculate_cvar_left(EuG + SMG, self.data.alpha)

        # Calculate revenues for L
        EuL = self.data.net_earnings_no_contract_priceL_L
        SML = ((self.data.price_L - self.results.strike_price) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

        # Calculate CVaR for L
        self.results.CVaRL = calculate_cvar_left(EuL + SML, self.data.alpha)


        self.results.utility_G = (1-self.data.A_G) * (EuG + SMG).mean() + self.data.A_G * self.results.CVaRG
        self.results.utility_L = (1-self.data.A_L) * (EuL + SML).mean() + self.data.A_L * self.results.CVaRL
        

        self.results.Nash_Product = ((self.results.utility_G - self.data.Zeta_G) * (self.results.utility_L - self.data.Zeta_L))
        # Save accumulated revenues
        self.results.accumulated_revenue_True_G = EuG + SMG.sum(axis=0)
        self.results.accumulated_revenue_True_L = EuL + SML.sum(axis=0)

        # Save Biased (What they expected)

    def run(self):
        """Run the optimization model."""
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            self._save_results()
            self.scipy_optimization()
            #self.display_results()
            self.scipy_display_results()
            BS = Barter_Set(self.data,self.results,self.scipy_results,self.old_obj_func)
            BS.Plotting_Barter_Set()
        
        else:
            #self.model.write("model.lp")
            #self.model.write("model.rlp")
            #self.model.write("model.mps")
            #self._save_results()
            BS = Barter_Set(self.data,self.results,self.scipy_results,self.old_obj_func)
            self.scipy_optimization()
            #self.scipy_display_results()
            BS.Plotting_Barter_Set()
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        """Display optimization results."""
        print("\n-------------------   RESULTS GUROBI  -------------------")
        print(f"Optimal Objective Value (Expected Value) (Log): {self.results.objective_value}")
        print(f"Optimal Objective Value (Expected Value): {np.exp(self.results.objective_value)}")
        print(f"Nash Product with optimal S and M: {self.results.Nash_Product}")
        print(f"Optimal Strike Price: {self.results.strike_price_hour}")
        print(f"Optimal Contract Amount: {self.results.contract_amount_hour}")
        print(f"Optimal Utility for G: {self.results.utility_G}")
        print(f"Optimal Utility for L: {self.results.utility_L}")
        print(f"Threat Point G: {self.data.Zeta_G}")
        print(f"Threat Point L: {self.data.Zeta_L}")

    def manual_optimization(self, plot=False, filename=None):
        """
        Perform manual optimization by grid search over strike prices.
        """
        strike_prices = np.linspace(self.data.strikeprice_min+3, self.data.strikeprice_max-3, 200)
        contract_amounts = np.linspace(self.data.contract_amount_min+150, self.data.contract_amount_max-150, 200)
        #M = 300  # Fixed contract amount

        utilities_G = np.zeros((len(strike_prices), len(contract_amounts)))
        utilities_L = np.zeros((len(strike_prices), len(contract_amounts)))
        combined_utilities = np.zeros((len(strike_prices), len(contract_amounts)))
        combined_utilities_log = np.zeros((len(strike_prices), len(contract_amounts)))


        """
        Could speed up by making SMG and SML numpya arrays from the start 
        """

        for i,strike in enumerate(tqdm(strike_prices, desc='loading...')):
            for j,M in enumerate(contract_amounts):
               
                # Calculate revenues for G
                SMG = (strike - self.data.price_G) * M
                

                # Calculate revenues for L
                SML = ( self.data.price_L-strike) * M

                # Calculate total revenues
                Scen_revenue_G = SMG.sum(axis=0) + self.data.net_earnings_no_contract_G
                Scen_revenue_L = SML.sum(axis=0) + self.data.net_earnings_no_contract_L

                # Calculate CVaR
                CVaRG = calculate_cvar_left(Scen_revenue_G.values, self.data.alpha)
                CVaRL = calculate_cvar_left(Scen_revenue_L, self.data.alpha)

                # Calculate utility using old and new objective functions
                if self.old_obj_func == True:
                    UG = (Scen_revenue_G.mean() + self.data.A_G * CVaRG)
                    UL = (Scen_revenue_L.mean() + self.data.A_L * CVaRL)
                    #UG = (1/(1+self.data.A_G)) * (Scen_revenue_G.mean() + self.data.A_G * CVaRG)
                    #UL = (1/(1+self.data.A_L)) * (Scen_revenue_L.mean() + self.data.A_L * CVaRL)
                else:
                    UG = (1 - self.data.A_G) * Scen_revenue_G.mean() + self.data.A_G * CVaRG
                    UL = (1 - self.data.A_L) * Scen_revenue_L.mean() + self.data.A_L * CVaRL

               


                # Calculate the Nash product (combined utility)
                combined_utility = (UG - self.data.Zeta_G ) * (UL - self.data.Zeta_L) 
                # Calculate the Nash product (combined utility)
                combined_utility_log = np.log(np.maximum(UG - self.data.Zeta_G , 1e-10)) + np.log(np.maximum(UL - self.data.Zeta_L , 1e-10)) # Avoid log(0) by using a small value
                combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
                """
                if combined_utility_log <0:
                    log_abs_diff_G = np.log(np.maximum(np.abs(UG - self.data.Zeta_G), 1e-10))
                    log_abs_diff_L = np.log(np.maximum(np.abs(UL - self.data.Zeta_L), 1e-10))
                    # Sum the logarithms
                    combined_utility_log = (log_abs_diff_G + log_abs_diff_L)

                    if combined_utility_log -combined_utility <1e-6 and strike < self.data.special_case_min_strike:
                        self.data.special_case_min_strike = strike
                    if combined_utility_log - combined_utility <1e-6 and strike > self.data.special_case_max_strike:
                        self.data.special_case_max_strike = strike
                """
                utilities_G[i,j] = UG
                utilities_L[i,j] = UL
                combined_utilities[i,j] = combined_utility
                combined_utilities_log[i,j] = combined_utility_log

       # Find indices of maximum utilities
        max_idx = np.unravel_index(np.argmax(combined_utilities), combined_utilities.shape)
        max_idx_log = np.unravel_index(np.argmax(combined_utilities_log), combined_utilities_log.shape)

        # Get optimal values
        optimal_strike_price = strike_prices[max_idx[0]]
        optimal_amount = contract_amounts[max_idx[1]]
        optimal_strike_price_log = strike_prices[max_idx_log[0]]
        optimal_amount_log = contract_amounts[max_idx_log[1]]

        # Get maximum utilities
        max_utility_G = utilities_G[max_idx]
        max_utility_L = utilities_L[max_idx]
        max_combined_utility = combined_utilities[max_idx]
        max_combined_utility_log = combined_utilities_log[max_idx_log]

        """
        Works if M is kept constant
        # Find first non-negative value
        first_positive_idx = next((i for i, x in enumerate(combined_utilities) if x >= 0), None)
        last_positive_idx = next((i for i, x in enumerate(reversed(combined_utilities)) if x >= 0), None)

        if first_positive_idx is not None:
            print(f"\nFirst non-negative combined utility:")
            print(f"Strike price: {strike_prices[first_positive_idx]:.5f}")

        if last_positive_idx is not None:
            last_positive_idx = len(combined_utilities) - last_positive_idx - 1
            print(f"\nLast non-negative combined utility:")
            print(f"Strike price: {strike_prices[last_positive_idx]:.5f}")
        """



        # Include threat points for comparison
        print("\n-------------------   RESULTS Iterative  -------------------")
        print(f"Threat Point G: {self.data.Zeta_G}")
        print(f"Threat Point L: {self.data.Zeta_L}")
        print(f'Risk Aversion G: {self.data.A_G}')
        print(f'Risk Aversion G: {self.data.A_L}')

        print(f"Optimal Strike Price: {optimal_strike_price}")
        print(f"Optimal Strike Price log: {optimal_strike_price_log}")
        print(f"Optimal  Amount: {optimal_amount}")
        print(f"Optimal Amount log: {optimal_amount_log}")
        print(f"Maximum Utility G: {max_utility_G}")
        print(f"Maximum Utility L: {max_utility_L}")

        print(f"Maximum Combined Utility (Nash Product): {max_combined_utility}")
        #print(f"Maximum Combined Utility Log (Nash Product): {np.exp(combined_utilities_log[max_combined_utility_index_log])}")
        print(f"Maximum Combined Utility Log (Nash Product): {max_combined_utility_log}")

        if plot == True:
            print("Plotting results...")
            #Example of how it looks - Would have to specify Amount to better see the results
            # Create Plots directory if it doesn't exist
           

            # Plot the utilities and Nash product for different strike prices
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))

            # First subplot: Utilities for G and L
            axs[0].plot(strike_prices, utilities_G, label="Utility G", color="blue")
            axs[0].plot(strike_prices, utilities_L, label="Utility L", color="green")
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

            """
            # Second subplot: Nash Product and Log Nash Product
            axs[2].plot(utilities_G, utilities_L,  color="orange", linestyle="-")
            axs[2].set_xlabel("Utility G")
            axs[2].set_ylabel("Utility L")
            axs[2].set_title("Utility G vs Utility L")
            axs[2].grid()
            fig.suptitle(f"Utilities and Nash Product for Different Strike Prices with with Risk AG{self.data.A_G} and Risk AL{self.data.A_L}")
            """
            # Adjust layout and show the plot
            plt.tight_layout()
            if filename:
                filepath = os.path.join(self.plots_dir, filename)
                plt.savefig(filepath)
                print(f"Histogram plot saved to {filepath}")
                plt.close(fig) # Close the figure after saving
            else:
                plt.show()

    def batch_manual_optimization(self, A_G_values, A_L_values, strike_min=None, strike_max=None, filename=None):
        """
        Perform manual optimization for multiple risk aversion values and plot combined results.
        
        Args:
            A_G_values: Array of GenCo risk aversion values
            A_L_values: Array of LSE risk aversion values
            strike_min: Minimum strike price (optional)
            strike_max: Maximum strike price (optional)
            filename: Optional filename to save the plot
        """
        # Update strike price bounds if provided
        if strike_min is None:
            strike_min = self.data.strikeprice_min
        if strike_max is None:
            strike_max = self.data.strikeprice_max 

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define strike prices once
        strike_prices = np.linspace(self.data.strikeprice_min, self.data.strikeprice_max, 200)

        # Create a color map for AG values - using tab20 for more distinct colors
        base_colors = plt.cm.tab20(np.linspace(0, 1, len(A_G_values)))
        # Number of gradient steps for each AL value
        gradient_steps = len(A_L_values)

        for ag_idx, a_G in enumerate(tqdm(A_G_values, desc='loading batch...')):
            base_color = base_colors[ag_idx]
            # Create gradient colors for this AG value - using a wider range of alpha values
            alpha_values = np.linspace(0.5, 1.0, gradient_steps)  # Increased minimum alpha for better visibility
            
            for al_idx, a_L in enumerate(A_L_values):
                utilities_G = []
                utilities_L = []
                combined_utilities_log = []
                
                # Update risk aversion values
                A_G = a_G
                A_L = a_L
                
                for strike in tqdm(strike_prices, desc='loading...'):
                    # Calculate revenues for G
                    SMG = (strike - self.data.price_G) * self.results.contract_amount

                    # Calculate revenues for L
                    SML = ( self.data.price_L-strike) * self.results.contract_amount

                    # Calculate total revenues
                    Scen_revenue_G = SMG.sum(axis=0) + self.data.net_earnings_no_contract_G
                    Scen_revenue_L = SML.sum(axis=0) + self.data.net_earnings_no_contract_L

                    # Calculate CVaR
                    CVaRG = calculate_cvar_left(Scen_revenue_G.values, self.data.alpha)
                    CVaRL = calculate_cvar_left(Scen_revenue_L.values, self.data.alpha)

                    # Calculate utility for G
                    if self.old_obj_func == True:
                        UG = (Scen_revenue_G.mean() + A_G * CVaRG)
                        UL = (Scen_revenue_L.mean() + A_L * CVaRL)
                    else:
                        UG = (1 - self.data.A_G) * Scen_revenue_G.mean() + A_G * CVaRG
                        UL = (1 - self.data.A_L) * Scen_revenue_L.mean() + A_L * CVaRL

                    utilities_G.append(UG)
                    utilities_L.append(UL)

                    # Calculate the Nash product (combined utility)
                    combined_utility_log = np.log(np.maximum(UG - self.data.Zeta_G, 1e-10)) + np.log(np.maximum(UL - self.data.Zeta_L, 1e-10))
                    combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
                    combined_utility_log = np.maximum(combined_utility_log, 0)  # Ensure non-negative values    
                    combined_utilities_log.append(combined_utility_log)

                # Plot with gradient color
                current_color = base_color.copy()
                current_color[3] = alpha_values[al_idx]  # Modify alpha for gradient effect
                ax.plot(strike_prices, combined_utilities_log, 
                    label=f'AG={a_G:.1f}, AL={a_L:.1f}',
                    color=current_color,
                    linewidth=2)  # Increased line width for better visibility

        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Nash Product (log)")
        ax.set_title("Nash Product (Log) for Different Risk Aversion Values")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)  # Reduced grid opacity for better contrast

        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()

    def scipy_optimization(self, plot=False, filename=None):
        """
        Perform Nash bargaining optimization using SciPy instead of Gurobi.
        This provides an alternative optimization approach that may be faster for some cases.
        
        Parameters
        ----------
        plot : bool, optional
            If True, creates a 2D contour plot of the Nash product, by default False
        filename : str, optional
            If provided, saves the plot to this filename, by default None
        
        Returns
        -------
        dict
            Dictionary containing optimization results including strike price, contract amount,
            utilities and Nash product value.
        """
        from utils import optimize_nash_product
        
        # Get the optimal values using SciPy optimization
        S_opt, M_opt, Nash_Product = optimize_nash_product(
            old_obj_func=self.old_obj_func,
            hours_in_year=self.data.hours_in_year,
            price_G = self.data.price_G,
            price_L = self.data.price_L,
            A_G=self.data.A_G,
            A_L=self.data.A_L,
            net_earnings_no_contract_G=self.data.net_earnings_no_contract_priceG_G,
            net_earnings_no_contract_L=self.data.net_earnings_no_contract_priceL_L,
            Zeta_G=self.data.Zeta_G,
            Zeta_L=self.data.Zeta_L,
            strikeprice_min=self.data.strikeprice_min,
            strikeprice_max=self.data.strikeprice_max,
            contract_amount_min=self.data.contract_amount_min,
            contract_amount_max=self.data.contract_amount_max,
            alpha=self.data.alpha,
            plot=plot,
            plot_dir=self.plots_dir if filename else None,
            filename=filename
        )
        
        # Store results in the same format as Gurobi optimization
        self.scipy_results.strike_price = S_opt
        self.scipy_results.contract_amount = M_opt # Convert to GWh/year
        self.scipy_results.strike_price_hour = S_opt / self.data.hours_in_year * 1000
        self.scipy_results.contract_amount_hour = M_opt /self.data.hours_in_year * 1000  # Convert to GWh/year
        self.scipy_results.objective_value = np.log(Nash_Product)  # Convert to log form to match Gurobi
        self.scipy_results.Nash_Product = Nash_Product
        
        #True Earnings with true probablity distribution
        # Calculate utilities with optimal values
        EuG = self.data.net_earnings_no_contract_priceG_G

        SMG = ((self.scipy_results.strike_price - self.data.price_G) * self.scipy_results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

        # Calculate CVaR for results G 
        self.scipy_results.CVaRG = calculate_cvar_left(EuG + SMG, self.data.alpha)

        # Calculate revenues for L
        EuL = self.data.net_earnings_no_contract_priceL_L
        SML = ((self.data.price_L - self.scipy_results.strike_price) * self.scipy_results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

        # Calculate CVaR for L
        self.scipy_results.CVaRL = calculate_cvar_left(EuL + SML, self.data.alpha)


        self.scipy_results.utility_G = (1-self.data.A_G) * (EuG + SMG).mean() + self.data.A_G * self.scipy_results.CVaRG
        self.scipy_results.utility_L = (1-self.data.A_L) * (EuL + SML).mean() + self.data.A_L * self.scipy_results.CVaRL
           

        self.scipy_results.Nash_Product = ((self.scipy_results.utility_G - self.data.Zeta_G) * (self.scipy_results.utility_L - self.data.Zeta_L))
        # Save accumulated revenues
        #re calcuate 
        self.scipy_results.accumulated_revenue_True_G = EuG + SMG.sum(axis=0)
        self.scipy_results.accumulated_revenue_True_L = EuL + SML.sum(axis=0)

        # Results with biased distribution 

        print("\n-------------------   RESULTS SCIPY  -------------------")
        print(f"Optimal Strike Price: {  self.scipy_results.strike_price}")
        print(f"Optimal Contract Amount: {  self.scipy_results.contract_amount}")
        print(f"Nash Product Value: {  self.scipy_results.objective_value}")
        print(f"Utility G: {  self.scipy_results.utility_G}")
        print(f"Utility L: {  self.scipy_results.utility_L}")
        print("SciPy optimization complete.")

        return {
            'strike_price': S_opt,
            'contract_amount': M_opt,
            'nash_product': Nash_Product,
            'utility_G':  self.scipy_results.utility_G,
            'utility_L':   self.scipy_results.utility_L 
        }

    def scipy_display_results(self):
      
        # Compare results
        print("\n-------------------  THREAT POINTS --------------------")
        print(f"Threat Point G: {self.data.Zeta_G}")
        print(f"Threat Point L: {self.data.Zeta_L}")

        print("\n-------------------   OPTIMIZATION COMPARISON  -------------------")
        print("Parameter       |    Gurobi    |    SciPy    |    Difference")
        print("-" * 60)
        print(f"Strike Price   | {self.results.strike_price_hour:2f} | {  self.scipy_results.strike_price_hour:10.4f} | {(self.results.strike_price -   self.scipy_results.strike_price):10.4f}")
        print(f"Contract Amt   | {self.results.contract_amount_hour:2f} | {  self.scipy_results.contract_amount_hour:10.10f} | {(self.results.contract_amount_hour -   self.scipy_results.contract_amount_hour):10.10f}")
        print(f"Nash Product   | {self.results.Nash_Product:2f} | {  self.scipy_results.Nash_Product:10.4f} | {(self.results.Nash_Product -   self.scipy_results.Nash_Product):10.4f}")
        print(f"Utility G     | {self.results.utility_G:2f} | {  self.scipy_results.utility_G:10.1f} | {(self.results.utility_G -   self.scipy_results.utility_G):10.4f}")
        print(f"Utility L     | {self.results.utility_L:2f} | {  self.scipy_results.utility_L:10.1f} | {(self.results.utility_L -   self.scipy_results.utility_L):10.4f}")
        print("")
    def plot_threat_points(self, A_G_values, A_L_values, filename=None):
        """
        Plot how threat points (Zeta) change with different risk aversion values.
        
        Parameters
        ----------
        A_G_values : array-like
            Array of GenCo risk aversion values to test
        A_L_values : array-like
            Array of LSE risk aversion values to test
        filename : str, optional
            If provided, saves the plot to this filename
        """
        # Create meshgrid for 3D surface plot

        A_values_old = np.round(np.linspace(0, 2, 15), 2)
        A_values_new = np.round(np.linspace(0, 1, 15), 2)
        zeta_G_old = np.zeros_like(A_values_old)
        zeta_L_old = np.zeros_like(A_values_new)

        zeta_G_new = np.zeros_like(A_values_old)
        zeta_L_new = np.zeros_like(A_values_new)

        # Calculate Zeta values for each combination of risk aversion parameters
        CVaR_no_contract_G = calculate_cvar_left(self.data.net_earnings_no_contract_true_G, self.data.alpha)
        CVaR_no_contract_L = calculate_cvar_left(self.data.net_earnings_no_contract_true_L, self.data.alpha)

        
        for i  in range(len(A_values_old)):

            zeta_G_old[i] = self.data.net_earnings_no_contract_true_G.mean() + A_values_old[i] * CVaR_no_contract_G
            zeta_G_new[i] = (1-A_values_new[i]) * self.data.net_earnings_no_contract_true_G.mean() +  A_values_new[i] * CVaR_no_contract_G

            zeta_L_old[i] = self.data.net_earnings_no_contract_true_L.mean() + A_values_old[i] * CVaR_no_contract_L
            zeta_L_new[i] = (1-A_values_new[i]) * self.data.net_earnings_no_contract_true_L.mean() +  A_values_new[i] * CVaR_no_contract_L

        # Create figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot G Revenue Histogram
        ax1 = axes[0]
        ax2 = axes[1]
        # First subplot for Zeta G

        ax1.plot(A_values_old, zeta_G_old,marker='o', color='orange')
        ax1.plot(A_values_new, zeta_G_new,marker='x', color='blue')
        ax1.set_xlabel('Risk Aversion G')
        ax1.set_ylabel('Threat Point G')
        ax1.legend(['Old Objective', 'New Objective'])
        ax1.set_title('Threat Point G vs Risk Aversion Parameters')

        # Second subplot for Zeta L

        ax2.plot(A_values_old, zeta_L_old, marker='o', color='orange')
        ax2.plot(A_values_new, zeta_L_new, marker='x', color='blue')
        ax2.set_xlabel('Risk Aversion L')
        ax2.set_ylabel('Threat point L')
        ax2.legend(['Old Objective', 'New Objective'])
        ax2.set_title('Threat Point L vs Risk Aversion Parameters')

        plt.tight_layout()

        if filename:
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Threat points plot saved to {filepath}")
            plt.close(fig)
        else:
            plt.show()
        print("test")