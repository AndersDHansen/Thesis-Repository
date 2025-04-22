"""
Contract negotiation implementation using Nash bargaining solution.
"""
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import GRB
from utils import Expando, build_dataframe, calculate_cvar, simulate_price_scenarios,analyze_price_distribution
from tqdm import tqdm
import scipy.stats as stats
from matplotlib.patches import Polygon # Import added at the top of the file
import seaborn as sns
import os

class ContractNegotiation:
    """
    Implements contract negotiation between generator and load using Nash bargaining.
    """
    def __init__(self, input_data, opf_results,Monte_Carlo=False):
        self.data = input_data
        self.opf_results = opf_results
        self.Monte_Carlo = Monte_Carlo
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
        # Build price dataframe
        self.data.price_df = build_dataframe(self.opf_results.price['N3'], 'price')
        self.data.price_true = self.data.price_df.values


        # Calculate biased prices
        self.data.price_G6 = (self.data.K_G6 * self.data.price_true.mean()) + self.data.price_df.values
        self.data.price_L2 = (self.data.K_L2 * self.data.price_true.mean()) + self.data.price_df.values

        # Generator production
        self.data.generator_production_df = build_dataframe(self.opf_results.generator_production['G6'], 'generator_production')
        self.data.generator_production = self.data.generator_production_df.values

        self.data.price_mean_true = self.data.price_true.mean() # Mean price at node N3
        self.data.price_std_true = self.data.price_true.std(ddof=1) # Std price at node N3

        # Make distributions for the price at node N3
        
        self.data.generator_production_df = build_dataframe(self.opf_results.generator_production['G6'],'generator_production') # Generator G6
        self.data.generator_production = self.data.generator_production_df.values # Generator G6 in numpy array for calculations

        self.data.price_aggregated = self.data.price_true.sum() # Aggregated price at node N3 (average over scenarios) 
        
        #Make this a call from utility function 
        self.data.net_earnings_no_contract_G6_df =build_dataframe(self.opf_results.generator_revenue['G6'],'generator_revenue')
        self.data.net_earnings_no_contract_G6 = self.data.net_earnings_no_contract_G6_df.sum() # Generator G6 earnings in numpy array for calculations ( Total Sum per Scenario)
        self.data.net_earnings_no_contract_L2_df = self.data.load_capacity['L2']*(self.data.retail_price-self.data.price_df)
        self.data.net_earnings_no_contract_L2 = self.data.net_earnings_no_contract_L2_df.sum() # Load L2 earnings in numpy array for calculations
        #CVaR calculation of No contract 
        self.data.CVaR_no_contract_G6 = calculate_cvar(self.data.net_earnings_no_contract_G6,self.data.alpha) # CVaR for G6

        self.data.CVaR_no_contract_L2 = calculate_cvar(self.data.net_earnings_no_contract_L2,self.data.alpha)

        # Threat Point G6
        self.data.Zeta_G6=(1-self.data.A_G6)*self.data.net_earnings_no_contract_G6.mean() + self.data.A_G6*self.data.CVaR_no_contract_G6

        #Threat Point for L2
        self.data.Zeta_L2=(1-self.data.A_L2)*self.data.net_earnings_no_contract_L2.mean() + self.data.A_L2*self.data.CVaR_no_contract_L2
        #CvaR prices 
        self.data.CVaR_price =calculate_cvar(self.data.price_true,self.data.alpha) # CVaR price at node N3

        # Distributions 
        # If assuming they have different observations of prices ( i.e., different K (bias) values)
        #pdf_True = stats.norm.pdf(self.data.price_G6, self.data.price_G6.mean(), self.data.price_G6.std()) # G6 observed distribution 
        #pdf_G6 = stats.norm.pdf(self.data.price_true, self.data.price_mean_true+(self.data.K_G6 * self.data.price_true.mean()), self.data.price_std_true) # L2 observed distribution 
        #pdf_L2 = stats.norm.pdf(self.data.price_true, self.data.price_mean_true+(self.data.K_L2 * self.data.price_true.mean()), self.data.price_std_true) # L2 observed distribution 
        
        num_scenarios = self.data.price_true.shape[1]  # Match your existing number of scenarios
        time_periods = self.data.price_true.shape[0]   # Match your existing time periods

        self.data.lambda_sum_true_per_scenario = self.data.price_true.sum(axis=0) # Sum across time periods for each scenario

        # Calculate E^P(λ∑) - Expected value of the sum over T (using TRUE distribution)
        self.data.expected_lambda_sum_true = self.data.lambda_sum_true_per_scenario.mean()

        # Calculate CVaR^P(λ∑) - CVaR of the sum over T (using TRUE distribution)
        # Assumes calculate_cvar returns the expected value of the variable *given* it's in the alpha-tail
        self.data.cvar_lambda_sum_true = calculate_cvar(self.data.lambda_sum_true_per_scenario, self.data.alpha)

        # Calculate CVaR^P(-λ∑) - CVaR of the negative sum over T (using TRUE distribution)
        # This corresponds to the risk of low LMPs
        self.data.cvar_neg_lambda_sum_true = calculate_cvar(-self.data.lambda_sum_true_per_scenario, self.data.alpha)

        # ... (calculations for no-contract earnings and threat points Zeta_G6, Zeta_L2) ...

        # <<< --- START OF SR* AND SU* IMPLEMENTATION --- >>>

        # --- Implementation of Equations (27) and (28) for SR* and SU* ---
        T = self.data.price_true.shape[0] # Number of time periods (contract length T)
        A_G = self.data.A_G6             # GenCo risk aversion factor
        A_L = self.data.A_L2             # LSE risk aversion factor

        # Determine the *absolute* bias constants K_G and K_L for the equations
        # Based on Theorem 1, K_G/K_L are absolute shifts.
        # We assume self.data.K_G6 and self.data.K_L2 store these *absolute* values.
        # If they stored relative factors (e.g., 0.01), you would calculate:
        # K_G_abs = self.data.K_G6 * self.data.expected_lambda_sum_true
        # K_L_abs = self.data.K_L2 * self.data.expected_lambda_sum_true
        # Using the current variables directly as the absolute K values:
        K_G_abs = self.data.K_G6
        K_L_abs = self.data.K_L2

        # Calculate the denominators T * (1 + A)
        denom_G = T * (1 + A_G)
        denom_L = T * (1 + A_L)

        # Avoid division by zero if A_G or A_L happens to be -1 (unlikely but safe)
        if abs(denom_G) < 1e-9: denom_G = 1e-9 # Use a small number instead of zero
        if abs(denom_L) < 1e-9: denom_L = 1e-9 # Use a small number instead of zero

        # Calculate Term 1 (GenCo perspective, risk of high LMPs)
        # E^P(λ∑) + A_G*CVaR^P(λ∑) + (1 + A_G)*K_G
        term1_G = (self.data.expected_lambda_sum_true + A_G * self.data.cvar_lambda_sum_true + (1 + A_G) * K_G_abs) / denom_G

        # Calculate Term 2 (GenCo perspective, risk of low LMPs -> using CVaR of negative sum)
        # E^P(λ∑) - A_G*CVaR^P(-λ∑) + (1 + A_G)*K_G
        term2_G = (self.data.expected_lambda_sum_true - A_G * self.data.cvar_neg_lambda_sum_true + (1 + A_G) * K_G_abs) / denom_G

        # Calculate Term 3 (LSE perspective, risk of low LMPs, for SR*)
        # E^P(λ∑) + (1 + A_L)*K_L - A_L*CVaR^P(-λ∑)
        term3_L_SR = (self.data.expected_lambda_sum_true + (1 + A_L) * K_L_abs - A_L * self.data.cvar_neg_lambda_sum_true) / denom_L

        # Calculate Term 4 (LSE perspective, risk of high LMPs, for SU*)
        # E^P(λ∑) + (1 + A_L)*K_L + A_L*CVaR^P(λ∑)
        term4_L_SU = (self.data.expected_lambda_sum_true + (1 + A_L) * K_L_abs + A_L * self.data.cvar_lambda_sum_true) / denom_L

        # Calculate SR* using Equation (27) - Minimum of the relevant terms
        self.data.SR_star = np.min([term1_G, term2_G, term3_L_SR])

        # Calculate SU* using Equation (28) - Maximum of the relevant terms
        self.data.SU_star = np.max([term1_G, term2_G, term4_L_SU])

        # <<< --- END OF SR* AND SU* IMPLEMENTATION --- >>>

        # ... (rest of the _build_statistics method, e.g., Monte Carlo simulations if enabled) ...

        # Print calculated critical bounds for verification during runtime
        print(f"Calculated SR* (Eq 27): {self.data.SR_star:.5f}")
        print(f"Calculated SU* (Eq 28): {self.data.SU_star:.5f}")


        # Generate scenarios using different distributions
        
        # Make distributions for the price at node N3
        self.data.pdf_true = stats.norm.pdf(self.data.price_true, self.data.price_mean_true, self.data.price_std_true) # True distribution 
        self.data.sim_prices_G6 = simulate_price_scenarios(self.data.price_true,self.data.K_G6, num_scenarios,time_periods, 'normal')
        self.data.sim_prices_L2 = simulate_price_scenarios(self.data.price_true,self.data.K_L2, num_scenarios,time_periods, 'normal')



        #lognormal_scenarios = simulate_price_scenarios(self.data.price_true, num_scenarios, 'lognormal')
        #empirical_scenarios = simulate_price_scenarios(self.data.price_true, num_scenarios, 'empirical')

        # Analyze distributions
        simulated_prices = {
            'normal G6': self.data.sim_prices_G6,
            'normal L2': self.data.sim_prices_L2,
            #'lognormal': lognormal_scenarios,
            #'empirical': empirical_scenarios
        }
        distribution_analysis = analyze_price_distribution(self.data.price_true, simulated_prices)


    def _build_variables(self):
        """Build optimization variables for contract negotiation."""
        # Auxiliary variables for logaritmic terms

        self.variables.arg_G6 = self.model.addVar(lb=0, name="UG6_minus_ZetaG6")
        self.variables.arg_L2 = self.model.addVar(lb=0, name="UL2_minus_ZetaL2")

        # Define logarithmic terms
        self.variables.log_arg_G6 = self.model.addVar(lb=0,name="log_arg_G6")
        self.variables.log_arg_L2 = self.model.addVar(lb=0,name="log_arg_L2")

           # build strike price variables
        self.variables.S = self.model.addVar(
            lb=self.data.strikeprice_min,
            ub=self.data.strikeprice_max,
            name='Strike_Price'
        )

    
        # build contract amount variables
        self.variables.M = self.model.addVar(
            lb=self.data.contract_amount_min,
            ub=self.data.contract_amount_max,
            name='Contract Amount'
        )
   
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
        """Build constraints for contract negotiation."""
         # Single Strike Price Constraint
        self.constraints.strike_price_constraint =  self.model.addLConstr(
        self.variables.S<=self.data.strikeprice_max,
        name='Strike_Price_Constraint_Max'
    )
        
         #Strike Price min  constraints
        self.constraints.strike_price_constraint_min = self.model.addLConstr(
            self.data.strikeprice_min <= self.variables.S,
            name='Strike_Price_Constraint_Min'
        )

        #Contract amounts 
        
        #Contract amount constraints
        self.constraints.contract_amount_constraint = self.model.addLConstr(
           self.variables.M <= self.data.contract_amount_max,
            name='Contract Amount Constraint Max'
        )
         #Contract amount constraints
        self.constraints.contract_amount_constraint_min = self.model.addLConstr(
            self.data.contract_amount_min <= self.variables.M,
            name='Contract Amount Constraint Min'
        )
        

        #Logarithmic constraints
        self.model.addGenConstrLog(self.variables.arg_G6, self.variables.log_arg_G6, "log_G6")
        self.model.addGenConstrLog(self.variables.arg_L2, self.variables.log_arg_L2, "log_L2")
        
        #Risk Constraints  
        if self.Monte_Carlo == False:     
            self.constraints.eta_G6_constraint = {
                s: self.model.addConstr(
                self.variables.eta_G6[s] >=self.variables.zeta_G6 - gp.quicksum((self.data.price_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s])
                                                                                +( self.variables.S- self.data.price_G6[t,s])*self.variables.M for t in self.data.TIME),
                name='Eta_Aversion_Constraint_G6_in_scenario_{0}'.format(s)
            )
            for s in self.data.SCENARIOS_L
            }
            self.constraints.eta_L2_constraint = {
                s: self.model.addConstr(
                self.variables.eta_L2[s] >=self.variables.zeta_L2 - gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price_L2[t,s])
                                                                                +(self.data.price_L2[t,s] - self.variables.S)*self.variables.M for t in self.data.TIME),
                name='Eta_Aversion_Constraint_L2_in_scenario_{0}'.format(s)
            )
            for s in self.data.SCENARIOS_L
            }
        else:
            self.constraints.eta_G6_constraint = {
                s: self.model.addConstr(
                self.variables.eta_G6[s] >=self.variables.zeta_G6 - gp.quicksum((self.data.sim_prices_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s])
                                                                                +( self.variables.S- self.data.sim_prices_G6[t,s])*self.variables.M for t in self.data.TIME),
                name='Eta_Aversion_Constraint_G6_in_scenario_{0}'.format(s)
            )
            for s in self.data.SCENARIOS_L
            }
            self.constraints.eta_L2_constraint = {
                s: self.model.addConstr(
                self.variables.eta_L2[s] >=self.variables.zeta_L2 - gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.sim_prices_L2[t,s])
                                                                                +(self.data.sim_prices_L2[t,s] - self.variables.S)*self.variables.M for t in self.data.TIME),
                name='Eta_Aversion_Constraint_L2_in_scenario_{0}'.format(s)
            )
            for s in self.data.SCENARIOS_L
            }

        self.model.update()

    def _build_objective(self):
        """Build the objective function for contract negotiation."""
        M = self.data.generator_capacity['G6'] # Baseload amount
        #Utility function for G6 
        EuG6 =  gp.quicksum((self.data.price_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG6 =  gp.quicksum(( self.variables.S- self.data.price_G6[t,s])*self.variables.M for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #CVaR for G6 

        CVaRG6 = self.variables.zeta_G6 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_G6[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UG6 = (1-self.data.A_G6) * self.data.PROB*(EuG6+SMG6) + self.data.A_G6*CVaRG6 # This bad boi is the issue


        #Utility function for L2
        EuL2 = gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.price_L2[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML2 =  gp.quicksum((self.data.price_L2[t,s] - self.variables.S)*self.variables.M for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #CvaR for L2
        CVaRL2 = self.variables.zeta_L2 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_L2[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UL2 = (1-self.data.A_L2)*self.data.PROB*(EuL2+ SML2) + self.data.A_L2*CVaRL2

         # Link auxiliary variables to expressions
        self.model.addConstr(self.variables.arg_G6 == UG6 - self.data.Zeta_G6, "arg_G6_constr")
        self.model.addConstr(self.variables.arg_L2 == UL2 - self.data.Zeta_L2, "arg_L2_constr")

        self.model.update()

        # Normal Objective function 
        #objective = (UG6 - self.data.Zeta_G6) * (UL2-self.data.Zeta_L2)
        #self.model.setObjective(objective,sense = gp.GRB.MAXIMIZE)
        
        # Set logarithmic objective
        self.model.setObjective(
            self.variables.log_arg_G6 + self.variables.log_arg_L2,
        GRB.MAXIMIZE) 
        self.model.update()

    def _build_objective_monte_carlo(self):

        M = self.data.generator_capacity['G6'] # Baseload amount
        #Utility function for G6 
        EuG6 =  gp.quicksum((self.data.sim_prices_G6[t,s]-self.data.generator_cost_a['G6'])*self.data.generator_production[t,s]-self.data.generator_cost_b['G6']*(self.data.generator_production[t,s]*self.data.generator_production[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SMG6 =  gp.quicksum(( self.variables.S- self.data.sim_prices_G6[t,s])*M for t in self.data.TIME for s in self.data.SCENARIOS_L)
        #CVaR for G6 

        CVaRG6 = self.variables.zeta_G6 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_G6[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UG6 = (1-self.data.A_G6) * self.data.PROB*(EuG6+SMG6) + self.data.A_G6*CVaRG6 # This bad boi is the issue


        #Utility function for L2
        EuL2 = gp.quicksum(self.data.load_capacity['L2'][t,s]*(self.data.retail_price-self.data.sim_prices_L2[t,s]) for t in self.data.TIME for s in self.data.SCENARIOS_L)
        SML2 =  gp.quicksum((self.data.sim_prices_L2[t,s] - self.variables.S)*M for t in self.data.TIME for s in self.data.SCENARIOS_L) # contract for capaicty - so am using generator capacity
        #CvaR for L2
        CVaRL2 = self.variables.zeta_L2 - (1/(1-self.data.alpha))*gp.quicksum((self.data.PROB*self.variables.eta_L2[s])  for s in self.data.SCENARIOS_L)
        #Expectation in utility function
        UL2 = (1-self.data.A_L2)*self.data.PROB*(EuL2+ SML2) + self.data.A_L2*CVaRL2

         # Link auxiliary variables to expressions
        self.model.addLConstr(self.variables.arg_G6 == UG6 - self.data.Zeta_G6, "arg_G6_constr")
        self.model.addLConstr(self.variables.arg_L2 == UL2 - self.data.Zeta_L2, "arg_L2_constr")

        self.model.update()

    def _build_model(self):
        """Initialize and build the complete optimization model."""
        self.model = gp.Model(name='Nash Bargaining Model')
        self.model.Params.NonConvex = 2
        #self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = 30
        
        self._build_variables()
        self._build_constraints()
        if self.Monte_Carlo == False:
            self._build_objective()
        else:
            self._build_objective_monte_carlo()
        self.model.update()

    def _save_results(self):
        """Save optimization results."""
        # Save objective value, strike price, and contract amount
        self.results.objective_value = self.model.ObjVal
        self.results.strike_price = self.variables.S.x
        self.results.contract_amount = self.variables.M.x

        # Calculate revenues with contract
        EuG6 = self.data.net_earnings_no_contract_G6
        if self.Monte_Carlo == False:
            SMG6 = {
                (t, s): (self.results.strike_price - self.data.price_G6[t, s]) * 
                self.results.contract_amount
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            }
        else:
            SMG6 = {
                (t, s): (self.results.strike_price - self.data.sim_prices_G6[t, s]) * 
                self.results.contract_amount
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            }
        SMG6_df = build_dataframe(SMG6, 'G6_revenue')

        # Calculate CVaR for G6
        self.results.CVaRG6 = calculate_cvar(EuG6 + SMG6_df.values.sum(), self.data.alpha)
        self.results.utility_G6 = (1-self.data.A_G6) * (EuG6 + SMG6_df.values.sum()).mean() + self.data.A_G6 * self.results.CVaRG6

        # Calculate revenues for L2
        EuL2 = self.data.net_earnings_no_contract_L2
        if self.Monte_Carlo == False:
            SML2 = {
                (t, s): (self.data.price_L2[t, s] - self.results.strike_price) * self.results.contract_amount
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L}
        else:
            SML2 = {
                (t, s): (self.data.sim_prices_L2[t, s] - self.results.strike_price) * self.results.contract_amount
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L}

        SML2_df = build_dataframe(SML2, 'L2_revenue')

        # Calculate CVaR for L2
        self.results.CVaRL2 = calculate_cvar(EuL2 + SML2_df.values.sum(), self.data.alpha)
        self.results.utility_L2 = (1-self.data.A_L2) * (EuL2 + SML2_df.values.sum()).mean() + self.data.A_L2 * self.results.CVaRL2

        # Save accumulated revenues
        self.results.accumulated_revenue_G6 = EuG6 + SMG6_df.sum(axis=0)
        self.results.accumulated_revenue_L2 = EuL2 + SML2_df.sum(axis=0)

    def run(self):
        """Run the optimization model."""
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            #self.model.write("model.lp")
            #self.model.write("model.rlp")
            #self.model.write("model.mps")
            self._save_results()
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        """Display optimization results."""
        print("\n-------------------   RESULTS GUROBI  -------------------")
        print(f"Optimal Objective Value (Log): {self.results.objective_value}")
        print(f"Optimal Objective Value: {np.exp(self.results.objective_value)}")
        print(f"Optimal Strike Price: {self.results.strike_price}")
        print(f"Optimal Contract Amount: {self.results.contract_amount}")
        print(f"Optimal Utility for G6: {self.results.utility_G6}")
        print(f"Optimal Utility for L2: {self.results.utility_L2}")
        print(f"Threat Point G6: {self.data.Zeta_G6}")
        print(f"Threat Point L2: {self.data.Zeta_L2}")

    def manual_optimization(self, plot=False, filename=None):
        """
        Perform manual optimization by grid search over strike prices.
        """
        strike_prices = np.linspace(self.data.strikeprice_min, self.data.strikeprice_max, 1000)
        M = 300  # Fixed contract amount

        utilities_G6 = []
        utilities_L2 = []
        combined_utilities = []
        combined_utilities_log = []

        for strike in tqdm(strike_prices, desc='loading...'):
            # Calculate revenues for G6
            SMG6 = {
                (t, s): (strike - self.data.price_G6[t, s]) * M 
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            }
            SMG6_df = build_dataframe(SMG6, 'G6_revenue')

            # Calculate revenues for L2
            SML2 = {
                (t, s): (self.data.price_L2[t, s] - strike) * M 
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            }
            SML2_df = build_dataframe(SML2, 'L2_revenue')

            # Calculate total revenues
            Scen_revenue_G6 = SMG6_df.sum() + self.data.net_earnings_no_contract_G6
            Scen_revenue_L2 = SML2_df.sum() + self.data.net_earnings_no_contract_L2

            # Calculate CVaR
            CVaRG6 = calculate_cvar(Scen_revenue_G6.values, self.data.alpha)
            CVaRL2 = calculate_cvar(Scen_revenue_L2.values, self.data.alpha)

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
            # Create Plots directory if it doesn't exist
           

            # Plot the utilities and Nash product for different strike prices
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))

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

            """
            # Second subplot: Nash Product and Log Nash Product
            axs[2].plot(utilities_G6, utilities_L2,  color="orange", linestyle="-")
            axs[2].set_xlabel("Utility G6")
            axs[2].set_ylabel("Utility L2")
            axs[2].set_title("Utility G6 vs Utility L2")
            axs[2].grid()
            fig.suptitle(f"Utilities and Nash Product for Different Strike Prices with with Risk AG{self.data.A_G6} and Risk AL{self.data.A_L2}")
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

    def batch_manual_optimization(self, A_G6_values, A_L2_values, strike_min=None, strike_max=None, filename=None):
        """
        Perform manual optimization for multiple risk aversion values and plot combined results.
        
        Args:
            A_G6_values: Array of GenCo risk aversion values
            A_L2_values: Array of LSE risk aversion values
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
        M = 300  # Fixed contract amount

        # Create a color map for AG values - using tab20 for more distinct colors
        base_colors = plt.cm.tab20(np.linspace(0, 1, len(A_G6_values)))
        # Number of gradient steps for each AL value
        gradient_steps = len(A_L2_values)

        for ag_idx, a_g6 in enumerate(tqdm(A_G6_values, desc='loading batch...')):
            base_color = base_colors[ag_idx]
            # Create gradient colors for this AG value - using a wider range of alpha values
            alpha_values = np.linspace(0.5, 1.0, gradient_steps)  # Increased minimum alpha for better visibility
            
            for al_idx, a_l2 in enumerate(A_L2_values):
                utilities_G6 = []
                utilities_L2 = []
                combined_utilities_log = []
                
                # Update risk aversion values
                A_G6 = a_g6
                A_L2 = a_l2
                
                for strike in tqdm(strike_prices, desc='loading...'):
                    # Calculate revenues for G6
                    SMG6 = {
                        (t, s): (strike - self.data.price_G6[t, s]) * M 
                        for t in self.data.TIME 
                        for s in self.data.SCENARIOS_L
                    }
                    SMG6_df = build_dataframe(SMG6, 'G6_revenue')

                    # Calculate revenues for L2
                    SML2 = {
                        (t, s): (self.data.price_L2[t, s] - strike) * M 
                        for t in self.data.TIME 
                        for s in self.data.SCENARIOS_L
                    }
                    SML2_df = build_dataframe(SML2, 'L2_revenue')

                    # Calculate total revenues
                    Scen_revenue_G6 = SMG6_df.sum() + self.data.net_earnings_no_contract_G6
                    Scen_revenue_L2 = SML2_df.sum() + self.data.net_earnings_no_contract_L2

                    # Calculate CVaR
                    CVaRG6 = calculate_cvar(Scen_revenue_G6.values, self.data.alpha)
                    CVaRL2 = calculate_cvar(Scen_revenue_L2.values, self.data.alpha)

                    # Calculate utility for G6
                    UG6 = (1 - self.data.A_G6) * Scen_revenue_G6.mean() + A_G6 * CVaRG6
                    utilities_G6.append(UG6)

                    # Calculate utility for L2
                    UL2 = (1 - self.data.A_L2) * Scen_revenue_L2.mean() + A_L2 * CVaRL2
                    utilities_L2.append(UL2)

                    # Calculate the Nash product (combined utility)
                    combined_utility_log = np.log(np.maximum(UG6 - self.data.Zeta_G6, 1e-10)) + np.log(np.maximum(UL2 - self.data.Zeta_L2, 1e-10))
                    combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
                    combined_utility_log = np.maximum(combined_utility_log, 0)  # Ensure non-negative values    
                    combined_utilities_log.append(combined_utility_log)

                # Plot with gradient color
                current_color = base_color.copy()
                current_color[3] = alpha_values[al_idx]  # Modify alpha for gradient effect
                ax.plot(strike_prices, combined_utilities_log, 
                    label=f'AG={a_g6:.1f}, AL={a_l2:.1f}',
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
        S_opt, M_opt, nash_value = optimize_nash_product(
            price_data=self.data.price_true,
            A_G6=self.data.A_G6,
            A_L2=self.data.A_L2,
            net_earnings_no_contract_G6=self.data.net_earnings_no_contract_G6,
            net_earnings_no_contract_L2=self.data.net_earnings_no_contract_L2,
            Zeta_G6=self.data.Zeta_G6,
            Zeta_L2=self.data.Zeta_L2,
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
        self.results.strike_price = S_opt
        self.results.contract_amount = M_opt
        self.results.objective_value = np.log(nash_value)  # Convert to log form to match Gurobi
        
        # Calculate utilities with optimal values
        EuG6 = self.data.net_earnings_no_contract_G6
        SMG6 = {(t, s): (S_opt - self.data.price_G6[t, s]) * M_opt
                for t in self.data.TIME for s in self.data.SCENARIOS_L}
        SMG6_df = build_dataframe(SMG6, 'G6_revenue')
        
        EuL2 = self.data.net_earnings_no_contract_L2
        SML2 = {(t, s): (self.data.price_L2[t, s] - S_opt) * M_opt
                for t in self.data.TIME for s in self.data.SCENARIOS_L}
        SML2_df = build_dataframe(SML2, 'L2_revenue')
        
        # Calculate CVaR and utilities
        self.scipy_results= calculate_cvar(EuG6 + SMG6_df.values.sum(), self.data.alpha)
        self.scipy_result.sutility_G6 = (1-self.data.A_G6) * (EuG6 + SMG6_df.values.sum()).mean() + self.data.A_G6 * self.results.CVaRG6
        
        self.scipy_results.CVaRL2 = calculate_cvar(EuL2 + SML2_df.values.sum(), self.data.alpha)
        self.scipy_results.utility_L2 = (1-self.data.A_L2) * (EuL2 + SML2_df.values.sum()).mean() + self.data.A_L2 * self.results.CVaRL2
        
        return {
            'strike_price': S_opt,
            'contract_amount': M_opt,
            'nash_product': nash_value,
            'utility_G6':  self.scipy_results.utility_G6,
            'utility_L2':   self.scipy_resultsutility_L2
        }



    def scipy_display_results(self):
        print("\n-------------------   RESULTS SCIPY  -------------------")
        print(f"Optimal Strike Price: {  self.scipy_results['strike_price']}")
        print(f"Optimal Contract Amount: {  self.scipy_results['contract_amount']}")
        print(f"Nash Product Value: {  self.scipy_results['nash_product']}")
        print(f"Utility G6: {  self.scipy_results['utility_G6']}")
        print(f"Utility L2: {  self.scipy_results['utility_L2']}")
        print("SciPy optimization complete.")

        # Compare results
        print("\n-------------------   OPTIMIZATION COMPARISON  -------------------")
        print("Parameter       |    Gurobi    |    SciPy    |    Difference")
        print("-" * 60)
        print(f"Strike Price   | {self.results.strike_price:11.4f} | {  self.scipy_results['strike_price']:10.4f} | {abs(self.results.strike_price -   self.scipy_results['strike_price']):10.4f}")
        print(f"Contract Amt   | {self.results.contract_amount:11.4f} | {  self.scipy_results['contract_amount']:10.4f} | {abs(self.results.contract_amount -   self.scipy_results['contract_amount']):10.4f}")
        print(f"Utility G6     | {self.results.utility_G6:11.4f} | {  self.scipy_results['utility_G6']:10.4f} | {abs(self.results.utility_G6 -   self.scipy_results['utility_G6']):10.4f}")
        print(f"Utility L2     | {self.results.utility_L2:11.4f} | {  self.scipy_results['utility_L2']:10.4f} | {abs(self.results.utility_L2 -   self.scipy_results['utility_L2']):10.4f}")
        