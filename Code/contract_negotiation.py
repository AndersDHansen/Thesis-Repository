"""
Contract negotiation implementation using Nash bargaining solution.
"""
import numpy as np
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import GRB
from utils import Expando, _calculate_S_star_BL_G,_calculate_S_star_BL_L, _calculate_S_star_PAP_L, build_dataframe, calculate_cvar_left, _right_tail_mask, _left_tail_mask, _calculate_S_star_PAP_G
from Barter_Set import Barter_Set
from tqdm import tqdm
import scipy.stats as stats
from matplotlib.patches import Polygon # Import added at the top of the file
import seaborn as sns
import os
from numpy.random import laplace
from scipy.optimize import minimize, NonlinearConstraint




class ContractNegotiation:
    def __init__(self, input_data):
        """Initialize contract negotiation model with loaded scenarios.
        
        Args:
            input_data: Input data object containing loaded scenarios and parameters
            provider: Provider object that supplies price, production and capture rate data
            
        Raises:
            ValueError: If required data is missing
        """        
        self.data = input_data
        self.relax = getattr(self.data, 'relax', False)
        self.contract_type = getattr(self.data, 'contract_type', 'Baseload')
        self.results = Expando()
        self.scipy_results = Expando()
        self.variables = Expando()
        self.constraints = Expando()
        
        self.plots_dir = os.path.join(os.path.dirname(__file__), 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
   
            
        # Build statistical measures and optimization model
        self._build_statistics()
        self._build_model()
            
    
    def _build_statistics(self):
        """Calculate statistical measures needed for negotiation using loaded scenarios."""
        # Data is already loaded in __init__

        ################### Calculate basic statistics from loaded scenarios #######
    
        ####### Prices ########
        mean_price_s = self.data.price_true.mean() # Per scenario across T
        mean_price = mean_price_s.mean()
        # Calculate scenario sums and expectations
        self.data.lambda_sum_true_per_scenario = self.data.price_true.sum(axis=0)
        self.data.expected_lambda_sum_true = self.data.lambda_sum_true_per_scenario.mean()

        # Calculate CVaR^P(λ∑) - CVaR of the sum over T (using TRUE distribution)
        # Assumes calculate_cvar returns the expected value of the variable *given* it's in the alpha-tail
        self.data.left_cvar_lambda_sum_true = calculate_cvar_left(self.data.lambda_sum_true_per_scenario, self.data.alpha)
        # Calculate CVaR^P(-λ∑) - CVaR of the negative sum over T (using TRUE distribution)
        # This corresponds to the risk of high LMPs
        self.data.left_Cvar_neg_lambda_sum_true = calculate_cvar_left(-self.data.lambda_sum_true_per_scenario, self.data.alpha) 
        self.data.left_Cvar_neg_lambdas = calculate_cvar_left(-self.data.price_true,self.data.alpha)
        # Production 
        mean_production = self.data.production.mean().mean()     

               
        ########### Capture Prices ##########
        # Calculate capture prices using loaded capture rates
        self.data.Capture_price_G = self.data.price_true * self.data.capture_rate
        self.data.Capture_price_h_G = self.data.Capture_price_G 
        self.data.Capture_price_G_avg = self.data.Capture_price_h_G.mean().mean()

        self.data.Capture_price_L = self.data.price_true * self.data.load_CR
        self.data.Capture_price_h_L = self.data.Capture_price_L 
        self.data.Capture_price_L_avg = self.data.Capture_price_h_L.mean().mean()

        # Per Scenario 
        self.data.production_per_scenario = self.data.production.sum(axis=0)
        self.data.expected_production = self.data.production_per_scenario.mean()


  

      

        
        
        # Calculate biased prices using mean price
        self.data.price_G = (self.data.K_G_price * mean_price) + self.data.price_true
        self.data.price_L = (self.data.K_L_price * mean_price) + self.data.price_true
        self.data.lambda_sum_G_per_scenario = self.data.price_G.sum(axis=0)
        self.data.lambda_sum_L_per_scenario = self.data.price_L.sum(axis=0)

        #Calculate biased production using mean production
        self.data.production_G = (self.data.K_G_prod * mean_production) + self.data.production
        self.data.production_L = (self.data.K_L_prod * mean_production) + self.data.production


        # Using loaded capture rate and production scenarios
        self.data.net_earnings_no_contract_G_df = pd.DataFrame(
            self.data.capture_rate * self.data.production * self.data.price_true
        )
        # Calculate true net earnings for generator with correct price distribution
        self.data.net_earnings_no_contract_true_G = self.data.net_earnings_no_contract_G_df.sum(axis=0)
        
        # Calculate biased net earnings for generator with price bias K_G
        self.data.net_earnings_no_contract_priceG_G_df = pd.DataFrame(
            self.data.capture_rate * self.data.production_G * self.data.price_G
        )
        self.data.net_earnings_no_contract_priceG_G = self.data.net_earnings_no_contract_priceG_G_df.sum(axis=0)
        
        # Calculate load earnings with true and biased prices
        self.data.net_earnings_no_contract_true_L =   (self.data.load_scenarios * (  self.data.retail_price-  self.data.load_CR * self.data.price_true)).sum(axis=0)
        self.data.net_earnings_no_contract_priceL_L = (self.data.load_scenarios * (  self.data.retail_price-  self.data.load_CR *self.data.price_L)).sum(axis=0)
          # Calculate CVaR for no-contract scenarios
        self.data.CVaR_no_contract_priceG_G = calculate_cvar_left(self.data.net_earnings_no_contract_priceG_G,self.data.alpha)
        self.data.CVaR_no_contract_priceL_L = calculate_cvar_left(self.data.net_earnings_no_contract_priceL_L,self.data.alpha)

        # Calculate utility functions using weighted average of mean and CVaR
        self.data.Zeta_G = (
            (1 - self.data.A_G) * self.data.net_earnings_no_contract_priceG_G.mean() + self.data.A_G * self.data.CVaR_no_contract_priceG_G)
        self.data.Zeta_L = (
            (1 - self.data.A_L) * self.data.net_earnings_no_contract_priceL_L.mean() + self.data.A_L * self.data.CVaR_no_contract_priceL_L)
        
        #Plotting Distributions
        #self.plot_distribution_with_p50()

        # Prepare for threat point calculations
        # Calculate dimensions from loaded scenarios
        num_scenarios = self.data.price_true.shape[1]
        time_periods = self.data.price_true.shape[0]
        
        time_periods_hours = time_periods * self.data.hours_in_year

        self.data.K_G_lambda_Sigma = self.data.K_G_price * self.data.expected_lambda_sum_true
        self.data.K_L_lambda_Sigma = self.data.K_L_price * self.data.expected_lambda_sum_true

        """ 
        if self.data.contract_type == 'Baseload': 
            # Determine the *absolute* bias constants K_G and K_L for the equations
            # Based on Theorem 1, K_G/K_L are absolute shifts.:
         
        
            # Using the current variables directly as the absolute K values:
    
            # Method 1 - Deterministic?
            self.data.term1_G =(( (1-self.data.A_G) * self.data.expected_lambda_sum_true +self.data.K_G_lambda_Sigma ) - self.data.A_G * self.data.left_Cvar_neg_lambda_sum_true)/ time_periods    # SR* numerator for Gen

            # high-price risk (feeds S_U*)
            self.data.term2_G = (((1-self.data.A_G) * self.data.expected_lambda_sum_true + self.data.K_G_lambda_Sigma ) + self.data.A_G * self.data.left_cvar_lambda_sum_true) / time_periods     # SU* numerator for Gen
    
            # Load-serving entity ––––––––––––––––––––––––––––––––––––––––––––––
            # low-price risk  (feeds S_R*)
            self.data.term3_L_SR = (self.data.expected_lambda_sum_true
                            + self.data.A_L * self.data.left_cvar_lambda_sum_true
                            + self.data.K_L_lambda_Sigma - self.data.A_L * self.data.expected_lambda_sum_true  ) / time_periods   # SU* numerator for LSE
            # high-price risk (feeds S_U*)
            self.data.term4_L_SU = ( self.data.expected_lambda_sum_true
                            - self.data.A_L * self.data.left_Cvar_neg_lambda_sum_true
                            + self.data.K_L_lambda_Sigma  - self.data.A_L * self.data.expected_lambda_sum_true )  / time_periods   # SR* numerator for LSE

            # Method 2 - Stochastic 
            mask_G = _left_tail_mask(self.data.net_earnings_no_contract_priceG_G, self.data.alpha)
            mask_L = _left_tail_mask(self.data.net_earnings_no_contract_priceL_L, self.data.alpha)
            neg_mask_G = _left_tail_mask(-self.data.net_earnings_no_contract_priceG_G, self.data.alpha)
            neg_mask_L = _left_tail_mask(-self.data.net_earnings_no_contract_priceL_L, self.data.alpha)

            tail_G = self.data.lambda_sum_G_per_scenario[mask_G].mean() if mask_G.any() else 0.0
            tail_L = self.data.lambda_sum_L_per_scenario[mask_L].mean() if mask_L.any() else 0.0
            neg_tail_G = self.data.lambda_sum_G_per_scenario[neg_mask_G].mean() if mask_G.any() else 0.0
            neg_tail_L = self.data.lambda_sum_L_per_scenario[neg_mask_L].mean() if mask_L.any() else 0.0

            self.data.term1_G_new =(( (1-self.data.A_G) * self.data.expected_lambda_sum_true +self.data.K_G_lambda_Sigma ) + self.data.A_G * neg_tail_G)/ time_periods    # SR* numerator for Gen

            # high-price risk (feeds S_U*)
            self.data.term2_G_new = (((1-self.data.A_G) * self.data.expected_lambda_sum_true + self.data.K_G_lambda_Sigma ) + self.data.A_G * tail_G) / time_periods     # SU* numerator for Gen

            # Load-serving entity ––––––––––––––––––––––––––––––––––––––––––––––
             # low-price risk (feeds S_R*)
            self.data.term3_L_SR_new = ( self.data.expected_lambda_sum_true
                                + self.data.A_L * neg_tail_L
                            + self.data.K_L_lambda_Sigma  - self.data.A_L * self.data.expected_lambda_sum_true )  / time_periods   # SR* numerator for LSE
            
            # high-price risk  (feeds S_U*)
            self.data.term4_L_SU_new = (self.data.expected_lambda_sum_true
                            + self.data.A_L * tail_L
                            + self.data.K_L_lambda_Sigma - self.data.A_L * self.data.expected_lambda_sum_true  ) / time_periods   # SU* numerator for LSE
           
          


            # Calculate SR* using Equation (27) - Minimum of the relevant terms
            self.data.SR_star = np.min([self.data.term1_G, self.data.term2_G, self.data.term3_L_SR])   # Convert from $/GWh to $/MWh
            self.data.SR_star_new = np.min([self.data.term1_G_new, self.data.term2_G_new, self.data.term4_L_SU_new])  # Convert from $/GWh to $/MWh

            # Calculate SU* using Equation (28) - Maximum of the relevant terms
            self.data.SU_star = np.max([self.data.term1_G, self.data.term2_G, self.data.term4_L_SU])  
            self.data.SU_star_new = np.max([self.data.term1_G_new, self.data.term3_L_SR_new, self.data.term4_L_SU_new])  # Convert from $/GWh to $/MWh

            print(f"Calculated SR* using New (Eq 27) (Hourly Price [EUR/MWh]): {self.data.SR_star*1e3:.4f}")
            print(f"Calculated SU* using new (Eq 28) (Hourly Price [EUR/MWh]: {self.data.SU_star*1e3:.4f}")

            
            print(f"Calculated SR* using New (Eq 27) (Hourly Price [EUR/MWh]): {self.data.SR_star_new*1e3:.4f}")
            print(f"Calculated SU* using new (Eq 28) (Hourly Price [EUR/MWh]: {self.data.SU_star_new*1e3:.4f}")
            #self.data.strikeprice_min = self.data.SR_star_new + 1e-3  # Add a small epsilon to avoid numerical issues
            #self.data.strikeprice_max = self.data.SR_star_new + 1e-3


            def constraint_S_star_G(x):
                S_star = _calculate_S_star_BL_G(x  , M,self.data.A_G, self.data.alpha, self.data.production_G, self.data.price_G,self.data.capture_rate)
                #print(f"Calculated S_star_G: {S_star}")
                return S_star

            def constraint_S_star_L(x):
                S_star = _calculate_S_star_BL_L(x, M,self.data.A_L, self.data.alpha, self.data.production_L, self.data.price_L,self.data.capture_rate,self.data.load_CR,self.data.load_scenarios)
               #print(f"Calculated S_star_L: {S_star}")
                return S_star

            nonlinear_constraint_S_star_G = NonlinearConstraint(constraint_S_star_G, 0, np.inf)
            nonlinear_constraint_S_star_L = NonlinearConstraint(constraint_S_star_L, 0, np.inf)

            # Define bounds for S (e.g., strike price range)
            bounds = [(self.data.strikeprice_min, self.data.strikeprice_max)]

            # Initial guess for S
            initial_guess = (self.data.strikeprice_max/2 ) 
            initial_guess_L = (self.data.strikeprice_max) 

            M =119.88
            result_G = minimize(
                _calculate_S_star_BL_G,
                x0=initial_guess,
                args=(M,self.data.A_G, self.data.alpha, self.data.production_G, self.data.price_G,self.data.capture_rate),
                bounds=bounds,
                constraints=[nonlinear_constraint_S_star_G],
                method='SLSQP',
                options={'disp': False, 'maxiter': 1000,'gtol': 1e-6,}
            )
            result_L = minimize(
                _calculate_S_star_BL_L,
                x0=initial_guess_L,
                args=(M,self.data.A_L, self.data.alpha, self.data.production_L, self.data.price_L,self.data.capture_rate,self.data.load_CR,self.data.load_scenarios),
                bounds=bounds,
                constraints=[nonlinear_constraint_S_star_L],
                method='SLSQP',
                options={'disp': False, 'maxiter': 1000,'gtol': 1e-6,}
            )

            # Optimal strike price
            #self.data.SR_star_new = result_G.x[0]
            #self.data.SU_star_new = result_L.x[0]

            print(f"Optimal Strike Price (Generator-side): {self.data.SR_star_new*1e3:.4f} EUR/MWh")
            print(f"Optimal Strike Price (Load-side): {self.data.SU_star_new*1e3:.4f} EUR/MWh")
            print()
        else:


            # Define bounds for S (e.g., strike price range)
            bounds = [(self.data.strikeprice_min, self.data.strikeprice_max)]

            # Initial guess for S
            initial_guess = (self.data.strikeprice_min ) 
            initial_guess_L = (self.data.strikeprice_max) 

            # Perform optimization
            #gamma = 1 since that is the most likely resulting contract amount for PAP 
            gamma = 0 

            def constraint_S_star_G(x):
                S_star = _calculate_S_star_PAP_G(x  , gamma,self.data.A_G, self.data.alpha, self.data.production_G, self.data.price_G,self.data.capture_rate)
                print(f"Calculated S_star_G: {S_star}")
                return S_star

            def constraint_S_star_L(x):
                S_star = _calculate_S_star_PAP_L(x, gamma,self.data.A_L, self.data.alpha, self.data.production_L, self.data.price_L,self.data.capture_rate,self.data.load_CR,self.data.load_scenarios)
                print(f"Calculated S_star_L: {S_star}")
                return S_star

            nonlinear_constraint_S_star_G = NonlinearConstraint(constraint_S_star_G, 0, np.inf)
            nonlinear_constraint_S_star_L = NonlinearConstraint(constraint_S_star_L, 0, np.inf)


            result_G = minimize(
                _calculate_S_star_PAP_G,
                x0=initial_guess,
                args=(gamma,self.data.A_G, self.data.alpha, self.data.production_G, self.data.price_G,self.data.capture_rate),
                bounds=bounds,
                constraints=[nonlinear_constraint_S_star_G],
                method='SLSQP',
                options={'disp': True, 'maxiter': 1000,'gtol': 1e-10,}
            )
            result_L = minimize(
                _calculate_S_star_PAP_L,
                x0=initial_guess,
                args=(gamma,self.data.A_L, self.data.alpha, self.data.production_L, self.data.price_L,self.data.capture_rate,self.data.load_CR,self.data.load_scenarios),
                bounds=bounds,
                constraints=[nonlinear_constraint_S_star_L],
                method='trust-constr',
                options={'disp': True, 'maxiter': 1000,'gtol': 1e-10,}
            )

            # Optimal strike price
            self.data.SR_star_new = result_G.x[0]
            self.data.SU_star_new = result_L.x[0]

            print(f"Optimal Strike Price (Generator-side): {self.data.SR_star_new*1e3:.4f} EUR/MWh")
            print(f"Optimal Strike Price (Load-side): {self.data.SU_star_new*1e3:.4f} EUR/MWh")
            print()
            """
    def _build_variables_PAP(self):

         # Auxiliary variables for logaritmic terms
        EPS = 1e-6            # pick something natural for your scale
        self.variables.arg_G = self.model.addVar(lb=EPS, name="UG_minus_ZetaG")
        self.variables.arg_L = self.model.addVar(lb=EPS, name="UL_minus_ZetaL")

        # Define logarithmic terms
        self.variables.log_arg_G = self.model.addVar(lb=EPS,name="log_arg_G")
        self.variables.log_arg_L = self.model.addVar(lb=EPS,name="log_arg_L")

           # build strike price variables
        self.variables.S = self.model.addVar(
            lb=self.data.strikeprice_min,  # Convert to $/MWh
            ub=self.data.strikeprice_max,
            name='Strike_Price'
        )

        # build contract amount variables
        self.variables.gamma = self.model.addVar(
            lb=0,
            ub=self.data.gamma_max,
            name='Proportion of production to go to PAP contract '
        )
   
        self.variables.zeta_G = self.model.addVar(
            name='Zeta_Auxillary_G',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
        self.variables.zeta_L = self.model.addVar(
            name='Zeta_Auxillary_L',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
      
        self.variables.eta_G = self.model.addVars(
            self.data.SCENARIOS_L,
            name='Auxillary_Variable_G',
            lb=0,
            ub=gp.GRB.INFINITY
)

        self.variables.eta_L = self.model.addVars(
            self.data.SCENARIOS_L,
            name='Auxillary_Variable_L',
            lb=0,
            ub=gp.GRB.INFINITY
        )
       
        self.model.update() 

    def _build_variables(self):
        """Build optimization variables for contract negotiation."""
        # Auxiliary variables for logaritmic terms
        EPS = 1e-6            # pick something natural for your scale
        self.variables.arg_G = self.model.addVar(lb=EPS, name="UG_minus_ZetaG")
        self.variables.arg_L = self.model.addVar(lb=EPS, name="UL_minus_ZetaL")

        # Define logarithmic terms
        self.variables.log_arg_G = self.model.addVar(lb=EPS,name="log_arg_G")
        self.variables.log_arg_L = self.model.addVar(lb=EPS,name="log_arg_L")

           # build strike price variables
        self.variables.S = self.model.addVar(
            lb=self.data.strikeprice_min,  # Convert to $/MWh
            ub=self.data.strikeprice_max,
            name='Strike_Price'
        )

        # build contract amount variables
        self.variables.M = self.model.addVar(
            lb=self.data.contract_amount_min,
            ub=self.data.contract_amount_max,
            name='Contract Amount'
        )
   
        self.variables.zeta_G = self.model.addVar(
            name='Zeta_Auxillary_G',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
        self.variables.zeta_L = self.model.addVar(
            name='Zeta_Auxillary_L',
            lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
      
        self.variables.eta_G = self.model.addVars(
            self.data.SCENARIOS_L,
            name='Auxillary_Variable_G',
            lb=0,
            ub=gp.GRB.INFINITY
        )

        self.variables.eta_L = self.model.addVars(
            self.data.SCENARIOS_L,
            name='Auxillary_Variable_L',
            lb=0,
            ub=gp.GRB.INFINITY
        )
  
        self.model.update() 

    def _build_constraints(self):
        """Build constraints for contract negotiation."""
         # Single Strike Price Constraint
        self.constraints.strike_price_constraint_max =  self.model.addLConstr(
        self.variables.S<=self.data.strikeprice_max ,
        name='Strike_Price_Constraint_Max'
        )
        
        
        self.constraints.strike_price_constraint_min =  self.model.addLConstr(
        self.variables.S>=self.data.strikeprice_min  ,
        name='Strike_Price_Constraint_Max'
        )

        #Contract amounts 
        
        #Contract amount constraints
        self.constraints.contract_amount_constraint_max = self.model.addLConstr(
           self.variables.M <= self.data.contract_amount_max ,  # Convert to GWh/year
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

        capture_rate_array = self.data.capture_rate.values
        price_G_array = self.data.price_G.values
        production_G_array = self.data.production_G.values
        load_scenarios_array = self.data.load_scenarios.values
        load_CR_array = self.data.load_CR.values
        price_L_array = self.data.price_L.values

        # Pre-compute constant terms for each scenario
        generator_const_per_scenario = (
            capture_rate_array * price_G_array * production_G_array
        ).sum(axis=0)  # Sum over time for each scenario

        load_const_per_scenario = (
            -load_scenarios_array *  load_CR_array * price_L_array
        ).sum(axis=0)  # Sum over time for each scenario

        # Batch create eta_G constraints
        self.constraints.eta_G_constraint = self.model.addConstrs(
        (self.variables.eta_G[s] >= 
         self.variables.zeta_G - (generator_const_per_scenario[s] + 
         gp.quicksum((self.variables.S - price_G_array[t,s]) * self.variables.M 
                     for t in self.data.TIME))
         for s in self.data.SCENARIOS_L),
        name='Eta_Aversion_Constraint_G'
    )
    
    # Batch create eta_L constraints  
        self.constraints.eta_L_constraint = self.model.addConstrs(
        (self.variables.eta_L[s] >= 
         self.variables.zeta_L - (load_const_per_scenario[s] + 
         gp.quicksum((price_L_array[t,s] - self.variables.S) * self.variables.M 
                     for t in self.data.TIME))
         for s in self.data.SCENARIOS_L),
        name='Eta_Aversion_Constraint_L'
    )

        self.model.update()

    def _build_constraints_PAP(self):

        """Build constraints for contract negotiation."""
         # Single Strike Price Constraint
        self.constraints.strike_price_constraint_max =  self.model.addLConstr(
        self.variables.S<=self.data.strikeprice_max ,
        name='Strike_Price_Constraint_Max'
        )
        
        
        self.constraints.strike_price_constraint_min =  self.model.addLConstr(
        self.variables.S>=self.data.strikeprice_min  ,
        name='Strike_Price_Constraint_Max'
        )

        #Contract amounts 
        
        #Contract amount constraints
        self.constraints.gamma_max = self.model.addLConstr(
           self.variables.gamma <= self.data.gamma_max ,  # Convert to GWh/year
            name='Contract Amount Constraint Max'
        )
         #Contract amount constraints
        self.constraints.gamma_min = self.model.addLConstr(
            0 <= self.variables.gamma ,  # Convert to GWh/year
            name='Contract Amount Constraint Min'
        )

        #Logarithmic constraints
        self.model.addGenConstrLog(self.variables.arg_G, self.variables.log_arg_G, "log_G")
        self.model.addGenConstrLog(self.variables.arg_L, self.variables.log_arg_L, "log_L")

    
        
        # Pre-extract all data as numpy arrays (do this once)
        production_G_vals = self.data.production_G.values  # Shape: (time, scenarios)
        capture_rate_vals = self.data.capture_rate.values  # Shape: (time, scenarios)
        price_G_vals = self.data.price_G.values  # Shape: (time, scenarios)
        price_L_vals = self.data.price_L.values  # Shape: (time, scenarios)
        load_CR_vals = self.data.load_CR.values  # Shape: (time, scenarios)
        load_scenarios_vals = self.data.load_scenarios.values  # Shape: (time, scenarios)
        production_L_vals = self.data.production_L.values  # Shape: (time, scenarios)

        # Pre-compute coefficients for each scenario
        # Generator constraints - coefficients for gamma*S term
        gamma_S_coeff_G_per_scenario = production_G_vals.sum(axis=0)  # Sum over time for each scenario

        # Generator constraints - constant terms for (1-gamma) term
        const_G_per_scenario = (capture_rate_vals * price_G_vals * production_G_vals).sum(axis=0)

        # Load constraints - constant terms
        const_L_per_scenario = (-price_L_vals * load_CR_vals * load_scenarios_vals).sum(axis=0)

        # Load constraints - coefficients for gamma term
        gamma_coeff_L_per_scenario = (production_L_vals * price_L_vals * capture_rate_vals).sum(axis=0)

        # Load constraints - coefficients for gamma*S term
        gamma_S_coeff_L_per_scenario = -production_L_vals.sum(axis=0)


        

        self.constraints.eta_G_constraint = self.model.addConstrs(
        (self.variables.eta_G[s] >= (
            self.variables.zeta_G 
            - self.variables.gamma * self.variables.S * gamma_S_coeff_G_per_scenario[s]
            - (1 - self.variables.gamma) * const_G_per_scenario[s]
            ) for s in range(len(self.data.SCENARIOS_L))),
            name='Eta_Aversion_Constraint_G'
        )

        self.constraints.eta_L_constraint = self.model.addConstrs(
            (self.variables.eta_L[s] >= (
        self.variables.zeta_L 
            - const_L_per_scenario[s]
            - self.variables.gamma * gamma_coeff_L_per_scenario[s]
            - self.variables.gamma * self.variables.S * gamma_S_coeff_L_per_scenario[s]
        ) for s in range(len(self.data.SCENARIOS_L))),
        name='Eta_Aversion_Constraint_L'
        )

    def _build_objectives_PAP(self):

        # Pre-extract all data as numpy arrays (do this once)
        prob_vals = self.data.PROB  # Shape: (scenarios,)
        production_G_vals = self.data.production_G.values  # Shape: (time, scenarios)
        capture_rate_vals = self.data.capture_rate.values  # Shape: (time, scenarios)
        price_G_vals = self.data.price_G.values  # Shape: (time, scenarios)
        price_L_vals = self.data.price_L.values  # Shape: (time, scenarios)
        load_CR_vals = self.data.load_CR.values  # Shape: (time, scenarios)
        load_scenarios_vals = self.data.load_scenarios.values  # Shape: (time, scenarios)
        production_L_vals = self.data.production_L.values  # Shape: (time, scenarios)

        # Pre-compute coefficients for all scenarios at once
        # Generator utility coefficients
        gamma_coeff_G = (prob_vals[np.newaxis, :] * production_G_vals).sum()  # Coefficient for gamma*S
        non_gamma_coeff_G = (prob_vals[np.newaxis, :] * capture_rate_vals * 
                            price_G_vals * production_G_vals).sum()  # Constant coefficient for (1-gamma) term

        # Load utility coefficients  
        load_base_cost = -(prob_vals[np.newaxis, :] * price_L_vals * load_CR_vals * 
                        load_scenarios_vals).sum()  # Constant term

        gamma_price_coeff_L = (prob_vals[np.newaxis, :] * production_L_vals * 
                            price_L_vals * capture_rate_vals).sum()  # Coefficient for gamma

        gamma_S_coeff_L = -(prob_vals[np.newaxis, :] * production_L_vals).sum()  # Coefficient for gamma*S

        # CVaR coefficients
        cvar_coeff = sum(self.data.PROB[s] for s in self.data.SCENARIOS_L)
        eta_G_sum = gp.quicksum(self.data.PROB[s] * self.variables.eta_G[s] 
                            for s in self.data.SCENARIOS_L)
        eta_L_sum = gp.quicksum(self.data.PROB[s] * self.variables.eta_L[s] 
                            for s in self.data.SCENARIOS_L)

        # Build expressions using pre-computed coefficients
        EuG = (self.variables.gamma * self.variables.S * gamma_coeff_G + 
            (1 - self.variables.gamma) * non_gamma_coeff_G)

        EuL = (load_base_cost + 
            self.variables.gamma * gamma_price_coeff_L + 
            self.variables.gamma * self.variables.S * gamma_S_coeff_L)

        # CVaR calculations
        CVaRG = self.variables.zeta_G - (1/(1-self.data.alpha)) * eta_G_sum
        CVaRL = self.variables.zeta_L - (1/(1-self.data.alpha)) * eta_L_sum

        # Calculate utilities with risk aversion
        UG = (1-self.data.A_G) * EuG + self.data.A_G * CVaRG
        UL = (1-self.data.A_L) * EuL + self.data.A_L * CVaRL

        # Set objective (same logic as before)
        if self.data.tau_G == 1:
            self.model.setObjective((UG - self.data.Zeta_G), GRB.MAXIMIZE)
            self.model.addConstr(UL - self.data.Zeta_L >= 0, "UL_non_negative")
            print("sheep")
        elif self.data.tau_L == 1:
            self.model.setObjective((UL - self.data.Zeta_L), GRB.MAXIMIZE)
            self.model.addConstr(UG - self.data.Zeta_G >= 0, "UG_non_negative")
            print("wolf")
        else:
            # Link auxiliary variables to expressions
            self.model.addConstr(self.variables.arg_G == (UG - self.data.Zeta_G), "arg_G_constr")
            self.model.addConstr(self.variables.arg_L == (UL - self.data.Zeta_L), "arg_L_constr")
            self.model.setObjective(
                (self.data.tau_G * self.variables.log_arg_G + self.data.tau_L * self.variables.log_arg_L),
                GRB.MAXIMIZE)
            print("goat") 

    def _build_objective(self):
        """Build the objective function for contract negotiation."""
    

        # Pre-extract all data as numpy arrays (do this once)
        prob_vals = self.data.PROB  # Shape: (scenarios,)
        capture_rate_vals = self.data.capture_rate.values  # Shape: (time, scenarios)
        price_G_vals = self.data.price_G.values  # Shape: (time, scenarios)
        production_G_vals = self.data.production_G.values  # Shape: (time, scenarios)
        load_scenarios_vals = self.data.load_scenarios.values  # Shape: (time, scenarios)
        load_CR_vals = self.data.load_CR.values  # Shape: (time, scenarios)
        price_L_vals = self.data.price_L.values  # Shape: (time, scenarios)

        # Pre-compute all coefficients using vectorized operations
        # Generator utility components
        gen_revenue_const = np.sum(prob_vals[np.newaxis, :] * capture_rate_vals * 
                                price_G_vals * production_G_vals)

        # Coefficients for S and M terms in generator utility
        S_coeff_G = np.sum(prob_vals) * len(self.data.TIME)  # Coefficient for S*M
        M_coeff_G = -np.sum(prob_vals[np.newaxis, :] * price_G_vals)  # Coefficient for M

        # Load utility components  
        load_revenue_const = np.sum(prob_vals[np.newaxis, :] * load_scenarios_vals * 
                                (self.data.retail_price - load_CR_vals * price_L_vals))

        # Coefficients for S and M terms in load utility
        S_coeff_L = -np.sum(prob_vals) * len(self.data.TIME)  # Coefficient for S*M  
        M_coeff_L = np.sum(prob_vals[np.newaxis, :] * price_L_vals)  # Coefficient for M

        # CVaR components (these are already efficient)
        eta_G_sum = gp.quicksum(self.data.PROB[s] * self.variables.eta_G[s] 
                            for s in self.data.SCENARIOS_L)
        eta_L_sum = gp.quicksum(self.data.PROB[s] * self.variables.eta_L[s] 
                            for s in self.data.SCENARIOS_L)

        # Build expressions using pre-computed coefficients
        EuG = (gen_revenue_const + 
            S_coeff_G * self.variables.S * self.variables.M + 
            M_coeff_G * self.variables.M)

        EuL = (load_revenue_const + 
            S_coeff_L * self.variables.S * self.variables.M + 
            M_coeff_L * self.variables.M)

        # CVaR calculations
        CVaRG = self.variables.zeta_G - (1/(1-self.data.alpha)) * eta_G_sum
        CVaRL = self.variables.zeta_L - (1/(1-self.data.alpha)) * eta_L_sum

        # Calculate utilities with risk aversion
        UG = (1-self.data.A_G) * EuG + self.data.A_G * CVaRG
        UL = (1-self.data.A_L) * EuL + self.data.A_L * CVaRL

        self.model.update()

        # Set logarithmic objective (same logic as before)
        if self.data.tau_G == 1:
            self.model.setObjective((UG - self.data.Zeta_G), GRB.MAXIMIZE)
            self.model.addConstr(UL - self.data.Zeta_L >= 1e-6, "UL_non_negative")
            print("sheep")
        elif self.data.tau_L == 1:
            self.model.setObjective((UL - self.data.Zeta_L), GRB.MAXIMIZE)
            self.model.addConstr(UG - self.data.Zeta_G >= 1e-6, "UG_non_negative")
            print("wolf")
        else:
            # Link auxiliary variables to expressions
            self.model.addConstr(self.variables.arg_G == (UG - self.data.Zeta_G), "arg_G_constr")
            self.model.addConstr(self.variables.arg_L == (UL - self.data.Zeta_L), "arg_L_constr")
            self.model.setObjective(
                (self.data.tau_G * self.variables.log_arg_G + self.data.tau_L * self.variables.log_arg_L),
                GRB.MAXIMIZE
            )
            self.model.update()
            print("goat")

    def _build_model(self):
        """Initialize and build the complete optimization model."""
        self.model = gp.Model(name='Nash Bargaining Model')
        self.model.Params.NonConvex = 2
        self.model.Params.FeasibilityTol = 1e-6
        self.model.Params.OutputFlag = 0  # Suppress output
        self.model.Params.TimeLimit = 420  # Set time limit to 7 minutes

        self.model.Params.ObjScale   = 1e-6
        #self.model.Params.NumericFocus = 1
        
        if self.contract_type == 'PAP':
            self._build_variables_PAP()
            self._build_constraints_PAP()
            self._build_objectives_PAP()
        else:
            self._build_variables()
            self._build_constraints()
            self._build_objective()
        self.model.update()

    def _save_results(self):
        """Save optimization results."""
        # Save objective value, strike price, and contract amount
        self.results.objective_value = self.model.ObjVal
        self.results.strike_price = self.variables.S.x * 1e3
        if self.contract_type == 'PAP':
            #self.results.contract_amount = self.variables.gamma.x * (self.data.production.sum().mean()/len(self.data.TIME))  # Convert to GWh/year
            self.results.contract_amount = self.variables.gamma.x  * self.data.generator_contract_capacity * self.data.hours_in_year # yearly
            self.results.gamma = self.variables.gamma.x
            self.results.contract_amount_hour = self.results.gamma * self.data.generator_contract_capacity  # hourly

        else:
            self.results.contract_amount = self.variables.M.x  # GWh/year
            self.results.contract_amount_hour = self.results.contract_amount / 8760 * 1e3  # Convert GWh/year to MWh/hour
        self.results.capture_price = self.data.Capture_price_L_avg

        # Save their 'actual' values based on the true distribution 
        strike = self.variables.S.x
        # Calculate revenues with contract
        if self.contract_type == 'PAP':
            EuG = ((1-self.results.gamma)* self.data.capture_rate * self.data.price_G * self.data.production_G).sum(axis=0)
            SMG = (self.results.gamma * self.data.production_G * strike).sum(axis=0)   # Sum across time periods for each scenario
            
            
            
            EuL = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios).sum(axis=0) # Sum across time periods for each scenario
            SML =   (self.results.gamma* self.data.production_L * self.data.price_L * self.data.capture_rate -  self.results.gamma * strike * self.data.production_L).sum(axis=0) # Sum across time periods for each scenario

            SMG_CP =   (self.results.gamma * self.data.production_G *  self.data.Capture_price_G_avg).sum(axis=0)   # Sum across time periods for each scenario
            SML_CP =     (self.results.gamma* self.data.production_L * self.data.price_L * self.data.capture_rate -  self.results.gamma * self.data.Capture_price_G_avg * self.data.production_L).sum(axis=0) # Sum across time periods for each scenario

            CV_CP_G = calculate_cvar_left(EuG + SMG_CP, self.data.alpha)
            CV_CP_L = calculate_cvar_left(EuL + SML_CP, self.data.alpha)

            # Calculate utilities with capture price
            self.results.utility_G_CP = (1-self.data.A_G) * (EuG + SMG_CP).mean() + self.data.A_G * CV_CP_G
            self.results.utility_L_CP = (1-self.data.A_L) * (EuL + SML_CP).mean() + self.data.A_L * CV_CP_L

            # Calculate expected earnings for G and L
            self.results.earnings_G_CP = EuG + SMG_CP
            self.results.earnings_L_CP = EuL + SML_CP
        else:
            EuG = self.data.net_earnings_no_contract_priceG_G
            EuL = self.data.net_earnings_no_contract_priceL_L
            SMG = ((strike - self.data.price_G) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario
            SML = ((self.data.price_L - strike) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

            # Caclculate CP_load price 

            SMG_CP = ((self.data.Capture_price_G_avg - self.data.price_G) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario
            SML_CP = ((self.data.price_L- self.data.Capture_price_G_avg) * self.results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

            CV_CP_G = calculate_cvar_left(EuG + SMG_CP, self.data.alpha)
            CV_CP_L = calculate_cvar_left(EuL + SML_CP, self.data.alpha)

            # Calculate utilities with capture price
            self.results.utility_G_CP = (1-self.data.A_G) * (EuG + SMG_CP).mean() + self.data.A_G * CV_CP_G
            self.results.utility_L_CP = (1-self.data.A_L) * (EuL + SML_CP).mean() + self.data.A_L * CV_CP_L

            # Calculate expected earnings for G and L
            self.results.earnings_G_CP = EuG + SMG_CP
            self.results.earnings_L_CP = EuL + SML_CP
        
        # Calculate CVaR for L and G
        self.results.CVaRG = calculate_cvar_left(EuG + SMG, self.data.alpha)
        self.results.CVaRL = calculate_cvar_left(EuL + SML, self.data.alpha)

        self.results.utility_G = (1-self.data.A_G) * (EuG + SMG).mean() + self.data.A_G * self.results.CVaRG
        self.results.utility_L = (1-self.data.A_L) * (EuL + SML).mean() + self.data.A_L * self.results.CVaRL
        
        print( self.results.utility_G)
        print( self.results.utility_L)

        self.results.Nash_Product = ((self.results.utility_G - self.data.Zeta_G)**(0.5) * (self.results.utility_L - self.data.Zeta_L)**(0.5))
        # Save accumulated revenues
        self.results.earnings_G = EuG + SMG
        self.results.earnings_L = EuL + SML
        # Calculate Alternative Net earnings if capture price was used 

    def run(self):
        """Run the optimization model."""
        #self.manual_optimization(plot=False)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            self._save_results()
            #self.scipy_optimization()
            #self.display_results()
            #self.scipy_display_results()

            if self.data.Barter == True:
                #self.manual_optimization(plot=True)
                #self.batch_manual_optimization(A_G_values= [0.0,0.5,0.9],A_L_values=[0.0])
                BS = Barter_Set(self.data,self.results,self.scipy_results)
                BS.Plotting_Barter_Set()
        else:
            #self._save_results()
            #BS = Barter_Set(self.data,self.results,self.scipy_results)
            #BS.Plotting_Barter_Set()

            #self.scipy_optimization()
            #self.scipy_display_results()
            #self.manual_optimization(plot=True)
            #self.batch_manual_optimization(A_G_values= [0.0,0.5,0.9],A_L_values=[0.0])
            #BS = Barter_Set(self.data,self.results,self.scipy_results)
            #BS.Plotting_Barter_Set()
            
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        """Display optimization results."""
        print("\n-------------------   RESULTS GUROBI  -------------------")
        print(f"Optimal Objective Value (Log): {self.results.objective_value:.5f}")
        print(f"Optimal Objective Value {np.exp(self.results.objective_value):.5f}")
        print(f"Nash Product with optimal S and M: {self.results.Nash_Product:.5f}")
        print(f"Optimal Strike Price(EUR/MWh): {self.results.strike_price:.5f}")
        if self.contract_type == 'PAP':
            print(f"Optimal Contract Capacity (%): {self.results.gamma:.5f}")
        print(f"Optimal Contract Amount(GWh/year): {self.results.contract_amount:.5f}")
        print(f"Optimal Contract Amount(MWh): {self.results.contract_amount_hour:.5f}")
        print(f"Optimal Utility for G: {self.results.utility_G:.5f}")
        print(f"Optimal Utility for L: {self.results.utility_L:.5f}")
        print(f"Threat Point G: {self.data.Zeta_G:.5f}")
        print(f"Threat Point L: {self.data.Zeta_L:.5f}")
        print(f"Delta G: {self.results.utility_G - self.data.Zeta_G:.5f}")
        print(f"Delta L: {self.results.utility_L - self.data.Zeta_L:.5f}")

    def manual_optimization(self, plot=False, filename=None):
        """
        Perform manual optimization by grid search over strike prices.
        """
        strike_prices = np.linspace(self.data.strikeprice_min, self.data.strikeprice_max, 3000)

        if self.contract_type == 'PAP':
            M_opt = 1 # %
        else:
            M_opt = 116 # MWh
            #contract_amounts = np.linspace(self.data.contract_amount_min, self.data.contract_amount_max, 200)
        #M = 300  # arbitarry
        #M_opt =  396.5062071155 

        #utilities_G = np.zeros((len(strike_prices), len(contract_amounts)))
        #utilities_L = np.zeros((len(strike_prices), len(contract_amounts)))
        #combined_utilities = np.zeros((len(strike_prices), len(contract_amounts)))
        #combined_utilities_log = np.zeros((len(strike_prices), len(contract_amounts)))

        utilities_G = np.zeros(len(strike_prices))
        utilities_L = np.zeros(len(strike_prices))
        combined_utilities =np.zeros(len(strike_prices))
        combined_utilities_log = np.zeros(len(strike_prices))


        #SR_test = np.zeros((len(strike_prices), len(contract_amounts)))
        #SU_test = np.zeros((len(strike_prices), len(contract_amounts)))
        
        """
        Could speed up by making SMG and SML numpya arrays from the start 
        """
        time = self.data.price_true.shape[0]

        for i,strike in enumerate(tqdm(strike_prices, desc='loading...')):
            #for j,M in enumerate(contract_amounts):

            #SR_test[i] = (1-self.data.A_L) * self.data.capture_prod_price_lambda_expected +self.data.K_L_lambda_Sigma +strike*time - self.data.A_L * self.data.left_Cvar_neg_capture_prod_price_lambda_sum_true 
            #SR_test[i] = (1-self.data.A_L) * self.data.capture_prod_price_lambda_expected +self.data.K_L_lambda_Sigma +strike*time + self.data.A_L * self.data.left_Cvar_capture_prod_price_lambda_sum_true 
            if self.contract_type == 'PAP':
                E_G = (1-M_opt) * self.data.capture_rate * self.data.price_G * self.data.production_G  # Expected earnings for G
                E_L = -self.data.price_L * self.data.load_CR * self.data.load_scenarios
                                                                

                SMG = M_opt * self.data.production_G * strike  # Sum across time periods for each scenario
                SML =M_opt* self.data.production_L * self.data.price_L * self.data.capture_rate - M_opt * strike * self.data.production_L  # Sum across time periods

                Scen_revenue_G = E_G.sum(axis=0) + SMG.sum(axis=0) 
                Scen_revenue_L = E_L.sum(axis=0) + SML.sum(axis=0)

            else:
            # Calculate revenues for G
                SMG = (strike - self.data.price_G) * M_opt
                # Calculate revenues for L
                SML = ( self.data.price_L-strike) * M_opt

                # Calculate total revenues
                Scen_revenue_G = SMG.sum(axis=0) + self.data.net_earnings_no_contract_priceG_G
                Scen_revenue_L = SML.sum(axis=0) + self.data.net_earnings_no_contract_priceL_L

            # Calculate CVaR
            CVaRG = calculate_cvar_left(Scen_revenue_G.values, self.data.alpha)
            CVaRL = calculate_cvar_left(Scen_revenue_L, self.data.alpha)

        
            UG = (1 - self.data.A_G) * Scen_revenue_G.mean() + self.data.A_G * CVaRG
            UL = (1 - self.data.A_L) * Scen_revenue_L.mean() + self.data.A_L * CVaRL
            # Calculate the Nash product (combined utility)
            combined_utility = (UG - self.data.Zeta_G ) * (UL - self.data.Zeta_L) 
            # Calculate the Nash product (combined utility)
            combined_utility_log = np.log(np.maximum(UG - self.data.Zeta_G , 1e-10)) + np.log(np.maximum(UL - self.data.Zeta_L , 1e-10)) # Avoid log(0) by using a small value
            combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
        
            utilities_G[i] = UG
            utilities_L[i] = UL
            combined_utilities[i] = combined_utility
            combined_utilities_log[i] = combined_utility_log

       # Find indices of maximum utilities
        max_idx = np.unravel_index(np.argmax(combined_utilities), combined_utilities.shape)
        max_idx_log = np.unravel_index(np.argmax(combined_utilities_log), combined_utilities_log.shape)

        # Get bhvn  values
        optimal_strike_price = strike_prices[max_idx[0]] * 1e3
        #optimal_amount = contract_amounts[max_idx[1]]
        #optimal_amount_hour = optimal_amount/8760*1e3
        optimal_strike_price_log = strike_prices[max_idx_log[0]] *1e3
        optimal_strike_price_log_hour = optimal_strike_price_log
        #optimal_amount_log = contract_amounts[max_idx_log[1]]

        # Get maximum utilities
        max_utility_G = utilities_G[max_idx]
        max_utility_L = utilities_L[max_idx]
        max_combined_utility = combined_utilities[max_idx]
        max_combined_utility_log = combined_utilities_log[max_idx_log]

        
        #Works if M is kept constant
        # Find first non-negative value
        """
        first_positive_idx = next((i for i, x in enumerate(combined_utilities) if x >= 0), None)
        last_positive_idx = next((i for i, x in enumerate(reversed(combined_utilities)) if x >= 0), None)

        if first_positive_idx is not None:
            print(f"\nFirst non-negative combined utility:")
            print(f"Strike price [hourly]: {strike_prices[first_positive_idx]/8760*1e3:.5f}")

        if last_positive_idx is not None:
            last_positive_idx = len(combined_utilities) - last_positive_idx - 1
            print(f"\nLast non-negative combined utility:")
            print(f"Strike price[hourly]: {strike_prices[last_positive_idx]/8760*1e3:.5f}")
        
        """
               # Find first and last positive combined utility values
        positive_indices = np.where(combined_utilities > 0)[0]
        if len(positive_indices) > 0:
            first_positive_idx = positive_indices[0]
            last_positive_idx = positive_indices[-1]
            first_positive_strike = strike_prices[first_positive_idx]
            last_positive_strike = strike_prices[last_positive_idx]
            print("\n-------------------   Positive Utility Range   -------------------")
            print(f"First positive strike price [EUR/MWh]: {first_positive_strike*1e3:.5f}")
            print(f"Last positive strike price [EUR/MWh]: {last_positive_strike*1e3:.5f}")
            print(f"Range width [EUR/MWh]: {(last_positive_strike*1e3 - first_positive_strike*1e3):.5f}")
            self.data.man_SR_star = first_positive_strike
            self.data.man_SU_star = last_positive_strike

            #self.data.strikeprice_min = (self.data.man_SR_star / 8760*1e3)-10
            #self.data.strikeprice_max = (self.data.man_SR_star /8760*1e3)-10

            #self.data.strikeprice_min =10
            #self.data.strikeprice_max =10


        # Include threat points for comparison
        print("\n-------------------   RESULTS Iterative  -------------------")
        print(f"Threat Point G: {self.data.Zeta_G}")
        print(f"Threat Point L: {self.data.Zeta_L}")
        print(f'Risk Aversion G: {self.data.A_G}')
        print(f'Risk Aversion G: {self.data.A_L}')

        #print(f"Optimal Strike Price: {optimal_strike_price}")
        print(f"Optimal Strike Price log[Yearly]: {optimal_strike_price_log}")
        print(f"Optimal Strike Price log[Hourly]: {optimal_strike_price_log_hour}")

        #print(f"Optimal  Amount [Yearly]: {optimal_amount}")
        #print(f"Optimal  Amount(On Average) [Hourly]: {optimal_amount_hour}")

        #print(f"Optimal Amount log: {optimal_amount_log}")
        print(f"Maximum Utility G: {max_utility_G}")
        print(f"Maximum Utility L: {max_utility_L}")
        print(f"Difference in Utility G: {max_utility_G - self.data.Zeta_G}")
        print(f"Difference in Utility L: {max_utility_L - self.data.Zeta_L}")

        print(f"Maximum Combined Utility (Nash Product): {max_combined_utility}")
        #print(f"Maximum Combined Utility Log (Nash Product): {np.exp(combined_utilities_log[max_combined_utility_index_log])}")
        print(f"Maximum Combined Utility Log (Nash Product): {max_combined_utility_log}")

        if plot == True:
            print("Plotting results...")
            #Example of how it looks - Would have to specify Amount to better see the results
            # Create Plots directory if it doesn't exist
            #fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            #axs[0].plot(SR_test,label="SR")
            #axs[1].plot(SU_test,label="SU")
            #plt.show()

            # Plot the utilities and Nash product for different strike prices
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))            # First subplot: Utilities for G and L
            sc = axs[0].scatter(utilities_G, utilities_L, c=combined_utilities, cmap='viridis', s=30)
            axs[0].set_xlabel("Utility L ")
            axs[0].set_ylabel("Utility G ")
            axs[0].set_title(f"Generator and Load Utilities vs Strike Price")
            axs[0].grid(True, alpha=0.3)            # Second subplot: Nash Product and Log Nash Product
            
            line_nash_log = axs[1].plot(strike_prices * 1e3, combined_utilities, label="Nash Product", color="green", linestyle="-")[0]
            line_nash = axs[1].plot(strike_prices, np.exp(combined_utilities_log), label="Log Nash Product", color="purple", linestyle="-")[0]
            optimal_line = axs[1].axvline(optimal_strike_price, color="red", linestyle="--", label=f"S*={optimal_strike_price:.2f}")
            optimal_log_line = axs[1].axvline(optimal_strike_price_log, color="orange", linestyle="--", label=f"S* log={optimal_strike_price_log:.2f}")
            axs[1].set_xlabel("Strike Price [EUR/MWh]")
            axs[1].set_ylabel("Nash Product Value")
            axs[1].set_title("Nash Product Comparison")
            handles = [line_nash_log, line_nash, optimal_line, optimal_log_line]
            axs[1].legend(handles=handles, loc='upper right')
            axs[1].grid(True, alpha=0.3)
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
            print()


            dA = self.data.Zeta_G        # threat points already computed in your code
            dB = self.data.Zeta_L

            # 1) Order points so they draw a clean frontier
            order        = np.argsort(utilities_G)
            uA_sorted    = utilities_G[order]
            uB_sorted    = utilities_L[order]
            nprod_sorted = combined_utilities[order]

            # 2) Locate the Nash point (maximum Nash-product)
            idx_N   = np.argmax(combined_utilities)
            uA_N    = utilities_G[idx_N]
            uB_N    = utilities_L[idx_N]
            K_star  = combined_utilities[idx_N]          # the constant in (uA−dA)(uB−dB)=K*

            # 3) Generate the iso-Nash hyperbola that passes through the Nash point
            uA_iso = np.linspace(dA+1, uA_N+3, 400)
            uB_iso = dB + K_star / (uA_iso - dA)

            # 4) Plot ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 6))

            # Pareto frontier g
            ax.plot(utilities_G, utilities_L,
                    lw=2, label=r'Pareto frontier $g$')

            # Threat point (dA,dB)
            ax.scatter(dA, dB, color='black', zorder=3)
            #ax.text(dA, dB, r'$(d_A,d_B)$', ha='right', va='top')

            # Nash point uN
            ax.scatter(uA_N, uB_N, s=80, color='red', zorder=4)
            #ax.text(uA_N, uB_N, r'$u^{N}$', ha='left', va='bottom')

            # Iso-Nash product curve
            ax.plot(uA_iso, uB_iso, ls='--',
                    label=r'$(u_A-d_A)(u_B-d_B)=\text{const}$')

            # Cosmetic tweaks
            ax.set_xlabel(r'$u_A$')
            ax.set_ylabel(r'$u_B$')
            ax.set_xlim(left=dA*0.9)
            #ax.set_ylim(bottom=dB*0.9)
            plt.axvline(x=dA, color='black', linestyle='--', alpha=0.3)
            plt.axhline(y=dB, color='black', linestyle='--', alpha=0.3)
            ax.grid(alpha=0.3)
            ax.legend()
            plt.tight_layout()
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
        strike_prices = np.linspace(self.data.strikeprice_min , self.data.strikeprice_max+100*1e-3, 1000)

        # Create a color map for AG values - using tab20 for more distinct colors
        base_colors = plt.cm.Set2(np.linspace(0, 1, len(A_G_values)))
        # Number of gradient steps for each AL value
        gradient_steps = len(A_L_values)

        for ag_idx, a_G in enumerate(tqdm(A_G_values, desc='loading batch...')):
            base_color = base_colors[ag_idx]
            # Create gradient colors for this AG value - using a wider range of alpha values
            alpha_values = np.linspace(0.5, 1.0, gradient_steps)  # Increased minimum alpha for better visibility
            

            Zeta_G = (1-a_G) * self.data.net_earnings_no_contract_priceG_G.mean() + a_G * self.data.CVaR_no_contract_priceG_G

            for al_idx, a_L in enumerate(A_L_values):
                Zeta_L = (1-a_L) * self.data.net_earnings_no_contract_priceL_L.mean() + a_L * self.data.CVaR_no_contract_priceL_L
                utilities_G = []
                utilities_L = []
                combined_utilities_log = []
                
                # Update risk aversion values
                A_G = a_G
                A_L = a_L
                
                for strike in tqdm(strike_prices, desc='loading...'):

                    if self.contract_type == 'PAP':
                        E_G = (1-1) * self.data.capture_rate * self.data.price_G * self.data.production  # Expected earnings for G
                        E_L = (-self.data.price_L * self.data.load_CR * self.data.load_scenarios)
                                                                        

                        SMG = 1 * self.data.production_G * strike  # Sum across time periods for each scenario
                        SML =  1 * self.data.production_L * self.data.price_L * self.data.capture_rate - 1* strike * self.data.production_L

                        Scen_revenue_G = E_G.sum(axis=0) + SMG.sum(axis=0) 
                        Scen_revenue_L = E_L.sum(axis=0) + SML.sum(axis=0)
                    else:
                        # Calculate revenues for G
                        SMG = (strike - self.data.price_G) * self.results.contract_amount

                        # Calculate revenues for L
                        SML = ( self.data.price_L-strike) * self.results.contract_amount

                        # Calculate total revenues
                        Scen_revenue_G = SMG.sum(axis=0) + self.data.net_earnings_no_contract_priceG_G
                        Scen_revenue_L = SML.sum(axis=0) + self.data.net_earnings_no_contract_priceL_L

                    # Calculate CVaR
                    CVaRG = calculate_cvar_left(Scen_revenue_G.values, self.data.alpha)
                    CVaRL = calculate_cvar_left(Scen_revenue_L.values, self.data.alpha)

                    # Calculate utility for G
                  
                    UG = (1 - self.data.A_G) * Scen_revenue_G.mean() + A_G * CVaRG
                    UL = (1 - self.data.A_L) * Scen_revenue_L.mean() + A_L * CVaRL

                    utilities_G.append(UG)
                    utilities_L.append(UL)

                    # Calculate the Nash product (combined utility)
                    combined_utility_log = np.log(np.maximum(UG - Zeta_G, 1e-10)) + np.log(np.maximum(UL - Zeta_L, 1e-10))
                    combined_utility_log = np.nan_to_num(combined_utility_log)  # Handle NaN values
                    combined_utility_log = np.maximum(combined_utility_log, 0)  # Ensure non-negative values    
                    combined_utilities_log.append(combined_utility_log)

                # Plot with gradient color
                current_color = base_color.copy()
                current_color[3] = alpha_values[al_idx]  # Modify alpha for gradient effect
                ax.plot(strike_prices*1e3, combined_utilities_log, 
                    label=f'AG={a_G:.1f}, AL={a_L:.1f}',
                    color=current_color,
                    linewidth=2,)  # Increased line width for better visibility

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
            hours_in_year=self.data.hours_in_year,
            price_G = self.data.price_G,
            price_L = self.data.price_L,
            A_G=self.data.A_G,
            A_L=self.data.A_L,
            tau_G=self.data.tau_G,
            tau_L=self.data.tau_L,
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
        self.scipy_results.strike_price = S_opt #Mio EuR /GWh
        self.scipy_results.strike_hour = S_opt*1000
        self.scipy_results.contract_amount = M_opt # Convert to GWh/year
        self.scipy_results.contract_amount_hour = M_opt /self.data.hours_in_year * 1000  # Convert to GWh/year
        self.scipy_results.objective_value = np.log(Nash_Product)  # Convert to log form to match Gurobi
        self.scipy_results.Nash_Product = Nash_Product
        
        #True Earnings with true probablity distribution
        # Calculate utilities with optimal values
        EuG = self.data.net_earnings_no_contract_true_G
        SMG = ((self.scipy_results.strike_price - self.data.price_true) * self.scipy_results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario
        # Calculate CVaR for results G 
        self.scipy_results.CVaRG = calculate_cvar_left(EuG + SMG, self.data.alpha)

        # Calculate revenues for L
        EuL = self.data.net_earnings_no_contract_true_L
        SML = ((self.data.price_true - self.scipy_results.strike_price) * self.scipy_results.contract_amount).sum(axis=0)  # Sum across time periods for each scenario

        # Calculate CVaR for L
        self.scipy_results.CVaRL = calculate_cvar_left(EuL + SML, self.data.alpha)


        self.scipy_results.utility_G = (1-self.data.A_G) * (EuG + SMG).mean() + self.data.A_G * self.scipy_results.CVaRG
        self.scipy_results.utility_L = (1-self.data.A_L) * (EuL + SML).mean() + self.data.A_L * self.scipy_results.CVaRL
           

        self.scipy_results.Nash_Product = ((self.scipy_results.utility_G - self.data.Zeta_G) * (self.scipy_results.utility_L - self.data.Zeta_L))
        # Save accumulated revenues
        #re calcuate 
        self.scipy_results.accumulated_revenue_True_G = (EuG + SMG)
        self.scipy_results.accumulated_revenue_True_L = (EuL + SML)

        # Results with biased distribution 

        print("\n-------------------  THREAT POINTS --------------------")
        print(f"Threat Point G: {self.data.Zeta_G}")
        print(f"Threat Point L: {self.data.Zeta_L}")

        print("\n-------------------   RESULTS SCIPY  -------------------")
        print(f"Optimal Strike Price: {  self.scipy_results.strike_hour}")
        print(f"Optimal Contract Amount: {  self.scipy_results.contract_amount}")
        print(f"Nash Product Value: {  self.scipy_results.objective_value}")
        print(f"Utility G: {  self.scipy_results.utility_G}")
        print(f"Utility L: {  self.scipy_results.utility_L}")
        print("SciPy optimization complete.")

         # Calculate average revenues
        g_no_contract = self.data.net_earnings_no_contract_true_G.mean()
        g_with_contract_scipy = self.scipy_results.accumulated_revenue_True_G.mean()
        
        l_no_contract = self.data.net_earnings_no_contract_true_L.mean()
        l_with_contract_scipy = self.scipy_results.accumulated_revenue_True_L.mean()
        
        print("\nGenerator Revenue (True Distribution):")
        print(f"Without Contract: {g_no_contract:.2f}")
        print(f"With Contract (SciPy): {g_with_contract_scipy:.2f}")
        print(f"Improvement (SciPy): {((g_with_contract_scipy - g_no_contract)/abs(g_no_contract))*100:.2f}%")
        
        print("\nLoad Revenue (True Distribution):")
        print(f"Without Contract: {l_no_contract:.2f}")
        print(f"With Contract (SciPy): {l_with_contract_scipy:.2f}")
        print(f"Improvement (SciPy): {((l_with_contract_scipy - l_no_contract)/abs(l_no_contract))*100:.2f}%")
        print("")

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
        print(f"Strike Price [EUR/MWh)]   | {self.results.strike_price:2f} | {  self.scipy_results.strike_hour:10.4f} | {(self.results.strike_price -   self.scipy_results.strike_hour):10.4f}")
        print(f"Contract Amt[MWh]   | {self.results.contract_amount_hour:2f} | {  self.scipy_results.contract_amount_hour:10.10f} | {(self.results.contract_amount_hour -   self.scipy_results.contract_amount_hour):10.10f}")
        print(f"Contract Amt[GWh/Year]   | {self.results.contract_amount:2f} | {  self.scipy_results.contract_amount:10.10f} | {(self.results.contract_amount -   self.scipy_results.contract_amount):10.10f}")
        print(f"Nash Product   | {self.results.Nash_Product:2f} | {  self.scipy_results.Nash_Product:10.4f} | {(self.results.Nash_Product -   self.scipy_results.Nash_Product):10.4f}")        
        print(f"Utility G     | {self.results.utility_G:2f} | {  self.scipy_results.utility_G:10.1f} | {(self.results.utility_G -   self.scipy_results.utility_G):10.4f}")
        print(f"Utility L     | {self.results.utility_L:2f} | {  self.scipy_results.utility_L:10.1f} | {(self.results.utility_L -   self.scipy_results.utility_L):10.4f}")
        
        print("\n-------------------   REVENUE COMPARISON  -------------------")
       
   