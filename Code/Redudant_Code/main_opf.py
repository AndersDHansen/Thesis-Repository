"""
Main execution script for power system contract negotiation analysis.
"""
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
from Code.Redudant_Code.data_management import load_data, InputData
from power_flow import OptimalPowerFlow
from visualization import Plotting_Class
from sensitivity_analysis import run_risk_sensitivity_analysis, run_bias_sensitivity_analysis
from contract_negotiation import ContractNegotiation
import copy
def run_contract_negotiation(input_data: InputData, opf_results, old_obj_func: bool, 
                           A_G6_values: np.ndarray, A_L2_values: np.ndarray) -> tuple:
    """Run contract negotiation and sensitivity analysis with given OPF results."""
    # Run Contract Negotiation
    # Store original values
    original_A_G6 = input_data.A_G6
    original_A_L2 = input_data.A_L2
    original_K_G6 = input_data.K_G6
    original_K_L2 = input_data.K_L2
    
    input_data.A_G6 = A_G6_values[1]  # Use middle value for A_G6
    input_data.A_L2 = A_L2_values[1]  # Use middle value for A_L2
  
    contract_model = ContractNegotiation(input_data, opf_results, old_obj_func=old_obj_func)
    contract_model.run()

    print(2*"\nRunning contract negotiation...")
    
    # Run Sensitivity Analyses
    #risk_sensitivity_results, earnings_sensitivity = run_risk_sensitivity_analysis(input_data, opf_results, A_G6_values, A_L2_values, old_obj_func=old_obj_func)

   # bias_sensitivity_results = run_bias_sensitivity_analysis(input_data, opf_results,old_obj_func)
    
             # Restore original values
    input_data.A_G6 = original_A_G6
    input_data.A_L2 = original_A_L2
    input_data.K_G6 = original_K_G6
    input_data.K_L2 = original_K_L2
    
    

    return contract_model#, risk_sensitivity_results, earnings_sensitivity#, bias_sensitivity_results



def main():    
    # Define simulation parameters
    HOURS = 24
    DAYS = 6
    SCENARIOS = 25
    A_L2 = 0.5 # Initial risk aversion
    A_G6 = 0.5 # Initial risk aversion
    

    # Load data and create InputData object 
    print("Loading data and preparing for simulation...")
    (GENERATORS, LOADS, NODES, TIME, SCENARIOS_L, PROB,
     fixed_Cost, generator_cost_a, generator_cost_b, generator_capacity,
     load_capacity, mapping_buses, mapping_generators, mapping_loads,
     branch_capacity_df, bus_reactance_df, System_data,
     strikeprice_min, retail_price, _, strikeprice_max,
     contract_amount_min, contract_amount_max,
     A_L2, A_G6, K_L2, K_G6, alpha) = load_data(
        hours=HOURS,
        days=DAYS,
        scen=SCENARIOS,
        A_G6=A_G6, #A_L2_value,
        A_L2=A_L2 #A_G6_value,
    )

    # Create InputData object
    input_data = InputData(
        GENERATORS=GENERATORS,
        LOADS=LOADS,
        NODES=NODES,
        TIME=TIME,
        SCENARIOS_L=SCENARIOS_L,
        PROB=PROB,
        generator_cost_fixed=fixed_Cost,
        generator_cost_a=generator_cost_a,
        generator_cost_b=generator_cost_b,
        generator_capacity=generator_capacity,
        load_capacity=load_capacity,
        mapping_buses=mapping_buses,
        mapping_generators=mapping_generators,
        mapping_loads=mapping_loads,
        branch_capacity=branch_capacity_df,
        bus_susceptance=bus_reactance_df,
        slack_bus='N1',
        system_data=System_data,
        retail_price=retail_price,
        strikeprice_min=strikeprice_min,
        strikeprice_max=strikeprice_max,
        contract_amount_min=contract_amount_min,
        contract_amount_max=contract_amount_max,
        A_L2=A_L2,
        A_G6=A_G6,
        K_L2=K_L2,
        K_G6=K_G6,
        alpha=alpha
    )

     # Define risk aversion parameters for both objective functions
    
    
    new_obj_params = {
        #'A_G6_values': np.round(np.linspace(1, 0.5 , 3), 2),  # A in [0,1]
        #'A_L2_values': np.round(np.linspace(1, 0.5, 3), 2)
        'A_G6_values':  np.array([0.1,0.5,0.9]),  # A in [0,1]
        'A_L2_values':  np.array([0.1,0.5,0.9])
    }

    old_obj_params = {
        #'A_G6_values': np.array([1, 1, 1]),  # A >= 0
        #A_L2_values': np.array([1, 1, 1])
        'A_G6_values': np.round(1/new_obj_params['A_G6_values']-1,1), 
        'A_L2_values': np.round(1/new_obj_params['A_L2_values']-1,1) # A >= 0
    }

      # Run OPF
    opf_model = OptimalPowerFlow(input_data)
    opf_model.run()
    opf_results = opf_model.results

    
    # Run contract negotiation for both objective functions with same OPF results
    print("\nRunning simulation with original objective function (E + A*CVaR)...")
    """
    old_results = run_contract_negotiation(
        copy.deepcopy(input_data), 
        opf_results,
        True, 
        old_obj_params['A_G6_values'], 
        old_obj_params['A_L2_values']
    )
   """
    
    print("\nRunning simulation with modified objective function ((1-A)*E + A*CVaR)...")
    new_results = run_contract_negotiation(
        copy.deepcopy(input_data),
        opf_results,
        False,
        new_obj_params['A_G6_values'],
        new_obj_params['A_L2_values']
    )
    
    # Create comparison plots
    print("\nGenerating comparison plots...")

    print(new_results[1])
    print(old_results[1])
    
    # Plot results for original objective function
    plot_obj_old = Plotting_Class(
        old_results[0].data,
        old_results[1],
        old_results[2],
        old_results[3]
    )
    
    # Plot results for new objective function
    plot_obj_new = Plotting_Class(
        new_results[0].data,
        new_results[1],
        new_results[2],
        new_results[3]
    )

        
       # Generate plots with distinctive filenames
    for plot_obj, obj_type, params in [
        (plot_obj_old, 'original', old_obj_params),
        (plot_obj_new, 'modified', new_obj_params)
    ]:
        # Risk sensitivity plots
        plot_obj._plot_sensitivity_results(
            filename=f"risk_sensitivity_{obj_type}.png"
        )
        
        # Earnings distribution plots
        plot_obj._plot_earnings_histograms(
            fixed_A_G6=params['A_G6_values'][1],  # Use middle value
            A_L2_to_plot=params['A_L2_values'].tolist(),
            filename=f"earnings_distribution_{obj_type}.png"
        )
        
        # Bias sensitivity plots
        plot_obj._plot_bias_sensitivity(
            filename=f"bias_sensitivity_{obj_type}.png"
        )
      
    
if __name__ == "__main__":
    main()