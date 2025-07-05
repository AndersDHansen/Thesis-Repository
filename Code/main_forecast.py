"""
Main execution script for power system contract negotiation analysis.
"""
import numpy as np
import pandas as pd 
from matplotlib.patches import Polygon
import os
from dataloader import load_data, InputData
from visualization import Plotting_Class
from sensitivity_analysis import (run_capture_price_analysis,run_risk_sensitivity_analysis, run_bias_sensitivity_analysis, 
                                  run_capture_rate_sensitivity_analysis, run_production_sensitivity_analysis, 
                                  run_no_contract_boundary_analysis_price, run_negotiation_power_sensitivity_analysis,
                                  run_load_scenario_sensitivity_analysis,run_load_capture_rate_sensitivity_analysis,
                                  run_price_sensitivity_analysis ,run_load_generation_ratio_sensitivity_analysis)
from contract_negotiation import ContractNegotiation
from utils import ForecastProvider
import copy
import json

# Global variable definition (module level)
DROPBOX_DIR = r'C:\Users\ande7\Dropbox\Apps\Overleaf\Thesis - Nash Bargaining\Figures'
plots_folder = os.path.join(os.path.dirname(__file__), 'Plots')
results_folder = os.path.join(os.path.dirname(__file__), 'Results')


def run_contract_negotiation_sensitivity(input_data: InputData, 
                           A_G_values: np.ndarray, A_L_values: np.ndarray, Beta_L_values: np.ndarray,Beta_G_values: np.ndarray) -> tuple:
    """Run contract negotiation and sensitivity analysis with given OPF results."""
    
    # Store original data
    original_data = copy.deepcopy(input_data)
    
    # Run base case with middle values
    base_input = copy.deepcopy(original_data)
   

    new_base = base_input
    new_base.K_G = 0.0  # Use middle value for A_G
    new_base.K_L = 0.0  # Use middle value for A
    
    contract_model = ContractNegotiation(base_input)
    contract_model.run()
    base_results = copy.deepcopy(contract_model.results)
    base_data = copy.deepcopy(contract_model.data)
    
    print(2*"\nRunning contract negotiation...")

    # Run Fixed Strike price of capture price G 
    capture_price_input = copy.deepcopy(original_data)
    CP_sensitivity_results, CP_earnings_sensitivity = run_capture_price_analysis(capture_price_input)

    
    # Run Sensitivity Analyses with fresh copies
    risk_input = copy.deepcopy(original_data)
    risk_sensitivity_results, earnings_sensitivity = run_risk_sensitivity_analysis(risk_input, A_G_values, A_L_values)
    
    bias_input = copy.deepcopy(original_data)
    bias_sensitivity_results = run_bias_sensitivity_analysis(bias_input)

    # Run price sensitivity analysis with a fresh copy (1)
    price_input = copy.deepcopy(original_data)
    price_sensitivity_results = run_price_sensitivity_analysis(price_input)

    # Run load scenario sensitivity analysis with a fresh copy (2)
    load_scenario_input = copy.deepcopy(original_data)
    load_scenario_sensitivity_results = run_load_scenario_sensitivity_analysis(load_scenario_input)

    # Run capture rate sensitivity analysis with a fresh copy (3)
    capture_rate_input = copy.deepcopy(original_data)
    capture_rate_sensitivity_results = run_capture_rate_sensitivity_analysis(capture_rate_input)

    # Run load capture rate sensitivity analysis with a fresh copy (4)
    load_capture_rate_input = copy.deepcopy(original_data)
    load_capture_rate_sensitivity_results = run_load_capture_rate_sensitivity_analysis(load_capture_rate_input)

    # Run production sensitivity analysis with a fresh copy (5)
    # Note: This is not the same as the production scenarios, it is the production rate of wind farm 
    production_rate_input = copy.deepcopy(original_data)
    production_sensitivity_results = run_production_sensitivity_analysis( production_rate_input)

    # Run Boundary analysis with a fresh copy 
    if boundary == True:
        boundary_data_input = copy.deepcopy(original_data)
        boundary_results_df = run_no_contract_boundary_analysis_price(boundary_data_input)
    else:
        boundary_results_df = pd.DataFrame()

    # Run sensitivity analysis with different negotations powers
    beta_input = copy.deepcopy(original_data)
    beta_sensitivity_results, beta_earnings_sensitivity = run_negotiation_power_sensitivity_analysis(
        beta_input, 
        Beta_G_values, 
        Beta_L_values
    )

    # Sensitivity analysis for Load / Generation ratio
    load_generation_ratio_input = copy.deepcopy(original_data)
    load_generation_ratio_sensitivity_results = run_load_generation_ratio_sensitivity_analysis(load_generation_ratio_input)



    #alpha_input = copy.deepcopy(original_data)
    #alpha_sensitivity_results,alpha_earnings_sensitivity = run_cvar_alpha_sensitivity_analysis(alpha_input    )



     # Prepare results as a dictionary
    results = {
        "capture_price_results": CP_sensitivity_results,
        "capture_price_earnings_sensitivity": CP_earnings_sensitivity,
        "risk_sensitivity": risk_sensitivity_results,
        "risk_earnings_sensitivity": earnings_sensitivity,
        "bias_sensitivity": bias_sensitivity_results,
        'price_sensitivity': price_sensitivity_results,
        "production_sensitivity": production_sensitivity_results,
        "load_sensitivity": load_scenario_sensitivity_results,
        "gen_capture_rate_sensitivity": capture_rate_sensitivity_results,
        "load_capture_rate_sensitivity": load_capture_rate_sensitivity_results,
        "boundary_results": boundary_results_df,
        "negotiation_power_sensitivity": beta_sensitivity_results,
        "negotiation_earnings_sensitivity": beta_earnings_sensitivity,
        "load_ratio_sensitivity": load_generation_ratio_sensitivity_results,
    }

    return contract_model , results


def run_contract_negotiation(input_data: InputData): 
        # Store original data
        original_data = copy.deepcopy(input_data)
        
        # Run base case with middle values
        base_input = copy.deepcopy(original_data)
        base_input.K_G = 0.0  # Use middle value for A_G
        base_input.K_L = 0.0  # Use middle value for A
        
        contract_model = ContractNegotiation(base_input)
        contract_model.run()
        base_results = copy.deepcopy(contract_model.results)

        return base_results

def save_results_to_csv(results_dict, contract_type,time_horizon, num_scenarios):
    # Create folders if they don't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Save each dataframe in the results tuple
    result_names_not_risk = [
         
        "capture_price_results", "capture_price_earnings_sensitivity",  # Capture price results
        "bias_sensitivity", "price_sensitivity",   # Bias is difference between G and L in what they think price will be, price sensitivity it is uniform between
        "production_sensitivity",'load_sensitivity',"gen_capture_rate_sensitivity", "load_capture_rate_sensitivity",
        "negotiation_power_sensitivity", "negotiation_earnings_sensitivity", "load_ratio_sensitivity" # beta results are negotation results
        ]
    
    result_names_risk = ["risk_sensitivity", "risk_earnings_sensitivity",'boundary_results']
    
    for i, result_name in enumerate(result_names_not_risk):
        base_filename = f"{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}"
        
        data = results_dict[result_name]
        # Skip if the data is not a DataFrame
        if isinstance(data, pd.DataFrame): 
            csv_path = os.path.join(results_folder, f"{base_filename}.csv")
            data.to_csv(csv_path)
            print(f"Saved {result_name} to CSV file")
        else:
            print(f"Skipping {result_name}: not a DataFrame")

    
    for i, result_name in enumerate(result_names_risk):
        base_filename = f"{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}"
        # Handle boundary_results specially since it's a list of dictionaries
        data = results_dict[result_name]
        if result_name == "boundary_results" and boundary == True:
                json_path = os.path.join(results_folder, f"{base_filename}.json")

            # Save as JSON file
                # Convert numpy arrays to lists for JSON serialization
       # Convert numpy arrays to lists for JSON serialization
                save_data = []
                for item in data:
                    # For each scenario in the boundary results
                    item_copy = item.copy()
                    if 'boundary_points' in item_copy:
                        item_copy['boundary_points'] = [list(point) for point in item_copy['boundary_points']]
                    
                    # Convert contract_grid to list if present
                    if 'contract_grid' in item_copy:
                        item_copy['contract_grid'] = item_copy['contract_grid'].tolist()
                    
                    # Convert numpy arrays in KL_range and KG_range
                    if 'KL_range' in item_copy:
                        item_copy['KL_range'] = item_copy['KL_range'].tolist()
                    if 'KG_range' in item_copy:
                        item_copy['KG_range'] = item_copy['KG_range'].tolist()
                    
                    save_data.append(item_copy)
                
                with open(json_path, 'w') as f:
                    json.dump(save_data, f)
                print(f"Saved {result_name} to JSON file")
        
        
        # Skip if the data is not a DataFrame
        if isinstance(data, pd.DataFrame): 
            csv_path = os.path.join(results_folder, f"{base_filename}.csv")
            data.to_csv(csv_path)
            print(f"Saved {result_name} to CSV file")
        else:
            print(f"Skipping {result_name}: not a DataFrame")


 
def main():      # Define simulation parameters
    time_horizon = 20  # Must match the scenarios that were generated
    num_scenarios = 1000  # Must match the scenarios that were generated
    global A_L , A_G, boundary, sensitivity , Barter
    A_L = 0.5  # Initial risk aversion
    A_G = 0.5  # Initial risk aversion    
    Beta_L = 0.5  # Asymmetry of power between load generator [0,1]
    Beta_G = 1-Beta_L  # Asymmetry of power between generation provider [0,1] - 1-beta_L
    Barter = False  # Whether to relax the problem (Mc Cormick's relaxation)
    contract_type = "PAP" # Either "Baseload" or "PAP"
    sensitivity = True  # Whether to run sensitivity analysis
    num_sensitivity = 6 # Number of sensitivity analysis points for Beta_L and Beta_G ( and A_G and A_L)  
    boundary = True  # Whether to run boundary analysis ( it takes awhile to run, so set to False for quick tests)
    print("Loading data and preparing for simulation...")
    input_data = load_data(
        time_horizon=time_horizon,
        num_scenarios=num_scenarios,
        A_G=A_G,
        A_L=A_L,
        Beta_L=Beta_L,
        Beta_G=Beta_G,
        Barter=Barter,
        contract_type=contract_type
    )    # InputData object is now created in load_data()    # Define risk aversion parameters for both objective functions
    params = {
        'A_G_values': np.array([0.1,0.5,0.9]),  # A in [0,1]
        'A_L_values': np.array([0.1,0.5,0.9]),  # A in [0,1]
        'Beta_L': np.linspace(0,1,num_sensitivity),  # Asymmetry of power between load generator [0,1]
        'Beta_G': np.ones(num_sensitivity)- np.linspace(0,1,num_sensitivity),  # Asymmetry of power between generation provider [
    }

    # Load scenarios from CSV files
    scenario_pattern = f"{{type}}_scenarios_{time_horizon}y_{num_scenarios}s.csv"

    # Load price scenarios
    prices_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='price')}", index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)

    # Load production scenarios
    prod_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='production')}", index_col=0)
    prod_df.index = pd.to_datetime(prod_df.index)

    # Load capture rate scenarios
    CR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='capture_rate')}", index_col=0)
    CR_df.index = pd.to_datetime(CR_df.index)    # Load load scenarios
    load_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load')}", index_col=0)
    load_df.index = pd.to_datetime(load_df.index)

    LR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load_capture_rate')}", index_col=0)
    LR_df.index = pd.to_datetime(LR_df.index)

    provider = ForecastProvider(prices_df, prod_df,CR_df,load_df,LR_df, prob=1/prices_df.shape[1])
    
    # Load data from provider into input_data
    input_data.load_data_from_provider(provider)
    
    # Run contract negotiation for both objective functions with same OPF results
    
    print("\nRunning simulation with modified objective function ((1-A)*E + A*CVaR)...")
    if sensitivity == True:
        cm_model , results = run_contract_negotiation_sensitivity(
            copy.deepcopy(input_data),
            params['A_G_values'],
            params['A_L_values'],
            params['Beta_L'],
            params['Beta_G']   
        )
        save_results_to_csv(results, contract_type,time_horizon, num_scenarios)
    else:
        cm_model = run_contract_negotiation(
            copy.deepcopy(input_data),
        )

    
    # Create comparison plots
    print("\nGenerating comparison plots...")


if __name__ == "__main__":
    main()