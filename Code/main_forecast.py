"""
Main execution script for power system contract negotiation analysis.
"""
import numpy as np
import pandas as pd 
from matplotlib.patches import Polygon
import os
from dataloader import load_data, InputData
from visualization import Plotting_Class
from sensitivity_analysis import (run_capture_price_analysis, run_no_contract_boundary_analysis_production, run_price_bias_sensitivity_analysis,run_risk_sensitivity_analysis, run_production_bias_sensitivity_analysis, 
                                  run_capture_rate_sensitivity_analysis, run_production_sensitivity_analysis, 
                                  run_no_contract_boundary_analysis_price, run_negotiation_power_sensitivity_analysis,
                                  run_load_scenario_sensitivity_analysis,run_load_capture_rate_sensitivity_analysis,
                                  run_price_sensitivity_analysis ,run_load_generation_ratio_sensitivity_analysis)
from contract_negotiation import ContractNegotiation
from utils import ForecastProvider
import copy
import json
import timeit


# Global variable definition (module level)
DROPBOX_DIR = r'C:\Users\ande7\Dropbox\Apps\Overleaf\Thesis - Nash Bargaining\Figures'
plots_folder = os.path.join(os.path.dirname(__file__), 'Plots')
results_folder = os.path.join(os.path.dirname(__file__), 'Results')


def run_contract_negotiation_sensitivity(input_data: InputData, 
                           A_G_values: np.ndarray, A_L_values: np.ndarray, tau_L_values: np.ndarray, tau_G_values: np.ndarray) -> tuple:
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
    
    # Run Price Bias sensitivity analysis with a fresh copy (0)
    price_bias_input = copy.deepcopy(original_data)
    price_bias_sensitivity_results = run_price_bias_sensitivity_analysis(price_bias_input)

    #Run Production Bias sensitivity analysis with a fresh copy (0)
    production_bias_input = copy.deepcopy(original_data)
    production_bias_sensitivity_results = run_production_bias_sensitivity_analysis(production_bias_input)

    # Run mean price sensitivity analysis with a fresh copy (1)
    price_input_mean = copy.deepcopy(original_data)
    price_sensitivity_results_mean = run_price_sensitivity_analysis(price_input_mean, sensitivity_type="mean")

    # Run standard deviation price sensitivity analysis with a fresh copy (1)
    price_input_std = copy.deepcopy(original_data)
    price_sensitivity_results_std = run_price_sensitivity_analysis(price_input_std, sensitivity_type="std")

    # Run mean load scenario sensitivity analysis with a fresh copy (2)
    load_scenario_input_mean = copy.deepcopy(original_data)
    load_scenario_sensitivity_results_mean = run_load_scenario_sensitivity_analysis(load_scenario_input_mean,  sensitivity_type="mean")

    # Run  standard deviation scenario sensitivity analysis with a fresh copy (2)
    load_scenario_input_std = copy.deepcopy(original_data)
    load_scenario_sensitivity_results_std = run_load_scenario_sensitivity_analysis(load_scenario_input_std,  sensitivity_type="std")

    # Run capture rate sensitivity analysis with a fresh copy (3)
    capture_rate_input = copy.deepcopy(original_data)
    capture_rate_sensitivity_results = run_capture_rate_sensitivity_analysis(capture_rate_input)

    # Run load capture rate sensitivity analysis with a fresh copy (4)
    load_capture_rate_input = copy.deepcopy(original_data)
    load_capture_rate_sensitivity_results = run_load_capture_rate_sensitivity_analysis(load_capture_rate_input)

    # Run mean production sensitivity analysis with a fresh copy (5)
    production_input_mean = copy.deepcopy(original_data)
    production_sensitivity_results_mean = run_production_sensitivity_analysis(production_input_mean, sensitivity_type="mean")

    # Run standard deviation production sensitivity analysis with a fresh copy (5)
    production_input_std = copy.deepcopy(original_data)
    production_sensitivity_results_std = run_production_sensitivity_analysis(production_input_std, sensitivity_type="std")

    # Run Boundary analysis with a fresh copy 
    if boundary == True:
        boundary_data_input_price = copy.deepcopy(original_data)
        boundary_results_df_price = run_no_contract_boundary_analysis_price(boundary_data_input_price)

        # Production boundary analysis
        boundary_data_input_production = copy.deepcopy(original_data)
        boundary_results_df_production = run_no_contract_boundary_analysis_production(boundary_data_input_production)
    else:
        boundary_results_df_price = pd.DataFrame()
        boundary_results_df_production = pd.DataFrame()

    # Run sensitivity analysis with different negotations powers
    tau_input = copy.deepcopy(original_data)
    tau_sensitivity_results, tau_earnings_sensitivity = run_negotiation_power_sensitivity_analysis(
        tau_input, 
        tau_G_values, 
        tau_L_values
    )

    # Sensitivity analysis for Load / Generation ratio
    load_generation_ratio_input = copy.deepcopy(original_data)
    #load_generation_ratio_sensitivity_results = run_load_generation_ratio_sensitivity_analysis(load_generation_ratio_input)
    load_generation_ratio_sensitivity_results = pd.DataFrame()  # Result is empty for load generation ratio sensitivity analysis so we create an empty DataFrame



    #alpha_input = copy.deepcopy(original_data)
    #alpha_sensitivity_results,alpha_earnings_sensitivity = run_cvar_alpha_sensitivity_analysis(alpha_input    )



     # Prepare results as a dictionary
    results = {
        "capture_price_results": CP_sensitivity_results,
        "capture_price_earnings_sensitivity": CP_earnings_sensitivity,
        "risk_sensitivity": risk_sensitivity_results,
        "risk_earnings_sensitivity": earnings_sensitivity,
        "price_bias_sensitivity": price_bias_sensitivity_results,
        'price_sensitivity_mean': price_sensitivity_results_mean,
        'price_sensitivity_std': price_sensitivity_results_std,
        'production_bias_sensitivity': production_bias_sensitivity_results,
        "production_sensitivity_mean": production_sensitivity_results_mean,
        "production_sensitivity_std": production_sensitivity_results_std,
        "load_sensitivity_mean": load_scenario_sensitivity_results_mean,
        "load_sensitivity_std": load_scenario_sensitivity_results_std,
        "gen_capture_rate_sensitivity": capture_rate_sensitivity_results,
        "load_capture_rate_sensitivity": load_capture_rate_sensitivity_results,
        "boundary_results_price": boundary_results_df_price,
        "boundary_results_production": boundary_results_df_production,
        "negotiation_power_sensitivity": tau_sensitivity_results,
        "negotiation_earnings_sensitivity": tau_earnings_sensitivity,
        "load_ratio_sensitivity": load_generation_ratio_sensitivity_results,
    }

    return contract_model , results


def run_contract_negotiation(input_data: InputData): 
        # Store original data
        original_data = copy.deepcopy(input_data)
        
        # Run base case with middle values
        base_input = copy.deepcopy(original_data)

        start_time = timeit.default_timer()
        contract_model = ContractNegotiation(base_input)
        contract_model.run()
        end_time = timeit.default_timer()
        print(f"Contract negotiation completed in {end_time - start_time:.2f} seconds.")
        base_results = copy.deepcopy(contract_model.results)

        print(" Running Capture Price ...")
        capture_price_input = copy.deepcopy(original_data)
        #CP_sensitivity_results, CP_earnings_sensitivity = run_capture_price_analysis(capture_price_input)


        return base_results

def save_results_to_csv(results_dict, contract_type,time_horizon, num_scenarios):
    # Create folders if they don't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Save each dataframe in the results tuple
    result_names_not_risk = [
         
        "capture_price_results", "capture_price_earnings_sensitivity",  # Capture price results
        "price_bias_sensitivity","production_bias_sensitivity",  # Price and production bias sensitivity results
        "price_sensitivity_mean","price_sensitivity_std",   # Bias is difference between G and L in what they think price will be, price sensitivity it is uniform between
        "production_sensitivity_mean", "production_sensitivity_std",  # Production sensitivity
        "load_sensitivity_mean", "load_sensitivity_std",  # Load sensitivity
        "gen_capture_rate_sensitivity", "load_capture_rate_sensitivity",
        "negotiation_power_sensitivity", "negotiation_earnings_sensitivity", "load_ratio_sensitivity" # tau results are negotation results
        ]
    
    result_names_risk = ["risk_sensitivity", "risk_earnings_sensitivity",'boundary_results_price', 'boundary_results_production']
    
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
        if result_name == ("boundary_results_price" or "boundary_results_production") and boundary == True:
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
    num_scenarios = 5000  # Must match the scenarios that were generated
    global A_L , A_G, boundary, sensitivity , Barter
    A_L = 0  # Initial risk aversion
    A_G = 0  # Initial risk aversion

    tau_L = 0.5  # Asymmetry of power between load generator [0,1]
    tau_G = 1-tau_L  # Asymmetry of power between generation provider [0,1] - 1-tau_L
    Barter = False  # Whether to relax the problem (Mc Cormick's relaxation)
    contract_type = "Baseload" # Either "Baseload" or "PAP"
    sensitivity = True  # Whether to run sensitivity analysis
    num_sensitivity = 5 # Number of sensitivity analysis points for tau_L and tau_G ( and A_G and A_L)  
    # Boundary analysis only on 20 years
    boundary = True  # Whether to run boundary analysis ( it takes awhile to run, so set to False for quick tests)
    print("Loading data and preparing for simulation...")
    input_data = load_data(
        time_horizon=time_horizon,
        num_scenarios=num_scenarios,
        A_G=A_G,
        A_L=A_L,
        tau_L=tau_L,
        tau_G=tau_G,
        Barter=Barter,
        contract_type=contract_type
    )    # InputData object is now created in load_data()    # Define risk aversion parameters for both objective functions
    params = {
        'A_G_values': np.array([0,0.1,0.25,0.5,0.75,0.9,1]),  # A in [0,1]
        'A_L_values':np.array([0,0.1,0.25,0.5,0.75,0.9,1]),  # A in [0,1]
        'tau_L': np.linspace(0,1,num_sensitivity),  # Asymmetry of power between load generator [0,1]
        'tau_G': np.ones(num_sensitivity)- np.linspace(0,1,num_sensitivity),  # Asymmetry of power between generation provider [
    }

    # Load scenarios from CSV files
    scenario_pattern_reduced = f"{{type}}_scenarios_reduced_{time_horizon}y_{num_scenarios}s.csv"
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

    prob_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='probabilities')}", index_col=0)
    prob_df = prob_df.values.flatten()
    prob_df = np.ones(num_scenarios) / num_scenarios  # Uniform probabilities remove laster
    provider = ForecastProvider(prices_df, prod_df, CR_df, load_df, LR_df, prob=prob_df)
    # Load data from provider into input_data
    input_data.load_data_from_provider(provider)
    
    # Run contract negotiation for both objective functions with same OPF results
    
    print("\nRunning simulation with modified objective function ((1-A)*E + A*CVaR)...")
    if sensitivity == True:
        cm_model , results = run_contract_negotiation_sensitivity(
            copy.deepcopy(input_data),
            params['A_G_values'],
            params['A_L_values'],
            params['tau_L'],
            params['tau_G']
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