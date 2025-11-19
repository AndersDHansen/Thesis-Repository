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
                                  run_price_sensitivity_analysis ,run_load_generation_ratio_sensitivity_analysis,
                                  run_negotiation_power_vs_risk_sensitivity_analysis, run_elasticity_vs_risk_sensitivity_analysis, run_bias_vs_risk_elasticity_sensitivity_analysis)
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
                                                     A_G_values: np.ndarray, A_L_values: np.ndarray, tau_L_values: np.ndarray, tau_G_values: np.ndarray,
                                                     selected_analyses: list | None = None) -> tuple:
    """Run selected sensitivity analyses and return only those results.

    If selected_analyses is None or empty and sensitivity is True, all standard analyses run.
    Supported keys:
        'capture_price','risk','price_bias','production_bias','price_mean','price_std',
        'production_mean','production_std','load_mean','load_std','gen_capture_rate','load_capture_rate',
    'boundary_price','boundary_production','negotiation','negotiation_vs_risk','elasticity_vs_risk','load_ratio'.
    """
    
    # Store original data
    original_data = copy.deepcopy(input_data)
    
    # Run base case with middle values
    base_input = copy.deepcopy(original_data)
   


    
    contract_model = ContractNegotiation(base_input)
    contract_model.run()
    base_results = copy.deepcopy(contract_model.results)
    base_data = copy.deepcopy(contract_model.data)
    
    print(2*"\nRunning contract negotiation...")

    # Determine which analyses to run
    run_all = selected_analyses is None or len(selected_analyses) == 0
    def sel(key: str) -> bool:
        return run_all or (key in selected_analyses)

    results = {}

    if sel('capture_price'):
        capture_price_input = copy.deepcopy(original_data)
        CP_sensitivity_results, CP_earnings_sensitivity = run_capture_price_analysis(capture_price_input)
        results["capture_price_results"] = CP_sensitivity_results
        results["capture_price_earnings_sensitivity"] = CP_earnings_sensitivity

    if sel('risk'):
        risk_input = copy.deepcopy(original_data)
        risk_sensitivity_results, earnings_sensitivity = run_risk_sensitivity_analysis(risk_input, A_G_values, A_L_values)
        results["risk_sensitivity"] = risk_sensitivity_results
        results["risk_earnings_sensitivity"] = earnings_sensitivity

    if sel('price_bias'):
        price_bias_input = copy.deepcopy(original_data)
        price_bias_sensitivity_results = run_price_bias_sensitivity_analysis(price_bias_input)
        results["price_bias_sensitivity"] = price_bias_sensitivity_results

    if sel('production_bias'):
        production_bias_input = copy.deepcopy(original_data)
        production_bias_sensitivity_results = run_production_bias_sensitivity_analysis(production_bias_input)
        results['production_bias_sensitivity'] = production_bias_sensitivity_results
     
    if sel('price_mean'):
        price_input_mean = copy.deepcopy(original_data)
        price_sensitivity_results_mean = run_price_sensitivity_analysis(price_input_mean, sensitivity_type="mean")
        results['price_sensitivity_mean'] = price_sensitivity_results_mean

    if sel('price_std'):
        price_input_std = copy.deepcopy(original_data)
        price_sensitivity_results_std = run_price_sensitivity_analysis(price_input_std, sensitivity_type="std")
        results['price_sensitivity_std'] = price_sensitivity_results_std

    if sel('load_mean'):
        load_scenario_input_mean = copy.deepcopy(original_data)
        load_scenario_sensitivity_results_mean = run_load_scenario_sensitivity_analysis(load_scenario_input_mean,  sensitivity_type="mean")
        results['load_sensitivity_mean'] = load_scenario_sensitivity_results_mean

    if sel('load_std'):
        load_scenario_input_std = copy.deepcopy(original_data)
        load_scenario_sensitivity_results_std = run_load_scenario_sensitivity_analysis(load_scenario_input_std,  sensitivity_type="std")
        results['load_sensitivity_std'] = load_scenario_sensitivity_results_std

    if sel('gen_capture_rate'):
        capture_rate_input = copy.deepcopy(original_data)
        capture_rate_sensitivity_results = run_capture_rate_sensitivity_analysis(capture_rate_input)
        results['gen_capture_rate_sensitivity'] = capture_rate_sensitivity_results

    if sel('load_capture_rate'):
        load_capture_rate_input = copy.deepcopy(original_data)
        load_capture_rate_sensitivity_results = run_load_capture_rate_sensitivity_analysis(load_capture_rate_input)
        results['load_capture_rate_sensitivity'] = load_capture_rate_sensitivity_results

    if sel('production_mean'):
        production_input_mean = copy.deepcopy(original_data)
        production_sensitivity_results_mean = run_production_sensitivity_analysis(production_input_mean, sensitivity_type="mean")
        results['production_sensitivity_mean'] = production_sensitivity_results_mean
  
    if sel('production_std'):
        production_input_std = copy.deepcopy(original_data)
        production_sensitivity_results_std = run_production_sensitivity_analysis(production_input_std, sensitivity_type="std")
        results['production_sensitivity_std'] = production_sensitivity_results_std
    
    if sel('boundary_production'):
        boundary_data_input_production = copy.deepcopy(original_data)
        boundary_results_df_production = run_no_contract_boundary_analysis_production(boundary_data_input_production)
        results['boundary_results_production'] = boundary_results_df_production

    
    if sel('boundary_price'):
        boundary_data_input_price = copy.deepcopy(original_data)
        boundary_results_df_price = run_no_contract_boundary_analysis_price(boundary_data_input_price)
        results['boundary_results_price'] = boundary_results_df_price
   
    if sel('negotiation'):
        tau_input = copy.deepcopy(original_data)
        tau_sensitivity_results, tau_earnings_sensitivity = run_negotiation_power_sensitivity_analysis(
            tau_input, 
            tau_G_values, 
            tau_L_values
        )
        results['negotiation_power_sensitivity'] = tau_sensitivity_results
        results['negotiation_earnings_sensitivity'] = tau_earnings_sensitivity
   
    # New: negotiation power vs multiple risk-aversion pairs
    if sel('negotiation_vs_risk'):
        nv_input = copy.deepcopy(original_data)
        sel_A_L_values = [0.0, 0.5, 0.9]  # Example values for A_L
        sel_A_G_values = [0.1, 0.5, 0.9]  # Example values for A_G
        negotiation_vs_risk_df = run_negotiation_power_vs_risk_sensitivity_analysis(
            nv_input,
            A_G_values=sel_A_G_values,
            A_L_values=sel_A_L_values,
            tau_L_values=tau_L_values,
        )
        results['negotiation_vs_risk_sensitivity'] = negotiation_vs_risk_df

    if sel('load_ratio'):
        load_generation_ratio_input = copy.deepcopy(original_data)
        load_generation_ratio_sensitivity_results = run_load_generation_ratio_sensitivity_analysis(load_generation_ratio_input)
        results['load_ratio_sensitivity'] = load_generation_ratio_sensitivity_results

    if sel('elasticity_vs_risk'):
        evr_input = copy.deepcopy(original_data)
        sel_A_L_values = [0.1, 0.5, 0.9]  # Example values for A_L
        sel_A_G_values = [0.1, 0.5, 0.9]  # Example values for A_G
        evr_df = run_elasticity_vs_risk_sensitivity_analysis(
            evr_input,
            A_G_values=sel_A_G_values,
            A_L_values=sel_A_L_values,
        )
        results['elasticity_vs_risk_sensitivity'] = evr_df

    if sel('bias_vs_risk_elasticity'):
        bre_input = copy.deepcopy(original_data)
        sel_A_L_values = [0.1, 0.5, 0.9]  # Example values for A_L
        sel_A_G_values = [0.1, 0.5, 0.9]  # Example values for A_G
        bre_df = run_bias_vs_risk_elasticity_sensitivity_analysis(
            bre_input,
            A_G_values=sel_A_G_values,
            A_L_values=sel_A_L_values,
        )
        results['bias_vs_risk_elasticity_sensitivity'] = bre_df

    return contract_model , results


def run_contract_negotiation(input_data: InputData): 
        # Store original data
        original_data = copy.deepcopy(input_data)
        
        # Run base case with middle values
        base_input = copy.deepcopy(original_data)

        start_time = timeit.default_timer()

        new_base = base_input
        new_base.K_G_price = 0  # Use middle value for A_G
        new_base.K_L_price = 0  # Use middle value for A_L

        contract_model = ContractNegotiation(new_base)
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
    
    # Save dynamically only what was produced
    risk_like_keys = {'negotiation_vs_risk_sensitivity', 'elasticity_vs_risk_sensitivity',"risk_sensitivity","risk_earnings_sensitivity","boundary_price","bias_vs_risk_elasticity_sensitivity"}

    for result_name, data in results_dict.items():
        # Build base filename
        if result_name in risk_like_keys:
            base_filename =  f"monte_{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}" if monte_price else f"{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}"
        else:
            base_filename =  f"monte_{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}" if monte_price else f"{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}"

        # Boundary results -> JSON
        if result_name in ("boundary_results_price", "boundary_results_production") and isinstance(data, (list, tuple)):
            json_path = os.path.join(results_folder, f"{base_filename}.json")
            save_data = []
            for item in data:
                item_copy = item.copy()
                if 'boundary_points' in item_copy:
                    item_copy['boundary_points'] = [list(point) for point in item_copy['boundary_points']]
                if 'contract_grid' in item_copy:
                    item_copy['contract_grid'] = np.array(item_copy['contract_grid']).tolist()
                if 'feas_mask' in item_copy:
                    item_copy['feas_mask'] = np.array(item_copy['feas_mask']).tolist()
                if 'pos_contract_mask' in item_copy:
                    item_copy['pos_contract_mask'] = np.array(item_copy['pos_contract_mask']).tolist()
                if 'KL_grid' in item_copy:
                    item_copy['KL_grid'] = np.array(item_copy['KL_grid']).tolist()
                if 'KG_grid' in item_copy:
                    item_copy['KG_grid'] = np.array(item_copy['KG_grid']).tolist()
                if 'KL_range' in item_copy:
                    item_copy['KL_range'] = np.array(item_copy['KL_range']).tolist()
                if 'KG_range' in item_copy:
                    item_copy['KG_range'] = np.array(item_copy['KG_range']).tolist()
                save_data.append(item_copy)
            with open(json_path, 'w') as f:
                json.dump(save_data, f)
            print(f"Saved {result_name} to JSON file")
            continue

        # DataFrames -> CSV
        if isinstance(data, pd.DataFrame):
            csv_path = os.path.join(results_folder, f"{base_filename}.csv")
            data.to_csv(csv_path)
            print(f"Saved {result_name} to CSV file")
        else:
            # Skip non-DF, non-boundary entries
            pass

 
def main():      # Define simulation parameters

    global A_L , A_G, d_G,d_L, boundary, sensitivity , Barter, scenario_time_horizon, opt_time_horizon, monte_price
    A_L = 0.5  # Initial risk aversion
    A_G = 0.5 # Initial risk aversion
    scenario_time_horizon = 20  # Must match the scenarios that were generated
    opt_time_horizon = 20  # Time horizon for optimization (in years)
    num_scenarios = 500  # Must match the scenarios that were generated
    d_G = 0.00  # Generator discount rate
    d_L = 0.00  # Load generator discount rate

    # Bool Statements
    monte_price = False  # Monte carlo price scenarios
    Barter = True  # Whether to relax the problem (Mc Cormick's relaxation)
    boundary = False  # Deprecated flag (use selected_analyses to include boundary_* if desired)
    sensitivity = False  # Whether to run sensitivity analyses at all
    Discount = True  # Whether to include discounting in the objective function



    tau_L = 1  # Asymmetry of power between load generator [0,1]
    tau_G = 1-tau_L  # Asymmetry of power between generation provider [0,1] - 1-tau_L
    contract_type = "Baseload" # Either "Baseload" or "PAP"
    # Choose which analyses to run; leave empty to run all when sensitivity=True
    #selected_analyses: list[str] = ["capture_price", "negotiation_vs_risk","risk","price_bias","production_bias",]
    selected_analyses: list[str] = ['elasticity_vs_risk']

    num_sensitivity = 5 # Number of sensitivity analysis points for tau_L and tau_G ( and A_G and A_L)  
    # Boundary analysis only on 20 years
    print("Loading data and preparing for simulation...")
    input_data = load_data(
        opt_time_horizon=opt_time_horizon,
        num_scenarios=num_scenarios,
        A_G=A_G,
        A_L=A_L,
        tau_L=tau_L,
        tau_G=tau_G,
        d_G=d_G,
        d_L=d_L,
        Barter=Barter,
        Discount=Discount,
        contract_type=contract_type
    )    # InputData object is now created in load_data()    # Define risk aversion parameters for both objective functions
    params = {
        'A_G_values': np.array([0,0.1,0.25,0.5,0.75,0.9,1]),  # A in [0,1]
        'A_L_values':np.array([0,0.1,0.25,0.5,0.75,0.9,1]),  # A in [0,1]
        'tau_L': np.linspace(0,1,num_sensitivity),  # Asymmetry of power between load generator [0,1]
        'tau_G': np.ones(num_sensitivity)- np.linspace(0,1,num_sensitivity),  # Asymmetry of power between generation provider [
    }

    # Load scenarios from CSV files
    scenario_pattern_reduced = f"{{type}}_scenarios_reduced_{scenario_time_horizon}y_{num_scenarios}s.csv"
    scenario_pattern_reduced_monte = f"{{type}}_scenarios_monte_{scenario_time_horizon}y_{num_scenarios}s.csv"


    #scenario_pattern = f"{{type}}_scenarios_{time_horizon}y_{num_scenarios}s.csv"

    prod_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='production')}", index_col=0)
    prod_df.index = pd.to_datetime(prod_df.index)
    # Load price scenarios
    if monte_price == True:
        prices_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced_monte.format(type='price')}", index_col=0)
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df.columns = prod_df.columns
        prob_df = np.ones(prices_df.shape[1]) / prices_df.shape[1]


        CR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced_monte.format(type='capture_rate')}", index_col=0)
        CR_df.index = pd.to_datetime(CR_df.index)

        # Load load scenarios
        load_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced_monte.format(type='load')}", index_col=0)
        load_df.index = pd.to_datetime(load_df.index)

        LR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced_monte.format(type='load_capture_rate')}", index_col=0)
        LR_df.index = pd.to_datetime(LR_df.index)

    else:
        prices_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='price')}", index_col=0)
        prices_df.index = pd.to_datetime(prices_df.index)
        prob_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='probabilities')}", index_col=0)
        prob_df = prob_df.values.flatten()
        #prob_df = np.ones(prices_df.shape[1]) / prices_df.shape[1]


        CR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='capture_rate')}", index_col=0)
        CR_df.index = pd.to_datetime(CR_df.index)

        # Load load scenarios
        load_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='load')}", index_col=0)
        load_df.index = pd.to_datetime(load_df.index)

        LR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced.format(type='load_capture_rate')}", index_col=0)
        LR_df.index = pd.to_datetime(LR_df.index)


    # Load production scenarios
 

    # Load capture rate scenarios
 
    
    #prob_df = np.ones(num_scenarios) / num_scenarios  # Uniform probabilities remove laster
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
            params['tau_G'],
            selected_analyses=selected_analyses
        )
        save_results_to_csv(results, contract_type,opt_time_horizon, num_scenarios)
    else:
        cm_model = run_contract_negotiation(
            copy.deepcopy(input_data),
        )

    
    # Create comparison plots
    print("\nGenerating comparison plots...")


if __name__ == "__main__":
    main()