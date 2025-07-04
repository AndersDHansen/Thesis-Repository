"""
Run this script to visualize the results without re-runnign the entire code 
Using saved Results from the previous runs
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import Plotting_Class
from contract_negotiation import ContractNegotiation
from dataloader import load_data, InputData
from utils import ForecastProvider
import json


DROPBOX_DIR = r'C:\Users\ande7\Dropbox\Apps\Overleaf\Thesis - Nash Bargaining\Figures'
plots_folder = os.path.join(os.path.dirname(__file__), 'Plots')
csv_folder = os.path.join(os.path.dirname(__file__), 'Results')



def load_sensitivity_results(contract_type,time_horizon, num_scenarios):
    """
    Load sensitivity analysis results from saved CSV files.
    
    Parameters:
    -----------
    contract_type : str
        Type of contract ("PAP" or "Baseload")
    num_scenarios : int
        Number of scenarios used in the simulation
    
    Returns:
    --------
    tuple
        Tuple containing all loaded dataframes
    """
   
    # Path to CSV files
    
    # Define filenames to load
    result_names = [
        "risk_sensitivity", "earnings_sensitivity", 
        "bias_sensitivity", "capture_rate_sensitivity", 
        "production_sensitivity",
        "negotiation_power_sensitivity", "negotiation_earnings_sensitivity",
        "cvar_alpha_sensitivity", "cvar_alpha_earnings_sensitivity"
    ]
    
    results = []
    
    # Load each CSV file
    for result_name in result_names:
        csv_filename = f"{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        file_path = os.path.join(csv_folder, csv_filename)
        
        df = pd.read_csv(file_path, index_col=0)
        results.append(df)
            
        print(f"Loaded {result_name} from {csv_filename}")
     
    return results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8]

def load_boundary_results(contract_type,time_horizon, num_scenarios):
    """
    Load boundary results from saved JSON file.
    
    Parameters:
    -----------
    contract_type : str
        Type of contract ("PAP" or "Baseload")
    num_scenarios : int
        Number of scenarios used in the simulation
    
    Returns:
    --------
    list
        List of dictionaries containing boundary results
    """
    # Path to JSON file
    json_filename = f"boundary_results_{contract_type}_{time_horizon}y_{num_scenarios}.json"
    json_path = os.path.join(os.path.dirname(__file__), 'Results', json_filename)
    
    try:
        with open(json_path, 'r') as f:
            boundary_data = json.load(f)
        
        # Convert lists back to numpy arrays
        for item in boundary_data:
            if 'boundary_points' in item:
                item['boundary_points'] = np.array(item['boundary_points'])
            if 'contract_grid' in item:
                item['contract_grid'] = np.array(item['contract_grid'])
            if 'KL_range' in item:
                item['KL_range'] = np.array(item['KL_range'])
            if 'KG_range' in item:
                item['KG_range'] = np.array(item['KG_range'])
        
        print(f"Loaded boundary results from {json_filename}")
        return boundary_data
    except FileNotFoundError:
        print(f"Warning: Could not find {json_filename}")
        return []

def main():
    # Configuration for loading data 
    time_horizon = 5  # Must match the scenarios that were generated
    num_scenarios = 1000  # Must match the scenarios that were generated
    A_L = 0.5  # Initial risk aversion
    A_G = 0.5  # Initial risk aversion
    Beta_L = 0.5  # Asymmetry of power between load generator [0,1]
    Beta_G = 1-Beta_L  # Asymmetry of power between generation provider [0,1] - 1-beta_L    
    Barter = False  # Whether to relax the problem (Mc Cormick's relaxation)
    contract_type = "PAP" # Either "Baseload" or "PAP"

    # Load data and create InputData object 
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
        'A_L_values': np.array([0.1,0.5,0.9])  # A in [0,1]
    }
    fixed_A_G=params['A_G_values'][1]
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
    # Load load scenarios
    load_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load')}", index_col=0)
    load_df.index = pd.to_datetime(load_df.index)
    #load Load Capture Rate scenarios
    LR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load_capture_rate')}", index_col=0)
    LR_df.index = pd.to_datetime(LR_df.index)

    provider = ForecastProvider(prices_df, prod_df,CR_df,load_df,LR_df, prob=1/prices_df.shape[1])
    
    # Load data from provider into input_data
    input_data.load_data_from_provider(provider)


    # Load sensitivity results from CSV
    risk_sensitivity, earnings_sensitivity, bias_sensitivity, \
    capture_rate_sensitivity, production_sensitivity , negotiation_sensitvity, negotiation_earnings, \
    alpha_sensitivity , alpha_earnings = \
    load_sensitivity_results(contract_type,time_horizon, num_scenarios)
    
    # Load boundary results from JSON
    boundary_results = load_boundary_results(contract_type,time_horizon, num_scenarios)
    
    # Create contract model data instance (simplified)
    contract_model = ContractNegotiation(input_data)
    contract_model.run()
    
    # Create plotting class instance
    plotter = Plotting_Class(
        contract_model_data=contract_model.data,
        risk_sensitivity_df=risk_sensitivity,
        sensitivity_earnings_df=earnings_sensitivity,
        bias_sensitivity_df=bias_sensitivity,
        capture_rate_sensitivity_df=capture_rate_sensitivity,
        production_sensitivity_df=production_sensitivity,
        boundary_results_df=boundary_results,
        negotiation_sensitivity_df=negotiation_sensitvity,
        negotiation_earnings_df=negotiation_earnings,
        alpha_sensitivity_df=alpha_sensitivity,
        alpha_earnings_df=alpha_earnings,

    )
    
    # Generate plots
    print("\nGenerating plots...")

    # Generate filenames
    risk_file = f"risk_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    earnings_file = f"earnings_distribution_AG={fixed_A_G}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    bias_file = f"bias_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    capture_file = f"capture_rate_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    production_file = f"production_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    boundary_file = f"no_contract_boundary_{contract_type}_{time_horizon}_{num_scenarios}.png"
    boundary_file_all = f"no_contract_boundary_all_{contract_type}_{time_horizon}_{num_scenarios}.png"
    negotiation_sensitivity_file = f"negotiation_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    negotation_earnings_file = f"negotiation_earnings_{contract_type}_{time_horizon}_{num_scenarios}.png"
    alpha_sensitivity_file = f"alpha_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    alpha_earnings_file = f"alpha_earnings_{contract_type}_{time_horizon}_{num_scenarios}.png"
    earnings_boxplot_file = f"earnings_boxplot_AG={fixed_A_G}_{contract_type}_{time_horizon}_{num_scenarios}.png"

        # Boxplot of earnings
    plotter._risk_plot_earnings_boxplot( fixed_A_G,  # Use middle value
        A_L_to_plot=params['A_L_values'].tolist(),filename=os.path.join(plots_folder, earnings_boxplot_file))
    


    plotter._nego_plot_earnings_boxplot(filename=os.path.join(plots_folder, negotation_earnings_file))
    plotter._plot_parameter_sensitivity_spider()
    plotter._plot_elasticity_tornado(metric='Utility_G')





    # Risk sensitivity plots - save to both locations
    plotter._plot_sensitivity_results_heatmap('risk',filename=os.path.join(plots_folder, risk_file))
    plotter._plot_sensitivity_results_heatmap('risk',filename=os.path.join(DROPBOX_DIR, risk_file))
            
    # Earnings distribution plots - save to both locations
    plotter._plot_earnings_histograms(
        fixed_A_G,  # Use middle value
        A_L_to_plot=params['A_L_values'].tolist(),
        filename=os.path.join(plots_folder, earnings_file)
    )
    plotter._plot_earnings_histograms(
        fixed_A_G,
        A_L_to_plot=params['A_L_values'].tolist(),
        filename=os.path.join(DROPBOX_DIR, earnings_file)
    )
    
    #Radar Chart 

    # Bias sensitivity plots - save to both locations
    plotter._plot_sensitivity_results_heatmap('bias',filename=os.path.join(plots_folder, bias_file))
    plotter._plot_sensitivity_results_heatmap('bias',filename=os.path.join(DROPBOX_DIR, bias_file))
            
    # Plot capture rate sensitivity plots - save to both locations
    plotter._plot_capture_rate_sensitivity(filename=os.path.join(plots_folder, capture_file))
    plotter._plot_capture_rate_sensitivity(filename=os.path.join(DROPBOX_DIR, capture_file))
    plotter._plot_production_sensitivity(filename=os.path.join(plots_folder, production_file))
    plotter._plot_production_sensitivity(filename=os.path.join(DROPBOX_DIR, production_file))

    # Plot negotiation sensitivity - save to both locations
    plotter._plot_sensitivity_results_line('negotiation',filename=os.path.join(plots_folder, negotiation_sensitivity_file))
    plotter._plot_sensitivity_results_line('negotiation',filename=os.path.join(DROPBOX_DIR, negotiation_sensitivity_file))


    # Plot alpha sensitivity - save to both locations
    #plotter._plot_sensitivity_results_line('alpha',filename=os.path.join(plots_folder, alpha_sensitivity_file))
    #plotter._plot_sensitivity_results_line('alpha',filename=os.path.join(DROPBOX_DIR, alpha_sensitivity_file))

    # Plot alpha earnings - save to both locations
    #plotter._plot_earnings_histograms_alpha(filename=os.path.join(plots_folder, alpha_earnings_file))
    #plotter._plot_earnings_histograms_alpha(filename=os.path.join(DROPBOX_DIR, alpha_earnings_file))
     
      
    if contract_type == "Baseload":
        plotter._plot_no_contract_boundaries(filename=os.path.join(plots_folder, boundary_file))
        plotter._plot_no_contract_boundaries(filename=os.path.join(DROPBOX_DIR, boundary_file))
        plotter._plot_no_contract_boundaries_all(filename=os.path.join(plots_folder, boundary_file_all))
        plotter._plot_no_contract_boundaries_all(filename=os.path.join(DROPBOX_DIR, boundary_file_all)) 
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()