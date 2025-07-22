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
    time_horizon : int
        Time horizon in years for the contract
    num_scenarios : int
        Number of scenarios used in the simulation
    
    Returns:
    --------
    dict
        dict containing all loaded dataframes
    """
   
    # Path to CSV files
    
    # Define filenames to load



    result_names_not_risk = [
        "capture_price_results", "capture_price_earnings_sensitivity",  # Capture price results
        "bias_sensitivity", "price_sensitivity",   # Bias is difference between G and L in what they think price will be, price sensitivity it is uniform between
        "production_sensitivity",'load_sensitivity',"gen_capture_rate_sensitivity", "load_capture_rate_sensitivity",
        "negotiation_power_sensitivity", "negotiation_earnings_sensitivity", "load_ratio_sensitivity" # beta results are negotation results
        ]
    
    result_names_risk = ["risk_sensitivity", "risk_earnings_sensitivity"]

    results = {}

    # Load each CSV file for risk results
    for result_name in result_names_risk:
        csv_filename = f"{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        file_path = os.path.join(csv_folder, csv_filename)
        df = pd.read_csv(file_path, index_col=0)
        results[result_name] = df
        print(f"Loaded {result_name} from {csv_filename}")

    # Load each CSV file for no-risk results
    for result_name in result_names_not_risk:
        csv_filename = f"{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        file_path = os.path.join(csv_folder, csv_filename)
        df = pd.read_csv(file_path, index_col=0)
        results[result_name] = df
        print(f"Loaded {result_name} from {csv_filename}")
            
        print(f"Loaded {result_name} from {csv_filename}")
     
    return results

def load_boundary_results(contract_type,time_horizon, num_scenarios):
    """
    Load boundary results from saved JSON file.
    
    Parameters:
    -----------
    contract_type : str
        Type of contract ("PAP" or "Baseload")
    time_horizon : int
        Time horizon in years for the contract
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
    global time_horizon, num_scenarios, A_G, A_L, Beta_L, Beta_G, Barter, contract_type
    # Configuration for loading data 
    time_horizon = 20  # Must match the scenarios that were generated
    num_scenarios = 1000  # Must match the scenarios that were generated
    A_L = 0.5  # Initial risk aversion
    A_G = 0.5  # Initial risk aversion
    Beta_L = 0.5  # Asymmetry of power between load generator [0,1]
    Beta_G = 1-Beta_L  # Asymmetry of power between generation provider [0,1] - 1-beta_L    
    Barter = False  # Whether to relax the problem (Mc Cormick's relaxation)
    contract_type = "Baseload" # Either "Baseload" or "PAP"

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

    sensitivity_results = load_sensitivity_results(contract_type,time_horizon, num_scenarios)
    
    # Load boundary results from JSON
    boundary_results = load_boundary_results(contract_type,time_horizon, num_scenarios)
    
    # Create contract model data instance (simplified)
    contract_model = ContractNegotiation(input_data)
    #contract_model.run()
    
    # Create plotting class instance
    plotter = Plotting_Class(
        contract_model_data=contract_model.data,
        CP_results_df=sensitivity_results['capture_price_results'],
        CP_earnings_df=sensitivity_results['capture_price_earnings_sensitivity'],
        risk_sensitivity_df=sensitivity_results['risk_sensitivity'],
        risk_earnings_df=sensitivity_results['risk_earnings_sensitivity'],
        bias_sensitivity_df=sensitivity_results['bias_sensitivity'],
        price_sensitivity_df=sensitivity_results['price_sensitivity'],
        production_sensitivity_df=sensitivity_results['production_sensitivity'],
        load_sensitivity_df =sensitivity_results['load_sensitivity'],
        gen_CR_sensitivity_df=sensitivity_results['gen_capture_rate_sensitivity'],
        load_CR_sensitivity_df=sensitivity_results['load_capture_rate_sensitivity'],
        boundary_results_df=boundary_results,
        negotiation_sensitivity_df=sensitivity_results['negotiation_power_sensitivity'],
        negotiation_earnings_df=sensitivity_results['negotiation_earnings_sensitivity'],
        load_ratio_df = sensitivity_results['load_ratio_sensitivity'],


    )
    
    # Generate plots
    print("\nGenerating plots...")

    # Generate filenames
    risk_file = f"risk_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    earnings_file = f"earnings_distribution_AG={fixed_A_G}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    bias_file = f"bias_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{time_horizon}_{num_scenarios}.png"
    price_file = f"price_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{time_horizon}_{num_scenarios}.png"
    production_file = f"production_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    load_file = f"load_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    prod_CR_file = f"prod_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{time_horizon}_{num_scenarios}.png"
    load_CR_file = f"load_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{time_horizon}_{num_scenarios}.png"
    boundary_file = f"no_contract_boundary_{contract_type}_{time_horizon}_{num_scenarios}.png"
    boundary_file_all = f"no_contract_boundary_all_{contract_type}_{time_horizon}_{num_scenarios}.png"
    negotiation_sensitivity_file = f"negotiation_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    negotation_earnings_file = f"negotiation_earnings_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    load_ratio_file = f"load_ratio_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    #alpha_sensitivity_file = f"alpha_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
    #alpha_earnings_file = f"alpha_earnings_{contract_type}_{time_horizon}_{num_scenarios}.png"
    earnings_boxplot_file = f"earnings_boxplot_AG={fixed_A_G}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    spider_file = f"parameter_sensitivity_spider_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}_{num_scenarios}.png"
    tornando_file = f"elasticity_tornado_AG={fixed_A_G}_{contract_type}_{time_horizon}_{num_scenarios}.png"

    
    
    # Generate plots 
        # Boxplot of earnings
    plotter._risk_plot_earnings_boxplot( fixed_A_G,  # Use middle value
        A_L_to_plot=params['A_L_values'].tolist(),filename=os.path.join(plots_folder, earnings_boxplot_file))
    plotter._plot_elasticity_tornado(['StrikePrice','ContractAmount',],filename=os.path.join(plots_folder, tornando_file))

    plotter._nego_plot_earnings_boxplot(filename=os.path.join(plots_folder, negotation_earnings_file))
    plotter._plot_parameter_sensitivity_spider(filename=os.path.join(plots_folder, spider_file))
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
            
   
    # Plot negotiation sensitivity - save to both locations
    plotter._plot_sensitivity_results_line('negotiation',filename=os.path.join(plots_folder, negotiation_sensitivity_file))
    plotter._plot_sensitivity_results_line('negotiation',filename=os.path.join(DROPBOX_DIR, negotiation_sensitivity_file))

    """ 
    # For production sensitivity
    plotter._plot_sensitivity( 'Production_Change', 'Production',filename=os.path.join(plots_folder, production_file))

    # For generator capture rate sensitivity
    plotter._plot_sensitivity( 'CaptureRate_Change', 'Gen Capture Rate',filename=os.path.join(plots_folder, prod_CR_file))

    # For load capture rate sensitivity
    plotter._plot_sensitivity('Load_CaptureRate_Change', 'Load Capture Rate'.foærmat(),filename=os.path.join(plots_folder, load_CR_file))

    # For price sensitivity
    plotter._plot_sensitivity( 'Price_Change', 'Price',filename=os.path.join(plots_folder, price_file))

    # For load sensitivity
    plotter._plot_sensitivity('Load_Change', 'Load',filename=os.path.join(plots_folder, load_file))
    """

    # Plot alpha sensitivity - save to both locations
    #plotter._plot_sensitivity_results_line('alpha',filename=os.path.join(plots_folder, alpha_sensitivity_file))
    #plotter._plot_sensitivity_results_line('alpha',filename=os.path.join(DROPBOX_DIR, alpha_sensitivity_file))

    # Plot alpha earnings - save to both locations
    #plotter._plot_earnings_histograms_alpha(filename=os.path.join(plots_folder, alpha_earnings_file))
    #plotter._plot_earnings_histograms_alpha(filename=os.path.join(DROPBOX_DIR, alpha_earnings_file))
     
      
    plotter._plot_no_contract_boundaries(filename=os.path.join(plots_folder, boundary_file))
    #plotter._plot_no_contract_boundaries(filename=os.path.join(DROPBOX_DIR, boundary_file))
    plotter._plot_no_contract_boundaries_all(filename=os.path.join(plots_folder, boundary_file_all))
    #plotter._plot_no_contract_boundaries_all(filename=os.path.join(DROPBOX_DIR, boundary_file_all)) 
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()