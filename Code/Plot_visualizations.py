"""
Run this script to visualize the results without re-runnign the entire code 
Using saved Results from the previous runs
"""

from fileinput import filename
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
        "price_bias_sensitivity","production_bias_sensitivity", # Price and production bias sensitivity results
        "price_sensitivity_mean","price_sensitivity_std",   # Bias is difference between G and L in what they think price will be, price sensitivity it is uniform between
        "production_sensitivity_mean", "production_sensitivity_std",  # Production sensitivity
        "load_sensitivity_mean", "load_sensitivity_std",  # Load sensitivity
        "gen_capture_rate_sensitivity", "load_capture_rate_sensitivity",
        "negotiation_power_sensitivity", "negotiation_earnings_sensitivity", "load_ratio_sensitivity" # tau results are negotation results
        ]
    
    result_names_risk = ["risk_sensitivity", "risk_earnings_sensitivity"]

    results = {}

    # Load each CSV file for risk results
    for result_name in result_names_risk:
        if monte_price == True:
            csv_filename = f"monte_{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        else:
            csv_filename = f"{result_name}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        file_path = os.path.join(csv_folder, csv_filename)
        df = pd.read_csv(file_path, index_col=0)
        results[result_name] = df
        print(f"Loaded {result_name} from {csv_filename}")

    # Load each CSV file for no-risk results
    for result_name in result_names_not_risk:
        if monte_price == True:
            csv_filename = f"monte_{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        else:
            csv_filename = f"{result_name}_AG={A_G}_AL={A_L}_{contract_type}_{time_horizon}y_{num_scenarios}.csv"
        file_path = os.path.join(csv_folder, csv_filename)
        df = pd.read_csv(file_path, index_col=0)
        results[result_name] = df
        print(f"Loaded {result_name} from {csv_filename}")
            
        print(f"Loaded {result_name} from {csv_filename}")
     
    return results

def load_boundary_results(sensitivity,contract_type,time_horizon, num_scenarios):
    """
    Load boundary results from saved JSON file.
    
    Parameters:
    -----------
    sensitivity : str
        Type of sensitivity analysis ("price" or "production")
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
    json_filename = f"boundary_results_{sensitivity}_{contract_type}_{time_horizon}y_{num_scenarios}.json"
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
    global scenario_time_horizon, opt_time_horizon, num_scenarios, A_G, A_L, tau_L, tau_G, Barter, contract_type, monte_price
    # Configuration for loading data 
    scenario_time_horizon = 20  # Must match the scenarios that were generated
    opt_time_horizon = 20  # Must match the scenarios that were generated
    num_scenarios = 5000  # Must match the scenarios that were generated
    A_L = 0.5  # Initial risk aversion
    A_G = 0.5  # Initial risk aversion
    tau_L = 0.5  # Asymmetry of power between load generator [0,1]
    tau_G = 1-tau_L  # Asymmetry of power between generation provider [0,1] - 1-tau_L
    Barter = False  # Whether to relax the problem (Mc Cormick's relaxation)
    contract_type = "PAP" # Either "Baseload" or "PAP"
    monte_price = False  # Whether to use Monte Carlo price scenarios
    # Load data and create InputData object 
    print("Loading data and preparing for simulation...")
    input_data = load_data(
        opt_time_horizon=opt_time_horizon,
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
        'A_L_values':np.array([0,0.1,0.25,0.5,0.75,0.9,1]),
    }
    fixed_A_G=1 # Fixed at middle value for plotting purposes
    # Load scenarios from CSV files
    scenario_pattern = f"{{type}}_scenarios_reduced_{scenario_time_horizon}y_{num_scenarios}s.csv"
    scenario_pattern_reduced_monte = f"{{type}}_scenarios_monte_{scenario_time_horizon}y_{num_scenarios}s.csv"
    
    
    # Load production scenarios
    prod_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='production')}", index_col=0)
    prod_df.index = pd.to_datetime(prod_df.index)
    
    # Load price scenarios
    if monte_price:
        prices_df = pd.read_csv(f"Code/scenarios/{scenario_pattern_reduced_monte.format(type='price')}", index_col=0)
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df.columns = prod_df.columns
        prob_df = np.ones(prices_df.shape[1]) / prices_df.shape[1]
    else:
        prices_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='price')}", index_col=0)
        prices_df.index = pd.to_datetime(prices_df.index)
        prob_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='probabilities')}", index_col=0)
        prob_df = prob_df.values.flatten()

    

    # Load capture rate scenarios
    CR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='capture_rate')}", index_col=0)
    CR_df.index = pd.to_datetime(CR_df.index)    # Load load scenarios
    # Load load scenarios
    load_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load')}", index_col=0)
    load_df.index = pd.to_datetime(load_df.index)
    #load Load Capture Rate scenarios
    LR_df = pd.read_csv(f"Code/scenarios/{scenario_pattern.format(type='load_capture_rate')}", index_col=0)
    LR_df.index = pd.to_datetime(LR_df.index)


    provider = ForecastProvider(prices_df, prod_df,CR_df,load_df,LR_df, prob=prob_df)
    
    # Load data from provider into input_data
    input_data.load_data_from_provider(provider)


    # Load sensitivity results from CSV

    sensitivity_results = load_sensitivity_results(contract_type,opt_time_horizon, num_scenarios)
    
    # Load boundary results from JSON
    boundary_results_price = load_boundary_results("price",contract_type,opt_time_horizon, num_scenarios)
    boundary_results_production = load_boundary_results("production",contract_type,opt_time_horizon, num_scenarios)

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
        price_bias_sensitivity_df=sensitivity_results['price_bias_sensitivity'],
        price_sensitivity_mean_df=sensitivity_results['price_sensitivity_mean'],
        price_sensitivity_std_df=sensitivity_results['price_sensitivity_std'],
        production_bias_sensitivity_df=sensitivity_results['production_bias_sensitivity'],
        production_sensitivity_mean_df=sensitivity_results['production_sensitivity_mean'],
        production_sensitivity_std_df=sensitivity_results['production_sensitivity_std'],
        load_sensitivity_mean_df=sensitivity_results['load_sensitivity_mean'],
        load_sensitivity_std_df=sensitivity_results['load_sensitivity_std'],
        gen_CR_sensitivity_df=sensitivity_results['gen_capture_rate_sensitivity'],
        load_CR_sensitivity_df=sensitivity_results['load_capture_rate_sensitivity'],
        boundary_results_df_price=boundary_results_price,
        boundary_results_df_production=boundary_results_production,
        negotiation_sensitivity_df=sensitivity_results['negotiation_power_sensitivity'],
        negotiation_earnings_df=sensitivity_results['negotiation_earnings_sensitivity'],
        load_ratio_df = sensitivity_results['load_ratio_sensitivity'],


    )
    
    # Generate plots
    print("\nGenerating plots...")

    # Generate filenames
    if monte_price:
        scenario_suffix = "monte"
        risk_file = f"{scenario_suffix}_risk_sensitivity_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        earnings_file = f"{scenario_suffix}_earnings_distribution_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_bias_file = f"{scenario_suffix}_price_bias_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_bias_file = f"{scenario_suffix}_production_bias_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_file_mean = f"{scenario_suffix}_price_sensitivity_mean_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_file_std = f"{scenario_suffix}_price_sensitivity_std_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_file_mean = f"{scenario_suffix}_production_sensitivity_mean_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_file_std = f"{scenario_suffix}_production_sensitivity_std_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_file_mean = f"{scenario_suffix}_load_sensitivity_mean_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_file_std = f"{scenario_suffix}_load_sensitivity_std_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        prod_CR_file = f"{scenario_suffix}_prod_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_CR_file = f"{scenario_suffix}_load_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_price = f"{scenario_suffix}_no_contract_boundary_price_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_all_price = f"{scenario_suffix}_no_contract_boundary_all_price_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_production = f"{scenario_suffix}_no_contract_boundary_production_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_all_production = f"{scenario_suffix}_no_contract_boundary_all_production_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        negotiation_sensitivity_file = f"{scenario_suffix}_negotiation_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        negotation_earnings_file = f"{scenario_suffix}_negotiation_earnings_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_ratio_file = f"{scenario_suffix}_load_ratio_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        #alpha_sensitivity_file = f"alpha_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
        #alpha_earnings_file = f"alpha_earnings_{contract_type}_{time_horizon}_{num_scenarios}.png"
        earnings_boxplot_file = f"{scenario_suffix}_earnings_boxplot_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        spider_file = f"{scenario_suffix}_parameter_sensitivity_spider_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        tornado_file = f"{scenario_suffix}_elasticity_tornado_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        risk_plot_3D_file = f"{scenario_suffix}_risk_3D_plot_AG_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_bias_3D_file = f"{scenario_suffix}_price_bias_3D_plot_AG_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"

        nash_product_file = f"{scenario_suffix}_nash_product_evolution_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        utility_space_file = f"{scenario_suffix}_utility_space_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        disagreement_points_file = f"{scenario_suffix}_disagreement_points_{opt_time_horizon}_{num_scenarios}.png"

    else:
        risk_file = f"risk_sensitivity_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        earnings_file = f"earnings_distribution_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_bias_file = f"price_bias_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_bias_file = f"production_bias_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_file_mean = f"price_sensitivity_mean_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_file_std = f"price_sensitivity_std_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_file_mean = f"production_sensitivity_mean_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        production_file_std = f"production_sensitivity_std_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_file_mean = f"load_sensitivity_mean_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_file_std = f"load_sensitivity_std_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        prod_CR_file = f"prod_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_CR_file = f"load_CR_sensitivity_AG={A_G}_AL={A_L}{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_price = f"no_contract_boundary_price_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_all_price = f"no_contract_boundary_all_price_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_production = f"no_contract_boundary_production_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        boundary_file_all_production = f"no_contract_boundary_all_production_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        negotiation_sensitivity_file = f"negotiation_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        negotation_earnings_file = f"negotiation_earnings_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        load_ratio_file = f"load_ratio_sensitivity_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        #alpha_sensitivity_file = f"alpha_sensitivity_{contract_type}_{time_horizon}_{num_scenarios}.png"
        #alpha_earnings_file = f"alpha_earnings_{contract_type}_{time_horizon}_{num_scenarios}.png"
        earnings_boxplot_file = f"earnings_boxplot_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        spider_file = f"parameter_sensitivity_spider_AG={A_G}_AL={A_L}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        tornado_file = f"elasticity_tornado_AG={fixed_A_G}_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        risk_plot_3D_file = f"risk_3D_plot_AG_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        price_bias_3D_file = f"price_bias_3D_plot_AG_{contract_type}_{opt_time_horizon}_{num_scenarios}.png" 

        nash_product_file = f"nash_product_evolution_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        utility_space_file = f"utility_space_{contract_type}_{opt_time_horizon}_{num_scenarios}.png"
        disagreement_points_file = f"disagreement_points_{opt_time_horizon}_{num_scenarios}.png"

    # Generate plots
    # Boxplot of earnings
    plotter._plot_disagreement_points(filename=os.path.join(plots_folder, disagreement_points_file))
    plotter._plot_disagreement_points(filename=os.path.join(DROPBOX_DIR, disagreement_points_file))
    plotter._risk_plot_earnings_boxplot( fixed_A_G,  # Use middle value
        A_L_to_plot=params['A_L_values'].tolist(),filename=os.path.join(plots_folder, earnings_boxplot_file))
    plotter._risk_plot_earnings_boxplot( fixed_A_G,  # Use middle value
        A_L_to_plot=params['A_L_values'].tolist(),filename=os.path.join(DROPBOX_DIR, earnings_boxplot_file))
    
    plotter._plot_elasticity_tornado(metrics=['StrikePrice','ContractAmount'],bias=False,filename=os.path.join(plots_folder, tornado_file))
    plotter._plot_elasticity_tornado(metrics=['StrikePrice','ContractAmount'],bias=False,filename=os.path.join(DROPBOX_DIR, tornado_file))

    plotter._plot_3D_sensitivity_results(sensitivity_type='risk', filename=os.path.join(plots_folder, risk_plot_3D_file))
    plotter._plot_3D_sensitivity_results(sensitivity_type='risk', filename=os.path.join(DROPBOX_DIR, risk_plot_3D_file))

    plotter._nego_plot_earnings_boxplot(filename=os.path.join(plots_folder, negotation_earnings_file))
    plotter._nego_plot_earnings_boxplot(filename=os.path.join(DROPBOX_DIR, negotation_earnings_file))
    #plotter._plot_parameter_sensitivity_spider(bias=False,filename=os.path.join(plots_folder, spider_file))
    # Risk sensitivity plots - save to both locations
    plotter._plot_sensitivity_results_heatmap('risk',filename=os.path.join(plots_folder, risk_file))
    plotter._plot_sensitivity_results_heatmap('risk',filename=os.path.join(DROPBOX_DIR, risk_file))
            
    # Earnings distribution plots - save to both locations
    plotter._plot_earnings_histograms(
        fixed_A_G,  # Use middle value
        A_L_to_plot=np.array([0.1,0.5,0.9]).tolist(),
        filename=os.path.join(plots_folder, earnings_file)
    )
    plotter._plot_earnings_histograms(
        fixed_A_G,
        A_L_to_plot=np.array([0.1,0.5,0.9]).tolist(),
        filename=os.path.join(DROPBOX_DIR, earnings_file)
    )

    #Threat Point 
    #plotter._plot_expected_versus_threatpoint(fixed_A_G,A_L_to_plot=np.array([0.1,0.5,0.9]).tolist(),filename=os.path.join(plots_folder, 'threat_point.png'))
    #plotter._plot_expected_versus_threatpoint(fixed_A_G,A_L_to_plot=np.array([0.1,0.5,0.9]).tolist(),filename=os.path.join(DROPBOX_DIR, 'threat_point.png'))

    #Radar Chart 

    # Price bias sensitivity plots - save to both locations
    # price_bias or production_bias 
    plotter._plot_sensitivity_results_heatmap('price_bias',filename=os.path.join(plots_folder, price_bias_file))
    plotter._plot_sensitivity_results_heatmap('price_bias',filename=os.path.join(DROPBOX_DIR, price_bias_file))
    plotter._plot_sensitivity_results_heatmap('production_bias',filename=os.path.join(plots_folder, production_bias_file))
    plotter._plot_sensitivity_results_heatmap('production_bias',filename=os.path.join(DROPBOX_DIR, production_bias_file))

    # Production bias sensitivity plots - save to both locations
    #plotter._plot_sensitivity_results_heatmap('production_bias',filename=os.path.join(plots_folder, production_bias_file))
    #plotter._plot_sensitivity_results_heatmap('production_bias',filename=os.path.join(DROPBOX_DIR, production_bias_file))


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

    plotter._plot_no_contract_boundaries(sensitivity_type='price', filename=os.path.join(plots_folder, boundary_file_price))
    plotter._plot_no_contract_boundaries(sensitivity_type='price',filename=os.path.join(DROPBOX_DIR, boundary_file_price))
    plotter._plot_no_contract_boundaries_all(sensitivity_type='price', filename=os.path.join(plots_folder, boundary_file_all_price))
    plotter._plot_no_contract_boundaries_all(sensitivity_type='price', filename=os.path.join(DROPBOX_DIR, boundary_file_all_price))

    #plotter._plot_no_contract_boundaries(sensitivity_type='production', filename=os.path.join(plots_folder, boundary_file_production))
    #plotter._plot_no_contract_boundaries(sensitivity_type='production',filename=os.path.join(DROPBOX_DIR, boundary_file_production))
    #plotter._plot_no_contract_boundaries_all(sensitivity_type='production', filename=os.path.join(plots_folder, boundary_file_all_production))
    #plotter._plot_no_contract_boundaries_all(sensitivity_type='production',filename=os.path.join(DROPBOX_DIR, boundary_file_all_production))
    #print("All plots generated successfully!")

    plotter._plot_nash_product_evolution(filename=os.path.join(plots_folder, nash_product_file))
    plotter._plot_nash_product_evolution(filename=os.path.join(DROPBOX_DIR, nash_product_file))
    #plotter._plot_summary_dashboard(filename='summary_dashboard.png')
    plotter._plot_utility_space(filename=os.path.join(plots_folder, utility_space_file))
    plotter._plot_utility_space(filename=os.path.join(DROPBOX_DIR, utility_space_file))


if __name__ == "__main__":
    main()