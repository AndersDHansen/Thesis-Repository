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
from dataclasses import dataclass
from typing import Optional, Tuple
import os
from dataloader import load_data, InputData
from power_flow import OptimalPowerFlow
from visualization import Plotting_Class
from sensitivity_analysis import run_risk_sensitivity_analysis, run_bias_sensitivity_analysis
from contract_negotiation import ContractNegotiation
from Forecasting import (PriceForecastConfig, PriceForecast,HistoricalProductionConfig, HistoricalProductionForecaster,MonteCarloConfig, MonteCarloSimulator,
                         CaptureRateConfig, CaptureRateForecaster)
from utils import OPFProvider, ForecastProvider
import copy



def run_contract_negotiation(input_data: InputData, provider, old_obj_func: bool, 
                           A_G_values: np.ndarray, A_L_values: np.ndarray) -> tuple:
    """Run contract negotiation and sensitivity analysis with given OPF results."""
    
    # Store original data
    original_data = copy.deepcopy(input_data)
    
    # Run base case with middle values
    base_input = copy.deepcopy(original_data)
    base_input.A_G = A_G_values[0]  # Use middle value for A_G
    base_input.A_L = A_L_values[0]  # Use middle value for A_L
    
    contract_model = ContractNegotiation(base_input, provider, old_obj_func=old_obj_func)
    contract_model.run()
    base_results = copy.deepcopy(contract_model.results)
    base_data = copy.deepcopy(contract_model.data)
    
    print(2*"\nRunning contract negotiation...")
    
    # Run Sensitivity Analyses with fresh copies
    risk_input = copy.deepcopy(original_data)
    risk_sensitivity_results, earnings_sensitivity = run_risk_sensitivity_analysis(
        risk_input, provider, A_G_values, A_L_values, old_obj_func=old_obj_func
    )
    
    bias_input = copy.deepcopy(original_data)
    bias_sensitivity_results = run_bias_sensitivity_analysis(
        bias_input, provider, old_obj_func
    )

    return base_data, risk_sensitivity_results, earnings_sensitivity, bias_sensitivity_results


def main():    
    # Define simulation parameters
    NUM_SCENARIOS = 100
    A_L = 0.5 # Initial risk aversion
    A_G = 0.5 # Initial risk aversion

    time_horizon = 2 # temporary number of years simulated  
    start_date = pd.to_datetime('2025-01-01')
  

    # Load data and create InputData object 
    print("Loading data and preparing for simulation...")
    (LOADS, TIME, SCENARIOS, PROB,
    generator_contract_capacity,
    load_scenarios, retail_price,
    strikeprice_min, strikeprice_min, strikeprice_max,
    contract_amount_min, contract_amount_max,
    A_L, A_G, K_L, K_G, alpha )= load_data(
    time_horizon=time_horizon,
    NUM_SCENARIOS=NUM_SCENARIOS,
    A_G=A_G, #A_L_value,
    A_L=A_L #A_G_value,
    )

    # Create InputData object
    input_data = InputData(
        LOADS=LOADS,    
        TIME=TIME,
        NUM_SCENARIOS=NUM_SCENARIOS,
        SCENARIOS=SCENARIOS,
        PROB=PROB,
        generator_contract_capacity = generator_contract_capacity,
        load_scenarios=load_scenarios,
        retail_price=retail_price,
        strikeprice_min=strikeprice_min,
        strikeprice_max=strikeprice_max,
        contract_amount_min=contract_amount_min,
        contract_amount_max=contract_amount_max,
        A_L=A_L,
        A_G=A_G,
        K_L=K_L,
        K_G=K_G,
        alpha=alpha
    )

     # Define risk aversion parameters for both objective functions
    
    
    new_obj_params = {
        #'A_G_values': np.round(np.linspace(1, 0.5 , 3), 2),  # A in [0,1]
        #'A_L_values': np.round(np.linspace(1, 0.5, 3), 2)
        'A_G_values':  np.array([0.1,0.5,0.9]),  # A in [0,1]
        'A_L_values':  np.array([0.1,0.5,0.9])
    }

    old_obj_params = {
        #'A_G_values': np.array([1, 1, 1]),  # A >= 0
        #A_L_values': np.array([1, 1, 1])
        'A_G_values': np.round(1/new_obj_params['A_G_values']-1,1), 
        'A_L_values': np.round(1/new_obj_params['A_L_values']-1,1) # A >= 0
    }

  

    # Run Forecasts      # ---------- Load scenarios from CSV files ---------------------------------
    time_horizon = 2  # Must match the scenarios that were generated
    num_scenarios = 10000  # Must match the scenarios that were generated    scenario_pattern = f"{{type}}_scenarios_{time_horizon}y_{num_scenarios}s.csv"

    # Load price scenarios
    prices_df = pd.read_csv(f"scenarios/{scenario_pattern.format(type='price')}", index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)

    # Load production scenarios
    prod_df = pd.read_csv(f"scenarios/{scenario_pattern.format(type='production')}", index_col=0)
    prod_df.index = pd.to_datetime(prod_df.index)

    # Load capture rate scenarios
    CR_df = pd.read_csv(f"scenarios/{scenario_pattern.format(type='capture_rate')}", index_col=0)
    CR_df.index = pd.to_datetime(CR_df.index)    # Load load scenarios
    load_df = pd.read_csv(f"scenarios/{scenario_pattern.format(type='load')}", index_col=0)
    load_df.index = pd.to_datetime(load_df.index)


    provider = ForecastProvider(prices_df, prod_df,CR_df, prob=1/prices_df.shape[1])

    #provider = OPFProvider(opf_model.results,prob=1/SCENARIOS)
    # Run contract negotiation for both objective functions with same OPF results
    print("\nRunning simulation with original objective function (E + A*CVaR)...")
    """
    old_results = run_contract_negotiation(
        copy.deepcopy(input_data), 
        provider,
        True, 
        old_obj_params['A_G_values'], 
        old_obj_params['A_L_values']
    )
   """
    
    print("\nRunning simulation with modified objective function ((1-A)*E + A*CVaR)...")
    new_results = run_contract_negotiation(
        copy.deepcopy(input_data),
        provider,
        False,
        new_obj_params['A_G_values'],
        new_obj_params['A_L_values']
    )
    
    # Create comparison plots
    print("\nGenerating comparison plots...")

    print(new_results[1])
    """
    print(old_results[1])
    
    # Plot results for original objective function
    plot_obj_old = Plotting_Class(
        old_results[0].data,
        old_results[1],
        old_results[2],
        old_results[3]
    )
    """
    
    # Plot results for new objective function
    plot_obj_new = Plotting_Class(
        new_results[0],
        new_results[1],
        new_results[2],
        new_results[3]
    )

        
       # Generate plots with distinctive filenames
    for plot_obj, obj_type, params in [
        #(plot_obj_old, 'original', old_obj_params),
        (plot_obj_new, 'modified', new_obj_params)
    ]:
        # Risk sensitivity plots
        plot_obj._plot_sensitivity_results(
            filename=f"risk_sensitivity_{obj_type}.png"
        )
        
        # Earnings distribution plots
        plot_obj._plot_earnings_histograms(
            fixed_A_G=params['A_G_values'][1],  # Use middle value
            A_L_to_plot=params['A_L_values'].tolist(),
            filename=f"earnings_distribution_{obj_type}.png"
        )
        
        # Bias sensitivity plots
        plot_obj._plot_bias_sensitivity(
            filename=f"bias_sensitivity_{obj_type}.png"
        )
      
    
if __name__ == "__main__":
    main()