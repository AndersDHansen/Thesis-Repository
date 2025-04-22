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
from data_management import load_data, InputData
from power_flow import OptimalPowerFlow
from visualization import Plotting_Class
from sensitivity_analysis import run_risk_sensitivity_analysis, run_bias_sensitivity_analysis
from contract_negotiation import ContractNegotiation
def main():
    # Define simulation parameters
    HOURS = 12 # Number of hours in a day
    DAYS = 5 # Number of days in the simulation (June - 30 days)
    SCENARIOS = 25
    
    # Define risk aversion parameter ranges
    #A_G6_values = np.insert(np.round(np.linspace(0.1, 0.9, 3), 2), 0, 0)
    #A_L2_values = np.insert(np.round(np.linspace(0.1, 0.9, 3), 2), 0, 0)
    A_G6_values = np.round(np.linspace(0 ,1, 4), 2)
    A_L2_values = np.round(np.linspace(0, 1, 4), 2)
    beta_L = 0.5  # Initial risk aversion
    beta_G = 0.5 # Initial risk aversion

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
        beta_L=beta_L, #A_L2_value,
        beta_G=beta_G #A_G6_value,
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

    # Run Optimal Power Flow
    print("\nRunning Optimal Power Flow simulation...")
    opf_model = OptimalPowerFlow(input_data)
    opf_model.run()
    print("OPF simulation complete.")

    # Run Contract Negotiation with Gurobi
    print("\nStarting contract negotiation with Gurobi...")
    contract_model = ContractNegotiation(input_data, opf_model.results)
    contract_model.run()
    contract_model.display_results()
    print("Gurobi optimization complete.")
    input_data.strikeprice_min = 20
    input_data.strikeprice_max = 21
    contract_model = ContractNegotiation(input_data, opf_model.results)
    contract_model.manual_optimization(plot=True)

    # Run Contract Negotiation with SciPy
    #print("\nStarting contract negotiation with SciPy...")
    #scipy_results = contract_model.scipy_optimization()

    # Run Risk Sensitivity Analysis
    print("\nStarting risk sensitivity analysis (No Monte Carlo)...")
    risk_sensitivity_results, earnings_sensitivity = run_risk_sensitivity_analysis(
        input_data,
        opf_model.results,
        A_G6_values,
        A_L2_values,
    )

    #risk_sensitivity_results_monte, earnings_sensitivity_monte = run_risk_sensitivity_analysis(input_data,opf_model.results,A_G6_values,A_L2_values,Monte_Carlo =True)

    print("\nRisk sensitivity analysis complete.")
    print("\nStarting price bias sensitivity analysis...")
    bias_sensitivity_results = run_bias_sensitivity_analysis(input_data,opf_model.results)
    print(earnings_sensitivity)

    # Run Bias Sensitivity Analysis
 
    print("\nPrice bias sensitivity analysis complete.")
    print(bias_sensitivity_results)

    # Create plots
    print("\nGenerating visualization plots...")
    plot_obj = Plotting_Class(contract_model.data,
        risk_sensitivity_results,
        earnings_sensitivity,
        bias_sensitivity_results
    )

    #plot_obj._plot_no_contract(
        #filename="no_contract.png"
    #    )
    plot_obj._plot_expected_versus_threatpoint(fixed_A_G6=A_G6_values[3],A_L2_to_plot=A_L2_values.tolist())

    # Plot Risk Sensitivity Results
    plot_obj._plot_sensitivity_results(
        filename="risk_sensitivity.png"
        )
    #plot_obj._plot_sensitivity_results(filename="risk_sensitivity_monte.png",new_data = risk_sensitivity_results_monte)

    # Plot Earnings Histograms
    plot_obj._plot_earnings_histograms(fixed_A_G6=A_G6_values[3],A_L2_to_plot=A_L2_values.tolist(),
        filename="earnings_distribution.png"
    )
    """
    plot_obj._plot_earnings_histograms(
        fixed_A_G6=A_G6_values[1],
        A_L2_to_plot=A_L2_values.tolist(),
        filename="earnings_distribution_monte.png",
        new_data = earnings_sensitivity_monte
    )
    """
    # Plot Bias Sensitivity Results
    plot_obj._plot_bias_sensitivity(filename=f"bias_sensitivity_AG6_{A_G6_values[0]:.1f}-{A_G6_values[-1]:.1f}_AL2_{A_L2_values[0]:.1f}-{A_L2_values[-1]:.1f}.png")
    #print("Visualization complete. Results saved to PNG files.")

if __name__ == "__main__":
    main()