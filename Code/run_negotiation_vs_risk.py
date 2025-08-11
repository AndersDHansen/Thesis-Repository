import os
import numpy as np
import pandas as pd
from dataloader import load_data
from utils import ForecastProvider
from sensitivity_analysis import run_negotiation_power_vs_risk_sensitivity_analysis

# Configuration
scenario_time_horizon = 20
opt_time_horizon = 20
num_scenarios = 500
A_L = 0.5
A_G = 0.5
tau_L = 0.5
tau_G = 1 - tau_L
Barter = False
contract_type = "PAP"  # "Baseload" or "PAP"
monte_price = False

# Paths
BASE_DIR = os.path.dirname(__file__)
SCEN_DIR = os.path.join(BASE_DIR, 'scenarios')
RES_DIR = os.path.join(BASE_DIR, 'Results')
os.makedirs(RES_DIR, exist_ok=True)

# Load scenarios
scenario_pattern = f"{{type}}_scenarios_reduced_{scenario_time_horizon}y_{num_scenarios}s.csv"
scenario_pattern_reduced_monte = f"{{type}}_scenarios_monte_{scenario_time_horizon}y_{num_scenarios}s.csv"

prod_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='production')), index_col=0)
prod_df.index = pd.to_datetime(prod_df.index)

if monte_price:
    prices_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern_reduced_monte.format(type='price')), index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.columns = prod_df.columns
    prob = np.ones(prices_df.shape[1]) / prices_df.shape[1]
    CR_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern_reduced_monte.format(type='capture_rate')), index_col=0)
    CR_df.index = pd.to_datetime(CR_df.index)
    load_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern_reduced_monte.format(type='load')), index_col=0)
    load_df.index = pd.to_datetime(load_df.index)
    LR_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern_reduced_monte.format(type='load_capture_rate')), index_col=0)
    LR_df.index = pd.to_datetime(LR_df.index)
else:
    prices_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='price')), index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index)
    prob_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='probabilities')), index_col=0)
    prob = prob_df.values.flatten()
    CR_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='capture_rate')), index_col=0)
    CR_df.index = pd.to_datetime(CR_df.index)
    load_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='load')), index_col=0)
    load_df.index = pd.to_datetime(load_df.index)
    LR_df = pd.read_csv(os.path.join(SCEN_DIR, scenario_pattern.format(type='load_capture_rate')), index_col=0)
    LR_df.index = pd.to_datetime(LR_df.index)

provider = ForecastProvider(prices_df, prod_df, CR_df, load_df, LR_df, prob=prob)

# Load data object
input_data = load_data(
    opt_time_horizon=opt_time_horizon,
    num_scenarios=num_scenarios,
    A_G=A_G,
    A_L=A_L,
    tau_L=tau_L,
    tau_G=tau_G,
    Barter=Barter,
    contract_type=contract_type,
)
input_data.load_data_from_provider(provider)

# Parameter grids
A_G_values = [0.1, 0.5, 0.9]
A_L_values = [0.1, 0.5, 0.9]
tau_L_values = np.linspace(0.0, 1.0, 11)

# Run analysis
results_df = run_negotiation_power_vs_risk_sensitivity_analysis(
    input_data,
    A_G_values=A_G_values,
    A_L_values=A_L_values,
    tau_L_values=tau_L_values,
)

# Save CSV with naming convention expected by Plot_visualizations
prefix = "monte_" if monte_price else ""
out_name = f"{prefix}negotiation_vs_risk_sensitivity_{contract_type}_{opt_time_horizon}y_{num_scenarios}.csv"
out_path = os.path.join(RES_DIR, out_name)
results_df.to_csv(out_path)
print(f"Saved: {out_path}")
