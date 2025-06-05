"""
Sensitivity analysis module for power system contract negotiation.
"""
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from contract_negotiation import ContractNegotiation

def run_risk_sensitivity_analysis(input_data_base, provider, A_G_values, A_L_values, old_obj_func=False):
    """Runs the ContractNegotiation for different combinations of A_G and A_L."""
    results_list = []
    earnings_list_scenarios = []

    for a_G in tqdm(A_G_values, desc="Iterating A_G"):
        for a_L in tqdm(A_L_values, desc="Iterating A_L", leave=False):
            # Use a fresh copy for each iteration
            current_input_data = copy.deepcopy(input_data_base)
            current_input_data.A_G = a_G
            current_input_data.A_L = a_L

            try:
                contract_model = ContractNegotiation(current_input_data, provider, old_obj_func=old_obj_func)
                contract_model.run()
                
                results_list.append({
                    'A_G': a_G,
                    'A_L': a_L,
                    'StrikePrice': contract_model.scipy_results.strike_price,
                    'ContractAmount': contract_model.scipy_results.contract_amount,
                    'Utility_G': contract_model.scipy_results.utility_G,
                    'Utility_L': contract_model.scipy_results.utility_L,
                    'NashProductLog': contract_model.scipy_results.objective_value,
                    'NashProduct': np.exp(contract_model.scipy_results.objective_value),
                    'ThreatPoint_G': contract_model.data.Zeta_G,
                    'ThreatPoint_L': contract_model.data.Zeta_L,
                })

                earnings_list_scenarios.append(pd.DataFrame({
                    'A_G': a_G,
                    'A_L': a_L,
                    'Revenue_G': contract_model.scipy_results.accumulated_revenue_True_G,
                    'Revenue_L': contract_model.scipy_results.accumulated_revenue_True_L,
                }))

            except Exception as e:
                print(f"Error for A_G={a_G}, A_L={a_L}: {e}")
                results_list.append({
                    'A_G': a_G, 'A_L': a_L,
                    'StrikePrice': np.nan, 'ContractAmount': np.nan,
                    'Utility_G': np.nan, 'Utility_L': np.nan,
                    'NashProductLog': np.nan, 'NashProduct': np.nan,
                    'ThreatPoint_G': np.nan, 'ThreatPoint_L': np.nan,
                })
                earnings_list_scenarios.append(pd.DataFrame({
                    'A_G': a_G, 'A_L': a_L,
                    'Revenue_G': np.nan, 'Revenue_L': np.nan,
                }))

            # Clean up
            del contract_model

    return pd.DataFrame(results_list), earnings_list_scenarios

def run_bias_sensitivity_analysis(input_data_base, provider, old_obj_func=False):
    """Performs sensitivity analysis on price bias factors (KG, KL)."""
    print("\n--- Starting Price Bias Sensitivity Analysis ---")

    K_G_factors = [-0.01, 0, 0.01]
    K_L_factors = [-0.01, 0, 0.01]
    results_sensitivity = []

    for kg_factor in K_G_factors:
        for kl_factor in K_L_factors:
            # Use a fresh copy for each iteration
            current_input_data = copy.deepcopy(input_data_base)
            current_input_data.K_G = kg_factor
            current_input_data.K_L = kl_factor

            try:
                contract_model = ContractNegotiation(current_input_data, provider, old_obj_func=old_obj_func)
                contract_model.run()

                results_sensitivity.append({
                    'KG_Factor': kg_factor,
                    'KL_Factor': kl_factor,
                    'StrikePrice': contract_model.scipy_results.strike_price,
                    'ContractAmount': contract_model.scipy_results.contract_amount,
                    'Utility_G': contract_model.scipy_results.utility_G,
                    'Utility_L': contract_model.scipy_results.utility_L,
                    'ThreatPoint_G': contract_model.data.Zeta_G,
                    'ThreatPoint_L': contract_model.data.Zeta_L,
                })

            except Exception as e:
                print(f"Error for KG={kg_factor}, KL={kl_factor}: {e}")
                results_sensitivity.append({
                    'KG_Factor': kg_factor, 'KL_Factor': kl_factor,
                    'StrikePrice': np.nan, 'ContractAmount': np.nan,
                    'Utility_G': np.nan, 'Utility_L': np.nan,
                    'ThreatPoint_G': np.nan, 'ThreatPoint_L': np.nan,
                })

            # Clean up
            del contract_model

    print("\n--- Price Bias Sensitivity Analysis Complete ---")
    return pd.DataFrame(results_sensitivity)