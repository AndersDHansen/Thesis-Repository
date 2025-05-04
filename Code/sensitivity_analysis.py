"""
Sensitivity analysis module for power system contract negotiation.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from contract_negotiation import ContractNegotiation
from utils import build_dataframe

def run_risk_sensitivity_analysis(input_data_base, opf_results, A_G6_values, A_L2_values,old_obj_func = False):
    """
    Runs the ContractNegotiation for different combinations of A_G6 and A_L2.

    Args:
        input_data_base: InputData object with base parameters
        opf_results: Results from OptimalPowerFlow run
        A_G6_values: Array of A_G6 values to test
        A_L2_values: Array of A_L2 values to test

    Returns:
        tuple: (DataFrame with sensitivity results, List of earnings DataFrames)
    """
    results_list = []
    earnings_list_scenarios = []
    current_input_data = input_data_base

    for a_g6 in tqdm(A_G6_values, desc="Iterating A_G6"):
        for a_l2 in tqdm(A_L2_values, desc="Iterating A_L2", leave=False):
            current_input_data.A_G6 = a_g6
            current_input_data.A_L2 = a_l2

            try:
                # Run contract negotiation
                contract_model = ContractNegotiation(current_input_data, opf_results, old_obj_func=old_obj_func)
                contract_model.run()
                
                # Store results
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': contract_model.results.strike_price,
                    'ContractAmount': contract_model.results.contract_amount,
                    'Utility_G6': contract_model.results.utility_G6,
                    'Utility_L2': contract_model.results.utility_L2,
                    'NashProductLog': contract_model.results.objective_value,
                    'NashProduct': np.exp(contract_model.results.objective_value),
                    'ThreatPoint_G6': contract_model.data.Zeta_G6,
                    'ThreatPoint_L2': contract_model.data.Zeta_L2,
                })

                earnings_list_scenarios.append(pd.DataFrame({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': contract_model.results.accumulated_revenue_G6,
                    'Revenue_L2': contract_model.results.accumulated_revenue_L2,
                }))

            except RuntimeError as e:
                print(f"Optimization failed for A_G6={a_g6}, A_L2={a_l2}: {e}")
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': np.nan,
                    'ContractAmount': np.nan,
                    'Utility_G6': np.nan,
                    'Utility_L2': np.nan,
                    'NashProductLog': np.nan,
                    'NashProduct': np.nan
                })
                earnings_list_scenarios.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': np.nan,
                    'Revenue_L2': np.nan,
                })

            except Exception as e:
                print(f"An error occurred for A_G6={a_g6}, A_L2={a_l2}: {e}")
                results_list.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'StrikePrice': np.nan,
                    'ContractAmount': np.nan,
                    'Utility_G6': np.nan,
                    'Utility_L2': np.nan,
                    'NashProductLog': np.nan,
                    'NashProduct': np.nan,
                    'ThreatPoint_G6': np.nan,
                    'ThreatPoint_L2': np.nan,
                })
                earnings_list_scenarios.append({
                    'A_G6': a_g6,
                    'A_L2': a_l2,
                    'Revenue_G6': np.nan,
                    'Revenue_L2': np.nan,
                })

    return pd.DataFrame(results_list), earnings_list_scenarios

def run_bias_sensitivity_analysis(input_data_base, opf_results,old_obj_func = False):
    """
    Performs sensitivity analysis on price bias factors (KG, KL).

    Args:
        input_data_base: InputData object with base parameters
        opf_results: Results from OptimalPowerFlow run

    Returns:
        pandas.DataFrame: DataFrame containing the sensitivity results
    """
    print("\n--- Starting Price Bias Sensitivity Analysis ---")

    # Calculate true expected price sum
    price_df = build_dataframe(opf_results.price['N3'], 'price')
    price_true = price_df.values
    lambda_sum_true = price_true.sum(axis=0)
    EP_lambda_sum_true = lambda_sum_true.mean() if lambda_sum_true.size > 0 else 0

    if EP_lambda_sum_true == 0:
        print("Warning: True Expected Sum of Prices (EP(λΣ)) is zero. Biases will be zero.")
    else:
        print(f"True Expected Sum of Prices (EP(λΣ)): {EP_lambda_sum_true:.4f}")

    # Define bias factors
    K_G6_factors = [-0.01, 0, 0.01]
    K_L2_factors = [-0.01, 0, 0.01]

    results_sensitivity = []
    for kg_factor in K_G6_factors:
        for kl_factor in K_L2_factors:
            try:
                print(f"\nRunning Negotiation for KG factor={kg_factor} "
                      f"(Bias={(1+kg_factor)*EP_lambda_sum_true:.2f}), "
                      f"KL factor={kl_factor} "
                      f"(Bias={(1+kl_factor)*EP_lambda_sum_true:.2f})")

                # Update bias factors
                input_data_base.K_G6 = kg_factor
                input_data_base.K_L2 = kl_factor

                # Run contract negotiation
                contract_model = ContractNegotiation(input_data_base, opf_results,old_obj_func=old_obj_func)
                contract_model.run()

                # Store results
                results_sensitivity.append({
                    'KG_Factor': kg_factor,
                    'KL_Factor': kl_factor,
                    'KG_Bias': (1+kg_factor)*EP_lambda_sum_true,
                    'KL_Bias': (1+kl_factor)*EP_lambda_sum_true,
                    'StrikePrice': contract_model.results.strike_price,
                    'ContractAmount': contract_model.results.contract_amount,
                    'threat_point_G6': contract_model.data.Zeta_G6,
                    'threat_point_L2': contract_model.data.Zeta_L2,
                })

                if np.isnan(contract_model.results.strike_price):
                    print(" -> Negotiation failed or no feasible solution found.")
                else:
                    print(f" -> Optimal S: {contract_model.results.strike_price:.4f}, "
                          f"Optimal M: {contract_model.results.contract_amount:.4f}")

            except Exception as e:
                print(f" -> Error during negotiation for KG factor={kg_factor}, "
                      f"KL factor={kl_factor}: {e}")
                results_sensitivity.append({
                    'KG_Factor': kg_factor,
                    'KL_Factor': kl_factor,
                    'KG_Bias': (1+kg_factor)*EP_lambda_sum_true,
                    'KL_Bias': (1+kl_factor)*EP_lambda_sum_true,
                    'StrikePrice': np.nan,
                    'ContractAmount': np.nan
                })

    print("\n--- Price Bias Sensitivity Analysis Complete ---")
    return pd.DataFrame(results_sensitivity)