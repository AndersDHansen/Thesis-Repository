"""
Sensitivity analysis module for contract negotiation.
"""
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from contract_negotiation import ContractNegotiation

def run_capture_price_analysis(input_data_base):
    """Performs sensitivity analysis on capture price value from simulations."""
    current_input_data = copy.deepcopy(input_data_base)

    if current_input_data.contract_type == "PAP":
        Capture_price_G = ((current_input_data.price_true * current_input_data.capture_rate * current_input_data.production).sum()/current_input_data.production.sum()).mean()
        avg_price = Capture_price_G
        # Set constraints for strike pices equal to the capture price of the generator
    else:
        Capture_price_L = ((current_input_data.price_true * current_input_data.load_CR * current_input_data.load_scenarios).sum()/current_input_data.load_scenarios.sum()).mean()

        avg_price = Capture_price_L
       
    current_input_data.strikeprice_max = avg_price
    current_input_data.strikeprice_min = avg_price  # Set Average price as the comparison case - since that is the 'base price'
    
        # Use a fresh copy for each iteration
    print(f"\n--- Starting Capture Price Sensitivity Analysis with Capture Price = {avg_price} ---")
    try:
        
        contract_model = ContractNegotiation(current_input_data)

        contract_model.run()
        
        # Create base result dictionary
        result_dict = {
            'Capture_Price': [avg_price],
            'StrikePrice': [contract_model.results.strike_price],
            'A_G': [current_input_data.A_G],
            'A_L': [current_input_data.A_L],
            'ContractAmount': [contract_model.results.contract_amount_hour],
            'Utility_G': [contract_model.results.utility_G],
            'Utility_L': [contract_model.results.utility_L],
            'ThreatPoint_G': [contract_model.data.Zeta_G],
            'ThreatPoint_L': [contract_model.data.Zeta_L],
            'Nash_Product': [(contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                            (contract_model.results.utility_L - contract_model.data.Zeta_L)]
        }
        
        # Add contract-type specific metrics
        if hasattr(current_input_data, 'contract_type'):
            if current_input_data.contract_type == "PAP":
                if hasattr(contract_model.results, 'gamma'):
                    result_dict['Gamma'] = contract_model.results.gamma
            elif current_input_data.contract_type == "Baseload":
                # Add Baseload-specific metrics if needed
                pass

          # Create earnings results for histograms
        earnings_df = pd.DataFrame({
            'A_G': current_input_data.A_G, 
            'A_L': current_input_data.A_L,
            'Revenue_G_CP': contract_model.results.earnings_G.values,
            'Revenue_L_CP': contract_model.results.earnings_L.values,

        })

        
    except Exception as e:
        print(f"Error for capture rate multiplier={avg_price}: {str(e)}")
        result_dict = {
            'CaptureRate_Change': [avg_price],
            'Avg_G_Capture_Rate': [np.nan],
            'A_G': [current_input_data.A_G],
            'A_L': [current_input_data.A_L],
            'StrikePrice': [np.nan],
            'ContractAmount': [np.nan],
            'Utility_G': [np.nan],
            'Utility_L': [np.nan],
            'ThreatPoint_G': [np.nan],
            'ThreatPoint_L': [np.nan],
            'Nash_Product': [np.nan]
        }
        earnings_df = pd.DataFrame({
                    'A_G': [current_input_data.A_G], 
                    'A_L': [current_input_data.A_L],
                    'Revenue_G': [np.nan], 
                    'Revenue_L': [np.nan],

                })
        
    # Clean up
    del contract_model
    
# Add analysis of results
    results_df = pd.DataFrame(result_dict)

    return results_df, earnings_df



def run_risk_sensitivity_analysis(input_data_base, A_G_values, A_L_values):
    """Runs the ContractNegotiation for different combinations of A_G and A_L."""
    print("\n--- Starting Risk Aversion Sensitivity Analysis ---")
    results_list = []
    earnings_list_scenarios = []

    for a_G in tqdm(A_G_values, desc="Iterating A_G"):
        for a_L in tqdm(A_L_values, desc="Iterating A_L", leave=False):
            # Use a fresh copy for each iteration
            current_input_data = copy.deepcopy(input_data_base)
            current_input_data.A_G = a_G
            current_input_data.A_L = a_L

            try:
                contract_model = ContractNegotiation(current_input_data)
                contract_model.run()
                
                # Create base result dictionary
                result_dict = {
                    'A_G': a_G,
                    'A_L': a_L,
                    'StrikePrice': contract_model.results.strike_price,
                    'ContractAmount': contract_model.results.contract_amount_hour,
                    'Utility_G': contract_model.results.utility_G,
                    'Utility_L': contract_model.results.utility_L,
                    'NashProductLog': contract_model.results.objective_value,
                    'Nash_Product': np.exp(contract_model.results.objective_value),
                    'ThreatPoint_G': contract_model.data.Zeta_G,
                    'ThreatPoint_L': contract_model.data.Zeta_L,
                }
                
                # Add contract-type specific metrics
                if hasattr(current_input_data, 'contract_type'):
                    if current_input_data.contract_type == "PAP":
                        if hasattr(contract_model.results, 'gamma'):
                            result_dict['Gamma'] = contract_model.results.gamma
                    elif current_input_data.contract_type == "Baseload":
                        # Add Baseload-specific metrics if needed
                        pass
                
                results_list.append(result_dict)
                
                # Create earnings results for histograms
                earnings_df = pd.DataFrame({
                    'A_G': a_G, 
                    'A_L': a_L,
                    'Revenue_G': contract_model.results.earnings_G.values,
                    'Revenue_L': contract_model.results.earnings_L.values,
                })
                earnings_list_scenarios.append(earnings_df)

            except Exception as e:
                print(f"Error for A_G={a_G}, A_L={a_L}: {str(e)}")
                results_list.append({
                    'A_G': a_G, 
                    'A_L': a_L,
                    'StrikePrice': np.nan, 
                    'ContractAmount': np.nan,
                    'Utility_G': np.nan, 
                    'Utility_L': np.nan,
                    'NashProductLog': np.nan,
                    'Nash_Product': np.nan,
                    'ThreatPoint_G': np.nan, 
                    'ThreatPoint_L': np.nan,
                })
                
                earnings_list_scenarios.append(pd.DataFrame({
                    'A_G': [a_G], 
                    'A_L': [a_L],
                    'Revenue_G': [np.nan], 
                    'Revenue_L': [np.nan],

                }))

            # Clean up
            del contract_model
    
    # Combine results and analyze
    results_df = pd.DataFrame(results_list)
    earnings_df = pd.concat(earnings_list_scenarios, ignore_index=True)

    print("\n--- Risk Aversion Sensitivity Analysis Complete ---")
    return results_df, earnings_df

def run_bias_sensitivity_analysis(input_data_base):
    """Performs sensitivity analysis on price bias factors."""
    print("\n--- Starting Price Bias Sensitivity Analysis ---")

    # Define bias factors - wider range for more insights
    #K_G_factors = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
    #K_L_factors = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
    K_G_factors = [-0.01, 0, 0.01]
    K_L_factors = [-0.01, 0, 0.01]
    results_sensitivity = []

    # Create parameter grid and use tqdm for progress tracking
    param_grid = [(kg, kl) for kg in K_G_factors for kl in K_L_factors]
    for kg_factor, kl_factor in tqdm(param_grid, desc="Testing bias combinations"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        current_input_data.K_G = kg_factor
        current_input_data.K_L = kl_factor

        try:
            contract_model = ContractNegotiation(current_input_data)
            contract_model.run()

            # Create base result dictionary
            result_dict = {
                'KG_Factor': kg_factor,
                'KL_Factor': kl_factor,
                 'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': contract_model.results.strike_price,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
                    
            results_sensitivity.append(result_dict)
            
        except Exception as e:
            print(f"Error for KG={kg_factor}, KL={kl_factor}: {str(e)}")
            results_sensitivity.append({
                'KG_Factor': kg_factor, 
                'KL_Factor': kl_factor,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan, 
                'ContractAmount': np.nan,
                'Utility_G': np.nan, 
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan, 
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })

        # Clean up
        del contract_model

    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)
  

    print("\n--- Price Bias Sensitivity Analysis Complete ---")
    return results_df

def run_capture_rate_sensitivity_analysis(input_data_base):
    """Performs sensitivity analysis on capture rate values."""
    print("\n--- Starting Capture Rate Sensitivity Analysis ---")
    
    # Define capture rate multipliers (as percentages of original values)
    # Testing from 70% to 130% of original capture rates
    cr_multipliers = [0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, ]
    results_sensitivity = []
    
    for cr_mult in tqdm(cr_multipliers, desc="Testing capture rate multipliers"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        
        try:
            # Create a modified provider with adjusted capture rates
            current_input_data.capture_rate = current_input_data.capture_rate * cr_mult
            
            
            # Run the contract negotiation with modified capture rates
            contract_model = ContractNegotiation(current_input_data)

            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'CaptureRate_Change': cr_mult,
                'Avg_G_Capture_Rate': contract_model.data.capture_rate.mean().mean(),
                'StrikePrice': contract_model.results.strike_price,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_sensitivity.append(result_dict)
            
        except Exception as e:
            print(f"Error for capture rate multiplier={cr_mult}: {str(e)}")
            results_sensitivity.append({
                'CaptureRate_Change': cr_mult,
                'Avg_G_Capture_Rate': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
            
        # Clean up
        del contract_model
        
    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)

    return results_df

def run_price_sensitivity_analysis(input_data_base):
    """Performs sensitivity analysis on price values."""
    print("\n--- Starting Capture Rate Sensitivity Analysis ---")
    
    # Define capture rate multipliers (as percentages of original values)
    # Testing from 70% to 130% of original capture rates
    cr_multipliers = [0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, ]
    results_sensitivity = []
    
    for price_mult in tqdm(cr_multipliers, desc="Testing capture rate multipliers"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        
        try:
            # Create a modified provider with adjusted capture rates
            current_input_data.price_true = current_input_data.price_true * price_mult
            
            
            # Run the contract negotiation with modified capture rates
            contract_model = ContractNegotiation(current_input_data)

            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'Price_Change': price_mult,
                'Avg_G_Price': contract_model.data.price_true.mean().mean(),
                'StrikePrice': contract_model.results.strike_price,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_sensitivity.append(result_dict)
            
        except Exception as e:
            print(f"Error for capture rate multiplier={price_mult}: {str(e)}")
            results_sensitivity.append({
                'Price_Change': price_mult,
                'Avg_G_Price': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
            
        # Clean up
        del contract_model
        
    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)

    return results_df

def run_production_sensitivity_analysis(input_data_base):

    """Performs sensitivity analysis on production rate values."""
    print("\n--- Starting Capture Rate Sensitivity Analysis ---")
    
    # Define capture rate multipliers (as percentages of original values)
    # Testing from 70% to 130% of original capture rates
    production_multipliers = [0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, ]
    results_sensitivity = []
    
    for prod_mult in tqdm(production_multipliers, desc="Testing Production multipliers"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        
       # try:
            # Create a modified provider with adjusted capture rates
        current_input_data.production = current_input_data.production * prod_mult
        try:
        
            
            # Run the contract negotiation with modified capture rates
            contract_model = ContractNegotiation(current_input_data)
            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'Production_Change': prod_mult,
                'Avg_Production': contract_model.data.production.mean().mean(),
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': contract_model.results.strike_price,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            results_sensitivity.append(result_dict)
        
                 
        except Exception as e:
            print(f"Error for capture rate multiplier={prod_mult}: {str(e)}")
            results_sensitivity.append({
                'Production_Change': prod_mult,
                'Avg_Capture_Rate': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
        
            

        
        # Clean up
        del contract_model
        
    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)

    return results_df

def run_load_capture_rate_sensitivity_analysis(input_data_base):
    """Performs sensitivity analysis on load capture rate values."""
    print("\n--- Starting Capture Rate Sensitivity Analysis ---")
    
    # Define capture rate multipliers (as percentages of original values)
    # Testing from 70% to 130% of original capture rates
    LR_multipliers = [0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, ]
    results_sensitivity = []
    
    for lr_mult in tqdm(LR_multipliers, desc="Testing capture rate multipliers"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        
        try:
            # Create a modified provider with adjusted capture rates
            current_input_data.load_CR = current_input_data.load_CR * lr_mult
            
            
            # Run the contract negotiation with modified capture rates
            contract_model = ContractNegotiation(current_input_data)

            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'Load_CaptureRate_Change': lr_mult,
                'Avg_Load_Capture_Rate': contract_model.data.load_CR.mean().mean(),
                'StrikePrice': contract_model.results.strike_price,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_sensitivity.append(result_dict)
            
        except Exception as e:
            print(f"Error for capture rate multiplier={lr_mult}: {str(e)}")
            results_sensitivity.append({
                'Load_CaptureRate_Change': lr_mult,
                'Avg_Load_Capture_Rate': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
            
        # Clean up
        del contract_model
        
    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)

    return results_df

def run_load_scenario_sensitivity_analysis(input_data_base):
    """Performs sensitivity analysis on load rate values."""
    print("\n--- Starting Capture Rate Sensitivity Analysis ---")
    
    # Define capture rate multipliers (as percentages of original values)
    # Testing from 70% to 130% of original capture rates
    Load_multipliers = [0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, ]
    results_sensitivity = []
    
    for load_mult in tqdm(Load_multipliers, desc="Testing capture rate multipliers"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        
        try:
            # Create a modified provider with adjusted capture rates
            current_input_data.load_scenarios = current_input_data.load_scenarios * load_mult
            
            
            # Run the contract negotiation with modified capture rates
            contract_model = ContractNegotiation(current_input_data)

            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'Load_Change': load_mult,
                'Avg_Load': contract_model.data.load_scenarios.mean().mean(),
                'StrikePrice': contract_model.results.strike_price,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_sensitivity.append(result_dict)
            
        except Exception as e:
            print(f"Error for capture rate multiplier={load_mult}: {str(e)}")
            results_sensitivity.append({
                'CapturLoad_ChangeeRate12w': load_mult,
                'Avg_Load': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
            
        # Clean up
        del contract_model
        
    # Add analysis of results
    results_df = pd.DataFrame(results_sensitivity)

    return results_df

def run_no_contract_boundary_analysis_price(input_data_base):

    """
    Run sensitivity analysis to find the no-contract boundaries for different risk aversion levels.
    Maps combinations of KG and KL where the optimal contract is zero or the problem is infeasible.
    """
    print("\n--- Starting No-Contract Boundary Analysis ---")
    
    # Define risk aversion scenarios to test
    risk_aversion_scenarios = [
        # Symmetrical scenarios

        {'A_G': 0.1, 'A_L': 0.1, 'label': 'Lower Boundary for (A_G=0.1, A_L=0.1)', 'linestyle': '-', 'linewidth': 4, 'color': 'blue'},
        {'A_G': 0.5, 'A_L': 0.5, 'label': 'Lower Boundary for (A_G=0.5, A_L=0.5)', 'linestyle': '-', 'linewidth': 2.5, 'color': 'green'},
        {'A_G': 0.9, 'A_L': 0.9, 'label': 'Lower Boundary for (A_G=0.9, A_L=0.9)', 'linestyle': '--', 'linewidth': 1.5, 'color': 'orange'},
        {'A_G': 1, 'A_L': 1, 'label': 'Lower Boundary for (A_G=1, A_L=1)', 'linestyle': '-.', 'linewidth': 1.5, 'color': 'yellow'},

        # New asymmetrical scenarios
        {'A_G': 0.1, 'A_L': 0.5, 'label': 'Asymmetric (A_G=0.1, A_L=0.5)', 
        'linestyle': ':', 'linewidth': 2.0, 'color': 'red'},
        {'A_G': 0.5, 'A_L': 0.1, 'label': 'Asymmetric (A_G=0.5, A_L=0.1)', 
        'linestyle': '-.', 'linewidth': 2.0, 'color': 'magenta'}
    
    ]
    
    # Define the grid of bias factors to test
    KL_range = np.linspace(-25, 25, 6) / 100  # -25% to 25%
    KG_range = np.linspace(-25, 25, 6) / 100  # -25% to 25%
    
    # Store results for each risk aversion scenario
    all_results = []
    
    # For each risk aversion scenario
    for scenario in risk_aversion_scenarios:
        print(f"\nAnalyzing scenario: A_G={scenario['A_G']}, A_L={scenario['A_L']}")
        
        # Create a grid to store contract amounts for this scenario
        contract_grid = np.zeros((len(KG_range), len(KL_range)))
        contract_grid.fill(np.nan)  # Initialize with NaN to indicate untested points
        
        # Test each combination of KG and KL
        for i, kg in enumerate(tqdm(KG_range, desc=f"Testing KG values for {scenario['label']}")):
            for j, kl in enumerate(KL_range):
                # Create a fresh copy of the input data
                current_input_data = copy.deepcopy(input_data_base)
                
                # Set the risk aversion and bias parameters
                current_input_data.A_G = scenario['A_G']
                current_input_data.A_L = scenario['A_L']
                current_input_data.K_G = kg
                current_input_data.K_L = kl
                
                try:
                    # Run the contract negotiation model
                    contract_model = ContractNegotiation(current_input_data)
                    contract_model.model.Params.FeasibilityTol = 1e-3

                    #contract_model.model.Params.SolutionLimit = 1  # Limit to first solution to avoid long runs (We are just checking for feasibility)
                    contract_model.run()
                    
                    
                    # Store the contract amount (or 0 if it's very small)
                    contract_amount = 0.0 if (hasattr(contract_model.results, 'contract_amount') and 
                                            contract_model.results.contract_amount < 1e-5) else \
                                    contract_model.results.contract_amount
                    contract_grid[i, j] = contract_amount
                    
                    del contract_model
                    
                except Exception as e:
                    # If optimization fails, mark as no-contract (set to 0)
                    print(f"  Error at KG={kg:.4f}, KL={kl:.4f}: {str(e)}")
                    contract_grid[i, j] = 0.0
        
        # Find the boundary points (transition from zero to positive contract)
        boundary_points = []
        
        # Scan rows (fixed KG) to find boundary points
        for i, kg in enumerate(KG_range):
            for j in range(1, len(KL_range)):
                if ((np.isnan(contract_grid[i, j-1]) or contract_grid[i, j-1] == 0) and 
                    contract_grid[i, j] > 0) or \
                   (contract_grid[i, j-1] > 0 and 
                    (np.isnan(contract_grid[i, j]) or contract_grid[i, j] == 0)):
                    # Found a boundary crossing
                    boundary_points.append((KL_range[j-1], kg))
                    boundary_points.append((KL_range[j], kg))
        
        # Scan columns (fixed KL) to find boundary points
        for j, kl in enumerate(KL_range):
            for i in range(1, len(KG_range)):
                if ((np.isnan(contract_grid[i-1, j]) or contract_grid[i-1, j] == 0) and 
                    contract_grid[i, j] > 0) or \
                   (contract_grid[i-1, j] > 0 and 
                    (np.isnan(contract_grid[i, j]) or contract_grid[i, j] == 0)):
                    # Found a boundary crossing
                    boundary_points.append((kl, KG_range[i-1]))
                    boundary_points.append((kl, KG_range[i]))
        
        
     

       # Store the results for this scenario
        all_results.append({
            'scenario': scenario,
            'contract_grid': contract_grid,
            'boundary_points': boundary_points,
            'KL_range': KL_range,
            'KG_range': KG_range
        })
    return all_results

def run_negotiation_power_sensitivity_analysis(input_data_base, Beta_G_values, Beta_L_values):
    """Runs the ContractNegotiation for different combinations of beta G and beta L."""
    print("\n--- Starting Negotation Power Sensitivity Analysis ---")
    results_list = []
    earnings_list_scenarios = []

    for Beta_G,Beta_L in tqdm(zip(Beta_G_values,Beta_L_values), desc="Iterating Beta_G"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        current_input_data.Beta_G = Beta_G
        current_input_data.Beta_L = Beta_L

        try:
            contract_model = ContractNegotiation(current_input_data)
            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'Beta_G': Beta_G,
                'Beta_L': Beta_L,
                'StrikePrice': contract_model.results.strike_price,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'NashProductLog': contract_model.results.objective_value,
                'Nash_Product': np.exp(contract_model.results.objective_value),
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_list.append(result_dict)
            
            # Create earnings results for histograms
            earnings_df = pd.DataFrame({
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'Beta_G': Beta_G, 
                'Beta_L': Beta_L,
                'Revenue_G': contract_model.results.earnings_G.values,
                'Revenue_L': contract_model.results.earnings_L.values,
                'CR_L_Revenue': contract_model.results.earnings_L_CP.values,
                'CR_G_Revenue': contract_model.results.earnings_G_CP.values
            })
            earnings_list_scenarios.append(earnings_df)

        except Exception as e:
            print(f"Error for Beta_G={Beta_G}, Beta_L={Beta_L}: {str(e)}")
            results_list.append({
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'Beta_G': Beta_G, 
                'Beta_L': Beta_L,
                'StrikePrice': np.nan, 
                'ContractAmount': np.nan,
                'Utility_G': np.nan, 
                'Utility_L': np.nan,
                'NashProductLog': np.nan,
                'Nash_Product': np.nan,
                'ThreatPoint_G': np.nan, 
                'ThreatPoint_L': np.nan,
            })
            
            earnings_list_scenarios.append(pd.DataFrame({
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'Beta_G': [Beta_G], 
                'Beta_L': [Beta_L],
                'Revenue_G': [np.nan], 
                'Revenue_L': [np.nan],
                'CP_L_Revenue': [np.nan],
                'CP_G_Revenue': [np.nan],
            }))

        # Clean up
        del contract_model
    
    # Combine results and analyze
    results_df = pd.DataFrame(results_list)
    earnings_df = pd.concat(earnings_list_scenarios, ignore_index=True)

    print("\n--- Negotation Power Sensitivity Analysis Complete ---")
    return results_df, earnings_df

def run_load_generation_ratio_sensitivity_analysis(input_data_base):
    """
    Sensitivity analysis for varying the Load/Generation ratio.
    For each ratio, scales load_scenarios and/or production so that:
        mean(load_scenarios) / mean(production) == ratio
    """
    print("\n--- Starting Load/Generation Ratio Sensitivity Analysis ---")
    
    ratio_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

    results_sensitivity = []

    base_mean_load = input_data_base.load_scenarios.mean().mean()
    base_mean_gen = input_data_base.production.mean().mean()

    for ratio in tqdm(ratio_values, desc="Testing Load/Gen ratios"):
        current_input_data = copy.deepcopy(input_data_base)
        # Option 1: Scale load to achieve desired ratio, keep generation fixed
        new_mean_load = ratio * base_mean_gen
        scale_factor = new_mean_load / base_mean_load
        current_input_data.load_scenarios = current_input_data.load_scenarios * scale_factor

        try:
            contract_model = ContractNegotiation(current_input_data)
            contract_model.run()

            result_dict = {
                'Load_Gen_Ratio': ratio,
                'Avg_Load': current_input_data.load_scenarios.mean().mean(),
                'Avg_Gen': current_input_data.production.mean().mean(),
                'StrikePrice': contract_model.results.strike_price,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
                'Nash_Product': (contract_model.results.utility_G - contract_model.data.Zeta_G) * 
                                (contract_model.results.utility_L - contract_model.data.Zeta_L)
            }
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
            results_sensitivity.append(result_dict)
        except Exception as e:
            print(f"Error for Load/Gen ratio={ratio}: {str(e)}")
            results_sensitivity.append({
                'Load_Gen_Ratio': ratio,
                'Avg_Load': np.nan,
                'Avg_Gen': np.nan,
                'A_G': current_input_data.A_G,
                'A_L': current_input_data.A_L,
                'StrikePrice': np.nan,
                'ContractAmount': np.nan,
                'Utility_G': np.nan,
                'Utility_L': np.nan,
                'ThreatPoint_G': np.nan,
                'ThreatPoint_L': np.nan,
                'Nash_Product': np.nan
            })
        del contract_model

    results_df = pd.DataFrame(results_sensitivity)
    return results_df



############## Unncessary Sensitivity Analysis Functions ##############

def run_cvar_alpha_sensitivity_analysis(input_data_base):
    """Runs the ContractNegotiation for different combinations of beta G and beta L."""
    print("\n--- Starting Negotation Power Sensitivity Analysis ---")
    results_list = []
    earnings_list_scenarios = []
    alpha_values = np.array([0.95])

    for alpha in tqdm(alpha_values, desc="Iterating Beta_G"):
        # Use a fresh copy for each iteration
        current_input_data = copy.deepcopy(input_data_base)
        current_input_data.alpha = alpha
        

        try:
            contract_model = ContractNegotiation(current_input_data)
            contract_model.run()
            
            # Create base result dictionary
            result_dict = {
                'alpha': alpha,
                'StrikePrice': contract_model.results.strike_price,
                'ContractAmount': contract_model.results.contract_amount_hour,
                'Utility_G': contract_model.results.utility_G,
                'Utility_L': contract_model.results.utility_L,
                'NashProductLog': contract_model.results.objective_value,
                'NashProduct': np.exp(contract_model.results.objective_value),
                'ThreatPoint_G': contract_model.data.Zeta_G,
                'ThreatPoint_L': contract_model.data.Zeta_L,
            }
            
            # Add contract-type specific metrics
            if hasattr(current_input_data, 'contract_type'):
                if current_input_data.contract_type == "PAP":
                    if hasattr(contract_model.results, 'gamma'):
                        result_dict['Gamma'] = contract_model.results.gamma
                elif current_input_data.contract_type == "Baseload":
                    # Add Baseload-specific metrics if needed
                    pass
            
            results_list.append(result_dict)
            
            # Create earnings results for histograms
            earnings_df = pd.DataFrame({
                'alpha': alpha, 
                'Revenue_G': contract_model.results.earnings_G.values,
                'Revenue_L': contract_model.results.earnings_L.values
            })
            earnings_list_scenarios.append(earnings_df)

        except Exception as e:
            print(f"Error for alpha={alpha}: {str(e)}")
            results_list.append({
                'alpha': alpha, 
                'StrikePrice': np.nan, 
                'ContractAmount': np.nan,
                'Utility_G': np.nan, 
                'Utility_L': np.nan,
                'NashProductLog': np.nan,
                'NashProduct': np.nan,
                'ThreatPoint_G': np.nan, 
                'ThreatPoint_L': np.nan,
            })
            
            earnings_list_scenarios.append(pd.DataFrame({
                'alpha': [alpha],
                'Revenue_G': [np.nan], 'Revenue_L': [np.nan],
            }))

        # Clean up
        del contract_model
    
    # Combine results and analyze
    results_df = pd.DataFrame(results_list)
    earnings_df = pd.concat(earnings_list_scenarios, ignore_index=True)

    print("\n--- Alpha Sensitivity Analysis Complete ---")
    return results_df, earnings_df