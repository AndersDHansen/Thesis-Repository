"""Generate Monte Carlo scenarios and save them to CSV files.

This script generates price, production, capture rate, and load scenarios using Monte Carlo
simulation and saves them to CSV files for later use.
"""

from Forecasting import (
    PriceForecastConfig,
    PriceForecast,
    HistoricalProductionConfig,
    HistoricalProductionForecaster,
    MonteCarloConfig,
    MonteCarloSimulator,
    CaptureRateConfig,
    CaptureRateForecaster
)
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def generate_scenario_batch(config):
    """Generate a batch of scenarios for parallel processing"""
    mc_cfg = MonteCarloConfig(n_simulations=config['batch_size'], random_seed=config['seed'])
    
    if config['type'] == 'price':
        cfg = PriceForecastConfig(
            time_horizon=config['time_horizon'],
            start_date=config['start_date'],
            csv_path=config['price_csv_path'],
            random_seed=config['seed']
        )
        sim = MonteCarloSimulator(cfg, mc_cfg, PriceForecast)
    elif config['type'] == 'production':
        cfg = HistoricalProductionConfig(
            time_horizon=config['time_horizon'],
            csv_path=config['prod_csv_path'],
            start_date=config['start_date'],
            capacity=config['generator_capacity'],
            random_seed=config['seed']
        )
        sim = MonteCarloSimulator(cfg, mc_cfg, HistoricalProductionForecaster)
    elif config['type'] == 'capture_rate':
        cfg = CaptureRateConfig(
            time_horizon=config['time_horizon'],
            start_date=config['start_date'],
            csv_path=config['price_csv_path'],
            random_seed=config['seed']
        )
        sim = MonteCarloSimulator(cfg, mc_cfg, CaptureRateForecaster)
    else:  # load
        rng = np.random.default_rng(config['seed'])
        load_mean = config['load_mean']
        load_std = config['load_std']
        scenarios = rng.normal(load_mean, load_std, 
                             size=(config['time_horizon'], config['batch_size'])) * 8760 / 1000
        return scenarios

    return sim.run().values

def generate_and_save_scenarios(
    time_horizon: int,
    num_scenarios: int,
    start_date: pd.Timestamp,
    price_csv_path: str,
    prod_csv_path: str,
    generator_capacity: float,
    output_dir: str,
    random_seed: int = 42
):
    """Generate Monte Carlo scenarios and save them to CSV files.

    Parameters
    ----------
    time_horizon : int
        Number of time periods to simulate
    num_scenarios : int
        Number of Monte Carlo scenarios to generate
    start_date : pd.Timestamp
        Start date for the scenarios
    price_csv_path : str
        Path to historical price data CSV
    prod_csv_path : str
        Path to historical production data CSV
    generator_capacity : float
        Generator capacity in MW
    output_dir : str
        Directory to save output CSV files
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    """    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename pattern
    file_pattern = f"{{type}}_scenarios_{time_horizon}y_{num_scenarios}s.csv"
    
    # Load configuration data for load scenarios
    daily_load_mean = np.array([
        337.01, 319.10, 285.94, 268.12, 318.61, 329.53, 335.84, 336.94, 
        316.81, 270.06, 250.76, 297.36, 310.81, 322.45, 338.52, 360.43, 
        341.99, 312.55, 351.49, 349.64, 363.59, 367.08, 336.56, 300.43, 
        285.71, 329.89, 335.36, 336.34, 337.69, 336.93
    ])
    load_mean = np.mean(daily_load_mean)
    load_std = np.sqrt(834.5748)

    # Configure parallel processing
    num_cores = multiprocessing.cpu_count()
    batch_size = num_scenarios // num_cores
    remaining = num_scenarios % num_cores

    # Prepare batch configurations
    all_configs = []
    base_config = {
        'time_horizon': time_horizon,
        'start_date': start_date,
        'price_csv_path': price_csv_path,
        'prod_csv_path': prod_csv_path,
        'generator_capacity': generator_capacity,
    }

    # Generate seeds for each batch to ensure reproducibility
    rng = np.random.default_rng(random_seed)
    batch_seeds = rng.integers(0, 2**32, size=num_cores * 4)  # 4 types of scenarios

    for scenario_type in ['price', 'production', 'capture_rate', 'load']:
        scenarios_list = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(num_cores):
                cfg = base_config.copy()
                cfg.update({
                    'type': scenario_type,
                    'batch_size': batch_size + (1 if i < remaining else 0),
                    'seed': batch_seeds[i],
                    'load_mean': load_mean,
                    'load_std': load_std
                })
                futures.append(executor.submit(generate_scenario_batch, cfg))
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc=f"Generating {scenario_type} scenarios"):
                scenarios_list.append(future.result())

        # Combine results and save
        all_scenarios = np.hstack(scenarios_list)
        df = pd.DataFrame(
            all_scenarios,
            index=pd.date_range(start=start_date, periods=time_horizon, freq='YS'),
            columns=[f'scenario_{i+1}' for i in range(num_scenarios)]
        )
        df.to_csv(Path(output_dir) / file_pattern.format(type=scenario_type))

if __name__ == "__main__":
    # Example usage
    time_horizon = 2  # 10 years
    num_scenarios = 1000

    # Wind Profile 
    csv_wind = "Code/Data/Wind/combined_wind_data.csv"  # adjust path if needed
    csv_solar = "Code/Data/Solar/combined_solar_data.csv"  # adjust path if needed

    start_date = pd.Timestamp("2025-01-01")
    price_csv_path =  "Code/Data/EnergyReport.csv"
    prod_csv_path = csv_wind
    generator_capacity = 100  # MW
    output_dir = "scenarios"    
    generate_and_save_scenarios(
        time_horizon=time_horizon,
        num_scenarios=num_scenarios,
        start_date=start_date,
        price_csv_path=price_csv_path,
        prod_csv_path=prod_csv_path,
        generator_capacity=generator_capacity,
        output_dir=output_dir
    )
    print("All scenarios generated successfully")
