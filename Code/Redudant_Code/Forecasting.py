"""Forecasting.py – synthetic price and production forecasting

Key components
--------------
1. Synthetic three-factor **price** model (deterministic + spikes + residual)
2. Monte-Carlo wrapper for the synthetic price model
3. Historical-based **production** forecaster (gaussian model only)
4. Monte-Carlo wrapper for the production forecaster
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm, beta,t,expon
from scipy import stats



# ==============================================================================
# 1)  Synthetic three‑factor *price* model
# ==============================================================================

@dataclass
class PriceForecastConfig:
    """Configuration for the synthetic 3‑factor price model."""

    time_horizon: int #Years 
    start_date: pd.Timestamp  # Not used in this implementation, but can be useful for time series
    csv_path: str # Path to historical price data 

    random_seed: Optional[int] = None


class PriceForecast:
    """Generate ONE synthetic price path (deterministic + spikes + residual)."""

    def __init__(self, cfg: PriceForecastConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        df= self._load_distribution()
        self.loc, self.scale = self._generate_distribution(df)

    def generate_curve(self) -> pd.Series:

        samples = expon.rvs(loc=self.loc, scale=self.scale, size=self.cfg.time_horizon, random_state=self.rng)
        dates = pd.date_range(start=self.cfg.start_date, periods=self.cfg.time_horizon, freq='YS')
        return pd.Series(samples, index=dates, name="price_EUR_MWh")

    def _load_distribution(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path, sep=";", decimal=",")
        df = df.drop(columns=['HourDK'])
        df.index = pd.to_datetime(df["HourUTC"])
        df = df.drop(columns=['HourUTC'])
        yearly_df = df.resample('YE').sum()
        yearly_price_df = yearly_df['DK2_EUR/MWh'] * 1000 * 1e-6  # Convert to EUR/MWh to EUR/GWh to Mio EUR/ GWH
        return  yearly_price_df

    def _generate_distribution(self, df: pd.DataFrame): 
        """Generate a distribution of prices based on historical data."""
        loc, scale = expon.fit(df.values)
        return loc, scale


# ==============================================================================
# 2)  Monte-Carlo wrapper for synthetic price model
# ==============================================================================

@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    summary_quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95)
    random_seed: Optional[int] = None


# ==============================================================================
# 3)  Historical-based *production* forecaster (gaussian model)
# ==============================================================================

@dataclass
class HistoricalProductionConfig:
    """Configuration for production forecasting from historical data.

    Forecasting model:
        • Draw each hour from N(μ, σ) estimated from historical time series.
        • Optional residual noise override and clipping by capacity.
    """

    time_horizon: int
    csv_path: str
    start_date: pd.Timestamp 

    datetime_column: str = "time"
    value_column: str = "electricity"  # kW in example CSV
    parse_format: str = "%d/%m/%Y %H.%M"

    capacity: Optional[float] = None     # MW
    random_seed: Optional[int] = None


class HistoricalProductionForecaster:
    """Forecast production using historical statistics (Gaussian model only)."""

    def __init__(self, cfg: HistoricalProductionConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.hours_year = 24*365 # 24 * 365
        self._load_and_calibrate()

    def _clip_monthly_production(self,df, capacity):
            
        """
        Clip the monthly sum of production to the physical maximum for each month.

        Parameters:
            values: 1D np.ndarray or pd.Series of production values Monthly
            months: 1D np.ndarray or pd.Series of month numbers (same length as values, e.g., 1-12)
            capacity: float, max MW

        Returns:
            np.ndarray of clipped values (same shape as input)
        """
        clipped = df.copy()
        df = pd.DataFrame(clipped)
        # Calculate max possible for each month
        #days_in_month = df.index.to_series().groupby(df.index.to_period('M')).first().dt.days_in_month # Remember this, will be used in strike price and amount
        max_possible = self.hours_year * capacity
        yearly_clipped = clipped.clip(upper=max_possible, axis=0)
        return yearly_clipped

    def _load_and_calibrate(self):
        #Load CSV
        df = pd.read_csv(self.cfg.csv_path)
        # Datetime
        df[self.cfg.datetime_column] = pd.to_datetime(df[self.cfg.datetime_column])
        # Set time as index and sort 
        df = df.set_index(self.cfg.datetime_column).sort_index()
        # Remove Jan with only one value
        year_counts = df.index.to_period('Y').value_counts().sort_index()
        # Verify no outlier data with only one value in January (happens)
        self.years_to_keep = year_counts[year_counts > 1].index
        # Filtered DF 
        self.df_filtered = df[df.index.to_period('Y').isin(self.years_to_keep)]
        # Get 
        series = (self.df_filtered[self.cfg.value_column].astype(float)).dropna().resample('YE').sum() # Resample to yearly sum
        self.mu = series.mean()
        self.sigma = series.std(ddof=0)
        self.std =self.sigma
        self.shape, self.loc, self.scale = stats.genextreme.fit(series.values)
        

    def generate_curve(self) -> pd.Series:
        values = stats.genextreme.rvs(self.shape, loc=self.loc, scale=self.scale, size=self.cfg.time_horizon, random_state=self.rng)/1000 # Convert MWh to GWh
        dates = pd.date_range(start=self.cfg.start_date, periods=self.cfg.time_horizon, freq='YS')   # YS-year start
        series = pd.Series(values, index=dates, name="production_GWh")
        series = self._clip_monthly_production(series, self.cfg.capacity)
        return series
    
# ==============================================================================
# 3)  Historical-based *production* forecaster (gaussian model)
# ==============================================================================

@dataclass
class CaptureRateConfig:
    time_horizon: int #Years 
    start_date: pd.Timestamp  # Not used in this implementation, but can be useful for time series
    csv_path: str # Path to historical price data 

    random_seed: Optional[int] = None

class CaptureRateForecaster:

    def __init__(self, cfg: CaptureRateConfig):
    
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self._load_and_calibrate() 
    def _load_and_calibrate(self):
        df = pd.read_csv(self.cfg.csv_path, sep=";", decimal=",")
        df = df.drop(columns=['HourDK'])
        df.index = pd.to_datetime(df["HourUTC"])
        df = df.drop(columns=['HourUTC'])

        # Mean and std for price 
        self.price_mu, self.price_std = df['DK2_EUR/MWh'].mean(), df['DK2_EUR/MWh'].std() # EUR/MWh

        # Mean and std for production
        self.prod_mu, self.prod_std = df['OnshoreWindGe50kW_MWhDK2'].mean(), df['OnshoreWindGe50kW_MWhDK2'].std() # MWh
        # Correlation across price and production [hour] to maintain correlation 
        self.corr = df['DK2_EUR/MWh'].corr(df['OnshoreWindGe50kW_MWhDK2'])
   
    def _noise(self):
        return self.rng.normal(loc=0, scale=0.5, size=self.cfg.time_horizon)
    
    def generate_curve(self) -> pd.Series:
        dates = pd.date_range(start=self.cfg.start_date, periods=self.cfg.time_horizon, freq='YS')   # YS = year-start
        noise = self._noise()
        corr_clipped = np.clip(self.corr + noise, -1, 1)  # Ensure noise is non-negative
        CR = 1 + corr_clipped* (self.price_std/self.price_mu) * (self.prod_std/self.prod_mu) 
        series = pd.Series(CR, index=dates, name="CaptureRate")
        return series


# ==============================================================================
# 4)  Monte-Carlo wrapper for Production , Price and Capture Rate forecasters
# ==============================================================================


class MonteCarloSimulator:
    def __init__(self, cfg_template, mc_cfg: MonteCarloConfig, forecaster_class):
        """
        Generic Monte Carlo simulator for any forecaster.

        Parameters
        ----------
        cfg_template : dataclass instance
            A config object that supports .__dict__ and includes .random_seed
        mc_cfg : MonteCarloConfig
        forecaster_class : class
            A class that accepts cfg_template and has .generate_curve() -> Series
        """
        self.cfg_template = cfg_template
        self.mc_cfg = mc_cfg
        self.forecaster_class = forecaster_class
        self.master_rng = np.random.default_rng(mc_cfg.random_seed)

    def run(self) -> pd.DataFrame:
        paths = []
        for _ in range(self.mc_cfg.n_simulations):
            seed = int(self.master_rng.integers(0, 2**32 - 1))
            #cfg_i = self._copy_config_with_seed(seed)
            forecaster = self.forecaster_class(self.cfg_template.__class__(**{**self.cfg_template.__dict__, "random_seed": seed}))
            #forecaster = self.forecaster_class(self.cfg_template)
            paths.append(forecaster.generate_curve())
        df = pd.concat(paths, axis=1, keys=range(self.mc_cfg.n_simulations))
        df.columns.name = "sim"
        return df

    def _copy_config_with_seed(self, seed: int):
        new_cfg = self.cfg_template.__class__(**{**self.cfg_template.__dict__, "random_seed": seed})
        return new_cfg

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = {"mean": df.mean(axis=1)}
        for q in self.mc_cfg.summary_quantiles:
            stats[f"q{int(q*100):02d}"] = df.quantile(q, axis=1)
        return pd.DataFrame(stats)