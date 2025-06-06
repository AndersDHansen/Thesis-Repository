"""Generate Monte Carlo scenarios and save them to CSV files.

This script generates price, production, capture rate, and load scenarios using Monte Carlo
simulation and saves them to CSV files for later use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ───────────────────────── helpers ──────────────────────────

def _yearly_index(start: str | pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start), periods=periods, freq="YS")


def _save_matrix(folder: Path, kind: str, mat: np.ndarray, start: str | pd.Timestamp):
    """Save a (years × sims) matrix as `<kind>_scenarios_<y>y_<s>s.csv`."""
    years, sims = mat.shape
    df = pd.DataFrame(mat, index=_yearly_index(start, years),
                      columns=pd.RangeIndex(sims, name="sim"))
    fname = f"{kind}_scenarios_{years}y_{sims}s.csv"
    df.to_csv(folder / fname, index_label="year")


# ───────────────────────── price model ──────────────────────

@dataclass
class PriceModel:
    loc: float
    scale: float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls, csv_path: str, seed: Optional[int] = None) -> "PriceModel":
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df.index = pd.to_datetime(df["HourUTC"])
        yearly = df["DK2_EUR/MWh"].resample("YE").sum().to_numpy(float)
        loc, scale = stats.expon.fit(yearly)
        return cls(loc, scale, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        return self.rng.exponential(self.scale, (years, sims)) + self.loc


# ──────────────────── production model (GEV) ─────────────────

@dataclass
class ProductionModel:
    shape: float
    loc: float
    scale: float
    cap_gwh: Optional[float]
    rng: np.random.Generator

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        capacity_mw: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "ProductionModel":
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        yearly = df.set_index("time")["electricity"].resample("YE").sum().to_numpy(float)
        shape, loc, scale = stats.genextreme.fit(yearly)
        cap_gwh = capacity_mw * 8_760 / 1_000 if capacity_mw else None
        return cls(shape, loc, scale, cap_gwh, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        draws = stats.genextreme.rvs(
            self.shape, loc=self.loc, scale=self.scale,
            size=(years, sims), random_state=self.rng
        ) / 1_000  # MWh → GWh
        if self.cap_gwh is not None:
            np.clip(draws, 0, self.cap_gwh, out=draws)
        return draws


# ───────────────────── capture‑rate model ───────────────────

@dataclass
class CaptureRateModel:
    price_mu: float
    price_std: float
    prod_mu: float
    prod_std:float
    corr_agg:float
    z_year_corr:np.ndarray
    mu_z_corr:float
    std_z_corr:float
    rng: np.random.Generator

    @classmethod
    def from_csv(cls, csv_path: str, seed: Optional[int] = None) -> "CaptureRateModel":
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df["HourUTC"] = pd.to_datetime(df["HourUTC"])
        df = df.set_index("HourUTC")

        # try to locate a production column; fall back to flat 1.0 rate
        prod_col = next((c for c in df.columns if "MWhDK2" in c or "prod" in c.lower()), None)
        if prod_col is None:
            return cls(1.0, 0.05, np.random.default_rng(seed))

        price = df["DK2_EUR/MWh"].to_numpy(float)
        prod = df[prod_col].to_numpy(float)

        # Mean and std for price
        price_mu, price_std = df['DK2_EUR/MWh'].mean(), df['DK2_EUR/MWh'].std()  # EUR/MWh

        # Mean and std for production
        prod_mu, prod_std = df['OnshoreWindGe50kW_MWhDK2'].mean(), df['OnshoreWindGe50kW_MWhDK2'].std()  # MWh

        # Correlation across price and production [hour] to maintain correlation
        # Add a 'year' column
        df['year'] = df.index.year

        # Compute hourly correlation between Onshore Wind and Price for each year
        corr_by_year = (
            df.groupby('year')[['OnshoreWindGe50kW_MWhDK2', 'DK2_EUR/MWh']]
            .corr()
            .iloc[0::2, 1]  # Select cross-correlations only
            .reset_index()
            .rename(columns={'DK2_EUR/MWh': 'hourly_corr'})
            .drop('level_1', axis=1))
        corr_by_year_arr = corr_by_year["hourly_corr"].to_numpy()[1:]

        corr_agg = df["DK2_EUR/MWh"].corr(df["OnshoreWindGe50kW_MWhDK2"])

        z_year_corr   = np.arctanh(corr_by_year_arr)   # Fisher-z
        std_z_corr  = z_year_corr.std(ddof=1)      # unbiased SD
        mu_z_corr     = z_year_corr.mean()    

        

        return cls(price_mu,price_std, prod_mu,prod_std,corr_agg,z_year_corr,mu_z_corr,std_z_corr, np.random.default_rng(seed))

    def simulate(self, years: int, sims: int) -> np.ndarray:
        
        noise    = np.random.normal(0, self.std_z_corr, size=(years,sims))
        rho_samples = self.mu_z_corr + noise
        rho_samp  = np.tanh(rho_samples)                                # ρ ∈ (-1,1)


        CR = 1 + rho_samp * (self.price_std/self.price_mu) *(self.prod_std/self.prod_mu)
        return CR


# ───────────────────────── load model ───────────────────────

_daily_mean = np.array([
    337.01, 319.10, 285.94, 268.12, 318.61, 329.53, 335.84, 336.94,
    316.81, 270.06, 250.76, 297.36, 310.81, 322.45, 338.52, 360.43,
    341.99, 312.55, 351.49, 349.64, 363.59, 367.08, 336.56, 300.43,
    285.71, 329.89, 335.36, 336.34, 337.69, 336.93,
])
_load_mean = _daily_mean.mean()            # MW per hour
_load_std = np.sqrt(834.5748)              # derived variance


def _simulate_load(years: int, sims: int, rng: np.random.Generator) -> np.ndarray:
    load = rng.normal(_load_mean, _load_std, (years, sims)) * 8_760 / 1_000  # GWh/yr
    return np.clip(load, 0, None)


# ──────────── one‑shot scenario generator ───────────

def run_scenarios(
    *,
    years: int,
    num_scenarios: int,
    start_time: str | pd.Timestamp,
    price_csv_path: str,
    prod_csv_path: str,
    capacity_mw: Optional[float],
    output_dir: str = "outputs",
    seed: int = 42,
):
    """Generate and save **price, production, capture‑rate and load** scenarios."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1) price
    price_mdl = PriceModel.from_csv(price_csv_path, seed)
    price_mat = price_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "price", price_mat, start_time)

    # 2) production
    prod_mdl = ProductionModel.from_csv(prod_csv_path, capacity_mw, seed)
    prod_mat = prod_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "production", prod_mat, start_time)

    # 3) capture‑rate (just needs price file that already has production col)
    cr_mdl = CaptureRateModel.from_csv(price_csv_path, seed)
    cr_mat = cr_mdl.simulate(years, num_scenarios)
    _save_matrix(out, "capture_rate", cr_mat, start_time)

    # 4) load
    load_mat = _simulate_load(years, num_scenarios, rng)
    _save_matrix(out, "load", load_mat, start_time)

    print(f"Wrote scenarios to {out.resolve()}")

if __name__ == "__main__":
    # Example usage
    years = 2  # 10 years
    num_scenarios = 1000

    # Wind Profile 
    csv_wind = "Code/Data/Wind/combined_wind_data.csv"  # adjust path if needed
    csv_solar = "Code/Data/Solar/combined_solar_data.csv"  # adjust path if needed

    start_time = pd.Timestamp("2025-01-01")
    price_csv_path =  "Code/Data/EnergyReport.csv"
    prod_csv_path = csv_wind
    capacity_mw = 100  # MW
    output_dir = "Code/scenarios"    
    run_scenarios(
        years=years,
        num_scenarios=num_scenarios,
        start_time=start_time,
        price_csv_path=price_csv_path,
        prod_csv_path=prod_csv_path,
        capacity_mw=capacity_mw,
        output_dir=output_dir
    )
    print("All scenarios generated successfully")
